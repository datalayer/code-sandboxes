# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""A live Kaggle kernel, with no service but Kaggle itself.

A batch job is stateless and a Kaggle interactive session has no public API —
the runtime URL its editor uses is minted by the web frontend and nothing
else. What both ends CAN reach is Kaggle's own datasets API, so that is the
whole transport: a batch job runs an agent that starts a real ipykernel and
polls a private dataset for code; the client pushes code there and polls a
second private dataset for the outputs the agent publishes. No tunnel, no
third-party relay — the rendezvous is the account's own storage.

The price is latency, and it is stated rather than hidden: a turn costs a
dataset version upload and a poll on each side — seconds, not milliseconds —
against a batch job's minute-plus per turn and no state at all. The state is
real: one kernel process lives for the whole session.

The agent embeds the account's API credentials in the (private) kernel source,
because a batch job has no other way to authenticate against the datasets API:
Kaggle attaches secrets in its UI only. The kernel is pushed private, to the
user's own account; still, it is their token in their artifact, so the mode is
opt-in (``live=True``).

@module kaggle_live
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable, Any, Optional

__all__ = ["KaggleLiveSession", "build_agent_code", "resolve_kaggle_credentials"]

#: How often each side looks for a message from the other, in seconds.
POLL_SECONDS = 3.0

#: While a turn is in flight — and shortly after one — both ends poll this
#: fast instead. The wait between polls is pure added latency on top of
#: Kaggle's own dataset-version processing, and an active REPL sends its next
#: input right away; only an idle session deserves the slow cadence.
FAST_POLL_SECONDS = 0.75

#: How long after the last activity the fast cadence is kept.
FAST_WINDOW_SECONDS = 60.0

#: The agent exits after this long without a message, so an abandoned session
#: does not burn quota until the batch timeout.
IDLE_TIMEOUT_SECONDS = 30 * 60

#: How long the client waits for the agent to boot: the job queues, the image
#: boots, the kernel starts — minutes on a busy day, longer with a GPU.
READY_TIMEOUT_SECONDS = 900


def _username_of_access_token(token: str) -> str:
    """The account an access token belongs to, asked of Kaggle itself.

    The token (``KGAT_…``) carries no username; the kaggle client learns it
    by introspection. Its own ``authenticate()`` calls ``exit(1)`` on
    failure — a library must not — so the access-token step is run alone.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api._load_config()
    if not api._authenticate_with_access_token():
        raise RuntimeError(
            "KAGGLE_API_TOKEN was not accepted by Kaggle. The token may be "
            "expired or revoked — create one at kaggle.com/settings."
        )
    username = api.config_values.get(api.CONFIG_NAME_USER)
    if not username:
        raise RuntimeError("Kaggle accepted the token but named no account.")
    return str(username)


def resolve_kaggle_credentials() -> tuple[str, dict[str, str]]:
    """The account's username, and the environment that authenticates it.

    The same three places the provider registry accepts, in the same order:
    ``KAGGLE_API_TOKEN`` — an access token (``KGAT_…``) or the contents of
    ``kaggle.json`` —, the ``KAGGLE_USERNAME``/``KAGGLE_KEY`` pair, and
    ``~/.kaggle/kaggle.json``.

    Returns ``(username, agent_env)``: the username names the dataset bus,
    and ``agent_env`` is exactly the environment the AGENT sets inside the
    batch job so its kaggle client authenticates as the same account.
    """
    import os

    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        try:
            parsed = json.loads(token)
            return parsed["username"], {
                "KAGGLE_USERNAME": parsed["username"],
                "KAGGLE_KEY": parsed["key"],
            }
        except (ValueError, KeyError):
            # Not the JSON of kaggle.json: an ACCESS token, which the kaggle
            # client authenticates by introspection — inside the job as well,
            # where the same variable is all it needs.
            return _username_of_access_token(token), {"KAGGLE_API_TOKEN": token}
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}
    config = Path("~/.kaggle/kaggle.json").expanduser()
    if config.is_file():
        parsed = json.loads(config.read_text(encoding="utf-8"))
        return parsed["username"], {
            "KAGGLE_USERNAME": parsed["username"],
            "KAGGLE_KEY": parsed["key"],
        }
    raise RuntimeError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN, "
        "KAGGLE_USERNAME/KAGGLE_KEY, or place ~/.kaggle/kaggle.json."
    )


def build_agent_code(
    agent_env: dict[str, str],
    c2k_ref: str,
    k2c_ref: str,
    *,
    poll_seconds: float = POLL_SECONDS,
    idle_timeout: float = IDLE_TIMEOUT_SECONDS,
) -> str:
    """The script the batch job runs: a kernel, fed from the dataset bus.

    Standalone on purpose — it imports nothing of this package, only what a
    Kaggle image already carries: the ``kaggle`` client, ``jupyter_client``
    and ``ipykernel``.
    """
    settings = json.dumps(
        {
            "auth": agent_env,
            "c2k": c2k_ref,
            "k2c": k2c_ref,
            "poll": poll_seconds,
            "fast": FAST_POLL_SECONDS,
            "fast_window": FAST_WINDOW_SECONDS,
            "idle": idle_timeout,
        }
    )
    template = '''
import json, os, shutil, tempfile, time
from pathlib import Path

SETTINGS = json.loads(%SETTINGS%)
os.environ.update(SETTINGS["auth"])

from kaggle import api  # noqa: E402  (needs the environment above)
from jupyter_client.manager import KernelManager  # noqa: E402

def publish(seq, reply):
    folder = Path(tempfile.mkdtemp(prefix="bus-"))
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": SETTINGS["k2c"].split("/")[-1],
        "id": SETTINGS["k2c"],
        "licenses": [{"name": "CC0-1.0"}],
    }))
    (folder / "message.json").write_text(json.dumps({"seq": seq, "reply": reply}))
    api.dataset_create_version(str(folder), version_notes=f"seq {seq}", quiet=True)
    shutil.rmtree(folder, ignore_errors=True)

def fetch():
    folder = Path(tempfile.mkdtemp(prefix="bus-"))
    try:
        api.dataset_download_files(SETTINGS["c2k"], path=str(folder), force=True, unzip=True)
        return json.loads((folder / "message.json").read_text())
    except Exception:
        return None
    finally:
        shutil.rmtree(folder, ignore_errors=True)

manager = KernelManager()
manager.start_kernel()
client = manager.client()
client.start_channels()
client.wait_for_ready(timeout=120)

def run(code):
    outputs = []
    state = {"status": "ok"}
    def sink(msg):
        kind = msg["msg_type"]
        content = msg["content"]
        if kind == "stream":
            outputs.append({"output_type": "stream", "name": content["name"], "text": content["text"]})
        elif kind in ("execute_result", "display_data"):
            outputs.append({"output_type": kind, "data": content.get("data", {}), "metadata": content.get("metadata", {})})
        elif kind == "error":
            state["status"] = "error"
            outputs.append({"output_type": "error", "ename": content["ename"], "evalue": content["evalue"], "traceback": content.get("traceback", [])})
    reply = client.execute_interactive(code, output_hook=sink, timeout=None)
    if reply["content"]["status"] == "error":
        state["status"] = "error"
    return {"status": state["status"], "outputs": outputs}

publish(0, {"status": "ok", "outputs": []})  # ready
last = 0
alive_since = time.time()
# The kernel just booted: the first input follows right away, so start fast.
active_since = time.time()
while time.time() - alive_since < SETTINGS["idle"]:
    message = fetch()
    if message and message.get("seq", 0) > last:
        last = message["seq"]
        alive_since = time.time()
        active_since = time.time()
        if message.get("op") == "shutdown":
            break
        publish(last, run(message.get("code", "")))
        active_since = time.time()
    fast = time.time() - active_since < SETTINGS["fast_window"]
    time.sleep(SETTINGS["fast"] if fast else SETTINGS["poll"])

client.stop_channels()
manager.shutdown_kernel(now=True)
print("agent: session closed")
'''
    return template.replace("%SETTINGS%", repr(settings))


class KaggleLiveSession:
    """A persistent kernel on Kaggle, spoken to over the dataset bus.

    Duck-types the kernel client the interactive sandbox path holds:
    ``execute(code, timeout=...) -> reply`` and ``stop()``, so
    :class:`KaggleSandbox` can hold one in place of a websocket client.
    """

    def __init__(
        self,
        executor: Any,
        *,
        api: Any = None,
        session_id: Optional[str] = None,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        self._executor = executor
        if api is None:
            from kaggle import api as kaggle_api

            api = kaggle_api
        self._api = api
        self._poll = poll_seconds
        self._session_id = session_id or uuid.uuid4().hex[:8]
        self._username: Optional[str] = None
        self._c2k: Optional[str] = None
        self._k2c: Optional[str] = None
        self._seq = 0
        self._slug: Optional[str] = None
        self.id = f"kaggle-live-{self._session_id}"

    # -- the bus ---------------------------------------------------------

    def _write_bus(self, ref: str, payload: dict, *, create: bool) -> None:
        folder = Path(tempfile.mkdtemp(prefix="bus-"))
        try:
            (folder / "dataset-metadata.json").write_text(
                json.dumps(
                    {
                        "title": ref.split("/")[-1],
                        "id": ref,
                        "licenses": [{"name": "CC0-1.0"}],
                    }
                ),
                encoding="utf-8",
            )
            (folder / "message.json").write_text(json.dumps(payload), encoding="utf-8")
            # `quiet` silences the progress bar only — the client prints
            # "Dataset URL: …" unconditionally, once per call, and the bus
            # is polled every few seconds.
            with contextlib.redirect_stdout(io.StringIO()):
                if create:
                    self._api.dataset_create_new(str(folder), public=False, quiet=True)
                else:
                    self._api.dataset_create_version(
                        str(folder), version_notes=f"seq {payload.get('seq')}", quiet=True
                    )
        finally:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)

    def _read_bus(self, ref: str) -> Optional[dict]:
        folder = Path(tempfile.mkdtemp(prefix="bus-"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self._api.dataset_download_files(
                    ref, path=str(folder), force=True, unzip=True
                )
            return json.loads((folder / "message.json").read_text(encoding="utf-8"))
        except Exception:
            # Not there yet, or a version still processing: the caller polls.
            return None
        finally:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)

    # -- the session -----------------------------------------------------

    def _agent_log_tail(self, lines: int = 25) -> str:
        """The last lines of the agent job's log, for a post-mortem."""
        if not self._slug:
            return ""
        folder = Path(tempfile.mkdtemp(prefix="bus-log-"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                files = self._executor.output(self._slug, str(folder))
            for name in files:
                path = Path(name)
                if path.suffix == ".log" or path.name == "output.log":
                    text = path.read_text(encoding="utf-8", errors="replace")
                    return "\n".join(text.splitlines()[-lines:])
        except Exception:  # noqa: BLE001 — the log is a courtesy, not a right
            pass
        finally:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)
        return ""

    def start(
        self,
        *,
        accelerator: Optional[str] = None,
        ready_timeout: float = READY_TIMEOUT_SECONDS,
        on_progress: Optional[Callable[[str], None]] = print,
    ) -> None:
        """Create the bus, submit the agent, wait until the kernel answers.

        Args:
            accelerator: A Kaggle accelerator name, for a GPU session.
            ready_timeout: How long to wait for the agent to boot.
            on_progress: Told what is happening while the job boots — the
                submitted job's URL, its status while queued, the moment the
                kernel answers. ``print`` by default, since whoever starts a
                live session is watching; a service passes ``None``.
        """
        say = on_progress or (lambda message: None)
        username, agent_env = resolve_kaggle_credentials()
        self._username = username
        base = f"code-sandbox-live-{self._session_id}"
        self._c2k = f"{username}/{base}-c2k"
        self._k2c = f"{username}/{base}-k2c"
        # Both ends of the bus exist before the agent boots, so neither side
        # ever has to create what the other is already polling.
        say(f"[kaggle-live] creating the private dataset bus ({base})…")
        self._write_bus(self._c2k, {"seq": 0}, create=True)
        self._write_bus(self._k2c, {"seq": -1}, create=True)

        agent = build_agent_code(agent_env, self._c2k, self._k2c, poll_seconds=self._poll)
        submitted = self._executor.execute(
            agent,
            slug=f"{base}-agent",
            kernel_type="script",
            wait=False,
            enable_internet=True,
            accelerator=accelerator,
            download_output=False,
        )
        self._slug = getattr(submitted, "slug", None)
        say(
            f"[kaggle-live] agent submitted: https://www.kaggle.com/code/{self._slug}\n"
            f"[kaggle-live] waiting for the kernel — a queued job takes minutes…"
        )

        started = time.monotonic()
        deadline = started + ready_timeout
        last_status: Optional[str] = None
        last_note = started
        while time.monotonic() < deadline:
            message = self._read_bus(self._k2c)
            if message and message.get("seq", -1) >= 0:
                say(f"[kaggle-live] the kernel is up ({time.monotonic() - started:.0f}s).")
                return
            # The job's own status: a dead agent must fail NOW, with its log,
            # not at the end of a fifteen-minute timeout.
            status: Optional[str] = None
            try:
                status = self._executor.status(self._slug) if self._slug else None
            except Exception:  # noqa: BLE001 — status is telemetry, not truth
                pass
            if status in ("ERROR", "CANCEL_ACKNOWLEDGED", "COMPLETE"):
                tail = self._agent_log_tail()
                hint = ""
                if "name resolution" in tail or "Failed to resolve" in tail:
                    # The job asked for internet and ran without it: Kaggle
                    # grants kernels internet only to phone-verified accounts.
                    hint = (
                        "\nThe job ran WITHOUT internet although it asked for "
                        "it — Kaggle only grants kernels internet once the "
                        "account is phone-verified: kaggle.com/settings, "
                        "'Phone verification'."
                    )
                raise RuntimeError(
                    f"The Kaggle agent job ended ({status}) before its kernel "
                    f"answered — https://www.kaggle.com/code/{self._slug}"
                    + (f"\n--- job log (tail) ---\n{tail}" if tail else "")
                    + hint
                )
            now = time.monotonic()
            if status != last_status or now - last_note >= 30:
                say(
                    f"[kaggle-live] still waiting — job {status or 'submitted'}, "
                    f"{now - started:.0f}s elapsed"
                )
                last_status, last_note = status, now
            time.sleep(self._poll)
        raise TimeoutError(
            f"The Kaggle live agent did not come up within {ready_timeout:.0f}s "
            f"(job {self._slug!r}). It may still be queued — a GPU job queues "
            "longest — or the account may not allow internet-enabled kernels."
        )

    def execute(self, code: str, timeout: Optional[float] = None) -> dict:
        """Run one snippet on the live kernel and return its reply."""
        if self._c2k is None or self._k2c is None:
            raise RuntimeError("The live session is not started.")
        self._seq += 1
        self._write_bus(self._c2k, {"seq": self._seq, "op": "execute", "code": code}, create=False)
        submitted = time.monotonic()
        deadline = submitted + (timeout or 600.0)
        while time.monotonic() < deadline:
            message = self._read_bus(self._k2c)
            if message and message.get("seq") == self._seq:
                return message.get("reply", {"status": "error", "outputs": []})
            # The wait between polls is latency the user feels on every turn:
            # poll fast while the answer is due, settle down only when the
            # turn has clearly become a long-running one.
            fast = time.monotonic() - submitted < FAST_WINDOW_SECONDS
            time.sleep(FAST_POLL_SECONDS if fast else self._poll)
        raise TimeoutError(f"No reply from the live kernel within {timeout or 600.0:.0f}s.")

    def stop(self, shutdown_kernel: bool = True) -> None:
        """Tell the agent to go; the datasets remain, small and private."""
        if shutdown_kernel and self._c2k:
            self._seq += 1
            try:
                self._write_bus(self._c2k, {"seq": self._seq, "op": "shutdown"}, create=False)
            except Exception:
                # The agent's idle timeout is the backstop.
                pass

    def get_variable(self, name: str) -> Any:
        raise NotImplementedError(
            "Variables of a live Kaggle kernel are read by running code on it."
        )
