# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Cloudflare sandbox implementation.

`Cloudflare Sandboxes <https://developers.cloudflare.com/sandbox/>`_ run
untrusted code in containers on Cloudflare's edge. The SDK for them is a
Workers binding, written in TypeScript, and a Python process cannot hold one:
``getSandbox(env.Sandbox, id)`` only means something inside a Worker.

What a Python process CAN talk to is the SANDBOX BRIDGE — a small Worker
Cloudflare publishes as a reference implementation, which exposes the SDK as
an HTTP API — so that is what this variant drives. It is deployed once per
account with

.. code-block:: bash

    npm create cloudflare -- sandbox-bridge \\
        --template=cloudflare/sandbox-sdk/bridge/worker

which returns a URL and generates a key. Both are what this variant needs:
``CLOUDFLARE_SANDBOX_API_URL`` and ``CLOUDFLARE_SANDBOX_API_KEY``.

The bridge offers a container and ``exec`` — one process per call, with its
output streamed back as server-sent events — and no way to feed a process
stdin after it has started. A namespace therefore CANNOT be held between
calls the way the CoreWeave and Modal variants hold one: each snippet runs in
a process of its own, and ``x = 1`` in one call is gone by the next. Combine
statements into a single snippet when they need to share state, or keep the
state in a file — the sandbox's filesystem does persist. Rich display data
has no channel either. The value of a trailing expression is reported.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import quote

from .base import Sandbox
from .exceptions import (
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxExecutionError,
    SandboxNotStartedError,
)
from .filesystem import SandboxFilesystem
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    Logs,
    OutputHandler,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

#: Where the bridge Worker answers, and the key it was deployed with.
API_URL_ENV_VAR = "CLOUDFLARE_SANDBOX_API_URL"
API_KEY_ENV_VAR = "CLOUDFLARE_SANDBOX_API_KEY"

#: The Python inside the container the bridge starts.
DEFAULT_PYTHON = "python3"

#: The program each snippet is run by. It takes its request as its one
#: argument rather than on stdin — the bridge starts a process and streams its
#: output, and gives nothing to write to — and answers with one JSON line, so
#: that what the code itself printed stays separable from the reply.
_RUNNER_SOURCE = """
import ast, contextlib, io, json, sys, traceback

request = json.loads(sys.argv[1])
out, err = io.StringIO(), io.StringIO()
reply = {"status": "ok"}
try:
    tree = ast.parse(request.get("code", ""), mode="exec")
    trailing = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = ast.Expression(tree.body.pop(-1).value)
    namespace = {"__name__": "__main__"}
    namespace.update(request.get("globals") or {})
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        if tree.body:
            exec(compile(tree, "<sandbox>", "exec"), namespace)
        if trailing is not None:
            value = eval(compile(trailing, "<sandbox>", "eval"), namespace)
            if value is not None:
                reply["result"] = repr(value)
except BaseException as error:
    reply["status"] = "error"
    reply["error"] = {
        "name": type(error).__name__,
        "value": str(error),
        "traceback": traceback.format_exc(),
    }
reply["stdout"] = out.getvalue()
reply["stderr"] = err.getvalue()
sys.stdout.write("\\n" + json.dumps(reply))
"""


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:
        raise SandboxConfigurationError(
            "httpx is required for CloudflareSandbox. Install it with: "
            "pip install code-sandboxes[cloudflare]"
        ) from exc
    return httpx


def _sse_events(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    """The (event, data) pairs of a server-sent-event stream.

    Written here rather than taken from a library because the bridge's stream
    is the whole of its execution protocol and this package should not grow a
    dependency for twenty lines. A record is a run of ``field: value`` lines
    ended by a blank one; ``data`` may appear more than once in a record, and
    the pieces are joined with newlines, as the specification says.
    """
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
            continue
        if line.startswith(":"):
            # A comment, which the bridge sends as a keep-alive.
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


class _CloudflareFilesystem(SandboxFilesystem):
    """The filesystem of a Cloudflare sandbox, read through the bridge.

    The base class reads a text file by running a snippet that binds the
    contents to a name and then reading that name back in a SECOND execution.
    That works wherever a namespace outlives a snippet, and here nothing does
    — so text reads and writes are served by the bridge's own file endpoints
    instead, which is one round trip rather than two and needs no session at
    all. The binary forms already went this way: they call `_read_file` and
    `_write_file`, which this variant overrides.
    """

    def read(self, path: str) -> str:
        return self._sandbox._read_file(path).decode("utf-8")

    def write(self, path: str, content: str, make_dirs: bool = True) -> None:
        # `make_dirs` is not honoured by the bridge's PUT, which creates the
        # parents it needs; the argument is kept for the shared interface.
        self._sandbox._write_file(path, content.encode("utf-8"))


class CloudflareSandbox(Sandbox):
    """Sandbox backed by a Cloudflare container, through the sandbox bridge.

    Args:
        config: Optional sandbox configuration.
        api_url: Where the bridge Worker answers, e.g.
            ``https://cloudflare-sandbox-bridge.example.workers.dev``. Read
            from ``CLOUDFLARE_SANDBOX_API_URL`` when omitted.
        api_key: The key the bridge was deployed with, sent as a bearer token.
            Read from ``CLOUDFLARE_SANDBOX_API_KEY`` when omitted. A bridge
            running locally for development may have none.
        python_executable: The Python inside the container.
        working_dir: Where snippets run. The container's ``/workspace`` unless
            the configuration names another.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        api_url: str | None = None,
        api_key: str | None = None,
        python_executable: str = DEFAULT_PYTHON,
        working_dir: str | None = None,
        **kwargs,
    ):
        super().__init__(config)
        self._api_url = (api_url or os.environ.get(API_URL_ENV_VAR) or "").rstrip("/")
        self._api_key = api_key or os.environ.get(API_KEY_ENV_VAR) or ""
        self._python_executable = python_executable
        self._working_dir = working_dir or self.config.working_dir or "/workspace"
        self._client: Any | None = None
        self._sandbox_id: str | None = None
        self._execution_count = 0
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """The environments this provider ships.

        A Cloudflare sandbox is one shape — the container the bridge Worker
        was deployed with — so there is one environment, and it is named after
        the bridge rather than after a machine.
        """
        return [
            SandboxEnvironment(
                name="cloudflare-default",
                title="Cloudflare",
                language="python",
                owner="cloudflare",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "cloudflare"},
            ),
        ]

    def start(self) -> None:
        if self._started:
            return

        # What this variant can never do comes first: a caller who asked for a
        # GPU, or for a network it cannot restrict, should hear that before
        # being sent to deploy a bridge for a request that was impossible
        # anyway.
        self._refuse_what_cannot_be_honoured()

        if not self._api_url:
            raise SandboxConfigurationError(
                "CloudflareSandbox needs the URL of a deployed sandbox bridge: "
                f"set {API_URL_ENV_VAR}, or pass api_url=. Deploy one with "
                "`npm create cloudflare -- sandbox-bridge "
                "--template=cloudflare/sandbox-sdk/bridge/worker`."
            )

        self._client = self.build_client()
        response = self._client.post("/v1/sandbox")
        self._raise_for_status(response, "create a sandbox")
        self._sandbox_id = str(response.json()["id"])

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="cloudflare",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "cloudflare_sandbox_id": self._sandbox_id,
                "api_url": self._api_url,
                "working_dir": self._working_dir,
            },
            resources=ResourceConfig(),
            config=self.config,
        )
        self._started = True

    def _refuse_what_cannot_be_honoured(self) -> None:
        """What this variant can never do, judged from the configuration alone."""
        if self.config.gpu:
            raise SandboxConfigurationError(
                "Cloudflare sandboxes have no GPU, so gpu=" + repr(self.config.gpu) + " "
                "cannot be honoured. Use the daytona, coreweave or modal "
                "variant for a GPU."
            )
        # The bridge exposes no networking controls at all — no egress rules,
        # no allowlist, no switch — so a policy asked for here could only be
        # accepted and then not applied. A sandbox believed to be cut off from
        # the network while it is not is the failure that matters.
        if self.config.network_policy in ("none", "allowlist") or self.config.allowed_hosts:
            raise SandboxConfigurationError(
                f"Cloudflare sandboxes cannot restrict the network, so "
                f"network_policy={self.config.network_policy!r} cannot be "
                "honoured. Use the e2b variant to cut a sandbox off, or the "
                "daytona or coreweave variant for an allowlist."
            )

    def build_client(self) -> Any:
        """An HTTP client for the bridge, with no sandbox behind it yet.

        Separate from :meth:`start` because talking to the bridge and HAVING a
        sandbox are different things: the manager deletes a sandbox by id and
        asks after one by id, and creating a throwaway container merely to get
        a client to do it with would leave a container running and billed.
        """
        if not self._api_url:
            raise SandboxConfigurationError(
                "CloudflareSandbox needs the URL of a deployed sandbox bridge: "
                f"set {API_URL_ENV_VAR}, or pass api_url=. Deploy one with "
                "`npm create cloudflare -- sandbox-bridge "
                "--template=cloudflare/sandbox-sdk/bridge/worker`."
            )
        httpx = _import_httpx()
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return httpx.Client(
            base_url=self._api_url,
            headers=headers,
            # Creating a container is slower than the default five seconds,
            # and an execution runs for as long as the caller asked it to.
            timeout=httpx.Timeout(30.0, read=None),
        )

    def _raise_for_status(self, response: Any, what: str) -> None:
        """Say which call failed, and what the bridge said about it.

        ``raise_for_status`` names the URL and the code and nothing else; a
        401 from the bridge means the key is wrong, and a caller should be
        able to read that from the message rather than infer it.
        """
        if response.status_code < 400:
            return
        detail = (response.text or "").strip()
        if response.status_code in (401, 403):
            detail = (
                f"{detail} — the bridge refused the key. Check "
                f"{API_KEY_ENV_VAR} against the secret it was deployed with."
            ).strip(" —")
        raise SandboxConnectionError(
            self._api_url,
            f"could not {what}: HTTP {response.status_code}. {detail}".strip(),
        )

    def stop(self) -> None:
        if not self._started:
            return
        if self._client is not None and self._sandbox_id:
            try:
                self._client.delete(f"/v1/sandbox/{quote(self._sandbox_id)}")
            except Exception:
                logger.debug(
                    "Ignoring error while destroying the Cloudflare sandbox", exc_info=True
                )
        if self._client is not None:
            with_close = getattr(self._client, "close", None)
            if with_close is not None:
                with_close()
        self._client = None
        self._sandbox_id = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def is_running(self) -> bool:
        """Whether the container is still up.

        Cloudflare puts a sandbox to sleep on its own schedule, so a sandbox
        this process created may be gone without this process having asked for
        it. The bridge answers the question directly.
        """
        if not self._started or self._client is None or not self._sandbox_id:
            return False
        try:
            response = self._client.get(f"/v1/sandbox/{quote(self._sandbox_id)}/running")
            return bool(response.json().get("running"))
        except Exception:
            return False

    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        if not self._started or self._client is None or not self._sandbox_id:
            raise SandboxNotStartedError()
        if language != "python":
            raise ValueError(f"CloudflareSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1
        seconds = timeout if timeout is not None else self.config.timeout
        # The environment of the CONFIGURATION as well as the one asked for
        # here. The bridge takes no environment when it creates a sandbox — it
        # has nowhere to put one — so `env_vars` would otherwise be accepted
        # and then silently dropped from every snippet. Per-call values win.
        environment = {**(self.config.env_vars or {}), **(envs or {})}
        request = json.dumps({"code": _with_envs(code, environment)})

        try:
            stdout, stderr, exit_code = self._exec(
                [self._python_executable, "-u", "-c", _RUNNER_SOURCE, request],
                seconds,
            )
        except Exception as error:
            # The bridge or the container, not the code.
            return ExecutionResult(
                execution_ok=False,
                execution_error=f"Failed to execute code on Cloudflare: {error}",
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        reply = _reply_of(stdout)
        if reply is None:
            # The runner never got to answer: the process was killed, or the
            # image has no Python where it was looked for. What the container
            # said is the only evidence, so it is what gets reported.
            return ExecutionResult(
                execution_ok=False,
                execution_error=(
                    f"The Cloudflare sandbox answered with nothing this package "
                    f"could read (exit code {exit_code}). {stderr or stdout}".strip()
                ),
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        return self._execution_result(
            reply, context, started_at, on_stdout, on_stderr, on_result, on_error
        )

    def _exec(self, argv: list[str], timeout: float) -> tuple[str, str, int | None]:
        """One process in the container, and what it wrote.

        The bridge answers with a server-sent-event stream: the output as it
        is written, base64-encoded, and one terminal event — the exit code, or
        an error that stopped it from running at all.
        """
        stdout: list[bytes] = []
        stderr: list[bytes] = []
        exit_code: int | None = None
        failure: str | None = None

        with self._client.stream(
            "POST",
            f"/v1/sandbox/{quote(self._sandbox_id or '')}/exec",
            json={
                "argv": argv,
                "cwd": self._working_dir,
                "timeout_ms": max(1, round(timeout * 1000)),
            },
            timeout=timeout + 30.0,
        ) as response:
            if response.status_code >= 400:
                response.read()
                self._raise_for_status(response, "run a command")
            for event, data in _sse_events(response.iter_lines()):
                if event == "stdout":
                    stdout.append(_decode(data))
                elif event == "stderr":
                    stderr.append(_decode(data))
                elif event == "exit":
                    with_code = _json_or_none(data) or {}
                    exit_code = with_code.get("exit_code")
                elif event == "error":
                    with_error = _json_or_none(data) or {}
                    failure = str(with_error.get("error") or data)

        if failure is not None:
            raise SandboxExecutionError("SandboxError", failure)
        return (
            b"".join(stdout).decode("utf-8", errors="replace"),
            b"".join(stderr).decode("utf-8", errors="replace"),
            exit_code,
        )

    def _execution_result(
        self,
        reply: dict,
        context: Context | None,
        started_at: float,
        on_stdout: OutputHandler[OutputMessage] | None,
        on_stderr: OutputHandler[OutputMessage] | None,
        on_result: OutputHandler[Result] | None,
        on_error: OutputHandler[CodeError] | None,
    ) -> ExecutionResult:
        """One reply of the runner, as an `ExecutionResult`.

        The runner collects the output and answers once, so the callbacks are
        called here, in order, on the lines it carried. A caller that streams
        sees the lines it would have seen, later.
        """
        now = time.time()
        stdout_messages: list[OutputMessage] = []
        for line in _lines(reply.get("stdout")):
            message = OutputMessage(line=line, timestamp=now, error=False)
            stdout_messages.append(message)
            if on_stdout:
                on_stdout(message)

        stderr_messages: list[OutputMessage] = []
        for line in _lines(reply.get("stderr")):
            message = OutputMessage(line=line, timestamp=now, error=True)
            stderr_messages.append(message)
            if on_stderr:
                on_stderr(message)

        results: list[Result] = []
        if reply.get("result") is not None:
            value = Result(data={"text/plain": reply["result"]}, is_main_result=True)
            results.append(value)
            if on_result:
                on_result(value)

        code_error: CodeError | None = None
        error = reply.get("error")
        if reply.get("status") == "error" and isinstance(error, dict):
            code_error = CodeError(
                name=error.get("name") or "Error",
                value=error.get("value") or "",
                traceback=error.get("traceback") or "",
            )
            if on_error:
                on_error(code_error)

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            execution_count=self._execution_count,
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=now,
        )

    def _do_interrupt(self) -> bool:
        """The bridge takes no interrupt; a timeout is the only stop."""
        return False

    @property
    def files(self) -> SandboxFilesystem:
        """The bridge-backed filesystem, rather than the base's code-driven one."""
        if self._files is None:
            self._files = _CloudflareFilesystem(self)
        return self._files

    def get_variable(self, name: str, context: Context | None = None) -> Any:
        """Refused: reading a variable takes a session, and there is none.

        The base class reads a variable in TWO executions — it binds the value
        to a name of its own, then reads that name back — which every other
        variant serves because its namespace outlives a snippet. Here the
        first execution's process is gone before the second starts, so the
        read would fail as "no such variable": true of the name, and entirely
        misleading about the reason.
        """
        raise SandboxConfigurationError(
            f"A Cloudflare sandbox runs each snippet in a process of its own, "
            f"so there is no session to read {name!r} from. Have the snippet "
            "print what you need and read it from the execution, or keep it "
            "in a file — the filesystem of the sandbox does persist."
        )

    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """The base class reaches this only through `get_variable`, which is
        refused above; it is implemented so the abstraction stays satisfied."""
        return self.get_variable(name, context)

    def _set_internal_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        if not self._started:
            raise SandboxNotStartedError()
        try:
            json.dumps(value)
        except TypeError as error:
            raise SandboxConfigurationError(
                f"A Cloudflare sandbox runs elsewhere, so {name!r} has to cross "
                "as JSON and this value cannot be encoded. Build it inside the "
                "sandbox with run_code instead."
            ) from error
        raise SandboxConfigurationError(
            "A Cloudflare sandbox runs each snippet in a process of its own, "
            f"so {name!r} would be gone before the next one reads it. Set it "
            "inside the snippet that uses it, or keep it in a file — the "
            "filesystem of the sandbox does persist."
        )

    def _write_file(self, path: str, content: bytes) -> None:
        """Straight to the filesystem of the container, not through the code."""
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        response = self._client.put(self._file_url(path), content=content)
        self._raise_for_status(response, f"write {path}")

    def _read_file(self, path: str) -> bytes:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        response = self._client.get(self._file_url(path))
        if response.status_code == 404:
            raise FileNotFoundError(f"Could not read file: {path}")
        self._raise_for_status(response, f"read {path}")
        return bytes(response.content)

    def _file_url(self, path: str) -> str:
        """The bridge's URL for one file of this sandbox.

        The path travels as the tail of the URL, so its separators must stay
        separators while everything else about it is escaped.
        """
        return f"/v1/sandbox/{quote(self._sandbox_id or '')}/file/{quote(path.lstrip('/'))}"


def _decode(data: str) -> bytes:
    """One base64 payload of the stream, or nothing when it is not one."""
    try:
        return base64.b64decode(data)
    except Exception:
        return data.encode()


def _json_or_none(data: str) -> dict | None:
    try:
        parsed = json.loads(data)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _reply_of(stdout: str) -> dict | None:
    """The runner's JSON reply, out of everything the process wrote.

    It is the LAST line: the runner writes it after a newline of its own, so
    whatever the container printed on its own account — a warning from the
    interpreter, a message from an entrypoint — comes before it.
    """
    for line in reversed(stdout.splitlines()):
        reply = _json_or_none(line)
        if reply is not None:
            return reply
    return None


def _lines(raw: Any) -> list[str]:
    """The lines of one stream, without the empty one a trailing newline makes."""
    if not raw:
        return []
    return str(raw).splitlines()


def _with_envs(code: str, envs: dict[str, str] | None) -> str:
    """The snippet, with the environment it asked for set first."""
    if not envs:
        return code
    assignments = "".join(
        f"_code_sandboxes_os.environ[{key!r}] = {value!r}\n" for key, value in envs.items()
    )
    return f"import os as _code_sandboxes_os\n{assignments}del _code_sandboxes_os\n{code}"
