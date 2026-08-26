# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Daytona sandbox implementation.

`Daytona <https://www.daytona.io/docs/>`_ runs code in cloud sandboxes that
start in well under a second. This variant drives one through its CODE
INTERPRETER — ``sandbox.code_interpreter`` — and not through
``sandbox.process.code_run``: the interpreter holds a namespace per context,
so ``x = 1`` in one call and ``print(x)`` in the next behave the way they do
in every other variant of this package, while ``code_run`` is a fresh process
per snippet and would not.

What the interpreter answers with is stdout, stderr, and the error when the
code raised — there is no execute_result on the wire. The value of a trailing
expression is therefore captured here rather than lost (see
:func:`_capture_trailing_value`). Rich display data — a figure, an HTML repr —
has no channel at all, and is not reported.
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import math
import textwrap
import time
from typing import Any

from .base import Sandbox
from .contents import (
    FILESYSTEM_PRIMITIVES,
    ContentAttachmentSpec,
    ContentCapabilities,
    ContentManifest,
    CreationTimeMounts,
    LocalBridgeCapability,
    PreparedAttachment,
)
from .exceptions import (
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxNotStartedError,
    VariableNotFoundError,
)
from .jupyter_ingress import preparation_command, resolved_options, websocket_url
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    JupyterServerEndpoint,
    JupyterServerOptions,
    Logs,
    OutputHandler,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
    gpu_memory,
)

logger = logging.getLogger(__name__)

#: What every sandbox this package creates is labelled with, so the ones it
#: made can be told from the rest of an organization's.
CREATED_BY_LABEL = "code-sandboxes"

#: The line the capture below writes, and the only line `run_code` takes back
#: out of the stream. Long and specific on purpose: a line of the code's own
#: output that began with it would be read as a value and disappear.
_VALUE_MARKER = "__code_sandboxes_daytona_value__:"

#: Names the capture binds inside the sandbox. Prefixed rather than short:
#: they share a namespace with everything the caller defines.
_VALUE_VAR = "_code_sandboxes_value"
_JSON_MOD = "_code_sandboxes_json"
_SYS_MOD = "_code_sandboxes_sys"


def _emit_text(expression: str) -> str:
    """Code writing the string `expression` evaluates to, on one line.

    JSON-encoded, so a value whose text runs over several lines still arrives
    as ONE line of stdout — which is what makes it separable from what the
    code itself printed.

    Written through ``sys.stdout`` rather than ``print``, and through modules
    imported under prefixed names: the code that just ran is free to have
    rebound ``print``, ``sys`` or ``json`` to anything it likes. The names are
    dropped again afterwards — the namespace they were bound in is the one the
    caller goes on working in, and `dir()` should not answer with ours.
    """
    return (
        f"import json as {_JSON_MOD}, sys as {_SYS_MOD}\n"
        f"{_SYS_MOD}.stdout.write({_VALUE_MARKER!r} + {_JSON_MOD}.dumps({expression}) + '\\n')\n"
        f"{_SYS_MOD}.stdout.flush()\n"
        f"del {_JSON_MOD}, {_SYS_MOD}\n"
    )


def _split_marker(line: str) -> tuple[str, str | None]:
    """The line as the caller should see it, and the value it carried.

    The marker is looked for ANYWHERE in the line, not only at its start. It
    is written last and nothing puts a newline in front of it, so code that
    left the stream mid-line — ``sys.stdout.write("x")``, a `print` with
    ``end=""`` — has the marker land right after its own text. Reading the
    line whole would have lost the value and shown the marker to the caller.
    """
    at = line.find(_VALUE_MARKER)
    if at < 0:
        return line, None
    try:
        value = json.loads(line[at + len(_VALUE_MARKER) :])
    except ValueError:
        return line, None
    if not isinstance(value, str):
        return line, None
    return line[:at], value


def _capture_trailing_value(code: str) -> str:
    """Make the value of a trailing expression readable, or leave the code alone.

    The interpreter of Daytona reports what the code PRINTED; the value of a
    last expression — the ``2`` of a cell holding ``1 + 1`` — is evaluated and
    dropped. Every other variant of this package answers with it, and the REPL
    and the tables that read `ExecutionResult.text` expect it, so it is asked
    for: the expression is bound to a name and its repr written to stdout
    behind a marker, which :meth:`DaytonaSandbox.run_code` takes back out.

    What comes BEFORE the expression is passed through untouched, byte for
    byte, and the expression stays on the line it was written on. That is the
    point of slicing the source rather than rewriting the tree: a traceback
    naming line 7 has to still mean line 7 of what the caller submitted.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Not ours to report: the sandbox raises it, with its own message and
        # its own position.
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if any(
        isinstance(node, (ast.Await, ast.Yield, ast.YieldFrom, ast.Starred))
        for node in ast.walk(last.value)
    ):
        # Binding these to a name means something else, or does not parse at
        # all. The code runs as written, and answers with its output alone.
        return code

    lines = code.splitlines(keepends=True)
    line = lines[last.lineno - 1]
    # `col_offset` counts the BYTES of the utf-8 line, not its characters.
    column = len(line.encode()[: last.col_offset].decode(errors="ignore"))
    start = sum(len(each) for each in lines[: last.lineno - 1]) + column
    head, expression = code[:start], code[start:]
    return (
        f"{head}{_VALUE_VAR} = ({expression})\n"
        f"if {_VALUE_VAR} is not None:\n"
        f"{textwrap.indent(_emit_text(f'repr({_VALUE_VAR})'), '    ')}"
        f"del {_VALUE_VAR}\n"
    )


class _Lines:
    """Whole lines out of a stream that arrives in chunks.

    The interpreter sends stdout as it is written, which cuts wherever the
    write did — mid-line as readily as at its end. A marker can only be
    recognised on a complete line, and an `OutputMessage` is a line, so the
    remainder of a chunk is held until the rest of it turns up.
    """

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> list[str]:
        self._pending += chunk
        *complete, self._pending = self._pending.split("\n")
        return complete

    def flush(self) -> list[str]:
        """Whatever never got its newline, once the execution is over."""
        rest, self._pending = self._pending, ""
        return [rest] if rest else []


def _gpu_types(flavors: str, daytona: Any) -> list[Any]:
    """The Daytona GPUs named, in the order they are preferred.

    One name asks for one GPU; several, comma-separated, ask for the first of
    them that Daytona can find — `"H100,H200,RTX-4090"` takes an H100 when
    there is one and an H200 when there is not, rather than failing. That is
    Daytona's own fallback, and it matters most for spot capacity, where what
    is free changes minute to minute.

    The flavours differ from one provider to the next — a `T4` is Modal's
    vocabulary, not Daytona's — so a name that means nothing here is said so
    at once, rather than reaching the API as an invalid enum.
    """
    offered = [
        candidate
        for candidate in daytona.GpuType
        if not candidate.value.lower().startswith("unknown")
    ]
    by_name = {candidate.value.upper(): candidate for candidate in offered}
    wanted: list[Any] = []
    for flavor in flavors.split(","):
        name = flavor.strip().upper().replace("_", "-")
        if not name:
            continue
        if name not in by_name:
            raise SandboxConfigurationError(
                f"Daytona has no GPU called {flavor.strip()!r}. It offers: "
                + ", ".join(candidate.value for candidate in offered)
                + "."
            )
        if by_name[name] not in wanted:
            wanted.append(by_name[name])
    return wanted


def _asks_for_a_gpu(resources: Any | None) -> bool:
    """Whether this specification carries a GPU.

    `Resources` sets `gpu` to a count, so "no GPU" arrives as either no
    specification at all or a count of zero.
    """
    return resources is not None and getattr(resources, "gpu", None) not in (None, 0)


def _import_daytona() -> Any:
    try:
        import daytona
    except ImportError as exc:
        raise SandboxConfigurationError(
            "daytona is required for DaytonaSandbox. Install it with: "
            "pip install code-sandboxes[daytona]"
        ) from exc
    return daytona


class DaytonaSandbox(Sandbox):
    """Sandbox backed by a Daytona cloud sandbox.

    Args:
        config: Optional sandbox configuration.
        api_key: Daytona API key. Read from ``DAYTONA_API_KEY`` when omitted.
        api_url: Daytona API URL. Read from ``DAYTONA_API_URL`` when omitted,
            which itself defaults to ``https://app.daytona.io/api``.
        target: Region the sandbox runs in. ``DAYTONA_TARGET`` when omitted.
        jwt_token: The other way of authenticating, with ``organization_id``.
        organization_id: Organization the JWT belongs to.
        snapshot: Name of the Daytona snapshot to create from. The default
            snapshot of the organization when omitted.
        image: A ``daytona.Image`` to create from instead of a snapshot.
            Resources — cpu, memory, a GPU — can only be asked for of an
            image, so asking for any of them builds one when none is given.
        python_version: Python of that image. Daytona's own default when
            omitted.
        gpu_count: How many GPUs to attach when one is asked for.
        spot: Whether to run on PREEMPTIBLE GPU capacity. Much cheaper than
            on-demand and outside the organization's GPU quota, at the price
            of being reclaimed without warning at any moment — see
            :meth:`preempted_at`. GPU-only, and Daytona refuses it for a
            sandbox that asks for none.
        delete_on_stop: Whether :meth:`stop` DELETES the sandbox, which is the
            default and what a ``with`` block should do, or merely stops it —
            leaving it in the organization, to be started again.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        jwt_token: str | None = None,
        organization_id: str | None = None,
        snapshot: str | None = None,
        image: Any | None = None,
        python_version: str | None = None,
        gpu_count: int = 1,
        spot: bool = False,
        delete_on_stop: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        self._api_key = api_key
        self._api_url = api_url
        self._target = target
        self._jwt_token = jwt_token
        self._organization_id = organization_id
        self._snapshot = snapshot
        self._image = image
        self._python_version = python_version
        self._gpu_count = gpu_count
        self._spot = spot
        self._delete_on_stop = delete_on_stop
        self._daytona: Any | None = None
        self._sandbox: Any | None = None
        #: The Daytona interpreter context standing for each of ours, made on
        #: first use — creating one is a round trip, and most callers use the
        #: default namespace and never need a second.
        self._contexts: dict[str, Any] = {}
        self._execution_count = 0
        self._jupyter_endpoint: JupyterServerEndpoint | None = None
        #: Volumes to mount, which Daytona takes only when the sandbox is
        #: created: `params.volumes=[VolumeMount(...)]`, nothing afterwards.
        self._volume_mounts = CreationTimeMounts()
        self._extra_kwargs = kwargs

    # -- Contents attachments ---------------------------------------------

    def content_capabilities(self) -> ContentCapabilities:
        return ContentCapabilities(
            provider="daytona",
            mount=True,
            bucket_mount=False,
            materialize=True,
            client=True,
            local_bridge_mount=LocalBridgeCapability(supported=False),
            filesystem_primitives=list(FILESYSTEM_PRIMITIVES),
        )

    def configure_contents(self, manifest: ContentManifest) -> None:
        super().configure_contents(manifest)
        self._volume_mounts.request_all(manifest)

    def _prepare_mount(self, spec: ContentAttachmentSpec, *, reconcile: bool) -> PreparedAttachment:
        del reconcile
        return self._volume_mounts.prepare(self, spec, provider="daytona")

    def _forget_mount(self, spec: ContentAttachmentSpec) -> None:
        self._volume_mounts.forget(spec)

    def prepare_jupyter_server(
        self, options: JupyterServerOptions | None = None
    ) -> JupyterServerEndpoint:
        """Prepare Jupyter and expose it through Daytona's preview ingress."""
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        if self._jupyter_endpoint is not None:
            return self._jupyter_endpoint

        value = resolved_options(options)
        response = self._sandbox.process.exec(
            preparation_command(value), timeout=max(1, math.ceil(value.install_timeout))
        )
        if getattr(response, "exit_code", 0) not in (0, None):
            raise SandboxConfigurationError(
                "Could not install and start Jupyter Server in the Daytona sandbox: "
                + str(getattr(response, "result", "unknown error"))
            )
        preview = self._sandbox.get_preview_link(value.port)
        self._jupyter_endpoint = JupyterServerEndpoint(
            port=value.port,
            http_url=preview.url.rstrip("/"),
            websocket_url=websocket_url(preview.url.rstrip("/")),
            headers={"X-Daytona-Preview-Token": preview.token},
            query={"token": value.token or ""},
        )
        return self._jupyter_endpoint

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """The environments this provider ships.

        Daytona takes a machine specification per sandbox rather than a
        catalogue of named ones, so what is offered here are the shapes worth
        naming — a plain sandbox, one with a GPU, and one on the preemptible
        capacity a GPU can be had cheaply on — and choosing an environment
        stays what it is everywhere else: choosing between named things.

        The shape is asked for by argument, not by name: `gpu=` and `spot=`.
        """
        return [
            SandboxEnvironment(
                name="daytona-default",
                title="Daytona",
                language="python",
                owner="daytona",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "daytona", "gpu": None},
            ),
            SandboxEnvironment(
                name="daytona-gpu",
                title="Daytona GPU",
                language="python",
                owner="daytona",
                visibility="cloud",
                burning_rate=0.0,
                gpu="H100",
                gpu_count=1,
                gpu_memory=gpu_memory("H100"),
                metadata={"variant": "daytona", "gpu": "H100", "spot": False},
            ),
            SandboxEnvironment(
                name="daytona-gpu-spot",
                title="Daytona GPU (spot)",
                language="python",
                owner="daytona",
                visibility="cloud",
                burning_rate=0.0,
                gpu="H100",
                gpu_count=1,
                gpu_memory=gpu_memory("H100"),
                metadata={"variant": "daytona", "gpu": "H100", "spot": True},
            ),
        ]

    def start(self) -> None:
        if self._started:
            return

        daytona = _import_daytona()
        self._daytona = daytona.Daytona(self._client_config(daytona))
        self._sandbox = self._daytona.create(self._create_params(daytona))
        self._volume_mounts.created()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox.id,
            variant="daytona",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "daytona_sandbox_id": self._sandbox.id,
                "snapshot": getattr(self._sandbox, "snapshot", None),
                "target": getattr(self._sandbox, "target", None),
                "spot": bool(getattr(self._sandbox, "spot", self._spot)),
            },
            resources=ResourceConfig(
                cpu=getattr(self._sandbox, "cpu", None),
                memory=getattr(self._sandbox, "memory", None),
                gpu=getattr(self._sandbox, "gpu_type", None),
                gpu_count=getattr(self._sandbox, "gpu", None) or 1,
            ),
            config=self.config,
        )
        self._started = True

    def _client_config(self, daytona: Any) -> Any | None:
        """The client settings that were given, and nothing more.

        A field left out is a field the SDK reads from the environment —
        ``DAYTONA_API_KEY`` and the rest — so passing ``None`` for everything
        would not be the same as passing nothing.
        """
        settings = {
            "api_key": self._api_key,
            "api_url": self._api_url,
            "target": self._target,
            "jwt_token": self._jwt_token,
            "organization_id": self._organization_id,
        }
        given = {key: value for key, value in settings.items() if value}
        return daytona.DaytonaConfig(**given) if given else None

    def _create_params(self, daytona: Any) -> Any:
        """What to ask Daytona for, from the configuration of this sandbox."""
        common: dict[str, Any] = {"labels": self._labels()}
        if self.config.env_vars:
            common["env_vars"] = dict(self.config.env_vars)
        if self.config.idle_timeout:
            common["auto_stop_interval"] = max(1, round(self.config.idle_timeout / 60))
        if self.config.max_lifetime:
            common["ttl_minutes"] = max(1, round(self.config.max_lifetime / 60))
        common.update(self._network_params())
        if self._volume_mounts.requested:
            # One mount per path, in the order they were asked for: the
            # record is keyed by path, so asking twice cannot mount twice.
            common["volumes"] = [
                daytona.VolumeMount(volume_id=volume_id, mount_path=mount_path)
                for mount_path, volume_id in self._volume_mounts.requested.items()
            ]

        resources = self._resources(daytona)
        if _asks_for_a_gpu(resources):
            # Daytona will not create a GPU sandbox that outlives its stop:
            # "GPU sandboxes must be ephemeral; set autoDeleteInterval to 0".
            # It is a property of asking for a GPU at all, not of asking for
            # preemptible capacity — which is where this used to live, so an
            # on-demand `gpu=` was refused by the API on creation.
            common["auto_delete_interval"] = 0
        if self._spot:
            common.update(self._spot_params(resources))
        if self._image is not None or resources is not None:
            image = self._image
            if image is None:
                image = (
                    daytona.Image.debian_slim(self._python_version)
                    if self._python_version
                    else daytona.Image.debian_slim()
                )
            return daytona.CreateSandboxFromImageParams(image=image, resources=resources, **common)
        return daytona.CreateSandboxFromSnapshotParams(snapshot=self._snapshot, **common)

    def _spot_params(self, resources: Any | None) -> dict[str, Any]:
        """What asking for preemptible capacity commits the sandbox to.

        Spot is GPU-only — Daytona refuses it for a sandbox that asks for no
        GPU — and it is built from an IMAGE rather than from a snapshot. Both
        are said here rather than left to come back as an API error a caller
        cannot act on. Being ephemeral is asked for alongside the GPU itself,
        in `_create_params`, since Daytona requires it of every GPU sandbox
        and not only of the preemptible ones.
        """
        if not _asks_for_a_gpu(resources):
            raise SandboxConfigurationError(
                "spot=True asks for preemptible GPU capacity, so it needs a "
                "GPU: pass gpu=... (for example gpu='H100', or "
                "gpu='H100,H200' to fall back to the second when the first "
                "is unavailable)."
            )
        if self._snapshot is not None:
            raise SandboxConfigurationError(
                "spot=True builds from an image, which is what carries a "
                "machine specification; snapshot= cannot be used with it. "
                "Pass image=, or leave both out for a Debian image."
            )
        return {"spot": True}

    def _labels(self) -> dict[str, str]:
        """The metadata the sandbox carries in Daytona.

        The NAME of the sandbox goes here rather than into Daytona's own
        ``name``: that one is unique within an organization and is how a
        sandbox is addressed, so a second sandbox asking for a name already
        taken is a conflict rather than a second sandbox. Ours are generated,
        and a caller is free to repeat one.
        """
        labels = {"created-by": CREATED_BY_LABEL}
        if self.config.name:
            labels["name"] = self.config.name
        labels.update(self._tags)
        return labels

    def _network_params(self) -> dict[str, Any]:
        """What the network policy of the configuration means to Daytona."""
        policy = self.config.network_policy
        if policy == "none":
            return {"network_block_all": True}
        if policy == "allowlist":
            if not self.config.allowed_hosts:
                raise SandboxConfigurationError(
                    "network_policy='allowlist' needs allowed_hosts: a sandbox "
                    "allowed nothing is a sandbox with no network at all, which "
                    "is network_policy='none'."
                )
            return {"domain_allow_list": ",".join(self.config.allowed_hosts)}
        return {}

    def _resources(self, daytona: Any) -> Any | None:
        """The machine asked for, or nothing when the defaults will do."""
        # Whole units here, and never zero: a fraction of a core or of a
        # GiB is still a request, and rounding it away asks for a machine
        # with none of that resource rather than for one with the default.
        cpu = max(1, math.ceil(self.config.cpu_limit)) if self.config.cpu_limit else None
        memory = None
        if self.config.memory_limit:
            memory = max(1, math.ceil(self.config.memory_limit / 1024**3))
        gpu_types = _gpu_types(self.config.gpu, daytona) if self.config.gpu else []
        if cpu is None and memory is None and not gpu_types:
            return None
        return daytona.Resources(
            cpu=cpu,
            memory=memory,
            gpu=max(1, self._gpu_count) if gpu_types else None,
            # One name goes as one name and several as the ordered list
            # Daytona falls back along; a list of one would work too, but the
            # simple case should read like the simple case in the request.
            gpu_type=gpu_types[0] if len(gpu_types) == 1 else (gpu_types or None),
        )

    def stop(self) -> None:
        if not self._started:
            return
        if self._sandbox is not None:
            try:
                if self._delete_on_stop:
                    self._sandbox.delete()
                else:
                    self._sandbox.stop()
            except Exception:
                logger.debug("Ignoring error while stopping the Daytona sandbox", exc_info=True)
            self._sandbox = None
        self._daytona = None
        self._jupyter_endpoint = None
        self._contexts.clear()
        self._volume_mounts.stopped()
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def preempted_at(self) -> str | None:
        """When this spot sandbox was reclaimed, or `None` while it still runs.

        Daytona takes preemptible capacity back without warning — no signal,
        no webhook — so the only way to know is to ask, which this does
        freshly rather than from the record the sandbox was made with. A
        reclaimed sandbox stays retrievable for 24 hours, which is long
        enough to find out what became of it.

        Always `None` for a sandbox that is not on spot capacity: there is
        nothing that would reclaim it.
        """
        if self._sandbox is None:
            return None
        with contextlib.suppress(Exception):
            # A sandbox that has just gone away cannot always be asked; the
            # answer then is whatever was last known, which is honest.
            self._sandbox.refresh_data()
        evicted = getattr(self._sandbox, "spot_evicted_at", None)
        return str(evicted) if evicted else None

    def _failure_reason(self, error: Exception) -> str:
        """Why the execution did not happen, named as closely as it can be.

        A reclaimed spot sandbox fails the way a dropped connection does, and
        a caller reading "the websocket closed" has no way to tell that it was
        outbid rather than broken. Asking costs a round trip, so it is only
        asked of a sandbox that could have been reclaimed at all.
        """
        if self._spot:
            evicted = self.preempted_at()
            if evicted:
                return (
                    f"The spot sandbox was reclaimed at {evicted}: preemptible "
                    "capacity is taken back when on-demand capacity needs it. "
                    "Run this again on a new sandbox, or ask for on-demand "
                    "capacity with spot=False."
                )
        return f"Failed to execute code on Daytona: {error}"

    def _interpreter_context(self, context: Context | None) -> Any | None:
        """The Daytona context one of ours stands for, made on first use.

        ``None`` — and the default context, which is the same namespace —
        is the shared one of the sandbox. Anything else is a context Daytona
        keeps apart, so :meth:`create_context` really does isolate.
        """
        if context is None or context.id == "default":
            return None
        existing = self._contexts.get(context.id)
        if existing is None:
            existing = self._sandbox.code_interpreter.create_context(cwd=context.cwd)
            self._contexts[context.id] = existing
        return existing

    def run_code(  # noqa: C901
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
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        if language != "python":
            raise ValueError(f"DaytonaSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1

        values: list[str] = []
        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        stdout_lines, stderr_lines = _Lines(), _Lines()

        def take_stdout(line: str) -> None:
            # The marker is OURS: the caller never sees it, neither streamed
            # nor in the logs it reads afterwards. What shared its line does
            # reach them — that part is the code's own output.
            text, value = _split_marker(line)
            if value is not None:
                values.append(value)
                if not text:
                    return
            message = OutputMessage(line=text, timestamp=time.time(), error=False)
            stdout_messages.append(message)
            if on_stdout:
                on_stdout(message)

        def take_stderr(line: str) -> None:
            message = OutputMessage(line=line, timestamp=time.time(), error=True)
            stderr_messages.append(message)
            if on_stderr:
                on_stderr(message)

        def feed_stdout(chunk: Any) -> None:
            for line in stdout_lines.feed(chunk.output):
                take_stdout(line)

        def feed_stderr(chunk: Any) -> None:
            for line in stderr_lines.feed(chunk.output):
                take_stderr(line)

        seconds = timeout if timeout is not None else self.config.timeout
        try:
            reply = self._sandbox.code_interpreter.run_code(
                _capture_trailing_value(code),
                context=self._interpreter_context(context),
                on_stdout=feed_stdout,
                on_stderr=feed_stderr,
                envs=envs,
                # Daytona counts in whole seconds, and reads 0 as "no limit".
                timeout=max(0, math.ceil(seconds)),
            )
        except Exception as error:
            return ExecutionResult(
                execution_ok=False,
                execution_error=self._failure_reason(error),
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        for line in stdout_lines.flush():
            take_stdout(line)
        for line in stderr_lines.flush():
            take_stderr(line)

        results: list[Result] = []
        for text in values:
            value = Result(data={"text/plain": text}, is_main_result=True)
            results.append(value)
            if on_result:
                on_result(value)

        code_error: CodeError | None = None
        if reply.error is not None:
            code_error = CodeError(
                name=reply.error.name or "Error",
                value=reply.error.value or "",
                traceback=reply.error.traceback or "",
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
            completed_at=time.time(),
        )

    def _do_interrupt(self) -> bool:
        """Daytona's interpreter takes no interrupt; a timeout is the only stop."""
        return False

    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """The value of a variable, carried back as JSON.

        A Daytona sandbox is a machine of its own: what comes back is what can
        be encoded, and anything that cannot arrives as its repr rather than
        raising — a partial answer being more use than none for the reading
        this serves, which is `commands.run` and the filesystem.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        execution = self.run_code(
            _emit_text(f"{_JSON_MOD}.dumps({name}, default=repr)"), context=context
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(execution.execution_error or "Sandbox execution failed")
        if execution.code_error is not None or execution.text is None:
            raise VariableNotFoundError(name)
        return json.loads(execution.text)

    def _set_internal_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        try:
            payload = json.dumps(value)
        except TypeError as error:
            raise SandboxConfigurationError(
                f"A Daytona sandbox runs elsewhere, so {name!r} has to cross as "
                "JSON and this value cannot be encoded. Build it inside the "
                "sandbox with run_code instead."
            ) from error
        execution = self.run_code(
            f"import json as {_JSON_MOD}\n"
            f"{name} = {_JSON_MOD}.loads({payload!r})\n"
            f"del {_JSON_MOD}\n",
            context=context,
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(execution.execution_error or "Sandbox execution failed")
        if execution.code_error is not None:
            raise SandboxExecutionError(str(execution.code_error))

    def _write_file(self, path: str, content: bytes) -> None:
        """Straight to the filesystem of the sandbox, not through the code.

        The base class writes a file by running a snippet that base64-decodes
        it, which is the only way when all a variant has is an interpreter.
        Daytona has a filesystem API, so a large file does not have to become
        a large program.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        self._sandbox.fs.upload_file(content, path)

    def _read_file(self, path: str) -> bytes:
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        content = self._sandbox.fs.download_file(path)
        if content is None:
            # A file that is not there raises out of the SDK, so this is the
            # other case: an answer carrying neither content nor error.
            # Reading it as an empty file would make a failed read look like
            # a successful one.
            raise FileNotFoundError(f"Could not read file: {path}")
        return content
