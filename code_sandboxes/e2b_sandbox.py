# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""E2B sandbox implementation.

`E2B <https://e2b.dev>`_ runs code in Firecracker microVMs that start in about
150 ms. This variant drives one through the CODE INTERPRETER SDK —
``e2b-code-interpreter`` — rather than through the plain ``e2b`` SDK: the
interpreter keeps a Jupyter kernel per context, so ``x = 1`` in one call and
``print(x)`` in the next behave the way they do in every other variant of this
package, and rich display data — a figure, an HTML repr — arrives as results
rather than being lost.

The mapping is unusually direct. E2B's ``run_code`` takes the same arguments
as :meth:`Sandbox.run_code` down to the names of the callbacks, and answers
with an execution carrying logs, results and an error. What is done here is
therefore mostly translation: E2B names each rich format with an attribute
(``png``, ``html``, …) where this package keys them by MIME type, and E2B
stamps its output messages with a millisecond integer where this package
counts in seconds.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .base import Sandbox
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
    MIMEType,
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

#: What every sandbox this package creates is tagged with, so the ones it made
#: can be told from the rest of an account's.
CREATED_BY_LABEL = "code-sandboxes"

#: The E2B template a sandbox is created from when the caller names none.
#:
#: The code interpreter's own, NOT E2B's ``base``: the interpreter talks to a
#: Jupyter kernel inside the sandbox, and only a template carrying one can
#: answer it. ``base`` is the default of the plain ``e2b`` SDK, where there is
#: no kernel to talk to, and a sandbox created from it here would start and
#: then fail every execution.
DEFAULT_TEMPLATE = "code-interpreter-v1"

#: What E2B calls each rich format, and the MIME type this package keys it by.
#: E2B answers with named attributes rather than a MIME dictionary, so the
#: translation has to be written down somewhere; here, once.
_FORMAT_MIME_TYPES: dict[str, str] = {
    "text": MIMEType.TEXT_PLAIN.value,
    "html": MIMEType.TEXT_HTML.value,
    "markdown": MIMEType.TEXT_MARKDOWN.value,
    "svg": MIMEType.IMAGE_SVG.value,
    "png": MIMEType.IMAGE_PNG.value,
    "jpeg": MIMEType.IMAGE_JPEG.value,
    "gif": MIMEType.IMAGE_GIF.value,
    "pdf": MIMEType.APPLICATION_PDF.value,
    "json": MIMEType.APPLICATION_JSON.value,
    "latex": "text/latex",
    "javascript": "application/javascript",
}


def _import_e2b() -> Any:
    try:
        import e2b_code_interpreter
    except ImportError as exc:
        raise SandboxConfigurationError(
            "e2b-code-interpreter is required for E2BSandbox. Install it with: "
            "pip install code-sandboxes[e2b]"
        ) from exc
    return e2b_code_interpreter


def _timestamp(value: Any) -> float:
    """One of E2B's timestamps, in the seconds this package counts in.

    E2B stamps an output message with an integer of MILLISECONDS since the
    epoch. Passing it through unchanged would put every line of output fifty
    thousand years in the future, which is what a reader comparing it against
    `started_at` would see.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return time.time()
    if number <= 0:
        return time.time()
    # Anything this large is not a count of seconds: 1e11 seconds is the year
    # 5138, while 1e11 milliseconds is 1973.
    return number / 1000.0 if number > 1e11 else number


def _result_data(result: Any) -> dict[str, Any]:
    """The formats one E2B result carries, keyed by MIME type.

    ``formats()`` answers with the names of the formats that are actually
    present, including the ones E2B does not have an attribute for — a custom
    MIME type a library declared — which arrive in ``extra``. Both are taken,
    so a result this package hands on is as complete as the one it was given.
    """
    data: dict[str, Any] = {}
    for name in result.formats():
        value = getattr(result, name, None)
        if value is None:
            extra = getattr(result, "extra", None) or {}
            value = extra.get(name)
        if value is None:
            continue
        data[_FORMAT_MIME_TYPES.get(name, name)] = value
    return data


class E2BSandbox(Sandbox):
    """Sandbox backed by an E2B microVM.

    Args:
        config: Optional sandbox configuration.
        api_key: E2B API key. Read from ``E2B_API_KEY`` when omitted.
        domain: E2B domain to talk to, for a self-hosted cluster. Read from
            ``E2B_DOMAIN`` when omitted, which itself defaults to ``e2b.dev``.
        template: The E2B template to create from — a base image with its
            packages already installed. :data:`DEFAULT_TEMPLATE` when omitted.
        allow_internet_access: Whether the sandbox may reach the network at
            all. ``network_policy='none'`` of the configuration says the same
            thing and wins when it is set.
        secure: E2B's own hardening of the sandbox's control plane. On by
            default, as it is in the SDK.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        api_key: str | None = None,
        domain: str | None = None,
        template: str | None = None,
        allow_internet_access: bool = True,
        secure: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        self._api_key = api_key
        self._domain = domain
        self._template = template
        self._allow_internet_access = allow_internet_access
        self._secure = secure
        self._sandbox: Any | None = None
        #: The E2B context standing for each of ours, made on first use —
        #: creating one starts a kernel, and most callers use the default
        #: namespace and never need a second.
        self._contexts: dict[str, Any] = {}
        self._execution_count = 0
        self._jupyter_endpoint: JupyterServerEndpoint | None = None
        self._extra_kwargs = kwargs

    def prepare_jupyter_server(
        self, options: JupyterServerOptions | None = None
    ) -> JupyterServerEndpoint:
        """Prepare Jupyter and expose it through E2B's per-port ingress."""
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        if self._jupyter_endpoint is not None:
            return self._jupyter_endpoint

        value = resolved_options(options)
        response = self._sandbox.commands.run(
            preparation_command(value), timeout=value.install_timeout
        )
        if getattr(response, "exit_code", 0) not in (0, None):
            raise SandboxConfigurationError(
                "Could not install and start Jupyter Server in the E2B sandbox: "
                + str(getattr(response, "stderr", "unknown error"))
            )
        url = f"https://{self._sandbox.get_host(value.port)}"
        traffic_token = getattr(self._sandbox, "traffic_access_token", None)
        headers = {"E2B-Traffic-Access-Token": traffic_token} if traffic_token else {}
        self._jupyter_endpoint = JupyterServerEndpoint(
            port=value.port,
            http_url=url,
            websocket_url=websocket_url(url),
            headers=headers,
            query={"token": value.token or ""},
        )
        return self._jupyter_endpoint

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """The environments this provider ships.

        E2B takes a TEMPLATE per sandbox rather than a catalogue of machines,
        and only a template carrying a Jupyter kernel can serve the code
        interpreter this variant drives — so there is ONE environment here,
        the interpreter's own template, rather than a menu including images
        that would start and then fail every execution. A template built on
        top of it is asked for by argument — `template=` — the way an image is
        everywhere else in this package.
        """
        return [
            SandboxEnvironment(
                name="e2b-code-interpreter",
                title="E2B",
                language="python",
                owner="e2b",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "e2b", "template": DEFAULT_TEMPLATE},
            ),
        ]

    def _refuse_what_cannot_be_honoured(self) -> None:
        """What this variant can never do, said before anything is installed.

        Judged from the configuration alone, so it comes BEFORE the import of
        the SDK: a caller who asked for a GPU E2B has not got should hear that
        first, rather than be sent to install a package and only then be told
        the request was impossible all along.
        """
        if self.config.gpu:
            # Silently running on a CPU is the worse failure: a sandbox that
            # looks as though it asked for an H100 and did not is one whose
            # timings mean nothing.
            raise SandboxConfigurationError(
                "E2B sandboxes have no GPU, so gpu=" + repr(self.config.gpu) + " "
                "cannot be honoured. Use the daytona, coreweave or modal "
                "variant for a GPU."
            )
        # The network policy is judged here too, for the same reason.
        self._network_allowed()

    def start(self) -> None:
        if self._started:
            return

        self._refuse_what_cannot_be_honoured()
        e2b = _import_e2b()
        self._sandbox = e2b.Sandbox.create(**self._create_params())

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox.sandbox_id,
            variant="e2b",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "e2b_sandbox_id": self._sandbox.sandbox_id,
                "template": self._template or DEFAULT_TEMPLATE,
                "domain": self._domain,
            },
            resources=ResourceConfig(
                cpu=self.config.cpu_limit,
                memory=self.config.memory_limit,
            ),
            config=self.config,
        )
        self._started = True

    def _create_params(self) -> dict[str, Any]:
        """What to ask E2B for, from the configuration of this sandbox.

        A setting that was not given is left out entirely rather than passed
        as ``None``: the SDK reads ``E2B_API_KEY`` and ``E2B_DOMAIN`` from the
        environment for exactly the arguments that are absent, and handing it
        an explicit ``None`` is not the same as handing it nothing.
        """
        params: dict[str, Any] = {
            "template": self._template or DEFAULT_TEMPLATE,
            "metadata": self._metadata(),
            "secure": self._secure,
            "allow_internet_access": self._network_allowed(),
        }
        if self._api_key:
            params["api_key"] = self._api_key
        if self._domain:
            params["domain"] = self._domain
        if self.config.env_vars:
            params["envs"] = dict(self.config.env_vars)
        if self.config.max_lifetime:
            # E2B counts the life of a sandbox in whole seconds, and takes it
            # down when they are up.
            params["timeout"] = max(1, round(self.config.max_lifetime))
        return params

    def _metadata(self) -> dict[str, str]:
        """The metadata the sandbox carries in E2B.

        E2B keeps this as a flat map of strings and lets a sandbox be looked
        up by it, so the name and the tags of the configuration go here — it
        is what makes `list` able to say which sandboxes this package made.
        """
        metadata = {"created-by": CREATED_BY_LABEL}
        if self.config.name:
            metadata["name"] = self.config.name
        metadata.update({key: str(value) for key, value in self._tags.items()})
        return metadata

    def _network_allowed(self) -> bool:
        """Whether this sandbox may reach the network.

        E2B offers one switch rather than an allowlist, so a policy naming
        hosts cannot be honoured and is refused instead of being silently
        widened to the whole internet — which is the failure that matters.
        """
        policy = self.config.network_policy
        if policy == "none":
            return False
        if policy == "allowlist":
            raise SandboxConfigurationError(
                "E2B has no host allowlist: a sandbox either reaches the "
                "network or it does not. Use network_policy='none' to cut it "
                "off, or 'all' to allow it."
            )
        return self._allow_internet_access

    def stop(self) -> None:
        if not self._started:
            return
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                logger.debug("Ignoring error while killing the E2B sandbox", exc_info=True)
            self._sandbox = None
        self._contexts.clear()
        self._jupyter_endpoint = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def set_timeout(self, seconds: float) -> None:
        """Give the sandbox longer to live, counted from now.

        E2B takes a sandbox down when its timeout runs out, whatever it is
        doing. A long-running job therefore has to say so before it starts,
        and this is how — the count restarts at the moment of the call.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        self._sandbox.set_timeout(max(1, round(seconds)))

    def get_host(self, port: int) -> str:
        """The host a service listening on `port` inside is reachable at.

        E2B gives every sandbox a public hostname per port, which is what
        makes a server started inside — a dashboard, an API under test —
        reachable from outside without a tunnel of one's own.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        return self._sandbox.get_host(port)

    def _e2b_context(self, context: Context | None) -> Any | None:
        """The E2B context one of ours stands for, made on first use.

        ``None`` — and the default context, which is the same namespace — is
        the sandbox's own kernel. Anything else is a context E2B keeps apart,
        so :meth:`create_context` really does isolate.
        """
        if context is None or context.id == "default":
            return None
        existing = self._contexts.get(context.id)
        if existing is None:
            existing = self._sandbox.create_code_context(cwd=context.cwd, language="python")
            self._contexts[context.id] = existing
        return existing

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
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        if language != "python":
            raise ValueError(f"E2BSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []

        def take_stdout(message: Any) -> None:
            translated = OutputMessage(
                line=message.line,
                timestamp=_timestamp(getattr(message, "timestamp", None)),
                error=False,
            )
            stdout_messages.append(translated)
            if on_stdout:
                on_stdout(translated)

        def take_stderr(message: Any) -> None:
            translated = OutputMessage(
                line=message.line,
                timestamp=_timestamp(getattr(message, "timestamp", None)),
                error=True,
            )
            stderr_messages.append(translated)
            if on_stderr:
                on_stderr(translated)

        def take_result(result: Any) -> None:
            translated = Result(
                data=_result_data(result),
                is_main_result=bool(getattr(result, "is_main_result", False)),
                extra=dict(getattr(result, "extra", None) or {}),
            )
            results.append(translated)
            if on_result:
                on_result(translated)

        seconds = timeout if timeout is not None else self.config.timeout
        try:
            execution = self._sandbox.run_code(
                code,
                language="python",
                context=self._e2b_context(context),
                on_stdout=take_stdout,
                on_stderr=take_stderr,
                on_result=take_result,
                envs=envs,
                timeout=seconds,
            )
        except Exception as error:
            # The microVM, not the code: a dropped connection, a sandbox that
            # E2B has already taken down. Reported rather than raised, so a
            # caller reads it the same way as every other infrastructure
            # failure in this package.
            return ExecutionResult(
                execution_ok=False,
                execution_error=f"Failed to execute code on E2B: {error}",
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        # The callbacks above collect everything as it streams, but E2B also
        # answers with the whole execution — and does NOT call `on_result` for
        # a sandbox that replayed a cached execution. Anything that only
        # appears in the answer is taken from there, without duplicating what
        # already arrived.
        for result in execution.results[len(results) :]:
            take_result(result)

        code_error: CodeError | None = None
        if execution.error is not None:
            code_error = CodeError(
                name=execution.error.name or "Error",
                value=execution.error.value or "",
                traceback=execution.error.traceback or "",
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
        """E2B's interpreter takes no interrupt; a timeout is the only stop."""
        return False

    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """The value of a variable, carried back as JSON.

        An E2B sandbox is a machine of its own: what comes back is what can be
        encoded, and anything that cannot arrives as its repr rather than
        raising — a partial answer being more use than none for the reading
        this serves, which is `commands.run` and the filesystem.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        execution = self.run_code(
            "import json as _code_sandboxes_json\n"
            f"print(_code_sandboxes_json.dumps({name}, default=repr))\n"
            "del _code_sandboxes_json\n",
            context=context,
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(
                "SandboxError", execution.execution_error or "Sandbox execution failed"
            )
        if execution.code_error is not None:
            raise VariableNotFoundError(name)
        printed = "\n".join(message.line for message in execution.logs.stdout).strip()
        if not printed:
            raise VariableNotFoundError(name)
        return json.loads(printed)

    def _set_internal_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        try:
            payload = json.dumps(value)
        except TypeError as error:
            raise SandboxConfigurationError(
                f"An E2B sandbox runs elsewhere, so {name!r} has to cross as "
                "JSON and this value cannot be encoded. Build it inside the "
                "sandbox with run_code instead."
            ) from error
        execution = self.run_code(
            "import json as _code_sandboxes_json\n"
            f"{name} = _code_sandboxes_json.loads({payload!r})\n"
            "del _code_sandboxes_json\n",
            context=context,
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(
                "SandboxError", execution.execution_error or "Sandbox execution failed"
            )
        if execution.code_error is not None:
            raise SandboxExecutionError(
                execution.code_error.name,
                execution.code_error.value,
                execution.code_error.traceback,
            )

    def _write_file(self, path: str, content: bytes) -> None:
        """Straight to the filesystem of the sandbox, not through the code.

        The base class writes a file by running a snippet that base64-decodes
        it, which is the only way when all a variant has is an interpreter.
        E2B has a filesystem API, so a large file does not have to become a
        large program.
        """
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        self._sandbox.files.write(path, content)

    def _read_file(self, path: str) -> bytes:
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        content = self._sandbox.files.read(path, format="bytes")
        if content is None:
            # A file that is not there raises out of the SDK, so this is the
            # other case: an answer carrying neither content nor error.
            # Reading it as an empty file would make a failed read look like a
            # successful one.
            raise FileNotFoundError(f"Could not read file: {path}")
        return bytes(content)
