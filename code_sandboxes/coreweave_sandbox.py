# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""CoreWeave sandbox implementation.

`CoreWeave Sandboxes <https://docs.coreweave.com/products/sandboxes>`_ run a
container on CoreWeave's own GPU cloud, started against a managed runner and
addressed through the ``cwsandbox`` SDK. What it offers is a container and
``exec`` — a process at a time, with its streams — and nothing that holds a
Python namespace between calls.

So one is held here. A single ``python -u -c`` process is started with the
sandbox and fed JSON lines on stdin, one request and one reply each, which is
the same arrangement the Modal variant uses and for the same reason: ``x = 1``
in one call and ``print(x)`` in the next behave the way they do in every other
variant of this package. A driver that cannot be started, or that goes away
mid-session, drops the sandbox back to a process per snippet — working, merely
stateless — rather than failing.

Rich display data — a figure, an HTML repr — has no channel in this
arrangement, and is not reported. The value of a trailing expression is.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from typing import Any

from .base import Sandbox
from .exceptions import (
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxNotStartedError,
    VariableNotFoundError,
)
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

#: What every sandbox this package creates is tagged with, so the ones it made
#: can be told from the rest of an organization's.
CREATED_BY_LABEL = "code-sandboxes"

#: The container a sandbox runs when the caller names no image. It is the
#: SDK's own default, and carries nothing but Python.
DEFAULT_CONTAINER_IMAGE = "python:3.11"

#: The session process. CoreWeave's `exec` is one process per call — whatever
#: a snippet defined is gone when its process exits, so a namespace has to be
#: kept by something that outlives them. This driver is started once and fed
#: JSON lines on stdin — one request, one reply — executing everything in a
#: single namespace, and answering with what the code printed, what its
#: trailing expression evaluated to, and the error it raised.
_DRIVER_SOURCE = """
import ast, contextlib, io, json, sys, traceback

namespace = {"__name__": "__main__"}
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    out, err = io.StringIO(), io.StringIO()
    reply = {"seq": request.get("seq"), "status": "ok"}
    try:
        tree = ast.parse(request.get("code", ""), mode="exec")
        trailing = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing = ast.Expression(tree.body.pop(-1).value)
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
    print(json.dumps(reply), flush=True)
"""

#: The program a snippet is run by when there is no driver — the fallback, and
#: what `commands.run` uses.
_STATELESS_SOURCE = """
import ast, contextlib, io, json, sys, traceback

request = json.loads(sys.stdin.read() or "{}")
out, err = io.StringIO(), io.StringIO()
reply = {"status": "ok"}
try:
    tree = ast.parse(request.get("code", ""), mode="exec")
    trailing = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        trailing = ast.Expression(tree.body.pop(-1).value)
    namespace = {"__name__": "__main__"}
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
print(json.dumps(reply), flush=True)
"""


def _import_cwsandbox() -> Any:
    try:
        import cwsandbox
    except ImportError as exc:
        raise SandboxConfigurationError(
            "cwsandbox is required for CoreWeaveSandbox. Install it with: "
            "pip install code-sandboxes[coreweave]"
        ) from exc
    return cwsandbox


class CoreWeaveSandbox(Sandbox):
    """Sandbox backed by a CoreWeave sandbox.

    Args:
        config: Optional sandbox configuration.
        api_key: CoreWeave API token. The SDK reads ``CWSANDBOX_API_KEY`` from
            the environment when this is omitted; passing it here sets that
            variable for the process, since the SDK offers no other way in.
        base_url: The control plane to talk to. ``CWSANDBOX_BASE_URL`` when
            omitted, which itself defaults to ``https://api.cwsandbox.com``.
        container_image: The image the sandbox runs.
            :data:`DEFAULT_CONTAINER_IMAGE` when omitted.
        profile_names: The sandbox profiles to run under — CoreWeave's own
            policy objects, which decide what a sandbox may do.
        runner_ids: Particular managed runners to place the sandbox on. Left
            to CoreWeave when omitted, which is the usual case.
        python_executable: The Python inside the container. ``python3`` unless
            the image keeps it elsewhere.
        stateful: Whether to keep one process for the session, so snippets
            share a namespace. On by default; turning it off runs each snippet
            in a process of its own.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        container_image: str | None = None,
        profile_names: list[str] | None = None,
        runner_ids: list[str] | None = None,
        python_executable: str = "python3",
        stateful: bool = True,
        **kwargs,
    ):
        super().__init__(config)
        self._api_key = api_key
        self._base_url = base_url
        self._container_image = container_image
        self._profile_names = profile_names
        self._runner_ids = runner_ids
        self._python_executable = python_executable
        self._stateful = stateful
        self._sandbox: Any | None = None
        self._driver: Any | None = None
        self._driver_replies: Any | None = None
        self._driver_seq = 0
        self._execution_count = 0
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """The environments this provider ships.

        CoreWeave takes an image and a machine specification per sandbox
        rather than a catalogue of named ones, so what is offered here are the
        shapes worth naming — a plain container and one with a GPU, which is
        what CoreWeave is for. The shape is asked for by argument: `gpu=`,
        `container_image=`.
        """
        return [
            SandboxEnvironment(
                name="coreweave-default",
                title="CoreWeave",
                language="python",
                owner="coreweave",
                visibility="cloud",
                burning_rate=0.0,
                metadata={
                    "variant": "coreweave",
                    "container_image": DEFAULT_CONTAINER_IMAGE,
                    "gpu": None,
                },
            ),
            SandboxEnvironment(
                name="coreweave-gpu",
                title="CoreWeave GPU",
                language="python",
                owner="coreweave",
                visibility="cloud",
                burning_rate=0.0,
                metadata={
                    "variant": "coreweave",
                    "container_image": DEFAULT_CONTAINER_IMAGE,
                    "gpu": "H100",
                },
            ),
        ]

    def start(self) -> None:
        if self._started:
            return

        cwsandbox = _import_cwsandbox()
        self._apply_credentials()
        self._sandbox = cwsandbox.Sandbox.run(**self._run_params(cwsandbox))
        # `run` sends the request and answers at once; the container is not
        # there to `exec` in until it is running. Waited for WITHOUT a deadline
        # of ours: `config.timeout` bounds how long a snippet may run, and
        # spending it on a cold start — pulling an image, finding a GPU
        # runner — would fail a sandbox that was merely slow to arrive. The
        # SDK's own request timeout is what bounds this.
        self._sandbox.wait()

        if self._stateful:
            self._start_driver()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox.sandbox_id,
            variant="coreweave",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "coreweave_sandbox_id": self._sandbox.sandbox_id,
                "container_image": self._container_image or DEFAULT_CONTAINER_IMAGE,
                "runner_id": getattr(self._sandbox, "runner_id", None),
                "stateful": self._driver is not None,
            },
            resources=ResourceConfig(
                cpu=self.config.cpu_limit,
                memory=self.config.memory_limit,
                gpu=self.config.gpu,
            ),
            config=self.config,
        )
        self._started = True

    def _apply_credentials(self) -> None:
        """Put an explicitly given token where the SDK looks for one.

        ``cwsandbox`` authenticates from ``CWSANDBOX_API_KEY`` — there is no
        argument for a token on `Sandbox.run` — so a caller who passes one
        here has it set for this process. A caller who passes none changes
        nothing, and the environment answers as it did.
        """
        import os

        if self._api_key:
            os.environ["CWSANDBOX_API_KEY"] = self._api_key
        if self._base_url:
            os.environ["CWSANDBOX_BASE_URL"] = self._base_url

    def _run_params(self, cwsandbox: Any) -> dict[str, Any]:
        """What to ask CoreWeave for, from the configuration of this sandbox."""
        params: dict[str, Any] = {
            "container_image": self._container_image or DEFAULT_CONTAINER_IMAGE,
            "tags": self._tag_list(),
        }
        if self.config.env_vars:
            params["environment_variables"] = dict(self.config.env_vars)
        if self.config.max_lifetime:
            params["max_lifetime_seconds"] = float(self.config.max_lifetime)
        if self._profile_names:
            params["profile_names"] = list(self._profile_names)
        if self._runner_ids:
            params["runner_ids"] = list(self._runner_ids)
        resources = self._resources(cwsandbox)
        if resources is not None:
            params["resources"] = resources
        network = self._network_params(cwsandbox)
        if network is not None:
            params["network"] = network
        return params

    def _tag_list(self) -> list[str]:
        """The metadata the sandbox carries in CoreWeave.

        CoreWeave keeps tags as a flat list of strings rather than as a map,
        so a pair is written ``key=value`` — which is how the name of the
        sandbox and the tags of the configuration survive the crossing and
        can be read back by `list`.
        """
        tags = [f"created-by={CREATED_BY_LABEL}"]
        if self.config.name:
            tags.append(f"name={self.config.name}")
        tags.extend(f"{key}={value}" for key, value in self._tags.items())
        return tags

    def _resources(self, cwsandbox: Any) -> Any | None:
        """The machine asked for, or nothing when the defaults will do.

        CoreWeave takes Kubernetes quantities — ``"2"`` cores, ``"4Gi"`` of
        memory — as requests and limits, and a GPU as a count and a kind.
        """
        requests: dict[str, str] = {}
        if self.config.cpu_limit:
            requests["cpu"] = str(max(1, math.ceil(self.config.cpu_limit)))
        if self.config.memory_limit:
            requests["memory"] = f"{max(1, math.ceil(self.config.memory_limit / 1024**3))}Gi"
        gpu: dict[str, Any] | None = None
        if self.config.gpu:
            # One name, or the first of several: CoreWeave places a sandbox on
            # a runner of one kind, and has no fallback list of its own.
            kind = self.config.gpu.split(",")[0].strip()
            if kind:
                gpu = {"count": 1, "type": kind}
        if not requests and gpu is None:
            return None
        return cwsandbox.ResourceOptions(
            requests=requests or None,
            limits=requests or None,
            gpu=gpu,
        )

    def _network_params(self, cwsandbox: Any) -> Any | None:
        """What the network policy of the configuration means to CoreWeave."""
        policy = self.config.network_policy
        if policy == "none":
            return cwsandbox.NetworkOptions(deny_egress=True, deny_ingress=True)
        if policy == "allowlist":
            if not self.config.allowed_hosts:
                raise SandboxConfigurationError(
                    "network_policy='allowlist' needs allowed_hosts: a sandbox "
                    "allowed nothing is a sandbox with no network at all, which "
                    "is network_policy='none'."
                )
            # An egress list IS the allowlist: naming what may be reached
            # denies everything else.
            return cwsandbox.NetworkOptions(
                egress=[cwsandbox.EgressRule(dns_name=host) for host in self.config.allowed_hosts]
            )
        return None

    def _start_driver(self) -> None:
        """Start the session process, and fall back to nothing on failure.

        A driver that cannot come up leaves `self._driver` unset, and
        `run_code` then executes each snippet in its own process — working,
        merely stateless.
        """
        import queue
        import threading

        try:
            driver = self._sandbox.exec(
                [self._python_executable, "-u", "-c", _DRIVER_SOURCE],
                stdin=True,
            )
        except Exception:
            logger.warning(
                "The CoreWeave session driver could not be started; snippets will not share state.",
                exc_info=True,
            )
            return
        replies: queue.Queue = queue.Queue()

        def pump() -> None:
            # The reader dies with the driver, and says so with the sentinel
            # below rather than with an exception nobody is there to catch.
            with contextlib.suppress(Exception):
                for line in driver.stdout:
                    replies.put(line)
            replies.put(None)

        # A thread reads the replies: the stream blocks, and a request that
        # never gets its answer must time out rather than hang run_code.
        thread = threading.Thread(target=pump, name="coreweave-driver-stdout", daemon=True)
        thread.start()
        self._driver = driver
        self._driver_replies = replies
        self._driver_seq = 0

    def _driver_request(self, code: str, timeout: float) -> dict | None:
        """One request to the session process, or None when it cannot serve."""
        import queue

        if self._driver is None:
            return None
        self._driver_seq += 1
        try:
            self._driver.stdin.writeline(json.dumps({"seq": self._driver_seq, "code": code}))
        except Exception:
            logger.warning("The CoreWeave session driver went away; restarting stateless.")
            self._driver = None
            return None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No reply from the CoreWeave session within {timeout:.0f}s.")
            try:
                line = self._driver_replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                # The reader reached EOF: the driver is gone.
                self._driver = None
                return None
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if reply.get("seq") == self._driver_seq:
                return reply

    def _stateless_request(self, code: str, timeout: float) -> dict:
        """One snippet in a process of its own, for when there is no driver."""
        process = self._sandbox.exec(
            [self._python_executable, "-u", "-c", _STATELESS_SOURCE],
            timeout_seconds=timeout,
            stdin=True,
        )
        process.stdin.writeline(json.dumps({"code": code}))
        process.stdin.close()
        result = process.result(timeout=timeout)
        printed = _text(getattr(result, "stdout_bytes", b""))
        for line in reversed(printed.splitlines()):
            # The reply is the LAST line: anything the program wrote outside
            # the redirect — a warning from the interpreter itself — comes
            # before it.
            with contextlib.suppress(ValueError):
                reply = json.loads(line)
                if isinstance(reply, dict):
                    return reply
        return {
            "status": "error",
            "stdout": printed,
            "stderr": _text(getattr(result, "stderr_bytes", b"")),
            "error": {
                "name": "SandboxError",
                "value": "The sandbox answered with nothing this package could read.",
                "traceback": "",
            },
        }

    def stop(self) -> None:
        if not self._started:
            return
        if self._driver is not None:
            with contextlib.suppress(Exception):
                self._driver.stdin.close()
            self._driver = None
            self._driver_replies = None
        if self._sandbox is not None:
            try:
                self._sandbox.stop().result()
            except Exception:
                logger.debug("Ignoring error while stopping the CoreWeave sandbox", exc_info=True)
            self._sandbox = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

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
            raise ValueError(f"CoreWeaveSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1
        seconds = timeout if timeout is not None else self.config.timeout
        prepared = _with_envs(code, envs)

        try:
            reply = self._driver_request(prepared, seconds)
            if reply is None:
                reply = self._stateless_request(prepared, seconds)
        except Exception as error:
            # The container, not the code: a sandbox CoreWeave has taken down,
            # a runner that went away, a session that never answered.
            return ExecutionResult(
                execution_ok=False,
                execution_error=f"Failed to execute code on CoreWeave: {error}",
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        return self._execution_result(
            reply, context, started_at, on_stdout, on_stderr, on_result, on_error
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
        """One reply of the driver, as an `ExecutionResult`.

        The output arrives whole rather than as it was written — the driver
        collects it and answers once — so the callbacks are called here, in
        order, on the lines it carried. A caller that streams sees the lines
        it would have seen, later.
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
        """The session process takes no interrupt; a timeout is the only stop."""
        return False

    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """The value of a variable, carried back as JSON."""
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
                f"A CoreWeave sandbox runs elsewhere, so {name!r} has to cross "
                "as JSON and this value cannot be encoded. Build it inside the "
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
        """Straight to the filesystem of the container, not through the code."""
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        self._sandbox.write_file(path, content).result()

    def _read_file(self, path: str) -> bytes:
        if not self._started or self._sandbox is None:
            raise SandboxNotStartedError()
        content = self._sandbox.read_file(path).result()
        if content is None:
            raise FileNotFoundError(f"Could not read file: {path}")
        return bytes(content)


def _text(raw: Any) -> str:
    """Whatever the SDK handed back, as text."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _lines(raw: Any) -> list[str]:
    """The lines of one stream, without the empty one a trailing newline makes."""
    text = _text(raw)
    if not text:
        return []
    return text.splitlines()


def _with_envs(code: str, envs: dict[str, str] | None) -> str:
    """The snippet, with the environment it asked for set first.

    The session process is started once, so variables meant for one execution
    cannot be passed to it the way they would be to a fresh process. They are
    set inside the namespace instead, which is where the code reads them from.
    """
    if not envs:
        return code
    assignments = "".join(
        f"_code_sandboxes_os.environ[{key!r}] = {value!r}\n" for key, value in envs.items()
    )
    return f"import os as _code_sandboxes_os\n{assignments}del _code_sandboxes_os\n{code}"
