# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Modal sandbox implementation.

`Modal <https://modal.com/docs/guide/sandbox>`_ provides secure, cloud-hosted
containers that can run arbitrary code. This sandbox uses ``modal.Sandbox`` to
provision a container and executes Python snippets inside it via ``sandbox.exec``.

One ``python -u -c`` process is started with the sandbox and fed JSON lines on
stdin — one request, one reply — so snippets share a namespace: ``x = 1`` in one
call is still there in the next. A session that cannot be started, or that goes
away mid-run, drops back to a fresh ``python -c`` process per snippet, which
works and merely forgets. Rich display outputs (images, HTML) are not captured;
stdout, stderr and the value of a trailing expression are.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
import uuid
from typing import Any

from .base import Sandbox
from .exceptions import SandboxConfigurationError, SandboxNotStartedError
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    Logs,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "code-sandboxes"
DEFAULT_MODAL_PYTHON_VERSION = "3.12"


def _resolve_modal_gpu(gpu_flavor: str, modal_module: Any) -> Any:
    """Resolve a GPU flavor string to a Modal GPU spec when possible.

    Falls back to the raw string when no structured constructor is available.
    """
    flavor = gpu_flavor.strip()
    gpu_ns = getattr(modal_module, "gpu", None)
    if gpu_ns is None:
        return flavor

    normalized = flavor.upper().replace("_", "-")

    if normalized == "A100-80GB" and hasattr(gpu_ns, "A100"):
        try:
            return gpu_ns.A100(size="80GB")
        except Exception:
            return flavor

    attr_by_flavor = {
        "T4": "T4",
        "L4": "L4",
        "A10G": "A10G",
        "A100": "A100",
        "H100": "H100",
    }
    attr_name = attr_by_flavor.get(normalized)
    if not attr_name or not hasattr(gpu_ns, attr_name):
        return flavor

    candidate = getattr(gpu_ns, attr_name)
    try:
        return candidate()
    except TypeError:
        return candidate


#: The process that holds the session inside the container.
#:
#: `sandbox.exec("python", "-c", code)` is a fresh interpreter per snippet, so
#: `x = 1` in one call and `x` in the next was a NameError: the container
#: persists, the namespace did not. This driver is started once and fed JSON
#: lines on stdin — one request, one reply — executing everything in a single
#: namespace, with stdout/stderr captured per request and the value of a
#: trailing expression repr'd the way a REPL would.
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


def _modal_exec_timeout_seconds(timeout: float | None, default: float) -> int:
    """Return a Modal-compatible timeout in integer seconds."""
    value = timeout if timeout is not None else default
    return max(1, math.ceil(value))


class ModalSandbox(Sandbox):
    """Sandbox backed by a Modal cloud container.

    Args:
        config: Optional sandbox configuration.
        app_name: Name of the Modal App to attach the sandbox to (created if missing).
        image: An optional pre-built ``modal.Image``. When omitted, a
            ``debian_slim`` image is used, optionally extended with ``pip_packages``.
        pip_packages: Optional list of pip packages to install in the default image.
        python_executable: Executable used to run snippets (default ``python``).
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        app_name: str = DEFAULT_APP_NAME,
        image: Any | None = None,
        pip_packages: list[str] | None = None,
        python_version: str = DEFAULT_MODAL_PYTHON_VERSION,
        python_executable: str = "python",
        **kwargs,
    ):
        super().__init__(config)
        self._app_name = app_name
        self._image = image
        self._pip_packages = pip_packages or []
        self._python_version = python_version
        self._python_executable = python_executable
        self._app = None
        self._sandbox = None
        self._sandbox_id = str(uuid.uuid4())
        self._execution_count = 0
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """The environments this provider ships.

        Modal takes a machine specification per sandbox; what is offered here
        are the two shapes worth naming — a plain container, and one with a
        GPU attached — so that choosing an environment is choosing between
        two named things, as it is with every other provider.
        """
        return [
            SandboxEnvironment(
                name="modal-cpu",
                title="Modal CPU",
                language="python",
                owner="modal",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "modal", "gpu": None},
            ),
            SandboxEnvironment(
                name="modal-gpu",
                title="Modal GPU",
                language="python",
                owner="modal",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "modal", "gpu": "T4"},
            ),
        ]

    def start(self) -> None:
        if self._started:
            return

        try:
            import modal
        except ImportError as exc:
            raise SandboxConfigurationError(
                "modal is required for ModalSandbox. Install it with: pip install modal"
            ) from exc

        self._app = modal.App.lookup(self._app_name, create_if_missing=True)

        image = self._image
        if image is None:
            image = modal.Image.debian_slim(python_version=self._python_version)
            if self._pip_packages:
                image = image.pip_install(*self._pip_packages)

        secrets = []
        if self.config.env_vars:
            secrets.append(modal.Secret.from_dict(dict(self.config.env_vars)))

        create_kwargs: dict[str, Any] = {
            "app": self._app,
            "image": image,
            "timeout": int(self.config.max_lifetime),
        }
        if self.config.gpu:
            create_kwargs["gpu"] = _resolve_modal_gpu(self.config.gpu, modal)
        if secrets:
            create_kwargs["secrets"] = secrets

        self._sandbox = modal.Sandbox.create(**create_kwargs)
        self._start_driver()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="modal",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "app_name": self._app_name,
                "modal_sandbox_id": getattr(self._sandbox, "object_id", None),
            },
            config=self.config,
        )
        self._started = True

    def _start_driver(self) -> None:
        """Start the session process, and fall back to nothing on failure.

        A driver that cannot come up leaves `self._driver` unset, and
        `run_code` then executes each snippet in its own process as before —
        working, merely stateless.
        """
        import queue
        import threading

        try:
            driver = self._sandbox.exec(self._python_executable, "-u", "-c", _DRIVER_SOURCE)
        except Exception:
            logger.warning(
                "The Modal session driver could not be started; snippets will not share state.",
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
        thread = threading.Thread(target=pump, name="modal-driver-stdout", daemon=True)
        thread.start()
        self._driver = driver
        self._driver_replies = replies
        self._driver_seq = 0

    def _driver_request(self, code: str, timeout: float) -> dict | None:
        """One request to the session process, or None when it cannot serve."""
        import json as json_module
        import queue

        if getattr(self, "_driver", None) is None:
            return None
        self._driver_seq += 1
        try:
            self._driver.stdin.write(
                json_module.dumps({"seq": self._driver_seq, "code": code}) + "\n"
            )
            self._driver.stdin.drain()
        except Exception:
            logger.warning("The Modal session driver went away; restarting stateless.")
            self._driver = None
            return None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No reply from the Modal session within {timeout:.0f}s.")
            try:
                line = self._driver_replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                # The reader reached EOF: the driver is gone.
                self._driver = None
                return None
            try:
                reply = json_module.loads(line)
            except ValueError:
                continue
            if reply.get("seq") == self._driver_seq:
                return reply

    def stop(self) -> None:
        if not self._started:
            return
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            except Exception:
                logger.debug("Ignoring error while terminating Modal sandbox", exc_info=True)
            try:
                self._sandbox.detach()
            except Exception:
                logger.debug("Ignoring error while detaching Modal sandbox", exc_info=True)
            self._sandbox = None
        self._app = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

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
            raise ValueError(f"ModalSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1

        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            code = f"{env_code}\n{code}"

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        code_error: CodeError | None = None

        # One process for the session: state persists between snippets, and a
        # trailing expression answers with its repr, as a REPL would. The
        # fresh-process path below stays as the fallback when the driver is
        # not there.
        reply = None
        try:
            reply = self._driver_request(code, timeout or self.config.timeout)
        except TimeoutError as error:
            return ExecutionResult(
                execution_ok=False,
                execution_error=str(error),
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )
        if reply is not None:
            current_time = time.time()
            results: list[Result] = []
            for line in (reply.get("stdout") or "").splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=False)
                stdout_messages.append(msg)
                if on_stdout:
                    on_stdout(msg)
            for line in (reply.get("stderr") or "").splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=True)
                stderr_messages.append(msg)
                if on_stderr:
                    on_stderr(msg)
            if reply.get("result") is not None:
                value = Result(data={"text/plain": reply["result"]}, is_main_result=True)
                results.append(value)
                if on_result:
                    on_result(value)
            if reply.get("status") == "error":
                detail = reply.get("error") or {}
                code_error = CodeError(
                    name=detail.get("name", "Error"),
                    value=detail.get("value", ""),
                    traceback=detail.get("traceback", ""),
                )
                if on_error:
                    on_error(code_error)
            return ExecutionResult(
                results=results,
                logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
                execution_ok=True,
                code_error=code_error,
                started_at=started_at,
                completed_at=current_time,
                execution_count=self._execution_count,
                context_id=context.id if context else "default",
            )

        try:
            process = self._sandbox.exec(
                self._python_executable,
                "-c",
                code,
                timeout=_modal_exec_timeout_seconds(timeout, self.config.timeout),
            )
            stdout_text = process.stdout.read()
            stderr_text = process.stderr.read()
            process.wait()
            returncode = process.returncode
        except Exception as e:
            return ExecutionResult(
                execution_ok=False,
                execution_error=f"Failed to execute code on Modal: {e}",
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
            )

        current_time = time.time()
        for line in (stdout_text or "").splitlines():
            msg = OutputMessage(line=line, timestamp=current_time, error=False)
            stdout_messages.append(msg)
            if on_stdout:
                on_stdout(msg)
        for line in (stderr_text or "").splitlines():
            msg = OutputMessage(line=line, timestamp=current_time, error=True)
            stderr_messages.append(msg)
            if on_stderr:
                on_stderr(msg)

        exit_code: int | None = None
        # A non-zero return code with stderr output indicates the user code
        # raised an exception. Surface it as a code error.
        if returncode not in (0, None) and stderr_text:
            last_line = stderr_text.strip().splitlines()[-1] if stderr_text.strip() else ""
            name = last_line.split(":", 1)[0].strip() or "Error"
            value = last_line.split(":", 1)[1].strip() if ":" in last_line else last_line
            code_error = CodeError(name=name, value=value, traceback=stderr_text)
            if on_error:
                on_error(code_error)
        elif returncode not in (0, None):
            exit_code = int(returncode)

        return ExecutionResult(
            results=[],
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            exit_code=exit_code,
            execution_count=self._execution_count,
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
        )

    def _do_interrupt(self) -> bool:
        """Modal does not support interrupts."""
        return False

    def _get_internal_variable(self, name: str, context: Context | None = None):
        raise NotImplementedError(
            "ModalSandbox holds variables in its session process; read them by "
            "running code, e.g. run_code(f'print({name})')."
        )

    def _set_internal_variable(self, name: str, value, context: Context | None = None) -> None:
        raise NotImplementedError(
            "ModalSandbox holds variables in its session process; set them by "
            "running code, e.g. run_code(f'{name} = ...')."
        )
