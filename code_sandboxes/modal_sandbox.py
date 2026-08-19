# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Modal sandbox implementation.

`Modal <https://modal.com/docs/guide/sandbox>`_ provides secure, cloud-hosted
containers that can run arbitrary code. This sandbox uses ``modal.Sandbox`` to
provision a container and executes Python snippets inside it via ``sandbox.exec``.

Each ``run_code`` call runs the snippet as a fresh ``python -c`` process, so
Python variables do **not** persist across calls (use the filesystem or a single
snippet for stateful workflows). Rich display outputs (images, HTML) are not
captured; only stdout/stderr text and the process exit code are returned.
"""

from __future__ import annotations

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


def _modal_exec_timeout_seconds(timeout: float | None, default: float) -> int:
    """Return a Modal-compatible timeout in integer seconds."""
    value = timeout if timeout is not None else default
    return max(1, int(math.ceil(value)))


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
            "ModalSandbox executes each snippet in a fresh process and does not "
            "support cross-call variable access."
        )

    def _set_internal_variable(self, name: str, value, context: Context | None = None) -> None:
        raise NotImplementedError(
            "ModalSandbox executes each snippet in a fresh process and does not "
            "support cross-call variable access."
        )
