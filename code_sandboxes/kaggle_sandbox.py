# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Kaggle sandbox implementation.

This sandbox connects to a Kaggle interactive notebook runtime and executes code
in its kernel using ``jupyter-kernel-client``'s :class:`KaggleKernelClient`.

When runtime connection details are not provided, it transparently falls back to
Kaggle's batch execution API via ``KaggleKernelExecutor``. This mode is useful
for server-side integrations (for example jupyter-mcp-server) because callers can
submit code without first attaching to an interactive kernel session.

Authentication supports two modes:

* **API token (default).** Provide a Kaggle API token via ``token`` or the
  ``KAGGLE_API_TOKEN`` environment variable. When ``kernel_id`` is omitted, a new
  kernel is created on the runtime.
* **Signed proxy URL.** Connect to an already-running notebook session using its
  ``server_url`` and ``kernel_id`` (typically derived from the WebSocket
  *channels* URL). The signed JWT embedded in the proxied ``server_url`` provides
  the authentication.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
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
_KAGGLE_TERMINAL_STATUSES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


class KaggleSandbox(Sandbox):
    """Sandbox backed by a Kaggle interactive notebook runtime.

    Args:
        config: Optional sandbox configuration.
        server_url: The Kaggle runtime proxy URL (ending in ``/proxy``).
        kernel_id: The Kaggle kernel identifier to connect to. When omitted, a new
            kernel is created on the runtime (requires a valid API token).
        channels_url: A Kaggle notebook session *channels* URL. When provided,
            ``server_url`` and ``kernel_id`` are parsed from it.
        token: The Kaggle API token used to authenticate interactive kernels. When
            ``None``, it falls back to the ``KAGGLE_API_TOKEN`` environment variable.
        gpu: Optional Kaggle accelerator name for batch mode. Supports friendly
            aliases such as ``T4`` or ``P100`` and Kaggle API values such as
            ``NvidiaTeslaT4``.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        server_url: str | None = None,
        kernel_id: str | None = None,
        channels_url: str | None = None,
        token: str | None = None,
        **kwargs,
    ):
        super().__init__(config)
        # Allow configuration via SandboxConfig extras as a fallback.
        extras = getattr(self.config, "model_extra", None) or {}
        self._server_url = server_url or extras.get("server_url")
        self._kernel_id = kernel_id or extras.get("kernel_id")
        self._channels_url = channels_url or extras.get("channels_url")
        self._token = token or extras.get("token")
        self._client = None
        self._executor = None
        self._batch_mode = False
        self._sandbox_id = str(uuid.uuid4())
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="kaggle",
                title="Kaggle",
                language="python",
                owner="kaggle",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "kaggle"},
            )
        ]

    def start(self) -> None:
        if self._started:
            return

        # Transparent batch mode: when no interactive runtime connection details
        # are available, fall back to Kaggle's official job API.
        if not self._server_url and not self._channels_url:
            try:
                from jupyter_kernel_client import KaggleKernelExecutor
            except ImportError as exc:
                raise SandboxConfigurationError(
                    "jupyter-kernel-client>=0.14.0 is required for Kaggle batch execution. "
                    "Install it with: pip install jupyter-kernel-client"
                ) from exc

            self._executor = KaggleKernelExecutor(
                username=self._extra_kwargs.get("username"),
                quiet=True,
            )
            self._batch_mode = True
            self._default_context = self.create_context("default")
            self._info = SandboxInfo(
                id=self._sandbox_id,
                variant="kaggle",
                status=SandboxStatus.RUNNING,
                created_at=time.time(),
                name=self.config.name,
                metadata={"mode": "batch"},
                config=self.config,
            )
            self._started = True
            return

        try:
            from jupyter_kernel_client import (
                KaggleKernelClient,
                parse_kaggle_channels_url,
            )
        except ImportError as exc:
            raise SandboxConfigurationError(
                "jupyter-kernel-client>=0.12.0 is required for KaggleSandbox. "
                "Install it with: pip install jupyter-kernel-client"
            ) from exc

        # Derive server_url / kernel_id from a channels URL when provided.
        if self._channels_url and (not self._server_url or not self._kernel_id):
            parsed_server_url, parsed_kernel_id = parse_kaggle_channels_url(self._channels_url)
            self._server_url = self._server_url or parsed_server_url
            self._kernel_id = self._kernel_id or parsed_kernel_id

        if not self._server_url:
            raise SandboxConfigurationError(
                "KaggleSandbox requires 'server_url' (and optionally 'kernel_id'), "
                "or a 'channels_url' to parse them from. Obtain them from the "
                "WebSocket channels URL of a running Kaggle notebook session."
            )

        self._client = KaggleKernelClient(
            server_url=self._server_url,
            kernel_id=self._kernel_id,
            token=self._token,
        )
        self._client.start()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="kaggle",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={"server_url": self._server_url, "kernel_id": self._kernel_id},
            config=self.config,
        )
        self._started = True

    def _setup_tool_caller(self) -> None:
        """Keep tool calling on the client side for Kaggle sandboxes."""
        return

    @staticmethod
    def _normalize_status(status: Any) -> str:
        """Normalize Kaggle status values (enum or string) to uppercase names."""
        name = getattr(status, "name", None)
        if name is None:
            name = str(status)
        return name.split(".")[-1].strip().upper()

    @staticmethod
    def _populate_artifacts_from_files(result: Any, files: list[str]) -> None:
        """Populate log/notebook fields from downloaded Kaggle output files."""
        if getattr(result, "log", None) is None:
            result.log = None
        if getattr(result, "notebook", None) is None:
            result.notebook = None

        for path_str in files:
            path = Path(path_str)
            if path.suffix == ".log" and result.log is None:
                try:
                    result.log = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
            elif path.suffix == ".ipynb" and result.notebook is None:
                try:
                    import json

                    result.notebook = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass

    def stop(self) -> None:
        if not self._started:
            return
        if self._client is not None:
            try:
                # Do not shut down the Kaggle kernel; we only disconnect.
                self._client.stop(shutdown_kernel=False)
            except Exception:
                logger.debug("Ignoring error while stopping Kaggle client", exc_info=True)
            self._client = None
        self._executor = None
        self._batch_mode = False
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
        if not self._started:
            raise SandboxNotStartedError()

        if self._batch_mode:
            return self._run_code_batch(
                code=code,
                language=language,
                context=context,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                on_result=on_result,
                on_error=on_error,
                envs=envs,
                timeout=timeout,
            )

        if self._client is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"KaggleSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._interrupt_requested.clear()
        self._executing_event.set()

        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            code = f"{env_code}\n{code}"

        try:
            reply = self._client.execute(code, timeout=timeout or self.config.timeout)
        except Exception as e:
            self._executing_event.clear()
            was_interrupted = self._interrupt_requested.is_set()
            self._interrupt_requested.clear()
            return ExecutionResult(
                execution_ok=False,
                execution_error=f"Failed to execute code: {e}" if not was_interrupted else None,
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
                interrupted=was_interrupted,
            )

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: CodeError | None = None
        exit_code: int | None = None

        current_time = time.time()
        for output in reply.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                name = output.get("name")
                text = output.get("text", "")
                for line in text.splitlines():
                    msg = OutputMessage(line=line, timestamp=current_time, error=name == "stderr")
                    if name == "stderr":
                        stderr_messages.append(msg)
                        if on_stderr:
                            on_stderr(msg)
                    else:
                        stdout_messages.append(msg)
                        if on_stdout:
                            on_stdout(msg)
            elif output_type in ("execute_result", "display_data"):
                result = Result(
                    data=output.get("data", {}),
                    is_main_result=output_type == "execute_result",
                    extra=output.get("metadata", {}),
                )
                results.append(result)
                if on_result:
                    on_result(result)
            elif output_type == "error":
                ename = output.get("ename", "Error")
                evalue = output.get("evalue", "")
                if ename == "SystemExit":
                    try:
                        exit_code = int(evalue) if evalue else 0
                    except (ValueError, TypeError):
                        exit_code = 1 if evalue else 0
                else:
                    code_error = CodeError(
                        name=ename,
                        value=evalue,
                        traceback="\n".join(output.get("traceback", [])),
                    )
                    if on_error:
                        on_error(code_error)

        self._executing_event.clear()
        was_interrupted = self._interrupt_requested.is_set()
        self._interrupt_requested.clear()

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            exit_code=exit_code,
            execution_count=reply.get("execution_count", 0),
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
            interrupted=was_interrupted,
        )

    def run_code_streaming(  # noqa: C901
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Iterator[OutputMessage | Result | CodeError]:
        """Execute code with streaming output.

        For Kaggle batch mode, this yields status progress while polling the
        remote job, then streams outputs from the downloaded artifacts.
        """
        if not self._batch_mode:
            yield from super().run_code_streaming(
                code=code,
                language=language,
                context=context,
                envs=envs,
                timeout=timeout,
            )
            return

        if self._executor is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"KaggleSandbox only supports Python, got: {language}")

        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            code = f"{env_code}\n{code}"

        self._interrupt_requested.clear()
        self._executing_event.set()

        accelerator = (
            self._extra_kwargs.get("accelerator")
            or self._extra_kwargs.get("gpu")
            or self.config.gpu
        )
        poll_interval = float(self._extra_kwargs.get("poll_interval", 2.0))
        timeout_seconds = float(timeout or self.config.timeout)

        try:
            submitted = self._executor.execute(
                code,
                wait=False,
                timeout=timeout_seconds,
                download_output=False,
                accelerator=accelerator,
            )
        except Exception as exc:
            self._executing_event.clear()
            self._interrupt_requested.clear()
            yield CodeError(
                name="SandboxExecutionError",
                value=f"Failed to submit Kaggle batch job: {exc}",
                traceback="",
            )
            return

        now = time.time()
        yield OutputMessage(
            line=f"[kaggle] submitted job: {submitted.slug}",
            timestamp=now,
            error=False,
        )

        status = self._normalize_status(getattr(submitted, "status", "QUEUED"))
        yield OutputMessage(line=f"[kaggle] status: {status}", timestamp=now, error=False)

        failure_message = getattr(submitted, "failure_message", None)
        deadline = time.monotonic() + timeout_seconds
        can_poll = hasattr(self._executor, "api") and hasattr(self._executor.api, "kernels_status")
        last_status = status

        while status not in _KAGGLE_TERMINAL_STATUSES and can_poll and time.monotonic() < deadline:
            time.sleep(poll_interval)
            response = self._executor.api.kernels_status(submitted.slug)
            status = self._normalize_status(getattr(response, "status", response))
            failure_message = getattr(response, "failure_message", None) or failure_message
            if status != last_status:
                yield OutputMessage(
                    line=f"[kaggle] status: {status}",
                    timestamp=time.time(),
                    error=False,
                )
                last_status = status

        if status not in _KAGGLE_TERMINAL_STATUSES and can_poll:
            failure_message = failure_message or (
                f"Timed out after {timeout_seconds:.1f}s waiting for Kaggle batch job"
            )
            status = status or "RUNNING"

        submitted.status = status
        submitted.failure_message = failure_message

        # Download artifacts and derive notebook/log fields for reply normalization.
        if hasattr(self._executor, "output"):
            try:
                output_dir = tempfile.mkdtemp(prefix="sandbox-kaggle-out-")
                files = self._executor.output(submitted.slug, output_dir)
                submitted.output_dir = output_dir
                submitted.output_files = list(files)
                self._populate_artifacts_from_files(submitted, list(files))
            except Exception as exc:
                yield OutputMessage(
                    line=f"[kaggle] warning: could not download outputs: {exc}",
                    timestamp=time.time(),
                    error=True,
                )

        reply = getattr(submitted, "kernel_reply", None)
        if reply is None and hasattr(submitted, "to_kernel_reply"):
            reply = submitted.to_kernel_reply()
            submitted.kernel_reply = reply

        if isinstance(reply, dict):
            for output in reply.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "stream":
                    stream_name = output.get("name", "stdout")
                    text = str(output.get("text", ""))
                    for line in text.splitlines():
                        yield OutputMessage(
                            line=line,
                            timestamp=time.time(),
                            error=stream_name == "stderr",
                        )
                elif output_type in ("execute_result", "display_data"):
                    yield Result(
                        data=output.get("data", {}),
                        is_main_result=output_type == "execute_result",
                        extra=output.get("metadata", {}),
                    )
                elif output_type == "error":
                    yield CodeError(
                        name=output.get("ename", "Error"),
                        value=output.get("evalue", ""),
                        traceback="\n".join(output.get("traceback", [])),
                    )

        if status != "COMPLETE":
            yield CodeError(
                name="KaggleExecutionError",
                value=failure_message or f"Kaggle execution failed with status: {status}",
                traceback=getattr(submitted, "log", "") or "",
            )

        self._executing_event.clear()
        self._interrupt_requested.clear()

    async def run_code_streaming_async(
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[OutputMessage | Result | CodeError]:
        """Async wrapper for Kaggle streaming execution."""
        for item in self.run_code_streaming(
            code=code,
            language=language,
            context=context,
            envs=envs,
            timeout=timeout,
        ):
            yield item

    def _run_code_batch(  # noqa: C901
        self,
        code: str,
        language: str,
        context: Context | None,
        on_stdout: OutputHandler[OutputMessage] | None,
        on_stderr: OutputHandler[OutputMessage] | None,
        on_result: OutputHandler[Result] | None,
        on_error: OutputHandler[CodeError] | None,
        envs: dict[str, str] | None,
        timeout: float | None,
    ) -> ExecutionResult:
        if self._executor is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"KaggleSandbox only supports Python, got: {language}")

        started_at = time.time()
        self._interrupt_requested.clear()
        self._executing_event.set()

        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            code = f"{env_code}\n{code}"

        accelerator = (
            self._extra_kwargs.get("accelerator")
            or self._extra_kwargs.get("gpu")
            or self.config.gpu
        )

        try:
            result = self._executor.execute(
                code,
                wait=True,
                timeout=float(timeout or self.config.timeout),
                download_output=True,
                accelerator=accelerator,
            )
        except Exception as e:
            self._executing_event.clear()
            was_interrupted = self._interrupt_requested.is_set()
            self._interrupt_requested.clear()
            execution_error = None
            if not was_interrupted:
                execution_error = f"Failed to execute Kaggle batch job: {e}"

            return ExecutionResult(
                execution_ok=False,
                execution_error=execution_error,
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
                interrupted=was_interrupted,
            )

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: CodeError | None = None

        now = time.time()
        reply = getattr(result, "kernel_reply", None)
        if reply is None and hasattr(result, "to_kernel_reply"):
            reply = result.to_kernel_reply()

        if isinstance(reply, dict):
            for output in reply.get("outputs", []):
                output_type = output.get("output_type")
                if output_type == "stream":
                    stream_name = output.get("name", "stdout")
                    text = output.get("text", "")
                    for line in str(text).splitlines():
                        msg = OutputMessage(line=line, timestamp=now, error=stream_name == "stderr")
                        if stream_name == "stderr":
                            stderr_messages.append(msg)
                            if on_stderr:
                                on_stderr(msg)
                        else:
                            stdout_messages.append(msg)
                            if on_stdout:
                                on_stdout(msg)
                elif output_type in ("execute_result", "display_data"):
                    output_result = Result(
                        data=output.get("data", {}),
                        is_main_result=output_type == "execute_result",
                        extra=output.get("metadata", {}),
                    )
                    results.append(output_result)
                    if on_result:
                        on_result(output_result)
                elif output_type == "error":
                    code_error = CodeError(
                        name=output.get("ename", "Error"),
                        value=output.get("evalue", ""),
                        traceback="\n".join(output.get("traceback", [])),
                    )
                    if on_error:
                        on_error(code_error)
        elif result.log:
            # Backward-compatible fallback for older jupyter-kernel-client versions.
            for line in result.log.splitlines():
                msg = OutputMessage(line=line, timestamp=now, error=False)
                stdout_messages.append(msg)
                if on_stdout:
                    on_stdout(msg)
            output_result = Result(
                data={"text/plain": result.log},
                is_main_result=True,
                extra={
                    "status": result.status,
                    "url": result.url,
                    "slug": result.slug,
                    "output_files": result.output_files,
                },
            )
            results.append(output_result)
            if on_result:
                on_result(output_result)

        if not result.succeeded and code_error is None:
            failure = result.failure_message or (
                f"Kaggle execution failed with status: {result.status}"
            )
            code_error = CodeError(
                name="KaggleExecutionError",
                value=failure,
                traceback=result.log or "",
            )
            if on_error:
                on_error(code_error)
            err_msg = OutputMessage(line=failure, timestamp=now, error=True)
            stderr_messages.append(err_msg)
            if on_stderr:
                on_stderr(err_msg)

        self._executing_event.clear()
        was_interrupted = self._interrupt_requested.is_set()
        self._interrupt_requested.clear()

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            execution_count=int(reply.get("execution_count", 0)) if isinstance(reply, dict) else 0,
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
            interrupted=was_interrupted,
        )

    def _get_internal_variable(self, name: str, context: Context | None = None):
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        return self._client.get_variable(name)

    def _set_internal_variable(self, name: str, value, context: Context | None = None) -> None:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        self._client.set_variable(name, value)
