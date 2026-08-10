# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Google Colab sandbox implementation.

This sandbox connects to an existing Google Colab runtime and executes code in
its kernel using :class:`code_sandboxes.google_colab.GoogleColabKernelClient`.

Unlike the Jupyter/Docker sandboxes, this sandbox does **not** provision a
runtime: a Colab runtime must already be running in a browser session. Reuse it
with either explicit ``server_url`` / ``kernel_id`` / ``proxy_token`` values or
by passing a Colab WebSocket ``channels_url``.
"""

from __future__ import annotations

import logging
import time
import uuid

from .base import Sandbox
from .google_colab import GoogleColabKernelClient, parse_google_colab_channels_url
from .exceptions import SandboxConfigurationError, SandboxNotStartedError
from .interfaces import ISandboxClient
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


class GoogleColabSandbox(Sandbox):
    """Sandbox backed by a Google Colab runtime.

    Args:
        config: Optional sandbox configuration.
        server_url: The Colab runtime proxy URL.
        kernel_id: The Colab kernel identifier to connect to.
        proxy_token: The Colab runtime proxy token.
        channels_url: Optional Colab channels URL to parse `server_url`,
            `kernel_id`, and `proxy_token` from.
        client_agent: Value advertised through the ``X-Colab-Client-Agent`` header.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        server_url: str | None = None,
        kernel_id: str | None = None,
        proxy_token: str | None = None,
        channels_url: str | None = None,
        client_agent: str = "code-sandboxes",
        **kwargs,
    ):
        super().__init__(config)
        # Allow configuration via SandboxConfig extras as a fallback.
        extras = getattr(self.config, "model_extra", None) or {}
        self._server_url = server_url or extras.get("server_url")
        self._kernel_id = kernel_id or extras.get("kernel_id")
        self._proxy_token = proxy_token or extras.get("proxy_token")
        self._channels_url = channels_url or extras.get("channels_url")
        self._client_agent = client_agent
        self._client = None
        self._sandbox_id = str(uuid.uuid4())
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="google_colab",
                title="Google Colab",
                language="python",
                owner="google",
                visibility="cloud",
                burning_rate=0.0,
                metadata={"variant": "google_colab"},
            )
        ]

    def start(self) -> None:
        if self._started:
            return

        if self._channels_url and (
            not self._server_url or not self._kernel_id or not self._proxy_token
        ):
            parsed_server_url, parsed_kernel_id, parsed_proxy_token = parse_google_colab_channels_url(
                self._channels_url
            )
            self._server_url = self._server_url or parsed_server_url
            self._kernel_id = self._kernel_id or parsed_kernel_id
            self._proxy_token = self._proxy_token or parsed_proxy_token

        if not self._server_url or not self._kernel_id or not self._proxy_token:
            raise SandboxConfigurationError(
                "GoogleColabSandbox requires 'server_url', 'kernel_id', and 'proxy_token'. "
                "Provide them directly, or pass 'channels_url' from an active Colab session."
            )

        self._client = GoogleColabKernelClient(
            server_url=self._server_url,
            kernel_id=self._kernel_id,
            proxy_token=self._proxy_token,
            client_agent=self._client_agent,
        )
        self._client.start()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="google_colab",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={"server_url": self._server_url, "kernel_id": self._client.id},
            config=self.config,
        )
        self._started = True

    @property
    def kernel_client(self) -> ISandboxClient | None:
        """The underlying Colab kernel client, if started."""
        return self._client

    def _setup_tool_caller(self) -> None:
        """Keep tool calling on the client side for Colab sandboxes."""
        return

    def stop(self) -> None:
        if not self._started:
            return
        if self._client is not None:
            try:
                # Do not shut down the Colab kernel; we only disconnect.
                self._client.stop(shutdown_kernel=False)
            except Exception:
                logger.debug("Ignoring error while stopping Colab client", exc_info=True)
            self._client = None
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
        if not self._started or self._client is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"GoogleColabSandbox only supports Python, got: {language}")

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

    def _get_internal_variable(self, name: str, context: Context | None = None):
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        return self._client.get_variable(name)

    def _set_internal_variable(self, name: str, value, context: Context | None = None) -> None:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        self._client.set_variable(name, value)
