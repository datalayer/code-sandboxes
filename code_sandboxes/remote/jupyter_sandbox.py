# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Local Jupyter-based sandbox implementation.

This sandbox runs a local Jupyter Server process (or connects to an existing
one) and uses ``jupyter-kernel-client`` to execute code in a persistent kernel.
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
import uuid
from urllib.parse import parse_qs, urlparse, urlunparse

import requests

import logging

from ..base import Sandbox
from ..exceptions import SandboxConfigurationError, SandboxNotStartedError
from ..models import (
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

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
DEFAULT_STARTUP_TIMEOUT = 30.0

logger = logging.getLogger(__name__)


class LocalJupyterSandbox(Sandbox):
    """Local Jupyter Server sandbox using a persistent kernel."""

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        server_url: Optional[str] = None,
        token: Optional[str] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        python_executable: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config)
        parsed_url = None
        parsed_token = None
        if server_url:
            parsed_url = urlparse(server_url)
            query = parse_qs(parsed_url.query)
            parsed_token = query.get("token", [None])[0]
            if parsed_token and token is None:
                token = parsed_token

        self._server_url = server_url
        if parsed_url and parsed_token:
            cleaned = parsed_url._replace(query="", fragment="")
            self._server_url = urlunparse(cleaned)
        self._token = token or uuid.uuid4().hex
        self._host = host
        self._port = port
        self._python_executable = python_executable or os.environ.get("PYTHON", "python")
        self._server_app = None
        self._server_thread: Optional[threading.Thread] = None
        self._client = None
        self._sandbox_id = str(uuid.uuid4())
        self._workdir: Optional[str] = None
        self._workdir_tmp: Optional[str] = None
        self._extra_kwargs = kwargs
        self._owns_server = server_url is None

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="local-jupyter",
                title="Local Jupyter",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": "local-jupyter"},
            )
        ]

    def _resolve_workdir(self) -> str:
        if self.config.working_dir:
            Path(self.config.working_dir).mkdir(parents=True, exist_ok=True)
            return self.config.working_dir
        if self._workdir:
            return self._workdir
        self._workdir_tmp = tempfile.mkdtemp(prefix="code-sandbox-")
        self._workdir = self._workdir_tmp
        return self._workdir

    @staticmethod
    def _is_port_available(host: str, port: int) -> bool:
        """Check if a port is available for binding.

        Args:
            host: The host address to check.
            port: The port number to check.

        Returns:
            True if the port is free, False otherwise.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
                return True
        except OSError:
            return False

    def _find_free_port(self) -> int:
        """Find a free port on the host by binding to port 0.

        The OS assigns an available random port.  The socket is closed
        immediately and the port number is returned.  A second check is
        performed to guard against the (unlikely) race where the port
        is grabbed between close and the Jupyter server bind.

        Returns:
            An available port number.

        Raises:
            SandboxConfigurationError: If no free port could be found.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self._host, 0))
                port = sock.getsockname()[1]
        except OSError as exc:
            raise SandboxConfigurationError(
                f"Could not find a free port on {self._host}: {exc}"
            ) from exc

        # Double-check that the port is still available after releasing
        if not self._is_port_available(self._host, port):
            raise SandboxConfigurationError(
                f"Port {port} was free but became unavailable immediately"
            )

        logger.info("Found free port %d on %s for Jupyter server", port, self._host)
        return port

    def _start_local_server(self) -> None:
        workdir = self._resolve_workdir()
        try:
            from jupyter_server.serverapp import ServerApp
        except Exception as exc:
            raise SandboxConfigurationError(
                "jupyter_server is required for LocalJupyterSandbox. "
                "Install it with: pip install code-sandboxes[test]"
            ) from exc

        # If port is 0, find a verified-free random port before starting.
        port = self._port
        if port == 0:
            port = self._find_free_port()
            self._port = port

        ServerApp.clear_instance()
        app = ServerApp.instance()
        app.initialize(
            argv=[
                "--no-browser",
                f"--ServerApp.token={self._token}",
                f"--ServerApp.port={port}",
                "--ServerApp.port_retries=0",
                "--ServerApp.allow_origin=*",
                f"--ServerApp.root_dir={workdir}",
            ],
        )
        self._server_app = app

        def _run_server():
            app.start()

        self._server_thread = threading.Thread(target=_run_server, daemon=True)
        self._server_thread.start()

        server_url = getattr(app, "connection_url", None) or getattr(app, "display_url", None)
        if not server_url:
            server_url = f"http://{self._host}:{app.port}"
        parsed = urlparse(server_url)
        query = parse_qs(parsed.query)
        token = query.get("token", [None])[0]
        cleaned = parsed._replace(query="", fragment="")

        self._server_url = urlunparse(cleaned).rstrip("/")
        self._token = token or self._token

    def _wait_for_server(self, timeout: float = DEFAULT_STARTUP_TIMEOUT) -> None:
        if not self._server_url:
            raise SandboxConfigurationError("Server URL not available")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get(
                    f"{self._server_url}/api/status",
                    params={"token": self._token},
                    timeout=2,
                )
                if response.ok:
                    return
            except Exception:
                time.sleep(0.5)
        raise SandboxConfigurationError("Timed out waiting for Jupyter Server")

    def _find_existing_kernel(self) -> str | None:
        """Find an existing pre-warmed kernel to reuse.

        Uses jupyter-server-client to list running kernels on the Jupyter server.
        If a pre-warmed kernel exists (idle, with 0 connections), returns its ID
        so we can connect to it instead of creating a new one.

        Returns:
            The kernel ID to reuse, or None if no suitable kernel is found.
        """
        try:
            from jupyter_server_client import JupyterServerClient
        except ImportError:
            logger.debug(
                "jupyter-server-client not available, will create a new kernel"
            )
            return None

        try:
            jsc = JupyterServerClient(
                base_url=self._server_url, token=self._token
            )
            kernels = jsc.kernels.list_kernels()
            if not kernels:
                logger.info("No existing kernels found, will create a new one")
                return None

            # Prefer a kernel with 0 connections (pre-warmed, not yet in use)
            for kernel in kernels:
                if kernel.connections == 0:
                    logger.info(
                        "Found pre-warmed kernel %s (connections=0, state=%s), reusing it",
                        kernel.id,
                        kernel.execution_state,
                    )
                    return kernel.id

            # Fall back to the first kernel if all have connections
            logger.info(
                "No idle kernel found, reusing first kernel %s (connections=%d)",
                kernels[0].id,
                kernels[0].connections,
            )
            return kernels[0].id
        except Exception as e:
            logger.warning(
                "Failed to list existing kernels: %s, will create a new one", e
            )
            return None

    def start(self) -> None:
        if self._started:
            return

        try:
            from jupyter_kernel_client import KernelClient
        except ImportError as exc:
            raise SandboxConfigurationError(
                "jupyter-kernel-client is required for LocalJupyterSandbox. "
                "Install it with: pip install code-sandboxes[test]"
            ) from exc

        if self._owns_server:
            self._start_local_server()

        self._wait_for_server(timeout=self.config.timeout or DEFAULT_STARTUP_TIMEOUT)

        # Try to reuse an existing pre-warmed kernel instead of creating a new one.
        # Jupyter runtimes pre-warm a kernel at startup; connecting to it avoids
        # unnecessary kernel proliferation (3 kernels → 2, or 2 → 1).
        kernel_id = self._find_existing_kernel()

        self._client = KernelClient(
            server_url=self._server_url, token=self._token, kernel_id=kernel_id
        )
        self._client.start()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="local-jupyter",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={"server_url": self._server_url},
            config=self.config,
        )
        self._started = True

    def _setup_tool_caller(self) -> None:
        """Keep tool calling on the client side for Jupyter sandboxes."""
        return

    def stop(self) -> None:
        if not self._started:
            return

        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None

        if self._server_app is not None and self._owns_server:
            try:
                if getattr(self._server_app, "io_loop", None):
                    self._server_app.io_loop.add_callback(self._server_app.stop)
                else:
                    self._server_app.stop()
            except Exception:
                pass
            self._server_app = None

        if self._server_thread is not None and self._owns_server:
            try:
                self._server_thread.join(timeout=5)
            except Exception:
                pass
            self._server_thread = None

        if self._workdir_tmp and os.path.isdir(self._workdir_tmp):
            try:
                import shutil

                shutil.rmtree(self._workdir_tmp, ignore_errors=True)
            except Exception:
                pass
            self._workdir_tmp = None

        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def _do_interrupt(self) -> bool:
        """Interrupt the Jupyter kernel via the REST API."""
        if not self._server_url or not self._client:
            return False
        try:
            # KernelClient exposes the kernel ID as the `.id` property
            kernel_id = getattr(self._client, 'id', None)
            if kernel_id:
                resp = requests.post(
                    f"{self._server_url}/api/kernels/{kernel_id}/interrupt",
                    params={"token": self._token},
                    timeout=5,
                )
                return resp.ok
            return False
        except Exception as e:
            logger.warning(f"Failed to interrupt Jupyter kernel: {e}")
            return False

    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[CodeError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"LocalJupyterSandbox only supports Python, got: {language}")

        started_at = time.time()

        # Track execution state for interrupt support
        self._interrupt_requested.clear()
        self._executing_event.set()

        if envs:
            env_code = "\n".join(
                f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items()
            )
            code = f"{env_code}\n{code}"

        try:
            reply = self._client.execute(code, timeout=timeout or self.config.timeout)
        except Exception as e:
            # Infrastructure failure or interruption
            self._executing_event.clear()
            was_interrupted = self._interrupt_requested.is_set()
            self._interrupt_requested.clear()
            return ExecutionResult(
                execution_ok=not was_interrupted,
                execution_error=f"Failed to execute code: {e}" if not was_interrupted else None,
                started_at=started_at,
                completed_at=time.time(),
                context_id=context.id if context else "default",
                interrupted=was_interrupted,
                code_error=CodeError(
                    name="KeyboardInterrupt",
                    value="Execution was interrupted",
                    traceback="",
                ) if was_interrupted else None,
            )

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: Optional[CodeError] = None
        exit_code: Optional[int] = None

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
                
                # Handle SystemExit specially - extract exit code
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

        # Clear execution tracking
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

    def _get_internal_variable(self, name: str, context: Optional[Context] = None):
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        return self._client.get_variable(name)

    def _set_internal_variable(
        self, name: str, value, context: Optional[Context] = None
    ) -> None:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        self._client.set_variable(name, value)
