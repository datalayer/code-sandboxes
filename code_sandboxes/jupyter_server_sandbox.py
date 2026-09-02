# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Jupyter-based sandbox implementation.

This sandbox runs a local Jupyter Server process (or connects to an existing
one) and uses ``jupyter-kernel-client`` to execute code in a persistent kernel.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Union
from urllib.parse import parse_qs, urlparse, urlunparse

import requests

from .base import Sandbox
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

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 0
DEFAULT_STARTUP_TIMEOUT = 30.0

#: How many of the server's last lines are kept, to quote when it will not
#: start. Enough for a traceback, not enough to hold a log in memory.
SERVER_OUTPUT_LINES = 50

logger = logging.getLogger(__name__)


class JupyterServerSandbox(Sandbox):
    """Jupyter Server sandbox using a persistent kernel.

    Pass ``headers`` to send extra HTTP headers on every request to an external
    Jupyter Server, for deployments whose credentials are not a token — for
    example a session ``Cookie`` plus ``X-XSRFToken`` from a password login.
    Such a deployment has no token to send, so pass ``token=None`` and let the
    headers authenticate: a token is only generated for a server this sandbox
    starts itself.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        server_url: str | None = None,
        token: str | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        python_executable: str | None = None,
        separate_process: bool = True,
        kernel_id: str | None = None,
        kernel_path: str | None = None,
        client_kwargs: dict | None = None,
        reuse_kernel: bool = True,
        headers: dict[str, str] | None = None,
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
        self._headers = dict(headers) if headers else {}
        # A generated token only means something for a server this sandbox
        # starts itself, where it is passed as --ServerApp.token. When talking to
        # an external server, honor the caller's token as given — including None,
        # so that other credentials (for example the session cookie and XSRF
        # header supplied via ``headers``) are what authenticate the requests
        # instead of a fabricated token the server has never seen.
        if token:
            self._token = token
        elif server_url is None:
            self._token = uuid.uuid4().hex
        else:
            self._token = None
        self._host = host
        self._port = port
        self._python_executable = python_executable or os.environ.get("PYTHON", "python")
        self._separate_process = separate_process
        self._server_app = None
        self._server_thread: threading.Thread | None = None
        self._server_process: subprocess.Popen | None = None
        #: The last lines the server wrote, to quote when it fails to start.
        self._server_output: deque[str] = deque(maxlen=SERVER_OUTPUT_LINES)
        self._client: ISandboxClient | None = None
        self._sandbox_id = str(uuid.uuid4())
        self._workdir: str | None = None
        self._workdir_tmp: str | None = None
        self._extra_kwargs = kwargs
        self._owns_server = server_url is None
        # Explicit kernel to connect to. When ``kernel_id`` is None and
        # ``reuse_kernel`` is True, the sandbox attempts to reuse a pre-warmed
        # kernel on the server; when ``kernel_id`` is None and
        # ``reuse_kernel`` is False, a brand-new kernel is created.
        self._kernel_id = kernel_id
        self._kernel_path = kernel_path
        self._client_kwargs = client_kwargs
        self._reuse_kernel = reuse_kernel

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="jupyter-server",
                title="Jupyter",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": "jupyter-server"},
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

        # If port is 0, find a verified-free random port before starting.
        port = self._port
        if port == 0:
            port = self._find_free_port()
            self._port = port

        if self._separate_process:
            self._start_local_server_subprocess(workdir, port)
        else:
            self._start_local_server_inprocess(workdir, port)

    def _start_local_server_subprocess(self, workdir: str, port: int) -> None:
        """Start the Jupyter server as a separate subprocess.

        This is the default mode.  It avoids event-loop conflicts when the
        caller already runs inside an async loop (e.g. uvicorn / uvloop).
        """
        cmd = [
            sys.executable,
            "-m",
            "jupyter_server",
            "--no-browser",
            f"--ServerApp.token={self._token}",
            f"--ServerApp.port={port}",
            "--ServerApp.port_retries=0",
            "--ServerApp.allow_origin=*",
            # A browser on another origin is a first-class client here: the
            # notebook and document editors of the workspace connect to this
            # server directly from a page served by the dev server or the SaaS.
            #
            # `allow_origin` alone is not enough for that. Starting a kernel is
            # a POST, and Jupyter's XSRF check rejects a cross-origin write
            # *before* the CORS headers are attached — so the browser reports a
            # missing `Access-Control-Allow-Origin` on a 403 and neither half of
            # the message names the real cause. Requests still have to carry the
            # token above; this only stops the cookie-based defence from
            # refusing a client that was never going to send a cookie.
            "--ServerApp.disable_check_xsrf=True",
            # Answer on both loopback names. The URL handed to the browser is
            # whatever the caller resolved, and a server bound only to
            # 127.0.0.1 refuses the same request addressed to localhost.
            f"--ServerApp.ip={self._host}",
            f"--ServerApp.root_dir={workdir}",
        ]

        logger.info(
            "Starting Jupyter server subprocess on %s:%d (workdir=%s)",
            self._host,
            port,
            workdir,
        )

        # Kept, not discarded.
        #
        # This was `DEVNULL` on both streams, so a server that failed to start
        # — a missing package, a port already taken, a bad argument — said why
        # into nothing, and the only thing anyone ever saw was "Timed out
        # waiting for Jupyter Server" thirty seconds later. Read on a thread so
        # the pipe cannot fill and block the server that IS starting.
        self._server_process = subprocess.Popen(  # noqa: S603 — argv built above, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Start in its own process group so we can kill the tree.
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        # A fresh buffer per start, so a previous run's drain thread cannot append
        # its last lines to the output quoted for this one. Declared in __init__.
        self._server_output = deque(maxlen=SERVER_OUTPUT_LINES)
        threading.Thread(
            target=self._drain_server_output,
            name=f"jupyter-server-{port}",
            daemon=True,
        ).start()

        self._server_url = f"http://{self._host}:{port}"

    def _drain_server_output(self) -> None:
        """Keep what the server says, so a failure can be quoted."""
        stream = getattr(self._server_process, "stdout", None)
        if stream is None:
            return
        with contextlib.suppress(Exception):
            for line in stream:
                text = line.rstrip()
                if text:
                    self._server_output.append(text)
                    logger.debug("[jupyter-server] %s", text)

    def _server_said(self) -> str:
        """The last of the server's own output, for an error message."""
        return "\n".join(getattr(self, "_server_output", ()))

    def _start_local_server_inprocess(self, workdir: str, port: int) -> None:
        """Start the Jupyter server in a daemon thread (legacy mode).

        Kept for environments where subprocess spawning is not desirable
        (e.g. unit tests, single-process setups).
        """
        try:
            from jupyter_server.serverapp import ServerApp
        except Exception as exc:
            raise SandboxConfigurationError(
                "jupyter_server is required for JupyterServerSandbox. "
                "Install it with: pip install code-sandboxes[test]"
            ) from exc

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
            # A server that has already exited is not going to answer, and
            # waiting out the timeout to say so turns a plain error — the
            # module is not installed, the port is taken — into a mystery.
            process = self._server_process
            if process is not None and process.poll() is not None:
                said = self._server_said()
                raise SandboxConfigurationError(
                    f"The Jupyter Server exited with code {process.returncode} "
                    f"before it was ready" + (f": {said}" if said else " and said nothing")
                )
            try:
                response = requests.get(
                    f"{self._server_url}/api/status",
                    params={"token": self._token},
                    headers=self._headers or None,
                    timeout=2,
                )
                if response.ok:
                    return
            except Exception:
                time.sleep(0.5)
        said = self._server_said()
        raise SandboxConfigurationError(
            f"Timed out waiting for Jupyter Server after {timeout:.0f}s"
            + (f": {said}" if said else "")
        )

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
            logger.debug("jupyter-server-client not available, will create a new kernel")
            return None

        try:
            jsc = JupyterServerClient(
                base_url=self._server_url,
                token=self._token,
                headers=self._headers or None,
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
            logger.warning("Failed to list existing kernels: %s, will create a new one", e)
            return None

    def start(self) -> None:
        if self._started:
            return

        try:
            from jupyter_kernel_client import JupyterKernelClient
        except ImportError as exc:
            raise SandboxConfigurationError(
                "jupyter-kernel-client is required for JupyterServerSandbox. "
                "Install it with: pip install code-sandboxes[test]"
            ) from exc

        if self._owns_server:
            self._start_local_server()

        self._wait_for_server(timeout=self.config.timeout or DEFAULT_STARTUP_TIMEOUT)

        # Decide which kernel to connect to:
        # - an explicit kernel_id always wins;
        # - otherwise, when reuse is enabled, reuse a pre-warmed kernel (Jupyter
        #   runtimes pre-warm one at startup) to avoid kernel proliferation;
        # - otherwise connect with no id so the client starts a brand-new kernel.
        if self._kernel_id is not None:
            kernel_id = self._kernel_id
        elif self._reuse_kernel:
            kernel_id = self._find_existing_kernel()
        else:
            kernel_id = None

        client_kwargs: dict = {
            "server_url": self._server_url,
            "token": self._token,
            "kernel_id": kernel_id,
            "client_kwargs": self._client_kwargs or None,
        }
        # Only forward headers when the caller supplied some, so the call stays
        # byte-identical for the common token/anonymous case.
        if self._headers:
            client_kwargs["headers"] = self._headers
        self._client = JupyterKernelClient(**client_kwargs)

        self._client.start(path=self._kernel_path)

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="jupyter-server",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={"server_url": self._server_url, "kernel_id": self._client.id},
            config=self.config,
        )
        self._started = True

    @property
    def kernel_client(self) -> ISandboxClient | None:
        """The underlying ``jupyter_kernel_client.JupyterKernelClient``.

        Exposed so callers that need the full low-level kernel API (for
        example streaming execution via ``execute_interactive``) can delegate
        to the same client the sandbox uses internally. ``None`` until
        :meth:`start` has been called.
        """
        return self._client

    def _setup_tool_caller(self) -> None:
        """Keep tool calling on the client side for Jupyter sandboxes."""
        return

    def stop(self) -> None:  # noqa: C901
        if not self._started:
            return

        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.stop()
            self._client = None

        # Terminate subprocess-based server
        if self._server_process is not None and self._owns_server:
            try:
                # Kill the entire process group (server + any children)
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self._server_process.pid), signal.SIGTERM)
                else:
                    self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                # It would not go quietly; there is nothing after `kill`.
                with contextlib.suppress(Exception):
                    self._server_process.kill()
            self._server_process = None

        # Terminate in-process server (legacy mode)
        if self._server_app is not None and self._owns_server:
            with contextlib.suppress(Exception):
                if getattr(self._server_app, "io_loop", None):
                    self._server_app.io_loop.add_callback(self._server_app.stop)
                else:
                    self._server_app.stop()
            self._server_app = None

        if self._server_thread is not None and self._owns_server:
            with contextlib.suppress(Exception):
                self._server_thread.join(timeout=5)
            self._server_thread = None

        if self._workdir_tmp and os.path.isdir(self._workdir_tmp):
            import shutil

            # `ignore_errors` already swallows what the tree throws.
            shutil.rmtree(self._workdir_tmp, ignore_errors=True)
            self._workdir_tmp = None

        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def _do_interrupt(self) -> bool:
        """Interrupt the Jupyter kernel via the REST API."""
        if not self._server_url or not self._client:
            return False
        try:
            # JupyterKernelClient exposes the kernel ID as the `.id` property
            kernel_id = getattr(self._client, "id", None)
            if kernel_id:
                resp = requests.post(
                    f"{self._server_url}/api/kernels/{kernel_id}/interrupt",
                    params={"token": self._token},
                    headers=self._headers or None,
                    timeout=5,
                )
                return resp.ok
            return False
        except Exception as e:
            logger.warning(f"Failed to interrupt Jupyter kernel: {e}")
            return False

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
            raise ValueError(f"JupyterServerSandbox only supports Python, got: {language}")

        started_at = time.time()

        # Track execution state for interrupt support
        self._interrupt_requested.clear()
        self._executing_event.set()

        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            code = f"{env_code}\n{code}"

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: CodeError | None = None
        exit_code: int | None = None

        def consume_stream(content: dict, timestamp: float) -> None:
            """One stream message, kept as the kernel wrote it.

            `splitlines()` was losing the one fact that distinguishes a
            finished line from a chunk written with `end=""`. A loop printing
            dots side by side arrived as six separate lines, because the
            terminator was discarded here and reinvented when the messages
            were joined — output no kernel had produced.

            `keepends=True` preserves it, and each message says whether its own
            chunk ended. The dots then reassemble as a notebook shows them: on
            one line, growing.
            """
            is_stderr = content.get("name") == "stderr"
            sink = stderr_messages if is_stderr else stdout_messages
            handler = on_stderr if is_stderr else on_stdout
            for part in content.get("text", "").splitlines(keepends=True):
                terminated = part.endswith("\n")
                msg = OutputMessage(
                    line=part.rstrip("\n"),
                    timestamp=timestamp,
                    error=is_stderr,
                    terminated=terminated,
                )
                sink.append(msg)
                if handler:
                    handler(msg)

        def consume_result(output_type: str, content: dict) -> None:
            """One rendered value. Only `execute_result` is the cell's own answer."""
            result = Result(
                data=content.get("data", {}),
                is_main_result=output_type == "execute_result",
                extra=content.get("metadata", {}),
            )
            results.append(result)
            if on_result:
                on_result(result)

        def consume_error(content: dict) -> None:
            """One raised exception — or, for SystemExit, a status rather than a failure."""
            nonlocal code_error, exit_code

            ename = content.get("ename", "Error")
            evalue = content.get("evalue", "")
            if ename == "SystemExit":
                try:
                    exit_code = int(evalue) if evalue else 0
                except (ValueError, TypeError):
                    exit_code = 1 if evalue else 0
                return
            code_error = CodeError(
                name=ename,
                value=evalue,
                traceback="\n".join(content.get("traceback", [])),
            )
            if on_error:
                on_error(code_error)

        def consume(message: dict) -> None:
            """Normalize one IOPub message while it is still arriving."""
            header = message.get("header", {})
            output_type = header.get("msg_type") or message.get("msg_type")
            content = message.get("content", {})

            if output_type == "stream":
                consume_stream(content, time.time())
            elif output_type in ("execute_result", "display_data"):
                consume_result(output_type, content)
            elif output_type == "error":
                consume_error(content)
            elif output_type == "clear_output" and not content.get("wait", False):
                # Not "wait": the cell asked for the output so far to be dropped now.
                stdout_messages.clear()
                stderr_messages.clear()
                results.clear()

        try:
            reply = self._client.execute_interactive(
                code,
                timeout=timeout or self.config.timeout,
                output_hook=consume,
            )
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
                )
                if was_interrupted
                else None,
            )

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
            execution_count=reply.get("content", {}).get("execution_count", 0),
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
            interrupted=was_interrupted,
        )

    async def run_code_streaming_async(
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[Union[OutputMessage, Result, CodeError]]:
        """Yield each output as the kernel produces it.

        The base implementation is streaming in shape only: it awaits the whole
        execution and then replays what it collected, so a cell that prints for
        three seconds delivered everything in one burst at the end. Nothing
        downstream could stream, however well it was written — the A2UI surface
        above this was emitting correct incremental messages, all of them
        microseconds apart, after the run had finished.

        Nothing new has to be observed to fix that. `run_code` already accepts
        `on_stdout`, `on_stderr`, `on_result` and `on_error`, and already calls
        them from its IOPub hook *while the messages are arriving* — the
        information was live and only the return value was not. This turns
        those callbacks into an async iterator.

        The bridge is a queue, because `execute_interactive` blocks: the
        execution runs on a worker thread and the callbacks hand items across
        with `call_soon_threadsafe`, which is the only safe way to touch an
        asyncio primitive from another thread. The generator then drains the
        queue as items land, rather than waiting on the thread.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Union[OutputMessage, Result, CodeError, None]] = (
            asyncio.Queue()
        )

        def emit(item: Union[OutputMessage, Result, CodeError]) -> None:
            # Called on the execution thread; hop to the loop's thread before
            # touching the queue.
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def execute() -> ExecutionResult:
            try:
                return self.run_code(
                    code=code,
                    language=language,
                    context=context,
                    on_stdout=emit,
                    on_stderr=emit,
                    on_result=emit,
                    on_error=emit,
                    envs=envs,
                    timeout=timeout,
                )
            finally:
                # The sentinel closes the generator whatever happened, so a
                # raising execution cannot leave a consumer awaiting forever.
                loop.call_soon_threadsafe(queue.put_nowait, None)

        execution_task = loop.run_in_executor(None, execute)

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        execution = await execution_task

        # An infrastructure failure never reaches the callbacks — it is the
        # reason there were none — so it is reported here, once, at the end.
        if not execution.execution_ok and execution.execution_error:
            yield CodeError(
                name="SandboxExecutionError",
                value=execution.execution_error,
                traceback="",
            )

    def _get_internal_variable(self, name: str, context: Context | None = None):
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        return self._client.get_variable(name)

    def _set_internal_variable(self, name: str, value, context: Context | None = None) -> None:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        self._client.set_variable(name, value)
