# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Datalayer Runtime-based sandbox implementation.

This sandbox uses the Datalayer platform for cloud-based code execution,
providing full isolation and scalable compute resources.

Inspired by E2B and Modal sandbox APIs.
"""

import time
import uuid
from typing import Any, Iterator, Optional

from ..base import Sandbox
from ..exceptions import (
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxNotStartedError,
    SandboxSnapshotError,
)
from ..models import (
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
    SnapshotInfo,
)


class DatalayerSandbox(Sandbox):
    """A sandbox using Datalayer Runtime for cloud-based code execution.

    This sandbox provides full isolation, scalable compute (CPU/GPU),
    and supports snapshots for state persistence.

    Inspired by E2B Code Interpreter and Modal Sandbox APIs:
    - E2B-like: Simple creation, timeout management, file operations
    - Modal-like: GPU support, exec, snapshots, tagging

    Example:
        from code_sandboxes import Sandbox

        # Simple E2B-style usage
        with Sandbox.create(timeout=60) as sandbox:
            result = sandbox.run_code("print('Hello!')")
            files = sandbox.files.list("/")

        # Modal-style with GPU
        with Sandbox.create(gpu="T4", environment="python-gpu-env") as sandbox:
            sandbox.run_code("import torch; print(torch.cuda.is_available())")

        # With explicit API key
        with Sandbox.create(token="your-api-key") as sandbox:
            sandbox.run_code("x = 1 + 1")

    Attributes:
        client: The Datalayer client instance.
        runtime: The runtime service instance.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
        snapshot_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Datalayer sandbox.

        Args:
            config: Sandbox configuration.
            token: Datalayer API token. If not provided, uses DATALAYER_API_KEY
                environment variable.
            run_url: Datalayer server URL. If not provided, uses default.
            snapshot_name: Name of snapshot to restore from (optional).
            **kwargs: Additional arguments passed to DatalayerClient.
        """
        super().__init__(config)
        self._token = token
        self._run_url = run_url
        self._snapshot_name = snapshot_name
        self._client = None
        self._runtime = None
        self._sandbox_id = str(uuid.uuid4())
        self._extra_kwargs = kwargs
        self._end_at: Optional[float] = None

    @property
    def client(self):
        """Get the Datalayer client instance."""
        return self._client

    @property
    def runtime(self):
        """Get the runtime service instance."""
        return self._runtime

    @property
    def object_id(self) -> str:
        """Get the sandbox object ID (Modal-style)."""
        return self._sandbox_id

    @classmethod
    def from_id(cls, sandbox_id: str, **kwargs) -> "DatalayerSandbox":
        """Retrieve an existing sandbox by its ID.

        Similar to Modal's Sandbox.from_id() method.

        Args:
            sandbox_id: The unique identifier of the sandbox.
            **kwargs: Additional arguments (token, run_url).

        Returns:
            A DatalayerSandbox instance connected to the existing runtime.
        """
        # Create a new sandbox instance with the existing ID
        sandbox = cls(**kwargs)
        sandbox._sandbox_id = sandbox_id
        # Connect to existing runtime - this would need runtime lookup
        # For now, this is a placeholder that would need datalayer_core support
        return sandbox

    @classmethod
    def list_all(
        cls,
        tags: Optional[dict[str, str]] = None,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
    ) -> Iterator["DatalayerSandbox"]:
        """List all running sandboxes.

        Similar to Modal's Sandbox.list() method.

        Args:
            tags: Filter sandboxes by tags.
            token: API token for authentication.
            run_url: Datalayer server URL.

        Yields:
            DatalayerSandbox instances.
        """
        try:
            from datalayer_core import DatalayerClient
            from datalayer_core.utils.urls import DatalayerURLs
        except ImportError:
            return

        try:
            if run_url:
                urls = DatalayerURLs.from_run_url(run_url)
                client = DatalayerClient(urls=urls, token=token)
            else:
                client = DatalayerClient(token=token)

            runtimes = client.list_runtimes()

            for runtime in runtimes:
                sandbox = cls(token=token, run_url=run_url)
                sandbox._client = client
                sandbox._runtime = runtime
                sandbox._sandbox_id = runtime.uid or str(uuid.uuid4())
                sandbox._started = True
                sandbox._info = SandboxInfo(
                    id=sandbox._sandbox_id,
                    variant="datalayer-runtime",
                    status=SandboxStatus.RUNNING,
                    created_at=time.time(),
                    name=runtime.name,
                    metadata={
                        "network_policy": sandbox.config.network_policy,
                        "allowed_hosts": sandbox.config.allowed_hosts,
                    },
                )
                yield sandbox
        except Exception:
            return

    @classmethod
    def list_environments(
        cls,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
    ) -> list[SandboxEnvironment]:
        try:
            from datalayer_core import DatalayerClient
            from datalayer_core.utils.urls import DatalayerURLs
        except ImportError:
            return []

        try:
            if run_url:
                urls = DatalayerURLs.from_run_url(run_url)
                client = DatalayerClient(urls=urls, token=token)
            else:
                client = DatalayerClient(token=token)

            environments = client.list_environments()
            return [
                SandboxEnvironment(
                    name=env.name,
                    title=env.title,
                    language=env.language,
                    owner=env.owner,
                    visibility=env.visibility,
                    burning_rate=float(env.burning_rate),
                    metadata=env.metadata,
                )
                for env in environments
            ]
        except Exception:
            return []

    def start(self) -> None:
        """Start the sandbox by creating a Datalayer runtime.

        Similar to E2B's sandbox creation with timeout support.

        Raises:
            SandboxConfigurationError: If configuration is invalid.
            SandboxConnectionError: If connection to Datalayer fails.
        """
        if self._started:
            return

        try:
            # Import here to avoid hard dependency
            from datalayer_core import DatalayerClient
            from datalayer_core.utils.urls import DatalayerURLs
        except ImportError as e:
            raise SandboxConfigurationError(
                "datalayer_core package is required for DatalayerSandbox. "
                "Install it with: pip install datalayer_core"
            ) from e

        try:
            # Create client with optional custom URL
            if self._run_url:
                urls = DatalayerURLs.from_run_url(self._run_url)
                self._client = DatalayerClient(urls=urls, token=self._token)
            else:
                self._client = DatalayerClient(token=self._token)

            # Calculate time reservation
            # Default to the platform default (10 minutes) unless max_lifetime is explicitly set
            from datalayer_core.utils.defaults import DEFAULT_TIME_RESERVATION

            default_max_lifetime = SandboxConfig().max_lifetime
            if self.config.max_lifetime != default_max_lifetime:
                lifetime_minutes = int(self.config.max_lifetime / 60)
                time_reservation = max(10, min(lifetime_minutes, 1440))  # Max 24 hours
            else:
                time_reservation = int(DEFAULT_TIME_RESERVATION)

            # Determine environment based on GPU config
            environment = self.config.environment
            if self.config.gpu and "gpu" not in environment.lower():
                # Try to use a GPU environment if GPU is requested
                environment = "python-gpu-env"

            # Build sandbox name
            sandbox_name = self.config.name or f"sandbox-{self._sandbox_id[:8]}"

            # Create the runtime (optionally from snapshot)
            if self._snapshot_name:
                self._runtime = self._client.create_runtime(
                    name=sandbox_name,
                    environment=environment,
                    time_reservation=time_reservation,
                    snapshot_name=self._snapshot_name,
                )
            else:
                self._runtime = self._client.create_runtime(
                    name=sandbox_name,
                    environment=environment,
                    time_reservation=time_reservation,
                )

            # Start the runtime
            self._runtime._start()

            self._default_context = self.create_context("default")

            # Calculate end time
            self._created_at = time.time()
            self._end_at = self._created_at + self.config.max_lifetime

            # Build resource config
            resources = None
            if self.config.gpu or self.config.cpu_limit or self.config.memory_limit:
                resources = ResourceConfig(
                    cpu=self.config.cpu_limit,
                    memory=self.config.memory_limit // (1024 * 1024) if self.config.memory_limit else None,
                    gpu=self.config.gpu,
                )

            self._info = SandboxInfo(
                id=self._sandbox_id,
                variant="datalayer-runtime",
                status=SandboxStatus.RUNNING,
                created_at=self._created_at,
                end_at=self._end_at,
                config=self.config,
                    metadata={
                        "network_policy": self.config.network_policy,
                        "allowed_hosts": self.config.allowed_hosts,
                    },
                name=sandbox_name,
                resources=resources,
            )
            self._started = True

        except Exception as e:
            url = self._run_url or "default"
            raise SandboxConnectionError(url, str(e)) from e

    def stop(self) -> None:
        """Stop the sandbox and release the Datalayer runtime.

        Similar to E2B's kill() and Modal's terminate().
        """
        if not self._started:
            return

        try:
            if self._runtime:
                self._runtime._stop()
        except Exception:
            pass  # Best effort cleanup

        self._runtime = None
        self._client = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    # Alias for Modal compatibility
    def terminate(self) -> None:
        """Terminate the sandbox. Alias for stop()."""
        self.stop()

    # Alias for E2B compatibility
    def kill(self) -> None:
        """Kill the sandbox. Alias for stop()."""
        self.stop()

    def set_timeout(self, timeout_seconds: float) -> None:
        """Change the sandbox timeout during runtime.

        Similar to E2B's set_timeout method. Resets the timeout to the new value.

        Args:
            timeout_seconds: New timeout in seconds from now.
        """
        if not self._started:
            raise SandboxNotStartedError()

        self._end_at = time.time() + timeout_seconds
        if self._info:
            self._info.end_at = self._end_at

    def get_info(self) -> SandboxInfo:
        """Retrieve sandbox information.

        Similar to E2B's getInfo() method.

        Returns:
            SandboxInfo object with current sandbox state.
        """
        if self._info:
            return self._info
        return SandboxInfo(
            id=self._sandbox_id,
            variant="datalayer-runtime",
            status=SandboxStatus.PENDING if not self._started else SandboxStatus.RUNNING,
        )

    def wait(self, raise_on_termination: bool = True) -> None:
        """Wait for the sandbox to finish.

        Similar to Modal's wait() method.

        Args:
            raise_on_termination: Whether to raise if sandbox terminates with error.
        """
        # For cloud sandboxes, this would wait for the runtime to complete
        # Currently just a placeholder
        pass

    def poll(self) -> Optional[int]:
        """Check if the sandbox has finished running.

        Similar to Modal's poll() method.

        Returns:
            None if still running, exit code otherwise.
        """
        if self._started:
            return None
        return 0

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
        """Execute code in the Datalayer runtime.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context (currently not used, runtime maintains state).
            on_stdout: Callback for stdout messages.
            on_stderr: Callback for stderr messages.
            on_result: Callback for results.
            on_error: Callback for code errors (Python exceptions).
            envs: Environment variables (set before execution).
            timeout: Timeout in seconds.

        Returns:
            Execution result.

        Raises:
            SandboxNotStartedError: If the sandbox hasn't been started.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        started_at = time.time()

        # Set environment variables if provided
        if envs:
            env_code = "\n".join(
                f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items()
            )
            self._runtime.execute(env_code)

        # Execute the code
        execution_timeout = timeout or self.config.timeout
        try:
            response = self._runtime.execute(code, timeout=execution_timeout)
        except Exception as e:
            # Infrastructure failure - couldn't execute the code
            return ExecutionResult(
                results=[],
                logs=Logs(),
                execution_ok=False,
                execution_error=f"Failed to execute code: {e}",
                execution_count=0,
                context_id=context.id if context else "default",
                started_at=started_at,
                completed_at=time.time(),
            )

        # Parse the response
        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: Optional[CodeError] = None

        current_time = time.time()

        # Process stdout
        if hasattr(response, "stdout") and response.stdout:
            for line in response.stdout.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=False)
                stdout_messages.append(msg)
                if on_stdout:
                    on_stdout(msg)

        # Process stderr
        if hasattr(response, "stderr") and response.stderr:
            for line in response.stderr.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=True)
                stderr_messages.append(msg)
                if on_stderr:
                    on_stderr(msg)

        # Process results
        if hasattr(response, "result") and response.result is not None:
            result = Result(
                data={"text/plain": str(response.result)},
                is_main_result=True,
            )
            results.append(result)
            if on_result:
                on_result(result)

        # Process display data (rich output)
        if hasattr(response, "display_data") and response.display_data:
            for display in response.display_data:
                result = Result(
                    data=display.get("data", {}),
                    is_main_result=False,
                    extra=display.get("metadata", {}),
                )
                results.append(result)
                if on_result:
                    on_result(result)

        # Process errors (code exceptions)
        if hasattr(response, "error") and response.error:
            code_error = CodeError(
                name=response.error.get("ename", "Error"),
                value=response.error.get("evalue", ""),
                traceback="\n".join(response.error.get("traceback", [])),
            )
            if on_error:
                on_error(code_error)

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            execution_count=getattr(response, "execution_count", 0),
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
        )

    def _get_internal_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Get a variable from the runtime.

        Args:
            name: Variable name.
            context: Context (not used, runtime maintains single namespace).

        Returns:
            The variable value.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        return self._runtime.get_variable(name)

    def _set_internal_variable(
        self, name: str, value: Any, context: Optional[Context] = None
    ) -> None:
        """Set a variable in the runtime.

        Args:
            name: Variable name.
            value: Value to set.
            context: Context (not used, runtime maintains single namespace).
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        self._runtime.set_variable(name, value)

    def create_snapshot(
        self,
        name: str,
        description: str = "",
    ) -> SnapshotInfo:
        """Create a snapshot of the current runtime state.

        Similar to Modal's snapshot_filesystem feature. This allows saving
        the current state of the sandbox for later restoration.

        Args:
            name: Name for the snapshot.
            description: Optional description.

        Returns:
            SnapshotInfo with the snapshot details.

        Raises:
            SandboxNotStartedError: If sandbox is not running.
            SandboxSnapshotError: If snapshot creation fails.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        try:
            snapshot = self._runtime.create_snapshot(name=name, description=description)
            return SnapshotInfo(
                id=snapshot.uid,
                name=name,
                sandbox_id=self._sandbox_id,
                created_at=time.time(),
                description=description,
            )
        except Exception as e:
            raise SandboxSnapshotError("create", str(e)) from e

    def list_snapshots(self) -> list[SnapshotInfo]:
        """List all snapshots.

        Returns:
            List of SnapshotInfo objects.
        """
        if not self._client:
            return []

        try:
            snapshots = self._client.list_snapshots()
            return [
                SnapshotInfo(
                    id=s.uid,
                    name=s.name,
                    sandbox_id="",
                    created_at=getattr(s, "created_at", 0),
                    description=getattr(s, "description", ""),
                )
                for s in snapshots
            ]
        except Exception:
            return []

    def install_packages(
        self, packages: list[str], timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Install Python packages in the runtime.

        Uses pip to install packages. Similar to E2B's package installation.

        Args:
            packages: List of package names to install.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.

        Example:
            sandbox.install_packages(["pandas", "numpy", "matplotlib"])
        """
        # Use %pip magic for better Jupyter integration
        pip_cmd = f"%pip install {' '.join(packages)}"
        return self.run_code(pip_cmd, timeout=timeout or 300)

    def install_requirements(
        self, requirements_path: str, timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Install packages from a requirements file.

        Args:
            requirements_path: Path to requirements.txt file in the sandbox.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.
        """
        pip_cmd = f"%pip install -r {requirements_path}"
        return self.run_code(pip_cmd, timeout=timeout or 300)

    def open_file(self, path: str, mode: str = "r") -> "SandboxFileHandle":
        """Open a file in the sandbox.

        Similar to Modal's sandbox.open() method.

        Args:
            path: Path to the file.
            mode: File mode ('r', 'w', 'rb', 'wb', 'a').

        Returns:
            SandboxFileHandle for file operations.
        """
        from ..filesystem import SandboxFileHandle
        return SandboxFileHandle(self, path, mode)
