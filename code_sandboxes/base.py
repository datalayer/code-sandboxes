# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Abstract base class for code sandboxes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterator, Optional, Union
import uuid

from .commands import SandboxCommands
from .filesystem import SandboxFilesystem
from .models import (
    Context,
    Execution,
    ExecutionError,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxVariant,
)



class Sandbox(ABC):
    """Abstract base class for code execution sandboxes.

    A sandbox provides a safe, isolated environment for executing code.
    Different implementations provide different isolation levels:
    - local-eval: Simple Python exec() based, minimal isolation
    - local-docker: Docker container based, good isolation
    - local-jupyter: Local Jupyter Server with persistent kernel state
    - datalayer-runtime: Cloud-based Datalayer runtime, full isolation

    Features inspired by E2B and Modal:
    - Code execution with result streaming
    - Filesystem operations (read, write, list, upload, download)
    - Command execution (run, exec, spawn)
    - Context management for state persistence
    - Snapshot support (for datalayer-runtime)
    - GPU/resource configuration (for datalayer-runtime)
    - Timeout and lifecycle management

    Example:
        with Sandbox.create(variant="datalayer-runtime") as sandbox:
            # Execute code
            result = sandbox.run_code("x = 1 + 1")
            result = sandbox.run_code("print(x)")  # prints 2

            # Use filesystem
            sandbox.files.write("/data/test.txt", "Hello")
            content = sandbox.files.read("/data/test.txt")

            # Run commands
            result = sandbox.commands.run("ls -la")

    Attributes:
        config: The sandbox configuration.
        info: Information about the running sandbox.
        files: Filesystem operations.
        commands: Command execution operations.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """Initialize sandbox with configuration.

        Args:
            config: Sandbox configuration. Uses defaults if not provided.
        """
        self.config = config or SandboxConfig()
        self._info: Optional[SandboxInfo] = None
        self._started = False
        self._default_context: Optional[Context] = None
        self._files: Optional[SandboxFilesystem] = None
        self._commands: Optional[SandboxCommands] = None
        self._tags: dict[str, str] = {}
        self._created_at: float = 0.0

    @property
    def info(self) -> Optional[SandboxInfo]:
        """Get information about this sandbox."""
        return self._info

    @property
    def is_started(self) -> bool:
        """Check if sandbox has been started."""
        return self._started

    @property
    def sandbox_id(self) -> Optional[str]:
        """Get the sandbox ID."""
        return self._info.id if self._info else None

    @property
    def files(self) -> SandboxFilesystem:
        """Get filesystem operations interface.

        Returns:
            SandboxFilesystem for file operations.
        """
        if self._files is None:
            self._files = SandboxFilesystem(self)
        return self._files

    @property
    def commands(self) -> SandboxCommands:
        """Get command execution interface.

        Returns:
            SandboxCommands for running terminal commands.
        """
        if self._commands is None:
            self._commands = SandboxCommands(self)
        return self._commands

    @property
    def tags(self) -> dict[str, str]:
        """Get sandbox tags (key-value pairs for metadata)."""
        return self._tags.copy()

    def set_tags(self, tags: dict[str, str]) -> None:
        """Set sandbox tags.

        Tags can be used to filter and organize sandboxes.
        Similar to Modal's sandbox tagging feature.

        Args:
            tags: Dictionary of tag names to values.
        """
        self._tags.update(tags)


    @classmethod
    def create(
        cls,
        variant: SandboxVariant | str = SandboxVariant.DATALAYER_RUNTIME,
        config: Optional[SandboxConfig] = None,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        environment: Optional[str] = None,
        gpu: Optional[str] = None,
        cpu: Optional[float] = None,
        memory: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
        network_policy: Optional[str] = None,
        allowed_hosts: Optional[list[str]] = None,
        tags: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> "Sandbox":
        """Factory method to create a sandbox of the specified variant.

        This method provides a simple interface similar to E2B and Modal:
        - E2B: Sandbox.create(timeout=60_000)
        - Modal: Sandbox.create(gpu="T4", timeout=300)

        Args:
            variant: The type of sandbox to create.
                - "local-eval": Simple Python exec() based, minimal isolation
                - "local-docker": Docker container based (requires Docker)
                - "local-jupyter": Local Jupyter Server with persistent kernel state
                - "datalayer-runtime": Cloud-based Datalayer runtime (default)
            config: Optional full configuration object (overrides individual params).
            timeout: Default timeout for code execution in seconds.
            name: Optional name for the sandbox.
            environment: Runtime environment (e.g., "python-cpu-env", "python-gpu-env").
            gpu: GPU type to use (e.g., "T4", "A100", "H100"). Only for datalayer-runtime.
            cpu: CPU cores to allocate.
            memory: Memory limit in MB.
            env: Environment variables to set in the sandbox.
            network_policy: Network access policy (inherit, none, allowlist, all).
            allowed_hosts: Allowlist of hosts when policy is allowlist.
            tags: Metadata tags for the sandbox.
            **kwargs: Additional arguments passed to the sandbox constructor.

        Returns:
            A Sandbox instance of the specified variant.

        Raises:
            ValueError: If the variant is not supported.

        Example:
            # Simple usage
            sandbox = Sandbox.create()

            # With timeout (like E2B)
            sandbox = Sandbox.create(timeout=60)

            # With GPU (like Modal)
            sandbox = Sandbox.create(gpu="T4", environment="python-gpu-env")

            # Local development
            sandbox = Sandbox.create(variant="local-eval")
        """
        # Build config from individual parameters if not provided
        if config is None:
            config = SandboxConfig(
                timeout=timeout or 30.0,
                environment=environment or "python-cpu-env",
                memory_limit=memory * 1024 * 1024 if memory else None,
                cpu_limit=cpu,
                env_vars=env or {},
                gpu=gpu,
                name=name,
                network_policy=network_policy or "inherit",
                allowed_hosts=allowed_hosts or [],
            )

        from .local.eval_sandbox import LocalEvalSandbox

        variant_value = variant.value if isinstance(variant, SandboxVariant) else variant

        if variant_value == "local-eval":
            sandbox = LocalEvalSandbox(config=config, **kwargs)
        elif variant_value == "local-docker":
            # Import here to avoid circular imports
            from .local.docker_sandbox import LocalDockerSandbox

            sandbox = LocalDockerSandbox(config=config, **kwargs)
        elif variant_value == "local-jupyter":
            from .local.jupyter_sandbox import LocalJupyterSandbox

            sandbox = LocalJupyterSandbox(config=config, **kwargs)
        elif variant_value == "datalayer-runtime":
            from .remote.datalayer_sandbox import DatalayerSandbox

            sandbox = DatalayerSandbox(config=config, **kwargs)
        else:
            raise ValueError(
                f"Unknown sandbox variant: {variant}. "
                "Supported variants: local-eval, local-docker, local-jupyter, "
                "datalayer-runtime"
            )

        # Set tags if provided
        if tags:
            sandbox.set_tags(tags)

        return sandbox

    @classmethod
    def from_id(cls, sandbox_id: str, **kwargs) -> "Sandbox":
        """Retrieve an existing sandbox by its ID.

        Similar to Modal's Sandbox.from_id() method.

        Args:
            sandbox_id: The unique identifier of the sandbox.
            **kwargs: Additional arguments.

        Returns:
            A Sandbox instance connected to the existing sandbox.

        Raises:
            SandboxNotFoundError: If no sandbox with the given ID exists.
        """
        # This is primarily for datalayer-runtime
        from .remote.datalayer_sandbox import DatalayerSandbox

        return DatalayerSandbox.from_id(sandbox_id, **kwargs)

    @classmethod
    def list_environments(
        cls,
        variant: SandboxVariant | str = SandboxVariant.DATALAYER_RUNTIME,
        **kwargs,
    ) -> list[SandboxEnvironment]:
        """List available environments for a given sandbox variant.

        Args:
            variant: Sandbox variant to query.
            **kwargs: Variant-specific parameters (e.g., token, run_url).

        Returns:
            List of SandboxEnvironment entries.
        """
        from .local.eval_sandbox import LocalEvalSandbox

        variant_value = variant.value if isinstance(variant, SandboxVariant) else variant

        if variant_value == "local-eval":
            return LocalEvalSandbox.list_environments()
        if variant_value == "local-docker":
            from .local.docker_sandbox import LocalDockerSandbox

            return LocalDockerSandbox.list_environments()
        if variant_value == "local-jupyter":
            from .local.jupyter_sandbox import LocalJupyterSandbox

            return LocalJupyterSandbox.list_environments()
        if variant_value == "datalayer-runtime":
            from .remote.datalayer_sandbox import DatalayerSandbox

            return DatalayerSandbox.list_environments(**kwargs)
        raise ValueError(
            f"Unknown sandbox variant: {variant}. "
            "Supported variants: local-eval, local-docker, local-jupyter, "
            "datalayer-runtime"
        )

    @classmethod
    def list(
        cls,
        tags: Optional[dict[str, str]] = None,
        **kwargs,
    ) -> Iterator["Sandbox"]:
        """List all running sandboxes.

        Similar to Modal's Sandbox.list() method.

        Args:
            tags: Filter sandboxes by tags.
            **kwargs: Additional filter arguments.

        Yields:
            Sandbox instances.
        """
        from .remote.datalayer_sandbox import DatalayerSandbox

        yield from DatalayerSandbox.list_all(tags=tags, **kwargs)

    def __enter__(self) -> "Sandbox":
        """Context manager entry - starts the sandbox."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the sandbox."""
        self.stop()

    async def __aenter__(self) -> "Sandbox":
        """Async context manager entry."""
        await self.start_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop_async()

    @abstractmethod
    def start(self) -> None:
        """Start the sandbox.

        Must be called before any code execution. Called automatically
        when using the sandbox as a context manager.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the sandbox and release resources.

        Called automatically when exiting the context manager.
        """
        pass

    async def start_async(self) -> None:
        """Async version of start(). Default implementation calls sync version."""
        self.start()

    async def stop_async(self) -> None:
        """Async version of stop(). Default implementation calls sync version."""
        self.stop()

    @abstractmethod
    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[ExecutionError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Execution:
        """Execute code in the sandbox.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context for maintaining state. If not provided,
                uses the default context.
            on_stdout: Callback for stdout messages.
            on_stderr: Callback for stderr messages.
            on_result: Callback for results.
            on_error: Callback for errors.
            envs: Additional environment variables for this execution.
            timeout: Timeout in seconds. Uses config default if not provided.

        Returns:
            Execution result containing output, results, and any errors.
        """
        pass

    async def run_code_async(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[ExecutionError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Execution:
        """Async version of run_code(). Default implementation calls sync version."""
        return self.run_code(
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

    def run_code_streaming(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[Union[OutputMessage, Result, ExecutionError]]:
        """Execute code with streaming output.

        Yields output messages, results, and errors as they are produced.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context for maintaining state.
            envs: Additional environment variables.
            timeout: Timeout in seconds.

        Yields:
            OutputMessage, Result, or ExecutionError objects.
        """
        # Default implementation: run and yield all at once
        execution = self.run_code(
            code=code,
            language=language,
            context=context,
            envs=envs,
            timeout=timeout,
        )
        for msg in execution.logs.stdout:
            yield msg
        for msg in execution.logs.stderr:
            yield msg
        for result in execution.results:
            yield result
        if execution.error:
            yield execution.error

    async def run_code_streaming_async(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[Union[OutputMessage, Result, ExecutionError]]:
        """Async version of run_code_streaming()."""
        execution = await self.run_code_async(
            code=code,
            language=language,
            context=context,
            envs=envs,
            timeout=timeout,
        )
        for msg in execution.logs.stdout:
            yield msg
        for msg in execution.logs.stderr:
            yield msg
        for result in execution.results:
            yield result
        if execution.error:
            yield execution.error

    def create_context(self, name: Optional[str] = None) -> Context:
        """Create a new execution context.

        A context maintains state (variables, imports, etc.) between executions.

        Args:
            name: Optional name for the context. Auto-generated if not provided.

        Returns:
            A new Context object.
        """
        context_id = name or str(uuid.uuid4())
        return Context(id=context_id, language="python", cwd=self.config.working_dir)

    def get_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Get a variable from the sandbox.

        Args:
            name: Name of the variable to retrieve.
            context: Context to get the variable from. Uses default if not provided.

        Returns:
            The value of the variable.

        Raises:
            VariableNotFoundError: If the variable doesn't exist.
        """
        # Default implementation using code execution
        execution = self.run_code(f"__result__ = {name}", context=context)
        if execution.error:
            from .exceptions import VariableNotFoundError

            raise VariableNotFoundError(name)
        return self._get_internal_variable("__result__", context)

    def set_variable(self, name: str, value: Any, context: Optional[Context] = None) -> None:
        """Set a variable in the sandbox.

        Args:
            name: Name of the variable to set.
            value: Value to assign.
            context: Context to set the variable in. Uses default if not provided.
        """
        self._set_internal_variable(name, value, context)

    def set_variables(
        self, variables: dict[str, Any], context: Optional[Context] = None
    ) -> None:
        """Set multiple variables in the sandbox.

        Args:
            variables: Dictionary of variable names to values.
            context: Context to set variables in. Uses default if not provided.
        """
        for name, value in variables.items():
            self.set_variable(name, value, context)

    @abstractmethod
    def _get_internal_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Internal method to get a variable. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _set_internal_variable(
        self, name: str, value: Any, context: Optional[Context] = None
    ) -> None:
        """Internal method to set a variable. Must be implemented by subclasses."""
        pass

    def install_packages(
        self, packages: list[str], timeout: Optional[float] = None
    ) -> Execution:
        """Install Python packages in the sandbox.

        Args:
            packages: List of package names to install.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.
        """
        install_cmd = f"import subprocess; subprocess.run(['pip', 'install'] + {packages!r}, check=True)"
        return self.run_code(install_cmd, timeout=timeout or 300)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the sandbox.

        Args:
            local_path: Path to the local file.
            remote_path: Destination path in the sandbox.
        """
        with open(local_path, "rb") as f:
            content = f.read()
        self._write_file(remote_path, content)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the sandbox.

        Args:
            remote_path: Path to the file in the sandbox.
            local_path: Destination path on the local filesystem.
        """
        content = self._read_file(remote_path)
        with open(local_path, "wb") as f:
            f.write(content)

    def _write_file(self, path: str, content: bytes) -> None:
        """Write a file in the sandbox. Override in subclasses for better performance."""
        import base64

        encoded = base64.b64encode(content).decode("utf-8")
        code = f"""
import base64
with open({path!r}, 'wb') as f:
    f.write(base64.b64decode({encoded!r}))
"""
        self.run_code(code)

    def _read_file(self, path: str) -> bytes:
        """Read a file from the sandbox. Override in subclasses for better performance."""
        import base64

        code = f"""
import base64
with open({path!r}, 'rb') as f:
    __file_content__ = base64.b64encode(f.read()).decode('utf-8')
"""
        self.run_code(code)
        encoded = self.get_variable("__file_content__")
        return base64.b64decode(encoded)
