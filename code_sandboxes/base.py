# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Abstract base class for code sandboxes."""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Any, Union

from .commands import SandboxCommands
from .filesystem import SandboxFilesystem
from .interfaces import ISandboxClient
from .models import (
    CodeError,
    Context,
    ExecutionResult,
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
    - eval: Simple Python exec() based, minimal isolation
    - docker: Docker container based, good isolation
    - jupyter: Jupyter Server with persistent kernel state
    - datalayer: Cloud-based Datalayer runtime, full isolation

    Features:
    - Code execution with result streaming
    - Filesystem operations (read, write, list, upload, download)
    - Command execution (run, exec, spawn)
    - Context management for state persistence
    - Snapshot support (for datalayer)
    - GPU/resource configuration (for datalayer)
    - Timeout and lifecycle management

    Example:
        with Sandbox.create(variant="datalayer") as sandbox:
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

    def __init__(self, config: SandboxConfig | None = None):
        """Initialize sandbox with configuration.

        Args:
            config: Sandbox configuration. Uses defaults if not provided.
        """
        self.config = config or SandboxConfig()
        self._info: SandboxInfo | None = None
        self._started = False
        self._default_context: Context | None = None
        self._files: SandboxFilesystem | None = None
        self._commands: SandboxCommands | None = None
        self._tags: dict[str, str] = {}
        self._created_at: float = 0.0
        self._tool_caller: Any | None = None  # Tool caller function for MCP tools
        self._executing_event = threading.Event()  # Set while code is running
        self._interrupt_requested = threading.Event()  # Set to request interruption

    @property
    def info(self) -> SandboxInfo | None:
        """Get information about this sandbox."""
        return self._info

    @property
    def is_started(self) -> bool:
        """Check if sandbox has been started."""
        return self._started

    @property
    def is_executing(self) -> bool:
        """Check if the sandbox is currently executing code."""
        return self._executing_event.is_set()

    @property
    def kernel_client(self) -> ISandboxClient | None:
        """Expose an optional kernel client interface for kernel-backed variants."""
        return None

    def interrupt(self) -> bool:
        """Request interruption of the currently running code.

        Returns:
            True if an interruption was requested (code was running),
            False if no code was running.
        """
        if not self._executing_event.is_set():
            return False
        self._interrupt_requested.set()
        return self._do_interrupt()

    def _do_interrupt(self) -> bool:
        """Perform the actual interrupt.  Subclasses should override this.

        The default implementation simply sets the interrupt flag.
        Subclasses can send kernel interrupt signals, raise async
        exceptions in threads, etc.

        Returns:
            True if the interrupt signal was delivered.
        """
        return True

    @property
    def sandbox_id(self) -> str | None:
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
    def create(  # noqa: C901
        cls,
        variant: SandboxVariant | str = SandboxVariant.DATALAYER,
        config: SandboxConfig | None = None,
        timeout: float | None = None,
        name: str | None = None,
        environment: str | None = None,
        gpu: str | None = None,
        cpu: float | None = None,
        memory: int | None = None,
        env: dict[str, str] | None = None,
        network_policy: str | None = None,
        allowed_hosts: list[str] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs,
    ) -> Sandbox:
        """Factory method to create a sandbox of the specified variant.

        This method provides a simple interface for creating sandboxes
        with different isolation levels and features:
        - Sandbox.create(timeout=60_000)
        - Sandbox.create(gpu="T4", timeout=300)

        Args:
            variant: The type of sandbox to create.
                - "eval": Simple Python exec() based, minimal isolation
                - "docker": Docker container based (requires Docker)
                - "jupyter": Jupyter Server with persistent kernel state
                - "datalayer": Cloud-based Datalayer runtime (default)
            config: Optional full configuration object (overrides individual params).
            timeout: Default timeout for code execution in seconds.
            name: Optional name for the sandbox.
            environment: Runtime environment (e.g., "python-cpu-env", "python-gpu-env").
            gpu: GPU type to use (e.g., "T4", "A100", "H100"). Only for datalayer.
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

            # With timeout
            sandbox = Sandbox.create(timeout=60)

            # With GPU (like Modal)
            sandbox = Sandbox.create(gpu="T4", environment="python-gpu-env")

            # development
            sandbox = Sandbox.create(variant="eval")
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

        from .eval_sandbox import EvalSandbox

        variant_value = variant.value if isinstance(variant, SandboxVariant) else variant

        if variant_value == "eval":
            sandbox = EvalSandbox(config=config, **kwargs)
        elif variant_value == "docker":
            # Import here to avoid circular imports
            from .docker_sandbox import DockerSandbox

            sandbox = DockerSandbox(config=config, **kwargs)
        elif variant_value == "jupyter":
            from .jupyter_sandbox import JupyterSandbox

            sandbox = JupyterSandbox(config=config, **kwargs)
        elif variant_value == "datalayer":
            from .datalayer_sandbox import DatalayerSandbox

            sandbox = DatalayerSandbox(config=config, **kwargs)
        elif variant_value == "colab":
            from .colab_sandbox import ColabSandbox

            sandbox = ColabSandbox(config=config, **kwargs)
        elif variant_value == "kaggle":
            from .kaggle_sandbox import KaggleSandbox

            sandbox = KaggleSandbox(config=config, **kwargs)
        elif variant_value == "monty":
            from .monty_sandbox import MontySandbox

            sandbox = MontySandbox(config=config, **kwargs)
        elif variant_value == "modal":
            from .modal_sandbox import ModalSandbox

            sandbox = ModalSandbox(config=config, **kwargs)
        else:
            raise ValueError(
                f"Unknown sandbox variant: {variant}. "
                "Supported variants: eval, docker, jupyter, "
                "datalayer, colab, kaggle, monty, modal"
            )

        # Set tags if provided
        if tags:
            sandbox.set_tags(tags)

        return sandbox

    @classmethod
    def from_id(cls, sandbox_id: str, **kwargs) -> Sandbox:
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
        # This is primarily for datalayer
        from .datalayer_sandbox import DatalayerSandbox

        return DatalayerSandbox.from_id(sandbox_id, **kwargs)

    @classmethod
    def list_environments(
        cls,
        variant: SandboxVariant | str = SandboxVariant.DATALAYER,
        **kwargs,
    ) -> list[SandboxEnvironment]:
        """List available environments for a given sandbox variant.

        Args:
            variant: Sandbox variant to query.
            **kwargs: Variant-specific parameters (e.g., token, run_url).

        Returns:
            List of SandboxEnvironment entries.
        """
        from .eval_sandbox import EvalSandbox

        variant_value = variant.value if isinstance(variant, SandboxVariant) else variant

        if variant_value == "eval":
            return EvalSandbox.list_environments()
        if variant_value == "docker":
            from .docker_sandbox import DockerSandbox

            return DockerSandbox.list_environments()
        if variant_value == "jupyter":
            from .jupyter_sandbox import JupyterSandbox

            return JupyterSandbox.list_environments()
        if variant_value == "datalayer":
            from .datalayer_sandbox import DatalayerSandbox

            return DatalayerSandbox.list_environments(**kwargs)
        raise ValueError(
            f"Unknown sandbox variant: {variant}. "
            "Supported variants: eval, docker, jupyter, "
            "datalayer"
        )

    @classmethod
    def list(
        cls,
        tags: dict[str, str] | None = None,
        **kwargs,
    ) -> Iterator[Sandbox]:
        """List all running sandboxes.

        Similar to Modal's Sandbox.list() method.

        Args:
            tags: Filter sandboxes by tags.
            **kwargs: Additional filter arguments.

        Yields:
            Sandbox instances.
        """
        from .datalayer_sandbox import DatalayerSandbox

        yield from DatalayerSandbox.list_all(tags=tags, **kwargs)

    def __enter__(self) -> Sandbox:
        """Context manager entry - starts the sandbox."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the sandbox."""
        self.stop()

    async def __aenter__(self) -> Sandbox:
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
        context: Context | None = None,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code in the sandbox.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context for maintaining state. If not provided,
                uses the default context.
            on_stdout: Callback for stdout messages.
            on_stderr: Callback for stderr messages.
            on_result: Callback for results.
            on_error: Callback for code errors (Python exceptions).
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
        context: Context | None = None,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
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
        context: Context | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Iterator[Union[OutputMessage, Result, CodeError]]:
        """Execute code with streaming output.

        Yields output messages, results, and errors as they are produced.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context for maintaining state.
            envs: Additional environment variables.
            timeout: Timeout in seconds.

        Yields:
            OutputMessage, Result, or CodeError objects.
        """
        # Default implementation: run and yield all at once
        execution = self.run_code(
            code=code,
            language=language,
            context=context,
            envs=envs,
            timeout=timeout,
        )
        yield from execution.logs.stdout
        yield from execution.logs.stderr
        yield from execution.results
        if not execution.execution_ok and execution.execution_error:
            yield CodeError(
                name="SandboxExecutionError", value=execution.execution_error, traceback=""
            )
        if execution.code_error:
            yield execution.code_error

    async def run_code_streaming_async(
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[Union[OutputMessage, Result, CodeError]]:
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
        if not execution.execution_ok and execution.execution_error:
            yield CodeError(
                name="SandboxExecutionError", value=execution.execution_error, traceback=""
            )
        if execution.code_error:
            yield execution.code_error

    def create_context(self, name: str | None = None) -> Context:
        """Create a new execution context.

        A context maintains state (variables, imports, etc.) between executions.

        Args:
            name: Optional name for the context. Auto-generated if not provided.

        Returns:
            A new Context object.
        """
        context_id = name or str(uuid.uuid4())
        return Context(id=context_id, language="python", cwd=self.config.working_dir)

    def get_variable(self, name: str, context: Context | None = None) -> Any:
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
        if not execution.execution_ok:
            from .exceptions import SandboxExecutionError

            raise SandboxExecutionError(execution.execution_error or "Sandbox execution failed")
        if execution.code_error:
            from .exceptions import VariableNotFoundError

            raise VariableNotFoundError(name)
        return self._get_internal_variable("__result__", context)

    def set_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        """Set a variable in the sandbox.

        Args:
            name: Name of the variable to set.
            value: Value to assign.
            context: Context to set the variable in. Uses default if not provided.
        """
        self._set_internal_variable(name, value, context)

    def set_variables(self, variables: dict[str, Any], context: Context | None = None) -> None:
        """Set multiple variables in the sandbox.

        Args:
            variables: Dictionary of variable names to values.
            context: Context to set variables in. Uses default if not provided.
        """
        for name, value in variables.items():
            self.set_variable(name, value, context)

    def register_tool_caller(self, tool_caller: Any) -> None:
        """Register a tool caller function for MCP tool invocations.

        The tool caller will be available to code running in the sandbox
        as `__call_tool__(tool_name, arguments)`. This allows sandbox code
        to invoke MCP tools through the provided caller.

        The tool caller is stored on the client side (sandbox object) and
        made available to sandbox code through the appropriate mechanism
        for each sandbox type.

        Args:
            tool_caller: An async function with signature:
                async def tool_caller(tool_name: str, arguments: dict) -> Any
        """
        self._tool_caller = tool_caller
        self._setup_tool_caller()

    def _setup_tool_caller(self) -> None:
        """Set up the tool caller in the sandbox environment.

        This method is called after registering a tool caller and should
        make `__call_tool__` available to sandbox code. Subclasses may
        override this for custom behavior.

        The default implementation injects the tool caller directly,
        which works for in-process sandboxes like eval.
        """
        if self._tool_caller is not None and self._started:
            self._set_internal_variable("__call_tool__", self._tool_caller)

    @abstractmethod
    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """Internal method to get a variable. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _set_internal_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        """Internal method to set a variable. Must be implemented by subclasses."""
        pass

    def install_packages(
        self, packages: list[str], timeout: float | None = None
    ) -> ExecutionResult:
        """Install Python packages in the sandbox.

        Args:
            packages: List of package names to install.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.
        """
        install_cmd = (
            f"import subprocess; subprocess.run(['pip', 'install'] + {packages!r}, check=True)"
        )
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
