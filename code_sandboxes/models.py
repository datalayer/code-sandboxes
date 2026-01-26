# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Models for code execution results and contexts.

Inspired by E2B Code Interpreter and Modal Sandbox models.

Uses Pydantic for:
- Automatic validation and type coercion
- JSON serialization/deserialization
- Better integration with FastAPI and modern Python APIs
- Clear schema definitions
"""

from enum import Enum
from typing import Any, Callable, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class SandboxEnvironment(BaseModel):
    """Environment information exposed by the sandbox API.

    Mirrors the Datalayer environment fields where available.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    title: str
    language: str = "python"
    owner: str = "local"
    visibility: str = "local"
    burning_rate: float = 0.0
    metadata: Optional[dict[str, Any]] = None


class MIMEType(str, Enum):
    """Common MIME types for execution results."""

    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    TEXT_MARKDOWN = "text/markdown"
    APPLICATION_JSON = "application/json"
    IMAGE_PNG = "image/png"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_SVG = "image/svg+xml"
    IMAGE_GIF = "image/gif"
    APPLICATION_PDF = "application/pdf"


class SandboxStatus(str, Enum):
    """Status of a sandbox."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"


class SandboxVariant(str, Enum):
    """Supported sandbox variants."""

    LOCAL_EVAL = "local-eval"
    LOCAL_DOCKER = "local-docker"
    LOCAL_JUPYTER = "local-jupyter"
    DATALAYER_RUNTIME = "datalayer-runtime"


class GPUType(str, Enum):
    """Available GPU types for cloud sandboxes."""

    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    H100 = "H100"
    L4 = "L4"


class ResourceConfig(BaseModel):
    """Resource configuration for sandbox.

    Similar to Modal's resource specification.

    Attributes:
        cpu: CPU cores to allocate.
        memory: Memory limit in MB.
        gpu: GPU type to use.
        gpu_count: Number of GPUs.
        disk: Disk size in GB.
    """

    model_config = ConfigDict(extra="allow")

    cpu: Optional[float] = None
    memory: Optional[int] = Field(default=None, description="Memory limit in MB")
    gpu: Optional[str] = None
    gpu_count: int = 1
    disk: Optional[int] = Field(default=None, description="Disk size in GB")

    def __repr__(self) -> str:
        parts = []
        if self.cpu:
            parts.append(f"cpu={self.cpu}")
        if self.memory:
            parts.append(f"memory={self.memory}MB")
        if self.gpu:
            parts.append(f"gpu={self.gpu}x{self.gpu_count}")
        return f"ResourceConfig({', '.join(parts) or 'default'})"


class OutputMessage(BaseModel):
    """A single output message from code execution.

    Attributes:
        line: The content of the output line.
        timestamp: Unix timestamp when the output was produced.
        error: Whether this is an error output (stderr).
    """

    model_config = ConfigDict(extra="allow")

    line: str
    timestamp: float = 0.0
    error: bool = False


class Logs(BaseModel):
    """Container for stdout and stderr logs.

    Attributes:
        stdout: List of stdout output messages.
        stderr: List of stderr output messages.
    """

    model_config = ConfigDict(extra="allow")

    stdout: list[OutputMessage] = Field(default_factory=list)
    stderr: list[OutputMessage] = Field(default_factory=list)

    @property
    def stdout_text(self) -> str:
        """Get stdout as a single string."""
        return "\n".join(msg.line for msg in self.stdout)

    @property
    def stderr_text(self) -> str:
        """Get stderr as a single string."""
        return "\n".join(msg.line for msg in self.stderr)


class Result(BaseModel):
    """A single result from code execution.

    Can contain multiple representations of the same data (e.g., text, HTML, image).

    Attributes:
        data: Dictionary mapping MIME types to their content.
        is_main_result: Whether this is the main result of the execution.
        extra: Additional metadata about the result.
    """

    model_config = ConfigDict(extra="allow")

    data: dict[str, Any] = Field(default_factory=dict)
    is_main_result: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> Optional[str]:
        """Get text/plain representation if available."""
        return self.data.get(MIMEType.TEXT_PLAIN) or self.data.get("text/plain")

    @property
    def html(self) -> Optional[str]:
        """Get text/html representation if available."""
        return self.data.get(MIMEType.TEXT_HTML) or self.data.get("text/html")

    @property
    def markdown(self) -> Optional[str]:
        """Get text/markdown representation if available."""
        return self.data.get(MIMEType.TEXT_MARKDOWN) or self.data.get("text/markdown")

    @property
    def json(self) -> Optional[Any]:
        """Get application/json representation if available."""
        return self.data.get(MIMEType.APPLICATION_JSON) or self.data.get("application/json")

    @property
    def png(self) -> Optional[str]:
        """Get base64-encoded PNG image if available."""
        return self.data.get(MIMEType.IMAGE_PNG) or self.data.get("image/png")

    @property
    def jpeg(self) -> Optional[str]:
        """Get base64-encoded JPEG image if available."""
        return self.data.get(MIMEType.IMAGE_JPEG) or self.data.get("image/jpeg")

    @property
    def svg(self) -> Optional[str]:
        """Get SVG image if available."""
        return self.data.get(MIMEType.IMAGE_SVG) or self.data.get("image/svg+xml")

    def __repr__(self) -> str:
        if self.text:
            return f"Result(text={self.text[:50]}...)" if len(self.text) > 50 else f"Result(text={self.text})"
        return f"Result(types={list(self.data.keys())})"


class CodeError(BaseModel):
    """Error information from code that raised an exception.

    This represents an error in the user's Python code, not an infrastructure failure.

    Attributes:
        name: The error class name (e.g., "ValueError", "SyntaxError").
        value: The error message.
        traceback: Full traceback as a string.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    value: str
    traceback: str = ""

    def __str__(self) -> str:
        if self.traceback:
            return f"{self.traceback}\n{self.name}: {self.value}"
        return f"{self.name}: {self.value}"

    def __repr__(self) -> str:
        return f"CodeError(name={self.name!r}, value={self.value!r})"


class Context(BaseModel):
    """Execution context for maintaining state across code executions.

    A context represents an isolated execution environment where variables,
    imports, and function definitions persist between executions.

    Attributes:
        id: Unique identifier for this context.
        language: Programming language for this context.
        cwd: Current working directory for file operations.
        env: Environment variables for this context.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    language: str = "python"
    cwd: Optional[str] = None
    env: dict[str, str] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Context(id={self.id!r}, language={self.language!r})"


class ExecutionResult(BaseModel):
    """Complete result of a code execution.

    The execution result distinguishes between two levels of failure:

    1. **Execution-level failure** (`execution_ok=False`): The sandbox infrastructure
       failed to execute the code. This could be due to connection issues, timeout
       waiting for kernel, resource exhaustion, etc. When `execution_ok` is False,
       `execution_error` contains details about what went wrong.

    2. **Code-level error** (`code_error` is not None): The sandbox successfully
       executed the code, but the Python code itself raised an exception. This is
       a normal execution result where the user's code encountered an error.

    Attributes:
        results: List of results produced by the execution (display outputs, return values).
        logs: Stdout and stderr logs from execution.

        # Execution-level status (infrastructure)
        execution_ok: Whether the sandbox infrastructure successfully executed the code.
                      False means a sandbox/infrastructure failure occurred.
        execution_error: Details about infrastructure failure when execution_ok=False.

        # Code-level status (user's code)
        code_error: Error information if the user's Python code raised an exception.
                    This is populated when code ran but encountered a runtime error.

        # Metadata
        execution_count: The execution counter (like Jupyter's In[n]).
        context_id: ID of the context where this was executed.
        started_at: Unix timestamp when execution started.
        completed_at: Unix timestamp when execution completed.
        interrupted: Whether execution was cancelled/interrupted.
    """

    model_config = ConfigDict(extra="allow")

    # Results and logs
    results: list[Result] = Field(default_factory=list)
    logs: Logs = Field(default_factory=Logs)

    # Execution-level (infrastructure) status
    execution_ok: bool = Field(
        default=True,
        description="Whether the sandbox infrastructure successfully executed the code"
    )
    execution_error: Optional[str] = Field(
        default=None,
        description="Details about infrastructure failure when execution_ok=False"
    )

    # Code-level (user code) status
    code_error: Optional[CodeError] = Field(
        default=None,
        description="Error information if the user's Python code raised an exception"
    )

    # Metadata
    execution_count: int = 0
    context_id: Optional[str] = None
    started_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when execution started"
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when execution completed"
    )
    interrupted: bool = Field(
        default=False,
        description="Whether execution was cancelled/interrupted"
    )
    exit_code: Optional[int] = Field(
        default=None,
        description="Exit code when code calls sys.exit() or script terminates. None means no explicit exit."
    )

    @property
    def text(self) -> Optional[str]:
        """Get the main text result if available."""
        for result in self.results:
            if result.text:
                return result.text
        return None

    @property
    def success(self) -> bool:
        """Whether execution completed successfully with no errors.

        Returns True only if:
        - Infrastructure executed the code successfully (execution_ok=True)
        - The code itself did not raise an exception (code_error=None)
        - Execution was not interrupted
        - Exit code is 0 or not set (None means normal completion without explicit exit)
        """
        return (
            self.execution_ok
            and self.code_error is None
            and not self.interrupted
            and (self.exit_code is None or self.exit_code == 0)
        )

    @property
    def duration(self) -> Optional[float]:
        """Execution duration in seconds, if timing info available."""
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None

    @property
    def stdout(self) -> str:
        """Get stdout as a single string."""
        return self.logs.stdout_text

    @property
    def stderr(self) -> str:
        """Get stderr as a single string."""
        return self.logs.stderr_text

    def __repr__(self) -> str:
        if not self.execution_ok:
            return f"ExecutionResult(execution_ok=False, execution_error={self.execution_error!r})"
        if self.success:
            status = "success"
        elif self.code_error:
            status = f"code_error={self.code_error.name}"
        elif self.exit_code is not None and self.exit_code != 0:
            status = f"exit_code={self.exit_code}"
        else:
            status = "failed"
        duration_str = f", duration={self.duration:.2f}s" if self.duration else ""
        return f"ExecutionResult({status}, results={len(self.results)}, execution_count={self.execution_count}{duration_str})"


# Type alias for output handlers (callbacks)
OutputHandler = Callable[[T], None]


class SandboxConfig(BaseModel):
    """Configuration for sandbox creation.

    Inspired by E2B and Modal configuration options.

    Attributes:
        timeout: Default timeout for code execution in seconds.
        memory_limit: Memory limit in bytes (for Docker/Datalayer sandboxes).
        cpu_limit: CPU limit (for Docker/Datalayer sandboxes).
        environment: Environment name for Datalayer sandboxes.
        working_dir: Default working directory.
        env_vars: Default environment variables.
        gpu: GPU type to use (e.g., "T4", "A100").
        name: Optional name for the sandbox.
        network_policy: Network access policy for code execution.
        allowed_hosts: Optional allowlist of hostnames/IPs when policy is allowlist.
        idle_timeout: Time in seconds before idle sandbox is terminated.
        max_lifetime: Maximum lifetime in seconds.
    """

    model_config = ConfigDict(extra="allow")

    timeout: float = 30.0
    memory_limit: Optional[int] = None
    cpu_limit: Optional[float] = None
    environment: str = "python-cpu-env"
    working_dir: Optional[str] = None
    env_vars: dict[str, str] = Field(default_factory=dict)
    gpu: Optional[str] = None
    name: Optional[str] = None
    network_policy: Literal["inherit", "none", "allowlist", "all"] = "inherit"
    allowed_hosts: list[str] = Field(default_factory=list)
    idle_timeout: Optional[float] = None
    max_lifetime: float = 86400.0  # 24 hours default like Modal


class SandboxInfo(BaseModel):
    """Information about a running sandbox.

    Inspired by E2B's getInfo() and Modal's sandbox info.

    Attributes:
        id: Unique identifier for the sandbox.
        variant: The sandbox variant (local-eval, local-docker, local-jupyter, datalayer-runtime).
        status: Current status of the sandbox.
        created_at: Unix timestamp when the sandbox was created.
        end_at: Unix timestamp when the sandbox will be terminated.
        config: The configuration used to create this sandbox.
        name: Name of the sandbox if set.
        metadata: Additional metadata about the sandbox.
        resources: Resource configuration for the sandbox.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    variant: str
    status: SandboxStatus = SandboxStatus.RUNNING
    created_at: float = 0.0
    end_at: Optional[float] = None
    config: Optional[SandboxConfig] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resources: Optional[ResourceConfig] = None

    @property
    def remaining_time(self) -> Optional[float]:
        """Get remaining time in seconds before sandbox terminates."""
        import time
        if self.end_at:
            return max(0, self.end_at - time.time())
        return None

    def __repr__(self) -> str:
        return f"SandboxInfo(id={self.id!r}, status={self.status.value}, variant={self.variant!r})"


class SnapshotInfo(BaseModel):
    """Information about a sandbox snapshot.

    Snapshots allow saving and restoring sandbox state.
    Similar to Modal's snapshot_filesystem feature.

    Attributes:
        id: Unique identifier for the snapshot.
        name: Name of the snapshot.
        sandbox_id: ID of the sandbox this snapshot was taken from.
        created_at: Unix timestamp when the snapshot was created.
        size: Size of the snapshot in bytes.
        description: Optional description.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    sandbox_id: str
    created_at: float = 0.0
    size: int = 0
    description: str = ""

    def __repr__(self) -> str:
        return f"SnapshotInfo(id={self.id!r}, name={self.name!r})"


class TunnelInfo(BaseModel):
    """Information about a tunnel to a sandbox port.

    Similar to Modal's Tunnel interface.

    Attributes:
        port: The port in the sandbox.
        url: The external URL to access the port.
        protocol: The protocol (http, https, tcp).
    """

    model_config = ConfigDict(extra="allow")

    port: int
    url: str
    protocol: str = "https"

    def __repr__(self) -> str:
        return f"TunnelInfo(port={self.port}, url={self.url!r})"
