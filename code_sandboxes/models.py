# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Models for code execution results and contexts.

Inspired by E2B Code Interpreter and Modal Sandbox models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional, TypeVar

T = TypeVar("T")


@dataclass
class SandboxEnvironment:
    """Environment information exposed by the sandbox API.

    Mirrors the Datalayer environment fields where available.
    """

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


@dataclass
class ResourceConfig:
    """Resource configuration for sandbox.

    Similar to Modal's resource specification.

    Attributes:
        cpu: CPU cores to allocate.
        memory: Memory limit in MB.
        gpu: GPU type to use.
        gpu_count: Number of GPUs.
        disk: Disk size in GB.
    """

    cpu: Optional[float] = None
    memory: Optional[int] = None  # MB
    gpu: Optional[str] = None
    gpu_count: int = 1
    disk: Optional[int] = None  # GB

    def __repr__(self) -> str:
        parts = []
        if self.cpu:
            parts.append(f"cpu={self.cpu}")
        if self.memory:
            parts.append(f"memory={self.memory}MB")
        if self.gpu:
            parts.append(f"gpu={self.gpu}x{self.gpu_count}")
        return f"ResourceConfig({', '.join(parts) or 'default'})"



@dataclass
class OutputMessage:
    """A single output message from code execution.

    Attributes:
        line: The content of the output line.
        timestamp: Unix timestamp when the output was produced.
        error: Whether this is an error output (stderr).
    """

    line: str
    timestamp: float = 0.0
    error: bool = False


@dataclass
class Logs:
    """Container for stdout and stderr logs.

    Attributes:
        stdout: List of stdout output messages.
        stderr: List of stderr output messages.
    """

    stdout: list[OutputMessage] = field(default_factory=list)
    stderr: list[OutputMessage] = field(default_factory=list)

    @property
    def stdout_text(self) -> str:
        """Get stdout as a single string."""
        return "\n".join(msg.line for msg in self.stdout)

    @property
    def stderr_text(self) -> str:
        """Get stderr as a single string."""
        return "\n".join(msg.line for msg in self.stderr)


@dataclass
class Result:
    """A single result from code execution.

    Can contain multiple representations of the same data (e.g., text, HTML, image).

    Attributes:
        data: Dictionary mapping MIME types to their content.
        is_main_result: Whether this is the main result of the execution.
        extra: Additional metadata about the result.
    """

    data: dict[str, Any] = field(default_factory=dict)
    is_main_result: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

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


@dataclass
class ExecutionError:
    """Error information from failed code execution.

    Attributes:
        name: The error class name (e.g., "ValueError", "SyntaxError").
        value: The error message.
        traceback: Full traceback as a string.
    """

    name: str
    value: str
    traceback: str = ""

    def __str__(self) -> str:
        if self.traceback:
            return f"{self.traceback}\n{self.name}: {self.value}"
        return f"{self.name}: {self.value}"

    def __repr__(self) -> str:
        return f"ExecutionError(name={self.name!r}, value={self.value!r})"


@dataclass
class Context:
    """Execution context for maintaining state across code executions.

    A context represents an isolated execution environment where variables,
    imports, and function definitions persist between executions.

    Attributes:
        id: Unique identifier for this context.
        language: Programming language for this context.
        cwd: Current working directory for file operations.
        env: Environment variables for this context.
    """

    id: str
    language: str = "python"
    cwd: Optional[str] = None
    env: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Context(id={self.id!r}, language={self.language!r})"


@dataclass
class Execution:
    """Complete result of a code execution.

    Attributes:
        results: List of results produced by the execution.
        logs: Stdout and stderr logs.
        error: Error information if execution failed.
        execution_count: The execution counter (like Jupyter's In[n]).
        context_id: ID of the context where this was executed.
    """

    results: list[Result] = field(default_factory=list)
    logs: Logs = field(default_factory=Logs)
    error: Optional[ExecutionError] = None
    execution_count: int = 0
    context_id: Optional[str] = None

    @property
    def text(self) -> Optional[str]:
        """Get the main text result if available."""
        for result in self.results:
            if result.text:
                return result.text
        return None

    @property
    def success(self) -> bool:
        """Whether the execution completed without errors."""
        return self.error is None

    @property
    def stdout(self) -> str:
        """Get stdout as a single string."""
        return self.logs.stdout_text

    @property
    def stderr(self) -> str:
        """Get stderr as a single string."""
        return self.logs.stderr_text

    def __repr__(self) -> str:
        status = "success" if self.success else f"error={self.error.name}"
        return f"Execution({status}, results={len(self.results)}, execution_count={self.execution_count})"


# Type alias for output handlers (callbacks)
OutputHandler = Callable[[T], None]


@dataclass
class SandboxConfig:
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

    timeout: float = 30.0
    memory_limit: Optional[int] = None
    cpu_limit: Optional[float] = None
    environment: str = "python-cpu-env"
    working_dir: Optional[str] = None
    env_vars: dict[str, str] = field(default_factory=dict)
    gpu: Optional[str] = None
    name: Optional[str] = None
    network_policy: Literal["inherit", "none", "allowlist", "all"] = "inherit"
    allowed_hosts: list[str] = field(default_factory=list)
    idle_timeout: Optional[float] = None
    max_lifetime: float = 86400.0  # 24 hours default like Modal


@dataclass
class SandboxInfo:
    """Information about a running sandbox.

    Inspired by E2B's getInfo() and Modal's sandbox info.

    Attributes:
        id: Unique identifier for the sandbox.
        variant: The sandbox variant (local-eval, local-docker, datalayer-runtime).
        status: Current status of the sandbox.
        created_at: Unix timestamp when the sandbox was created.
        end_at: Unix timestamp when the sandbox will be terminated.
        config: The configuration used to create this sandbox.
        name: Name of the sandbox if set.
        metadata: Additional metadata about the sandbox.
        resources: Resource configuration for the sandbox.
    """

    id: str
    variant: str
    status: SandboxStatus = SandboxStatus.RUNNING
    created_at: float = 0.0
    end_at: Optional[float] = None
    config: Optional[SandboxConfig] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
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


@dataclass
class SnapshotInfo:
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

    id: str
    name: str
    sandbox_id: str
    created_at: float = 0.0
    size: int = 0
    description: str = ""

    def __repr__(self) -> str:
        return f"SnapshotInfo(id={self.id!r}, name={self.name!r})"


@dataclass
class TunnelInfo:
    """Information about a tunnel to a sandbox port.

    Similar to Modal's Tunnel interface.

    Attributes:
        port: The port in the sandbox.
        url: The external URL to access the port.
        protocol: The protocol (http, https, tcp).
    """

    port: int
    url: str
    protocol: str = "https"

    def __repr__(self) -> str:
        return f"TunnelInfo(port={self.port}, url={self.url!r})"
