# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Code Sandboxes - Safe, isolated environments for AI code execution.

This package provides different sandbox implementations for executing
code safely.

sandboxes (in-process execution):
    - EvalSandbox: Simple Python exec() based, for development/testing
    - MontySandbox: Minimal secure Python interpreter (pydantic-monty)

Remote sandboxes (out-of-process execution via Jupyter kernel protocol):
    - DockerSandbox: Docker container based, good isolation
    - JupyterServerSandbox: Jupyter Server with persistent kernel state
    - DatalayerSandbox: Cloud-based Datalayer runtime, full isolation
    - GoogleColabSandbox: Google Colab runtime, connects to an assigned kernel
    - KaggleSandbox: Kaggle runtime, connects to an interactive notebook kernel

Cloud container sandboxes:
    - ModalSandbox: Modal cloud containers, per-snippet process execution
    - DaytonaSandbox: Daytona cloud sandboxes, stateful Python interpreter
    - E2BSandbox: E2B microVMs, stateful Python kernel and rich outputs
    - CoreWeaveSandbox: CoreWeave containers, stateful Python session
    - CloudflareSandbox: Cloudflare containers, through a sandbox bridge Worker

Features:
- Code execution with streaming support
- Filesystem operations (read, write, list, upload, download)
- Command execution (run, exec, spawn)
- Context management for state persistence
- Snapshot support (for datalayer)
- GPU and resource configuration

Example:
    from code_sandboxes import Sandbox

    # Create an eval sandbox
    with Sandbox.create(variant="eval") as sandbox:
        # Execute code
        result = sandbox.run_code("x = 1 + 1")
        result = sandbox.run_code("print(x)")  # prints 2

        # Filesystem operations
        sandbox.files.write("/data/test.txt", "Hello World")
        content = sandbox.files.read("/data/test.txt")

        # Command execution
        result = sandbox.commands.run("ls -la")

Style usage:
    sandbox = Sandbox.create(timeout=60)  # 60 second timeout
    result = sandbox.run_code('print("hello")')
    files = sandbox.files.list("/")

Style usage:
    sandbox = Sandbox.create(gpu="T4", environment="python-gpu-env")
    process = sandbox.commands.exec("python", "-c", "print('hello')")
    for line in process.stdout:
        print(line)
"""

from .base import Sandbox
from .builds import (
    ENVIRONMENT_CONTENTS_MANIFEST,
    BuildEntry,
    BuiltArtifact,
    EnvironmentBuild,
    build_artifact,
    dockerfile_fragment,
    installed_environment_contents,
)
from .client import CodeExecutionOutcome, CodeSandboxClient, execution_result_to_reply
from .cloudflare_sandbox import CloudflareSandbox
from .commands import CommandResult, ProcessHandle, SandboxCommands
from .console import (
    EXIT_COMMANDS,
    example_code,
    repl_prompt,
    run_repl,
    show_and_run,
    show_code,
    show_examples,
    show_result,
)
from .contents import (
    ContentAttachmentError,
    ContentAttachmentSpec,
    ContentCapabilities,
    ContentManifest,
    LocalBridgeCapability,
    ManifestLocation,
    MaterializeEntry,
    PreparedAttachment,
)
from .coreweave_sandbox import CoreWeaveSandbox
from .datalayer_sandbox import DatalayerSandbox
from .daytona_sandbox import DaytonaSandbox
from .docker_sandbox import DockerSandbox
from .e2b_sandbox import E2BSandbox
from .eval_sandbox import EvalSandbox
from .exceptions import (
    ContextNotFoundError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxError,
    SandboxExecutionError,
    SandboxNotStartedError,
    SandboxQuotaExceededError,
    SandboxResourceError,
    SandboxNotFoundError,
    SandboxSnapshotError,
    SandboxTimeoutError,
    VariableNotFoundError,
)
from .filesystem import (
    FileInfo,
    FileType,
    FileWatchEvent,
    FileWatchEventType,
    SandboxFileHandle,
    SandboxFilesystem,
)
from .google_colab import (
    GoogleColabKernelClient,
    parse_google_colab_channels_url,
)
from .google_colab_sandbox import GoogleColabSandbox
from .interfaces import ISandboxClient
from .jupyter_server_sandbox import JupyterServerSandbox
from .kaggle import KAGGLE_API_TOKEN_ENV, KaggleKernelClient, parse_kaggle_channels_url
from .kaggle_execute import KaggleExecutionResult, KaggleKernelExecutor
from .kaggle_sandbox import KaggleSandbox
from .lifecycle import (
    LIFECYCLE_OPERATIONS,
    RUNTIMES_API_PREFIX,
    MANAGER_OPERATIONS,
    INSTANCE_OPERATIONS,
    SandboxLifecycle,
    SandboxManagerLifecycle,
    SandboxOperationNotSupported,
    runtime_checkpoints_url,
    runtime_pause_url,
    runtime_resume_url,
    runtime_url,
    runtimes_url,
    sandbox_snapshot_url,
    sandbox_snapshots_url,
    unsupported,
)
from .manage import (
    SandboxManagementError,
    SandboxManager,
    get_manager,
    manageable_variants,
)
from .modal_sandbox import ModalSandbox
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    GPUType,
    JupyterServerEndpoint,
    JupyterServerOptions,
    Logs,
    MIMEType,
    OutputHandler,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
    SandboxVariant,
    SnapshotInfo,
    TunnelInfo,
    normalize_variant,
)
from .monty_sandbox import MontySandbox
from .provider_ingress import provider_ingress_execution
from .providers import (
    PROVIDERS,
    ProviderRequirement,
    SandboxProvider,
    available_providers,
    get_provider,
)

#: Everything this package exports, in one sorted list — the groups it
#: used to be split into stopped matching what they sat above.
__all__ = [
    "BuildEntry",
    "BuiltArtifact",
    "CloudflareSandbox",
    "CodeError",
    "CodeExecutionOutcome",
    "CodeSandboxClient",
    "CommandResult",
    "ContentAttachmentError",
    "ContentAttachmentSpec",
    "ContentCapabilities",
    "ContentManifest",
    "Context",
    "ContextNotFoundError",
    "CoreWeaveSandbox",
    "DatalayerSandbox",
    "DaytonaSandbox",
    "DockerSandbox",
    "E2BSandbox",
    "ENVIRONMENT_CONTENTS_MANIFEST",
    "EXIT_COMMANDS",
    "EnvironmentBuild",
    "EvalSandbox",
    "ExecutionResult",
    "FileInfo",
    "FileType",
    "FileWatchEvent",
    "FileWatchEventType",
    "GPUType",
    "GoogleColabKernelClient",
    "GoogleColabSandbox",
    "INSTANCE_OPERATIONS",
    "ISandboxClient",
    "JupyterServerEndpoint",
    "JupyterServerOptions",
    "JupyterServerSandbox",
    "KAGGLE_API_TOKEN_ENV",
    "KaggleExecutionResult",
    "KaggleKernelClient",
    "KaggleKernelExecutor",
    "KaggleSandbox",
    "LIFECYCLE_OPERATIONS",
    "LocalBridgeCapability",
    "Logs",
    "MANAGER_OPERATIONS",
    "MIMEType",
    "ManifestLocation",
    "MaterializeEntry",
    "ModalSandbox",
    "MontySandbox",
    "OutputHandler",
    "OutputMessage",
    "PROVIDERS",
    "PreparedAttachment",
    "ProcessHandle",
    "ProviderRequirement",
    "RUNTIMES_API_PREFIX",
    "ResourceConfig",
    "Result",
    "Sandbox",
    "SandboxAuthenticationError",
    "SandboxCommands",
    "SandboxConfig",
    "SandboxConfigurationError",
    "SandboxConnectionError",
    "SandboxEnvironment",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxFileHandle",
    "SandboxFilesystem",
    "SandboxInfo",
    "SandboxLifecycle",
    "SandboxManagementError",
    "SandboxManager",
    "SandboxManagerLifecycle",
    "SandboxNotStartedError",
    "SandboxOperationNotSupported",
    "SandboxProvider",
    "SandboxQuotaExceededError",
    "SandboxResourceError",
    "SandboxNotFoundError",
    "SandboxSnapshotError",
    "SandboxStatus",
    "SandboxTimeoutError",
    "SandboxVariant",
    "SnapshotInfo",
    "TunnelInfo",
    "VariableNotFoundError",
    "available_providers",
    "build_artifact",
    "dockerfile_fragment",
    "example_code",
    "execution_result_to_reply",
    "get_manager",
    "get_provider",
    "installed_environment_contents",
    "manageable_variants",
    "normalize_variant",
    "parse_google_colab_channels_url",
    "parse_kaggle_channels_url",
    "provider_ingress_execution",
    "repl_prompt",
    "run_repl",
    "runtime_checkpoints_url",
    "runtime_pause_url",
    "runtime_resume_url",
    "runtime_url",
    "runtimes_url",
    "sandbox_snapshot_url",
    "sandbox_snapshots_url",
    "show_and_run",
    "show_code",
    "show_examples",
    "show_result",
    "unsupported",
]
