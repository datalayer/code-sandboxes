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
from .client import CodeExecutionOutcome, CodeSandboxClient, execution_result_to_reply
from .cloudflare_sandbox import CloudflareSandbox
from .commands import CommandResult, ProcessHandle, SandboxCommands
from .console import (
    EXIT_COMMANDS,
    repl_prompt,
    run_repl,
    show_and_run,
    show_code,
    show_result,
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
    "EXIT_COMMANDS",
    "KAGGLE_API_TOKEN_ENV",
    "PROVIDERS",
    "CloudflareSandbox",
    "CodeError",
    "CodeExecutionOutcome",
    "CodeSandboxClient",
    "CommandResult",
    "Context",
    "ContextNotFoundError",
    "CoreWeaveSandbox",
    "DatalayerSandbox",
    "DaytonaSandbox",
    "DockerSandbox",
    "E2BSandbox",
    "EvalSandbox",
    "ExecutionResult",
    "FileInfo",
    "FileType",
    "FileWatchEvent",
    "FileWatchEventType",
    "GPUType",
    "GoogleColabKernelClient",
    "GoogleColabSandbox",
    "ISandboxClient",
    "JupyterServerSandbox",
    "KaggleExecutionResult",
    "KaggleKernelClient",
    "KaggleKernelExecutor",
    "KaggleSandbox",
    "Logs",
    "MIMEType",
    "ModalSandbox",
    "MontySandbox",
    "OutputHandler",
    "OutputMessage",
    "ProcessHandle",
    "ProviderRequirement",
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
    "SandboxManagementError",
    "SandboxManager",
    "SandboxNotStartedError",
    "SandboxProvider",
    "SandboxQuotaExceededError",
    "SandboxResourceError",
    "SandboxSnapshotError",
    "SandboxStatus",
    "SandboxTimeoutError",
    "SandboxVariant",
    "SnapshotInfo",
    "TunnelInfo",
    "VariableNotFoundError",
    "available_providers",
    "execution_result_to_reply",
    "get_manager",
    "get_provider",
    "manageable_variants",
    "normalize_variant",
    "parse_google_colab_channels_url",
    "parse_kaggle_channels_url",
    "repl_prompt",
    "run_repl",
    "show_and_run",
    "show_code",
    "show_result",
]
