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
    - JupyterSandbox: Jupyter Server with persistent kernel state
    - DatalayerSandbox: Cloud-based Datalayer runtime, full isolation
    - ColabSandbox: Google Colab runtime, connects to an assigned kernel
    - KaggleSandbox: Kaggle runtime, connects to an interactive notebook kernel

Cloud container sandboxes:
    - ModalSandbox: Modal cloud containers, per-snippet process execution

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
from .client import CodeExecutionOutcome, CodeSandboxClient
from .colab_sandbox import ColabSandbox
from .commands import CommandResult, ProcessHandle, SandboxCommands
from .datalayer_sandbox import DatalayerSandbox
from .docker_sandbox import DockerSandbox
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
from .interfaces import IKernelClient, ISandboxClient
from .jupyter_sandbox import JupyterSandbox
from .kaggle_sandbox import KaggleSandbox
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
)
from .monty_sandbox import MontySandbox

__all__ = [
    # Models
    "CodeError",
    "CodeExecutionOutcome",
    "CodeSandboxClient",
    "ColabSandbox",
    "CommandResult",
    "Context",
    "ContextNotFoundError",
    "DatalayerSandbox",
    "DockerSandbox",
    # Sandbox implementations
    "EvalSandbox",
    "ExecutionResult",
    "FileInfo",
    "FileType",
    "FileWatchEvent",
    "FileWatchEventType",
    "GPUType",
    "JupyterSandbox",
    "IKernelClient",
    "ISandboxClient",
    "KaggleSandbox",
    "Logs",
    "MIMEType",
    "ModalSandbox",
    "MontySandbox",
    "OutputHandler",
    "OutputMessage",
    "ProcessHandle",
    "ResourceConfig",
    "Result",
    # Main sandbox class
    "Sandbox",
    "SandboxAuthenticationError",
    # Commands
    "SandboxCommands",
    "SandboxConfig",
    "SandboxConfigurationError",
    "SandboxConnectionError",
    "SandboxEnvironment",
    # Exceptions
    "SandboxError",
    "SandboxExecutionError",
    "SandboxFileHandle",
    # Filesystem
    "SandboxFilesystem",
    "SandboxInfo",
    "SandboxNotStartedError",
    "SandboxQuotaExceededError",
    "SandboxResourceError",
    "SandboxSnapshotError",
    "SandboxStatus",
    "SandboxTimeoutError",
    "SandboxVariant",
    "SandboxVariant",
    "SnapshotInfo",
    "TunnelInfo",
    "VariableNotFoundError",
]
