# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Code Sandboxes - Safe, isolated environments for AI code execution.

This package provides different sandbox implementations for executing
code safely, inspired by E2B and Modal:

Local sandboxes (in-process execution):
    - LocalEvalSandbox: Simple Python exec() based, for development/testing

Remote sandboxes (out-of-process execution via Jupyter kernel protocol):
    - LocalDockerSandbox: Docker container based, good isolation
    - LocalJupyterSandbox: Jupyter Server with persistent kernel state
    - DatalayerSandbox: Cloud-based Datalayer runtime, full isolation

Features:
- Code execution with streaming support
- Filesystem operations (read, write, list, upload, download)
- Command execution (run, exec, spawn)
- Context management for state persistence
- Snapshot support (for datalayer-runtime)
- GPU and resource configuration

Example:
    from code_sandboxes import Sandbox

    # Create a sandbox (defaults to datalayer-runtime)
    with Sandbox.create(variant="local-eval") as sandbox:
        # Execute code
        result = sandbox.run_code("x = 1 + 1")
        result = sandbox.run_code("print(x)")  # prints 2

        # Filesystem operations
        sandbox.files.write("/data/test.txt", "Hello World")
        content = sandbox.files.read("/data/test.txt")

        # Command execution
        result = sandbox.commands.run("ls -la")

E2B-style usage:
    sandbox = Sandbox.create(timeout=60)  # 60 second timeout
    result = sandbox.run_code('print("hello")')
    files = sandbox.files.list("/")

Modal-style usage:
    sandbox = Sandbox.create(gpu="T4", environment="python-gpu-env")
    process = sandbox.commands.exec("python", "-c", "print('hello')")
    for line in process.stdout:
        print(line)
"""

from .base import Sandbox, SandboxVariant
from .commands import CommandResult, ProcessHandle, SandboxCommands
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
from .local.eval_sandbox import LocalEvalSandbox
from .remote.docker_sandbox import LocalDockerSandbox
from .remote.jupyter_sandbox import LocalJupyterSandbox
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
from .remote.datalayer_sandbox import DatalayerSandbox

__all__ = [
    # Main sandbox class
    "Sandbox",
    "SandboxVariant",
    # Sandbox implementations
    "LocalEvalSandbox",
    "LocalDockerSandbox",
    "LocalJupyterSandbox",
    "DatalayerSandbox",
    # Filesystem
    "SandboxFilesystem",
    "SandboxFileHandle",
    "FileInfo",
    "FileType",
    "FileWatchEvent",
    "FileWatchEventType",
    # Commands
    "SandboxCommands",
    "CommandResult",
    "ProcessHandle",
    # Models
    "CodeError",
    "Context",
    "ExecutionResult",
    "Logs",
    "MIMEType",
    "OutputHandler",
    "OutputMessage",
    "Result",
    "SandboxConfig",
    "SandboxEnvironment",
    "SandboxInfo",
    "SandboxStatus",
    "SandboxVariant",
    "ResourceConfig",
    "GPUType",
    "SnapshotInfo",
    "TunnelInfo",
    # Exceptions
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxExecutionError",
    "SandboxNotStartedError",
    "SandboxConnectionError",
    "SandboxConfigurationError",
    "SandboxSnapshotError",
    "SandboxResourceError",
    "SandboxAuthenticationError",
    "SandboxQuotaExceededError",
    "ContextNotFoundError",
    "VariableNotFoundError",
]
