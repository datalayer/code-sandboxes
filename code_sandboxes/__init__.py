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
    - MarimoSandbox: a Jupyter Server kernel holding Marimo's reactive cell graph
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
from .exceptions import (
    ContextNotFoundError,
    SandboxAuthenticationError,
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxError,
    SandboxExecutionError,
    SandboxNotFoundError,
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
from .interfaces import ISandboxClient
from .lifecycle import (
    INSTANCE_OPERATIONS,
    LIFECYCLE_OPERATIONS,
    MANAGER_OPERATIONS,
    RUNTIMES_API_PREFIX,
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
    Reaction,
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
from .provider_ingress import provider_ingress_execution
from .providers import (
    PROVIDERS,
    ProviderRequirement,
    SandboxProvider,
    available_providers,
    get_provider,
)
from .sandboxes.cloudflare import CloudflareSandbox
from .sandboxes.coreweave import CoreWeaveSandbox
from .sandboxes.datalayer import DatalayerSandbox
from .sandboxes.daytona import DaytonaSandbox
from .sandboxes.docker import DockerSandbox
from .sandboxes.e2b import E2BSandbox
from .sandboxes.eval import EvalSandbox
from .sandboxes.google_colab import GoogleColabSandbox
from .sandboxes.google_colab.client import (
    GoogleColabKernelClient,
    parse_google_colab_channels_url,
)
from .sandboxes.jupyter_server import JupyterServerSandbox
from .sandboxes.kaggle import KaggleSandbox
from .sandboxes.kaggle.client import (
    KAGGLE_API_TOKEN_ENV,
    KaggleKernelClient,
    parse_kaggle_channels_url,
)
from .sandboxes.kaggle.execute import KaggleExecutionResult, KaggleKernelExecutor
from .sandboxes.marimo import CellRun, MarimoRun, MarimoSandbox
from .sandboxes.marimo.cells import CellReply, CellsRun, MarimoCells
from .sandboxes.modal import ModalSandbox
from .sandboxes.monty import MontySandbox

#: Everything this package exports, in one sorted list — the groups it
#: used to be split into stopped matching what they sat above.
__all__ = [
    "ENVIRONMENT_CONTENTS_MANIFEST",
    "EXIT_COMMANDS",
    "INSTANCE_OPERATIONS",
    "KAGGLE_API_TOKEN_ENV",
    "LIFECYCLE_OPERATIONS",
    "MANAGER_OPERATIONS",
    "PROVIDERS",
    "RUNTIMES_API_PREFIX",
    "BuildEntry",
    "BuiltArtifact",
    "CellReply",
    "CellRun",
    "CellsRun",
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
    "ISandboxClient",
    "JupyterServerEndpoint",
    "JupyterServerOptions",
    "JupyterServerSandbox",
    "KaggleExecutionResult",
    "KaggleKernelClient",
    "KaggleKernelExecutor",
    "KaggleSandbox",
    "LocalBridgeCapability",
    "Logs",
    "MIMEType",
    "ManifestLocation",
    "MarimoCells",
    "MarimoRun",
    "MarimoSandbox",
    "MaterializeEntry",
    "ModalSandbox",
    "MontySandbox",
    "OutputHandler",
    "OutputMessage",
    "PreparedAttachment",
    "ProcessHandle",
    "ProviderRequirement",
    "Reaction",
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
    "SandboxNotFoundError",
    "SandboxNotStartedError",
    "SandboxOperationNotSupported",
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


# -- The flat module paths of 1.9.x, kept as aliases -------------------------
#
# 1.10.0 moved every provider into `code_sandboxes.sandboxes.<provider>`.
# Released consumers import the old paths — `code_sandboxes.datalayer_sandbox`
# in jupyter-mcp-sandboxes 0.2.6, which the MCP gateway image installs from
# PyPI — and a fresh install would break them until each is re-released. So
# each old dotted path names the *same module object* as its new home:
# `import code_sandboxes.datalayer_sandbox`, `from code_sandboxes.kaggle
# import …` and `patch("code_sandboxes.datalayer_sandbox.X")` keep working and
# act on the real module. Deprecated: new code imports from `code_sandboxes`
# or the new paths, and the aliases go in 2.0.
#: The 1.9.x flat path of each moved module, and its home since 1.10.0.
_FLAT_PATHS: dict[str, str] = {
    "cloudflare_sandbox": "sandboxes.cloudflare.cloudflare",
    "coreweave_sandbox": "sandboxes.coreweave.coreweave",
    "datalayer_sandbox": "sandboxes.datalayer.datalayer",
    "daytona_sandbox": "sandboxes.daytona.daytona",
    "docker_sandbox": "sandboxes.docker.docker",
    "e2b_sandbox": "sandboxes.e2b.e2b",
    "eval_sandbox": "sandboxes.eval.eval",
    "google_colab": "sandboxes.google_colab.client",
    "google_colab_sandbox": "sandboxes.google_colab.google_colab",
    "jupyter_server_sandbox": "sandboxes.jupyter_server.jupyter_server",
    "kaggle": "sandboxes.kaggle.client",
    "kaggle_execute": "sandboxes.kaggle.execute",
    "kaggle_live": "sandboxes.kaggle.live",
    "kaggle_sandbox": "sandboxes.kaggle.kaggle",
    "marimo_cells": "sandboxes.marimo.cells",
    "marimo_reactive": "sandboxes.marimo.reactive",
    "marimo_sandbox": "sandboxes.marimo.marimo",
    "modal_sandbox": "sandboxes.modal.modal",
    "monty_sandbox": "sandboxes.monty.monty",
}


def _alias_the_flat_paths() -> None:
    import importlib
    import sys

    for old, new in _FLAT_PATHS.items():
        module = importlib.import_module(f"{__name__}.{new}")
        sys.modules.setdefault(f"{__name__}.{old}", module)
        globals().setdefault(old, module)


_alias_the_flat_paths()
del _alias_the_flat_paths
