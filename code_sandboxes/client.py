# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""High-level, variant-agnostic client for executing code in a sandbox.

The :class:`CodeSandboxClient` wraps any concrete :class:`~code_sandboxes.base.Sandbox`
implementation (eval, docker, jupyter, datalayer) behind a small, ergonomic
API that returns a normalized :class:`CodeExecutionOutcome`. It is meant to be
the single entry point consumers should reach for when all they need is
"run this code / command and give me the stdout, stderr and success flag",
without caring about the underlying sandbox variant.

Example:
    from code_sandboxes import CodeSandboxClient

    # Create + own the sandbox lifecycle.
    with CodeSandboxClient.create(variant="jupyter-server", jupyter_url=url) as client:
        outcome = client.execute_code("x = 1")
        outcome = client.execute_code("print(x)")
        print(outcome.stdout)  # "1"

    # Or wrap an already-running sandbox managed elsewhere.
    client = CodeSandboxClient(existing_sandbox)
    outcome = await client.execute_code_async("print('hi')")

Kubernetes / colocated-sidecar contract:
    The client is deliberately variant-agnostic and performs **no** fallback of
    its own. It never picks a variant and never silently spins up an ``eval``
    sandbox. When it wraps a shared/managed sandbox (e.g. agent-runtimes'
    ``ManagedSandbox`` proxy over a per-pod Jupyter sidecar), every call is
    delegated to that sandbox, so:

    * code and skill execution reuse the *existing* colocated Jupyter kernel
      (state persists across executions), and
    * a sidecar that is configured but not yet reachable fails fast — the
      wrapped sandbox raises rather than degrading to an in-process ``eval``.

    Owning the sandbox lifecycle (``owns_sandbox`` / :meth:`create`) is meant
    for local/standalone callers; in the pod the sandbox is owned by the
    manager and the client is created with ``owns_sandbox=False``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Union

from .base import Sandbox
from .commands import CommandResult
from .contents import (
    ContentAttachmentError,
    ContentCapabilities,
    ContentManifest,
    ManifestLocation,
    PreparedAttachment,
    install_manifest,
)
from .filesystem import FileInfo, SandboxFilesystem
from .models import (
    CodeError,
    ExecutionResult,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxInfo,
    SandboxVariant,
)

__all__ = ["CodeExecutionOutcome", "CodeSandboxClient", "execution_result_to_reply"]

StreamingItem = Union[OutputMessage, Result, CodeError]


def execution_result_to_reply(execution: ExecutionResult) -> dict[str, Any]:
    """Convert a variant-neutral execution result to a Jupyter-shaped reply."""
    outputs: list[dict[str, Any]] = []

    if execution.logs.stdout:
        outputs.append(
            {
                "output_type": "stream",
                "name": "stdout",
                "text": "\n".join(message.line for message in execution.logs.stdout) + "\n",
            }
        )
    if execution.logs.stderr:
        outputs.append(
            {
                "output_type": "stream",
                "name": "stderr",
                "text": "\n".join(message.line for message in execution.logs.stderr) + "\n",
            }
        )

    for result in execution.results:
        outputs.append(
            {
                "output_type": "execute_result" if result.is_main_result else "display_data",
                "data": result.data,
                "metadata": result.extra,
            }
        )

    if execution.code_error is not None:
        traceback = execution.code_error.traceback or ""
        outputs.append(
            {
                "output_type": "error",
                "ename": execution.code_error.name,
                "evalue": execution.code_error.value,
                "traceback": traceback.splitlines(),
            }
        )

    return {
        "execution_count": execution.execution_count,
        "outputs": outputs,
        "status": "ok" if execution.success else "error",
    }


@dataclass
class CodeExecutionOutcome:
    """Normalized result of a code execution, independent of sandbox variant.

    This is a faithful *superset* of :class:`ExecutionResult`: it distinguishes
    the same two failure levels (infrastructure vs. user-code) plus intentional
    ``sys.exit()`` codes, so it can drive both plain code execution (the TUX /
    ``/sandbox/execute`` endpoint) and skill-script execution
    (``agent_skills.SandboxExecutor``) without losing information.

    Attributes:
        success: True when the infrastructure ran the code and the code itself
            raised no error, was not interrupted, and exited cleanly.
        execution_ok: True when the sandbox infrastructure ran the code, even if
            the user's code raised an exception.
        stdout: Combined standard output text.
        stderr: Combined standard error text.
        results: Textual representation of rich results (display data / return
            values) produced by the execution.
        error: Human-readable error message when ``success`` is False, otherwise
            ``None``.
        execution_error: Infrastructure failure detail when ``execution_ok`` is
            False (connection loss, kernel timeout, sidecar unavailable, …).
        code_error: Structured user-code exception ``{name, value, traceback}``
            when the code ran but raised, otherwise ``None``.
        exit_code: Exit code when the code called ``sys.exit()``; ``None`` for a
            normal completion without an explicit exit.
        interrupted: Whether the execution was cancelled/interrupted.
    """

    success: bool
    execution_ok: bool
    stdout: str = ""
    stderr: str = ""
    results: list[str] = field(default_factory=list)
    error: str | None = None
    execution_error: str | None = None
    code_error: dict[str, str] | None = None
    exit_code: int | None = None
    interrupted: bool = False

    @classmethod
    def from_execution_result(cls, execution: ExecutionResult) -> CodeExecutionOutcome:
        """Build a normalized outcome from a raw :class:`ExecutionResult`."""
        results: list[str] = []
        for result in execution.results:
            text = getattr(result, "text", None)
            if text:
                results.append(text)

        code_error_dict: dict[str, str] | None = None
        if execution.code_error is not None:
            code_error = execution.code_error
            code_error_dict = {
                "name": getattr(code_error, "name", None) or "Error",
                "value": getattr(code_error, "value", None) or "",
                "traceback": getattr(code_error, "traceback", None) or "",
            }

        exit_code = getattr(execution, "exit_code", None)

        error: str | None = None
        if not execution.execution_ok:
            error = execution.execution_error or "Sandbox infrastructure failure"
        elif code_error_dict is not None:
            name = code_error_dict["name"]
            value = code_error_dict["value"]
            error = f"{name}: {value}".strip().rstrip(":")
        elif execution.interrupted:
            error = "Execution interrupted"
        elif exit_code is not None and exit_code != 0:
            error = f"Script exited with code {exit_code}"

        return cls(
            success=execution.success,
            execution_ok=execution.execution_ok,
            stdout=execution.logs.stdout_text,
            stderr=execution.logs.stderr_text,
            results=results,
            error=error,
            execution_error=execution.execution_error,
            code_error=code_error_dict,
            exit_code=exit_code,
            interrupted=execution.interrupted,
        )


class CodeSandboxClient:
    """Variant-agnostic facade over a :class:`Sandbox`.

    The client owns no variant-specific logic: it simply delegates to the
    wrapped sandbox and normalizes the result. Callers can either let the
    client create and manage the sandbox (:meth:`create`) or hand it an
    existing sandbox instance that is managed elsewhere.
    """

    def __init__(self, sandbox: Sandbox, *, owns_sandbox: bool = False) -> None:
        """Wrap an existing sandbox.

        Args:
            sandbox: The concrete sandbox to delegate execution to.
            owns_sandbox: When True the client will stop the sandbox on
                :meth:`close` / context-manager exit. Defaults to False so that
                wrapping a shared/managed sandbox never shuts it down.
        """
        self._sandbox = sandbox
        self._owns_sandbox = owns_sandbox
        self._contents_location: ManifestLocation | None = None

    @classmethod
    def create(
        cls,
        variant: SandboxVariant | str = SandboxVariant.EVAL,
        config: SandboxConfig | None = None,
        **kwargs,
    ) -> CodeSandboxClient:
        """Create a client that owns a freshly created sandbox of ``variant``.

        Accepts the same keyword arguments as :meth:`Sandbox.create`.
        """
        sandbox = Sandbox.create(variant=variant, config=config, **kwargs)
        return cls(sandbox, owns_sandbox=True)

    @property
    def sandbox(self) -> Sandbox:
        """The wrapped sandbox instance."""
        return self._sandbox

    @property
    def config(self) -> SandboxConfig:
        """Variant-neutral configuration for the wrapped sandbox."""
        return self._sandbox.config

    @property
    def info(self) -> SandboxInfo | None:
        """Runtime information for the wrapped sandbox, when started."""
        return self._sandbox.info

    @property
    def id(self) -> str | None:
        """Stable execution-backend identifier when the variant exposes one."""
        info = getattr(self._sandbox, "info", None)
        metadata = getattr(info, "metadata", None) or {}
        kernel_id = metadata.get("kernel_id")
        if kernel_id:
            return str(kernel_id)
        backend = getattr(self._sandbox, "kernel_client", None)
        backend_id = getattr(backend, "id", None)
        return str(backend_id) if backend_id else self._sandbox.sandbox_id

    @property
    def kernel_info(self) -> dict[str, Any]:
        """Language metadata without exposing a variant's underlying client."""
        backend = getattr(self._sandbox, "kernel_client", None)
        info = getattr(backend, "kernel_info", None)
        if isinstance(info, dict):
            return info
        environments = self._sandbox.list_environments()
        language = environments[0].language if environments else "python"
        return {"language_info": {"name": language}}

    @property
    def variant(self) -> SandboxVariant | None:
        """The variant of the wrapped sandbox, if known.

        ``SandboxConfig`` carries no ``variant`` field, so real sandboxes only
        report their variant through ``info``, which is populated on start. The
        config is still consulted first for duck-typed sandboxes that expose it.
        """
        variant = getattr(self._sandbox.config, "variant", None)
        if variant is None:
            info = getattr(self._sandbox, "info", None)
            variant = getattr(info, "variant", None)
        return variant

    @property
    def is_started(self) -> bool:
        """Whether the wrapped sandbox has been started.

        Sandboxes that do not expose ``is_started`` (e.g. minimal duck-typed
        objects that only implement ``run_code``) are treated as ready.
        """
        return bool(getattr(self._sandbox, "is_started", True))

    def start(self) -> None:
        """Start the wrapped sandbox if it exposes a lifecycle and is not started."""
        start_fn = getattr(self._sandbox, "start", None)
        if callable(start_fn) and not self.is_started:
            start_fn()

    async def start_async(self) -> None:
        """Async variant of :meth:`start`."""
        if self.is_started:
            return
        start_async_fn = getattr(self._sandbox, "start_async", None)
        if callable(start_async_fn):
            await start_async_fn()
        else:
            self.start()

    def close(self) -> None:
        """Stop the wrapped sandbox when this client owns it."""
        stop_fn = getattr(self._sandbox, "stop", None)
        if self._owns_sandbox and callable(stop_fn) and self.is_started:
            stop_fn()

    def stop(self, shutdown_kernel: bool = True) -> None:
        """Release the client, optionally preserving a borrowed remote backend."""
        if shutdown_kernel:
            self.close()
            return
        backend = getattr(self._sandbox, "kernel_client", None)
        backend_stop = getattr(backend, "stop", None)
        if callable(backend_stop):
            backend_stop(shutdown_kernel=False)
        # The backend connection is now closed, so the sandbox must no longer
        # report itself as started; otherwise start() would no-op and later
        # executions would run against a closed backend.
        mark_stopped = getattr(self._sandbox, "mark_stopped", None)
        if callable(mark_stopped):
            mark_stopped()

    async def close_async(self) -> None:
        """Async variant of :meth:`close`."""
        if not (self._owns_sandbox and self.is_started):
            return
        stop_async_fn = getattr(self._sandbox, "stop_async", None)
        if callable(stop_async_fn):
            await stop_async_fn()
        else:
            self.close()

    def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionOutcome:
        """Execute code and return a normalized outcome.

        The sandbox is started automatically if needed.
        """
        self.start()
        execution = self._sandbox.run_code(code, language=language, timeout=timeout, envs=envs)
        return CodeExecutionOutcome.from_execution_result(execution)

    def execute(
        self,
        code: str,
        silent: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute code and return a backend-neutral Jupyter-shaped reply."""
        del silent, kwargs
        self.start()
        execution = self._sandbox.run_code(code, timeout=timeout)
        return execution_result_to_reply(execution)

    def execute_interactive(
        self,
        code: str,
        silent: bool = False,
        timeout: float | None = None,
        output_hook: Callable[[dict[str, Any]], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute code and optionally emit each output to a callback.

        Mirrors :meth:`jupyter_kernel_client.JupyterKernelClient.execute_interactive`
        so kernel-client consumers work without an adapter:

        * each output-hook message carries a Jupyter ``header`` with ``msg_type``
          (read as ``message["header"]["msg_type"]``), and
        * the reply nests ``status`` / ``execution_count`` under ``content``
          (read as ``reply["content"]["status"]``).

        The flat top-level ``status`` / ``execution_count`` / ``outputs`` keys are
        preserved for callers that consume the backend-neutral reply shape, so the
        emitted messages and reply are a superset satisfying both contracts.
        """
        reply = self.execute(code, silent=silent, timeout=timeout, **kwargs)
        if output_hook is not None:
            for output in reply["outputs"]:
                msg_type = output.get("output_type", "display_data")
                output_hook(
                    {
                        "header": {"msg_type": msg_type},
                        "msg_type": msg_type,
                        "content": output,
                    }
                )
        reply["content"] = {
            "status": reply.get("status", "ok"),
            "execution_count": reply.get("execution_count"),
        }
        return reply

    def get_variable(self, name: str) -> Any:
        """Read a variable through the wrapped sandbox."""
        self.start()
        return self._sandbox.get_variable(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable through the wrapped sandbox."""
        self.start()
        self._sandbox.set_variable(name, value)

    def set_variables(self, variables: dict[str, Any]) -> None:
        """Set multiple variables through the wrapped sandbox."""
        self.start()
        self._sandbox.set_variables(variables)

    def register_tool_caller(self, caller: Callable[..., Any]) -> None:
        """Register the callable used by generated tools inside the sandbox."""
        self.start()
        self._sandbox.register_tool_caller(caller)

    def interrupt(self) -> bool:
        """Interrupt the active execution when supported by the variant."""
        return self._sandbox.interrupt()

    def is_alive(self) -> bool:
        """Whether the sandbox is started and available for execution."""
        return self.is_started

    def restart(self) -> None:
        """Restart the wrapped sandbox through its public lifecycle."""
        self._sandbox.stop()
        self._sandbox.start()

    async def execute_code_async(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionOutcome:
        """Async variant of :meth:`execute_code`."""
        await self.start_async()
        execution = await self._sandbox.run_code_async(
            code, language=language, timeout=timeout, envs=envs
        )
        return CodeExecutionOutcome.from_execution_result(execution)

    def execute_code_streaming(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        envs: dict[str, str] | None = None,
    ) -> Iterator[StreamingItem]:
        """Execute code and stream output events.

        This is a thin variant-agnostic wrapper over
        ``Sandbox.run_code_streaming``.
        """
        self.start()
        yield from self._sandbox.run_code_streaming(
            code,
            language=language,
            timeout=timeout,
            envs=envs,
        )

    async def execute_code_streaming_async(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
        envs: dict[str, str] | None = None,
    ) -> AsyncIterator[StreamingItem]:
        """Async variant of :meth:`execute_code_streaming`."""
        await self.start_async()
        async for item in self._sandbox.run_code_streaming_async(
            code,
            language=language,
            timeout=timeout,
            envs=envs,
        ):
            yield item

    def run_command(self, command: str, timeout: float | None = None) -> CommandResult:
        """Run a shell command inside the sandbox."""
        self.start()
        return self._sandbox.commands.run(command, timeout=timeout)

    # -- the filesystem ----------------------------------------------------
    #
    # A file browser and a terminal look at the same tree, and both of them
    # reach it through this client rather than through a provider's SDK: that
    # is what makes a workflow written for one sandbox work on the next. Each
    # of these starts the sandbox first, because asking about files before
    # there is a sandbox to ask is a mistake worth answering rather than
    # crashing on.

    @property
    def files(self) -> SandboxFilesystem:
        """Filesystem operations, whichever provider is underneath."""
        self.start()
        return self._sandbox.files

    def list_files(self, path: str = "/") -> list[FileInfo]:
        """What is in a directory: names, sizes and what each one is."""
        return self.files.list(path)

    def stat_file(self, path: str) -> FileInfo:
        """One entry, or `FileNotFoundError` if the path names nothing."""
        return self.files.get_info(path)

    def read_file(self, path: str, *, binary: bool = False) -> str | bytes:
        """Read a whole file; `binary` for anything that is not text."""
        return self.files.read_bytes(path) if binary else self.files.read(path)

    def write_file(self, path: str, content: str | bytes, *, make_dirs: bool = True) -> None:
        """Write a whole file, creating the directories above it by default."""
        if isinstance(content, bytes):
            self.files.write_bytes(path, content, make_dirs=make_dirs)
        else:
            self.files.write(path, content, make_dirs=make_dirs)

    def stream_file(self, path: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        """Read a file in pieces, for something too big to hold at once.

        The pieces come from one read of the sandbox: the providers'
        filesystem APIs answer a whole file, and pretending otherwise would
        promise a memory bound this cannot keep. What it does keep is the
        shape a caller writes against, so a ranged read behind one provider
        does not change the code above it.
        """
        content = self.files.read_bytes(path)
        for start in range(0, len(content), max(1, chunk_size)):
            yield content[start : start + chunk_size]

    def delete_file(self, path: str, *, recursive: bool = False) -> None:
        """Remove a file, or a directory when `recursive` says so."""
        self.files.rm(path, recursive=recursive)

    def make_directory(self, path: str, *, parents: bool = True) -> None:
        """Create a directory, and the ones above it by default."""
        self.files.mkdir(path, parents=parents)

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Put a local file into the sandbox."""
        self.files.upload(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Take a file out of the sandbox."""
        self.files.download(remote_path, local_path)

    # -- Contents attachments ---------------------------------------------
    #
    # The Contents service hands over a manifest; the sandbox is given what
    # it names. The client's part is the order of things — configure before
    # the sandbox exists, start, install the manifest inside, then prepare —
    # and the one strict verb, `attach`, that refuses to pretend a required
    # attachment is there when it is not.

    @property
    def contents_location(self) -> ManifestLocation | None:
        """Where the manifest was last written inside the sandbox, if it was."""
        return self._contents_location

    def content_capabilities(self) -> ContentCapabilities:
        """What the wrapped sandbox's provider can do with an attachment."""
        return self._sandbox.content_capabilities()

    def prepare_contents(self, manifest: ContentManifest) -> list[PreparedAttachment]:
        """Give the sandbox what the manifest names, and say what became of it.

        A sandbox not yet started is configured first — the environment, and
        the mounts a provider only makes at creation — then started, then
        given the manifest file and the credentials to Contents, and only
        then are the attachments prepared. Nothing here raises for an
        attachment that could not be honoured; :meth:`attach` does.
        """
        self._stage_contents(manifest)
        return self._sandbox.prepare_contents(manifest)

    def reconcile_contents(self, manifest: ContentManifest) -> list[PreparedAttachment]:
        """Re-check every attachment and repair what is missing.

        After a restart, a reconnect, or a manifest that grew: the same as
        :meth:`prepare_contents`, minus the work already done.
        """
        self._stage_contents(manifest)
        return self._sandbox.reconcile_contents(manifest)

    def attach(self, manifest: ContentManifest) -> list[PreparedAttachment]:
        """Prepare the manifest and insist that every required attachment is ready.

        Raises:
            ContentAttachmentError: naming the first required attachment that
                is not ready and why, with the whole list on `attachments`.
                The sandbox is LEFT RUNNING — whether a sandbox without its
                data is still worth having is the caller's call.
        """
        prepared = self.prepare_contents(manifest)
        for result in prepared:
            spec = manifest.attachment(result.uid)
            if spec is not None and spec.required and result.status != "ready":
                raise ContentAttachmentError(
                    result.uid,
                    result.error_code or "ATTACHMENT_NOT_READY",
                    f"Content attachment {result.uid} is {result.status}"
                    + (f": {result.detail}" if result.detail else ""),
                    attachments=prepared,
                )
        return prepared

    def attachment_status(self, uid: str) -> PreparedAttachment | None:
        """What the last prepare or reconcile said of one attachment."""
        return self._sandbox.attachment_status(uid)

    def detach(self, uid: str) -> None:
        """Take away what one attachment delivered; the source is untouched."""
        self._sandbox.remove_attachment(uid)

    def _stage_contents(self, manifest: ContentManifest) -> None:
        """Configure, start, and put the manifest inside — in that order.

        Configured every time, running or not: the configuration is what the
        provider is handed at the NEXT creation — a restart — and it must say
        what this manifest says, not what an earlier one did.
        """
        self._sandbox.configure_contents(manifest)
        self.start()
        self._contents_location = install_manifest(self._sandbox, manifest)

    def __enter__(self) -> CodeSandboxClient:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    async def __aenter__(self) -> CodeSandboxClient:
        await self.start_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_async()
