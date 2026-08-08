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
    with CodeSandboxClient.create(variant="jupyter", jupyter_url=url) as client:
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
        """The variant of the wrapped sandbox, if known."""
        return getattr(self._sandbox.config, "variant", None)

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
        """Execute code and optionally emit each normalized output to a callback."""
        reply = self.execute(code, silent=silent, timeout=timeout, **kwargs)
        if output_hook is not None:
            for output in reply["outputs"]:
                output_hook(
                    {
                        "msg_type": output.get("output_type", "display_data"),
                        "content": output,
                    }
                )
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
