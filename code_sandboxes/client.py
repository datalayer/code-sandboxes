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

from dataclasses import dataclass, field

from .base import Sandbox
from .commands import CommandResult
from .models import ExecutionResult, SandboxConfig, SandboxVariant

__all__ = ["CodeExecutionOutcome", "CodeSandboxClient"]


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
