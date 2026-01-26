# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Local eval-based sandbox implementation.

This is a simple sandbox that uses Python's exec() for code execution.
It provides minimal isolation and is suitable for development and testing.

WARNING: This sandbox does NOT provide security isolation. Do not use
for executing untrusted code.
"""

import ast
import asyncio
import io
import socket
import threading
import time
import textwrap
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from contextlib import contextmanager
from typing import Any, Optional

from ..base import Sandbox
from ..exceptions import SandboxNotStartedError
from ..models import (
    CodeError,
    Context,
    ExecutionResult,
    Logs,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
)


class LocalEvalSandbox(Sandbox):
    """A simple sandbox using Python's exec() for code execution.

    This sandbox maintains separate namespaces for each context, allowing
    variable persistence between executions within the same context.

    WARNING: This provides NO security isolation. Only use for trusted code.

    Example:
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("x = 42")
            result = sandbox.run_code("print(x * 2)")  # prints 84
    """

    def __init__(self, config: Optional[SandboxConfig] = None, **kwargs):
        """Initialize the local eval sandbox.

        Args:
            config: Sandbox configuration.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(config)
        self._namespaces: dict[str, dict[str, Any]] = {}
        self._execution_count: dict[str, int] = {}
        self._sandbox_id = str(uuid.uuid4())

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="local-eval",
                title="Local Eval",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": "local-eval"},
            )
        ]

    def start(self) -> None:
        """Start the sandbox (initializes the default namespace)."""
        if self._started:
            return

        self._default_context = self.create_context("default")
        self._namespaces[self._default_context.id] = {"__builtins__": __builtins__}
        self._execution_count[self._default_context.id] = 0

        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="local-eval",
            status="running",
            created_at=time.time(),
            config=self.config,
        )
        self._started = True

    def stop(self) -> None:
        """Stop the sandbox (clears all namespaces)."""
        if not self._started:
            return

        self._namespaces.clear()
        self._execution_count.clear()
        self._started = False
        if self._info:
            self._info.status = "stopped"

    def create_context(self, name: Optional[str] = None) -> Context:
        """Create a new execution context with its own namespace.

        Args:
            name: Optional name for the context.

        Returns:
            A new Context object.
        """
        context = super().create_context(name)
        if context.id not in self._namespaces:
            self._namespaces[context.id] = {"__builtins__": __builtins__}
            self._execution_count[context.id] = 0
        return context

    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[CodeError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute Python code using exec().

        Args:
            code: The Python code to execute.
            language: Must be "python" for this sandbox.
            context: Execution context (uses default if not provided).
            on_stdout: Callback for stdout messages.
            on_stderr: Callback for stderr messages.
            on_result: Callback for results.
            on_error: Callback for errors.
            envs: Environment variables (applied to os.environ temporarily).
            timeout: Timeout in seconds (not enforced in this simple implementation).

        Returns:
            Execution result.

        Raises:
            SandboxNotStartedError: If the sandbox hasn't been started.
            ValueError: If language is not "python".
        """
        if not self._started:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"LocalEvalSandbox only supports Python, got: {language}")

        # Normalize indentation for multiline snippets
        code = textwrap.dedent(code)

        # Get or create context
        ctx = context or self._default_context
        if ctx.id not in self._namespaces:
            self._namespaces[ctx.id] = {"__builtins__": __builtins__}
            self._execution_count[ctx.id] = 0

        namespace = self._namespaces[ctx.id]
        self._execution_count[ctx.id] += 1
        execution_count = self._execution_count[ctx.id]

        started_at = time.time()

        # Set up environment variables temporarily
        old_env = {}
        if envs:
            import os

            for key, value in envs.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: Optional[CodeError] = None

        def _run_coroutine_sync(coro):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop and running_loop.is_running():
                result_container: dict[str, Any] = {}
                error_container: dict[str, BaseException] = {}

                def _runner():
                    try:
                        result_container["value"] = asyncio.run(coro)
                    except BaseException as exc:
                        error_container["error"] = exc

                thread = threading.Thread(target=_runner)
                thread.start()
                thread.join()
                if "error" in error_container:
                    raise error_container["error"]
                return result_container.get("value")

            return asyncio.run(coro)

        try:
            with self._network_guard(), redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                if "await " in code or "async " in code:
                    def _indent_code(value: str, spaces: int) -> str:
                        indent = " " * spaces
                        return "\n".join(indent + line for line in value.split("\n"))

                    async_wrapper = f"""
async def __user_code__():
{_indent_code(code, 4)}
    return locals()
"""
                    import sys
                    print(f"[SANDBOX] Executing async code, wrapping in __user_code__", file=sys.stderr, flush=True)
                    exec(async_wrapper, namespace, namespace)
                    result_value = namespace["__user_code__"]()
                    print(f"[SANDBOX] Calling _run_coroutine_sync...", file=sys.stderr, flush=True)
                    locals_value = _run_coroutine_sync(result_value)
                    print(f"[SANDBOX] _run_coroutine_sync returned", file=sys.stderr, flush=True)
                    if isinstance(locals_value, dict):
                        for key, value in locals_value.items():
                            # Skip Python internals but preserve user variables starting with __
                            if key in ("__builtins__", "__name__", "__doc__", "__package__", 
                                     "__loader__", "__spec__", "__annotations__", "__cached__",
                                     "__file__"):
                                continue
                            namespace[key] = value
                else:
                    # Try to evaluate as an expression first
                    try:
                        compiled = compile(code, "<sandbox>", "eval")
                        result_value = eval(compiled, namespace)
                        if asyncio.iscoroutine(result_value):
                            result_value = _run_coroutine_sync(result_value)
                        if result_value is not None:
                            result = Result(
                                data={"text/plain": repr(result_value)},
                                is_main_result=True,
                            )
                            results.append(result)
                            if on_result:
                                on_result(result)
                    except SyntaxError:
                        # Not a pure expression, execute as statements
                        # If the last statement is an expression, evaluate it and capture the result
                        try:
                            parsed = ast.parse(code, mode="exec")
                            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                                last_expr = parsed.body.pop()

                                if parsed.body:
                                    exec(compile(parsed, "<sandbox>", "exec"), namespace)

                                expr_code = ast.Expression(last_expr.value)
                                result_value = eval(compile(expr_code, "<sandbox>", "eval"), namespace)
                                if asyncio.iscoroutine(result_value):
                                    result_value = _run_coroutine_sync(result_value)
                                if result_value is not None:
                                    result = Result(
                                        data={"text/plain": repr(result_value)},
                                        is_main_result=True,
                                    )
                                    results.append(result)
                                    if on_result:
                                        on_result(result)
                            else:
                                exec(compile(parsed, "<sandbox>", "exec"), namespace)
                        except Exception:
                            exec(code, namespace)

        except Exception as e:
            # Capture the error
            tb = traceback.format_exc()
            code_error = CodeError(
                name=type(e).__name__,
                value=str(e),
                traceback=tb,
            )
            if on_error:
                on_error(code_error)

        finally:
            # Restore environment variables
            if envs:
                import os

                for key, old_value in old_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

        # Process stdout
        stdout_content = stdout_buffer.getvalue()
        if stdout_content:
            current_time = time.time()
            for line in stdout_content.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=False)
                stdout_messages.append(msg)
                if on_stdout:
                    on_stdout(msg)

        # Process stderr
        stderr_content = stderr_buffer.getvalue()
        if stderr_content:
            current_time = time.time()
            for line in stderr_content.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=True)
                stderr_messages.append(msg)
                if on_stderr:
                    on_stderr(msg)

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            execution_count=execution_count,
            context_id=ctx.id,
            started_at=started_at,
            completed_at=time.time(),
        )

    @contextmanager
    def _network_guard(self):
        """Apply the configured network policy for local eval."""
        policy = getattr(self.config, "network_policy", "inherit")
        if policy in ("inherit", "all"):
            yield
            return

        original_create_connection = socket.create_connection
        original_socket_connect = socket.socket.connect

        def _extract_host(address):
            if isinstance(address, tuple) and address:
                return address[0]
            if isinstance(address, str):
                return address
            return None

        def _is_allowed(host: str | None) -> bool:
            if host is None:
                return False
            allowed = getattr(self.config, "allowed_hosts", [])
            return host in allowed

        def _blocked(*_args, **_kwargs):
            raise RuntimeError("Network access is disabled by sandbox policy")

        def _guarded_create_connection(address, *args, **kwargs):
            host = _extract_host(address)
            if policy == "none":
                return _blocked()
            if policy == "allowlist" and not _is_allowed(host):
                return _blocked()
            return original_create_connection(address, *args, **kwargs)

        def _guarded_connect(sock, address):
            host = _extract_host(address)
            if policy == "none":
                return _blocked()
            if policy == "allowlist" and not _is_allowed(host):
                return _blocked()
            return original_socket_connect(sock, address)

        try:
            socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
            socket.socket.connect = _guarded_connect  # type: ignore[assignment]
            yield
        finally:
            socket.create_connection = original_create_connection  # type: ignore[assignment]
            socket.socket.connect = original_socket_connect  # type: ignore[assignment]

    def _get_internal_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Get a variable from the namespace.

        Args:
            name: Variable name.
            context: Context to get from.

        Returns:
            The variable value.

        Raises:
            VariableNotFoundError: If variable doesn't exist.
        """
        ctx = context or self._default_context
        if ctx.id not in self._namespaces:
            from ..exceptions import VariableNotFoundError

            raise VariableNotFoundError(name)

        namespace = self._namespaces[ctx.id]
        if name not in namespace:
            from ..exceptions import VariableNotFoundError

            raise VariableNotFoundError(name)

        return namespace[name]

    def _set_internal_variable(
        self, name: str, value: Any, context: Optional[Context] = None
    ) -> None:
        """Set a variable in the namespace.

        Args:
            name: Variable name.
            value: Value to set.
            context: Context to set in.
        """
        ctx = context or self._default_context
        if ctx.id not in self._namespaces:
            self._namespaces[ctx.id] = {"__builtins__": __builtins__}

        self._namespaces[ctx.id][name] = value

    def get_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Get a variable from the sandbox.

        This is more efficient than the base class implementation as it
        directly accesses the namespace.

        Args:
            name: Variable name.
            context: Context to get from.

        Returns:
            The variable value.
        """
        return self._get_internal_variable(name, context)

    def set_variable(self, name: str, value: Any, context: Optional[Context] = None) -> None:
        """Set a variable in the sandbox.

        This is more efficient than the base class implementation as it
        directly accesses the namespace.

        Args:
            name: Variable name.
            value: Value to set.
            context: Context to set in.
        """
        self._set_internal_variable(name, value, context)
