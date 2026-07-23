# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Monty sandbox implementation.

`Monty <https://github.com/pydantic/monty>`_ is a minimal, secure Python
interpreter written in Rust (distributed as ``pydantic-monty``). It runs a
restricted subset of Python in-process with microsecond startup times and no
access to the host filesystem, environment or network unless explicitly granted.

This makes it an excellent fit for running short, LLM-generated snippets where a
full container/kernel would be overkill. Note that Monty only supports a subset
of Python (no third-party libraries, limited stdlib), so rich display outputs and
filesystem/command operations are not available.

This sandbox uses ``MontyRepl``, whose session state (heap and namespace)
persists across successive ``run_code`` calls.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

from .base import Sandbox
from .exceptions import SandboxConfigurationError, SandboxNotStartedError
from .models import (
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
    SandboxStatus,
)

logger = logging.getLogger(__name__)


class MontySandbox(Sandbox):
    """Sandbox backed by the Monty secure Python interpreter.

    Args:
        config: Optional sandbox configuration.
        type_check: Whether Monty should type-check the code before running it.
        type_check_stubs: Optional type stub definitions used when ``type_check``
            is enabled.
        external_functions: Mapping of names to host callables the sandboxed code
            is allowed to call.
        limits: Optional Monty ``ResourceLimits`` mapping (memory, stack depth,
            execution time, ...).
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        type_check: bool = False,
        type_check_stubs: Optional[str] = None,
        external_functions: Optional[dict[str, Any]] = None,
        limits: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(config)
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._external_functions = external_functions or {}
        self._limits = limits
        self._repl = None
        self._collect_streams = None
        self._sandbox_id = str(uuid.uuid4())
        self._execution_count = 0
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="monty",
                title="Monty",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": "monty"},
            )
        ]

    def start(self) -> None:
        if self._started:
            return

        try:
            import pydantic_monty
        except ImportError as exc:
            raise SandboxConfigurationError(
                "pydantic-monty is required for MontySandbox. "
                "Install it with: pip install pydantic-monty"
            ) from exc

        repl_kwargs: dict[str, Any] = {"type_check": self._type_check}
        if self._type_check_stubs is not None:
            repl_kwargs["type_check_stubs"] = self._type_check_stubs
        if self._limits is not None:
            repl_kwargs["limits"] = self._limits

        # A stateful REPL session; heap and namespace persist across feed_run calls.
        self._repl = pydantic_monty.MontyRepl(**repl_kwargs)
        self._collect_streams = pydantic_monty.CollectStreams

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="monty",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={"interpreter": "monty"},
            config=self.config,
        )
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._repl = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
        return str(value)

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
        if not self._started or self._repl is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"MontySandbox only supports Python, got: {language}")

        started_at = time.time()
        self._execution_count += 1

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: Optional[CodeError] = None

        # Fresh collector per call so we only capture this snippet's output.
        collector = self._collect_streams()
        return_value: Any = None
        raised = False
        try:
            return_value = self._repl.feed_run(
                code,
                external_functions=self._external_functions or None,
                print_callback=collector,
            )
        except Exception as e:
            raised = True
            code_error = CodeError(
                name=type(e).__name__,
                value=str(e),
                traceback="",
            )
            if on_error:
                on_error(code_error)

        current_time = time.time()

        # ``collector.output`` yields a list of (stream, text) tuples.
        for stream, text in collector.output or []:
            is_err = stream == "stderr"
            for line in self._coerce_text(text).splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=is_err)
                if is_err:
                    stderr_messages.append(msg)
                    if on_stderr:
                        on_stderr(msg)
                else:
                    stdout_messages.append(msg)
                    if on_stdout:
                        on_stdout(msg)

        if not raised and return_value is not None:
            result = Result(
                data={"text/plain": self._coerce_text(return_value)},
                is_main_result=True,
            )
            results.append(result)
            if on_result:
                on_result(result)

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            execution_count=self._execution_count,
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
        )

    def _get_internal_variable(self, name: str, context: Optional[Context] = None):
        if not self._started or self._repl is None:
            raise SandboxNotStartedError()
        return self._repl.feed_run(name)

    def _set_internal_variable(self, name: str, value, context: Optional[Context] = None) -> None:
        if not self._started or self._repl is None:
            raise SandboxNotStartedError()
        self._repl.feed_run(f"{name} = {value!r}")
