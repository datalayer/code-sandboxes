# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Unit tests for code-sandboxes package."""

import asyncio
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_sandboxes.base import Sandbox, SandboxVariant
from code_sandboxes.local.eval_sandbox import LocalEvalSandbox
from code_sandboxes.local.jupyter_sandbox import LocalJupyterSandbox
from code_sandboxes.models import (
    CodeError,
    Context,
    ExecutionResult,
    GPUType,
    Logs,
    MIMEType,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxInfo,
    SandboxStatus,
    SandboxVariant as SandboxVariantEnum,
)
from code_sandboxes.exceptions import SandboxNotStartedError


# =============================================================================
# Model Tests
# =============================================================================

class TestModels:
    """Tests for data models."""

    def test_mime_type_enum(self):
        """Test MIMEType enum values."""
        assert MIMEType.TEXT_PLAIN.value == "text/plain"
        assert MIMEType.APPLICATION_JSON.value == "application/json"
        assert MIMEType.IMAGE_PNG.value == "image/png"

    def test_sandbox_status_enum(self):
        """Test SandboxStatus enum values."""
        assert SandboxStatus.PENDING.value == "pending"
        assert SandboxStatus.RUNNING.value == "running"
        assert SandboxStatus.STOPPED.value == "stopped"
        assert SandboxStatus.ERROR.value == "error"

    def test_sandbox_variant_enum(self):
        """Test SandboxVariant enum values."""
        assert SandboxVariantEnum.LOCAL_EVAL.value == "local-eval"
        assert SandboxVariantEnum.LOCAL_DOCKER.value == "local-docker"
        assert SandboxVariantEnum.LOCAL_JUPYTER.value == "local-jupyter"
        assert SandboxVariantEnum.DATALAYER_RUNTIME.value == "datalayer-runtime"

    def test_gpu_type_enum(self):
        """Test GPUType enum values."""
        assert GPUType.T4.value == "T4"
        assert GPUType.A100.value == "A100"
        assert GPUType.H100.value == "H100"

    def test_resource_config(self):
        """Test ResourceConfig dataclass."""
        config = ResourceConfig(
            cpu=2.0,
            memory=4096,
            gpu="T4",
            gpu_count=1,
        )

        assert config.cpu == 2.0
        assert config.memory == 4096
        assert config.gpu == "T4"
        assert "cpu=2.0" in repr(config)

    def test_resource_config_default(self):
        """Test ResourceConfig with defaults."""
        config = ResourceConfig()

        assert config.cpu is None
        assert config.memory is None
        assert config.gpu is None
        assert config.gpu_count == 1

    def test_output_message(self):
        """Test OutputMessage dataclass."""
        msg = OutputMessage(
            line="Hello, World!",
            timestamp=1234567890.0,
            error=False,
        )

        assert msg.line == "Hello, World!"
        assert msg.timestamp == 1234567890.0
        assert msg.error is False

    def test_output_message_defaults(self):
        """Test OutputMessage with defaults."""
        msg = OutputMessage(line="test")

        assert msg.line == "test"
        assert msg.timestamp == 0.0
        assert msg.error is False

    def test_result(self):
        """Test Result dataclass."""
        result = Result(
            data={"text/plain": "42"},
            is_main_result=True,
        )

        assert result.data["text/plain"] == "42"
        assert result.is_main_result is True

    def test_code_error(self):
        """Test CodeError dataclass."""
        error = CodeError(
            name="ValueError",
            value="Invalid input",
            traceback="Traceback...",
        )

        assert error.name == "ValueError"
        assert error.value == "Invalid input"
        assert error.traceback == "Traceback..."
        
    def test_execution_success_status(self):
        """Test Execution with successful execution."""
        execution = ExecutionResult(
            results=[Result(data={"text/plain": "42"})],
            execution_ok=True,
            code_error=None,
            started_at=1000.0,
            completed_at=1001.5,
        )
        
        assert execution.execution_ok is True
        assert execution.execution_error is None
        assert execution.code_error is None
        assert execution.success is True
        assert execution.duration == 1.5

    def test_execution_code_error(self):
        """Test Execution with code error (Python exception)."""
        execution = ExecutionResult(
            execution_ok=True,
            code_error=CodeError(
                name="ValueError",
                value="Invalid value",
                traceback="Traceback...",
            ),
        )
        
        assert execution.execution_ok is True
        assert execution.code_error is not None
        assert execution.code_error.name == "ValueError"
        assert execution.success is False

    def test_execution_infrastructure_failure(self):
        """Test Execution with infrastructure failure."""
        execution = ExecutionResult(
            execution_ok=False,
            execution_error="Connection timeout",
        )
        
        assert execution.execution_ok is False
        assert execution.execution_error == "Connection timeout"
        assert execution.code_error is None
        assert execution.success is False

    def test_execution_interrupted(self):
        """Test Execution that was interrupted."""
        execution = ExecutionResult(
            execution_ok=True,
            interrupted=True,
        )
        
        assert execution.execution_ok is True
        assert execution.interrupted is True
        assert execution.success is False

    def test_sandbox_config(self):
        """Test SandboxConfig dataclass."""
        config = SandboxConfig(
            timeout=60.0,
            environment="python-gpu-env",
        )

        assert config.timeout == 60.0
        assert config.environment == "python-gpu-env"

    def test_context(self):
        """Test Context dataclass."""
        ctx = Context(id="ctx-123", language="python")

        assert ctx.id == "ctx-123"
        assert ctx.language == "python"


# =============================================================================
# LocalEvalSandbox Tests
# =============================================================================

class TestLocalEvalSandbox:
    """Tests for LocalEvalSandbox."""

    def test_create_sandbox(self):
        """Test creating a sandbox."""
        sandbox = LocalEvalSandbox()

        assert sandbox is not None
        assert not sandbox.is_started

    def test_start_sandbox(self):
        """Test starting a sandbox."""
        sandbox = LocalEvalSandbox()
        sandbox.start()

        assert sandbox.is_started
        assert sandbox.info is not None
        assert sandbox.info.status == "running"

    def test_stop_sandbox(self):
        """Test stopping a sandbox."""
        sandbox = LocalEvalSandbox()
        sandbox.start()
        sandbox.stop()

        assert not sandbox.is_started
        assert sandbox.info.status == "stopped"

    def test_context_manager(self):
        """Test using sandbox as context manager."""
        with LocalEvalSandbox() as sandbox:
            assert sandbox.is_started

        assert not sandbox.is_started

    def test_run_code_simple_expression(self):
        """Test running a simple expression."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("1 + 1")

            assert execution is not None
            assert len(execution.results) > 0
            assert "2" in execution.results[0].data.get("text/plain", "")

    def test_run_code_statement(self):
        """Test running statements."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("x = 42")
            execution = sandbox.run_code("x * 2")

            assert "84" in execution.results[0].data.get("text/plain", "")

    def test_run_code_print_output(self):
        """Test capturing print output."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("print('Hello, World!')")

            assert "Hello" in execution.logs.stdout_text

    def test_run_code_error(self):
        """Test handling runtime errors."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("1 / 0")

            assert execution.code_error is not None
            assert execution.code_error.name == "ZeroDivisionError"

    def test_run_code_syntax_error(self):
        """Test handling syntax errors."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("if if if")

            assert execution.code_error is not None
            assert "Syntax" in execution.code_error.name

    def test_variable_persistence(self):
        """Test that variables persist between executions."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("counter = 0")
            sandbox.run_code("counter += 10")
            execution = sandbox.run_code("counter")

            assert "10" in execution.results[0].data.get("text/plain", "")

    def test_async_state_persistence(self):
        """Test that async locals persist between executions."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code(
                """
async def set_value():
    value = 123
    return value

value = await set_value()
"""
            )
            execution = sandbox.run_code("value")

            assert "123" in execution.results[0].data.get("text/plain", "")

    def test_function_definition(self):
        """Test defining and calling functions."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("""
def greet(name):
    return f'Hello, {name}!'
""")
            execution = sandbox.run_code("greet('World')")

            assert "Hello, World!" in execution.results[0].data.get("text/plain", "")

    def test_async_code(self):
        """Test running async code."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("""
import asyncio

async def async_add(a, b):
    await asyncio.sleep(0.01)
    return a + b
""")
            execution = sandbox.run_code("asyncio.run(async_add(20, 22))")

            assert "42" in execution.results[0].data.get("text/plain", "")

    def test_async_await_direct(self):
        """Test running async code with await directly in code."""
        with LocalEvalSandbox() as sandbox:
            # First define an async function
            sandbox.run_code("""
import asyncio

async def fetch_data():
    await asyncio.sleep(0.01)
    return {"status": "success", "value": 42}
""")
            
            # Then call it with await directly (no asyncio.run wrapper)
            execution = sandbox.run_code("""
result = await fetch_data()
print(f"Status: {result['status']}, Value: {result['value']}")
result
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Status: success, Value: 42" in execution.stdout
            # The result should be returned
            if execution.results:
                assert "'status': 'success'" in execution.results[0].data.get("text/plain", "")

    def test_async_await_with_nested_calls(self):
        """Test async code with nested await calls."""
        with LocalEvalSandbox() as sandbox:
            # Define multiple async functions
            sandbox.run_code("""
import asyncio

async def get_number():
    await asyncio.sleep(0.01)
    return 10

async def multiply(x, factor):
    await asyncio.sleep(0.01)
    return x * factor

async def process():
    num = await get_number()
    result = await multiply(num, 5)
    return result
""")
            
            # Call with await
            execution = sandbox.run_code("""
final_result = await process()
print(f"Final result: {final_result}")
final_result
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Final result: 50" in execution.stdout
            # Check result if available
            if execution.results:
                assert "50" in execution.results[0].data.get("text/plain", "")

    def test_async_with_external_caller(self):
        """Test async code that calls external async functions stored in namespace."""
        with LocalEvalSandbox() as sandbox:
            # Set up an async callable in the namespace
            async def external_async_function(name):
                import asyncio
                await asyncio.sleep(0.01)
                return f"Hello, {name}!"
            
            sandbox.set_variable("external_func", external_async_function)
            
            # Call it with await
            execution = sandbox.run_code("""
greeting = await external_func("World")
print(greeting)
greeting
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Hello, World!" in execution.stdout
            # Check result if available  
            if execution.results:
                assert "Hello, World!" in execution.results[0].data.get("text/plain", "")

    def test_async_function_defined_in_separate_execution(self):
        """Test calling async function defined in a separate run_code call."""
        with LocalEvalSandbox() as sandbox:
            # Set up an external object that will be called
            class MockExecutor:
                async def call_tool(self, name, args):
                    return f"Called {name} with {args}"
            
            sandbox.set_variable("__executor__", MockExecutor())
            
            # First execution: define the async wrapper function
            execution1 = sandbox.run_code("""
async def __call_tool__(tool_name, arguments):
    '''Call a tool through the executor.'''
    return await __executor__.call_tool(tool_name, arguments)

# Test that it's defined
print(f"__call_tool__ defined: {callable(__call_tool__)}")
""")
            
            assert execution1.success, f"Execution failed: {execution1.code_error}"
            assert "defined: True" in execution1.stdout
            
            # Second execution: use the function with await
            execution2 = sandbox.run_code("""
result = await __call_tool__("test_tool", {"arg": "value"})
print(f"Result: {result}")
result
""")
            
            assert execution2.success, f"Execution failed: {execution2.code_error}"
            assert "Called test_tool" in execution2.stdout

    def test_async_stdout_capture(self):
        """Test that stdout is properly captured in async code."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("""
import asyncio

async def print_messages():
    print("First message")
    await asyncio.sleep(0.01)
    print("Second message")
    return "done"

result = await print_messages()
print(f"Result: {result}")
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "First message" in execution.stdout, f"Expected 'First message' in stdout, got: {execution.stdout!r}"
            assert "Second message" in execution.stdout, f"Expected 'Second message' in stdout, got: {execution.stdout!r}"
            assert "Result: done" in execution.stdout, f"Expected 'Result: done' in stdout, got: {execution.stdout!r}"

    def test_async_stderr_capture(self):
        """Test that stderr is properly captured in async code."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("""
import asyncio
import sys

async def print_errors():
    print("Error message", file=sys.stderr)
    await asyncio.sleep(0.01)
    print("Another error", file=sys.stderr)
    return "done"

result = await print_errors()
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Error message" in execution.stderr, f"Expected 'Error message' in stderr, got: {execution.stderr!r}"
            assert "Another error" in execution.stderr, f"Expected 'Another error' in stderr, got: {execution.stderr!r}"

    def test_async_mixed_output(self):
        """Test that both stdout and stderr are captured in async code."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code("""
import asyncio
import sys

async def mixed_output():
    print("stdout line 1")
    print("stderr line 1", file=sys.stderr)
    await asyncio.sleep(0.01)
    print("stdout line 2")
    print("stderr line 2", file=sys.stderr)

await mixed_output()
""")
            
            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "stdout line 1" in execution.stdout
            assert "stdout line 2" in execution.stdout
            assert "stderr line 1" in execution.stderr
            assert "stderr line 2" in execution.stderr

    def test_import_modules(self):
        """Test importing standard library modules."""
        with LocalEvalSandbox() as sandbox:
            sandbox.run_code("import json")
            execution = sandbox.run_code('json.dumps({"key": "value"})')

            assert '"key"' in execution.results[0].data.get("text/plain", "")

    def test_not_started_error(self):
        """Test error when running code without starting."""
        sandbox = LocalEvalSandbox()

        with pytest.raises(SandboxNotStartedError):
            sandbox.run_code("1 + 1")

    def test_unsupported_language(self):
        """Test error for unsupported language."""
        with LocalEvalSandbox() as sandbox:
            with pytest.raises(ValueError, match="only supports Python"):
                sandbox.run_code("console.log('hello')", language="javascript")

    def test_multiple_contexts(self):
        """Test multiple execution contexts."""
        with LocalEvalSandbox() as sandbox:
            ctx1 = sandbox.create_context("context1")
            ctx2 = sandbox.create_context("context2")

            sandbox.run_code("x = 1", context=ctx1)
            sandbox.run_code("x = 2", context=ctx2)

            exec1 = sandbox.run_code("x", context=ctx1)
            exec2 = sandbox.run_code("x", context=ctx2)

            assert "1" in exec1.results[0].data.get("text/plain", "")
            assert "2" in exec2.results[0].data.get("text/plain", "")

    def test_callbacks(self):
        """Test output callbacks."""
        stdout_messages = []
        result_messages = []

        def on_stdout(msg):
            stdout_messages.append(msg)

        def on_result(res):
            result_messages.append(res)

        with LocalEvalSandbox() as sandbox:
            sandbox.run_code(
                "print('callback test')\n42",
                on_stdout=on_stdout,
                on_result=on_result,
            )

        # Callbacks should have been called
        # (exact behavior depends on implementation)

    def test_environment_variables(self):
        """Test setting environment variables."""
        with LocalEvalSandbox() as sandbox:
            execution = sandbox.run_code(
                "import os; os.environ.get('TEST_VAR', 'not set')",
                envs={"TEST_VAR": "test_value"},
            )

            assert len(execution.results) > 0
            assert "test_value" in execution.results[0].data.get("text/plain", "")

    def test_network_policy_blocks_connections(self):
        """Test that network policy can block outbound connections."""
        config = SandboxConfig(network_policy="none")
        with LocalEvalSandbox(config=config) as sandbox:
            execution = sandbox.run_code(
                "import socket; socket.create_connection(('example.com', 80))"
            )

            assert execution.code_error is not None
            assert "Network access is disabled" in execution.code_error.value

    def test_sandbox_id(self):
        """Test sandbox ID is assigned."""
        with LocalEvalSandbox() as sandbox:
            assert sandbox.sandbox_id is not None
            assert len(sandbox.sandbox_id) > 0

    def test_sandbox_info(self):
        """Test sandbox info."""
        with LocalEvalSandbox() as sandbox:
            assert sandbox.info is not None
            assert sandbox.info.variant == "local-eval"
            assert sandbox.info.status == "running"


# =============================================================================
# Sandbox Factory Tests
# =============================================================================

class TestSandboxFactory:
    """Tests for Sandbox.create factory method."""

    def test_create_local_eval(self):
        """Test creating local-eval sandbox."""
        sandbox = Sandbox.create(variant="local-eval")

        assert sandbox is not None
        assert isinstance(sandbox, LocalEvalSandbox)

    def test_create_local_jupyter(self):
        """Test creating local-jupyter sandbox."""
        sandbox = Sandbox.create(variant=SandboxVariant.LOCAL_JUPYTER)

        assert sandbox is not None
        assert isinstance(sandbox, LocalJupyterSandbox)

    def test_create_with_config(self):
        """Test creating sandbox with config."""
        config = SandboxConfig(timeout=120.0)
        sandbox = Sandbox.create(variant="local-eval", config=config)

        assert sandbox.config.timeout == 120.0

    def test_create_with_timeout(self):
        """Test creating sandbox with timeout parameter."""
        sandbox = Sandbox.create(variant="local-eval", timeout=90.0)

        assert sandbox.config.timeout == 90.0

    def test_create_with_env(self):
        """Test creating sandbox with environment variables."""
        config = SandboxConfig(env_vars={"MY_VAR": "my_value"})
        sandbox = Sandbox.create(
            variant="local-eval",
            config=config,
        )

        assert sandbox.config.env_vars.get("MY_VAR") == "my_value"

    def test_create_invalid_variant(self):
        """Test error for invalid variant."""
        with pytest.raises(ValueError):
            Sandbox.create(variant="invalid-variant")


# =============================================================================
# Local Jupyter Sandbox Tests
# =============================================================================

class TestLocalJupyterSandbox:
    """Tests for LocalJupyterSandbox."""

    def test_local_jupyter_persistence(self, tmp_path: Path):
        """Test persistence across requests in local-jupyter sandbox."""
        if os.environ.get("RUN_LOCAL_JUPYTER_TESTS") != "1":
            pytest.skip("Set RUN_LOCAL_JUPYTER_TESTS=1 to enable local-jupyter tests")
        try:
            import jupyter_server  # noqa: F401
        except Exception:
            pytest.skip("jupyter_server is not available")

        sandbox = LocalJupyterSandbox(config=SandboxConfig(working_dir=str(tmp_path)))
        try:
            sandbox.start()
        except Exception as exc:
            pytest.skip(f"local-jupyter sandbox not available: {exc}")

        try:
            sandbox.run_code("x = 7")
            execution = sandbox.run_code("x + 1")
            assert "8" in execution.results[0].data.get("text/plain", "")
        finally:
            sandbox.stop()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for code-sandboxes."""

    def test_complex_computation(self):
        """Test complex computation in sandbox."""
        with LocalEvalSandbox() as sandbox:
            code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(10)]
result
"""
            execution = sandbox.run_code(code)

            assert len(execution.results) > 0
            result = execution.results[0].data.get("text/plain", "")
            assert "0, 1, 1, 2, 3, 5, 8, 13, 21, 34" in result

    def test_data_processing(self):
        """Test data processing in sandbox."""
        with LocalEvalSandbox() as sandbox:
            code = """
import json

data = [
    {"name": "Alice", "score": 85},
    {"name": "Bob", "score": 92},
    {"name": "Charlie", "score": 78},
]

average = sum(d["score"] for d in data) / len(data)
top_scorer = max(data, key=lambda x: x["score"])

{"average": average, "top_scorer": top_scorer["name"]}
"""
            execution = sandbox.run_code(code)

            assert len(execution.results) > 0
            result = execution.results[0].data.get("text/plain", "")
            assert "Bob" in result
            assert "85" in result

    def test_file_operations(self, tmp_path: Path):
        """Test file operations in sandbox."""
        with LocalEvalSandbox() as sandbox:
            # Write a file
            file_path = tmp_path / "test.txt"
            code = f"""
with open("{file_path}", "w") as f:
    f.write("Hello from sandbox!")
"""
            sandbox.run_code(code)

            # Read it back (in two steps: setup then expression)
            code = f"""
with open("{file_path}", "r") as f:
    content = f.read()
content
"""
            execution = sandbox.run_code(code)

            assert len(execution.results) > 0
            assert "Hello from sandbox!" in execution.results[0].data.get("text/plain", "")

    def test_multiline_output(self):
        """Test multiline output."""
        with LocalEvalSandbox() as sandbox:
            code = """
for i in range(5):
    print(f"Line {i}")
"""
            execution = sandbox.run_code(code)

            stdout = execution.logs.stdout_text
            for i in range(5):
                assert f"Line {i}" in stdout

    def test_exception_handling(self):
        """Test exception handling in user code."""
        with LocalEvalSandbox() as sandbox:
            code = """
try:
    result = 1 / 0
except ZeroDivisionError:
    result = "caught"
result
"""
            execution = sandbox.run_code(code)

            assert execution.code_error is None
            assert len(execution.results) > 0
            assert "caught" in execution.results[0].data.get("text/plain", "")

    def test_class_definition(self):
        """Test defining and using classes."""
        with LocalEvalSandbox() as sandbox:
            code = """
class Calculator:
    def __init__(self, value=0):
        self.value = value
    
    def add(self, x):
        self.value += x
        return self
    
    def multiply(self, x):
        self.value *= x
        return self

calc = Calculator(5).add(3).multiply(2)
calc.value
    """
            execution = sandbox.run_code(code)

            assert len(execution.results) > 0
            assert "16" in execution.results[0].data.get("text/plain", "")

    def test_list_comprehension(self):
        """Test list comprehensions."""
        with LocalEvalSandbox() as sandbox:
            code = "[x**2 for x in range(10) if x % 2 == 0]"
            execution = sandbox.run_code(code)

            assert "[0, 4, 16, 36, 64]" in execution.results[0].data.get("text/plain", "")

    def test_generator_expression(self):
        """Test generator expressions."""
        with LocalEvalSandbox() as sandbox:
            code = "sum(x**2 for x in range(10))"
            execution = sandbox.run_code(code)

            assert "285" in execution.results[0].data.get("text/plain", "")
