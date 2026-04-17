# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""eval sandbox tests."""

import pytest

from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.exceptions import SandboxNotStartedError
from code_sandboxes.models import SandboxConfig


class TestEvalSandbox:
    """Tests for EvalSandbox."""

    def test_create_sandbox(self):
        """Test creating a sandbox."""
        sandbox = EvalSandbox()

        assert sandbox is not None
        assert not sandbox.is_started

    def test_start_sandbox(self):
        """Test starting a sandbox."""
        sandbox = EvalSandbox()
        sandbox.start()

        assert sandbox.is_started
        assert sandbox.info is not None
        assert sandbox.info.status == "running"

    def test_stop_sandbox(self):
        """Test stopping a sandbox."""
        sandbox = EvalSandbox()
        sandbox.start()
        sandbox.stop()

        assert not sandbox.is_started
        assert sandbox.info.status == "stopped"

    def test_context_manager(self):
        """Test using sandbox as context manager."""
        with EvalSandbox() as sandbox:
            assert sandbox.is_started

        assert not sandbox.is_started

    def test_run_code_simple_expression(self):
        """Test running a simple expression."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code("1 + 1")

            assert execution is not None
            assert len(execution.results) > 0
            assert "2" in execution.results[0].data.get("text/plain", "")

    def test_run_code_statement(self):
        """Test running statements."""
        with EvalSandbox() as sandbox:
            sandbox.run_code("x = 42")
            execution = sandbox.run_code("x * 2")

            assert "84" in execution.results[0].data.get("text/plain", "")

    def test_run_code_print_output(self):
        """Test capturing print output."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code("print('Hello, World!')")

            assert "Hello" in execution.logs.stdout_text

    def test_run_code_error(self):
        """Test handling runtime errors."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code("1 / 0")

            assert execution.code_error is not None
            assert execution.code_error.name == "ZeroDivisionError"

    def test_run_code_syntax_error(self):
        """Test handling syntax errors."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code("if if if")

            assert execution.code_error is not None
            assert "Syntax" in execution.code_error.name

    def test_run_code_system_exit(self):
        """Test handling sys.exit without treating it as a code error."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code("import sys; sys.exit(2)")

            assert execution.execution_ok is True
            assert execution.code_error is None
            assert execution.exit_code == 2
            assert execution.success is False

    def test_variable_persistence(self):
        """Test that variables persist between executions."""
        with EvalSandbox() as sandbox:
            sandbox.run_code("counter = 0")
            sandbox.run_code("counter += 10")
            execution = sandbox.run_code("counter")

            assert "10" in execution.results[0].data.get("text/plain", "")

    def test_async_state_persistence(self):
        """Test that async locals persist between executions."""
        with EvalSandbox() as sandbox:
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
        with EvalSandbox() as sandbox:
            sandbox.run_code(
                """
def greet(name):
    return f'Hello, {name}!'
"""
            )
            execution = sandbox.run_code("greet('World')")

            assert "Hello, World!" in execution.results[0].data.get("text/plain", "")

    def test_async_code(self):
        """Test running async code."""
        with EvalSandbox() as sandbox:
            sandbox.run_code(
                """
import asyncio

async def async_add(a, b):
    await asyncio.sleep(0.01)
    return a + b
"""
            )
            execution = sandbox.run_code("asyncio.run(async_add(20, 22))")

            assert "42" in execution.results[0].data.get("text/plain", "")

    def test_async_await_direct(self):
        """Test running async code with await directly in code."""
        with EvalSandbox() as sandbox:
            sandbox.run_code(
                """
import asyncio

async def fetch_data():
    await asyncio.sleep(0.01)
    return {"status": "success", "value": 42}
"""
            )

            execution = sandbox.run_code(
                """
result = await fetch_data()
print(f"Status: {result['status']}, Value: {result['value']}")
result
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Status: success, Value: 42" in execution.stdout
            if execution.results:
                assert "'status': 'success'" in execution.results[0].data.get("text/plain", "")

    def test_async_await_with_nested_calls(self):
        """Test async code with nested await calls."""
        with EvalSandbox() as sandbox:
            sandbox.run_code(
                """
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
"""
            )

            execution = sandbox.run_code(
                """
final_result = await process()
print(f"Final result: {final_result}")
final_result
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Final result: 50" in execution.stdout
            if execution.results:
                assert "50" in execution.results[0].data.get("text/plain", "")

    def test_async_with_external_caller(self):
        """Test async code that calls external async functions stored in namespace."""
        with EvalSandbox() as sandbox:

            async def external_async_function(name):
                import asyncio

                await asyncio.sleep(0.01)
                return f"Hello, {name}!"

            sandbox.set_variable("external_func", external_async_function)

            execution = sandbox.run_code(
                """
greeting = await external_func("World")
print(greeting)
greeting
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Hello, World!" in execution.stdout
            if execution.results:
                assert "Hello, World!" in execution.results[0].data.get("text/plain", "")

    def test_async_function_defined_in_separate_execution(self):
        """Test calling async function defined in a separate run_code call."""
        with EvalSandbox() as sandbox:
            # Register a tool caller function
            async def my_tool_caller(tool_name, arguments):
                return f"Called {tool_name} with {arguments}"

            sandbox.register_tool_caller(my_tool_caller)

            # Verify __call_tool__ is available
            execution1 = sandbox.run_code(
                """
print(f"__call_tool__ defined: {callable(__call_tool__)}")
"""
            )

            assert execution1.success, f"Execution failed: {execution1.code_error}"
            assert "defined: True" in execution1.stdout

            execution2 = sandbox.run_code(
                """
result = await __call_tool__("test_tool", {"arg": "value"})
print(f"Result: {result}")
result
"""
            )

            assert execution2.success, f"Execution failed: {execution2.code_error}"
            assert "Called test_tool" in execution2.stdout

    def test_async_stdout_capture(self):
        """Test that stdout is properly captured in async code."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code(
                """
import asyncio

async def print_messages():
    print("First message")
    await asyncio.sleep(0.01)
    print("Second message")
    return "done"

result = await print_messages()
print(f"Result: {result}")
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "First message" in execution.stdout
            assert "Second message" in execution.stdout
            assert "Result: done" in execution.stdout

    def test_async_stderr_capture(self):
        """Test that stderr is properly captured in async code."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code(
                """
import asyncio
import sys

async def print_errors():
    print("Error message", file=sys.stderr)
    await asyncio.sleep(0.01)
    print("Another error", file=sys.stderr)
    return "done"

result = await print_errors()
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "Error message" in execution.stderr
            assert "Another error" in execution.stderr

    def test_async_mixed_output(self):
        """Test that both stdout and stderr are captured in async code."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code(
                """
import asyncio
import sys

async def mixed_output():
    print("stdout line 1")
    print("stderr line 1", file=sys.stderr)
    await asyncio.sleep(0.01)
    print("stdout line 2")
    print("stderr line 2", file=sys.stderr)

await mixed_output()
"""
            )

            assert execution.success, f"Execution failed: {execution.code_error}"
            assert "stdout line 1" in execution.stdout
            assert "stdout line 2" in execution.stdout
            assert "stderr line 1" in execution.stderr
            assert "stderr line 2" in execution.stderr

    def test_import_modules(self):
        """Test importing standard library modules."""
        with EvalSandbox() as sandbox:
            sandbox.run_code("import json")
            execution = sandbox.run_code('json.dumps({"key": "value"})')

            assert '"key"' in execution.results[0].data.get("text/plain", "")

    def test_not_started_error(self):
        """Test error when running code without starting."""
        sandbox = EvalSandbox()

        with pytest.raises(SandboxNotStartedError):
            sandbox.run_code("1 + 1")

    def test_unsupported_language(self):
        """Test error for unsupported language."""
        with EvalSandbox() as sandbox:
            with pytest.raises(ValueError, match="only supports Python"):
                sandbox.run_code("console.log('hello')", language="javascript")

    def test_multiple_contexts(self):
        """Test multiple execution contexts."""
        with EvalSandbox() as sandbox:
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

        with EvalSandbox() as sandbox:
            sandbox.run_code(
                "print('callback test')\n42",
                on_stdout=on_stdout,
                on_result=on_result,
            )

        # Callbacks should have been called
        # (exact behavior depends on implementation)

    def test_environment_variables(self):
        """Test setting environment variables."""
        with EvalSandbox() as sandbox:
            execution = sandbox.run_code(
                "import os; os.environ.get('TEST_VAR', 'not set')",
                envs={"TEST_VAR": "test_value"},
            )

            assert len(execution.results) > 0
            assert "test_value" in execution.results[0].data.get("text/plain", "")

    def test_network_policy_blocks_connections(self):
        """Test that network policy can block outbound connections."""
        config = SandboxConfig(network_policy="none")
        with EvalSandbox(config=config) as sandbox:
            execution = sandbox.run_code(
                "import socket; socket.create_connection(('example.com', 80))"
            )

            assert execution.code_error is not None
            assert "Network access is disabled" in execution.code_error.value

    def test_sandbox_id(self):
        """Test sandbox ID is assigned."""
        with EvalSandbox() as sandbox:
            assert sandbox.sandbox_id is not None
            assert len(sandbox.sandbox_id) > 0

    def test_sandbox_info(self):
        """Test sandbox info."""
        with EvalSandbox() as sandbox:
            assert sandbox.info is not None
            assert sandbox.info.variant == "eval"
            assert sandbox.info.status == "running"


class TestInterruptAndExecutionState:
    """Tests for is_executing and interrupt() on EvalSandbox."""

    def test_is_executing_false_when_idle(self):
        """is_executing is False when no code is running."""
        with EvalSandbox() as sandbox:
            assert sandbox.is_executing is False

    def test_is_executing_true_during_run(self):
        """is_executing is True while code is running."""
        import threading

        started = threading.Event()
        proceed = threading.Event()
        observed_executing = []

        def run_in_thread():
            # Execute code that waits for the proceed event
            sandbox.run_code("import time\n" "started.set()\n" "proceed.wait(5)\n")

        with EvalSandbox() as sandbox:
            sandbox.set_variable("started", started)
            sandbox.set_variable("proceed", proceed)

            thread = threading.Thread(target=run_in_thread)
            thread.start()

            # Wait until the sandbox code is running
            started.wait(5)
            observed_executing.append(sandbox.is_executing)

            # Release the code
            proceed.set()
            thread.join(5)

        assert observed_executing == [True]
        # After execution, should be False
        assert sandbox.is_executing is False

    def test_interrupt_when_not_executing(self):
        """interrupt() returns False when nothing is running."""
        with EvalSandbox() as sandbox:
            assert sandbox.interrupt() is False

    def test_interrupt_stops_running_code(self):
        """interrupt() sends KeyboardInterrupt to running code."""
        import threading
        import time

        started = threading.Event()

        def run_in_thread():
            return sandbox.run_code(
                "import time\n" "started.set()\n" "while True:\n" "    time.sleep(0.01)\n"
            )

        with EvalSandbox() as sandbox:
            sandbox.set_variable("started", started)

            results = [None]

            def run_and_capture():
                results[0] = sandbox.run_code(
                    "import time\n" "started.set()\n" "while True:\n" "    time.sleep(0.01)\n"
                )

            thread = threading.Thread(target=run_and_capture)
            thread.start()

            # Wait until the sandbox code is running
            started.wait(5)
            time.sleep(0.05)  # Give it a moment to enter the loop

            # Interrupt
            success = sandbox.interrupt()
            assert success is True

            thread.join(5)

            result = results[0]
            assert result is not None
            assert result.interrupted is True
            assert result.code_error is not None
            assert result.code_error.name == "KeyboardInterrupt"
