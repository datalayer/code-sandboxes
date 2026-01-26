# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Integration tests for code-sandboxes."""

from pathlib import Path

from code_sandboxes.local.eval_sandbox import LocalEvalSandbox


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
            file_path = tmp_path / "test.txt"
            code = f"""
with open("{file_path}", "w") as f:
    f.write("Hello from sandbox!")
"""
            sandbox.run_code(code)

            code = f"""
with open("{file_path}", "r") as f:
    content = f.read()
content
"""
            execution = sandbox.run_code(code)

            assert len(execution.results) > 0
            assert "Hello from sandbox!" in execution.results[0].data.get(
                "text/plain", ""
            )

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

            assert "[0, 4, 16, 36, 64]" in execution.results[0].data.get(
                "text/plain", ""
            )

    def test_generator_expression(self):
        """Test generator expressions."""
        with LocalEvalSandbox() as sandbox:
            code = "sum(x**2 for x in range(10))"
            execution = sandbox.run_code(code)

            assert "285" in execution.results[0].data.get("text/plain", "")
