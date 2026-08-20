# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Shared helper for exec-style sandbox examples.

Every run reads the same way in the terminal, whatever the provider: the code
that was submitted, then what came back — stdout, the value of the last
expression, and the error when there is one. An example that printed only its
own commentary left the reader guessing which snippet produced which lines.
"""

from __future__ import annotations

from code_sandboxes import Sandbox


def show_code(code: str) -> None:
    """Print the code about to be submitted, indented under its marker."""
    print(">>> code:")
    for line in code.strip("\n").splitlines():
        print(f"    {line}")


def show_result(result) -> None:
    """Print what an execution came back with, and only what it came back with."""
    stdout = (result.stdout or "").strip("\n")
    if stdout:
        print("<<< stdout:")
        for line in stdout.splitlines():
            print(f"    {line}")
    text = (result.text or "").strip()
    # The value of the last expression, when it is not just the stdout again.
    if text and text != stdout.strip():
        print(f"<<< result: {text}")
    error = getattr(result, "code_error", None)
    if error is not None:
        print(f"<<< error: {error.name}: {error.value}")
    if not stdout and not text and error is None:
        print("<<< (no output)")


def show_and_run(sandbox: Sandbox, code: str, **kwargs):
    """Print the code, run it, print what came back, and return the result."""
    show_code(code)
    result = sandbox.run_code(code, **kwargs)
    show_result(result)
    return result
