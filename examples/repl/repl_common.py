# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Shared REPL helper for sandbox examples."""

from __future__ import annotations

from code_sandboxes import Sandbox


def run_repl(sandbox: Sandbox) -> None:
    """Run a small interactive Python REPL on a sandbox."""
    print("Sandbox REPL ready.")
    print("Type Python code and press Enter.")
    print("Use :quit or :exit to leave, :help for help.")

    while True:
        try:
            code = input("sandbox>>> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nInterrupted. Use :quit to exit.")
            continue

        if not code:
            continue
        if code in {":quit", ":exit"}:
            break
        if code == ":help":
            print("Enter Python expressions/statements.")
            print(":quit or :exit to leave.")
            continue

        result = sandbox.run_code(code)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.text and result.text != result.stdout.strip():
            print(result.text)
        if result.stderr:
            print(result.stderr.rstrip())
        if result.code_error:
            print(f"{result.code_error.name}: {result.code_error.value}")
        if not result.execution_ok and result.execution_error:
            print(f"Execution error: {result.execution_error}")

    print("REPL closed.")
