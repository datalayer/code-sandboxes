# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Shared REPL helper for sandbox examples."""

from __future__ import annotations

from code_sandboxes import Sandbox


def _build_prompt(sandbox: Sandbox) -> str:
    info = sandbox.info
    if info is None:
        return "sandbox>>> "

    variant = info.variant or "unknown"
    name_or_id = info.name or (info.id[:8] if info.id else "sandbox")
    return f"sandbox({variant}:{name_or_id})>>> "


def _read_input(prompt: str) -> str | None:
    try:
        return input(prompt).strip()
    except EOFError:
        print()
        return None
    except KeyboardInterrupt:
        print("\nInterrupted. Use :quit to exit.")
        return ""


def _handle_repl_command(code: str) -> bool:
    if code in {":quit", ":exit"}:
        return False
    if code == ":help":
        print("Enter Python expressions/statements.")
        print(":quit or :exit to leave.")
    return True


def _print_result(result) -> None:
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


def run_repl(sandbox: Sandbox) -> None:
    """Run a small interactive Python REPL on a sandbox."""
    prompt = _build_prompt(sandbox)

    print("Sandbox REPL ready.")
    print("Type Python code and press Enter.")
    print("Use :quit or :exit to leave, :help for help.")

    while True:
        code = _read_input(prompt)
        if code is None:
            break
        if not code:
            continue
        if code.startswith(":"):
            if not _handle_repl_command(code):
                break
            continue

        result = sandbox.run_code(code)
        _print_result(result)

    print("REPL closed.")
