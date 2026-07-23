# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Shared helper for exec-style sandbox examples.

Prints the code that will be executed before running it, so the output
clearly shows the snippet associated with each result.
"""

from __future__ import annotations

from code_sandboxes import Sandbox


def show_and_run(sandbox: Sandbox, code: str, **kwargs):
    """Print the code snippet, execute it on the sandbox, and return the result."""
    print(">>> code:")
    for line in code.strip("\n").splitlines():
        print(f"    {line}")
    return sandbox.run_code(code, **kwargs)
