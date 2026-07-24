# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: monty sandbox (secure in-process interpreter)."""

from repl_common import run_repl

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(variant="monty", timeout=30, name="monty1") as sandbox:
            run_repl(sandbox)
    except ModuleNotFoundError as exc:
        print("monty sandbox is not available:", exc)
    except Exception as exc:
        print("monty REPL failed:", exc)


if __name__ == "__main__":
    main()
