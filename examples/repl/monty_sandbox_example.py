# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: monty sandbox (secure in-process interpreter)."""

from code_sandboxes import Sandbox

from repl_common import run_repl


def main() -> None:
    try:
        with Sandbox.create(variant="monty", timeout=30) as sandbox:
            run_repl(sandbox)
    except ModuleNotFoundError as exc:
        print("monty sandbox is not available:", exc)
    except Exception as exc:
        print("monty REPL failed:", exc)


if __name__ == "__main__":
    main()
