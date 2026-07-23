# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: jupyter sandbox (persistent kernel state)."""

from code_sandboxes import Sandbox

from repl_common import run_repl


def main() -> None:
    try:
        with Sandbox.create(variant="jupyter", timeout=30) as sandbox:
            run_repl(sandbox)
    except ModuleNotFoundError as exc:
        print("jupyter sandbox is not available:", exc)
    except Exception as exc:
        print("jupyter REPL failed:", exc)


if __name__ == "__main__":
    main()
