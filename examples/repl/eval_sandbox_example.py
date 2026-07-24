# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: eval sandbox (no isolation)."""

from repl_common import run_repl

from code_sandboxes import Sandbox


def main() -> None:
    with Sandbox.create(variant="eval", timeout=30) as sandbox:
        run_repl(sandbox)


if __name__ == "__main__":
    main()
