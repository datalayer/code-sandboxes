# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: docker sandbox (container isolation)."""

from code_sandboxes import Sandbox

from repl_common import run_repl


def main() -> None:
    try:
        with Sandbox.create(
            variant="docker",
            timeout=30,
            image="code-sandboxes-jupyter:latest",
        ) as sandbox:
            run_repl(sandbox)
    except ModuleNotFoundError as exc:
        print("docker sandbox is not available:", exc)
    except Exception as exc:
        print("docker REPL failed:", exc)


if __name__ == "__main__":
    main()
