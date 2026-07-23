# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: datalayer sandbox (cloud runtime)."""

from repl_common import run_repl

from code_sandboxes import Sandbox


def main() -> None:
    try:
        environments = Sandbox.list_environments(variant="datalayer")
        if not environments:
            raise RuntimeError("No environments available.")

        first_env = environments[0]
        print(f"Using environment: {first_env.name} ({first_env.title})")

        with Sandbox.create(
            variant="datalayer",
            timeout=60,
            environment=first_env.name,
        ) as sandbox:
            run_repl(sandbox)
    except Exception as exc:
        print("datalayer REPL failed:", exc)


if __name__ == "__main__":
    main()
