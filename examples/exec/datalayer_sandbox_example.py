# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: datalayer sandbox (cloud runtime).

Run with:
    python examples/datalayer_runtime_example.py

This requires Datalayer runtime credentials/config.
"""

from code_sandboxes import Sandbox, show_and_run


def main() -> None:
    try:
        environments = Sandbox.list_environments(variant="datalayer")
        if not environments:
            raise RuntimeError("No environments available.")

        print("Available environments:")
        for env in environments:
            print(f"- {env.name} ({env.title})")

        first_env = environments[0]
        with Sandbox.create(
            variant="datalayer",
            timeout=60,
            environment=first_env.name,
        ) as sandbox:
            show_and_run(sandbox, "print('hello from datalayer runtime')")
    except Exception as exc:
        print("datalayer example failed:", exc)
        print("Exception type:", type(exc))
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
