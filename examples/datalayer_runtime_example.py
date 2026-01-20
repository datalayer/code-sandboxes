# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: datalayer-runtime sandbox (cloud runtime).

Run with:
    python examples/datalayer_runtime_example.py

This requires Datalayer runtime credentials/config.
"""

from code_sandboxes import Sandbox


def main() -> None:
    try:
        environments = Sandbox.list_environments(variant="datalayer-runtime")
        if not environments:
            raise RuntimeError("No environments available.")

        print("Available environments:")
        for env in environments:
            print(f"- {env.name} ({env.title})")

        first_env = environments[0]
        with Sandbox.create(
            variant="datalayer-runtime",
            timeout=60,
            environment=first_env.name,
        ) as sandbox:
            result = sandbox.run_code("print('hello from datalayer runtime')")
            print("stdout:", result.stdout)
    except Exception as exc:  # noqa: BLE001
        print("datalayer-runtime example failed:", exc)
        print("Exception type:", type(exc))
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
