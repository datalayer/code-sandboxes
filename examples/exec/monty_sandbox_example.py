# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: monty sandbox (secure in-process interpreter).

Run with:
  python examples/monty_sandbox_example.py

Note: This requires code-sandboxes[monty] / pydantic-monty.
"""

from code_sandboxes import Sandbox


def main() -> None:
    try:
        with Sandbox.create(variant="monty", timeout=30) as sandbox:
            sandbox.run_code("x = 21")
            result = sandbox.run_code("x * 2")
            print("result:", result.text)

            result = sandbox.run_code("print('hello from monty')")
            print("stdout:", result.stdout)

            error_result = sandbox.run_code("raise ValueError('monty failure example')")
            if error_result.code_error:
                print(
                    "code_error:",
                    f"{error_result.code_error.name}: {error_result.code_error.value}",
                )
    except ModuleNotFoundError as exc:
        print("monty sandbox is not available:", exc)
    except Exception as exc:
        print("monty example failed:", exc)


if __name__ == "__main__":
    main()
