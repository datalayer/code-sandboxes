# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: monty sandbox (secure in-process interpreter).

Run with:
  python examples/monty_sandbox_example.py

Note: This requires code-sandboxes[monty] / pydantic-monty.
"""

from code_sandboxes import Sandbox

from exec_common import show_and_run


def main() -> None:
    try:
        with Sandbox.create(variant="monty", timeout=30) as sandbox:
            show_and_run(sandbox, "x = 21")
            result = show_and_run(sandbox, "x * 2")
            print("result:", result.text)

            result = show_and_run(sandbox, "print('hello from monty')")
            print("stdout:", result.stdout)

            error_result = show_and_run(sandbox, "raise ValueError('monty failure example')")
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
