# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: monty sandbox (secure in-process interpreter)."""

from code_sandboxes import Sandbox, run_repl


def _examples() -> list[tuple[str, str]]:
    """Snippets worth pasting into this sandbox, for `:examples`."""
    return [
        (
            "State is kept between lines",
            """
            totals = [1, 2, 3]
            totals.append(4)
            sum(totals)
            """,
        ),
        (
            "Pure computation is what this interpreter is for",
            """
            fib = lambda n: n if n < 2 else fib(n - 1) + fib(n - 2)
            [fib(n) for n in range(12)]
            """,
        ),
        (
            "What it refuses: there is no filesystem and no network here",
            """
            import os
            os.listdir("/")
            """,
        ),
    ]


def main() -> None:
    try:
        with Sandbox.create(variant="monty", timeout=30, name="monty1") as sandbox:
            run_repl(sandbox, examples=_examples())
    except ModuleNotFoundError as exc:
        print("monty sandbox is not available:", exc)
    except Exception as exc:
        print("monty REPL failed:", exc)


if __name__ == "__main__":
    main()
