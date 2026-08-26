# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: eval sandbox (no isolation)."""

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
            "Where this is running",
            """
            import platform, sys
            platform.node(), platform.platform(), sys.version.split()[0]
            """,
        ),
        (
            "It isolates NOTHING — this is your own process and your own disk",
            """
            import os
            os.getcwd(), len(os.listdir("."))
            """,
        ),
    ]


def main() -> None:
    with Sandbox.create(variant="eval", timeout=30, examples=_examples()) as sandbox:
        run_repl(sandbox)


if __name__ == "__main__":
    main()
