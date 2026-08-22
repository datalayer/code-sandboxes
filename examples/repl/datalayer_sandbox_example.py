# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: datalayer sandbox (cloud runtime)."""

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
            "What the runtime was given",
            """
            import os
            {k: v for k, v in os.environ.items() if k.startswith("DATALAYER_")}
            """,
        ),
        (
            "The filesystem is the sandbox's own",
            """
            from pathlib import Path
            Path("/tmp/notes.txt").write_text("written inside the sandbox")
            Path("/tmp/notes.txt").read_text()
            """,
        ),
    ]


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
            run_repl(sandbox, examples=_examples())
    except Exception as exc:
        print("datalayer REPL failed:", exc)


if __name__ == "__main__":
    main()
