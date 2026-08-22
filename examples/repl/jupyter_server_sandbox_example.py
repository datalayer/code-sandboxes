# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: jupyter sandbox (persistent kernel state)."""

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
            "A real kernel, so a figure comes back as a figure",
            """
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([1, 4, 9, 16])
            fig
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
        with Sandbox.create(variant="jupyter-server", timeout=30) as sandbox:
            run_repl(sandbox, examples=_examples())
    except ModuleNotFoundError as exc:
        print("jupyter sandbox is not available:", exc)
    except Exception as exc:
        print("jupyter REPL failed:", exc)


if __name__ == "__main__":
    main()
