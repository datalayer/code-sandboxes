# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: docker sandbox (container isolation)."""

from code_sandboxes import Sandbox, run_repl


def _examples() -> list[tuple[str, str]]:
    """Snippets worth pasting into this sandbox, for `:examples`."""
    return [
        (
            "Where this is running",
            """
            import platform, sys
            platform.node(), platform.platform(), sys.version.split()[0]
            """,
        ),
        (
            "State is kept between lines",
            """
            totals = [1, 2, 3]
            totals.append(4)
            sum(totals)
            """,
        ),
        (
            "Which image this container came from",
            """
            from pathlib import Path
            print(Path("/etc/os-release").read_text())
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
        (
            "Install a package into the sandbox",
            """
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx"], check=True)
            import httpx; httpx.__version__
            """,
        ),
    ]


def main() -> None:
    try:
        with Sandbox.create(
            variant="docker",
            timeout=30,
            image="code-sandboxes-jupyter:latest",
            examples=_examples(),
        ) as sandbox:
            run_repl(sandbox)
    except ModuleNotFoundError as exc:
        print("docker sandbox is not available:", exc)
    except Exception as exc:
        print("docker REPL failed:", exc)


if __name__ == "__main__":
    main()
