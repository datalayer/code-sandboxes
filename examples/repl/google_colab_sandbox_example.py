# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: colab sandbox (Google Colab runtime)."""

import os

from code_sandboxes import Sandbox, run_repl


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


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
            "What the Colab runtime was given, GPU included when there is one",
            """
            import subprocess
            print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "no GPU")
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
        runtime_url = _require("RUNTIME_URL")
        runtime_id = _require("RUNTIME_ID")
        runtime_proxy_token = _require("RUNTIME_PROXY_TOKEN")

        with Sandbox.create(
            variant="google-colab",
            timeout=60,
            server_url=runtime_url,
            kernel_id=runtime_id,
            proxy_token=runtime_proxy_token,
        ) as sandbox:
            run_repl(sandbox, examples=_examples())
    except Exception as exc:
        print("colab REPL failed:", exc)
        print(
            "Hint: export RUNTIME_URL, RUNTIME_ID, and RUNTIME_PROXY_TOKEN "
            "from an active Colab runtime session."
        )


if __name__ == "__main__":
    main()
