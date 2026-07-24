# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: colab sandbox (Google Colab runtime)."""

import os

from repl_common import run_repl

from code_sandboxes import Sandbox


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    try:
        runtime_url = _require("RUNTIME_URL")
        runtime_id = _require("RUNTIME_ID")
        runtime_proxy_token = _require("RUNTIME_PROXY_TOKEN")

        with Sandbox.create(
            variant="colab",
            timeout=60,
            server_url=runtime_url,
            kernel_id=runtime_id,
            proxy_token=runtime_proxy_token,
        ) as sandbox:
            run_repl(sandbox)
    except Exception as exc:
        print("colab REPL failed:", exc)
        print(
            "Hint: export RUNTIME_URL, RUNTIME_ID, and RUNTIME_PROXY_TOKEN "
            "from an active Colab runtime session."
        )


if __name__ == "__main__":
    main()
