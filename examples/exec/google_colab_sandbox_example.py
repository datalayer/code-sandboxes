# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: colab sandbox (Google Colab runtime).

Run with:
  RUNTIME_URL=... RUNTIME_ID=... RUNTIME_PROXY_TOKEN=... \\
  python examples/colab_sandbox_example.py
"""

import os

from exec_common import show_and_run

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
            variant="google_colab",
            timeout=60,
            server_url=runtime_url,
            kernel_id=runtime_id,
            proxy_token=runtime_proxy_token,
        ) as sandbox:
            show_and_run(sandbox, "x = 40")
            result = show_and_run(sandbox, "x + 2")

            result = show_and_run(sandbox, "print('hello from colab')")
    except Exception as exc:
        print("colab example failed:", exc)
        print(
            "Hint: export RUNTIME_URL, RUNTIME_ID, and RUNTIME_PROXY_TOKEN "
            "from an active Colab runtime session."
        )


if __name__ == "__main__":
    main()
