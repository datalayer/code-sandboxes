# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: kaggle sandbox (Kaggle interactive notebook runtime)."""

import os

from repl_common import run_repl

from code_sandboxes import Sandbox


def main() -> None:
    try:
        channels_url = os.environ.get("RUNTIME_CHANNELS_URL")
        runtime_url = os.environ.get("RUNTIME_URL")
        runtime_id = os.environ.get("RUNTIME_ID")

        if channels_url:
            kwargs = {"channels_url": channels_url}
        elif runtime_url:
            kwargs = {"server_url": runtime_url}
            if runtime_id:
                kwargs["kernel_id"] = runtime_id
        else:
            raise RuntimeError(
                "Set RUNTIME_CHANNELS_URL, or RUNTIME_URL (and optionally RUNTIME_ID). "
                "To create a new kernel, set KAGGLE_API_TOKEN and RUNTIME_URL only."
            )

        with Sandbox.create(variant="kaggle", timeout=60, **kwargs) as sandbox:
            run_repl(sandbox)
    except Exception as exc:
        print("kaggle REPL failed:", exc)
        print(
            "Hint: set KAGGLE_API_TOKEN and RUNTIME_URL to create a kernel, or "
            "export RUNTIME_CHANNELS_URL (the WebSocket channels URL of a running "
            "Kaggle notebook session), or RUNTIME_URL and RUNTIME_ID."
        )


if __name__ == "__main__":
    main()
