# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: kaggle sandbox (Kaggle interactive notebook runtime).

Authentication supports two modes:

* API token (default): set ``KAGGLE_API_TOKEN`` and provide ``RUNTIME_URL``
  (ending in ``/proxy``). Omitting ``RUNTIME_ID`` creates a new kernel.
* Signed proxy URL: provide the WebSocket channels URL of a running Kaggle
  notebook session (``RUNTIME_CHANNELS_URL``), or ``RUNTIME_URL`` and
  ``RUNTIME_ID`` of an existing kernel.

Run with:
  KAGGLE_API_TOKEN=... RUNTIME_URL='https://.../proxy' \\
  python examples/kaggle_sandbox_example.py

or:
  RUNTIME_CHANNELS_URL='wss://.../proxy/api/kernels/<id>/channels?...' \\
  python examples/kaggle_sandbox_example.py

or:
  RUNTIME_URL='https://.../proxy' RUNTIME_ID=<id> \\
  python examples/kaggle_sandbox_example.py
"""

import os

from exec_common import show_and_run

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
            show_and_run(sandbox, "x = 40")
            result = show_and_run(sandbox, "x + 2")
            print("result:", result.text)

            result = show_and_run(sandbox, "print('hello from kaggle')")
            print("stdout:", result.stdout)
    except Exception as exc:
        print("kaggle example failed:", exc)
        print(
            "Hint: set KAGGLE_API_TOKEN and RUNTIME_URL to create a kernel, or "
            "export RUNTIME_CHANNELS_URL (the WebSocket channels URL of a running "
            "Kaggle notebook session), or RUNTIME_URL and RUNTIME_ID."
        )


if __name__ == "__main__":
    main()
