# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: kaggle sandbox (batch by default, interactive when pointed at one).

With no ``RUNTIME_URL``/``RUNTIME_CHANNELS_URL`` set, the REPL runs in
Kaggle batch mode: every input is submitted as its own batch kernel through
the official ``kaggle`` package — expect about a minute per input, and no
state carried between them. Credentials: ``KAGGLE_API_TOKEN``, or
``KAGGLE_USERNAME``/``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``.

Point it at a live kaggle.com notebook session (``RUNTIME_URL`` or
``RUNTIME_CHANNELS_URL``) for a stateful, immediate REPL.
"""

import os

from repl_common import run_repl

from code_sandboxes import Sandbox


def main() -> None:
    channels_url = os.environ.get("RUNTIME_CHANNELS_URL")
    runtime_url = os.environ.get("RUNTIME_URL")
    runtime_id = os.environ.get("RUNTIME_ID")
    try:
        if channels_url:
            print("mode: interactive (live kaggle.com session)")
            kwargs = {"channels_url": channels_url, "timeout": 60}
        elif runtime_url:
            print("mode: interactive (live kaggle.com session)")
            kwargs = {"server_url": runtime_url, "timeout": 60}
            if runtime_id:
                kwargs["kernel_id"] = runtime_id
        else:
            print("mode: batch — each input is a Kaggle batch job (about a")
            print("minute each, stateless). Set RUNTIME_URL for a live session.")
            kwargs = {"timeout": 600}

        with Sandbox.create(variant="kaggle", **kwargs) as sandbox:
            run_repl(sandbox)
    except Exception as exc:
        print("kaggle REPL failed:", exc)
        print(
            "Hint: batch mode needs Kaggle credentials — KAGGLE_API_TOKEN, or "
            "KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json. For a "
            "stateful REPL, start a notebook session on kaggle.com and set "
            "RUNTIME_URL (its proxy URL) or RUNTIME_CHANNELS_URL."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
