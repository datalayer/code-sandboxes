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
        elif os.environ.get("KAGGLE_LIVE"):
            print("mode: live — one batch job holds a real kernel, the code")
            print("travels over a private dataset bus of your account.")
            print("Booting the agent takes a few minutes (the job queues);")
            print("each turn then costs seconds, and state persists.")
            print()
            print("REQUIREMENT: a phone-verified Kaggle account. The agent")
            print("needs internet from inside the job to reach the dataset")
            print("bus, and Kaggle grants kernels internet only after phone")
            print("verification (kaggle.com/settings). Without it the job")
            print("dies unable to resolve api.kaggle.com.")
            kwargs = {"live": True, "timeout": 600}
            gpu = os.environ.get("KAGGLE_GPU")
            if gpu:
                # `1`/`true` ask for "a GPU" — the T4 is the everyday one —
                # anything else names the accelerator (t4, p100, l4, …).
                kwargs["gpu"] = (
                    "t4" if gpu.strip().lower() in ("1", "true", "yes") else gpu
                )
                print()
                print(f"accelerator: {kwargs['gpu']} — a GPU job queues")
                print("longer than a CPU one before it boots.")
        else:
            print("mode: batch — each input is a Kaggle batch job (about a")
            print("minute each). State carries over: each turn replays the")
            print("session's code before the new input, so `x = 1` then")
            print("`print(x)` works — at the price of re-running everything")
            print("each turn. Set KAGGLE_LIVE=1 for one persistent kernel.")
            kwargs = {"timeout": 600}
            gpu = os.environ.get("KAGGLE_GPU")
            if gpu:
                kwargs["gpu"] = (
                    "t4" if gpu.strip().lower() in ("1", "true", "yes") else gpu
                )
                print()
                print(f"accelerator: {kwargs['gpu']} — every batch job runs")
                print("with it, and queues longer than a CPU one.")

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
