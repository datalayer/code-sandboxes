# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: modal sandbox (cloud container execution).

Run with:
  python examples/modal_sandbox_example.py

Auth options:
- `modal token new` (writes ~/.modal.toml), or
- set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.

GPU options:
- pass `--gpu T4` (or A10G/A100/H100), or
- set MODAL_GPU in the environment.
"""

import argparse
import os
from pathlib import Path

from code_sandboxes import Sandbox


def _has_modal_auth() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return Path.home().joinpath(".modal.toml").exists()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modal sandbox example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("MODAL_GPU"),
        help="Optional GPU flavor (for example: T4, A10G, A100, H100).",
    )
    return parser.parse_args()


def _gpu_probe_code() -> str:
    return """
import shutil
import subprocess

print("nvidia-smi available:", shutil.which("nvidia-smi") is not None)
if shutil.which("nvidia-smi"):
    result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    print(result.stdout or "(no nvidia-smi output)")

try:
    import torch  # type: ignore
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())
except Exception as exc:
    print("torch check unavailable:", exc)
"""


def main() -> None:
    args = _parse_args()

    if not _has_modal_auth():
        print("Modal auth not found.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET or run: modal token new")
        return

    gpu = args.gpu
    if gpu:
        print(f"Launching modal sandbox with GPU flavor: {gpu}")
    else:
        print("Launching modal sandbox without GPU.")

    try:
        with Sandbox.create(
            variant="modal",
            timeout=60,
            gpu=gpu,
            pip_packages=["numpy"],
        ) as sandbox:
            result = sandbox.run_code("import numpy as np; print(int(np.arange(5).sum()))")
            print("stdout:", result.stdout.strip())

            if gpu:
                gpu_result = sandbox.run_code(_gpu_probe_code())
                print("gpu_probe:\n", gpu_result.stdout.strip())

            error_result = sandbox.run_code("raise RuntimeError('modal failure example')")
            if error_result.code_error:
                print(
                    "code_error:",
                    f"{error_result.code_error.name}: {error_result.code_error.value}",
                )
    except Exception as exc:
        print("modal example failed:", exc)


if __name__ == "__main__":
    main()
