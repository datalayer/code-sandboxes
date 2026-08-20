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

from exec_common import show_and_run

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
    """Code that PROVES a GPU, not merely looks for one.

    Prints one `GPU-PROBE:` line per fact so the caller can assert on them:
    the driver must be there and must list at least one device.
    """
    return """
import shutil
import subprocess

smi = shutil.which("nvidia-smi")
print("GPU-PROBE: nvidia-smi", "present" if smi else "MISSING")
if smi:
    listing = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True).stdout.strip()
    print("GPU-PROBE: devices", listing or "NONE")

try:
    import torch  # type: ignore
    print("GPU-PROBE: torch", torch.__version__, "cuda", torch.cuda.is_available(), "count", torch.cuda.device_count())
except Exception as exc:
    print("GPU-PROBE: torch unavailable:", exc)
"""


def main() -> None:
    args = _parse_args()
    if not _has_modal_auth():
        print("Modal auth not found.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET or run: modal token new")
        raise SystemExit(1)

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
            show_and_run(sandbox, "import numpy as np; print(int(np.arange(5).sum()))")

            if gpu:
                gpu_result = show_and_run(sandbox, _gpu_probe_code())
                probe = gpu_result.stdout.strip()
                print("gpu_probe:\n", probe)
                # A GPU run must PROVE the GPU: the driver present, at
                # least one device listed, and the requested flavor named.
                if "nvidia-smi present" not in probe:
                    raise RuntimeError("GPU requested but nvidia-smi is missing in the sandbox.")
                if "devices NONE" in probe or "GPU-PROBE: devices" not in probe:
                    raise RuntimeError("GPU requested but no device is listed by nvidia-smi.")
                if gpu.lower() not in probe.lower():
                    raise RuntimeError(
                        f"GPU flavor {gpu!r} requested but not named by nvidia-smi: {probe!r}"
                    )
                print(f"GPU verified: flavor {gpu} is present.")

            # The error path, demonstrated ON PURPOSE: the run must not die,
            # the failure must come back as a `code_error` on the result. Said
            # before it happens, or the example's last lines read as a crash.
            print("-- error handling: the next snippet raises deliberately --")
            error_result = show_and_run(sandbox, "raise RuntimeError('modal failure example')")
            if error_result.code_error is None:
                raise RuntimeError(
                    "The deliberate failure did not surface as a code_error."
                )
            print("error captured as expected — modal example completed.")
    except Exception as exc:
        print("modal example failed:", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
