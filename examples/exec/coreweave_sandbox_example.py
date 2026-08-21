# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: coreweave sandbox (container on CoreWeave, optionally on a GPU).

Run with:
  python examples/exec/coreweave_sandbox_example.py

Auth:
- create an access token in the CoreWeave console and export CWSANDBOX_API_KEY.
- export CWSANDBOX_BASE_URL to talk to another control plane than
  https://api.cwsandbox.com.

GPU options:
- pass `--gpu H100`, or set COREWEAVE_GPU in the environment.
- a GPU is a machine specification, and the image has to carry what the code
  needs — `--image` says which one to run.

CoreWeave offers a container and `exec`, a process at a time, and nothing that
holds a Python namespace between calls. So the variant holds one itself: a
single session process is started with the sandbox and fed one snippet at a
time. A driver that cannot start, or that goes away mid-session, drops the
sandbox back to a process per snippet — working, merely stateless — which is
why the state check below asks the sandbox which of the two it got.
"""

import argparse
import os

from code_sandboxes import Sandbox, show_and_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coreweave sandbox example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("COREWEAVE_GPU"),
        help="Optional GPU (for example: H100, H200, A100).",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("COREWEAVE_IMAGE"),
        help="The container image the sandbox runs. `python:3.11` unless given.",
    )
    return parser.parse_args()


def _gpu_probe_code() -> str:
    """Code that PROVES a GPU, not merely looks for one.

    Prints one `GPU-PROBE:` line per fact so the caller can assert on them:
    the driver must be there, and must list at least one device.
    """
    return """
import shutil
import subprocess

smi = shutil.which("nvidia-smi")
print("GPU-PROBE: nvidia-smi", "present" if smi else "MISSING")
if smi:
    listing = subprocess.run(
        ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
    ).stdout.strip()
    print("GPU-PROBE: devices", listing or "NONE")
"""


def _verify_gpu(sandbox, gpu: str) -> None:
    """Prove the GPU is there rather than take the request for an answer."""
    probe = show_and_run(sandbox, _gpu_probe_code()).stdout.strip()
    if "nvidia-smi present" not in probe:
        raise RuntimeError("GPU requested but nvidia-smi is missing in the sandbox.")
    if "devices NONE" in probe or "GPU-PROBE: devices" not in probe:
        raise RuntimeError("GPU requested but no device is listed by nvidia-smi.")
    print(f"GPU verified: {gpu} is present.")


def main() -> None:
    args = _parse_args()
    if not os.environ.get("CWSANDBOX_API_KEY"):
        print("CoreWeave auth not found.")
        print("Set CWSANDBOX_API_KEY to an access token from the CoreWeave console.")
        print("See https://docs.coreweave.com/products/sandboxes.")
        raise SystemExit(1)

    if args.gpu:
        print(f"Launching coreweave sandbox with GPU: {args.gpu}")
    else:
        print("Launching coreweave sandbox without GPU.")

    try:
        with Sandbox.create(
            variant="coreweave",
            timeout=60,
            gpu=args.gpu,
            container_image=args.image,
        ) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")

            show_and_run(sandbox, "x = 40")
            show_and_run(sandbox, "import sys; print(sys.version.split()[0])")
            state = show_and_run(sandbox, "x + 2")

            # The sandbox says which arrangement it ended up with, and only a
            # session process promises the namespace crosses. Asserting on
            # state that was never promised would fail an honest fallback.
            info = sandbox.info
            stateful = bool(info.metadata.get("stateful")) if info else False
            if stateful:
                if state.text != "42":
                    raise RuntimeError(
                        f"State did not survive between snippets: x + 2 gave {state.text!r}."
                    )
                print("state verified: the session process shares one namespace.")
            else:
                print("no session process: each snippet ran on its own, so x did not cross.")

            # Bytes take the filesystem of the container, not a program that
            # decodes them.
            sandbox.files.write_bytes("/tmp/hello.bin", b"from coreweave")
            print("file round trip:", sandbox.files.read_bytes("/tmp/hello.bin"))

            if args.gpu:
                _verify_gpu(sandbox, args.gpu)

            # The error path, demonstrated ON PURPOSE: the run must not die,
            # the failure must come back as a `code_error` on the result. Said
            # before it happens, or the example's last lines read as a crash.
            print("-- error handling: the next snippet raises deliberately --")
            error_result = show_and_run(sandbox, "raise RuntimeError('coreweave failure example')")
            if error_result.code_error is None:
                raise RuntimeError("The deliberate failure did not surface as a code_error.")
            print("error captured as expected — coreweave example completed.")
    except Exception as exc:
        print("coreweave example failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
