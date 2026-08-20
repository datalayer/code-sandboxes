# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: daytona sandbox (cloud sandbox with a stateful interpreter).

Run with:
  python examples/exec/daytona_sandbox_example.py

Auth:
- create an API key at https://app.daytona.io and export DAYTONA_API_KEY, or
- export DAYTONA_JWT_TOKEN together with DAYTONA_ORGANIZATION_ID.

GPU options:
- pass `--gpu H100` (or H200/RTX-4090/RTX-5090/RTX-PRO-6000), or
- set DAYTONA_GPU in the environment.

Asking for a GPU asks for a machine specification, and Daytona takes one only
when the sandbox is built from an IMAGE — so a GPU run starts from a Debian
image rather than from the default snapshot, and takes longer to come up.
"""

import argparse
import os

from exec_common import show_and_run

from code_sandboxes import Sandbox


def _has_daytona_auth() -> bool:
    if os.environ.get("DAYTONA_API_KEY"):
        return True
    return bool(os.environ.get("DAYTONA_JWT_TOKEN") and os.environ.get("DAYTONA_ORGANIZATION_ID"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daytona sandbox example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("DAYTONA_GPU"),
        help="Optional GPU (for example: H100, H200, RTX-4090).",
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


def main() -> None:
    args = _parse_args()
    if not _has_daytona_auth():
        print("Daytona auth not found.")
        print("Set DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID.")
        print("Create a key at https://app.daytona.io.")
        raise SystemExit(1)

    if args.gpu:
        print(f"Launching daytona sandbox with GPU: {args.gpu}")
    else:
        print("Launching daytona sandbox without GPU.")

    try:
        with Sandbox.create(variant="daytona", timeout=60, gpu=args.gpu) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")

            # What tells this variant apart from a per-snippet runner: the
            # interpreter holds one namespace, so the second snippet sees what
            # the first defined.
            show_and_run(sandbox, "x = 40")
            show_and_run(sandbox, "import sys; print(sys.version.split()[0])")
            state = show_and_run(sandbox, "x + 2")
            if state.text != "42":
                raise RuntimeError(
                    f"State did not survive between snippets: x + 2 gave {state.text!r}."
                )
            print("state verified: the namespace is shared between snippets.")

            # Bytes take the filesystem of the sandbox, not a program that
            # decodes them.
            sandbox.files.write_bytes("/tmp/hello.bin", b"from daytona")
            print("file round trip:", sandbox.files.read_bytes("/tmp/hello.bin"))

            if args.gpu:
                gpu_result = show_and_run(sandbox, _gpu_probe_code())
                probe = gpu_result.stdout.strip()
                # A GPU run must PROVE the GPU: the driver present, and at
                # least one device listed.
                if "nvidia-smi present" not in probe:
                    raise RuntimeError("GPU requested but nvidia-smi is missing in the sandbox.")
                if "devices NONE" in probe or "GPU-PROBE: devices" not in probe:
                    raise RuntimeError("GPU requested but no device is listed by nvidia-smi.")
                print(f"GPU verified: {args.gpu} is present.")

            # The error path, demonstrated ON PURPOSE: the run must not die,
            # the failure must come back as a `code_error` on the result. Said
            # before it happens, or the example's last lines read as a crash.
            print("-- error handling: the next snippet raises deliberately --")
            error_result = show_and_run(sandbox, "raise RuntimeError('daytona failure example')")
            if error_result.code_error is None:
                raise RuntimeError("The deliberate failure did not surface as a code_error.")
            print("error captured as expected — daytona example completed.")
    except Exception as exc:
        print("daytona example failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
