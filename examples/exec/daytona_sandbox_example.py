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
- name several — `--gpu H100,H200` — to fall back to the second when the
  first is unavailable.
- add `--spot` for preemptible capacity: far cheaper, outside the GPU quota,
  and reclaimed without warning. The run below says which happened.

Asking for a GPU asks for a machine specification, and Daytona takes one only
when the sandbox is built from an IMAGE — so a GPU run starts from a Debian
image rather than from the default snapshot, and takes longer to come up.
"""

import argparse
import os

from code_sandboxes import Sandbox, provider_ingress_execution, show_and_run


def _has_daytona_auth() -> bool:
    if os.environ.get("DAYTONA_API_KEY"):
        return True
    return bool(os.environ.get("DAYTONA_JWT_TOKEN") and os.environ.get("DAYTONA_ORGANIZATION_ID"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daytona sandbox example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("DAYTONA_GPU"),
        help=(
            "Optional GPU (for example: H100, H200, RTX-4090). Several, "
            "comma-separated, are an ordered list of preferences Daytona "
            "falls back along."
        ),
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        default=bool(os.environ.get("DAYTONA_SPOT")),
        help=(
            "Run on preemptible GPU capacity: far cheaper and outside the GPU "
            "quota, and reclaimed without warning. Needs --gpu."
        ),
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Execute directly through the Daytona SDK adapter.",
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


def _verify_gpu(sandbox, provider, gpu: str, *, spot: bool) -> None:
    """Prove the GPU is there, and say whether spot capacity still holds it."""
    probe = show_and_run(sandbox, _gpu_probe_code()).stdout.strip()
    # A GPU run must PROVE the GPU: the driver present, and at least one
    # device listed.
    if "nvidia-smi present" not in probe:
        raise RuntimeError("GPU requested but nvidia-smi is missing in the sandbox.")
    if "devices NONE" in probe or "GPU-PROBE: devices" not in probe:
        raise RuntimeError("GPU requested but no device is listed by nvidia-smi.")
    print(f"GPU verified: {gpu} is present.")
    if spot:
        # No warning is given before spot capacity is taken back, so the only
        # way to know is to ask.
        print("spot: reclaimed at", provider.preempted_at() or "not yet")


def main() -> None:
    args = _parse_args()
    if not _has_daytona_auth():
        print("Daytona auth not found.")
        print("Set DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID.")
        print("Create a key at https://app.daytona.io.")
        raise SystemExit(1)

    if args.gpu:
        capacity = "spot (preemptible)" if args.spot else "on-demand"
        print(f"Launching daytona sandbox with GPU: {args.gpu} on {capacity} capacity")
    elif args.spot:
        print("--spot needs --gpu: preemptible capacity is GPU capacity.")
        raise SystemExit(1)
    else:
        print("Launching daytona sandbox without GPU.")

    try:
        with (
            Sandbox.create(variant="daytona", timeout=60, gpu=args.gpu, spot=args.spot) as provider,
            provider_ingress_execution(provider, direct=args.direct) as sandbox,
        ):
            print(f"Sandbox: {provider.sandbox_id}")

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

            print("-- streaming: one number should appear every second --")
            show_and_run(
                sandbox,
                "import time\nfor i in range(1, 10):\n    print(i)\n    time.sleep(1)",
            )

            # Bytes take the filesystem of the sandbox, not a program that
            # decodes them.
            provider.files.write_bytes("/tmp/hello.bin", b"from daytona")
            print("file round trip:", provider.files.read_bytes("/tmp/hello.bin"))

            if args.gpu:
                _verify_gpu(sandbox, provider, args.gpu, spot=args.spot)

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
