# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: kaggle sandbox (batch by default, interactive when pointed at one).

Two modes, chosen by the environment:

* Batch (default, nothing but credentials needed): ``Sandbox.create`` with no
  runtime URL submits the code as a Kaggle batch kernel through the official
  ``kaggle`` package, polls it, and streams status and outputs back. Each run
  is an independent job — no state carries between executions — so this mode
  runs one self-contained snippet. Credentials: ``KAGGLE_API_TOKEN``, or
  ``KAGGLE_USERNAME``/``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``.
  ``--gpu [FLAVOR]`` requests an accelerator on the job (T4 or P100; the
  account must be phone-verified on kaggle.com) and the run fails unless
  the probe sees it.
* Interactive (stateful): set ``RUNTIME_CHANNELS_URL``, or ``RUNTIME_URL``
  (and optionally ``RUNTIME_ID``) of a notebook session started on
  kaggle.com. There ``--gpu`` can only verify — the accelerator is chosen
  when the session is started on kaggle.com.

Run with:
  python kaggle_sandbox_example.py                # batch, CPU
  python kaggle_sandbox_example.py --gpu T4       # batch, on a T4
  RUNTIME_URL='https://.../proxy' python kaggle_sandbox_example.py
"""

import argparse
import os

from exec_common import show_and_run

from code_sandboxes import CodeError, Sandbox


GPU_PROBE = """
import shutil
import subprocess

smi = shutil.which("nvidia-smi")
print("GPU-PROBE: nvidia-smi", "present" if smi else "MISSING")
if smi:
    listing = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True).stdout.strip()
    print("GPU-PROBE: devices", listing or "NONE")
"""

BATCH_SNIPPET = """
import sys
print("python:", sys.version.split()[0])
x = 40
print("result:", x + 2)
print("hello from kaggle batch")
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the kaggle sandbox example.")
    parser.add_argument(
        "--gpu",
        nargs="?",
        const="T4",
        default=os.environ.get("KAGGLE_GPU") or None,
        metavar="FLAVOR",
        help=(
            "Request (batch mode) or verify (interactive mode) a GPU. "
            "Batch mode passes the flavor as the job's accelerator — T4 and "
            "P100 are the shapes Kaggle batch supports. In interactive mode "
            "the accelerator was chosen on kaggle.com; this only proves it "
            "is there."
        ),
    )
    return parser.parse_args()


def _check_gpu_probe(probe: str, flavor: str) -> None:
    """Fail loudly unless the probe saw a device of the requested flavor."""
    print("gpu_probe:\n", probe)
    if "nvidia-smi present" not in probe or "devices NONE" in probe:
        raise RuntimeError(
            f"GPU verification failed: no GPU in the Kaggle runtime "
            f"(requested {flavor})."
        )
    # 'T4 x2' must match a 'Tesla T4' listing: the family name is the check.
    family = flavor.split()[0].lower()
    devices = next(
        (line for line in probe.splitlines() if "devices" in line), ""
    ).lower()
    if family not in devices:
        raise RuntimeError(
            f"GPU verification failed: requested {flavor} but nvidia-smi "
            f"lists: {devices or 'nothing'}"
        )
    print(f"GPU verified: flavor {flavor} is present.")


def _run_batch(gpu: str | None) -> None:
    """Submit one self-contained snippet as a Kaggle batch job and stream it."""
    print("mode: batch (no RUNTIME_URL set — the job runs on kaggle.com)")
    if gpu:
        print(f"accelerator requested: {gpu}")
    code = BATCH_SNIPPET + (GPU_PROBE if gpu else "")
    # A batch job queues, boots and converts the notebook: minutes, not
    # seconds — and a GPU job queues longer than a CPU one.
    timeout = 900 if gpu else 600
    create_kwargs = {"gpu": gpu} if gpu else {}
    with Sandbox.create(variant="kaggle", timeout=timeout, **create_kwargs) as sandbox:
        lines: list[str] = []
        error = None
        for event in sandbox.run_code_streaming(code, timeout=timeout):
            if isinstance(event, CodeError):
                error = event
                print(f"[error] {event.name}: {event.value}")
            elif hasattr(event, "line"):
                lines.append(event.line)
                print(event.line, flush=True)
            else:
                print("[result]", getattr(event, "data", event))
        if error is not None:
            raise RuntimeError(f"{error.name}: {error.value}")
        stdout = "\n".join(lines)
        if "result: 42" not in stdout or "hello from kaggle batch" not in stdout:
            raise RuntimeError("The batch job completed without the expected output.")
        if gpu:
            probe = "\n".join(line for line in lines if line.startswith("GPU-PROBE:"))
            try:
                _check_gpu_probe(probe, gpu)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} The job was submitted with the accelerator, so a "
                    "missing GPU is Kaggle declining it: batch accelerators "
                    "need a phone-verified account (kaggle.com/settings) and "
                    "count against the weekly GPU quota. Valid batch shapes "
                    "are T4 and P100 (TPU: Tpu1VmV38)."
                ) from exc
        print("batch job verified: outputs are all present.")


def _run_interactive(gpu: str | None) -> None:
    """Talk to a live kaggle.com notebook session: stateful executions."""
    channels_url = os.environ.get("RUNTIME_CHANNELS_URL")
    runtime_url = os.environ.get("RUNTIME_URL")
    runtime_id = os.environ.get("RUNTIME_ID")
    if channels_url:
        kwargs = {"channels_url": channels_url}
    else:
        kwargs = {"server_url": runtime_url}
        if runtime_id:
            kwargs["kernel_id"] = runtime_id
    print("mode: interactive (live kaggle.com session)")
    with Sandbox.create(variant="kaggle", timeout=60, **kwargs) as sandbox:
        show_and_run(sandbox, "x = 40")
        result = show_and_run(sandbox, "x + 2")
        print("result:", result.text)

        result = show_and_run(sandbox, "print('hello from kaggle')")
        print("stdout:", result.stdout)

        if gpu:
            probe = show_and_run(sandbox, GPU_PROBE).stdout.strip()
            _check_gpu_probe(probe, gpu)


def main() -> None:
    args = _parse_args()
    interactive = bool(
        os.environ.get("RUNTIME_CHANNELS_URL") or os.environ.get("RUNTIME_URL")
    )
    try:
        if interactive:
            _run_interactive(args.gpu)
        else:
            _run_batch(args.gpu)
    except Exception as exc:
        print("kaggle example failed:", exc)
        if not interactive:
            print(
                "Hint: batch mode needs Kaggle credentials — KAGGLE_API_TOKEN, "
                "or KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json. "
                "For a stateful session instead, start a notebook on kaggle.com "
                "and set RUNTIME_URL (its https://….jupyter-proxy.kaggle.net/"
                "k/…/proxy URL) or RUNTIME_CHANNELS_URL."
            )
        else:
            print(
                "Hint: RUNTIME_URL must be the proxy URL of a session that is "
                "still running on kaggle.com; KAGGLE_API_TOKEN authenticates "
                "kernel creation on it. Unset RUNTIME_URL to use batch mode, "
                "which needs no session."
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
