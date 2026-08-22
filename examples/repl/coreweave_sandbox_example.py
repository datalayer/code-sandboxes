# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: coreweave sandbox (container on CoreWeave, optionally on a GPU).

Run with:
  python examples/repl/coreweave_sandbox_example.py

Auth:
- create an access token in the CoreWeave console and export CWSANDBOX_API_KEY.
- export CWSANDBOX_BASE_URL to talk to another control plane than
  https://api.cwsandbox.com.

Definitions persist between lines because the variant keeps one session process
for the sandbox and feeds it a line at a time. When that process cannot be
started, the sandbox falls back to a process per line — still working, no
longer stateful — and the prompt says so before it opens.
"""

import argparse
import os

from code_sandboxes import Sandbox, run_repl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the coreweave sandbox REPL example.")
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


def _examples(gpu: str | None) -> list[tuple[str, str]]:
    """Snippets worth pasting into this sandbox, for `:examples`.

    A sandbox with a card in it is worth different lines from one without, so
    the GPU set replaces the general one rather than being appended to it.
    """
    if gpu:
        return [
            (
                "What the GPU is, straight from the driver",
                """
            import subprocess
            print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
            """,
            ),
            (
                "The same from Python, once torch is there",
                """
            import torch
            torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0)
            """,
            ),
            (
                "Install torch if the image has none (a minute or two)",
                """
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"], check=True)
            """,
            ),
            (
                "A workload that actually uses it: a matmul, timed on the device",
                """
            import time, torch
            a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
            b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize(); start = time.perf_counter()
            [a @ b for _ in range(10)] and torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / 10
            f"{2 * 8192 ** 3 / seconds / 1e12:.1f} TFLOP/s"
            """,
            ),
            (
                "How much memory the card has, and how much this used",
                """
            import torch
            free, total = torch.cuda.mem_get_info()
            f"{(total - free) / 1e9:.1f} GB used of {total / 1e9:.1f} GB"
            """,
            ),
        ]
    return [
        (
            "State is kept between lines",
            """
            totals = [1, 2, 3]
            totals.append(4)
            sum(totals)
            """,
        ),
        (
            "Where this is running",
            """
            import platform, sys
            platform.node(), platform.platform(), sys.version.split()[0]
            """,
        ),
        (
            "The filesystem is the sandbox's own",
            """
            from pathlib import Path
            Path("/tmp/notes.txt").write_text("written inside the sandbox")
            Path("/tmp/notes.txt").read_text()
            """,
        ),
        (
            "Install a package into the sandbox",
            """
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "httpx"], check=True)
            import httpx; httpx.__version__
            """,
        ),
    ]


def main() -> None:
    args = _parse_args()
    if not os.environ.get("CWSANDBOX_API_KEY"):
        print("CoreWeave auth not found.")
        print("Set CWSANDBOX_API_KEY to an access token from the CoreWeave console.")
        print("See https://docs.coreweave.com/products/sandboxes.")
        raise SystemExit(1)

    if args.gpu:
        print(f"Launching coreweave sandbox REPL with GPU: {args.gpu}")
    else:
        print("Launching coreweave sandbox REPL without GPU.")

    try:
        with Sandbox.create(
            variant="coreweave",
            timeout=60,
            gpu=args.gpu,
            container_image=args.image,
        ) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")
            info = sandbox.info
            if info and info.metadata.get("stateful"):
                print("Definitions persist between lines: one session process holds them.")
            else:
                print("No session process: each line runs on its own, and nothing crosses.")
            run_repl(sandbox, examples=_examples(args.gpu))
    except Exception as exc:
        print("coreweave REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
