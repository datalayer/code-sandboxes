# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: modal sandbox (cloud container execution)."""

import argparse
import os
from pathlib import Path

from code_sandboxes import Sandbox, provider_ingress_execution, run_repl


def _has_modal_auth() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return Path.home().joinpath(".modal.toml").exists()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modal sandbox REPL example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("MODAL_GPU"),
        help="Optional GPU flavor (for example: T4, A10G, A100, H100).",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Execute directly through the Modal process adapter.",
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
    if not _has_modal_auth():
        print("Modal auth not found.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET or run: modal token new")
        raise SystemExit(1)

    if args.gpu:
        print(f"Launching modal sandbox REPL with GPU flavor: {args.gpu}")
    else:
        print("Launching modal sandbox REPL without GPU.")

    try:
        with (
            Sandbox.create(
                variant="modal",
                timeout=60,
                gpu=args.gpu,
                examples=_examples(args.gpu),
            ) as provider,
            provider_ingress_execution(provider, direct=args.direct) as sandbox,
        ):
            run_repl(sandbox)
    except Exception as exc:
        print("modal REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
