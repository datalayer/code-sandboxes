# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: daytona sandbox (cloud sandbox with a stateful interpreter).

Run with:
  python examples/repl/daytona_sandbox_example.py

Auth:
- create an API key at https://app.daytona.io and export DAYTONA_API_KEY, or
- export DAYTONA_JWT_TOKEN together with DAYTONA_ORGANIZATION_ID.

The prompt behaves as a REPL should: definitions persist between lines, and a
line that is an expression answers with its value.
"""

import argparse
import os

from code_sandboxes import Sandbox, provider_ingress_execution, run_repl


def _has_daytona_auth() -> bool:
    if os.environ.get("DAYTONA_API_KEY"):
        return True
    return bool(os.environ.get("DAYTONA_JWT_TOKEN") and os.environ.get("DAYTONA_ORGANIZATION_ID"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daytona sandbox REPL example.")
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
        "--keep",
        action="store_true",
        help=(
            "Leave the sandbox in the organization when the REPL closes, "
            "stopped rather than deleted, so it can be started again."
        ),
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Execute directly through the Daytona SDK adapter.",
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
    if not _has_daytona_auth():
        print("Daytona auth not found.")
        print("Set DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID.")
        print("Create a key at https://app.daytona.io.")
        raise SystemExit(1)

    if args.gpu:
        capacity = "spot (preemptible)" if args.spot else "on-demand"
        print(f"Launching daytona sandbox REPL with GPU: {args.gpu} on {capacity}")
    elif args.spot:
        print("--spot needs --gpu: preemptible capacity is GPU capacity.")
        raise SystemExit(1)
    else:
        print("Launching daytona sandbox REPL without GPU.")

    try:
        with Sandbox.create(
            variant="daytona",
            timeout=60,
            gpu=args.gpu,
            spot=args.spot,
            delete_on_stop=not args.keep,
            examples=_examples(args.gpu),
        ) as provider, provider_ingress_execution(
            provider, direct=args.direct
        ) as sandbox:
            print(f"Sandbox: {provider.sandbox_id}")
            run_repl(sandbox)
            if args.keep:
                print(
                    "Kept: the sandbox is stopped, not deleted — "
                    "`code-sandboxes list -v daytona` shows it."
                )
    except Exception as exc:
        print("daytona REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
