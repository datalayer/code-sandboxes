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
            run_repl(sandbox)
    except Exception as exc:
        print("coreweave REPL failed:", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
