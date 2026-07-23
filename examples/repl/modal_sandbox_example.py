# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""REPL example: modal sandbox (cloud container execution)."""

import argparse
import os
from pathlib import Path

from code_sandboxes import Sandbox

from repl_common import run_repl


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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if not _has_modal_auth():
        print("Modal auth not found.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET or run: modal token new")
        return

    if args.gpu:
        print(f"Launching modal sandbox REPL with GPU flavor: {args.gpu}")
    else:
        print("Launching modal sandbox REPL without GPU.")

    try:
        with Sandbox.create(
            variant="modal",
            timeout=60,
            gpu=args.gpu,
        ) as sandbox:
            run_repl(sandbox)
    except Exception as exc:
        print("modal REPL failed:", exc)


if __name__ == "__main__":
    main()
