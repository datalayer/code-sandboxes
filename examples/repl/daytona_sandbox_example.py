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

from repl_common import run_repl

from code_sandboxes import Sandbox


def _has_daytona_auth() -> bool:
    if os.environ.get("DAYTONA_API_KEY"):
        return True
    return bool(os.environ.get("DAYTONA_JWT_TOKEN") and os.environ.get("DAYTONA_ORGANIZATION_ID"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daytona sandbox REPL example.")
    parser.add_argument(
        "--gpu",
        default=os.environ.get("DAYTONA_GPU"),
        help="Optional GPU (for example: H100, H200, RTX-4090).",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help=(
            "Leave the sandbox in the organization when the REPL closes, "
            "stopped rather than deleted, so it can be started again."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not _has_daytona_auth():
        print("Daytona auth not found.")
        print("Set DAYTONA_API_KEY, or DAYTONA_JWT_TOKEN with DAYTONA_ORGANIZATION_ID.")
        print("Create a key at https://app.daytona.io.")
        raise SystemExit(1)

    if args.gpu:
        print(f"Launching daytona sandbox REPL with GPU: {args.gpu}")
    else:
        print("Launching daytona sandbox REPL without GPU.")

    try:
        with Sandbox.create(
            variant="daytona",
            timeout=60,
            gpu=args.gpu,
            delete_on_stop=not args.keep,
        ) as sandbox:
            print(f"Sandbox: {sandbox.sandbox_id}")
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
