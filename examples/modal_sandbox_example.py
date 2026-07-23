# Copyright (c) 2025-2026 Datalayer, Inc.
# BSD 3-Clause License

"""Example: modal sandbox (cloud container execution).

Run with:
  python examples/modal_sandbox_example.py

Auth options:
- `modal token new` (writes ~/.modal.toml), or
- set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.
"""

import os
from pathlib import Path

from code_sandboxes import Sandbox


def _has_modal_auth() -> bool:
    if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
        return True
    return Path.home().joinpath(".modal.toml").exists()


def main() -> None:
    if not _has_modal_auth():
        print("Modal auth not found.")
        print("Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET or run: modal token new")
        return

    try:
        with Sandbox.create(
            variant="modal",
            timeout=60,
            pip_packages=["numpy"],
        ) as sandbox:
            result = sandbox.run_code("import numpy as np; print(int(np.arange(5).sum()))")
            print("stdout:", result.stdout.strip())

            error_result = sandbox.run_code("raise RuntimeError('modal failure example')")
            if error_result.code_error:
                print(
                    "code_error:",
                    f"{error_result.code_error.name}: {error_result.code_error.value}",
                )
    except Exception as exc:
        print("modal example failed:", exc)


if __name__ == "__main__":
    main()
