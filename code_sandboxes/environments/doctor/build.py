# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Build ``datalayer-sandbox``, the doctor, as a single-file zipapp.

    python -m code_sandboxes.environments.doctor.build dist/datalayer-sandbox

The archive holds the checker alone — no package, no dependencies — and runs
with the image's ``python3``. Images install it at
``/opt/datalayer/bin/datalayer-sandbox``.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipapp
from pathlib import Path

__all__ = ["build_zipapp"]

_SOURCE = Path(__file__).with_name("datalayer_sandbox.py")


def build_zipapp(target: str | Path, *, interpreter: str = "/usr/bin/env python3") -> Path:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        shutil.copy(_SOURCE, Path(staging) / _SOURCE.name)
        zipapp.create_archive(
            staging, target, interpreter=interpreter, main="datalayer_sandbox:run"
        )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m code_sandboxes.environments.doctor.build")
    parser.add_argument("target", nargs="?", default="dist/datalayer-sandbox")
    arguments = parser.parse_args(argv)
    print(build_zipapp(arguments.target))  # noqa: T201 - a command's own output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
