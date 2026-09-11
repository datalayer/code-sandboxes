# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Generate what is generated from the Environment models.

python -m code_sandboxes.environments schema --write [PATH]
python -m code_sandboxes.environments schema --check [PATH]
python -m code_sandboxes.environments contract --markdown docs/docs/environments/contract.mdx
python -m code_sandboxes.environments contract --check docs/docs/environments/contract.mdx
"""

from __future__ import annotations

import sys

from . import contract, schema

_COMMANDS = {"schema": schema.main, "contract": contract.main}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in _COMMANDS:
        print(__doc__.strip(), file=sys.stderr)  # noqa: T201 - a command's own output
        return 2
    return _COMMANDS[arguments[0]](arguments[1:])


if __name__ == "__main__":
    raise SystemExit(main())
