# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Modal sandbox implementation.

The implementation lives in :mod:`.modal`; this package re-exports it.
"""

from .modal import (
    DEFAULT_APP_NAME,
    DEFAULT_MODAL_PYTHON_VERSION,
    ModalSandbox,
    logger,
)

__all__ = [
    "DEFAULT_APP_NAME",
    "DEFAULT_MODAL_PYTHON_VERSION",
    "ModalSandbox",
    "logger",
]
