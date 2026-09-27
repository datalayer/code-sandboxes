# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""E2B sandbox implementation.

The implementation lives in :mod:`.e2b`; this package re-exports it.
"""

from .e2b import (
    CREATED_BY_LABEL,
    DEFAULT_TEMPLATE,
    E2BSandbox,
    logger,
)

__all__ = [
    "CREATED_BY_LABEL",
    "DEFAULT_TEMPLATE",
    "E2BSandbox",
    "logger",
]
