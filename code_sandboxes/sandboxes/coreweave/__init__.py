# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""CoreWeave sandbox implementation.

The implementation lives in :mod:`.coreweave`; this package re-exports it.
"""

from .coreweave import (
    CREATED_BY_LABEL,
    DEFAULT_CONTAINER_IMAGE,
    CoreWeaveSandbox,
    logger,
)

__all__ = [
    "CREATED_BY_LABEL",
    "DEFAULT_CONTAINER_IMAGE",
    "CoreWeaveSandbox",
    "logger",
]
