# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Kaggle sandbox implementation.

The implementation lives in :mod:`.kaggle`; this package re-exports it.
"""

from .kaggle import (
    KaggleSandbox,
    logger,
)

__all__ = [
    "KaggleSandbox",
    "logger",
]
