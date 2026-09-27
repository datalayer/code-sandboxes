# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Google Colab sandbox implementation.

The implementation lives in :mod:`.google_colab`; this package re-exports it.
"""

from .google_colab import (
    GoogleColabSandbox,
    logger,
)

__all__ = [
    "GoogleColabSandbox",
    "logger",
]
