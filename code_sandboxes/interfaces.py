# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typing protocols for sandbox clients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jupyter_kernel_client.interfaces import IJupyterKernelClient


@runtime_checkable
class ISandboxClient(IJupyterKernelClient, Protocol):
    """Kernel client protocol exposed by sandbox variants.

    This currently matches ``IJupyterKernelClient`` exactly and acts as an extension
    point for sandbox-specific client capabilities.
    """
