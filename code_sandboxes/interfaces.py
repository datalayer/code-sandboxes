# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typing protocols for sandbox clients."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

try:
    from jupyter_kernel_client.interfaces import IKernelClient
except Exception:  # pragma: no cover - fallback for optional dependency contexts
    class IKernelClient(Protocol):
        """Fallback protocol when jupyter-kernel-client is unavailable."""


@runtime_checkable
class ISandboxClient(IKernelClient, Protocol):
    """Kernel client protocol exposed by sandbox variants.

    This currently matches ``IKernelClient`` exactly and acts as an extension
    point for sandbox-specific client capabilities.
    """
