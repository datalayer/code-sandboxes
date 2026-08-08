# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Typing protocols for sandbox clients."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ISandboxClient(Protocol):
    """Internal execution backend used by kernel-backed sandbox variants."""

    @property
    def id(self) -> str | None: ...

    @property
    def kernel_info(self) -> dict[str, Any] | None: ...

    def start(self, **kwargs: Any) -> None: ...

    def stop(self, shutdown_kernel: bool = True) -> None: ...

    def execute(self, code: str, **kwargs: Any) -> dict[str, Any]: ...

    def execute_interactive(self, code: str, **kwargs: Any) -> dict[str, Any]: ...

    def get_variable(self, name: str) -> Any: ...

    def set_variable(self, name: str, value: Any) -> None: ...

    def interrupt(self) -> bool: ...
