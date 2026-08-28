# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""One vocabulary for a sandbox's life, whoever is speaking it.

The same nouns had two spellings. This library said `Sandbox.create()`,
`.start()`, `.stop()`, `.run_code()`; the Runtimes API said `POST /runtimes`,
`DELETE /runtimes/{pod}`, `POST /runtimes/{pod}/pause`. Same concepts, different
words — so every caller that wanted to work with both wrote a translation layer,
and every translation layer disagreed slightly with the next.

The answer is convergence rather than a facade: one set of verbs, spelled the
same way wherever it appears. A facade would have frozen the divergence behind
something nobody wants to own.

    create · start · stop · pause · resume · list · get · snapshot · execute

This module states that vocabulary and nothing else. It is a `Protocol`, so a
class conforms by having the methods rather than by inheriting anything, and the
Runtimes client conforms by being written to it — which is what "converged"
means here.

Not every provider can do every verb. A sandbox that cannot pause says so
through `supports()` rather than raising when someone tries: the caller asking
"can I?" deserves an answer before it commits.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from code_sandboxes.exceptions import SandboxError


class SandboxOperationNotSupported(SandboxError):
    """The provider cannot do this, and said so rather than half-doing it."""

    def __init__(self, operation: str, variant: str = "") -> None:
        where = f" by the {variant} sandbox" if variant else ""
        super().__init__(
            f"{operation!r} is not supported{where}. "
            "Ask `supports()` before committing to it."
        )
        self.operation = operation
        self.variant = variant


#: The converged vocabulary. Every entry is a verb a caller may use, and the
#: REST spelling it corresponds to on the Runtimes API.
LIFECYCLE_OPERATIONS: dict[str, str] = {
    "create": "POST /runtimes",
    "start": "POST /runtimes (implied) — begin a created sandbox",
    "stop": "DELETE /runtimes/{pod_name}",
    "pause": "POST /runtimes/{pod_name}/pause",
    "resume": "POST /runtimes/{pod_name}/resume",
    "list": "GET /runtimes",
    "get": "GET /runtimes/{pod_name}",
    "snapshot": "POST /runtimes/{pod_name}/snapshots",
    "execute": "run code in the sandbox",
}


@runtime_checkable
class SandboxLifecycle(Protocol):
    """What every sandbox can be asked, in one vocabulary.

    Implemented by `code_sandboxes.Sandbox` for in-process providers and by the
    Runtimes client for pods. A caller written against this works with either
    without knowing which answered.
    """

    def supports(self, operation: str) -> bool:
        """Whether this sandbox can do a verb at all.

        Asked *before* committing, because "pause this and come back tomorrow"
        is a plan a caller makes, not a call it makes.
        """
        ...

    def start(self, **kwargs: Any) -> None:
        """Bring the sandbox up. Idempotent: starting a running one is a no-op."""
        ...

    def stop(self, **kwargs: Any) -> None:
        """Take it down. Idempotent, and safe on one that never started."""
        ...

    def pause(self, **kwargs: Any) -> None:
        """Suspend it, keeping its state, so it can be resumed later."""
        ...

    def resume(self, **kwargs: Any) -> None:
        """Bring a paused sandbox back with its state intact."""
        ...

    def snapshot(self, name: str, **kwargs: Any) -> Any:
        """Capture its state under a name, without ending it."""
        ...

    def run_code(self, code: str, **kwargs: Any) -> Any:
        """Execute code and return what it produced."""
        ...


def unsupported(operation: str, variant: str = "") -> SandboxOperationNotSupported:
    """Build the refusal, so every provider refuses the same way."""
    return SandboxOperationNotSupported(operation, variant)
