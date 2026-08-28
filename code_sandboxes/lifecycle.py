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

    create · start · stop · pause · resume · list · get · update · snapshot · execute

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
    "stop": "DELETE /runtimes/{runtime_name}",
    "pause": "POST /runtimes/{runtime_name}/pause",
    "resume": "POST /runtimes/{runtime_name}/resume",
    "list": "GET /runtimes",
    "get": "GET /runtimes/{runtime_name}",
    "update": "PUT /runtimes/{runtime_name}",
    "snapshot": "POST /sandbox-snapshots",
    "execute": "run code in the sandbox",
}

#: The verbs that belong to one sandbox, and so appear on `SandboxLifecycle`.
#: The rest — `create`, `list`, `get`, `update` — belong to whoever manages a
#: collection of them, which is a different object with a different shape.
INSTANCE_OPERATIONS: frozenset[str] = frozenset(
    {"start", "stop", "pause", "resume", "snapshot", "execute"}
)

#: The verbs that belong to a manager of sandboxes rather than to one sandbox.
MANAGER_OPERATIONS: frozenset[str] = frozenset({"create", "list", "get", "update"})


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


#: The Runtimes API prefix, below whichever host serves it.
RUNTIMES_API_PREFIX = "api/runtimes/v1"


def _join(base_url: str, path: str) -> str:
    """Join a base to a path without doubling or dropping the slash between."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def runtimes_url(base_url: str) -> str:
    """Every runtime — the collection. `create` posts here, `list` gets here."""
    return _join(base_url, f"{RUNTIMES_API_PREFIX}/runtimes")


def runtime_url(base_url: str, runtime_name: str) -> str:
    """One runtime, by name.

    `get` reads it, `update` puts to it, and `stop` deletes it — whether the
    runtime is running or paused, which is why there is no second path for the
    paused case.
    """
    return f"{runtimes_url(base_url)}/{runtime_name}"


def runtime_pause_url(base_url: str, runtime_name: str) -> str:
    """Suspend one runtime, keeping its state."""
    return f"{runtime_url(base_url, runtime_name)}/pause"


def runtime_resume_url(base_url: str, runtime_name: str) -> str:
    """Bring one paused runtime back."""
    return f"{runtime_url(base_url, runtime_name)}/resume"


def sandbox_snapshots_url(base_url: str) -> str:
    """The snapshots collection.

    A snapshot outlives the runtime it came from, which is why it is its own
    resource rather than a sub-path.
    """
    return _join(base_url, f"{RUNTIMES_API_PREFIX}/sandbox-snapshots")


def sandbox_snapshot_url(base_url: str, snapshot_id: str) -> str:
    """One snapshot, by UID."""
    return f"{sandbox_snapshots_url(base_url)}/{snapshot_id}"


def runtime_checkpoints_url(base_url: str) -> str:
    """The checkpoint records behind pause and resume."""
    return _join(base_url, f"{RUNTIMES_API_PREFIX}/runtime-checkpoints")


@runtime_checkable
class SandboxManagerLifecycle(Protocol):
    """What a *collection* of sandboxes can be asked.

    `SandboxLifecycle` is one sandbox; this is whoever hands them out. The
    Runtimes client is both — it manages pods, and `handle()` narrows it to one.
    Keeping the two apart is why `Sandbox.create()` can stay a classmethod
    while a client's `create()` is an ordinary call.
    """

    def supports(self, operation: str) -> bool:
        """Whether this manager can do a verb at all."""
        ...

    def create(self, **kwargs: Any) -> Any:
        """Bring a new sandbox into being and return what identifies it."""
        ...

    def list(self, **kwargs: Any) -> Any:
        """Every sandbox this caller can see."""
        ...

    def get(self, identifier: str, **kwargs: Any) -> Any:
        """One sandbox, by whatever names it."""
        ...

    def update(self, identifier: str, **kwargs: Any) -> Any:
        """Change a sandbox in place, without restarting it."""
        ...


def unsupported(operation: str, variant: str = "") -> SandboxOperationNotSupported:
    """Build the refusal, so every provider refuses the same way."""
    return SandboxOperationNotSupported(operation, variant)
