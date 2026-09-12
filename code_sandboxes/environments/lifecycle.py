# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The life of an Environment Version, as data.

A version is editable only while it is a draft. Once submitted it resolves
into a lock, builds one artifact per variant from that lock, validates them,
and ends ``ready``, ``partially_ready`` or ``failed``; a ready version is
later ``deprecated``. Every legal move is one :class:`Transition` below, and
:func:`transition` refuses the rest — a state written to a record without
going through it is how two services end up disagreeing about what a version
is.

Backfilling a variant onto a ``ready`` or ``partially_ready`` version builds
from the stored lock without moving the version; the only move a backfill
makes is ``partially_ready`` to ``ready``, when the last unavailable variant
arrives.

One move is made only by asking for it: retrying a build that failed under a
retryable code reopens its version, ``failed`` back to ``building`` (the
owner's decision of 2026-09-11). A version never walks it on its own — without
the :data:`RETRY` event, and without a failure the taxonomy calls retryable,
``failed`` is where a version stops and the answer is a new version.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "LAUNCHABLE_STATES",
    "PROMOTABLE_STATES",
    "RETRY",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "InvalidTransitionError",
    "Transition",
    "VersionState",
    "accepts_new_sandboxes",
    "build_outcome",
    "can_promote",
    "can_transition",
    "is_editable",
    "is_terminal",
    "next_states",
    "promotion_needs_acknowledgement",
    "reopened_by_retry",
    "transition",
    "unavailable_variants",
]


class VersionState(str, Enum):
    DRAFT = "draft"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"
    PARTIALLY_READY = "partially_ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


#: The event that reopens a failed version: a retry of a build whose failure
#: the taxonomy calls retryable. The only event a move is asked for by name.
RETRY = "retry"


@dataclass(frozen=True)
class Transition:
    """One legal move, and what makes it."""

    source: VersionState
    target: VersionState
    event: str
    #: True when the move is made only by naming its event, and never on its
    #: own: nothing reaches this target by walking the machine.
    on_request: bool = False


S = VersionState

TRANSITIONS: tuple[Transition, ...] = (
    Transition(S.DRAFT, S.RESOLVING, "submit"),
    Transition(S.RESOLVING, S.RESOLVED, "lock produced"),
    Transition(S.RESOLVING, S.FAILED, "resolution conflict"),
    Transition(S.RESOLVED, S.BUILDING, "build requested"),
    Transition(S.BUILDING, S.VALIDATING, "all builds terminal"),
    Transition(S.BUILDING, S.FAILED, "required variant failed"),
    Transition(S.VALIDATING, S.READY, "all required variants pass the smoke test"),
    Transition(S.VALIDATING, S.PARTIALLY_READY, "required variants pass, an optional one failed"),
    Transition(S.VALIDATING, S.FAILED, "required variant failed validation"),
    Transition(S.READY, S.DEPRECATED, "deprecate"),
    Transition(S.PARTIALLY_READY, S.READY, "backfill build succeeds"),
    Transition(S.PARTIALLY_READY, S.DEPRECATED, "deprecate"),
    Transition(S.FAILED, S.BUILDING, RETRY, on_request=True),
)

#: Nothing leaves these on its own. A retry of a retryable failure is the one
#: move out, and it is made only by asking for it by name.
TERMINAL_STATES: frozenset[VersionState] = frozenset({S.FAILED, S.DEPRECATED})

#: What may become an Environment's promoted version.
PROMOTABLE_STATES: frozenset[VersionState] = frozenset({S.READY, S.PARTIALLY_READY})

#: What a new sandbox may start from. A deprecated version keeps the
#: sandboxes already running on it and takes no new ones.
LAUNCHABLE_STATES: frozenset[VersionState] = PROMOTABLE_STATES


class InvalidTransitionError(ValueError):
    """A move the lifecycle does not allow."""


def _state(value: VersionState | str) -> VersionState:
    try:
        return VersionState(value)
    except ValueError:
        names = ", ".join(state.value for state in VersionState)
        raise ValueError(f"no version state {value!r}; the states are {names}") from None


def next_states(state: VersionState | str, *, event: str | None = None) -> frozenset[VersionState]:
    """Where a version in this state may go on its own, and where the event takes it."""
    source = _state(state)
    return frozenset(
        item.target
        for item in TRANSITIONS
        if item.source is source and (not item.on_request or item.event == event)
    )


def _move(source: VersionState, target: VersionState, event: str | None) -> Transition | None:
    """The transition from one state to the other, when this event makes it."""
    for item in TRANSITIONS:
        if item.source is not source or item.target is not target:
            continue
        if not item.on_request or item.event == event:
            return item
    return None


def can_transition(
    source: VersionState | str,
    target: VersionState | str,
    *,
    event: str | None = None,
    retryable: bool | None = None,
) -> bool:
    """Whether this move is legal, under the event asked for.

    The retry that reopens a failed version also needs its failure to be
    retryable (:func:`errors.retryable_failure` decides that).
    """
    move = _move(_state(source), _state(target), event)
    return move is not None and (move.event != RETRY or bool(retryable))


def reopened_by_retry(state: VersionState | str, *, retryable: bool) -> bool:
    """Whether retrying a build of a version in this state moves it back to `building`."""
    return can_transition(state, S.BUILDING, event=RETRY, retryable=retryable)


def transition(
    source: VersionState | str,
    target: VersionState | str,
    *,
    event: str | None = None,
    retryable: bool | None = None,
) -> VersionState:
    """The target, when the move is legal; otherwise an error naming the legal ones."""
    origin, destination = _state(source), _state(target)
    move = _move(origin, destination, event)
    if move is not None:
        if move.event == RETRY and not retryable:
            raise InvalidTransitionError(
                f"a {origin.value} version is reopened only by a retry of a retryable failure; "
                "a failure that is not retryable needs a new version"
            )
        return destination
    legal = sorted(state.value for state in next_states(origin, event=event))
    if not legal:
        raise InvalidTransitionError(f"a {origin.value} version cannot change state")
    raise InvalidTransitionError(
        f"a {origin.value} version cannot become {destination.value}; it can become "
        + ", ".join(legal)
    )


def is_terminal(state: VersionState | str) -> bool:
    return _state(state) in TERMINAL_STATES


def is_editable(state: VersionState | str) -> bool:
    """Only a draft is edited; editing anything else makes a new draft."""
    return _state(state) is S.DRAFT


def can_promote(state: VersionState | str) -> bool:
    return _state(state) in PROMOTABLE_STATES


def promotion_needs_acknowledgement(state: VersionState | str) -> bool:
    """A partially ready version is promoted only with its gaps acknowledged."""
    return _state(state) is S.PARTIALLY_READY


def accepts_new_sandboxes(state: VersionState | str) -> bool:
    return _state(state) in LAUNCHABLE_STATES


def build_outcome(required: Mapping[str, bool], optional: Mapping[str, bool]) -> VersionState:
    """Where a version lands once every variant's build and validation are terminal.

    ``required`` and ``optional`` map each variant to whether it passed. Any
    failed required variant fails the version; with every required variant
    passing, one failed optional variant makes it partially ready.
    """
    if not required:
        raise ValueError("a build needs at least one required variant")
    both = set(required) & set(optional)
    if both:
        raise ValueError(f"a variant is either required or optional, not both: {sorted(both)}")
    if not all(required.values()):
        return S.FAILED
    if all(optional.values()):
        return S.READY
    return S.PARTIALLY_READY


def unavailable_variants(optional: Mapping[str, bool]) -> list[str]:
    """The optional variants a partially ready version lacks, as promotion names them."""
    return sorted(variant for variant, passed in optional.items() if not passed)
