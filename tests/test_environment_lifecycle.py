# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The version lifecycle: every legal move allowed, every other one refused."""

from __future__ import annotations

import itertools

import pytest

from code_sandboxes.environments.lifecycle import (
    TERMINAL_STATES,
    TRANSITIONS,
    InvalidTransitionError,
    VersionState,
    accepts_new_sandboxes,
    build_outcome,
    can_promote,
    can_transition,
    is_editable,
    is_terminal,
    next_states,
    promotion_needs_acknowledgement,
    transition,
    unavailable_variants,
)

S = VersionState

#: PLAN_ENV.md §2.1, edge by edge.
EDGES = {
    (S.DRAFT, S.RESOLVING),
    (S.RESOLVING, S.RESOLVED),
    (S.RESOLVING, S.FAILED),
    (S.RESOLVED, S.BUILDING),
    (S.BUILDING, S.VALIDATING),
    (S.BUILDING, S.FAILED),
    (S.VALIDATING, S.READY),
    (S.VALIDATING, S.PARTIALLY_READY),
    (S.VALIDATING, S.FAILED),
    (S.READY, S.DEPRECATED),
    (S.PARTIALLY_READY, S.READY),
    (S.PARTIALLY_READY, S.DEPRECATED),
}


def test_the_transitions_are_exactly_the_diagram() -> None:
    assert {(item.source, item.target) for item in TRANSITIONS} == EDGES
    assert all(item.event for item in TRANSITIONS)


@pytest.mark.parametrize(("source", "target"), sorted(EDGES, key=str))
def test_every_legal_move_is_allowed(source: VersionState, target: VersionState) -> None:
    assert can_transition(source, target)
    assert transition(source.value, target.value) is target


ILLEGAL = sorted(set(itertools.product(VersionState, VersionState)) - EDGES, key=str)


@pytest.mark.parametrize(("source", "target"), ILLEGAL)
def test_every_other_move_is_refused(source: VersionState, target: VersionState) -> None:
    assert not can_transition(source, target)
    with pytest.raises(InvalidTransitionError):
        transition(source, target)


def test_a_refusal_names_the_moves_that_are_legal() -> None:
    with pytest.raises(InvalidTransitionError, match="it can become failed, resolved"):
        transition("resolving", "ready")
    with pytest.raises(InvalidTransitionError, match="cannot change state"):
        transition("failed", "draft")


def test_nothing_leaves_a_terminal_state() -> None:
    assert TERMINAL_STATES == {S.FAILED, S.DEPRECATED}
    for state in VersionState:
        assert is_terminal(state) == (not next_states(state))


def test_only_a_draft_is_edited() -> None:
    assert [state for state in VersionState if is_editable(state)] == [S.DRAFT]


def test_promotion_and_launch_take_ready_and_partially_ready_versions_only() -> None:
    assert {state for state in VersionState if can_promote(state)} == {S.READY, S.PARTIALLY_READY}
    assert {state for state in VersionState if accepts_new_sandboxes(state)} == {
        S.READY,
        S.PARTIALLY_READY,
    }
    assert promotion_needs_acknowledgement(S.PARTIALLY_READY)
    assert not promotion_needs_acknowledgement(S.READY)
    assert not accepts_new_sandboxes(S.DEPRECATED)


def test_an_unknown_state_names_the_states() -> None:
    with pytest.raises(ValueError, match="partially_ready"):
        is_terminal("almost_ready")


@pytest.mark.parametrize(
    ("required", "optional", "outcome"),
    [
        ({"datalayer": True}, {}, S.READY),
        ({"datalayer": True}, {"e2b": True, "modal": True}, S.READY),
        ({"datalayer": True}, {"e2b": True, "modal": False}, S.PARTIALLY_READY),
        ({"datalayer": False}, {"e2b": True}, S.FAILED),
        ({"datalayer": True, "daytona": False}, {"modal": False}, S.FAILED),
    ],
)
def test_the_outcome_follows_required_and_optional_variants(
    required: dict[str, bool], optional: dict[str, bool], outcome: VersionState
) -> None:
    assert build_outcome(required, optional) is outcome


def test_a_build_needs_a_required_variant_and_no_variant_twice() -> None:
    with pytest.raises(ValueError, match="at least one required"):
        build_outcome({}, {"modal": True})
    with pytest.raises(ValueError, match="not both"):
        build_outcome({"modal": True}, {"modal": True})


def test_the_unavailable_variants_are_the_failed_optional_ones() -> None:
    assert unavailable_variants({"modal": False, "e2b": True, "daytona": False}) == [
        "daytona",
        "modal",
    ]
