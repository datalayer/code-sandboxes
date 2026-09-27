# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The kernel helper is real code, and the shipped text is that code.

`reactive_kernel.py` is imported here — typed, coverable — and `reactive`
ships its file text to the kernel. The tests hold the two ends together and
cover the graph behaviours the drivers rely on: replacement, conflicts,
cycles, plans, and the wire encoding.
"""

from __future__ import annotations

import pytest

marimo = pytest.importorskip("marimo")

from code_sandboxes.sandboxes.marimo import reactive, reactive_kernel  # noqa: E402


@pytest.fixture
def graph():
    return reactive_kernel._MarimoReactive()


def test_the_shipped_source_is_the_modules_own_file():
    from importlib import resources

    text = (
        resources.files("code_sandboxes.sandboxes.marimo")
        .joinpath("reactive_kernel.py")
        .read_text(encoding="utf-8")
    )
    assert reactive.KERNEL_HELPER_SOURCE == text
    assert "class _MarimoReactive" in reactive.KERNEL_HELPER_SOURCE


def test_the_markers_agree_between_the_two_sides():
    assert reactive.ANSWER_MARKER == reactive_kernel.ANSWER_MARKER
    assert reactive.HELPER_NAME == "__marimo_reactive__"
    assert hasattr(reactive_kernel, reactive.HELPER_NAME)


def test_the_module_and_the_executed_text_are_the_same_code(graph):
    executed = reactive.local_helper()
    assert type(executed).__name__ == type(graph).__name__
    for helper in (graph, executed):
        answer = helper.register("a", "x = 1")
        assert answer == {"cell": "a", "defs": ["x"], "refs": [], "conflicts": [], "cycle": False}


def test_reexecuting_the_source_keeps_the_graph():
    """A second driver on the same kernel must not wipe the first one's cells."""
    namespace: dict = {}
    exec(reactive.KERNEL_HELPER_SOURCE, namespace)  # noqa: S102 - our own module text
    namespace[reactive.HELPER_NAME].register("a", "x = 1")
    exec(reactive.KERNEL_HELPER_SOURCE, namespace)  # noqa: S102
    assert namespace[reactive.HELPER_NAME].codes() == {"a": "x = 1"}


def test_registering_a_name_again_replaces_the_cell(graph):
    graph.register("a", "x = 1")
    answer = graph.register("a", "x = 2\ny = x")
    assert answer["defs"] == ["x", "y"]
    assert graph.codes() == {"a": "x = 2\ny = x"}


def test_two_definers_are_a_conflict_and_both_parent_the_reader(graph):
    graph.register("a", "x = 1")
    conflicted = graph.register("b", "x = 2")
    graph.register("c", "print(x)")
    assert conflicted["conflicts"] == ["x"]
    snapshot = graph.snapshot()
    assert snapshot["conflicts"] == ["x"]
    assert snapshot["cells"]["c"]["parents"] == ["a", "b"]
    assert graph.plan("a") == ["c"]
    assert graph.plan("b") == ["c"]


def test_a_cycle_is_reported_on_the_closing_cell_and_in_the_snapshot(graph):
    first = graph.register("d", "p = q")
    closing = graph.register("e", "q = p")
    assert first["cycle"] is False
    assert closing["cycle"] is True
    assert graph.snapshot()["cycles"] == ["d", "e"]


def test_a_chain_plans_in_dependency_order(graph):
    graph.register("c", "z = y + 1")
    graph.register("b", "y = x + 1")
    graph.register("a", "x = 1")
    assert graph.plan("a") == ["b", "c"]
    assert graph.plan_all() == ["a", "b", "c"]
    assert graph.plan("c") == []


def test_a_removed_cell_leaves_the_plan_and_the_snapshot(graph):
    graph.register("a", "x = 1")
    graph.register("b", "print(x)")
    assert graph.remove("b") == {"cell": "b", "removed": True}
    assert graph.plan("a") == []
    assert "b" not in graph.snapshot()["cells"]
    # Removing what is not there answers the same way, so a driver retrying
    # after a lost reply is not punished.
    assert graph.remove("b") == {"cell": "b", "removed": True}


def test_a_plan_for_an_unregistered_cell_is_empty(graph):
    assert graph.plan("nope") == []


def test_the_wire_round_trip(graph, capsys):
    graph.register("a", "x = 1")
    graph.answer("plan", "a")
    lines = capsys.readouterr().out.splitlines()
    assert reactive.decode_answer(lines) == []
    graph.answer("codes")
    assert reactive.decode_answer(capsys.readouterr().out.splitlines()) == {"a": "x = 1"}


def test_decode_answer_reads_the_last_answer_and_ignores_noise(graph, capsys):
    graph.answer("codes")
    graph.register("a", "x = 1")
    graph.answer("codes")
    lines = ["some print output", *capsys.readouterr().out.splitlines(), "trailing noise"]
    assert reactive.decode_answer(lines) == {"a": "x = 1"}


def test_no_answer_is_a_lookup_error():
    with pytest.raises(LookupError):
        reactive.decode_answer(["nothing here"])


def test_question_quotes_its_arguments():
    assert reactive.question("plan", "a") == "__marimo_reactive__.answer('plan', 'a')"
    assert reactive.question("codes") == "__marimo_reactive__.answer('codes')"
    tricky = reactive.question("register", "a", "x = 'quoted'")
    namespace: dict = {}
    exec(reactive.KERNEL_HELPER_SOURCE, namespace)  # noqa: S102
    exec(tricky, namespace)  # noqa: S102 - the question we built one line up
