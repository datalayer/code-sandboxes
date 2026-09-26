# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Marimo's cells, driven through a Jupyter-shaped `execute` on any client.

A fake client executes in one namespace and answers Jupyter-shaped replies,
the way `CodeSandboxClient.execute` does for every variant: the driver only
ever sees replies, so this is the path the MCP server's toolset takes on a
Datalayer runtime.
"""

from __future__ import annotations

import contextlib
import io
import traceback

import pytest

marimo = pytest.importorskip("marimo")

from code_sandboxes.marimo_cells import MarimoCells  # noqa: E402


class ReplyingClient:
    """`execute(code)` → a Jupyter-shaped reply, from code run right here."""

    def __init__(self, namespace=None):
        self.namespace = namespace if namespace is not None else {"__name__": "__main__"}
        self.executed: list[str] = []

    def execute(self, code, timeout=None, **kwargs):
        self.executed.append(code)
        out, err = io.StringIO(), io.StringIO()
        error = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(compile(code, "<cell>", "exec"), self.namespace)  # noqa: S102 - the test's own code
            except BaseException as raised:
                error = raised
        outputs = []
        if out.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": out.getvalue()})
        if err.getvalue():
            outputs.append({"output_type": "stream", "name": "stderr", "text": err.getvalue()})
        if error is not None:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "traceback": traceback.format_exception(error),
                }
            )
        return {
            "status": "error" if error else "ok",
            "execution_count": len(self.executed),
            "outputs": outputs,
        }


@pytest.fixture
def cells():
    return MarimoCells(ReplyingClient(), install_marimo=False)


def test_the_helper_is_installed_once_by_execute(cells):
    cells.install()
    cells.install()
    assert sum("class _MarimoReactive" in code for code in cells._client.executed) == 1


def test_running_a_cell_re_runs_what_depends_on_it(cells):
    cells.run_cell("a", "x = 1")
    cells.run_cell("b", "print(x + 1)")
    run = cells.run_cell("a", "x = 10")
    assert run.ok
    assert [reaction.cell_id for reaction in run.reactions] == ["b"]
    assert run.reactions[0].outputs == [{"output_type": "stream", "name": "stdout", "text": "11\n"}]


def test_a_failing_reaction_stops_the_chain_and_is_reported(cells):
    cells.run_cell("a", "x = 1")
    cells.run_cell("b", "assert x < 5")
    cells.run_cell("c", "print('c reads', x)")
    run = cells.run_cell("a", "x = 10")
    assert run.ok
    assert [(r.cell_id, r.ok) for r in run.reactions] == [("b", False)]
    assert run.to_dict()["reactions"][0]["status"] == "error"


def test_a_cell_that_does_not_parse_never_runs(cells):
    run = cells.run_cell("a", "def broken(:")
    assert not run.ok
    assert "SyntaxError" in run.registration["error"]
    assert run.cell.outputs[0]["ename"] == "SyntaxError"
    assert "a" not in cells.cells


def test_the_graph_and_the_plan_are_readable(cells):
    cells.register_cell("a", "x = 1")
    cells.register_cell("b", "y = x")
    cells.register_cell("c", "z = y")
    assert cells.plan("a") == ["b", "c"]
    graph = cells.graph()
    assert graph["cells"]["b"]["parents"] == ["a"] and graph["cells"]["b"]["children"] == ["c"]
    cells.remove_cell("c")
    assert cells.plan("a") == ["b"]


def test_run_code_is_a_cell_of_its_own(cells):
    cells.run_code("base = 2")
    cells.run_code("print(base * 3)")
    run = cells.run_code("base = 5")
    assert run.cell.cell_id == "cell-3"
    assert run.reactions[0].outputs[0]["text"] == "15\n"


def test_a_second_driver_on_the_same_kernel_reads_the_cells_back():
    client = ReplyingClient()
    first = MarimoCells(client, install_marimo=False)
    first.run_cell("a", "x = 1")
    first.run_cell("b", "print(x)")
    second = MarimoCells(client, install_marimo=False)
    assert second.cells == {"a": "x = 1", "b": "print(x)"}
    run = second.run_cell("a", "x = 7")
    assert run.reactions[0].outputs[0]["text"] == "7\n"


def test_a_missing_marimo_is_refused_when_installing_is_off():
    namespace = {"__name__": "__main__", "__builtins__": {**__builtins__, "__import__": _no_marimo}}
    with pytest.raises(RuntimeError, match="install_marimo is False"):
        MarimoCells(ReplyingClient(namespace), install_marimo=False).install()


def _no_marimo(name, *args, **kwargs):
    if name == "marimo":
        raise ImportError("no marimo here")
    return __import__(name, *args, **kwargs)
