# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A Marimo sandbox re-runs what depends on what just ran.

Run against an in-process kernel: the fake client executes code in one
namespace and reports it as IOPub messages, so the whole path — the helper
installed by an execute request, the questions asked through stdout, the
reactions run through further requests — is the one a real kernel takes.
"""

from __future__ import annotations

import contextlib
import io
import sys
import traceback
import types

import pytest

marimo = pytest.importorskip("marimo")

from code_sandboxes import marimo_reactive  # noqa: E402
from code_sandboxes.marimo_sandbox import MarimoSandbox  # noqa: E402


class InProcessKernel:
    """The client a Marimo sandbox talks to, running code right here."""

    def __init__(self, server_url, token, kernel_id, client_kwargs=None):
        self.id = kernel_id or "in-process"
        self.namespace: dict = {"__name__": "__main__"}
        self.executed: list[str] = []
        self.kernel_info = {"language_info": {"name": "python"}}

    def start(self, path=None):
        return None

    def stop(self, shutdown_kernel=True):
        return None

    def execute_interactive(self, code, timeout=None, output_hook=None, **kwargs):
        self.executed.append(code)
        out, err = io.StringIO(), io.StringIO()
        error = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                exec(compile(code, "<cell>", "exec"), self.namespace)  # noqa: S102 - the test's own code
            except BaseException as raised:
                error = raised
        if output_hook:
            if out.getvalue():
                output_hook(
                    {"msg_type": "stream", "content": {"name": "stdout", "text": out.getvalue()}}
                )
            if err.getvalue():
                output_hook(
                    {"msg_type": "stream", "content": {"name": "stderr", "text": err.getvalue()}}
                )
            if error is not None:
                output_hook(
                    {
                        "msg_type": "error",
                        "content": {
                            "ename": type(error).__name__,
                            "evalue": str(error),
                            "traceback": traceback.format_exception(error),
                        },
                    }
                )
        return {
            "content": {"status": "error" if error else "ok", "execution_count": len(self.executed)}
        }

    def execute(self, code, **kwargs):
        return self.execute_interactive(code)

    def interrupt(self):
        return True

    def get_variable(self, name):
        return self.namespace[name]

    def set_variable(self, name, value):
        self.namespace[name] = value


@pytest.fixture
def sandbox(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=InProcessKernel),
    )
    box = MarimoSandbox(server_url="http://localhost:8888", install_marimo=False)
    monkeypatch.setattr(box, "_wait_for_server", lambda timeout=None: None)
    box.start()
    yield box
    box.stop()


def stdout(result) -> str:
    return "".join(
        message.line + ("\n" if message.terminated else "") for message in result.logs.stdout
    )


def test_the_helper_is_installed_by_one_execute_request(sandbox):
    kernel = sandbox.kernel_client
    assert marimo_reactive.HELPER_NAME in kernel.namespace
    assert sandbox.info.variant == "marimo"


def test_running_a_cell_re_runs_what_depends_on_it(sandbox):
    sandbox.run_cell("a", "x = 1")
    b = sandbox.run_cell("b", "y = x + 1\nprint(y)")
    assert stdout(b.result) == "2\n"
    assert b.reactions == []

    again = sandbox.run_cell("a", "x = 10")
    assert [reaction.cell_id for reaction in again.reactions] == ["b"]
    assert stdout(again.reactions[0].result) == "11\n"
    assert sandbox.kernel_client.namespace["y"] == 11


def test_reactions_run_in_dependency_order(sandbox):
    sandbox.run_cell("c", "print('c', b_)")  # declared first, runs last
    sandbox.run_cell("b", "b_ = a_ * 2")
    sandbox.run_cell("a", "a_ = 1")
    run = sandbox.run_cell("a", "a_ = 3")
    assert [reaction.cell_id for reaction in run.reactions] == ["b", "c"]
    assert stdout(run.reactions[1].result) == "c 6\n"


def test_a_failing_cell_stops_the_reaction(sandbox):
    sandbox.run_cell("a", "n = 2")
    sandbox.run_cell("b", "half = 10 / n")
    sandbox.run_cell("c", "print(half)")
    run = sandbox.run_cell("a", "n = 0")
    assert [reaction.cell_id for reaction in run.reactions] == ["b"]
    assert run.reactions[0].result.code_error is not None
    assert run.reactions[0].result.code_error.name == "ZeroDivisionError"


def test_a_cell_that_does_not_parse_is_refused_before_running(sandbox):
    run = sandbox.run_cell("bad", "def (")
    assert not run.ok
    assert run.result.code_error.name == "SyntaxError"
    assert "def (" not in sandbox.kernel_client.executed
    assert "bad" not in sandbox.cells


def test_the_graph_reports_names_conflicts_and_cycles(sandbox):
    first = sandbox.run_cell("a", "x = 1")
    assert first.registration["defs"] == ["x"] and first.registration["conflicts"] == []
    second = sandbox.run_cell("a2", "x = 2")
    assert second.registration["conflicts"] == ["x"]
    sandbox.register_cell("p", "q_ = p_ + 1")
    cycle = sandbox.register_cell("q", "p_ = q_ + 1")
    assert cycle["cycle"] is True
    graph = sandbox.graph()
    assert "x" in graph["conflicts"]
    assert set(graph["cycles"]) >= {"p", "q"}
    assert graph["cells"]["p"]["refs"] == ["p_"] and graph["cells"]["p"]["children"] == ["q"]


def test_removing_a_cell_ends_its_reactions(sandbox):
    sandbox.run_cell("a", "x = 1")
    sandbox.run_cell("b", "print(x)")
    sandbox.remove_cell("b")
    assert sandbox.run_cell("a", "x = 2").reactions == []
    assert "b" not in sandbox.cells


def test_run_code_is_a_cell_of_its_own_and_reacts(sandbox):
    sandbox.run_code("base = 5")
    sandbox.run_code("print(base * 2)")
    result = sandbox.run_code("base = 7")
    assert result.code_error is None
    assert getattr(result, "marimo_reactions", None) == ["cell-2"]
    assert sandbox.kernel_client.executed[-1] == "print(base * 2)"


def test_react_false_runs_the_cell_alone(sandbox):
    sandbox.run_cell("a", "x = 1")
    sandbox.run_cell("b", "print(x)")
    run = sandbox.run_cell("a", "x = 2", react=False)
    assert run.reactions == []


def test_the_local_helper_is_the_kernel_helper():
    helper = marimo_reactive.local_helper()
    helper.register("a", "x = 1")
    helper.register("b", "y = x")
    assert helper.plan("a") == ["b"]
    assert marimo_reactive.decode_answer(
        ["noise", "__MARIMO__" + __import__("base64").b64encode(b'{"k": 1}').decode()]
    ) == {"k": 1}
    with pytest.raises(LookupError):
        marimo_reactive.decode_answer(["nothing here"])


def test_the_variant_is_known():
    from code_sandboxes import Sandbox, SandboxVariant, normalize_variant

    assert normalize_variant(SandboxVariant.MARIMO) == "marimo"
    assert [env.name for env in Sandbox.list_environments("marimo")] == ["marimo"]
