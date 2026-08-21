# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""`code-sandboxes exec`: one snippet, and a status a shell can read.

Run against the `eval` variant, which needs nothing and runs in this process,
so these are the real command end to end rather than a mock of it.
"""

from __future__ import annotations

from typer.testing import CliRunner

from code_sandboxes import cli as sandbox_cli

runner = CliRunner()


def _exec(*args: str, **kwargs) -> object:
    return runner.invoke(sandbox_cli.app, ["exec", "--variant", "eval", *args], **kwargs)


def test_code_given_as_an_argument_runs():
    result = _exec("print(40 + 2)")

    assert result.exit_code == 0
    assert "42" in result.stdout


def test_the_value_of_the_last_expression_comes_back():
    result = _exec("x = 40\nx + 2")

    assert result.exit_code == 0
    assert "<<< result: 42" in result.stdout


def test_code_can_come_from_a_file(tmp_path):
    snippet = tmp_path / "snippet.py"
    snippet.write_text("print('from a file')\n", encoding="utf-8")

    result = _exec("--file", str(snippet))

    assert result.exit_code == 0
    assert "from a file" in result.stdout


def test_code_can_come_from_stdin():
    result = _exec(input="print('from stdin')\n")

    assert result.exit_code == 0
    assert "from stdin" in result.stdout


def test_giving_it_both_ways_is_refused(tmp_path):
    snippet = tmp_path / "snippet.py"
    snippet.write_text("print('file')\n", encoding="utf-8")

    result = _exec("print('argument')", "--file", str(snippet))

    assert result.exit_code != 0


def test_the_status_is_the_code_s_own():
    """What makes this composable in a shell: a snippet that raised fails."""
    assert _exec("print('fine')").exit_code == 0
    assert _exec("raise ValueError('boom')").exit_code == 1


def test_a_failing_snippet_still_shows_its_error():
    result = _exec("raise ValueError('boom')")

    assert "ValueError: boom" in result.stdout


def test_quiet_prints_only_what_the_code_produced():
    result = _exec("--quiet", "print(40 + 2)")

    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


def test_the_sandbox_is_terminated_either_way(monkeypatch):
    stopped: list[bool] = []
    real_stop = sandbox_cli.Sandbox.stop

    def spy(self):
        stopped.append(True)
        return real_stop(self)

    monkeypatch.setattr("code_sandboxes.eval_sandbox.EvalSandbox.stop", spy)

    _exec("print('ok')")
    _exec("raise ValueError('boom')")

    assert len(stopped) == 2


def test_the_variant_and_its_settings_are_forwarded(monkeypatch):
    captured: dict = {}

    def fake_create(*_args, **kwargs):
        captured.update(kwargs)
        from code_sandboxes.eval_sandbox import EvalSandbox

        return EvalSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(fake_create))

    result = runner.invoke(
        sandbox_cli.app,
        ["exec", "--variant", "datalayer", "--gpu", "T4", "--timeout", "12", "1 + 1"],
    )

    assert result.exit_code == 0
    assert captured["variant"] == "datalayer"
    assert captured["gpu"] == "T4"
    assert captured["timeout"] == 12.0


def test_an_unknown_variant_is_refused_before_anything_starts():
    result = runner.invoke(sandbox_cli.app, ["exec", "--variant", "nonesuch", "1 + 1"])

    assert result.exit_code != 0
