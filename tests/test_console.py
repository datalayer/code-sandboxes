# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The one way a run is shown, and the one prompt that shows it repeatedly.

This machinery used to exist three times over — in the CLI, in the REPL
examples and in the exec examples — so what is worth pinning down is the
things those copies disagreed about: whether the value of the last expression
is shown, whether stderr is told apart, and what ends a session.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest
from rich.console import Console

from code_sandboxes.console import (
    EXIT_COMMANDS,
    repl_prompt,
    run_repl,
    show_and_run,
    show_result,
)
from code_sandboxes.models import (
    CodeError,
    ExecutionResult,
    Logs,
    OutputMessage,
    Result,
    SandboxInfo,
)


def _console(width: int = 200) -> tuple[Console, list[str]]:
    """A console that keeps what it was told, instead of a terminal."""
    console = Console(record=True, width=width, force_terminal=False, no_color=True)
    lines: list[str] = []
    return console, lines


def _rendered(console: Console) -> list[str]:
    return [line.rstrip() for line in console.export_text().splitlines()]


def _result(
    stdout: str = "",
    stderr: str = "",
    text: str | None = None,
    error: CodeError | None = None,
    ok: bool = True,
    failure: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        results=[Result(data={"text/plain": text}, is_main_result=True)] if text else [],
        logs=Logs(
            stdout=[OutputMessage(line=line) for line in stdout.splitlines()],
            stderr=[OutputMessage(line=line, error=True) for line in stderr.splitlines()],
        ),
        execution_ok=ok,
        execution_error=failure,
        code_error=error,
    )


class _FakeSandbox:
    """Enough of a sandbox to be shown: an identity and an answer."""

    def __init__(self, answers: dict[str, ExecutionResult] | None = None):
        self.info = SandboxInfo(id="abcdef123456", variant="eval", name=None)
        self.ran: list[str] = []
        self._answers = answers or {}

    def run_code(self, code: str, **_kwargs) -> ExecutionResult:
        self.ran.append(code)
        return self._answers.get(code, _result(stdout=f"ran {code}"))


# --- What a result looks like --------------------------------------------


def test_the_value_of_the_last_expression_is_shown():
    console, _ = _console()

    show_result(_result(text="42"), console=console)

    assert "<<< result: 42" in _rendered(console)


def test_a_value_that_is_only_the_stdout_again_is_not_shown_twice():
    console, _ = _console()

    show_result(_result(stdout="42", text="42"), console=console)

    rendered = _rendered(console)
    assert "<<< stdout:" in rendered
    assert not any(line.startswith("<<< result:") for line in rendered)


def test_stderr_is_told_apart_from_stdout():
    console, _ = _console()

    show_result(_result(stdout="fine", stderr="careful"), console=console)

    rendered = _rendered(console)
    assert rendered.index("<<< stdout:") < rendered.index("<<< stderr:")
    assert "    careful" in rendered


def test_a_raising_snippet_shows_its_error():
    console, _ = _console()

    show_result(_result(error=CodeError(name="ValueError", value="boom")), console=console)

    assert "<<< error: ValueError: boom" in _rendered(console)


def test_a_sandbox_that_could_not_run_it_says_so_differently():
    """The code failing and the sandbox failing are not the same event."""
    console, _ = _console()

    show_result(_result(ok=False, failure="the websocket closed"), console=console)

    rendered = _rendered(console)
    assert "<<< execution error: the websocket closed" in rendered
    assert not any(line.startswith("<<< error:") for line in rendered)


def test_a_snippet_that_produced_nothing_says_so():
    console, _ = _console()

    show_result(_result(), console=console)

    assert "<<< (no output)" in _rendered(console)


def test_the_prompt_form_drops_the_markers():
    """At a prompt the reader typed the line; the labels are noise."""
    console, _ = _console()

    show_result(_result(stdout="hello", text="42"), console=console, labelled=False)

    assert _rendered(console) == ["hello", "42"]


def test_output_that_looks_like_markup_is_shown_as_itself():
    """`rich` reads `[dim]` as a style; a sandbox printing it means the text."""
    console, _ = _console()

    show_result(_result(stdout="[dim]not a style[/dim]"), console=console)

    assert "    [dim]not a style[/dim]" in _rendered(console)


def test_show_and_run_prints_the_code_then_the_answer():
    console, _ = _console()
    sandbox = _FakeSandbox({"1 + 1": _result(text="2")})

    result = show_and_run(sandbox, "1 + 1", console=console)

    assert sandbox.ran == ["1 + 1"]
    assert result.text == "2"
    rendered = _rendered(console)
    assert rendered.index(">>> code:") < rendered.index("<<< result: 2")
    assert "    1 + 1" in rendered


def test_show_and_run_prints_callback_output_before_execution_returns():
    console, _ = _console()

    class StreamingSandbox(_FakeSandbox):
        def run_code(self, code: str, **kwargs) -> ExecutionResult:
            kwargs["on_stdout"](OutputMessage(line="first"))
            # The first event is visible while execution is still in progress.
            assert "first" in console.export_text(clear=False)
            kwargs["on_stdout"](OutputMessage(line="second"))
            return _result(stdout="first\nsecond")

    result = show_and_run(StreamingSandbox(), "stream()", console=console)

    assert result.stdout == "first\nsecond"
    rendered = _rendered(console)
    assert rendered.count("    first") == 1
    assert rendered.count("    second") == 1


# --- The prompt -----------------------------------------------------------


def test_the_prompt_names_the_sandbox():
    sandbox = _FakeSandbox()

    assert repl_prompt(sandbox) == "sandbox(eval:abcdef12)>>> "

    sandbox.info.name = "tan-law-5384"
    assert repl_prompt(sandbox) == "sandbox(eval:tan-law-5384)>>> "


def _typing(monkeypatch, *lines: str) -> None:
    """Stand in for someone at the keyboard, who eventually stops typing."""
    typed = iter(lines)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(typed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)


@pytest.mark.parametrize("command", sorted(EXIT_COMMANDS))
def test_every_exit_command_leaves(monkeypatch, command):
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch, command, "never_reached")

    run_repl(sandbox, console=console, banner=False)

    assert sandbox.ran == []


def test_examples_prints_the_snippets_and_runs_none_of_them(monkeypatch):
    """`:examples` is for a person with a cursor: it shows, it does not run."""
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch, ":examples", ":exit")

    run_repl(
        sandbox,
        console=console,
        banner=False,
        examples=[("Discover the GPU", "import torch\ntorch.cuda.get_device_name(0)")],
    )

    rendered = _rendered(console)
    assert "# 1. Discover the GPU" in rendered
    assert "torch.cuda.get_device_name(0)" in rendered
    # Shown, never executed: the sandbox was not asked to run a thing.
    assert sandbox.ran == []


def test_examples_come_from_the_sandbox_they_were_created_with(monkeypatch):
    """Declared once at creation; the prompt needs no arrangement of its own."""
    console, _ = _console()
    sandbox = _FakeSandbox()
    sandbox.config = SimpleNamespace(examples=[("From the config", "1 + 1")])
    _typing(monkeypatch, ":examples", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert "# 1. From the config" in _rendered(console)


def test_an_example_can_be_run_by_its_number(monkeypatch):
    """`:examples:2` is for a reader who wants the answer, not the paste."""
    console, _ = _console()
    sandbox = _FakeSandbox({"second()": _result(text="ok")})
    _typing(monkeypatch, ":examples:2", ":exit")

    run_repl(
        sandbox,
        console=console,
        banner=False,
        examples=[("First", "first()"), ("Second", "second()")],
    )

    # Run, and shown before it ran: an answer with no visible cause is worse.
    assert sandbox.ran == ["second()"]
    assert "    second()" in _rendered(console)


def test_a_number_that_is_not_an_example_is_refused_without_running_anything(monkeypatch):
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch, ":examples:9", ":examples:x", ":exit")

    run_repl(sandbox, console=console, banner=False, examples=[("Only one", "1")])

    assert sandbox.ran == []
    assert sum("no example" in line for line in _rendered(console)) == 2


def test_brackets_in_a_snippet_survive_being_printed(monkeypatch):
    """Rich reads `[...]` as a style tag, so a type hint would print mangled —
    and be wrong exactly where someone was about to copy it."""
    console, _ = _console()
    _typing(monkeypatch, ":examples", ":exit")

    run_repl(
        _FakeSandbox(),
        console=console,
        banner=False,
        examples=[("Types", "items: list[str] = []\nitems[0:1]")],
    )

    rendered = _rendered(console)
    assert "items: list[str] = []" in rendered
    assert "items[0:1]" in rendered


def test_a_prompt_with_no_examples_says_so_rather_than_printing_nothing(monkeypatch):
    console, _ = _console()
    _typing(monkeypatch, ":examples", ":exit")

    run_repl(_FakeSandbox(), console=console, banner=False)

    assert "This sandbox ships no examples." in _rendered(console)


def test_the_help_names_examples_only_when_there_are_some(monkeypatch):
    """An empty command in the help is worse than no mention of it."""
    with_examples, _ = _console()
    _typing(monkeypatch, ":help", ":exit")
    run_repl(_FakeSandbox(), console=with_examples, banner=False, examples=[("A", "1")])
    assert any(":examples" in line for line in _rendered(with_examples))

    without, _ = _console()
    _typing(monkeypatch, ":help", ":exit")
    run_repl(_FakeSandbox(), console=without, banner=False)
    assert not any(":examples" in line for line in _rendered(without))


def test_a_snippet_is_printed_dedented_so_it_can_be_pasted(monkeypatch):
    """It is copied straight into the prompt, so leading indentation would
    reach the interpreter as an IndentationError."""
    console, _ = _console()
    _typing(monkeypatch, ":examples", ":exit")

    run_repl(
        _FakeSandbox(),
        console=console,
        banner=False,
        examples=[("Indented in the source", "\n            x = 21\n            x * 2\n")],
    )

    rendered = _rendered(console)
    # Flush left, exactly as it must arrive at the prompt.
    assert "x = 21" in rendered
    assert not any(line.startswith(" ") and line.strip() == "x = 21" for line in rendered)


def test_the_prompt_runs_what_is_typed_and_shows_the_answer(monkeypatch):
    console, _ = _console()
    sandbox = _FakeSandbox({"1 + 1": _result(text="2")})
    _typing(monkeypatch, "1 + 1", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert sandbox.ran == ["1 + 1"]
    assert "2" in _rendered(console)


def test_blank_lines_are_not_run(monkeypatch):
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch, "", "   ", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert sandbox.ran == []


def test_running_out_of_input_leaves(monkeypatch):
    """Ctrl-D ends the session; it is not an error."""
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch)

    run_repl(sandbox, console=console, banner=False)

    assert "REPL closed." in _rendered(console)


def test_help_is_shown_without_running_anything(monkeypatch):
    console, _ = _console()
    sandbox = _FakeSandbox()
    _typing(monkeypatch, ":help", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert sandbox.ran == []
    assert any(":help" in line for line in _rendered(console))


def test_a_line_that_blows_up_the_sandbox_does_not_end_the_session(monkeypatch):
    """A prompt outlives its lines: the next one still gets to run."""
    console, _ = _console()
    sandbox = _FakeSandbox()
    calls: list[str] = []

    def explode(code: str, **_kwargs):
        calls.append(code)
        if code == "boom":
            raise RuntimeError("the connection went away")
        return _result(stdout="still here")

    sandbox.run_code = explode
    _typing(monkeypatch, "boom", "after", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert calls == ["boom", "after"]
    rendered = _rendered(console)
    assert any("the connection went away" in line for line in rendered)
    assert "still here" in rendered


def test_an_interrupted_line_does_not_end_the_session(monkeypatch):
    console, _ = _console()
    sandbox = _FakeSandbox()

    def interrupted(_code: str, **_kwargs):
        raise KeyboardInterrupt

    sandbox.run_code = interrupted
    _typing(monkeypatch, "while True: pass", ":exit")

    run_repl(sandbox, console=console, banner=False)

    assert any("interrupted" in line.lower() for line in _rendered(console))
