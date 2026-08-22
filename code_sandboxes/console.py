# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Showing a sandbox at work — one snippet, or a prompt full of them.

Two things are wanted of every sandbox here: run a snippet and show what came
back, or hold a prompt open and do that repeatedly. Three programs had each
grown their own copy — the CLI had a REPL with one way of rendering a result,
the REPL examples had a second, the exec examples a third — and they disagreed
about the things a reader actually notices: whether the value of the last
expression is shown at all, whether stderr is told apart from stdout, which
words end a session.

This is that machinery, once. `code-sandboxes exec` and `code-sandboxes repl`
run on it, and so does every example — which is what makes an example worth
reading: it shows how to use a sandbox rather than how to print things.
"""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from .base import Sandbox
    from .models import ExecutionResult

__all__ = [
    "EXIT_COMMANDS",
    "example_code",
    "repl_prompt",
    "run_repl",
    "show_and_run",
    "show_code",
    "show_examples",
    "show_result",
]

#: What ends a session, typed at the prompt. Both spellings: `:exit` is what
#: the help offers, and `exit` is what someone types anyway.
EXIT_COMMANDS = frozenset({":exit", ":quit", "exit", "quit"})

_console = Console()


def _out(console: Console | None) -> Console:
    return console if console is not None else _console


def _write(console: Console, line: str, *, style: str | None = None, indent: bool = False) -> None:
    """One line of a sandbox's output, printed as the text it is.

    `markup` and `highlight` off, always: what comes back from a sandbox is
    data, and rich reads `[1, 2]` as a style tag and colours anything that
    looks like a number or a path. A REPL that renders a list as a colour is
    not showing the list.
    """
    console.print(
        f"    {line}" if indent else line,
        style=style,
        markup=False,
        highlight=False,
    )


def show_code(code: str, *, console: Console | None = None) -> None:
    """Print the code about to be submitted, indented under its marker."""
    out = _out(console)
    out.print(">>> code:", style="cyan")
    for line in code.strip("\n").splitlines():
        _write(out, line, indent=True)


def show_result(
    result: ExecutionResult,
    *,
    console: Console | None = None,
    labelled: bool = True,
) -> None:
    """Print what an execution came back with, and only that.

    `labelled` puts each stream under a marker of its own, which is what a
    transcript of several snippets needs to stay readable. A REPL, where the
    reader typed the line a moment ago and is watching for the answer, has no
    use for it and reads better without.
    """
    out = _out(console)
    stdout = (result.stdout or "").strip("\n")
    stderr = (result.stderr or "").strip("\n")
    text = (result.text or "").strip()

    def block(marker: str, lines: list[str], style: str | None = None) -> None:
        if labelled:
            out.print(marker, style=style or "cyan")
        for line in lines:
            _write(out, line, style=style, indent=labelled)

    if stdout:
        block("<<< stdout:", stdout.splitlines())
    # The value of the last expression, when it is not the stdout again.
    if text and text != stdout.strip():
        _write(out, f"<<< result: {text}" if labelled else text)
    if stderr:
        block("<<< stderr:", stderr.splitlines(), style="yellow")
    if result.code_error is not None:
        error = f"{result.code_error.name}: {result.code_error.value}"
        _write(out, f"<<< error: {error}" if labelled else error, style="red")
    if not result.execution_ok:
        # Not the code failing but the sandbox failing to run it, which is a
        # different thing and says so.
        failed = result.execution_error or "the sandbox could not run this"
        _write(out, f"<<< execution error: {failed}", style="red")
    elif labelled and not stdout and not text and result.code_error is None:
        # A snippet can run perfectly and say nothing; in a transcript that
        # has to be visible, or the reader looks for output that never came.
        out.print("<<< (no output)", style="dim")


def show_and_run(
    sandbox: Sandbox,
    code: str,
    *,
    console: Console | None = None,
    **kwargs: Any,
) -> ExecutionResult:
    """Print the code, run it, print what came back, and return the result."""
    show_code(code, console=console)
    result = sandbox.run_code(code, **kwargs)
    show_result(result, console=console)
    return result


def repl_prompt(sandbox: Sandbox) -> str:
    """The prompt, naming which sandbox the line is about to be run in.

    A bare `>>>` is ambiguous the moment a second terminal is open, and these
    prompts are usually opened in pairs — one against a local kernel and one
    against something in a cloud, to compare them.
    """
    info = sandbox.info
    if info is None:
        return "sandbox>>> "
    name = info.name or (info.id[:8] if info.id else "sandbox")
    return f"sandbox({info.variant or 'unknown'}:{name})>>> "


def _show_help(console: Console, has_examples: bool = False) -> None:
    console.print("Type Python statements or expressions.", style="dim")
    console.print(
        "State is kept between lines, and the value of an expression is shown.",
        style="dim",
    )
    console.print(
        f"{', '.join(sorted(EXIT_COMMANDS))} — leave, terminating the sandbox.",
        style="dim",
    )
    if has_examples:
        console.print(":examples — the snippets for this sandbox, numbered.", style="dim")
        console.print(":examples:2 — run the second one, without pasting it.", style="dim")
    console.print(":help — this.", style="dim")


def example_code(examples: Sequence[tuple[str, str]], number: int) -> str | None:
    """The code of example `number`, counted from one, or None if there is no
    such example."""
    if 1 <= number <= len(examples):
        return dedent(examples[number - 1][1]).strip("\n")
    return None


def show_examples(
    examples: Sequence[tuple[str, str]],
    console: Console | None = None,
) -> None:
    """Print the snippets, numbered, each under what it does.

    Numbered so they can be asked for by number — `:examples:2` runs the
    second — and printed plainly rather than boxed, because a reader who wants
    to paste one instead selects it with the cursor and anything drawn around
    it would come along.

    `markup=False` throughout: Rich reads `[...]` as a style tag, so a snippet
    holding `list[str]` or `data[1:3]` would print with the brackets eaten and
    be wrong in exactly the place someone was about to copy.
    """
    out = _out(console)
    if not examples:
        out.print("This sandbox ships no examples.", style="dim")
        return
    out.print("")
    for number, (title, code) in enumerate(examples, start=1):
        out.print(f"# {number}. {title}", style="cyan", markup=False, highlight=False)
        for line in dedent(code).strip("\n").splitlines():
            out.print(line, style="white", markup=False, highlight=False)
        out.print("")
    out.print(
        f":examples:N runs one of them — 1 to {len(examples)}.",
        style="dim",
        markup=False,
    )


def run_repl(  # noqa: C901
    sandbox: Sandbox,
    *,
    console: Console | None = None,
    banner: bool = True,
    examples: Sequence[tuple[str, str]] | None = None,
) -> None:
    """Hold a prompt open against a sandbox that is already started.

    Leaving the loop does NOT stop the sandbox: whoever started it decides
    when it goes, which for every caller here is the `with` block around this.

    Args:
        examples: Title-and-code pairs, overriding whatever the sandbox was
            created with. Normally left out: they are declared once at
            `Sandbox.create(examples=...)` and read from there, so a caller
            holding a sandbox already has them and a prompt opened on it
            offers the right ones without being told twice.
    """
    out = _out(console)
    prompt = repl_prompt(sandbox)
    # The sandbox's own, unless this call brought its own list.
    if examples is None:
        # Reached through two `getattr`s on purpose: `run_repl` takes anything
        # that runs code, and a stand-in without a `config` should open a
        # prompt with no examples rather than fail to open one at all.
        examples = list(getattr(getattr(sandbox, "config", None), "examples", None) or [])
    if banner:
        out.print("Sandbox REPL ready. Type Python and press Enter.", style="green")
        hint = ":exit or Ctrl-D to leave, :help for help."
        if examples:
            hint = ":examples for snippets to paste, :exit to leave, :help for help."
        out.print(hint, style="dim")

    while True:
        try:
            line = input(prompt)
        except EOFError:
            out.print("")
            break
        except KeyboardInterrupt:
            # Ctrl-C abandons the line, as a shell does; it does not leave.
            out.print("\nInterrupted. Use :exit to leave.", style="yellow")
            continue

        code = line.strip()
        if not code:
            continue
        if code in EXIT_COMMANDS:
            break
        if code == ":help":
            _show_help(out, has_examples=bool(examples))
            continue
        if code == ":examples":
            show_examples(examples or [], console=out)
            continue
        if code.startswith(":examples:"):
            asked = code[len(":examples:") :].strip()
            wanted = int(asked) if asked.isdigit() else 0
            chosen = example_code(examples or [], wanted)
            if chosen is None:
                out.print(
                    f"There is no example {asked!r}. :examples lists them.",
                    style="yellow",
                    markup=False,
                )
                continue
            # Shown before it runs: an example that executed invisibly would
            # leave the reader with an answer and no idea what produced it.
            show_code(chosen, console=out)
            code = chosen

        try:
            result = sandbox.run_code(code)
        except KeyboardInterrupt:
            out.print("\nExecution interrupted.", style="yellow")
            continue
        except Exception as exc:
            # A prompt outlives its lines: whatever the sandbox threw is
            # reported, and the next line still gets to run.
            _write(out, f"Execution failed: {exc}", style="red")
            continue

        show_result(result, console=out, labelled=False)

    out.print("REPL closed.", style="green")
