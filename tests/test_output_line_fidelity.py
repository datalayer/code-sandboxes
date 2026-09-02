# Copyright (c) 2023-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""Output is reassembled as the kernel wrote it, not as whole lines.

`print(".", end="")` sends a chunk with no terminator and a notebook puts the
next one beside it — which is how a progress line is written. Splitting output
into "lines" and rejoining them with newlines discards that and then invents it
back, so six dots printed side by side came out as six lines of one dot.
"""

from code_sandboxes.models import Logs, OutputMessage


def test_chunks_without_a_terminator_stay_on_one_line() -> None:
    messages = [OutputMessage(line="Mars", terminated=True)]
    messages += [OutputMessage(line=".", terminated=False) for _ in range(6)]
    messages.append(OutputMessage(line="", terminated=True))

    assert Logs(stdout=messages).stdout_text == "Mars\n......"


def test_whole_lines_are_unchanged() -> None:
    # Every producer that predates the flag says nothing about it, so the
    # default has to reassemble exactly as `"\n".join` used to — including
    # the absence of a trailing newline, which callers compare against.
    messages = [OutputMessage(line="a"), OutputMessage(line="b")]
    assert Logs(stdout=messages).stdout_text == "a\nb"


def test_no_output_is_the_empty_string() -> None:
    assert Logs().stdout_text == ""
    assert Logs().stderr_text == ""


def test_stderr_follows_the_same_rule() -> None:
    messages = [
        OutputMessage(line="warn", terminated=False, error=True),
        OutputMessage(line="ing", terminated=True, error=True),
    ]
    assert Logs(stderr=messages).stderr_text == "warning"
