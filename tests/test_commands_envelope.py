# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""`commands.run` carries its result on stdout, not through a variable read.

It used to run the subprocess inside the kernel, store the result in
`__cmd_output__`, and read it back with `get_variable`. Not every provider can
read a variable out of its session — Modal cannot — so `run_command` there
answered `exit_code=-1` for every command, and the live matrix's Modal row had
never passed. Stdout is the one channel every provider returns.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from code_sandboxes.commands import _ENVELOPE_END, _ENVELOPE_START, SandboxCommands


class FakeSandbox:
    """Answers `run_code` with what a provider prints, and **cannot** read variables."""

    def __init__(self, stdout: str, ok: bool = True) -> None:
        self.stdout = stdout
        self.ok = ok
        self.ran: list[str] = []

    def run_code(self, code: str, timeout=None):
        self.ran.append(code)
        lines = [SimpleNamespace(line=part) for part in self.stdout.splitlines(keepends=True)]
        return SimpleNamespace(
            execution_ok=self.ok,
            execution_error=None if self.ok else "boom",
            logs=SimpleNamespace(stdout=lines, stderr=[]),
        )

    def get_variable(self, name, context=None):
        raise NotImplementedError("this sandbox holds variables in its session process")


def _envelope(exit_code: int, stdout: str, stderr: str = "") -> str:
    return _ENVELOPE_START + json.dumps({"exit_code": exit_code, "stdout": stdout, "stderr": stderr}) + _ENVELOPE_END


def test_the_result_is_read_from_stdout_and_no_variable_is_needed() -> None:
    sandbox = FakeSandbox("some earlier output\n" + _envelope(0, "hello\n") + "\n")

    result = SandboxCommands(sandbox).run("echo hello")

    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    # And the code it sent prints the envelope itself, rather than leaving the
    # result in a variable for a read that some providers cannot make.
    assert "print(" in sandbox.ran[0] and _ENVELOPE_START in sandbox.ran[0]


def test_a_command_that_fails_reports_its_own_exit_code() -> None:
    sandbox = FakeSandbox(_envelope(2, "", "no such file"))

    result = SandboxCommands(sandbox).run("ls /nope")

    assert (result.exit_code, result.stderr) == (2, "no such file")


def test_output_the_command_wrote_before_the_envelope_does_not_confuse_it() -> None:
    """A command that prints the marker text itself: the *last* envelope wins."""
    decoy = _envelope(99, "decoy")
    sandbox = FakeSandbox(decoy + "\n" + _envelope(0, "real"))

    result = SandboxCommands(sandbox).run("printf decoy")

    assert (result.exit_code, result.stdout) == (0, "real")


def test_a_missing_envelope_is_an_error_not_a_silent_success() -> None:
    sandbox = FakeSandbox("the kernel printed this and nothing else\n")

    result = SandboxCommands(sandbox).run("true")

    assert result.exit_code == -1
    assert "envelope" in result.stderr
    assert "kernel printed" in result.stdout, "what was printed is kept for diagnosis"


def test_an_infrastructure_failure_is_reported_as_such() -> None:
    sandbox = FakeSandbox("", ok=False)

    result = SandboxCommands(sandbox).run("true")

    assert result.exit_code == -1
    assert result.stderr == "boom"
