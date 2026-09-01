# Copyright (c) 2023-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""A failed execution carries its traceback as an output, not only as a string.

The exception used to be captured into ``code_error`` and flattened into a
one-line ``error`` message, while ``outputs`` carried only the results. Anything
rendering those outputs therefore had no traceback to render and had to invent a
stand-in — one ``KeyError: 'east'`` line where a notebook shows the frames.
"""

from types import SimpleNamespace

from code_sandboxes.client import CodeExecutionOutcome


def _execution(*, code_error, results=(), success=False):
    return SimpleNamespace(
        success=success,
        execution_ok=True,
        logs=SimpleNamespace(stdout_text="this much worked", stderr_text=""),
        results=list(results),
        code_error=code_error,
        execution_error=None,
        interrupted=False,
        exit_code=None,
        execution_count=1,
    )


def test_a_failure_appends_an_error_output() -> None:
    outcome = CodeExecutionOutcome.from_execution_result(
        _execution(
            code_error=SimpleNamespace(
                name="KeyError",
                value="'east'",
                traceback="Traceback (most recent call last):\n  line 3\nKeyError: 'east'",
            )
        )
    )
    errors = [o for o in outcome.outputs if o.get("output_type") == "error"]
    assert len(errors) == 1
    error = errors[0]
    assert error["ename"] == "KeyError"
    assert error["evalue"] == "'east'"
    # Lines, per nbformat — a renderer joins them, it does not split them.
    assert error["traceback"] == [
        "Traceback (most recent call last):",
        "  line 3",
        "KeyError: 'east'",
    ]


def test_the_one_line_message_is_still_there() -> None:
    # The summary string has callers of its own; the output is additional.
    outcome = CodeExecutionOutcome.from_execution_result(
        _execution(
            code_error=SimpleNamespace(
                name="KeyError", value="'east'", traceback="KeyError: 'east'"
            )
        )
    )
    assert outcome.error == "KeyError: 'east'"


def test_a_clean_run_appends_nothing() -> None:
    outcome = CodeExecutionOutcome.from_execution_result(
        _execution(code_error=None, success=True)
    )
    assert not [o for o in outcome.outputs if o.get("output_type") == "error"]
