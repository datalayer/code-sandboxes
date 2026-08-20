# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""The Modal session driver, run as a real subprocess."""

import json
import re
import subprocess
import sys


def _driver_source() -> str:
    text = open("code_sandboxes/modal_sandbox.py").read()
    match = re.search(r'_DRIVER_SOURCE = """(.*?)"""', text, re.S)
    assert match
    return match.group(1)


def _speak(requests):
    stdin = "".join(json.dumps(r) + "\n" for r in requests)
    completed = subprocess.run(  # noqa: S603 — this interpreter, and the driver of this repo
        [sys.executable, "-u", "-c", _driver_source()],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def test_state_survives_between_requests():
    replies = _speak(
        [
            {"seq": 1, "code": "x = 1"},
            {"seq": 2, "code": "x"},
        ]
    )
    assert replies[0]["status"] == "ok"
    assert replies[1]["status"] == "ok"
    assert replies[1]["result"] == "1"


def test_stdout_and_errors_come_back_per_request():
    replies = _speak(
        [
            {"seq": 1, "code": "print('hello')"},
            {"seq": 2, "code": "1 / 0"},
            {"seq": 3, "code": "print('still alive')"},
        ]
    )
    assert replies[0]["stdout"] == "hello\n"
    assert replies[1]["status"] == "error"
    assert replies[1]["error"]["name"] == "ZeroDivisionError"
    # One failing request does not take the session down.
    assert replies[2]["stdout"] == "still alive\n"


def test_trailing_expression_answers_like_a_repl():
    replies = _speak([{"seq": 1, "code": "y = 20\ny * 2 + 2"}])
    assert replies[0]["result"] == "42"
