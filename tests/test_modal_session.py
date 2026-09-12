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


# --- One driver per context (PLAN_ENV.md E2-02) ---------------------------


class _Driver:
    """One driver process, doubled: what it was asked, and what it answers."""

    def __init__(self, namespace: dict) -> None:
        self.namespace = namespace
        self.written: list[dict] = []
        self.stdout = self
        self._lines: list[str] = []

        class _Stdin:
            def __init__(self, driver):
                self._driver = driver

            def write(self, text):
                self._driver._ask(text)

            def drain(self):
                return None

        self.stdin = _Stdin(self)

    def _ask(self, text: str) -> None:
        request = json.loads(text)
        self.written.append(request)
        code = request.get("code", "")
        reply = {"seq": request["seq"], "status": "ok", "stdout": "", "stderr": ""}
        try:
            value = eval(compile(code, "<t>", "eval"), self.namespace)  # noqa: S307
            if value is not None:
                reply["result"] = repr(value)
        except SyntaxError:
            exec(compile(code, "<t>", "exec"), self.namespace)  # noqa: S102
        except BaseException as error:
            reply = {
                "seq": request["seq"],
                "status": "error",
                "stdout": "",
                "stderr": "",
                "error": {"name": type(error).__name__, "value": str(error), "traceback": ""},
            }
        self._lines.append(json.dumps(reply) + "\n")

    def __iter__(self):
        # The pump thread reads this: every line written so far, then it waits
        # without spinning — a busy loop here burns a core for as long as the
        # test session lives.
        import time as _time

        while True:
            if self._lines:
                yield self._lines.pop(0)
            else:
                _time.sleep(0.001)


def _modal_with_drivers():
    """A ModalSandbox whose `exec` hands out one fresh driver per call."""
    from types import SimpleNamespace

    from code_sandboxes.modal_sandbox import ModalSandbox
    from code_sandboxes.models import SandboxConfig

    drivers: list[_Driver] = []

    def exec_(*_args, **_kwargs):
        driver = _Driver(namespace={"__name__": "__main__"})
        drivers.append(driver)
        return driver

    sandbox = ModalSandbox(config=SandboxConfig(timeout=5.0))
    sandbox._sandbox = SimpleNamespace(exec=exec_, object_id="sb-1")
    sandbox._started = True
    sandbox._default_context = sandbox.create_context("default")
    return sandbox, drivers


def test_each_context_gets_its_own_driver_and_its_own_namespace():
    """Check 14 of E0-04: one driver made `create_context` a label.

    A driver holds one namespace, so with a single one `x = 1` in one context
    was readable from another — which is not isolation, whatever the result's
    `context_id` said.
    """
    sandbox, drivers = _modal_with_drivers()
    first = sandbox.create_context("first")
    second = sandbox.create_context("second")

    sandbox.run_code("x = 41", context=first)
    mine = sandbox.run_code("x + 1", context=first)
    theirs = sandbox.run_code("x", context=second)

    assert [value.data["text/plain"] for value in mine.results] == ["42"]
    # The other context never saw `x`: its own driver, its own namespace.
    assert theirs.results == []
    assert theirs.code_error is not None and theirs.code_error.name == "NameError"
    assert len(drivers) == 2, "one driver per context, made on first use"


def test_the_default_context_keeps_one_driver_between_snippets():
    sandbox, drivers = _modal_with_drivers()

    sandbox.run_code("y = 7")
    again = sandbox.run_code("y")

    assert [value.data["text/plain"] for value in again.results] == ["7"]
    assert len(drivers) == 1


def test_a_context_used_twice_does_not_start_a_second_driver():
    sandbox, drivers = _modal_with_drivers()
    context = sandbox.create_context("analysis")

    sandbox.run_code("z = 1", context=context)
    sandbox.run_code("z", context=context)

    assert len(drivers) == 1
