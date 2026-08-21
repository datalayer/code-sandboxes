# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The CoreWeave sandbox, against a container that really runs the driver.

The fake below does not answer with canned replies: it runs the driver source
the variant sends — as a real subprocess, in
:func:`test_the_driver_source_is_a_program_that_answers_what_it_promises`, and
in-process everywhere else — because the driver IS the protocol here. A change
to it that broke the framing would otherwise pass every test in this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.coreweave_sandbox import (
    _DRIVER_SOURCE,
    DEFAULT_CONTAINER_IMAGE,
    CoreWeaveSandbox,
    _with_envs,
)
from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.manage import get_manager, manageable_variants
from code_sandboxes.models import SandboxConfig, SandboxVariant
from code_sandboxes.providers import get_provider


class _FakeStdin:
    def __init__(self, on_line, on_close=None) -> None:
        self._on_line = on_line
        self._on_close = on_close
        self.closed = False

    def writeline(self, text: str):
        self._on_line(text)
        return SimpleNamespace(result=lambda timeout=None: None)

    def close(self):
        self.closed = True
        if self._on_close is not None:
            self._on_close()
        return SimpleNamespace(result=lambda timeout=None: None)


def _serve(code: str, namespace: dict, seq=None) -> dict:
    """Run one request the way the driver does, and answer as it answers."""
    import ast
    import contextlib
    import io
    import traceback

    out, err = io.StringIO(), io.StringIO()
    reply: dict = {"status": "ok"}
    if seq is not None:
        reply["seq"] = seq
    try:
        tree = ast.parse(code, mode="exec")
        trailing = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            trailing = ast.Expression(tree.body.pop(-1).value)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if tree.body:
                exec(compile(tree, "<sandbox>", "exec"), namespace)  # noqa: S102
            if trailing is not None:
                value = eval(compile(trailing, "<sandbox>", "eval"), namespace)  # noqa: S307
                if value is not None:
                    reply["result"] = repr(value)
    except BaseException as error:  # the point of the fake: run it for real
        reply["status"] = "error"
        reply["error"] = {
            "name": type(error).__name__,
            "value": str(error),
            "traceback": traceback.format_exc(),
        }
    reply["stdout"] = out.getvalue()
    reply["stderr"] = err.getvalue()
    return reply


class _FakeDriver:
    """The session process: one namespace, one reply per request.

    Its stdout BLOCKS the way a real stream does — the variant reads it from a
    thread, and a stream that ended as soon as it was empty would look like a
    driver that had died before the first request was even written.
    """

    def __init__(self) -> None:
        import queue

        self.namespace: dict = {"__name__": "__main__"}
        self.requests: list[dict] = []
        self._replies: queue.Queue = queue.Queue()
        self.stdin = _FakeStdin(self.serve, on_close=lambda: self._replies.put(None))
        self.stdout = self._lines()

    def _lines(self):
        while True:
            line = self._replies.get()
            if line is None:
                return
            yield line

    def serve(self, line: str) -> None:
        request = json.loads(line)
        self.requests.append(request)
        reply = _serve(request.get("code", ""), self.namespace, seq=request.get("seq"))
        self._replies.put(json.dumps(reply))


class _FakeProcess:
    """One process of its own: the stateless fallback, and what it answers."""

    def __init__(self) -> None:
        self._reply: dict | None = None
        self.stdin = _FakeStdin(self._run)

    def _run(self, line: str) -> None:
        request = json.loads(line)
        self._reply = _serve(request.get("code", ""), {"__name__": "__main__"})

    def result(self, timeout=None):
        printed = json.dumps(self._reply or {}) + "\n"
        return SimpleNamespace(stdout_bytes=printed.encode(), stderr_bytes=b"")


class _FakeCoreWeaveSandbox:
    def __init__(self, **params) -> None:
        self.params = params
        self.sandbox_id = "cw-sbx-1"
        self.runner_id = "runner-1"
        self.files: dict[str, bytes] = {}
        self.execs: list[list[str]] = []
        self.driver: _FakeDriver | None = None
        self.stopped = False
        self.waited = False

    def wait(self, timeout=None):
        self.waited = True
        return self

    def exec(self, command, *, cwd=None, check=False, timeout_seconds=None, stdin=False):
        self.execs.append(list(command))
        # Which program was started decides what it is: the session driver, or
        # one process for one snippet.
        if _DRIVER_SOURCE in command:
            self.driver = _FakeDriver()
            return self.driver
        return _FakeProcess()

    def stop(self, **_):
        self.stopped = True
        return SimpleNamespace(result=lambda timeout=None: None)

    def write_file(self, filepath, contents, **_):
        self.files[filepath] = contents
        return SimpleNamespace(result=lambda timeout=None: None)

    def read_file(self, filepath, **_):
        content = self.files.get(filepath)
        return SimpleNamespace(result=lambda timeout=None: content)


def _started(config: SandboxConfig | None = None, *, stateful: bool = True, **kwargs):
    """A sandbox that believes it started, holding the fake above."""
    sandbox = CoreWeaveSandbox(config=config, stateful=stateful, **kwargs)
    sandbox._sandbox = _FakeCoreWeaveSandbox()
    sandbox._started = True
    if stateful:
        sandbox._start_driver()
    sandbox._default_context = sandbox.create_context("default")
    return sandbox


# --- The driver, as a real program -----------------------------------------


def test_the_driver_source_is_a_program_that_answers_what_it_promises():
    """Run it for real: the driver is the protocol, and must hold its shape."""
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-u", "-c", _DRIVER_SOURCE],
        input='{"seq": 1, "code": "x = 21"}\n{"seq": 2, "code": "print(x); x * 2"}\n',
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    replies = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert [reply["seq"] for reply in replies] == [1, 2]
    # The namespace is shared, the trailing expression is reported, and what
    # was printed stays separate from it.
    assert replies[1]["stdout"] == "21\n"
    assert replies[1]["result"] == "42"
    assert replies[1]["status"] == "ok"


def test_the_driver_reports_a_raising_snippet_without_dying():
    process = subprocess.run(  # noqa: S603
        [sys.executable, "-u", "-c", _DRIVER_SOURCE],
        input='{"seq": 1, "code": "1 / 0"}\n{"seq": 2, "code": "40 + 2"}\n',
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    replies = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    assert replies[0]["status"] == "error"
    assert replies[0]["error"]["name"] == "ZeroDivisionError"
    # Still serving afterwards: one bad snippet does not end the session.
    assert replies[1]["result"] == "42"


# --- Executing --------------------------------------------------------------


def test_a_snippet_keeps_what_the_one_before_it_defined():
    sandbox = _started()

    sandbox.run_code("x = 41")
    execution = sandbox.run_code("print(x + 1)")

    assert [message.line for message in execution.logs.stdout] == ["42"]
    assert execution.execution_ok
    assert execution.code_error is None


def test_the_value_of_a_trailing_expression_is_answered_with():
    sandbox = _started()

    execution = sandbox.run_code("1 + 1")

    assert [result.data["text/plain"] for result in execution.results] == ["2"]
    assert execution.text == "2"


def test_output_reaches_a_caller_that_streams():
    sandbox = _started()
    streamed: list[str] = []
    errors: list[str] = []

    sandbox.run_code(
        "import sys\nprint('one')\nprint('bad', file=sys.stderr)",
        on_stdout=lambda message: streamed.append(message.line),
        on_stderr=lambda message: errors.append(message.line),
    )

    assert streamed == ["one"]
    assert errors == ["bad"]


def test_a_raising_snippet_is_reported_as_the_codes_error_not_the_sandboxs():
    sandbox = _started()

    execution = sandbox.run_code("1 / 0")

    assert execution.execution_ok
    assert execution.code_error is not None
    assert execution.code_error.name == "ZeroDivisionError"


def test_a_session_that_never_answers_times_out_rather_than_hanging():
    sandbox = _started()
    # A driver that takes the request and says nothing.
    sandbox._driver.stdin.writeline = lambda text: None

    execution = sandbox.run_code("1 + 1", timeout=0.05)

    assert not execution.execution_ok
    assert "within" in (execution.execution_error or "")


def test_a_session_that_went_away_falls_back_to_a_process_per_snippet():
    """Working, merely stateless: the sandbox must not stop being usable."""
    sandbox = _started()

    def gone(text):
        raise BrokenPipeError("the driver is gone")

    sandbox._driver.stdin.writeline = gone
    execution = sandbox.run_code("print('still here')")

    assert sandbox._driver is None
    assert execution.execution_ok
    assert [message.line for message in execution.logs.stdout] == ["still here"]


def test_without_a_driver_each_snippet_runs_in_its_own_process():
    sandbox = _started(stateful=False)

    assert sandbox._driver is None
    execution = sandbox.run_code("21 * 2")

    assert execution.text == "42"


def test_only_python_is_offered():
    with pytest.raises(ValueError, match="only supports Python"):
        _started().run_code("console.log(1)", language="javascript")


def test_an_environment_asked_for_reaches_the_snippet():
    """The session process is started once, so envs are set in the namespace."""
    sandbox = _started()

    execution = sandbox.run_code("import os\nprint(os.environ['TOKEN'])", envs={"TOKEN": "shhh"})

    assert [message.line for message in execution.logs.stdout] == ["shhh"]
    assert "TOKEN" in _with_envs("pass", {"TOKEN": "shhh"})


# --- Variables and files ----------------------------------------------------


def test_a_variable_crosses_as_json_in_both_directions():
    sandbox = _started()

    sandbox.set_variable("payload", {"a": [1, 2]})

    assert sandbox.get_variable("payload") == {"a": [1, 2]}


def test_a_value_that_cannot_be_encoded_is_refused_where_it_is_set():
    with pytest.raises(SandboxConfigurationError, match="cannot be encoded"):
        _started().set_variable("payload", object())


def test_without_a_session_a_variable_is_refused_rather_than_lost():
    """A set that reports success and then vanishes is the worse answer."""
    sandbox = _started(stateful=False)

    with pytest.raises(SandboxConfigurationError, match="no session process"):
        sandbox.set_variable("payload", {"a": 1})
    with pytest.raises(SandboxConfigurationError, match="no session process"):
        sandbox.get_variable("payload")


def test_a_timed_out_snippet_has_its_session_stopped():
    """Giving up on the answer is not giving up on the work: it must be cut."""
    sandbox = _started()
    driver = sandbox._driver
    cancelled: list[bool] = []
    driver.cancel = lambda: cancelled.append(True)
    # A session that takes the request and never answers.
    driver.stdin.writeline = lambda text: None

    execution = sandbox.run_code("1 + 1", timeout=0.05)

    assert not execution.execution_ok
    assert "stopped" in (execution.execution_error or "")
    assert cancelled == [True]
    # Dropped, so the next execution starts a session of its own.
    assert sandbox._driver is None


def test_files_go_through_the_filesystem_api_not_through_the_code():
    sandbox = _started()

    sandbox._write_file("/workspace/notes.txt", b"hello")

    assert sandbox._sandbox.files["/workspace/notes.txt"] == b"hello"
    assert sandbox._read_file("/workspace/notes.txt") == b"hello"


def test_a_file_that_is_not_there_is_not_read_as_an_empty_one():
    with pytest.raises(FileNotFoundError):
        _started()._read_file("/workspace/missing.txt")


# --- Creating ---------------------------------------------------------------


def test_the_configuration_becomes_what_coreweave_is_asked_for():
    cwsandbox = pytest.importorskip("cwsandbox")
    sandbox = CoreWeaveSandbox(
        SandboxConfig(
            name="mine",
            max_lifetime=600.0,
            env_vars={"TOKEN": "t"},
            cpu_limit=2.0,
            memory_limit=4 * 1024**3,
            gpu="H100",
        )
    )

    params = sandbox._run_params(cwsandbox)

    assert params["container_image"] == DEFAULT_CONTAINER_IMAGE
    assert params["max_lifetime_seconds"] == 600.0
    assert params["environment_variables"] == {"TOKEN": "t"}
    assert "name=mine" in params["tags"]
    assert "created-by=code-sandboxes" in params["tags"]
    assert params["resources"].requests == {"cpu": "2", "memory": "4Gi"}
    assert params["resources"].gpu == {"count": 1, "type": "H100"}


def test_a_sandbox_cut_off_from_the_network_says_so():
    cwsandbox = pytest.importorskip("cwsandbox")

    options = CoreWeaveSandbox(SandboxConfig(network_policy="none"))._network_params(cwsandbox)

    assert options.deny_egress is True


def test_an_allowlist_of_nothing_is_refused_rather_than_silently_meaning_none():
    cwsandbox = pytest.importorskip("cwsandbox")
    sandbox = CoreWeaveSandbox(SandboxConfig(network_policy="allowlist"))

    with pytest.raises(SandboxConfigurationError, match="needs allowed_hosts"):
        sandbox._network_params(cwsandbox)


def test_the_defaults_ask_for_no_particular_machine():
    cwsandbox = pytest.importorskip("cwsandbox")

    assert CoreWeaveSandbox()._resources(cwsandbox) is None


def test_stopping_stops_the_sandbox_and_closes_the_session():
    sandbox = _started()
    fake = sandbox._sandbox
    driver = sandbox._driver

    sandbox.stop()

    assert driver.stdin.closed
    assert fake.stopped
    assert not sandbox.is_started


# --- Registration -----------------------------------------------------------


def test_the_variant_is_registered_everywhere_a_variant_is_named():
    assert SandboxVariant.COREWEAVE.value == "coreweave"
    assert isinstance(Sandbox.create(variant="coreweave"), CoreWeaveSandbox)
    assert [env.name for env in Sandbox.list_environments(variant="coreweave")] == [
        "coreweave-default",
        "coreweave-gpu",
    ]
    assert get_provider("coreweave") is not None
    assert "coreweave" in manageable_variants()
    assert get_manager("coreweave").variant == "coreweave"


def test_the_provider_says_what_it_needs():
    provider = get_provider("coreweave")

    assert provider.extra == "coreweave"
    assert not provider.is_available({})
    assert provider.is_available({"CWSANDBOX_API_KEY": "cw_key"})


def test_the_sdk_is_only_needed_when_the_sandbox_starts():
    sandbox = Sandbox.create(variant="coreweave")

    assert isinstance(sandbox, CoreWeaveSandbox)
    assert not sandbox.is_started
