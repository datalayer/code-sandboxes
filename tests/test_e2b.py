# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The E2B sandbox, against an interpreter that really runs the code.

The fake below is not a mock answering with canned strings: it executes what
the variant sends it and answers in E2B's own shapes. What is worth testing
here is the translation — E2B names each rich format with an attribute where
this package keys them by MIME type, and stamps its output in milliseconds
where this package counts in seconds — so the fake has to speak E2B, and the
assertions read this package's models.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from types import SimpleNamespace

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.e2b_sandbox import E2BSandbox, _result_data, _timestamp
from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.manage import get_manager, manageable_variants
from code_sandboxes.models import SandboxConfig, SandboxVariant
from code_sandboxes.providers import get_provider


class _FakeResult:
    """One of E2B's results: named formats, and `formats()` over them."""

    def __init__(self, **data) -> None:
        self.is_main_result = bool(data.pop("is_main_result", False))
        self.extra = data.pop("extra", {}) or {}
        for name, value in data.items():
            setattr(self, name, value)
        self._names = list(data) + list(self.extra)

    def formats(self):
        return list(self._names)


class _FakeFiles:
    def __init__(self) -> None:
        self.written: dict[str, bytes] = {}

    def write(self, path, data, **_):
        self.written[path] = data if isinstance(data, bytes) else str(data).encode()

    def read(self, path, format="text", **_):  # noqa: A002 - E2B names it so
        if path not in self.written:
            raise FileNotFoundError(path)
        content = self.written[path]
        return content if format == "bytes" else content.decode()


class _FakeE2BSandbox:
    """E2B's code interpreter, executed here instead of over there."""

    def __init__(self, **params) -> None:
        self.params = params
        self.sandbox_id = "e2b-sbx-1"
        self.files = _FakeFiles()
        self.namespaces: dict[str | None, dict] = {None: {"__name__": "__main__"}}
        self.contexts: list[SimpleNamespace] = []
        self.calls: list[dict] = []
        self.killed = False
        self.timeouts: list[int] = []

    def create_code_context(self, cwd=None, language=None, **_):
        context = SimpleNamespace(
            context_id=f"ctx-{len(self.contexts)}", cwd=cwd, language=language
        )
        self.contexts.append(context)
        self.namespaces[context.context_id] = {"__name__": "__main__"}
        return context

    def run_code(
        self,
        code,
        language=None,
        context=None,
        on_stdout=None,
        on_stderr=None,
        on_result=None,
        envs=None,
        timeout=None,
        **_,
    ):
        self.calls.append({"code": code, "context": context, "envs": envs, "timeout": timeout})
        key = context.context_id if context is not None else None
        namespace = self.namespaces.setdefault(key, {"__name__": "__main__"})
        out, err = io.StringIO(), io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
        except BaseException as raised:  # the point of the fake: run it for real
            error = SimpleNamespace(
                name=type(raised).__name__,
                value=str(raised),
                traceback=traceback.format_exc(),
            )

        stdout_lines = out.getvalue().splitlines()
        stderr_lines = err.getvalue().splitlines()
        for line in stdout_lines:
            if on_stdout:
                # E2B stamps in MILLISECONDS since the epoch.
                on_stdout(SimpleNamespace(line=line, timestamp=1_700_000_000_000, error=False))
        for line in stderr_lines:
            if on_stderr:
                on_stderr(SimpleNamespace(line=line, timestamp=1_700_000_000_000, error=True))

        results = list(namespace.pop("_test_results", []))
        for result in results:
            if on_result:
                on_result(result)
        return SimpleNamespace(
            results=results,
            logs=SimpleNamespace(stdout=stdout_lines, stderr=stderr_lines),
            error=error,
            execution_count=len(self.calls),
        )

    def kill(self, **_):
        self.killed = True
        return True

    def set_timeout(self, timeout, **_):
        self.timeouts.append(timeout)

    def get_host(self, port):
        return f"{port}-{self.sandbox_id}.e2b.app"


def _started(config: SandboxConfig | None = None, **kwargs) -> E2BSandbox:
    """A sandbox that believes it started, holding the fake above."""
    sandbox = E2BSandbox(config=config, **kwargs)
    sandbox._sandbox = _FakeE2BSandbox()
    sandbox._started = True
    sandbox._default_context = sandbox.create_context("default")
    return sandbox


# --- Executing ------------------------------------------------------------


def test_a_snippet_keeps_what_the_one_before_it_defined():
    sandbox = _started()

    sandbox.run_code("x = 41")
    execution = sandbox.run_code("print(x + 1)")

    assert [message.line for message in execution.logs.stdout] == ["42"]
    assert execution.code_error is None
    assert execution.execution_ok


def test_output_arrives_as_it_is_written_and_is_kept():
    sandbox = _started()
    streamed: list[str] = []

    execution = sandbox.run_code(
        "import sys\nprint('one')\nprint('two')\nprint('bad', file=sys.stderr)",
        on_stdout=lambda message: streamed.append(message.line),
    )

    assert streamed == ["one", "two"]
    assert [message.line for message in execution.logs.stdout] == ["one", "two"]
    assert [message.line for message in execution.logs.stderr] == ["bad"]


def test_output_is_stamped_in_the_seconds_this_package_counts_in():
    """E2B stamps in milliseconds; a line an epoch in the future is the bug."""
    sandbox = _started()

    execution = sandbox.run_code("print('now')")

    assert execution.logs.stdout[0].timestamp == pytest.approx(1_700_000_000.0)


def test_a_raising_snippet_is_reported_as_the_codes_error_not_the_sandboxs():
    sandbox = _started()
    errors: list = []

    execution = sandbox.run_code("1 / 0", on_error=errors.append)

    # The sandbox did its job; the code is what failed.
    assert execution.execution_ok
    assert execution.code_error is not None
    assert execution.code_error.name == "ZeroDivisionError"
    assert errors and errors[0].name == "ZeroDivisionError"


def test_a_sandbox_that_went_away_is_reported_rather_than_raised():
    sandbox = _started()

    def gone(*_, **__):
        raise ConnectionError("sandbox not found")

    sandbox._sandbox.run_code = gone
    execution = sandbox.run_code("1 + 1")

    assert not execution.execution_ok
    assert "sandbox not found" in (execution.execution_error or "")


def test_rich_formats_are_keyed_by_mime_type():
    result = _FakeResult(
        text="<Figure>",
        png="aGk=",
        html="<b>hi</b>",
        extra={"application/vnd.plotly.v1+json": {"data": []}},
        is_main_result=True,
    )

    data = _result_data(result)

    assert data["text/plain"] == "<Figure>"
    assert data["image/png"] == "aGk="
    assert data["text/html"] == "<b>hi</b>"
    # A format E2B has no attribute for keeps the name it arrived under.
    assert data["application/vnd.plotly.v1+json"] == {"data": []}


def test_a_result_the_sandbox_only_reported_at_the_end_is_not_lost():
    """E2B skips `on_result` for a replayed execution; the answer still has it."""
    sandbox = _started()
    sandbox._sandbox.run_code = lambda *args, **kwargs: SimpleNamespace(
        results=[_FakeResult(text="42", is_main_result=True)],
        logs=SimpleNamespace(stdout=[], stderr=[]),
        error=None,
        execution_count=1,
    )

    execution = sandbox.run_code("40 + 2")

    assert [result.data["text/plain"] for result in execution.results] == ["42"]


def test_a_second_context_is_a_namespace_of_its_own():
    sandbox = _started()
    other = sandbox.create_context("other")

    sandbox.run_code("x = 'default'")
    execution = sandbox.run_code("print('x' in dir())", context=other)

    assert [message.line for message in execution.logs.stdout] == ["False"]


def test_only_python_is_offered():
    with pytest.raises(ValueError, match="only supports Python"):
        _started().run_code("console.log(1)", language="javascript")


# --- Variables and files ---------------------------------------------------


def test_a_variable_crosses_as_json_in_both_directions():
    sandbox = _started()

    sandbox.set_variable("payload", {"a": [1, 2]})

    assert sandbox.get_variable("payload") == {"a": [1, 2]}


def test_a_value_that_cannot_be_encoded_is_refused_where_it_is_set():
    sandbox = _started()

    with pytest.raises(SandboxConfigurationError, match="cannot be encoded"):
        sandbox.set_variable("payload", object())


def test_files_go_through_the_filesystem_api_not_through_the_code():
    sandbox = _started()

    sandbox._write_file("/home/user/notes.txt", b"hello")

    assert sandbox._sandbox.files.written["/home/user/notes.txt"] == b"hello"
    assert sandbox._read_file("/home/user/notes.txt") == b"hello"


# --- Creating --------------------------------------------------------------


def test_the_configuration_becomes_what_e2b_is_asked_for():
    sandbox = E2BSandbox(
        SandboxConfig(name="mine", max_lifetime=600.0, env_vars={"TOKEN": "t"}),
        api_key="e2b_key",
    )

    params = sandbox._create_params()

    # The interpreter's template, not E2B's `base`: `base` carries no kernel
    # for the interpreter to talk to.
    assert params["template"] == "code-interpreter-v1"
    assert params["timeout"] == 600
    assert params["envs"] == {"TOKEN": "t"}
    assert params["api_key"] == "e2b_key"
    assert params["metadata"]["name"] == "mine"
    assert params["metadata"]["created-by"] == "code-sandboxes"


def test_a_setting_that_was_not_given_is_left_for_the_environment():
    """Passing an explicit None is not the same as passing nothing."""
    params = E2BSandbox()._create_params()

    assert "api_key" not in params
    assert "domain" not in params


def test_a_gpu_that_cannot_be_given_is_refused_rather_than_ignored():
    """Running on a CPU while looking as though it asked for a GPU is worse."""
    sandbox = E2BSandbox(SandboxConfig(gpu="H100"))

    with pytest.raises(SandboxConfigurationError, match="no GPU"):
        sandbox._create_params()


def test_a_sandbox_cut_off_from_the_network_says_so():
    params = E2BSandbox(SandboxConfig(network_policy="none"))._create_params()

    assert params["allow_internet_access"] is False


def test_an_allowlist_is_refused_rather_than_widened_to_the_whole_internet():
    sandbox = E2BSandbox(SandboxConfig(network_policy="allowlist", allowed_hosts=["pypi.org"]))

    with pytest.raises(SandboxConfigurationError, match="no host allowlist"):
        sandbox._create_params()


def test_stopping_kills_the_sandbox_and_says_it_stopped():
    sandbox = _started()
    fake = sandbox._sandbox
    sandbox._info = None

    sandbox.stop()

    assert fake.killed
    assert not sandbox.is_started


def test_the_life_of_a_sandbox_can_be_extended_while_it_runs():
    sandbox = _started()

    sandbox.set_timeout(120.4)

    assert sandbox._sandbox.timeouts == [120]


def test_a_port_inside_has_a_host_outside():
    assert _started().get_host(8888).startswith("8888-")


def test_a_timestamp_that_is_not_one_falls_back_to_now():
    assert _timestamp(None) > 0
    assert _timestamp("nonsense") > 0
    # Seconds are left alone; milliseconds are divided.
    assert _timestamp(1_700_000_000) == pytest.approx(1_700_000_000.0)
    assert _timestamp(1_700_000_000_000) == pytest.approx(1_700_000_000.0)


# --- Registration ----------------------------------------------------------


def test_the_variant_is_registered_everywhere_a_variant_is_named():
    assert SandboxVariant.E2B.value == "e2b"
    assert isinstance(Sandbox.create(variant="e2b"), E2BSandbox)
    assert [env.name for env in Sandbox.list_environments(variant="e2b")] == [
        "e2b-code-interpreter",
    ]
    assert get_provider("e2b") is not None
    assert "e2b" in manageable_variants()
    assert get_manager("e2b").variant == "e2b"


def test_the_provider_says_what_it_needs():
    provider = get_provider("e2b")

    assert provider.extra == "e2b"
    assert not provider.is_available({})
    assert provider.is_available({"E2B_API_KEY": "e2b_key"})


def test_the_sdk_is_only_needed_when_the_sandbox_starts():
    """Creating one must not import e2b: a listing names every variant."""
    sandbox = Sandbox.create(variant="e2b")

    assert isinstance(sandbox, E2BSandbox)
    assert not sandbox.is_started
