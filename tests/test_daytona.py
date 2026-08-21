# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Daytona sandbox, against an interpreter that really runs the code.

The fake below is not a mock answering with canned strings. What this variant
does that is worth testing happens on either side of the wire — the code is
rewritten before it goes out, and the stream that comes back is cut into lines
and read for a marker — and only running the code it actually sends exercises
both.
"""

from __future__ import annotations

import contextlib
import io
import traceback
from types import SimpleNamespace

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.daytona_sandbox import (
    _VALUE_MARKER,
    DaytonaSandbox,
    _capture_trailing_value,
    _Lines,
    _split_marker,
)
from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.manage import get_manager, manageable_variants
from code_sandboxes.models import SandboxConfig, SandboxVariant
from code_sandboxes.providers import get_provider


def _chunks(text: str, size: int = 7) -> list[str]:
    """The text as the websocket would deliver it: cut anywhere at all."""
    return [text[at : at + size] for at in range(0, len(text), size)] if text else []


class _FakeInterpreter:
    """Daytona's code interpreter, executed here instead of over there."""

    def __init__(self) -> None:
        self.namespaces: dict[str | None, dict] = {None: {"__name__": "__main__"}}
        self.contexts: list[SimpleNamespace] = []
        self.calls: list[SimpleNamespace] = []

    def create_context(self, cwd=None):
        context = SimpleNamespace(id=f"ctx-{len(self.contexts)}", cwd=cwd)
        self.contexts.append(context)
        self.namespaces[context.id] = {"__name__": "__main__"}
        return context

    def run_code(
        self,
        code,
        *,
        context=None,
        on_stdout=None,
        on_stderr=None,
        on_error=None,
        envs=None,
        timeout=None,
    ):
        self.calls.append(SimpleNamespace(code=code, context=context, envs=envs, timeout=timeout))
        namespace = self.namespaces[context.id if context else None]
        out, err, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<daytona>", "exec"), namespace)  # noqa: S102
        except BaseException as exc:  # the sandbox reports a failure, it never raises
            error = SimpleNamespace(
                name=type(exc).__name__, value=str(exc), traceback=traceback.format_exc()
            )
        for chunk in _chunks(out.getvalue()):
            if on_stdout:
                on_stdout(SimpleNamespace(output=chunk))
        for chunk in _chunks(err.getvalue()):
            if on_stderr:
                on_stderr(SimpleNamespace(output=chunk))
        if error is not None and on_error:
            on_error(error)
        return SimpleNamespace(stdout=out.getvalue(), stderr=err.getvalue(), error=error)


class _FakeFilesystem:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def upload_file(self, src, dst, timeout=1800):
        if isinstance(src, bytes):
            self.files[dst] = src
            return
        with open(src, "rb") as handle:
            self.files[dst] = handle.read()

    def download_file(self, *args):
        return self.files.get(args[0])


class _FakeDaytonaSandbox:
    def __init__(self) -> None:
        self.id = "sbx-test"
        self.name = None
        self.labels: dict[str, str] = {}
        self.code_interpreter = _FakeInterpreter()
        self.fs = _FakeFilesystem()
        self.deleted = False
        self.stopped = False
        self.spot_evicted_at = None
        #: How often the record was re-read. Asking Daytona is a round trip,
        #: so a test can say when one was worth making.
        self.refreshes = 0

    def refresh_data(self):
        self.refreshes += 1

    def delete(self):
        self.deleted = True

    def stop(self):
        self.stopped = True


def _started(config: SandboxConfig | None = None, **kwargs) -> DaytonaSandbox:
    """A sandbox wired to the fake, as `start()` would have left it."""
    sandbox = DaytonaSandbox(config=config or SandboxConfig(timeout=10.0), **kwargs)
    sandbox._started = True
    sandbox._sandbox = _FakeDaytonaSandbox()
    sandbox._default_context = sandbox.create_context("default")
    return sandbox


# --- What the caller gets back ------------------------------------------


def test_state_survives_between_calls():
    """One namespace per sandbox: that is why the interpreter is used at all."""
    sandbox = _started()

    sandbox.run_code("x = 40")
    result = sandbox.run_code("x + 2")

    assert result.success
    assert result.text == "42"


def test_trailing_expression_answers_like_a_repl():
    sandbox = _started()

    assert sandbox.run_code("1 + 1").text == "2"
    # A statement has no value, and none is invented for it.
    assert sandbox.run_code("y = 3").text is None
    # Neither has a call that returns None.
    assert sandbox.run_code("print('hi')").text is None


def test_the_marker_never_reaches_the_caller():
    """The line carrying the value is ours, and is taken back out."""
    sandbox = _started()
    streamed: list[str] = []

    result = sandbox.run_code("print('hello')\n1 + 1", on_stdout=lambda m: streamed.append(m.line))

    assert result.stdout == "hello"
    assert streamed == ["hello"]
    assert _VALUE_MARKER not in result.stdout


def test_output_cut_across_chunks_is_put_back_together():
    """A websocket cuts where it likes; an OutputMessage is still a line."""
    sandbox = _started()
    long_line = "x" * 40

    result = sandbox.run_code(f"print({long_line!r})\nprint('second')")

    assert [message.line for message in result.logs.stdout] == [long_line, "second"]


def test_a_line_never_terminated_keeps_the_marker_out_of_it():
    """`write` leaves the stream mid-line, so the marker lands on that line."""
    sandbox = _started()

    result = sandbox.run_code("import sys; sys.stdout.write('no newline')")

    assert result.stdout == "no newline"
    # And the value written by that call — the number of characters — is read.
    assert result.text == "10"


def test_stderr_is_kept_apart():
    sandbox = _started()

    result = sandbox.run_code("import sys; sys.stderr.write('warned\\n')")

    assert result.stderr == "warned"
    assert result.stdout == ""


def test_raising_code_comes_back_as_a_code_error():
    """The sandbox worked; the code did not. Those are different failures."""
    sandbox = _started()

    result = sandbox.run_code("raise ValueError('boom')")

    assert result.execution_ok
    assert result.code_error is not None
    assert result.code_error.name == "ValueError"
    assert result.code_error.value == "boom"
    assert not result.success


def test_the_interpreter_refusing_is_an_execution_failure():
    sandbox = _started()

    def explode(*_args, **_kwargs):
        raise RuntimeError("websocket closed")

    sandbox._sandbox.code_interpreter.run_code = explode
    result = sandbox.run_code("1 + 1")

    assert not result.execution_ok
    assert "websocket closed" in (result.execution_error or "")


def test_a_traceback_names_the_line_the_caller_wrote():
    """The capture rewrites the tail; everything before it keeps its place."""
    sandbox = _started()

    result = sandbox.run_code("x = 1\n\n\nundefined_name\n")

    assert result.code_error is not None
    assert result.code_error.name == "NameError"
    assert "line 4" in result.code_error.traceback


def test_the_capture_leaves_nothing_behind_in_the_namespace():
    sandbox = _started()

    sandbox.run_code("1 + 1")
    names = sandbox._sandbox.code_interpreter.namespaces[None]

    assert [name for name in names if name.startswith("_code_sandboxes")] == []


# --- What is sent out ----------------------------------------------------


def test_a_sub_second_timeout_is_rounded_up_to_a_whole_one():
    """Daytona counts in whole seconds, and reads 0 as no limit at all."""
    sandbox = _started()

    sandbox.run_code("1", timeout=0.5)

    assert sandbox._sandbox.code_interpreter.calls[-1].timeout == 1


def test_environment_variables_are_passed_through():
    sandbox = _started()

    sandbox.run_code("1", envs={"TOKEN": "secret"})

    assert sandbox._sandbox.code_interpreter.calls[-1].envs == {"TOKEN": "secret"}


def test_the_default_context_is_daytonas_own():
    sandbox = _started()

    sandbox.run_code("1", context=sandbox._default_context)

    assert sandbox._sandbox.code_interpreter.calls[-1].context is None
    assert sandbox._sandbox.code_interpreter.contexts == []


def test_a_created_context_really_is_isolated():
    sandbox = _started()
    other = sandbox.create_context()

    sandbox.run_code("x = 1")
    result = sandbox.run_code("x", context=other)

    assert result.code_error is not None
    assert result.code_error.name == "NameError"
    # And made once, however often it is used.
    sandbox.run_code("2", context=other)
    assert len(sandbox._sandbox.code_interpreter.contexts) == 1


def test_a_non_python_language_is_refused():
    sandbox = _started()

    with pytest.raises(ValueError, match="only supports Python"):
        sandbox.run_code("SELECT 1", language="sql")


# --- The code that is not rewritten --------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        "x = 1",  # no trailing expression
        "x = (",  # will not parse: the sandbox reports the syntax error
        "async def f():\n    pass",
        "value = await thing()",
    ],
)
def test_code_without_a_capturable_value_is_sent_verbatim(code):
    assert _capture_trailing_value(code) == code


def test_an_awaited_trailing_expression_is_left_alone():
    """Binding an await to a name outside a coroutine does not parse."""
    code = "await thing()"

    assert _capture_trailing_value(code) == code


# --- Lines ---------------------------------------------------------------


def test_lines_holds_a_partial_line_until_the_rest_arrives():
    lines = _Lines()

    assert lines.feed("ab") == []
    assert lines.feed("c\nde") == ["abc"]
    assert lines.feed("f\ng\n") == ["def", "g"]
    assert lines.flush() == []


def test_a_marker_sharing_a_line_with_real_output_is_split_off():
    text, value = _split_marker(f'partial{_VALUE_MARKER}"7"')

    assert (text, value) == ("partial", "7")


def test_a_line_that_merely_looks_like_a_marker_is_left_whole():
    line = f"{_VALUE_MARKER}not json"

    assert _split_marker(line) == (line, None)


def test_lines_gives_up_what_never_got_a_newline():
    lines = _Lines()

    lines.feed("tail")

    assert lines.flush() == ["tail"]
    assert lines.flush() == []


# --- Variables and files -------------------------------------------------


def test_variables_cross_as_json():
    sandbox = _started()

    sandbox.set_variable("payload", {"a": [1, 2], "b": "three"})

    assert sandbox.get_variable("payload") == {"a": [1, 2], "b": "three"}


def test_a_value_that_cannot_be_encoded_is_refused_with_a_reason():
    sandbox = _started()

    with pytest.raises(SandboxConfigurationError, match="cannot be encoded"):
        sandbox.set_variable("fn", lambda: None)


def test_bytes_go_through_the_filesystem_api_not_through_the_code():
    """A large file should not have to become a large program."""
    sandbox = _started()

    sandbox.files.write_bytes("/work/hello.bin", b"content", make_dirs=False)

    assert sandbox._sandbox.fs.files["/work/hello.bin"] == b"content"
    assert sandbox.files.read_bytes("/work/hello.bin") == b"content"
    # Nothing was executed to move those bytes.
    assert sandbox._sandbox.code_interpreter.calls == []


# --- Lifecycle -----------------------------------------------------------


def test_stopping_deletes_the_sandbox_by_default():
    sandbox = _started()
    fake = sandbox._sandbox

    sandbox.stop()

    assert fake.deleted
    assert not sandbox.is_started


def test_a_sandbox_can_be_left_standing_for_later():
    sandbox = _started(delete_on_stop=False)
    fake = sandbox._sandbox

    sandbox.stop()

    assert fake.stopped
    assert not fake.deleted


# --- Configuration -------------------------------------------------------


def test_the_network_policy_becomes_daytonas_own_settings():
    blocked = _started(SandboxConfig(network_policy="none"))
    assert blocked._network_params() == {"network_block_all": True}

    allowed = _started(
        SandboxConfig(
            network_policy="allowlist",
            allowed_hosts=["pypi.org", "files.pythonhosted.org"],
        )
    )
    assert allowed._network_params() == {"domain_allow_list": "pypi.org,files.pythonhosted.org"}

    assert _started(SandboxConfig(network_policy="inherit"))._network_params() == {}


def test_an_allowlist_of_nothing_is_refused():
    sandbox = _started(SandboxConfig(network_policy="allowlist"))

    with pytest.raises(SandboxConfigurationError, match="allowed_hosts"):
        sandbox._network_params()


def test_the_name_travels_as_a_label_not_as_daytonas_name():
    """Daytona names address a sandbox and must be unique; ours are generated."""
    sandbox = _started(SandboxConfig(name="tan-law-5384"))
    sandbox.set_tags({"team": "ai"})

    assert sandbox._labels() == {
        "created-by": "code-sandboxes",
        "name": "tan-law-5384",
        "team": "ai",
    }


def test_a_gpu_daytona_does_not_have_is_refused_by_name():
    daytona = pytest.importorskip("daytona")
    from code_sandboxes.daytona_sandbox import _gpu_types

    with pytest.raises(SandboxConfigurationError, match="no GPU called 'T4'"):
        _gpu_types("T4", daytona)

    assert _gpu_types("h100", daytona) == [daytona.GpuType.H100]


def test_several_gpus_are_an_ordered_list_of_preferences():
    """Daytona takes the first of them it can find, which is the point."""
    daytona = pytest.importorskip("daytona")
    from code_sandboxes.daytona_sandbox import _gpu_types

    assert _gpu_types("H100, rtx_4090 ,H200", daytona) == [
        daytona.GpuType.H100,
        daytona.GpuType.RTX_4090,
        daytona.GpuType.H200,
    ]
    # A name repeated is still one preference, in the place it was first named.
    assert _gpu_types("H100,H100", daytona) == [daytona.GpuType.H100]
    # And one bad name in a list is still a refusal: falling back silently to
    # the rest would hand out a GPU nobody asked for.
    with pytest.raises(SandboxConfigurationError, match="no GPU called 'T4'"):
        _gpu_types("H100,T4", daytona)


def test_resources_are_only_asked_for_when_the_configuration_says_so():
    daytona = pytest.importorskip("daytona")

    assert _started(SandboxConfig())._resources(daytona) is None
    resources = _started(
        SandboxConfig(cpu_limit=2.0, memory_limit=4 * 1024**3, gpu="H100")
    )._resources(daytona)
    assert (resources.cpu, resources.memory, resources.gpu) == (2, 4, 1)
    assert resources.gpu_type == daytona.GpuType.H100


def test_spot_asks_for_exactly_what_the_documented_example_does():
    """Preemptible capacity: an image, `spot`, and no lingering afterwards."""
    daytona = pytest.importorskip("daytona")

    params = _started(SandboxConfig(gpu="H100,H200"), spot=True, gpu_count=2)._create_params(
        daytona
    )

    assert isinstance(params, daytona.CreateSandboxFromImageParams)
    assert params.spot is True
    # Mandatory for spot: a reclaimed sandbox does not hang about.
    assert params.auto_delete_interval == 0
    assert params.resources.gpu == 2
    assert params.resources.gpu_type == [daytona.GpuType.H100, daytona.GpuType.H200]


def test_one_gpu_goes_as_one_name_not_as_a_list_of_one():
    daytona = pytest.importorskip("daytona")

    resources = _started(SandboxConfig(gpu="H100"))._resources(daytona)

    assert resources.gpu_type == daytona.GpuType.H100


def test_spot_without_a_gpu_is_refused_with_the_reason():
    """Daytona rejects it too; saying so here costs no round trip."""
    daytona = pytest.importorskip("daytona")

    with pytest.raises(SandboxConfigurationError, match="needs a GPU"):
        _started(SandboxConfig(), spot=True)._create_params(daytona)


def test_spot_from_a_snapshot_is_refused():
    """Only an image carries a machine specification, and spot needs one."""
    daytona = pytest.importorskip("daytona")

    sandbox = _started(SandboxConfig(gpu="H100"), spot=True, snapshot="my-snapshot")
    with pytest.raises(SandboxConfigurationError, match="snapshot"):
        sandbox._create_params(daytona)


def test_a_sandbox_that_was_not_asked_for_on_spot_does_not_say_spot():
    daytona = pytest.importorskip("daytona")

    params = _started(SandboxConfig(gpu="H100"))._create_params(daytona)

    assert params.spot is None


def test_being_reclaimed_is_reported_as_being_reclaimed():
    """A dropped connection and an eviction read alike without asking."""
    sandbox = _started(SandboxConfig(gpu="H100"), spot=True)
    sandbox._sandbox.spot_evicted_at = "2026-08-21T10:00:00Z"

    def gone(*_args, **_kwargs):
        raise RuntimeError("websocket closed")

    sandbox._sandbox.code_interpreter.run_code = gone
    result = sandbox.run_code("1 + 1")

    assert not result.execution_ok
    assert "reclaimed at 2026-08-21T10:00:00Z" in (result.execution_error or "")


def test_a_failure_that_is_not_an_eviction_is_still_reported_as_itself():
    sandbox = _started(SandboxConfig(gpu="H100"), spot=True)

    def gone(*_args, **_kwargs):
        raise RuntimeError("websocket closed")

    sandbox._sandbox.code_interpreter.run_code = gone
    result = sandbox.run_code("1 + 1")

    assert "websocket closed" in (result.execution_error or "")
    assert "reclaimed" not in (result.execution_error or "")


def test_a_sandbox_on_demand_is_never_asked_whether_it_was_reclaimed():
    """The question costs a round trip and has one answer for on-demand."""
    sandbox = _started(SandboxConfig())

    def gone(*_args, **_kwargs):
        raise RuntimeError("websocket closed")

    sandbox._sandbox.code_interpreter.run_code = gone
    result = sandbox.run_code("1 + 1")

    assert "websocket closed" in (result.execution_error or "")
    assert sandbox._sandbox.refreshes == 0


def test_the_eviction_is_read_freshly_rather_than_from_the_old_record():
    """It happened after the sandbox was made, so the copy it holds is stale."""
    sandbox = _started(SandboxConfig(gpu="H100"), spot=True)

    assert sandbox.preempted_at() is None
    assert sandbox._sandbox.refreshes == 1

    sandbox._sandbox.spot_evicted_at = "2026-08-21T10:00:00Z"
    assert sandbox.preempted_at() == "2026-08-21T10:00:00Z"


def test_asking_for_resources_creates_from_an_image():
    """Daytona takes a machine specification with an image, not a snapshot."""
    daytona = pytest.importorskip("daytona")

    from_snapshot = _started(SandboxConfig())._create_params(daytona)
    assert isinstance(from_snapshot, daytona.CreateSandboxFromSnapshotParams)

    from_image = _started(SandboxConfig(cpu_limit=2.0))._create_params(daytona)
    assert isinstance(from_image, daytona.CreateSandboxFromImageParams)
    assert from_image.resources.cpu == 2


def test_only_the_client_settings_that_were_given_are_passed_on():
    """What is left out is what the SDK reads from the environment."""
    daytona = pytest.importorskip("daytona")

    assert _started()._client_config(daytona) is None
    config = _started(api_key="dtn_key")._client_config(daytona)
    assert config.api_key == "dtn_key"


# --- Registration --------------------------------------------------------


def test_the_variant_is_registered_everywhere_a_variant_is_named():
    assert SandboxVariant.DAYTONA.value == "daytona"
    assert isinstance(Sandbox.create(variant="daytona"), DaytonaSandbox)
    assert [env.name for env in Sandbox.list_environments(variant="daytona")] == [
        "daytona-default",
        "daytona-gpu",
        "daytona-gpu-spot",
    ]
    assert get_provider("daytona") is not None
    assert "daytona" in manageable_variants()
    assert get_manager("daytona").variant == "daytona"


def test_the_provider_says_what_it_needs():
    provider = get_provider("daytona")

    assert provider.extra == "daytona"
    assert not provider.is_available({})
    assert provider.is_available({"DAYTONA_API_KEY": "dtn_key"})
    assert provider.is_available({"DAYTONA_JWT_TOKEN": "jwt", "DAYTONA_ORGANIZATION_ID": "org"})
