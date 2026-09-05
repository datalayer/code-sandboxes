# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Reaching a Datalayer deployment, and saying so when it cannot be reached.

Both things pinned here failed in production the same way: something moved in
a package this one talks to, and what the user was told pointed somewhere
else entirely.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from code_sandboxes.datalayer_sandbox import DatalayerSandbox, _urls_for_run
from code_sandboxes.exceptions import SandboxConfigurationError, SandboxNotFoundError
from code_sandboxes.models import SandboxConfig

#: Importing the SDK warns — about its coming move to platformdirs, about
#: pydantic's class-based config. Neither is what these tests are about, and
#: the suite turns warnings into errors.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_execute_preserves_native_jupyter_mime_bundles():
    """A rendered image must not collapse to its ``text/plain`` fallback."""

    class Runtime:
        def execute(self, code, timeout):
            return SimpleNamespace(
                execute_response=[
                    {
                        "output_type": "display_data",
                        "data": {
                            "image/png": "iVBORw0KGgo=",
                            "text/plain": "<IPython.core.display.Image object>",
                        },
                        "metadata": {"image/png": {"width": 500}},
                    }
                ],
                # These legacy fields must not replace or duplicate the
                # native output above.
                result="<IPython.core.display.Image object>",
                stdout="",
                stderr="",
                error=None,
            )

    sandbox = DatalayerSandbox(SandboxConfig())
    sandbox._runtime = Runtime()
    sandbox._started = True

    execution = sandbox.run_code("display(image)")

    assert len(execution.results) == 1
    assert execution.results[0].data == {
        "image/png": "iVBORw0KGgo=",
        "text/plain": "<IPython.core.display.Image object>",
    }
    assert execution.results[0].extra == {"image/png": {"width": 500}}


def test_every_service_the_sdk_knows_about_points_at_the_one_origin():
    """A run serves all of its services from a single host.

    Read off the SDK rather than from a list written here: the list went
    stale when `mcp_server_url` was renamed, and every execution died on the
    unexpected keyword — a failure with no visible connection to the rename.
    """
    # The SDK arrives with the `datalayer` extra, which an install without it
    # does not have. With no signature to read there is nothing here to check,
    # so this is a skip and not a failure.
    sdk_urls = pytest.importorskip("datalayer_core.utils.urls")

    urls = _urls_for_run("https://prod1.datalayer.run/")

    services = [
        name
        for name in inspect.signature(sdk_urls.DatalayerURLs.from_environment).parameters
        if name.endswith("_url")
    ]
    assert services, "the SDK declares no service URLs; this test is testing nothing"
    for name in services:
        assert getattr(urls, name) == "https://prod1.datalayer.run"


def test_a_backend_that_cannot_be_imported_says_what_actually_failed(monkeypatch):
    """The reason, not the usual reason.

    The message named the missing package and the command that installs it
    whatever the import error said. With the package installed and one name
    moved inside it, that sent the reader to reinstall a dependency that was
    already there.
    """
    import builtins
    import sys
    import types

    # `start()` reaches for the SDK before it reaches for the backend, so in an
    # install without the `datalayer` extra the SDK is what fails first and the
    # failure under test never happens. Stand in for the SDK: its absence is a
    # different fault with its own message, and not the one pinned here.
    for name in ("datalayer_core", "datalayer_core.utils", "datalayer_core.utils.urls"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("agent_runtimes"):
            raise ImportError("cannot import name 'DEFAULT_TIME_RESERVATION'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    sandbox = DatalayerSandbox(SandboxConfig())
    with pytest.raises(SandboxConfigurationError) as raised:
        sandbox.start()

    message = str(raised.value)
    assert "DEFAULT_TIME_RESERVATION" in message
    # The old advice is still there for the case where it IS the answer.
    assert "code-sandboxes[datalayer]" in message


# ---------------------------------------------------------------------------
# Reaching a runtime that is already running
# ---------------------------------------------------------------------------


class _Runtime:
    """The shape `AgentClient` answers with, as far as this class reads it."""

    def __init__(self, uid: str, runtime_name: str, name: str = "notebook") -> None:
        self.uid = uid
        self.runtime_name = runtime_name
        self.name = name


class _Client:
    """An `AgentClient` that knows about some runtimes and nothing else."""

    def __init__(self, runtimes: list[_Runtime]) -> None:
        self.runtimes = runtimes
        self.asked: list[str] = []
        self.listed = 0

    def get_runtime(self, name: str) -> _Runtime:
        self.asked.append(name)
        for runtime in self.runtimes:
            if runtime.runtime_name == name:
                return runtime
        raise RuntimeError(f"Failed to get runtime '{name}': not found")

    def list_runtimes(self) -> list[_Runtime]:
        self.listed += 1
        return list(self.runtimes)


@pytest.fixture
def one_runtime(monkeypatch):
    """`from_id` against a client that has one runtime."""
    import agent_runtimes.client as client_module

    client = _Client([_Runtime(uid="01UID", runtime_name="sb-01arz", name="my notebook")])
    monkeypatch.setattr(client_module, "AgentClient", lambda **kwargs: client)
    return client


def test_from_id_answers_a_sandbox_connected_to_the_runtime(one_runtime):
    """It used to answer an object connected to nothing, with a comment
    saying it "would need agent-runtimes support" — which was there all
    along."""
    sandbox = DatalayerSandbox.from_id("sb-01arz")

    assert sandbox._runtime is not None
    assert sandbox._started is True
    assert sandbox._info.name == "my notebook"
    assert sandbox._info.status.value == "running"


def test_the_id_list_all_hands_out_is_one_from_id_accepts(one_runtime):
    """The round trip. `list_all` sets the sandbox id from the runtime's
    `uid`, and `AgentClient` looks up by `runtime_name` — so a `from_id` that
    took only a name would refuse the very id this class had just given the
    caller."""
    listed = list(DatalayerSandbox.list_all())
    assert len(listed) == 1

    same = DatalayerSandbox.from_id(listed[0].object_id)
    assert same._runtime.runtime_name == listed[0]._runtime.runtime_name


def test_the_sandbox_id_is_the_runtimes_uid(one_runtime):
    """`object_id` is what a caller stores and comes back with later.

    The uid rather than the pod name, because the pod name is derived and
    can be re-derived for a relaunch while the uid names this runtime and
    only ever this one.
    """
    assert DatalayerSandbox.from_id("sb-01arz").object_id == "01UID"
    assert next(iter(DatalayerSandbox.list_all())).object_id == "01UID"


def test_a_runtime_name_is_one_request_and_a_uid_falls_back_to_a_scan(one_runtime):
    DatalayerSandbox.from_id("sb-01arz")
    assert one_runtime.asked == ["sb-01arz"] and one_runtime.listed == 0

    DatalayerSandbox.from_id("01UID")
    assert one_runtime.listed == 1


def test_an_id_that_names_nothing_raises_rather_than_answering_a_dead_object(one_runtime):
    """Every call on an unconnected sandbox fails later, somewhere else, with
    a message about whatever it touched first rather than about the wrong
    id."""
    with pytest.raises(SandboxNotFoundError) as raised:
        DatalayerSandbox.from_id("nothing-like-this")
    assert raised.value.sandbox_id == "nothing-like-this"


def test_a_lookup_that_cannot_be_made_says_what_is_missing(monkeypatch):
    """No Datalayer client installed is a different problem from a wrong id,
    and the message has to say which."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("agent_runtimes"):
            raise ImportError("no agent_runtimes here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(SandboxNotFoundError) as raised:
        DatalayerSandbox.from_id("sb-01arz")
    assert "not installed" in str(raised.value)


def test_from_id_and_list_all_adopt_a_runtime_the_same_way(one_runtime):
    """Two adoptions that drifted is how `from_id` comes to return something
    that behaves unlike what iteration yields."""
    from_iteration = next(iter(DatalayerSandbox.list_all()))
    directly = DatalayerSandbox.from_id("sb-01arz")

    assert from_iteration.object_id == directly.object_id
    assert from_iteration._info.name == directly._info.name
    assert from_iteration._info.status == directly._info.status
    assert from_iteration._started == directly._started


class _FakeModel:
    """The part of a runtime's model these properties read."""

    def __init__(self, kernel_id=None):
        self.kernel_id = kernel_id


class _FakeRuntime:
    """A runtime, as `AgentClient.create_runtime` hands one back."""

    def __init__(self, ingress=None, jupyter_token=None, kernel_id=None):
        self.ingress = ingress
        self.jupyter_token = jupyter_token
        self.model = _FakeModel(kernel_id)


class TestDatalayerSandboxIngress:
    """The sandbox publishes where its kernel lives.

    A Datalayer runtime is not only somewhere the agent executes — it is a
    kernel a person may want to look at. The notebook and document surfaces in
    a browser build their own connection to the same server, and until these
    properties existed there was no address to build it from: the runtime
    started, appeared in the console, and left the editors connected to
    nothing.
    """

    def test_reports_nothing_before_it_starts(self):
        sandbox = DatalayerSandbox(token="api-key")
        assert sandbox.server_url is None
        assert sandbox.jupyter_token is None
        assert sandbox.kernel_id is None

    def test_publishes_the_runtime_ingress(self):
        sandbox = DatalayerSandbox(token="api-key")
        sandbox._runtime = _FakeRuntime(
            ingress="https://runtime.example/api/jupyter-server",
            jupyter_token="jupyter-secret",
            kernel_id="kernel-1",
        )

        assert sandbox.server_url == "https://runtime.example/api/jupyter-server"
        # Same value under the name older callers probe for.
        assert sandbox._server_url == sandbox.server_url
        assert sandbox.kernel_id == "kernel-1"

    def test_never_offers_the_api_key_as_a_jupyter_token(self):
        """The distinction this exists to keep.

        `_token` authenticates this process to Datalayer. Serving it as a
        Jupyter token would fail the connection and leak a credential in the
        same breath.
        """
        sandbox = DatalayerSandbox(token="api-key")
        sandbox._runtime = _FakeRuntime(
            ingress="https://runtime.example",
            jupyter_token="jupyter-secret",
        )

        assert sandbox.jupyter_token == "jupyter-secret"
        assert sandbox.jupyter_token != sandbox._token

    def test_follows_the_runtime_rather_than_copying_it(self):
        """A restarted runtime gets a new kernel; a copy would go stale."""
        sandbox = DatalayerSandbox(token="api-key")
        runtime = _FakeRuntime(ingress="https://runtime.example", kernel_id="k1")
        sandbox._runtime = runtime

        assert sandbox.kernel_id == "k1"
        runtime.model.kernel_id = "k2"
        assert sandbox.kernel_id == "k2"


def test_a_launched_sandbox_is_named_by_its_runtimes_uid(monkeypatch):
    """The uuid drawn at construction is a placeholder for a sandbox that
    does not exist yet. It stayed after the runtime did, so a worker sharing
    "the sandbox it launched" named Runtimes a uuid nothing had heard of."""

    class _Launched(_Runtime):
        def start(self) -> None:
            return None

    class _Launcher:
        def create_runtime(self, **kwargs):
            return _Launched(uid="01LAUNCHED", runtime_name="01launched", name=kwargs.get("name", ""))

    import agent_runtimes.client as client_module

    monkeypatch.setattr(client_module, "AgentClient", lambda **kwargs: _Launcher())
    monkeypatch.setattr(DatalayerSandbox, "create_context", lambda self, name: name)
    sandbox = DatalayerSandbox(SandboxConfig())
    placeholder = sandbox._sandbox_id
    sandbox.start()

    assert sandbox._sandbox_id == "01LAUNCHED" != placeholder
    assert sandbox._info.id == "01LAUNCHED"
