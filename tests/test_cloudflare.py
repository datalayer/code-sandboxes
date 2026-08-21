# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Cloudflare sandbox, against a bridge that really answers.

The fake below is an HTTP transport, not a mock of the variant's methods: it
speaks the bridge's protocol — a server-sent-event stream of base64 chunks and
a terminal event — and runs the program the variant asks it to run. That is
what is worth testing here, because the protocol is where this variant lives:
everything else is the same `ExecutionResult` as every other variant's.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.cloudflare_sandbox import (
    _RUNNER_SOURCE,
    API_KEY_ENV_VAR,
    API_URL_ENV_VAR,
    CloudflareSandbox,
    _reply_of,
    _sse_events,
)
from code_sandboxes.exceptions import (
    SandboxConfigurationError,
    SandboxConnectionError,
)
from code_sandboxes.manage import get_manager, manageable_variants
from code_sandboxes.models import SandboxConfig, SandboxVariant
from code_sandboxes.providers import get_provider

BRIDGE_URL = "https://bridge.example.workers.dev"


def _sse(*events: tuple[str, object]) -> bytes:
    """Those events as the bridge would write them on the wire."""
    records = []
    for name, payload in events:
        if name in ("stdout", "stderr"):
            data = base64.b64encode(str(payload).encode()).decode()
        else:
            data = json.dumps(payload)
        records.append(f"event: {name}\ndata: {data}\n\n")
    # A keep-alive comment, which the reader must skip rather than choke on.
    return (":\n\n" + "".join(records)).encode()


class _FakeBridge:
    """The sandbox bridge Worker, answering over httpx's transport hook."""

    def __init__(self) -> None:
        self.sandboxes: list[str] = []
        self.deleted: list[str] = []
        self.files: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []
        self.authorization: str | None = None
        self.refuse = False

    def handle(self, request):
        import httpx

        self.requests.append((request.method, request.url.path))
        self.authorization = request.headers.get("authorization")
        if self.refuse:
            return httpx.Response(401, text="invalid key")

        path = request.url.path
        if request.method == "POST" and path == "/v1/sandbox":
            sandbox_id = f"cf-sbx-{len(self.sandboxes) + 1}"
            self.sandboxes.append(sandbox_id)
            return httpx.Response(200, json={"id": sandbox_id})
        if request.method == "DELETE" and path.count("/") == 3:
            self.deleted.append(path.rsplit("/", 1)[-1])
            return httpx.Response(204)
        if path.endswith("/running"):
            sandbox_id = path.split("/")[3]
            running = sandbox_id in self.sandboxes and sandbox_id not in self.deleted
            return httpx.Response(200, json={"running": running})
        if path.endswith("/exec"):
            return self._exec(httpx, json.loads(request.content))
        if "/file/" in path:
            name = path.split("/file/", 1)[1]
            if request.method == "PUT":
                self.files[name] = request.content
                return httpx.Response(200, json={"ok": True})
            if name not in self.files:
                return httpx.Response(404, text="no such file")
            return httpx.Response(200, content=self.files[name])
        return httpx.Response(404, text=f"no route for {path}")

    def _exec(self, httpx, body):
        """Really run the argv, and stream what it wrote."""
        argv = list(body["argv"])
        # The variant asks for `python3`; this machine's Python is the one
        # that is actually here.
        argv[0] = sys.executable
        finished = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, timeout=60, check=False
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                ("stdout", finished.stdout),
                ("stderr", finished.stderr),
                ("exit", {"exit_code": finished.returncode}),
            ),
        )


def _started(bridge: _FakeBridge | None = None, config: SandboxConfig | None = None, **kwargs):
    """A sandbox talking to the fake bridge over a real httpx client."""
    import httpx

    bridge = bridge or _FakeBridge()
    sandbox = CloudflareSandbox(config=config, api_url=BRIDGE_URL, api_key="cf_key", **kwargs)
    # The client is built in `start`; here it is built with the transport that
    # answers, so the protocol is exercised end to end.
    sandbox._client = httpx.Client(
        base_url=BRIDGE_URL,
        headers={"Authorization": "Bearer cf_key"},
        transport=httpx.MockTransport(bridge.handle),
    )
    response = sandbox._client.post("/v1/sandbox")
    sandbox._sandbox_id = response.json()["id"]
    sandbox._started = True
    sandbox._default_context = sandbox.create_context("default")
    sandbox.bridge = bridge  # for the assertions
    return sandbox


pytest.importorskip("httpx")


# --- The stream ------------------------------------------------------------


def test_the_events_of_a_stream_are_read_the_way_the_specification_says():
    stream = (
        ":keep-alive\n"
        "event: stdout\n"
        "data: aGVsbG8=\n"
        "\n"
        "event: message\n"
        "data: one\n"
        "data: two\n"
        "\n"
        "event: exit\n"
        'data: {"exit_code": 0}\n'
        "\n"
    )

    events = list(_sse_events(iter(stream.split("\n"))))

    assert events[0] == ("stdout", "aGVsbG8=")
    # Two data lines in one record are joined with a newline, not dropped.
    assert events[1] == ("message", "one\ntwo")
    assert events[2] == ("exit", '{"exit_code": 0}')


def test_the_reply_is_taken_from_the_last_line_the_process_wrote():
    """Anything the container printed on its own account comes before it."""
    stdout = "warning: something\n" + json.dumps({"status": "ok", "result": "42"})

    assert _reply_of(stdout) == {"status": "ok", "result": "42"}
    assert _reply_of("nothing json here") is None


def test_the_runner_source_is_a_program_that_answers_what_it_promises():
    request = json.dumps({"code": "print('hi'); 40 + 2"})
    finished = subprocess.run(  # noqa: S603
        [sys.executable, "-u", "-c", _RUNNER_SOURCE, request],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    reply = _reply_of(finished.stdout)
    assert reply["status"] == "ok"
    assert reply["stdout"] == "hi\n"
    assert reply["result"] == "42"


# --- Executing -------------------------------------------------------------


def test_a_snippet_runs_and_its_output_comes_back():
    sandbox = _started()
    streamed: list[str] = []

    execution = sandbox.run_code(
        "import sys\nprint('one')\nprint('bad', file=sys.stderr)",
        on_stdout=lambda message: streamed.append(message.line),
    )

    assert execution.execution_ok
    assert streamed == ["one"]
    assert [message.line for message in execution.logs.stderr] == ["bad"]


def test_the_value_of_a_trailing_expression_is_answered_with():
    execution = _started().run_code("1 + 1")

    assert execution.text == "2"


def test_a_raising_snippet_is_reported_as_the_codes_error_not_the_sandboxs():
    execution = _started().run_code("1 / 0")

    assert execution.execution_ok
    assert execution.code_error is not None
    assert execution.code_error.name == "ZeroDivisionError"


def test_each_snippet_runs_in_a_process_of_its_own():
    """This variant cannot hold a namespace; the docs say so, and so does this."""
    sandbox = _started()

    sandbox.run_code("x = 41")
    execution = sandbox.run_code("print(x)")

    assert execution.code_error is not None
    assert execution.code_error.name == "NameError"


def test_an_environment_asked_for_reaches_the_snippet():
    execution = _started().run_code("import os\nprint(os.environ['TOKEN'])", envs={"TOKEN": "shhh"})

    assert [message.line for message in execution.logs.stdout] == ["shhh"]


def test_a_bridge_that_refuses_the_key_says_which_variable_to_check():
    sandbox = _started()
    sandbox.bridge.refuse = True

    execution = sandbox.run_code("1 + 1")

    assert not execution.execution_ok
    assert API_KEY_ENV_VAR in (execution.execution_error or "")


def test_only_python_is_offered():
    with pytest.raises(ValueError, match="only supports Python"):
        _started().run_code("console.log(1)", language="javascript")


# --- Files and lifetime ----------------------------------------------------


def test_files_go_through_the_bridge_not_through_the_code():
    sandbox = _started()

    sandbox._write_file("/workspace/notes.txt", b"hello")

    assert sandbox._read_file("/workspace/notes.txt") == b"hello"
    assert ("PUT", f"/v1/sandbox/{sandbox._sandbox_id}/file/workspace/notes.txt") in (
        sandbox.bridge.requests
    )


def test_a_file_that_is_not_there_is_not_read_as_an_empty_one():
    with pytest.raises(FileNotFoundError):
        _started()._read_file("/workspace/missing.txt")


def test_a_sandbox_says_whether_it_is_still_up():
    sandbox = _started()

    assert sandbox.is_running()

    sandbox.stop()

    assert not sandbox.is_running()


def test_stopping_destroys_the_container():
    sandbox = _started()
    bridge, sandbox_id = sandbox.bridge, sandbox._sandbox_id

    sandbox.stop()

    assert bridge.deleted == [sandbox_id]
    assert not sandbox.is_started


def test_the_configured_environment_reaches_every_snippet():
    """`env_vars` has nowhere to go at creation, so it rides with each call."""
    sandbox = _started(config=SandboxConfig(env_vars={"TOKEN": "from-config"}))

    execution = sandbox.run_code("import os\nprint(os.environ['TOKEN'])")

    assert [message.line for message in execution.logs.stdout] == ["from-config"]


def test_a_per_call_environment_wins_over_the_configured_one():
    sandbox = _started(config=SandboxConfig(env_vars={"TOKEN": "from-config", "KEEP": "yes"}))

    execution = sandbox.run_code(
        "import os\nprint(os.environ['TOKEN'], os.environ['KEEP'])",
        envs={"TOKEN": "from-call"},
    )

    assert [message.line for message in execution.logs.stdout] == ["from-call yes"]


def test_a_network_policy_that_cannot_be_honoured_is_refused_not_ignored():
    """Believing a sandbox is cut off while it is not is the failure here."""
    for policy in ("none", "allowlist"):
        sandbox = CloudflareSandbox(
            SandboxConfig(network_policy=policy, allowed_hosts=["pypi.org"]),
            api_url=BRIDGE_URL,
            api_key="cf_key",
        )
        with pytest.raises(SandboxConfigurationError, match="cannot restrict the network"):
            sandbox.start()


def test_reading_a_variable_says_there_is_no_session_rather_than_no_variable():
    """The base class reads in two executions; the first process is gone."""
    sandbox = _started()

    with pytest.raises(SandboxConfigurationError, match="no session to read"):
        sandbox.get_variable("x")


def test_text_files_go_through_the_bridge_so_they_need_no_session():
    sandbox = _started()

    sandbox.files.write("/workspace/notes.txt", "hello")

    assert sandbox.files.read("/workspace/notes.txt") == "hello"
    # One round trip each, straight at the file endpoints.
    assert ("PUT", f"/v1/sandbox/{sandbox._sandbox_id}/file/workspace/notes.txt") in (
        sandbox.bridge.requests
    )


def test_a_variable_cannot_be_set_for_a_later_snippet_and_says_why():
    """Refused rather than silently lost between two processes."""
    sandbox = _started()

    with pytest.raises(SandboxConfigurationError, match="process of its own"):
        sandbox.set_variable("payload", {"a": 1})


# --- Configuring -----------------------------------------------------------


def test_starting_without_a_bridge_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv(API_URL_ENV_VAR, raising=False)
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)

    with pytest.raises(SandboxConfigurationError, match="sandbox bridge"):
        CloudflareSandbox().start()


def test_a_gpu_that_cannot_be_given_is_refused_rather_than_ignored():
    """Running on a CPU while looking as though it asked for a GPU is worse."""
    sandbox = CloudflareSandbox(SandboxConfig(gpu="H100"), api_url=BRIDGE_URL, api_key="k")

    with pytest.raises(SandboxConfigurationError, match="no GPU"):
        sandbox.start()


def test_the_bridge_is_read_from_the_environment_when_it_is_not_given(monkeypatch):
    monkeypatch.setenv(API_URL_ENV_VAR, f"{BRIDGE_URL}/")
    monkeypatch.setenv(API_KEY_ENV_VAR, "from-env")

    sandbox = CloudflareSandbox()

    # The trailing slash is dropped: the paths this variant builds start with one.
    assert sandbox._api_url == BRIDGE_URL
    assert sandbox._api_key == "from-env"


def test_the_client_sends_the_key_it_was_given_as_a_bearer_token():
    """Every authenticated call rides on this header; a placeholder here would
    make the bridge refuse all of them."""
    client = CloudflareSandbox(api_url=BRIDGE_URL, api_key="cf_secret_value").build_client()

    try:
        assert client.headers["authorization"] == "Bearer cf_secret_value"
    finally:
        client.close()

    # And no header at all when there is no key: a bridge run locally for
    # development has none, and `Bearer ` alone would be a wrong answer.
    anonymous = CloudflareSandbox(api_url=BRIDGE_URL, api_key="").build_client()
    try:
        assert "authorization" not in anonymous.headers
    finally:
        anonymous.close()


def test_a_refused_call_names_the_bridge_and_the_key():
    sandbox = _started()
    sandbox.bridge.refuse = True

    with pytest.raises(SandboxConnectionError, match=API_KEY_ENV_VAR):
        sandbox._write_file("/workspace/x", b"1")


# --- Registration ----------------------------------------------------------


def test_the_variant_is_registered_everywhere_a_variant_is_named():
    assert SandboxVariant.CLOUDFLARE.value == "cloudflare"
    assert isinstance(Sandbox.create(variant="cloudflare"), CloudflareSandbox)
    assert [env.name for env in Sandbox.list_environments(variant="cloudflare")] == [
        "cloudflare-default",
    ]
    assert get_provider("cloudflare") is not None
    assert "cloudflare" in manageable_variants()
    assert get_manager("cloudflare").variant == "cloudflare"


def test_the_provider_says_what_it_needs():
    provider = get_provider("cloudflare")

    assert provider.extra == "cloudflare"
    assert not provider.is_available({})
    # Both halves: a URL without a key is a bridge that will refuse it.
    assert not provider.is_available({API_URL_ENV_VAR: BRIDGE_URL})
    assert provider.is_available({API_URL_ENV_VAR: BRIDGE_URL, API_KEY_ENV_VAR: "k"})


def test_the_manager_deletes_by_id_without_creating_a_container_to_do_it():
    """A container made merely to hold a client is a container left billed."""
    import httpx

    from code_sandboxes import cloudflare_sandbox

    bridge = _FakeBridge()
    bridge.sandboxes.append("cf-sbx-existing")
    manager = get_manager("cloudflare", api_url=BRIDGE_URL, api_key="cf_key")
    # Every client the manager builds answers from the fake bridge.
    original = cloudflare_sandbox.CloudflareSandbox.build_client
    cloudflare_sandbox.CloudflareSandbox.build_client = lambda self: httpx.Client(
        base_url=BRIDGE_URL,
        headers={"Authorization": "Bearer cf_key"},
        transport=httpx.MockTransport(bridge.handle),
    )
    try:
        assert manager.get("cf-sbx-existing").status.value == "running"
        assert manager.delete("cf-sbx-existing") is True
        assert manager.get("cf-sbx-existing").status.value == "stopped"
    finally:
        cloudflare_sandbox.CloudflareSandbox.build_client = original

    # Nothing was created along the way: only the one that was already there.
    assert bridge.sandboxes == ["cf-sbx-existing"]
    assert ("POST", "/v1/sandbox") not in bridge.requests
    assert bridge.deleted == ["cf-sbx-existing"]


def test_the_manager_says_it_cannot_list_rather_than_answering_none():
    """ "None" and "cannot know" are different facts, and must read differently."""
    manager = get_manager("cloudflare")

    assert "list" not in manager.capabilities
    with pytest.raises(Exception, match="list"):
        manager.list()
