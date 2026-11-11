# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The agentcore variant and the AgentCore browser session, against a stubbed
boto3 client: every call's parameters asserted.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.exceptions import UnknownServiceError  # noqa: E402
from botocore.stub import Stubber  # noqa: E402

try:
    boto3.client("bedrock-agentcore", region_name="us-east-1")
except UnknownServiceError:  # pragma: no cover - an older botocore
    pytest.skip("botocore has no AgentCore service model", allow_module_level=True)

from code_sandboxes import Sandbox  # noqa: E402
from code_sandboxes.exceptions import (  # noqa: E402
    SandboxConfigurationError,
    SandboxExecutionError,
)
from code_sandboxes.manage import get_manager  # noqa: E402
from code_sandboxes.models import SandboxConfig, SandboxStatus  # noqa: E402
from code_sandboxes.providers import get_provider  # noqa: E402
from code_sandboxes.sandboxes.agentcore import (  # noqa: E402
    DEFAULT_BROWSER,
    DEFAULT_CODE_INTERPRETER,
    AgentCoreBrowser,
    AgentCoreSandbox,
    list_sessions,
)

REGION = "us-east-1"
SESSION = "01M74ZEAJR9QZ5EZNZ5E8S0DB8"
NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)


class _EventStreamStubber(Stubber):
    """A Stubber whose responses may carry an event stream as a list of events.

    botocore validates a stubbed response against the output shape, where an
    event stream is a structure; the real client answers with an iterable of
    events. Everything else in the response is still validated.
    """

    def _validate_operation_response(self, operation_name: str, service_response: Any) -> None:
        if isinstance(service_response.get("stream"), list):
            rest = {key: value for key, value in service_response.items() if key != "stream"}
            shape = self.client.meta.service_model.operation_model(operation_name).output_shape
            for name in shape.required_members:
                if name != "stream" and name not in rest:
                    raise AssertionError(f"{operation_name} answer misses {name}")
            return
        super()._validate_operation_response(operation_name, service_response)


@pytest.fixture()
def data() -> Iterator[tuple[Any, Stubber]]:
    client = boto3.client(
        "bedrock-agentcore",
        region_name=REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    with _EventStreamStubber(client) as stub:
        yield client, stub
        stub.assert_no_pending_responses()


def _started(stub: Stubber, name: str = "check", timeout: int | None = None) -> None:
    expected: dict[str, Any] = {
        "codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER,
        "name": name,
    }
    if timeout is not None:
        expected["sessionTimeoutSeconds"] = timeout
    stub.add_response(
        "start_code_interpreter_session",
        {
            "codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER,
            "sessionId": SESSION,
            "createdAt": NOW,
        },
        expected,
    )


def _stopped(stub: Stubber) -> None:
    stub.add_response(
        "stop_code_interpreter_session",
        {
            "codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER,
            "sessionId": SESSION,
            "lastUpdatedAt": NOW,
        },
        {"codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER, "sessionId": SESSION},
    )


def _invoked(
    stub: Stubber, name: str, arguments: dict[str, Any], events: list[dict[str, Any]]
) -> None:
    stub.add_response(
        "invoke_code_interpreter",
        {"sessionId": SESSION, "stream": events},
        {
            "codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER,
            "sessionId": SESSION,
            "name": name,
            "arguments": arguments,
        },
    )


def _executed(
    stub: Stubber, code: str, stdout: str = "", stderr: str = "", exit_code: int = 0
) -> None:
    _invoked(
        stub,
        "executeCode",
        {"code": code, "language": "python"},
        [
            {
                "result": {
                    "content": [{"type": "text", "text": stdout or stderr}],
                    "structuredContent": {
                        "stdout": stdout,
                        "stderr": stderr,
                        "exitCode": exit_code,
                    },
                    "isError": exit_code != 0,
                }
            }
        ],
    )


def _sandbox(client: Any, session_timeout: float | None = None, **config: Any) -> AgentCoreSandbox:
    return AgentCoreSandbox(
        config=SandboxConfig(name="check", **config),
        session_timeout=session_timeout,
        client=client,
    )


class TestTheCodeInterpreter:
    def test_a_session_runs_code_and_is_stopped(self, data):
        client, stub = data
        _started(stub, timeout=600)
        _executed(stub, "print(6*7)", stdout="42")
        _stopped(stub)

        sandbox = _sandbox(client, session_timeout=600)
        sandbox.start()
        assert sandbox.info.id == SESSION
        assert sandbox.info.variant == "agentcore"
        assert sandbox.info.metadata["code_interpreter_id"] == DEFAULT_CODE_INTERPRETER
        result = sandbox.run_code("print(6*7)")
        assert result.success
        assert result.stdout == "42"
        assert result.execution_count == 1
        sandbox.stop()
        assert not sandbox.is_started
        assert sandbox.info.status == SandboxStatus.STOPPED
        sandbox.stop()  # a second stop calls nothing

    def test_the_factory_makes_it(self):
        assert isinstance(Sandbox.create(variant="agentcore"), AgentCoreSandbox)
        assert isinstance(Sandbox.create(variant="AgentCore"), AgentCoreSandbox)
        [environment] = Sandbox.list_environments(variant="agentcore")
        assert environment.metadata["code_interpreter_id"] == DEFAULT_CODE_INTERPRETER

    def test_a_raised_error_is_a_code_error_with_its_name(self, data):
        client, stub = data
        _started(stub)
        traceback = (
            "-----\nValueError                                Traceback (most recent call last)\n"
            'Cell In[6], line 1\n----> 1 raise ValueError("boom")\n\nValueError: boom'
        )
        _executed(stub, 'raise ValueError("boom")', stderr=traceback, exit_code=1)
        seen = []
        sandbox = _sandbox(client)
        sandbox.start()
        result = sandbox.run_code('raise ValueError("boom")', on_error=seen.append)
        assert result.execution_ok
        assert result.code_error.name == "ValueError"
        assert result.code_error.value == "boom"
        assert result.code_error.traceback == traceback
        assert seen == [result.code_error]
        assert result.stderr == ""

    def test_stderr_of_a_run_that_succeeded_is_stderr(self, data):
        client, stub = data
        _started(stub)
        _executed(stub, "warn()", stdout="a\nb", stderr="careful")
        sandbox = _sandbox(client)
        sandbox.start()
        out, err = [], []
        result = sandbox.run_code("warn()", on_stdout=out.append, on_stderr=err.append)
        assert result.code_error is None
        assert [m.line for m in out] == ["a", "b"]
        assert [m.line for m in err] == ["careful"]
        assert result.stderr == "careful"

    def test_an_exception_in_the_stream_is_an_execution_failure(self, data):
        client, stub = data
        _started(stub)
        _invoked(
            stub,
            "executeCode",
            {"code": "1", "language": "python"},
            [{"throttlingException": {"message": "slow down"}}],
        )
        sandbox = _sandbox(client)
        sandbox.start()
        result = sandbox.run_code("1")
        assert not result.execution_ok
        assert "ThrottlingException: slow down" in result.execution_error

    def test_an_image_comes_back_as_a_result(self, data):
        client, stub = data
        _started(stub)
        _invoked(
            stub,
            "executeCode",
            {"code": "plot()", "language": "python"},
            [
                {
                    "result": {
                        "content": [{"type": "image", "data": b"\x89PNG", "mimeType": "image/png"}],
                        "structuredContent": {"stdout": "", "stderr": "", "exitCode": 0},
                        "isError": False,
                    }
                }
            ],
        )
        sandbox = _sandbox(client)
        sandbox.start()
        [image] = sandbox.run_code("plot()").results
        assert image.data == {"image/png": "iVBORw=="}

    def test_javascript_is_passed_as_its_language(self, data):
        client, stub = data
        _started(stub)
        _invoked(
            stub,
            "executeCode",
            {"code": "console.log(42)", "language": "javascript"},
            [{"result": {"content": [], "structuredContent": {"stdout": "42", "exitCode": 0}}}],
        )
        sandbox = _sandbox(client)
        sandbox.start()
        assert sandbox.run_code("console.log(42)", language="javascript").stdout == "42"

    def test_files_go_through_the_file_tools(self, data):
        client, stub = data
        _started(stub)
        _invoked(
            stub,
            "writeFiles",
            {"content": [{"path": "a.txt", "text": "hello"}]},
            [{"result": {"content": [{"type": "text", "text": "Successfully wrote all 1 files"}]}}],
        )
        _invoked(
            stub,
            "writeFiles",
            {"content": [{"path": "b.bin", "blob": b"\xff\x00"}]},
            [{"result": {"content": [{"type": "text", "text": "ok"}]}}],
        )
        _invoked(
            stub,
            "readFiles",
            {"paths": ["a.txt"]},
            [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {
                                    "type": "text",
                                    "uri": "file:///a.txt",
                                    "text": "hello",
                                },
                            }
                        ]
                    }
                }
            ],
        )
        _invoked(
            stub,
            "readFiles",
            {"paths": ["missing.txt"]},
            [{"result": {"content": [{"type": "text", "text": "No such file"}], "isError": True}}],
        )
        sandbox = _sandbox(client)
        sandbox.start()
        sandbox._write_file("a.txt", b"hello")
        sandbox._write_file("b.bin", b"\xff\x00")
        assert sandbox._read_file("a.txt") == b"hello"
        with pytest.raises(SandboxExecutionError, match="No such file"):
            sandbox._read_file("missing.txt")

    def test_variables_cross_as_json(self, data):
        client, stub = data
        _started(stub)
        set_code = (
            "import json as _code_sandboxes_json\n"
            "x = _code_sandboxes_json.loads('[1, 2]')\n"
            "del _code_sandboxes_json\n"
        )
        get_code = (
            "import json as _code_sandboxes_json\n"
            "print(_code_sandboxes_json.dumps(__result__, default=repr))\n"
            "del _code_sandboxes_json\n"
        )
        _executed(stub, set_code)
        _executed(stub, "__result__ = x")
        _executed(stub, get_code, stdout="[1, 2]")
        sandbox = _sandbox(client)
        sandbox.start()
        sandbox.set_variable("x", [1, 2])
        assert sandbox.get_variable("x") == [1, 2]

    def test_what_cannot_be_honoured_is_refused_before_a_session(self, data):
        client, _ = data
        with pytest.raises(SandboxConfigurationError, match="no GPU"):
            _sandbox(client, gpu="T4").start()
        with pytest.raises(SandboxConfigurationError, match="no host allowlist"):
            _sandbox(client, network_policy="allowlist").start()
        with pytest.raises(SandboxConfigurationError, match="PUBLIC network mode"):
            _sandbox(client, network_policy="all").start()
        with pytest.raises(SandboxConfigurationError, match="between 1 and 28800"):
            _sandbox(client, session_timeout=30_000).start()

    def test_one_kernel_per_session(self, data):
        client, stub = data
        _started(stub)
        sandbox = _sandbox(client)
        sandbox.start()
        with pytest.raises(SandboxConfigurationError, match="one kernel"):
            sandbox.create_context("other")
        with pytest.raises(ValueError, match="python, javascript, typescript"):
            sandbox.run_code("puts 1", language="ruby")
        with pytest.raises(SandboxConfigurationError, match="environment variables"):
            sandbox.run_code("1", envs={"A": "1"})

    def test_a_region_is_required(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        with pytest.raises(SandboxConfigurationError, match="AWS region"):
            AgentCoreSandbox(config=SandboxConfig(name="check")).start()

    def test_it_takes_no_interrupt(self, data):
        client, _ = data
        assert _sandbox(client)._do_interrupt() is False


class TestTheManager:
    def test_list_create_delete(self, data):
        client, stub = data
        listed = {
            "items": [
                {
                    "codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER,
                    "sessionId": SESSION,
                    "name": "check",
                    "status": "READY",
                    "createdAt": NOW,
                }
            ]
        }
        stub.add_response(
            "list_code_interpreter_sessions",
            listed,
            {"codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER, "status": "READY"},
        )
        _started(stub)
        stub.add_response(
            "list_code_interpreter_sessions",
            listed,
            {"codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER, "status": "READY"},
        )
        _stopped(stub)
        stub.add_response(
            "list_code_interpreter_sessions",
            {"items": []},
            {"codeInterpreterIdentifier": DEFAULT_CODE_INTERPRETER, "status": "READY"},
        )

        manager = get_manager("agentcore", client=client)
        [info] = manager.list()
        assert (info.id, info.status, info.name) == (SESSION, SandboxStatus.RUNNING, "check")
        assert info.created_at == NOW.timestamp()
        assert manager.create(name="check").id == SESSION
        assert manager.delete(SESSION) is True
        assert manager.delete(SESSION) is False

    def test_the_provider_is_listed(self):
        provider = get_provider("agentcore")
        assert provider is not None
        assert provider.extra == "agentcore"
        assert [e.name for e in provider.environments()] == ["agentcore-code-interpreter"]


class TestTheBrowser:
    def test_a_session_is_started_signed_and_stopped(self, data):
        client, stub = data
        automation = f"wss://bedrock-agentcore.{REGION}.amazonaws.com/browser-streams/{DEFAULT_BROWSER}/sessions/{SESSION}/automation"
        live = f"https://bedrock-agentcore.{REGION}.amazonaws.com/browser-streams/{DEFAULT_BROWSER}/sessions/{SESSION}/live-view"
        stub.add_response(
            "start_browser_session",
            {
                "browserIdentifier": DEFAULT_BROWSER,
                "sessionId": SESSION,
                "createdAt": NOW,
                "streams": {
                    "automationStream": {"streamEndpoint": automation, "streamStatus": "ENABLED"},
                    "liveViewStream": {"streamEndpoint": live},
                },
            },
            {
                "browserIdentifier": DEFAULT_BROWSER,
                "name": "check",
                "sessionTimeoutSeconds": 300,
                "viewPort": {"width": 1280, "height": 720},
            },
        )
        stub.add_response(
            "stop_browser_session",
            {"browserIdentifier": DEFAULT_BROWSER, "sessionId": SESSION, "lastUpdatedAt": NOW},
            {"browserIdentifier": DEFAULT_BROWSER, "sessionId": SESSION},
        )
        with AgentCoreBrowser(
            name="check", session_timeout=300, viewport=(1280, 720), client=client
        ) as browser:
            assert browser.session_id == SESSION
            assert browser.automation_endpoint == automation
            headers = browser.automation_headers()
            assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=testing/")
            assert f"/{REGION}/bedrock-agentcore/aws4_request" in headers["Authorization"]
            assert headers["Host"] == f"bedrock-agentcore.{REGION}.amazonaws.com"
            assert "X-Amz-Date" in headers
            query = parse_qs(urlparse(browser.live_view_url(expires=120)).query)
            assert query["X-Amz-Expires"] == ["120"]
            assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert browser.session_id is None
        assert browser.automation_endpoint is None

    def test_sessions_are_listed_page_by_page(self, data):
        client, stub = data
        item = {
            "browserIdentifier": DEFAULT_BROWSER,
            "sessionId": SESSION,
            "status": "READY",
            "createdAt": NOW,
        }
        stub.add_response(
            "list_browser_sessions",
            {"items": [item], "nextToken": "next"},
            {"browserIdentifier": DEFAULT_BROWSER, "status": "READY"},
        )
        stub.add_response(
            "list_browser_sessions",
            {"items": [item]},
            {"browserIdentifier": DEFAULT_BROWSER, "status": "READY", "nextToken": "next"},
        )
        assert len(list_sessions("browser", DEFAULT_BROWSER, client)) == 2
        with pytest.raises(ValueError, match="code-interpreter and browser"):
            list_sessions("runtime", "x", client)
