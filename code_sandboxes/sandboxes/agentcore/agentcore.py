# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Amazon Bedrock AgentCore sandboxes: the Code Interpreter and the Browser.

AgentCore offers two managed sandboxes, each reached through the data plane
(`bedrock-agentcore` in boto3) with the person's own AWS credentials:

- the **Code Interpreter** — a session of a code interpreter, by default the
  AWS-managed ``aws.codeinterpreter.v1``. `AgentCoreSandbox` drives it with
  ``StartCodeInterpreterSession``, ``InvokeCodeInterpreter`` (``executeCode``,
  ``writeFiles``, ``readFiles``) and ``StopCodeInterpreterSession``. A session
  keeps one Python kernel, so ``x = 1`` in one call and ``print(x)`` in the
  next behave as in every other variant of this package.
- the **Browser** — a headless Chrome session of a browser, by default the
  AWS-managed ``aws.browser.v1``. It runs no code of ours, so it is not a
  `Sandbox`: `AgentCoreBrowser` starts and stops a session and hands out what
  a client needs to drive it — the automation stream (Chrome DevTools
  Protocol over a SigV4-signed WebSocket, what Playwright's
  ``connect_over_cdp`` takes) and the live view a person watches.

What AgentCore answers is translated, never guessed at: its interpreter
returns one result per call carrying ``stdout``, ``stderr`` and an exit code,
with the value of a last expression printed into ``stdout`` (it has no
separate channel for it), and an error as a traceback in ``stderr``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import uuid
from typing import Any
from urllib.parse import urlparse, urlunparse

from ...base import Sandbox, marks_execution
from ...exceptions import (
    SandboxConfigurationError,
    SandboxExecutionError,
    SandboxNotStartedError,
    VariableNotFoundError,
)
from ...models import (
    CodeError,
    Context,
    ExecutionResult,
    Logs,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
)

logger = logging.getLogger(__name__)

#: The AWS-managed code interpreter a session is started on when the caller
#: names none. It runs in AgentCore's SANDBOX network mode: no internet.
DEFAULT_CODE_INTERPRETER = "aws.codeinterpreter.v1"

#: The AWS-managed browser a session is started on when the caller names none.
DEFAULT_BROWSER = "aws.browser.v1"

#: The boto3 service of AgentCore's data plane, and the SigV4 service name.
DATA_PLANE_SERVICE = "bedrock-agentcore"

#: The languages AgentCore's interpreter runs.
LANGUAGES = ("python", "javascript", "typescript")

#: AgentCore keeps a session for at most eight hours.
MAX_SESSION_TIMEOUT = 28_800

#: What AgentCore calls the life of a session, in the words used here.
SESSION_STATES = {
    "READY": SandboxStatus.RUNNING,
    "TERMINATED": SandboxStatus.TERMINATED,
}

#: The exceptions AgentCore may send inside the response stream.
_STREAM_EXCEPTIONS = (
    "accessDeniedException",
    "conflictException",
    "internalServerException",
    "resourceNotFoundException",
    "serviceQuotaExceededException",
    "throttlingException",
    "validationException",
)

#: The last line of a Python traceback: ``ValueError: boom``.
_ERROR_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit|Warning)?): ?(.*)$")


def _import_boto3() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise SandboxConfigurationError(
            "boto3 is required for the agentcore variant. Install it with: "
            "pip install code-sandboxes[agentcore]"
        ) from exc
    return boto3


def resolve_region(region: str | None = None) -> str:
    """The AWS region: the one given, else ``AWS_REGION``, else ``AWS_DEFAULT_REGION``.

    Refused when none says: a session started in a region nobody chose is
    one nobody finds again to stop.
    """
    chosen = (
        region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
    ).strip()
    if not chosen:
        raise SandboxConfigurationError(
            "The agentcore variant needs an AWS region: pass region= or set AWS_REGION."
        )
    return chosen


def data_plane_client(region: str | None = None) -> Any:
    """A boto3 client of AgentCore's data plane, with the caller's own credentials."""
    return _import_boto3().client(DATA_PLANE_SERVICE, region_name=resolve_region(region))


def _session_name(name: str | None) -> str:
    """A session name AgentCore takes (1 to 100 characters)."""
    value = (name or "").strip() or f"code-sandboxes-{uuid.uuid4().hex[:8]}"
    return value[:100]


def _session_timeout(seconds: float | None) -> int | None:
    """A session timeout AgentCore takes, in whole seconds, or None for its default."""
    if not seconds:
        return None
    whole = round(seconds)
    if whole < 1 or whole > MAX_SESSION_TIMEOUT:
        raise SandboxConfigurationError(
            f"AgentCore keeps a session between 1 and {MAX_SESSION_TIMEOUT} seconds, not {whole}."
        )
    return whole


def _stream_failure(event: dict[str, Any]) -> str | None:
    """The exception AgentCore sent inside the stream, in a sentence, if any."""
    for key in _STREAM_EXCEPTIONS:
        if key in event:
            message = (event.get(key) or {}).get("message") or ""
            return f"{key[0].upper()}{key[1:]}: {message}".rstrip(": ")
    return None


def _results_of(stream: Any) -> tuple[list[dict[str, Any]], str | None]:
    """The results of one invocation, and the failure the stream carried, if any."""
    results: list[dict[str, Any]] = []
    for event in stream:
        failure = _stream_failure(event)
        if failure:
            return results, failure
        result = event.get("result")
        if isinstance(result, dict):
            results.append(result)
    return results, None


def _code_error(stderr: str) -> CodeError:
    """The error a failed execution raised, read off its traceback."""
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    last = lines[-1].strip() if lines else ""
    match = _ERROR_LINE.match(last)
    if match:
        return CodeError(name=match.group(1), value=match.group(2), traceback=stderr)
    return CodeError(name="Error", value=last, traceback=stderr)


def _lines(
    text: str | None, error: bool, handler: OutputHandler[OutputMessage] | None
) -> list[OutputMessage]:
    """One stream of an execution, line by line, each handed to its handler."""
    now = time.time()
    messages = [
        OutputMessage(line=line, timestamp=now, error=error) for line in (text or "").splitlines()
    ]
    if handler:
        for message in messages:
            handler(message)
    return messages


def _images(raw: dict[str, Any], handler: OutputHandler[Result] | None) -> list[Result]:
    """The images one result carried, base64-encoded and keyed by MIME type."""
    results: list[Result] = []
    for block in raw.get("content") or []:
        data = block.get("data")
        if block.get("type") != "image" or data is None:
            continue
        encoded = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
        result = Result(data={block.get("mimeType") or "image/png": encoded})
        if handler:
            handler(result)
        results.append(result)
    return results


class AgentCoreSandbox(Sandbox):
    """A session of an AgentCore code interpreter.

    Args:
        config: Optional sandbox configuration. ``name`` names the session.
        region: The AWS region; ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` when
            omitted.
        code_interpreter_id: The code interpreter to start the session on.
            :data:`DEFAULT_CODE_INTERPRETER` when omitted; a code interpreter
            of your own (created with ``CreateCodeInterpreter``, for instance
            in PUBLIC network mode) by its id.
        session_timeout: Seconds before AgentCore ends the session, at most
            eight hours; AgentCore's own default (15 minutes) when omitted.
            Not ``max_lifetime``, whose default of a day AgentCore cannot keep.
        client: A boto3 ``bedrock-agentcore`` client to use instead of one
            made from the default credentials.
    """

    def __init__(
        self,
        config: SandboxConfig | None = None,
        region: str | None = None,
        code_interpreter_id: str | None = None,
        session_timeout: float | None = None,
        client: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(config)
        self._region = region
        self._session_timeout = session_timeout
        self._code_interpreter_id = code_interpreter_id or DEFAULT_CODE_INTERPRETER
        self._client = client
        self._session_id: str | None = None
        self._execution_count = 0
        self._extra_kwargs = kwargs

    @property
    def code_interpreter_id(self) -> str:
        """The code interpreter this sandbox's session runs on."""
        return self._code_interpreter_id

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        """One environment: the AWS-managed code interpreter.

        A code interpreter of your own is asked for by argument —
        ``code_interpreter_id=`` — the way a template is for E2B.
        """
        return [
            SandboxEnvironment(
                name="agentcore-code-interpreter",
                title="AgentCore Code Interpreter",
                language="python",
                owner="agentcore",
                visibility="cloud",
                burning_rate=0.0,
                metadata={
                    "variant": "agentcore",
                    "code_interpreter_id": DEFAULT_CODE_INTERPRETER,
                    "features": [],
                },
            ),
        ]

    def _refuse_what_cannot_be_honoured(self) -> None:
        """What a session can never do, said before anything is started."""
        if self.config.gpu:
            raise SandboxConfigurationError(
                "AgentCore's code interpreter has no GPU, so gpu="
                + repr(self.config.gpu)
                + " cannot be honoured."
            )
        policy = self.config.network_policy
        if policy == "allowlist":
            raise SandboxConfigurationError(
                "AgentCore has no host allowlist per session: the network is the "
                "code interpreter's own mode (SANDBOX or PUBLIC)."
            )
        if policy == "all" and self._code_interpreter_id == DEFAULT_CODE_INTERPRETER:
            raise SandboxConfigurationError(
                f"{DEFAULT_CODE_INTERPRETER} reaches no internet (SANDBOX network "
                "mode). Create a code interpreter in PUBLIC network mode and pass "
                "its id as code_interpreter_id."
            )
        _session_timeout(self._session_timeout)

    def _data_plane(self) -> Any:
        if self._client is None:
            self._client = data_plane_client(self._region)
        return self._client

    def start(self) -> None:
        if self._started:
            return
        self._refuse_what_cannot_be_honoured()
        client = self._data_plane()
        params: dict[str, Any] = {
            "codeInterpreterIdentifier": self._code_interpreter_id,
            "name": _session_name(self.config.name),
        }
        timeout = _session_timeout(self._session_timeout)
        if timeout is not None:
            params["sessionTimeoutSeconds"] = timeout
        answer = client.start_code_interpreter_session(**params)
        self._session_id = answer["sessionId"]
        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._session_id,
            variant="agentcore",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=params["name"],
            metadata={
                "code_interpreter_id": self._code_interpreter_id,
                "session_id": self._session_id,
                "region": client.meta.region_name,
            },
            config=self.config,
        )
        self._started = True

    def stop(self) -> None:
        # Guarded on the session itself, not `_started`: a failure after the
        # session was started must still let it be stopped.
        if self._session_id is None:
            return
        session_id = self._session_id
        self._data_plane().stop_code_interpreter_session(
            codeInterpreterIdentifier=self._code_interpreter_id,
            sessionId=session_id,
        )
        self._session_id = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def _invoke(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """One tool of the interpreter; its results, or the failure raised."""
        if not self._started or self._session_id is None:
            raise SandboxNotStartedError()
        answer = self._data_plane().invoke_code_interpreter(
            codeInterpreterIdentifier=self._code_interpreter_id,
            sessionId=self._session_id,
            name=name,
            arguments=arguments,
        )
        results, failure = _results_of(answer["stream"])
        if failure:
            raise SandboxExecutionError("AgentCoreError", failure)
        for result in results:
            if result.get("isError"):
                text = " ".join(
                    block.get("text") or ""
                    for block in result.get("content") or []
                    if block.get("type") == "text"
                )
                raise SandboxExecutionError("AgentCoreError", text or f"{name} failed")
        return results

    def create_context(self, name: str | None = None) -> Context:
        """The session's one kernel: AgentCore keeps no second namespace per session."""
        if name != "default":
            raise SandboxConfigurationError(
                "An AgentCore session has one kernel: start another sandbox for a "
                "namespace of its own."
            )
        return super().create_context("default")

    @marks_execution
    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Context | None = None,
        on_stdout: OutputHandler[OutputMessage] | None = None,
        on_stderr: OutputHandler[OutputMessage] | None = None,
        on_result: OutputHandler[Result] | None = None,
        on_error: OutputHandler[CodeError] | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        if not self._started or self._session_id is None:
            raise SandboxNotStartedError()
        if language not in LANGUAGES:
            raise ValueError(
                f"AgentCore's code interpreter runs {', '.join(LANGUAGES)}, not {language}."
            )
        if envs:
            raise SandboxConfigurationError(
                "AgentCore's code interpreter takes no environment variables per call."
            )
        del timeout  # AgentCore takes no timeout per call; the session's applies.

        started_at = time.time()
        self._execution_count += 1
        try:
            answer = self._data_plane().invoke_code_interpreter(
                codeInterpreterIdentifier=self._code_interpreter_id,
                sessionId=self._session_id,
                name="executeCode",
                arguments={"code": code, "language": language},
            )
            raw_results, failure = _results_of(answer["stream"])
            failure = f"AgentCore refused the execution: {failure}" if failure else None
        except Exception as error:
            raw_results, failure = [], f"Failed to execute code on AgentCore: {error}"
        if failure:
            # The service, not the code: reported, as every variant reports
            # an infrastructure failure, rather than raised.
            return ExecutionResult(
                execution_ok=False,
                execution_error=failure,
                started_at=started_at,
                completed_at=time.time(),
                context_id="default",
            )

        stdout: list[OutputMessage] = []
        stderr: list[OutputMessage] = []
        results: list[Result] = []
        code_error: CodeError | None = None
        for raw in raw_results:
            structured = raw.get("structuredContent") or {}
            stdout += _lines(structured.get("stdout"), False, on_stdout)
            failed = bool(raw.get("isError")) or (structured.get("exitCode") or 0) != 0
            if not failed:
                stderr += _lines(structured.get("stderr"), True, on_stderr)
            elif code_error is None:
                code_error = _code_error(structured.get("stderr") or structured.get("stdout") or "")
                if on_error:
                    on_error(code_error)
            results += _images(raw, on_result)

        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout, stderr=stderr),
            execution_ok=True,
            code_error=code_error,
            execution_count=self._execution_count,
            context_id="default",
            started_at=started_at,
            completed_at=time.time(),
        )

    def _do_interrupt(self) -> bool:
        """AgentCore's executeCode takes no interrupt; the session timeout is the stop."""
        return False

    def _get_internal_variable(self, name: str, context: Context | None = None) -> Any:
        """The value of a variable, carried back as JSON (its repr when it cannot be)."""
        execution = self.run_code(
            "import json as _code_sandboxes_json\n"
            f"print(_code_sandboxes_json.dumps({name}, default=repr))\n"
            "del _code_sandboxes_json\n",
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(
                "SandboxError", execution.execution_error or "Sandbox execution failed"
            )
        if execution.code_error is not None:
            raise VariableNotFoundError(name)
        printed = "\n".join(message.line for message in execution.logs.stdout).strip()
        if not printed:
            raise VariableNotFoundError(name)
        return json.loads(printed.splitlines()[-1])

    def _set_internal_variable(self, name: str, value: Any, context: Context | None = None) -> None:
        try:
            payload = json.dumps(value)
        except TypeError as error:
            raise SandboxConfigurationError(
                f"An AgentCore session runs elsewhere, so {name!r} has to cross as "
                "JSON and this value cannot be encoded. Build it inside the "
                "sandbox with run_code instead."
            ) from error
        execution = self.run_code(
            "import json as _code_sandboxes_json\n"
            f"{name} = _code_sandboxes_json.loads({payload!r})\n"
            "del _code_sandboxes_json\n",
        )
        if not execution.execution_ok:
            raise SandboxExecutionError(
                "SandboxError", execution.execution_error or "Sandbox execution failed"
            )
        if execution.code_error is not None:
            raise SandboxExecutionError(
                execution.code_error.name,
                execution.code_error.value,
                execution.code_error.traceback,
            )

    def _write_file(self, path: str, content: bytes) -> None:
        """Through the interpreter's ``writeFiles`` tool, not through code."""
        try:
            block: dict[str, Any] = {"path": path, "text": content.decode("utf-8")}
        except UnicodeDecodeError:
            block = {"path": path, "blob": content}
        self._invoke("writeFiles", {"content": [block]})

    def _read_file(self, path: str) -> bytes:
        """Through the interpreter's ``readFiles`` tool."""
        for result in self._invoke("readFiles", {"paths": [path]}):
            for block in result.get("content") or []:
                resource = block.get("resource") or {}
                if resource.get("blob") is not None:
                    return bytes(resource["blob"])
                if resource.get("text") is not None:
                    return str(resource["text"]).encode("utf-8")
        raise FileNotFoundError(f"Could not read file: {path}")


class AgentCoreBrowser:
    """A session of an AgentCore browser: a headless Chrome AWS runs.

    It runs no code of ours, so it is not a :class:`Sandbox`: it is started,
    driven from outside over its automation stream (the Chrome DevTools
    Protocol on a SigV4-signed WebSocket), watched through its live view, and
    stopped.

    Args:
        region: The AWS region; ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` when
            omitted.
        browser_id: The browser to start the session on;
            :data:`DEFAULT_BROWSER` when omitted.
        name: The session's name.
        session_timeout: Seconds before AgentCore ends the session (its own
            default when omitted, at most eight hours).
        viewport: ``(width, height)`` of the page, when given.
        client: A boto3 ``bedrock-agentcore`` client to use instead of one
            made from the default credentials.

    Example::

        with AgentCoreBrowser(region="us-east-1") as browser:
            endpoint, headers = browser.automation_endpoint, browser.automation_headers()
            # playwright: chromium.connect_over_cdp(endpoint, headers=headers)
    """

    def __init__(
        self,
        region: str | None = None,
        browser_id: str | None = None,
        name: str | None = None,
        session_timeout: float | None = None,
        viewport: tuple[int, int] | None = None,
        client: Any | None = None,
    ):
        self._region = region
        self.browser_id = browser_id or DEFAULT_BROWSER
        self.name = _session_name(name)
        self._timeout = _session_timeout(session_timeout)
        self._viewport = viewport
        self._client = client
        self.session_id: str | None = None
        self.automation_endpoint: str | None = None
        self.live_view_endpoint: str | None = None

    def _data_plane(self) -> Any:
        if self._client is None:
            self._client = data_plane_client(self._region)
        return self._client

    @property
    def is_started(self) -> bool:
        return self.session_id is not None

    def start(self) -> None:
        if self.session_id is not None:
            return
        params: dict[str, Any] = {"browserIdentifier": self.browser_id, "name": self.name}
        if self._timeout is not None:
            params["sessionTimeoutSeconds"] = self._timeout
        if self._viewport is not None:
            params["viewPort"] = {"width": self._viewport[0], "height": self._viewport[1]}
        answer = self._data_plane().start_browser_session(**params)
        self.session_id = answer["sessionId"]
        streams = answer.get("streams") or {}
        self.automation_endpoint = (streams.get("automationStream") or {}).get("streamEndpoint")
        self.live_view_endpoint = (streams.get("liveViewStream") or {}).get("streamEndpoint")

    def stop(self) -> None:
        if self.session_id is None:
            return
        self._data_plane().stop_browser_session(
            browserIdentifier=self.browser_id, sessionId=self.session_id
        )
        self.session_id = None
        self.automation_endpoint = None
        self.live_view_endpoint = None

    def _signer(self) -> tuple[Any, str]:
        client = self._data_plane()
        credentials = client._request_signer._credentials
        if credentials is None:
            raise SandboxConfigurationError("No AWS credentials to sign the browser stream with.")
        return credentials.get_frozen_credentials(), client.meta.region_name

    def automation_headers(self) -> dict[str, str]:
        """The SigV4-signed headers that open the automation WebSocket.

        What a CDP client (Playwright's ``connect_over_cdp``) sends with the
        handshake to :attr:`automation_endpoint`. Signed now, so asked for
        right before connecting.
        """
        if self.automation_endpoint is None:
            raise SandboxNotStartedError()
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        credentials, region = self._signer()
        parsed = urlparse(self.automation_endpoint)
        https_url = urlunparse(parsed._replace(scheme="https"))
        request = AWSRequest(method="GET", url=https_url, headers={"host": parsed.netloc})
        SigV4Auth(credentials, DATA_PLANE_SERVICE, region).add_auth(request)
        headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
        headers["Host"] = parsed.netloc
        return headers

    def live_view_url(self, expires: int = 300) -> str:
        """A presigned URL of the live view, valid ``expires`` seconds."""
        if self.live_view_endpoint is None:
            raise SandboxNotStartedError()
        from botocore.auth import SigV4QueryAuth
        from botocore.awsrequest import AWSRequest

        credentials, region = self._signer()
        request = AWSRequest(method="GET", url=self.live_view_endpoint)
        SigV4QueryAuth(credentials, DATA_PLANE_SERVICE, region, expires=expires).add_auth(request)
        return request.url

    def __enter__(self) -> AgentCoreBrowser:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def list_sessions(
    kind: str, identifier: str, client: Any, status: str | None = "READY"
) -> list[dict[str, Any]]:
    """Every session of a code interpreter (``kind="code-interpreter"``) or a browser."""
    if kind == "code-interpreter":
        call, key = client.list_code_interpreter_sessions, "codeInterpreterIdentifier"
    elif kind == "browser":
        call, key = client.list_browser_sessions, "browserIdentifier"
    else:
        raise ValueError(f"AgentCore has code-interpreter and browser sessions, not {kind!r}.")
    params: dict[str, Any] = {key: identifier}
    if status:
        params["status"] = status
    items: list[dict[str, Any]] = []
    while True:
        answer = call(**params)
        items.extend(answer.get("items") or [])
        token = answer.get("nextToken")
        if not token:
            return items
        params["nextToken"] = token
