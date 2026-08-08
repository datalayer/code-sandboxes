# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for CodeSandboxClient."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from code_sandboxes.client import CodeSandboxClient, execution_result_to_reply
from code_sandboxes.models import (
    CodeError,
    ExecutionResult,
    Logs,
    OutputMessage,
    Result,
    SandboxEnvironment,
)


class _FakeSandbox:
    def __init__(self):
        self._started = False
        self._variables = {}
        self._tool_caller = None
        self.config = SimpleNamespace(variant="kaggle")
        self.info = SimpleNamespace(metadata={"kernel_id": "execution-id"})
        self.sandbox_id = "sandbox-id"

    @property
    def is_started(self):
        return self._started

    def start(self):
        self._started = True

    async def start_async(self):
        self.start()

    def stop(self):
        self._started = False

    def run_code(self, code: str, language: str = "python", timeout=None, envs=None):
        _ = (code, language, timeout, envs)
        return ExecutionResult(
            execution_ok=True,
            execution_count=2,
            logs=Logs(stdout=[OutputMessage(line="hello")]),
            results=[Result(data={"text/plain": "42"}, is_main_result=True)],
        )

    def get_variable(self, name):
        return self._variables[name]

    def set_variable(self, name, value):
        self._variables[name] = value

    def set_variables(self, variables):
        self._variables.update(variables)

    def interrupt(self):
        return True

    def register_tool_caller(self, caller):
        self._tool_caller = caller

    @classmethod
    def list_environments(cls):
        return [SandboxEnvironment(name="fake", title="Fake", language="python")]

    def run_code_streaming(self, code: str, language: str = "python", timeout=None, envs=None):
        _ = (code, language, timeout, envs)
        yield OutputMessage(line="hello", timestamp=0.0, error=False)
        yield Result(data={"text/plain": "42"}, is_main_result=True, extra={})
        yield CodeError(name="ValueError", value="boom", traceback="")

    async def run_code_streaming_async(
        self, code: str, language: str = "python", timeout=None, envs=None
    ):
        _ = (code, language, timeout, envs)
        for item in self.run_code_streaming(code, language=language, timeout=timeout, envs=envs):
            await asyncio.sleep(0)
            yield item


def test_execute_code_streaming_proxies_sandbox_events():
    client = CodeSandboxClient(_FakeSandbox())

    events = list(client.execute_code_streaming("print('hi')"))

    assert isinstance(events[0], OutputMessage)
    assert events[0].line == "hello"
    assert isinstance(events[1], Result)
    assert events[1].text == "42"
    assert isinstance(events[2], CodeError)
    assert events[2].name == "ValueError"


@pytest.mark.asyncio
async def test_execute_code_streaming_async_proxies_sandbox_events():
    client = CodeSandboxClient(_FakeSandbox())

    events = []
    async for item in client.execute_code_streaming_async("print('hi')"):
        events.append(item)

    assert len(events) == 3
    assert isinstance(events[0], OutputMessage)
    assert isinstance(events[1], Result)
    assert isinstance(events[2], CodeError)


def test_execute_returns_variant_neutral_reply_and_metadata():
    client = CodeSandboxClient(_FakeSandbox())

    reply = client.execute("print('hello')")

    assert client.id == "execution-id"
    assert client.info is not None
    assert client.config.variant == "kaggle"
    assert client.kernel_info == {"language_info": {"name": "python"}}
    assert reply == {
        "execution_count": 2,
        "outputs": [
            {"output_type": "stream", "name": "stdout", "text": "hello\n"},
            {
                "output_type": "execute_result",
                "data": {"text/plain": "42"},
                "metadata": {},
            },
        ],
        "status": "ok",
    }


def test_variables_interrupt_and_restart_delegate_to_sandbox():
    sandbox = _FakeSandbox()
    client = CodeSandboxClient(sandbox)

    client.set_variable("one", 1)
    client.set_variables({"two": 2})

    def tool_caller():
        return None

    client.register_tool_caller(tool_caller)
    assert client.get_variable("one") == 1
    assert client.get_variable("two") == 2
    assert sandbox._tool_caller is tool_caller
    assert client.interrupt() is True

    client.restart()
    assert client.is_alive() is True


def test_execution_error_converts_to_error_output():
    execution = ExecutionResult(
        execution_ok=True,
        execution_count=3,
        code_error=CodeError(name="ValueError", value="boom", traceback="line 1\nline 2"),
    )

    reply = execution_result_to_reply(execution)

    assert reply["status"] == "error"
    assert reply["outputs"] == [
        {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "boom",
            "traceback": ["line 1", "line 2"],
        }
    ]


class _FakeKernelClient:
    """Minimal kernel client that records how it was stopped."""

    def __init__(self):
        self.stopped_with = None

    def stop(self, shutdown_kernel=True):
        self.stopped_with = shutdown_kernel


class _FakeKernelBackedSandbox(_FakeSandbox):
    """Sandbox exposing a borrowed kernel client, like colab/kaggle variants."""

    def __init__(self):
        super().__init__()
        self.kernel_client = _FakeKernelClient()

    def mark_stopped(self):
        self._started = False


def test_stop_without_shutdown_disconnects_backend_and_clears_started():
    sandbox = _FakeKernelBackedSandbox()
    client = CodeSandboxClient(sandbox)
    client.start()
    assert client.is_started is True

    client.stop(shutdown_kernel=False)

    # The borrowed kernel is disconnected, not shut down...
    assert sandbox.kernel_client.stopped_with is False
    # ...and the sandbox no longer claims to be started, so a later start()
    # reconnects instead of reusing the closed backend.
    assert client.is_started is False
    client.start()
    assert client.is_started is True
