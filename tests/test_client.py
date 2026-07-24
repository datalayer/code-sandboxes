# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for CodeSandboxClient."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from code_sandboxes.client import CodeSandboxClient
from code_sandboxes.models import CodeError, OutputMessage, Result


class _FakeSandbox:
    def __init__(self):
        self._started = False
        self.config = SimpleNamespace(variant="kaggle")

    @property
    def is_started(self):
        return self._started

    def start(self):
        self._started = True

    async def start_async(self):
        self.start()

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
