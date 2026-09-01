# Copyright (c) 2023-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""Output arrives while the code is still running, not after it finishes.

The base `run_code_streaming_async` is streaming in shape only: it awaits the
whole execution and then replays what it collected. Everything above it was
therefore unable to stream however well it was written — the A2UI surface built
correct incremental messages and emitted all of them microseconds apart, once
the run was over.

These tests are about *when* items arrive, because that is the whole property.
A test that only checked what arrived passed against the replay.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from code_sandboxes.jupyter_server_sandbox import JupyterServerSandbox
from code_sandboxes.models import CodeError, ExecutionResult, Logs, OutputMessage, Result


class _SlowSandbox(JupyterServerSandbox):
    """A sandbox whose `run_code` emits over time, like a real kernel."""

    def __init__(self, *, fail: bool = False) -> None:  # noqa: D107
        self.fail = fail

    def run_code(  # type: ignore[override]
        self,
        code: str,
        language: str = "python",
        context=None,
        on_stdout=None,
        on_stderr=None,
        on_result=None,
        on_error=None,
        envs=None,
        timeout=None,
    ) -> ExecutionResult:
        for index in range(3):
            time.sleep(0.15)
            if on_stdout:
                on_stdout(
                    OutputMessage(line=f"tick {index}", timestamp=time.time(), error=False)
                )
        if on_result:
            on_result(Result(data={"text/plain": "42"}, is_main_result=True, extra={}))
        if self.fail:
            return ExecutionResult(
                execution_ok=False,
                execution_error="the sandbox fell over",
                logs=Logs(),
            )
        return ExecutionResult(execution_ok=True, logs=Logs())


@pytest.mark.asyncio
async def test_items_arrive_while_the_code_is_still_running() -> None:
    sandbox = _SlowSandbox()
    started = time.monotonic()
    arrivals: list[tuple[float, object]] = []

    async for item in sandbox.run_code_streaming_async("irrelevant"):
        arrivals.append((time.monotonic() - started, item))

    assert len(arrivals) == 4  # three lines and the result

    # The first line lands well before the run ends. Against the replay
    # implementation every arrival is at the end and this is the assertion
    # that fails.
    first_at, _ = arrivals[0]
    last_at, _ = arrivals[-1]
    assert first_at < 0.3, f"first item arrived at {first_at:.2f}s"
    assert last_at - first_at > 0.2, "everything arrived at once"


@pytest.mark.asyncio
async def test_the_items_themselves_are_unchanged() -> None:
    sandbox = _SlowSandbox()
    items = [item async for item in sandbox.run_code_streaming_async("irrelevant")]

    lines = [i.line for i in items if isinstance(i, OutputMessage)]
    assert lines == ["tick 0", "tick 1", "tick 2"]
    assert any(isinstance(i, Result) for i in items)


@pytest.mark.asyncio
async def test_an_infrastructure_failure_is_reported_once_at_the_end() -> None:
    # It never reaches the callbacks — being the reason there were none — so
    # the generator has to add it after the execution returns.
    sandbox = _SlowSandbox(fail=True)
    items = [item async for item in sandbox.run_code_streaming_async("irrelevant")]

    errors = [i for i in items if isinstance(i, CodeError)]
    assert len(errors) == 1
    assert errors[0].value == "the sandbox fell over"


@pytest.mark.asyncio
async def test_a_raising_execution_does_not_hang_the_consumer() -> None:
    """The sentinel closes the generator whatever happened."""

    class _Exploding(_SlowSandbox):
        def run_code(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await asyncio.wait_for(
            _drain(_Exploding().run_code_streaming_async("irrelevant")), timeout=5
        )


async def _drain(generator) -> list[object]:
    return [item async for item in generator]
