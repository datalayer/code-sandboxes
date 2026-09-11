# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Datalayer manager: what `create` asks for reaches the runtime, and `delete` stops it.

Both CLIs reach the platform through `DatalayerSandboxManager`, and until
2026-09-11 neither could choose an environment or a name, nor delete a
runtime. The keywords fell into `DatalayerSandbox`'s extra arguments, so every
runtime started in `ai-agents-env` under a generated name, and `delete` called
a client method that does not exist and answered "nothing deleted".
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from code_sandboxes import datalayer_sandbox
from code_sandboxes.manage import DatalayerSandboxManager, SandboxManagementError
from code_sandboxes.models import SandboxConfig, SandboxInfo, SandboxStatus


class _Sandbox:
    started: ClassVar[list[_Sandbox]] = []

    def __init__(self, config=None, token=None, run_url=None, **kwargs):
        self.config = config
        self.kwargs = kwargs
        _Sandbox.started.append(self)

    def start(self):
        return None

    @property
    def info(self):
        return SandboxInfo(
            id="01runtime",
            variant="datalayer",
            status=SandboxStatus.RUNNING,
            name=self.config.name or "",
        )


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    _Sandbox.started.clear()
    monkeypatch.setattr(datalayer_sandbox, "DatalayerSandbox", _Sandbox)


@pytest.mark.parametrize("key", ["environment_name", "environment"])
def test_create_starts_the_environment_asked_for_under_its_name(key):
    DatalayerSandboxManager(token="t").create(**{key: "python-cpu-env"}, name="e0-11-check")
    sandbox = _Sandbox.started[-1]
    assert sandbox.config.environment == "python-cpu-env"
    assert sandbox.config.name == "e0-11-check"
    assert sandbox.kwargs == {}


def test_create_with_nothing_asked_keeps_the_defaults():
    DatalayerSandboxManager(token="t").create()
    assert _Sandbox.started[-1].config == SandboxConfig()


def test_a_gpu_asked_for_reaches_the_config():
    DatalayerSandboxManager(token="t").create(environment="ai-env", gpu="A100")
    assert _Sandbox.started[-1].config.gpu == "A100"


class _Client:
    def __init__(self, answer=True, error=None):
        self.answer = answer
        self.error = error
        self.stopped = []

    def stop_runtime(self, runtime):
        if self.error:
            raise self.error
        self.stopped.append(runtime)
        return self.answer


def _manager_with(client):
    manager = DatalayerSandboxManager(token="t")
    manager._client = client
    return manager


def test_delete_stops_the_runtime_by_its_uid():
    client = _Client()
    assert _manager_with(client).delete("01m28m0b8f822s8487n5vgrv2n") is True
    assert client.stopped == ["01m28m0b8f822s8487n5vgrv2n"]


def test_delete_answers_false_when_the_platform_did_not_stop_it():
    assert _manager_with(_Client(answer=False)).delete("01x") is False


def test_a_failed_delete_says_why():
    with pytest.raises(SandboxManagementError, match="did not stop runtime 01x: unreachable"):
        _manager_with(_Client(error=RuntimeError("unreachable"))).delete("01x")
