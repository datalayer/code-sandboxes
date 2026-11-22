# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A Datalayer sandbox's network: the web, plus the hosts it names.

A Datalayer sandbox reaches any public host on ports 80 and 443 only. The
``host:port`` named in ``allowed_hosts`` (the CLI's ``--egress``) are sent as
the launch's ``egress`` and added by the sandbox's firewall; the platform
refuses, in a sentence, what a sandbox may not have. A network policy the
platform cannot honour is refused before anything is launched.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.manage import DatalayerSandboxManager
from code_sandboxes.models import SandboxConfig, SandboxInfo, SandboxStatus
from code_sandboxes.sandboxes import datalayer as datalayer_sandbox
from code_sandboxes.sandboxes.datalayer.datalayer import DatalayerSandbox

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _Launched:
    def __init__(self, egress: list[str]) -> None:
        self.uid = "01LAUNCHED"
        self.runtime_name = "01launched"
        self.name = "s"
        self.egress = egress

    def start(self) -> None:
        return None


@pytest.fixture
def launcher(monkeypatch):
    import agent_runtimes.client as client_module

    asked: list[dict[str, Any]] = []

    class _Launcher:
        def create_runtime(self, **kwargs: Any) -> _Launched:
            asked.append(kwargs)
            return _Launched(list(kwargs.get("egress") or []))

    monkeypatch.setattr(client_module, "AgentClient", lambda **kwargs: _Launcher())
    monkeypatch.setattr(DatalayerSandbox, "create_context", lambda self, name: name)
    return asked


def test_the_allowed_hosts_are_the_launchs_egress(launcher):
    sandbox = DatalayerSandbox(SandboxConfig(allowed_hosts=["github.com:22"]), connect=False)
    sandbox.start()
    assert launcher[-1]["egress"] == ["github.com:22"]
    assert sandbox.info.metadata["egress"] == ["github.com:22"]


def test_no_allowed_hosts_sends_no_egress(launcher):
    DatalayerSandbox(SandboxConfig(), connect=False).start()
    assert "egress" not in launcher[-1]


@pytest.mark.parametrize("policy", ["none", "allowlist", "all"])
def test_a_policy_the_platform_cannot_honour_is_refused_before_the_launch(launcher, policy):
    sandbox = DatalayerSandbox(
        SandboxConfig(network_policy=policy, allowed_hosts=["github.com:22"]), connect=False
    )
    with pytest.raises(SandboxConfigurationError, match="ports 80 and 443"):
        sandbox.start()
    assert launcher == []


class _Sandbox:
    started: ClassVar[list[_Sandbox]] = []

    def __init__(self, config=None, token=None, run_url=None, **kwargs):
        self.config = config
        _Sandbox.started.append(self)

    def start(self):
        return None

    @property
    def info(self):
        return SandboxInfo(id="01runtime", variant="datalayer", status=SandboxStatus.RUNNING)


def test_the_managers_egress_reaches_the_config(monkeypatch):
    monkeypatch.setattr(datalayer_sandbox, "DatalayerSandbox", _Sandbox)
    DatalayerSandboxManager(token="t").create(egress=["github.com:22", "1.1.1.1:53"])
    assert _Sandbox.started[-1].config.allowed_hosts == ["github.com:22", "1.1.1.1:53"]
    DatalayerSandboxManager(token="t").create()
    assert _Sandbox.started[-1].config == SandboxConfig()


def test_the_managers_listing_says_each_sandboxs_egress():
    class _Runtime:
        uid = "01UID"
        name = "s"
        environment_name = "ai-agents-env"
        runtime_name = "01uid"
        egress: ClassVar[list[str]] = ["github.com:22"]

    info = DatalayerSandboxManager._info(_Runtime())
    assert info.metadata["egress"] == ["github.com:22"]
    _Runtime.egress = []
    assert DatalayerSandboxManager._info(_Runtime()).metadata["egress"] == []


def test_the_cli_asks_for_each_egress_of_a_datalayer_sandbox(monkeypatch):
    from typer.testing import CliRunner

    from code_sandboxes import cli

    asked: dict[str, Any] = {}

    class _Manager:
        def create(self, **kwargs: Any) -> SandboxInfo:
            asked.update(kwargs)
            return SandboxInfo(
                id="01RUNTIME",
                variant="datalayer",
                status=SandboxStatus.RUNNING,
                metadata={"egress": list(kwargs.get("egress") or []), "allowed_hosts": []},
            )

    monkeypatch.setattr(cli, "get_manager", lambda variant, **kwargs: _Manager())
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        ["create", "-v", "datalayer", "--egress", "github.com:22", "--egress", "1.1.1.1:53"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0, result.output
    assert asked["egress"] == ["github.com:22", "1.1.1.1:53"]
    assert "egress=github.com:22 1.1.1.1:53" in result.output
    assert "allowed_hosts" not in result.output
    refused = runner.invoke(cli.app, ["create", "-v", "modal", "--egress", "github.com:22"])
    assert refused.exit_code == 1 and "Datalayer sandbox" in refused.output
