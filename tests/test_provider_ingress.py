# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Selection of real Jupyter ingress versus direct provider execution."""

from types import SimpleNamespace

from code_sandboxes import JupyterServerEndpoint, SandboxConfig
from code_sandboxes.provider_ingress import provider_ingress_execution


class _Provider:
    config = SandboxConfig()

    def __init__(self):
        self.prepared = 0

    def prepare_jupyter_server(self, options):
        self.prepared += 1
        return JupyterServerEndpoint(
            port=8888,
            http_url="https://provider.example",
            websocket_url="wss://provider.example",
            headers={"X-Provider-Token": "secret"},
            query={"token": "jupyter-secret"},
        )


def test_direct_mode_keeps_the_provider_adapter():
    provider = _Provider()

    with provider_ingress_execution(provider, direct=True) as execution:
        assert execution is provider

    assert provider.prepared == 0


def test_default_connects_a_jupyter_sandbox(monkeypatch):
    calls = []

    class _Jupyter:
        def __init__(self, **kwargs):
            calls.append(SimpleNamespace(kind="init", kwargs=kwargs))

        def start(self):
            calls.append(SimpleNamespace(kind="start"))

        def stop(self):
            calls.append(SimpleNamespace(kind="stop"))

    monkeypatch.setattr(
        "code_sandboxes.provider_ingress.JupyterServerSandbox", _Jupyter
    )
    provider = _Provider()

    with provider_ingress_execution(provider) as execution:
        assert isinstance(execution, _Jupyter)

    assert provider.prepared == 1
    assert [call.kind for call in calls] == ["init", "start", "stop"]
    assert calls[0].kwargs["server_url"] == "https://provider.example"
    assert calls[0].kwargs["headers"] == {"X-Provider-Token": "secret"}
    assert calls[0].kwargs["token"] == "jupyter-secret"
