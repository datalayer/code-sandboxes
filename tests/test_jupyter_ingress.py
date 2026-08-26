# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Provider-independent Jupyter ingress preparation and credentials."""

# ruff: noqa: S106

from types import SimpleNamespace

from code_sandboxes import JupyterServerOptions
from code_sandboxes.daytona_sandbox import DaytonaSandbox
from code_sandboxes.e2b_sandbox import E2BSandbox
from code_sandboxes.jupyter_ingress import preparation_command
from code_sandboxes.modal_sandbox import ModalSandbox


class _Result:
    exit_code = 0
    result = ""
    stderr = ""


class _Process:
    returncode = 0
    stderr = SimpleNamespace(read=lambda: "")

    def wait(self):
        return 0


def test_preparation_checks_before_installing_and_waits_for_readiness():
    command = preparation_command(JupyterServerOptions(port=9999, token="secret"))

    assert "import jupyter_server, ipykernel" in command
    assert "|| python -m pip install" in command
    assert "--ServerApp.port=9999" in command
    assert "socket.create_connection(('127.0.0.1', 9999)" in command


def test_daytona_uses_preview_ingress_and_caches_preparation():
    calls = []
    remote = SimpleNamespace(
        process=SimpleNamespace(exec=lambda command, timeout: calls.append(command) or _Result()),
        get_preview_link=lambda port: SimpleNamespace(
            url=f"https://daytona.example/{port}", token="preview-secret"
        ),
    )
    sandbox = DaytonaSandbox()
    sandbox._sandbox = remote
    sandbox._started = True

    endpoint = sandbox.prepare_jupyter_server(JupyterServerOptions(token="jupyter-secret"))
    again = sandbox.prepare_jupyter_server()

    assert endpoint is again
    assert len(calls) == 1
    assert endpoint.websocket_url == "wss://daytona.example/8888"
    assert endpoint.headers == {"X-Daytona-Preview-Token": "preview-secret"}
    assert endpoint.query == {"token": "jupyter-secret"}


def test_e2b_uses_traffic_access_token():
    calls = []
    remote = SimpleNamespace(
        commands=SimpleNamespace(run=lambda command, timeout: calls.append(command) or _Result()),
        get_host=lambda port: f"{port}-sandbox.e2b.app",
        traffic_access_token="traffic-secret",
    )
    sandbox = E2BSandbox()
    sandbox._sandbox = remote
    sandbox._started = True

    endpoint = sandbox.prepare_jupyter_server(JupyterServerOptions(token="jupyter-secret"))

    assert len(calls) == 1
    assert endpoint.http_url == "https://8888-sandbox.e2b.app"
    assert endpoint.headers == {"E2B-Traffic-Access-Token": "traffic-secret"}


def test_modal_keeps_connect_and_jupyter_credentials_separate():
    calls = []
    remote = SimpleNamespace(
        exec=lambda *args, **kwargs: calls.append((args, kwargs)) or _Process(),
        create_connect_token=lambda port: SimpleNamespace(
            url=f"https://modal.example/{port}/", token="connect-secret"
        ),
    )
    sandbox = ModalSandbox()
    sandbox._sandbox = remote
    sandbox._started = True

    endpoint = sandbox.prepare_jupyter_server(JupyterServerOptions(token="jupyter-secret"))

    assert len(calls) == 1
    assert endpoint.headers == {"Authorization": "Bearer connect-secret"}
    assert endpoint.query == {"token": "jupyter-secret"}
