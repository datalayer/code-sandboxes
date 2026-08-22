# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""jupyter sandbox tests."""

import os
import sys
import time
import types
import uuid
from pathlib import Path

import pytest

from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.jupyter_server_sandbox import JupyterServerSandbox
from code_sandboxes.models import SandboxConfig


def test_explicit_kernel_id_wins_over_reuse(monkeypatch):
    """Explicit kernel_id takes precedence even when reuse is enabled."""

    captured: dict[str, object] = {}

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None):
            self.id = kernel_id or "started-kernel"
            captured["server_url"] = server_url
            captured["token"] = token
            captured["kernel_id"] = kernel_id
            captured["client_kwargs"] = client_kwargs

        def start(self, path=None):
            captured["path"] = path

        def stop(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=_KernelClientStub),
    )

    sandbox = JupyterServerSandbox(
        server_url="http://localhost:8888",
        kernel_id="explicit-kernel",
        reuse_kernel=True,
    )

    monkeypatch.setattr(sandbox, "_wait_for_server", lambda timeout=None: None)

    def _should_not_be_called():
        raise AssertionError("_find_existing_kernel should not be called with explicit kernel_id")

    monkeypatch.setattr(sandbox, "_find_existing_kernel", _should_not_be_called)

    sandbox.start()
    try:
        assert captured["kernel_id"] == "explicit-kernel"
    finally:
        sandbox.stop()


def test_reuse_kernel_false_forces_new_kernel(monkeypatch):
    """When reuse_kernel is False and no kernel_id is provided, connect with kernel_id=None."""

    captured: dict[str, object] = {}

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None):
            self.id = kernel_id or "started-kernel"
            captured["kernel_id"] = kernel_id

        def start(self, path=None):
            return None

        def stop(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=_KernelClientStub),
    )

    sandbox = JupyterServerSandbox(
        server_url="http://localhost:8888",
        kernel_id=None,
        reuse_kernel=False,
    )

    monkeypatch.setattr(sandbox, "_wait_for_server", lambda timeout=None: None)

    def _should_not_be_called():
        raise AssertionError("_find_existing_kernel should not be called when reuse_kernel=False")

    monkeypatch.setattr(sandbox, "_find_existing_kernel", _should_not_be_called)

    sandbox.start()
    try:
        assert captured["kernel_id"] is None
    finally:
        sandbox.stop()


def test_kernel_client_forwards_client_kwargs(monkeypatch, tmp_path: Path):
    """JupyterServerSandbox forwards client_kwargs to JupyterKernelClient."""

    captured: dict[str, object] = {}

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None):
            self.id = kernel_id or "started-kernel"
            captured["server_url"] = server_url
            captured["token"] = token
            captured["kernel_id"] = kernel_id
            captured["client_kwargs"] = client_kwargs

        def start(self, path=None):
            captured["start_path"] = path

        def stop(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=_KernelClientStub),
    )

    notebook_path = str(tmp_path / "notebook.ipynb")

    sandbox = JupyterServerSandbox(
        server_url="http://localhost:8888",
        kernel_id="kernel-1",
        kernel_path=notebook_path,
        client_kwargs={"reconnect_interval": 5},
        reuse_kernel=False,
    )

    monkeypatch.setattr(sandbox, "_wait_for_server", lambda timeout=None: None)

    sandbox.start()
    try:
        assert captured["kernel_id"] == "kernel-1"
        assert captured.get("client_kwargs") == {"reconnect_interval": 5}
        assert captured.get("start_path") == notebook_path
    finally:
        sandbox.stop()


class TestJupyterServerSandbox:
    """Tests for JupyterServerSandbox."""

    def test_local_jupyter_persistence(self, tmp_path: Path):
        """Test persistence across requests in jupyter sandbox."""
        if os.environ.get("RUN_JUPYTER_TESTS") != "1":
            pytest.skip("Set RUN_JUPYTER_TESTS=1 to enable jupyter tests")
        try:
            import jupyter_server  # noqa: F401
        except Exception:
            pytest.skip("jupyter_server is not available")

        sandbox = JupyterServerSandbox(config=SandboxConfig(working_dir=str(tmp_path)))
        try:
            sandbox.start()
        except Exception as exc:
            pytest.skip(f"jupyter sandbox not available: {exc}")

        try:
            sandbox.run_code("x = 7")
            execution = sandbox.run_code("x + 1")
            assert "8" in execution.results[0].data.get("text/plain", "")
        finally:
            sandbox.stop()


def _kernel_client_stub(captured: dict):
    """Build a JupyterKernelClient stub that records the kwargs it was built with."""

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None, **kwargs):
            self.id = kernel_id or "started-kernel"
            captured["server_url"] = server_url
            captured["token"] = token
            captured["kernel_id"] = kernel_id
            captured["client_kwargs"] = client_kwargs
            captured["headers"] = kwargs.get("headers")

        def start(self, path=None):
            captured["start_path"] = path

        def stop(self):
            return None

    return _KernelClientStub


def test_headers_are_forwarded_to_the_kernel_client(monkeypatch):
    """Extra headers reach the kernel client, for cookie/XSRF authenticated servers."""

    captured: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=_kernel_client_stub(captured)),
    )

    auth_headers = {"Cookie": "username-localhost=abc; _xsrf=tok", "X-XSRFToken": "tok"}
    sandbox = JupyterServerSandbox(
        server_url="http://localhost:8888",
        token=None,
        kernel_id="kernel-1",
        reuse_kernel=False,
        headers=auth_headers,
    )

    monkeypatch.setattr(sandbox, "_wait_for_server", lambda timeout=None: None)

    sandbox.start()
    try:
        assert captured["headers"] == auth_headers
    finally:
        sandbox.stop()


def test_no_headers_kwarg_when_none_supplied(monkeypatch):
    """Without headers the kernel client is built exactly as before."""

    captured: dict[str, object] = {}
    monkeypatch.setitem(
        sys.modules,
        "jupyter_kernel_client",
        types.SimpleNamespace(JupyterKernelClient=_kernel_client_stub(captured)),
    )

    credential = uuid.uuid4().hex
    sandbox = JupyterServerSandbox(
        server_url="http://localhost:8888",
        token=credential,
        kernel_id="kernel-1",
        reuse_kernel=False,
    )

    monkeypatch.setattr(sandbox, "_wait_for_server", lambda timeout=None: None)

    sandbox.start()
    try:
        assert captured["headers"] is None
        assert captured["token"] == credential
    finally:
        sandbox.stop()


def test_external_server_keeps_token_none():
    """An external server with no token must not get a fabricated one.

    Password-authenticated deployments have no token to send; generating one
    would put a credential the server never issued on every request.
    """

    sandbox = JupyterServerSandbox(server_url="http://localhost:8888", token=None)

    assert sandbox._token is None


def test_owned_server_still_generates_a_token():
    """A sandbox that starts its own server still needs a token to secure it."""

    sandbox = JupyterServerSandbox()

    assert sandbox._token


class TestAServerThatWillNotStart:
    """What the user is told when the Jupyter Server never comes up.

    Both streams went to `DEVNULL`, so a server that died on the way up —
    a module not installed, a port already taken — explained itself into
    nothing, and the caller waited out the whole timeout to be told
    "Timed out waiting for Jupyter Server". The reason was there all along.
    """

    def _sandbox(self, returncode, said):
        from collections import deque

        sandbox = JupyterServerSandbox.__new__(JupyterServerSandbox)
        sandbox._server_url = "http://127.0.0.1:1"
        sandbox._token = "not-a-secret"  # noqa: S105 - a stand-in, not a credential
        sandbox._headers = {}
        sandbox._server_output = deque(said)
        sandbox._server_process = type(
            "P", (), {"poll": lambda self: returncode, "returncode": returncode}
        )()
        return sandbox

    def test_a_dead_server_is_reported_at_once_with_what_it_said(self):
        sandbox = self._sandbox(1, ["ModuleNotFoundError: No module named 'jupyter_server'"])

        started = time.time()
        with pytest.raises(SandboxConfigurationError) as raised:
            sandbox._wait_for_server(timeout=30)

        assert time.time() - started < 5, "it waited out the timeout"
        assert "exited with code 1" in str(raised.value)
        assert "No module named 'jupyter_server'" in str(raised.value)

    def test_a_server_still_running_is_waited_for_and_then_quoted(self):
        sandbox = self._sandbox(None, ["[W] something looked wrong"])

        with pytest.raises(SandboxConfigurationError) as raised:
            sandbox._wait_for_server(timeout=1)

        assert "Timed out" in str(raised.value)
        assert "something looked wrong" in str(raised.value)
