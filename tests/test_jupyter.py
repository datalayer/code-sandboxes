# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""jupyter sandbox tests."""

import os
import sys
import types
from pathlib import Path

import pytest

from code_sandboxes.jupyter_sandbox import JupyterSandbox
from code_sandboxes.models import SandboxConfig


def test_explicit_kernel_id_wins_over_reuse(monkeypatch):
    """Explicit kernel_id takes precedence even when reuse is enabled."""

    captured: dict[str, object] = {}

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None):
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

    sandbox = JupyterSandbox(
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

    sandbox = JupyterSandbox(
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
    """JupyterSandbox forwards client_kwargs to JupyterKernelClient."""

    captured: dict[str, object] = {}

    class _KernelClientStub:
        def __init__(self, server_url, token, kernel_id, client_kwargs=None):
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

    sandbox = JupyterSandbox(
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


class TestJupyterSandbox:
    """Tests for JupyterSandbox."""

    def test_local_jupyter_persistence(self, tmp_path: Path):
        """Test persistence across requests in jupyter sandbox."""
        if os.environ.get("RUN_JUPYTER_TESTS") != "1":
            pytest.skip("Set RUN_JUPYTER_TESTS=1 to enable jupyter tests")
        try:
            import jupyter_server  # noqa: F401
        except Exception:
            pytest.skip("jupyter_server is not available")

        sandbox = JupyterSandbox(config=SandboxConfig(working_dir=str(tmp_path)))
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
