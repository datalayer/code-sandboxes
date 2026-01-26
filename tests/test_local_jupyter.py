# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Local jupyter sandbox tests."""

import os
from pathlib import Path

import pytest

from code_sandboxes.local.jupyter_sandbox import LocalJupyterSandbox
from code_sandboxes.models import SandboxConfig


class TestLocalJupyterSandbox:
    """Tests for LocalJupyterSandbox."""

    def test_local_jupyter_persistence(self, tmp_path: Path):
        """Test persistence across requests in local-jupyter sandbox."""
        if os.environ.get("RUN_LOCAL_JUPYTER_TESTS") != "1":
            pytest.skip("Set RUN_LOCAL_JUPYTER_TESTS=1 to enable local-jupyter tests")
        try:
            import jupyter_server  # noqa: F401
        except Exception:
            pytest.skip("jupyter_server is not available")

        sandbox = LocalJupyterSandbox(config=SandboxConfig(working_dir=str(tmp_path)))
        try:
            sandbox.start()
        except Exception as exc:
            pytest.skip(f"local-jupyter sandbox not available: {exc}")

        try:
            sandbox.run_code("x = 7")
            execution = sandbox.run_code("x + 1")
            assert "8" in execution.results[0].data.get("text/plain", "")
        finally:
            sandbox.stop()
