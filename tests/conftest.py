# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Pytest configuration and fixtures for code-sandboxes tests."""

from pathlib import Path

import pytest

from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.models import SandboxConfig


@pytest.fixture
def sandbox():
    """Create a local eval sandbox for testing."""
    sandbox = EvalSandbox()
    sandbox.start()
    yield sandbox
    sandbox.stop()


@pytest.fixture
def sandbox_with_config():
    """Create a sandbox with custom configuration."""
    config = SandboxConfig(timeout=30.0)
    sandbox = EvalSandbox(config=config)
    sandbox.start()
    yield sandbox
    sandbox.stop()


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory."""
    work_path = tmp_path / "workspace"
    work_path.mkdir()
    return work_path
