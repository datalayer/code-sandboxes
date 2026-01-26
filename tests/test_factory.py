# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Sandbox factory tests."""

import pytest

from code_sandboxes.base import Sandbox, SandboxVariant
from code_sandboxes.local.eval_sandbox import LocalEvalSandbox
from code_sandboxes.local.jupyter_sandbox import LocalJupyterSandbox
from code_sandboxes.models import SandboxConfig


class TestSandboxFactory:
    """Tests for Sandbox.create factory method."""

    def test_create_local_eval(self):
        """Test creating local-eval sandbox."""
        sandbox = Sandbox.create(variant="local-eval")

        assert sandbox is not None
        assert isinstance(sandbox, LocalEvalSandbox)

    def test_create_local_jupyter(self):
        """Test creating local-jupyter sandbox."""
        sandbox = Sandbox.create(variant=SandboxVariant.LOCAL_JUPYTER)

        assert sandbox is not None
        assert isinstance(sandbox, LocalJupyterSandbox)

    def test_create_with_config(self):
        """Test creating sandbox with config."""
        config = SandboxConfig(timeout=120.0)
        sandbox = Sandbox.create(variant="local-eval", config=config)

        assert sandbox.config.timeout == 120.0

    def test_create_with_timeout(self):
        """Test creating sandbox with timeout parameter."""
        sandbox = Sandbox.create(variant="local-eval", timeout=90.0)

        assert sandbox.config.timeout == 90.0

    def test_create_with_env(self):
        """Test creating sandbox with environment variables."""
        config = SandboxConfig(env_vars={"MY_VAR": "my_value"})
        sandbox = Sandbox.create(
            variant="local-eval",
            config=config,
        )

        assert sandbox.config.env_vars.get("MY_VAR") == "my_value"

    def test_create_invalid_variant(self):
        """Test error for invalid variant."""
        with pytest.raises(ValueError):
            Sandbox.create(variant="invalid-variant")
