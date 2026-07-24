# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Sandbox factory tests."""

import pytest

from code_sandboxes.base import Sandbox, SandboxVariant
from code_sandboxes.colab_sandbox import ColabSandbox
from code_sandboxes.datalayer_sandbox import DatalayerSandbox
from code_sandboxes.docker_sandbox import DockerSandbox
from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.jupyter_sandbox import JupyterSandbox
from code_sandboxes.kaggle_sandbox import KaggleSandbox
from code_sandboxes.modal_sandbox import ModalSandbox
from code_sandboxes.models import SandboxConfig
from code_sandboxes.monty_sandbox import MontySandbox


class TestSandboxFactory:
    """Tests for Sandbox.create factory method."""

    def test_create_local_eval(self):
        """Test creating eval sandbox."""
        sandbox = Sandbox.create(variant="eval")

        assert sandbox is not None
        assert isinstance(sandbox, EvalSandbox)

    def test_create_local_jupyter(self):
        """Test creating jupyter sandbox."""
        sandbox = Sandbox.create(variant=SandboxVariant.JUPYTER)

        assert sandbox is not None
        assert isinstance(sandbox, JupyterSandbox)

    def test_create_with_config(self):
        """Test creating sandbox with config."""
        config = SandboxConfig(timeout=120.0)
        sandbox = Sandbox.create(variant="eval", config=config)

        assert sandbox.config.timeout == 120.0

    def test_create_with_timeout(self):
        """Test creating sandbox with timeout parameter."""
        sandbox = Sandbox.create(variant="eval", timeout=90.0)

        assert sandbox.config.timeout == 90.0

    def test_create_with_env(self):
        """Test creating sandbox with environment variables."""
        config = SandboxConfig(env_vars={"MY_VAR": "my_value"})
        sandbox = Sandbox.create(
            variant="eval",
            config=config,
        )

        assert sandbox.config.env_vars.get("MY_VAR") == "my_value"

    def test_create_invalid_variant(self):
        """Test error for invalid variant."""
        with pytest.raises(ValueError):
            Sandbox.create(variant="invalid-variant")

    @pytest.mark.parametrize(
        "variant,expected_type",
        [
            ("eval", EvalSandbox),
            ("jupyter", JupyterSandbox),
            ("docker", DockerSandbox),
            ("datalayer", DatalayerSandbox),
            ("colab", ColabSandbox),
            ("kaggle", KaggleSandbox),
            ("monty", MontySandbox),
            ("modal", ModalSandbox),
        ],
    )
    def test_create_all_supported_variants(self, variant, expected_type):
        """Test that all supported variants resolve to the expected sandbox class."""
        sandbox = Sandbox.create(variant=variant)
        assert isinstance(sandbox, expected_type)

    def test_create_default_variant_is_datalayer(self):
        """Test that omitting variant uses the datalayer sandbox by default."""
        sandbox = Sandbox.create()
        assert isinstance(sandbox, DatalayerSandbox)

    def test_create_colab_forwards_connection_kwargs(self):
        """Test that Colab-specific connection kwargs are propagated."""
        sandbox = Sandbox.create(
            variant="colab",
            server_url="https://colab-host.example",
            kernel_id="kernel-id",
            proxy_token="proxy-token",  # noqa: S106
            channels_url=(
                "wss://colab-host.example/api/kernels/kernel-id/channels"
                "?colab-runtime-proxy-token=proxy-token"
            ),
            client_agent="agent-name",
        )
        assert isinstance(sandbox, ColabSandbox)
        assert sandbox._server_url == "https://colab-host.example"
        assert sandbox._kernel_id == "kernel-id"
        assert sandbox._proxy_token == "proxy-token"  # noqa: S105
        assert sandbox._channels_url.startswith("wss://colab-host.example")

    def test_create_kaggle_forwards_connection_kwargs(self):
        """Test that Kaggle-specific connection kwargs are propagated."""
        sandbox = Sandbox.create(
            variant="kaggle",
            server_url="https://kaggle-host.example/proxy",
            kernel_id="kernel-id",
            token="api-token",  # noqa: S106
        )
        assert isinstance(sandbox, KaggleSandbox)
        assert sandbox._server_url == "https://kaggle-host.example/proxy"
        assert sandbox._kernel_id == "kernel-id"
        assert sandbox._token == "api-token"  # noqa: S105

    def test_create_datalayer_forwards_runtime_kwargs(self):
        """Test that datalayer-specific kwargs are propagated."""
        sandbox = Sandbox.create(
            variant="datalayer",
            token="api-token",  # noqa: S106
            run_url="https://run.example",
            snapshot_name="snap-1",
        )
        assert isinstance(sandbox, DatalayerSandbox)
        assert sandbox._token == "api-token"  # noqa: S105
        assert sandbox._run_url == "https://run.example"
        assert sandbox._snapshot_name == "snap-1"
