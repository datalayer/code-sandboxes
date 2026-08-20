# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for the Typer REPL CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from code_sandboxes import cli as sandbox_cli
from code_sandboxes.models import ExecutionResult, Logs, Result


class _FakeSandbox:
    def __init__(self):
        self.sandbox_id = "sandbox-123"
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exited = True

    def run_code(self, code: str):
        return ExecutionResult(
            results=[Result(data={"text/plain": f"echo:{code}"}, is_main_result=True)],
            logs=Logs(),
            execution_ok=True,
        )


def test_repl_jupyter_variant_uses_random_port(monkeypatch):
    runner = CliRunner()
    captured: dict = {}
    fake_sandbox = _FakeSandbox()

    def _fake_create(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_sandbox

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))

    result = runner.invoke(sandbox_cli.app, ["repl", "--variant", "jupyter-server"], input=":exit\n")

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "jupyter-server"
    assert captured["kwargs"]["port"] == 0
    assert fake_sandbox.exited is True


def test_repl_colab_prompts_and_forwards_credentials(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))

    # Prompts: server_url, kernel_id, proxy_token, then repl command.
    user_input = "https://colab-host.example\nkernel-abc\nproxy-xyz\n:exit\n"
    result = runner.invoke(
        sandbox_cli.app,
        ["repl", "--variant", "google-colab"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "google_colab"
    assert captured["kwargs"]["server_url"] == "https://colab-host.example"
    assert captured["kwargs"]["kernel_id"] == "kernel-abc"
    assert captured["kwargs"]["proxy_token"] == "proxy-xyz"  # noqa: S105


def test_repl_kaggle_prompts_and_forwards_credentials(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "env-token")

    # Prompts: server_url, kernel_id, then repl command.
    user_input = "https://kaggle-host.example/proxy\nkernel-abc\n:exit\n"
    result = runner.invoke(sandbox_cli.app, ["repl", "--variant", "kaggle"], input=user_input)

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "kaggle"
    assert captured["kwargs"]["server_url"] == "https://kaggle-host.example/proxy"
    assert captured["kwargs"]["kernel_id"] == "kernel-abc"
    assert captured["kwargs"]["token"] == "env-token"  # noqa: S105


def test_repl_kaggle_creates_kernel_without_kernel_id(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "env-token")

    # Prompts: server_url, empty kernel_id (create new), then repl command.
    user_input = "https://kaggle-host.example/proxy\n\n:exit\n"
    result = runner.invoke(sandbox_cli.app, ["repl", "--variant", "kaggle"], input=user_input)

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "kaggle"
    assert captured["kwargs"]["server_url"] == "https://kaggle-host.example/proxy"
    assert "kernel_id" not in captured["kwargs"]
    assert captured["kwargs"]["token"] == "env-token"  # noqa: S105


def test_root_defaults_to_jupyter_repl(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))

    result = runner.invoke(sandbox_cli.app, [], input=":exit\n")

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "jupyter-server"
    assert captured["kwargs"]["port"] == 0


def test_repl_modal_gpu_is_forwarded(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))

    result = runner.invoke(
        sandbox_cli.app,
        ["repl", "--variant", "modal", "--gpu", "A100"],
        input=":exit\n",
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "modal"
    assert captured["kwargs"]["gpu"] == "A100"


def test_repl_kaggle_gpu_is_forwarded(monkeypatch):
    runner = CliRunner()
    captured: dict = {}

    def _fake_create(*args, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeSandbox()

    monkeypatch.setattr(sandbox_cli.Sandbox, "create", staticmethod(_fake_create))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "env-token")

    # Prompts: server_url, empty kernel_id (create new), then repl command.
    user_input = "https://kaggle-host.example/proxy\n\n:exit\n"
    result = runner.invoke(
        sandbox_cli.app,
        ["repl", "--variant", "kaggle", "--gpu", "T4"],
        input=user_input,
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["variant"] == "kaggle"
    assert captured["kwargs"]["gpu"] == "T4"
