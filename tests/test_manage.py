# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Tests for the CRUD managers and their CLI commands."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from code_sandboxes import cli as sandbox_cli
from code_sandboxes.manage import (
    DockerSandboxManager,
    KaggleSandboxManager,
    SandboxManagementError,
    get_manager,
    manageable_variants,
)
from code_sandboxes.models import SandboxInfo, SandboxStatus


def test_every_variant_has_a_manager():
    assert manageable_variants() == [
        "datalayer",
        "daytona",
        "docker",
        "eval",
        "google_colab",
        "jupyter-server",
        "kaggle",
        "modal",
        "monty",
    ]
    for variant in manageable_variants():
        assert get_manager(variant).variant == variant


def test_the_colab_spelling_with_a_dash_is_accepted(monkeypatch):
    monkeypatch.setenv("RUNTIME_URL", "https://colab.example/proxy")
    assert get_manager("google-colab").variant == "google_colab"


def test_an_unknown_variant_is_named_in_the_error():
    with pytest.raises(ValueError, match="carrier-pigeon"):
        get_manager("carrier-pigeon")


def test_the_ephemeral_variants_answer_the_truth():
    """An eval sandbox dies with its process: nothing to list, delete says why."""
    manager = get_manager("eval")
    assert manager.list() == []
    assert manager.capabilities == frozenset({"list"})
    with pytest.raises(SandboxManagementError, match="dies with it"):
        manager.delete("anything")
    with pytest.raises(SandboxManagementError, match=r"Sandbox\.create"):
        manager.create()


class _FakeContainer:
    def __init__(self, container_id, name, status="running", tags=None):
        self.id = container_id
        self.name = name
        self.status = status
        self.image = type("Image", (), {"tags": tags or ["code-sandboxes-jupyter:latest"]})()
        self.attrs = {"Created": "2026-08-16T00:00:00Z"}
        self.removed = False

    def remove(self, force=False):
        self.removed = True

    def rename(self, name):
        self.name = name

    def reload(self):
        pass


class _FakeDockerClient:
    def __init__(self, containers):
        self._containers = containers
        self.containers = self

    def list(self, **kwargs):
        return list(self._containers)


def test_docker_manager_lists_and_deletes_by_prefix():
    container = _FakeContainer("abcdef123456789", "musing-darwin")
    manager = DockerSandboxManager(docker_client=_FakeDockerClient([container]))

    infos = manager.list()
    assert [info.id for info in infos] == ["abcdef123456"]
    assert infos[0].status == SandboxStatus.RUNNING

    assert manager.get("abcdef") is not None
    assert manager.get("musing-darwin") is not None
    assert manager.get("nope") is None

    assert manager.delete("abcdef") is True
    assert container.removed


def test_docker_manager_update_renames():
    container = _FakeContainer("abcdef123456789", "musing-darwin")
    manager = DockerSandboxManager(docker_client=_FakeDockerClient([container]))

    info = manager.update("abcdef", name="proud-noether")
    assert container.name == "proud-noether"
    assert info.name == "proud-noether"

    with pytest.raises(SandboxManagementError, match="only the name"):
        manager.update("abcdef")
    with pytest.raises(SandboxManagementError, match="No docker sandbox"):
        manager.update("nope", name="anything")


def test_the_ephemeral_variants_refuse_update():
    with pytest.raises(SandboxManagementError, match="dies with it"):
        get_manager("eval").update("anything", name="new")


class _FakeKernel:
    def __init__(self, ref, title="A kernel"):
        self.ref = ref
        self.title = title
        self.author = "someone"
        self.last_run_time = "2026-08-16"


class _FakeKaggleApi:
    def __init__(self, kernels):
        self.kernels = kernels
        self.deleted = []

    def kernels_list(self, mine=False, page_size=20):
        return list(self.kernels)

    def kernels_delete(self, ref, no_confirm=False):
        self.deleted.append(ref)

    def kernels_status(self, ref):
        return type("Status", (), {"status": "KernelWorkerStatus.COMPLETE"})()


class _FakeExecutor:
    def __init__(self, api):
        self.api = api

    def _resolve_username(self):
        return "someone"


def test_kaggle_manager_qualifies_bare_slugs():
    api = _FakeKaggleApi([_FakeKernel("someone/my-kernel")])
    manager = KaggleSandboxManager()
    manager._executor = _FakeExecutor(api)

    assert [info.id for info in manager.list()] == ["someone/my-kernel"]
    # A bare slug means "mine": the username is filled in.
    info = manager.get("my-kernel")
    assert info is not None and info.metadata["run_status"].endswith("COMPLETE")

    assert manager.delete("my-kernel") is True
    assert api.deleted == ["someone/my-kernel"]
    assert manager.delete("not-there") is False


def test_cli_list_renders_a_table(monkeypatch):
    class _FakeManager:
        variant = "kaggle"

        def list(self):
            return [
                SandboxInfo(
                    id="someone/my-kernel",
                    variant="kaggle",
                    status=SandboxStatus.STOPPED,
                    name="My Kernel",
                    metadata={"url": "https://example"},
                )
            ]

    monkeypatch.setattr(sandbox_cli, "get_manager", lambda *a, **k: _FakeManager())
    result = CliRunner().invoke(sandbox_cli.app, ["list", "-v", "kaggle"])
    assert result.exit_code == 0
    assert "my-kernel" in result.output
    assert "stopped" in result.output


def test_cli_delete_asks_unless_yes(monkeypatch):
    deleted = []

    class _FakeManager:
        def delete(self, sandbox_id):
            deleted.append(sandbox_id)
            return True

    monkeypatch.setattr(sandbox_cli, "get_manager", lambda *a, **k: _FakeManager())
    runner = CliRunner()

    result = runner.invoke(sandbox_cli.app, ["delete", "the-id", "-v", "modal"], input="n\n")
    assert result.exit_code == 0
    assert deleted == []

    result = runner.invoke(sandbox_cli.app, ["delete", "the-id", "-v", "modal", "--yes"])
    assert result.exit_code == 0
    assert deleted == ["the-id"]


def test_cli_update_parses_tags_and_renders_the_result(monkeypatch):
    received = {}

    class _FakeManager:
        def update(self, sandbox_id, **changes):
            received[sandbox_id] = changes
            return SandboxInfo(
                id=sandbox_id,
                variant="modal",
                status=SandboxStatus.RUNNING,
                name="code-sandboxes",
                metadata={"tags": changes.get("tags", {})},
            )

    monkeypatch.setattr(sandbox_cli, "get_manager", lambda *a, **k: _FakeManager())
    runner = CliRunner()

    result = runner.invoke(
        sandbox_cli.app,
        ["update", "sb-1", "-v", "modal", "--tag", "team=ai", "--tag", "env=dev"],
    )
    assert result.exit_code == 0
    assert received["sb-1"] == {"tags": {"team": "ai", "env": "dev"}}
    assert "sb-1" in result.output

    result = runner.invoke(
        sandbox_cli.app, ["update", "sb-1", "-v", "modal", "--tag", "notavalue"]
    )
    assert result.exit_code == 1
    assert "Not a key=value tag" in result.output


def test_cli_list_of_a_failing_variant_exits_nonzero(monkeypatch):
    def _raise(*a, **k):
        raise SandboxManagementError("no backend today")

    monkeypatch.setattr(sandbox_cli, "get_manager", _raise)
    result = CliRunner().invoke(sandbox_cli.app, ["list", "-v", "docker"])
    assert result.exit_code == 1
    assert "no backend today" in result.output
