# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""An Environment's `contents:` manifest, built into every provider's artifact.

The same verified fetch on every provider — `curl` then `sha256sum -c` — and
the same manifest written beside the files, so a sandbox can say what its
artifact carries. What differs is what the provider calls an artifact, and
each of those is exercised here against a fake of its SDK: no network, no
Docker, no CLI.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from code_sandboxes import builds
from code_sandboxes.builds import (
    ENVIRONMENT_CONTENTS_MANIFEST,
    BuildEntry,
    EnvironmentBuild,
    build_artifact,
    build_commands,
    dockerfile_fragment,
    environment_contents_manifest,
    fetch_command,
    installed_environment_contents,
    manifest_command,
)
from code_sandboxes.exceptions import SandboxConfigurationError

SHA_A = "a" * 64
SHA_B = "b" * 64


def _build(provider: str = "datalayer", **fields) -> EnvironmentBuild:
    return EnvironmentBuild(
        environment="python-science-env",
        provider=provider,
        entries=[
            BuildEntry(
                source_uri="https://datalayer.example/contents/iris.csv",
                destination_path="/opt/datalayer/contents/iris.csv",
                sha256=SHA_A,
                size_bytes=4551,
            ),
            BuildEntry(
                source_uri="s3://datalayer-datasets-prod/cards/model-card.md",
                destination_path="/opt/datalayer/contents/model-card.md",
                sha256=SHA_B,
            ),
        ],
        **fields,
    )


# --- The commands ---------------------------------------------------------------


def test_the_fragment_fetches_and_verifies_every_entry():
    """A checksum that does not match fails the build, and the fragment says so."""
    fragment = dockerfile_fragment(_build())

    for entry in _build().entries:
        assert f"curl -fsSL {entry.source_uri} -o {entry.destination_path}" in fragment
        assert f'echo "{entry.sha256}  {entry.destination_path}" | sha256sum -c' in fragment
    lines = [line for line in fragment.splitlines() if not line.startswith("#")]
    assert all(line.startswith("RUN ") for line in lines)
    assert sum("sha256sum -c" in line for line in lines) == 2
    assert "fails the build" in fragment
    assert ENVIRONMENT_CONTENTS_MANIFEST in fragment
    assert not fragment.startswith("FROM")
    assert dockerfile_fragment(_build(), base_image="python:3.12").startswith("FROM python:3.12\n")


def test_the_commands_end_with_the_manifest_of_what_was_built():
    commands = build_commands(_build())
    assert len(commands) == 3
    assert commands[-1] == manifest_command(_build())
    assert ENVIRONMENT_CONTENTS_MANIFEST in commands[-1]
    manifest = environment_contents_manifest(_build())
    assert manifest["environment"] == "python-science-env"
    assert manifest["entries"] == [
        {
            "path": "/opt/datalayer/contents/iris.csv",
            "sha256": SHA_A,
            "source": "https://datalayer.example/contents/iris.csv",
            "size": 4551,
        },
        {
            "path": "/opt/datalayer/contents/model-card.md",
            "sha256": SHA_B,
            "source": "s3://datalayer-datasets-prod/cards/model-card.md",
        },
    ]


def test_a_relative_destination_is_refused():
    with pytest.raises(SandboxConfigurationError, match="absolute"):
        fetch_command(BuildEntry(source_uri="https://x/y", destination_path="y", sha256=SHA_A))


@pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("sha256sum") is None, reason="needs a shell"
)
def test_the_manifest_command_writes_valid_json_through_a_real_shell(tmp_path, monkeypatch):
    """The quoting is what a shell sees; a real one is the only judge of it."""
    target = tmp_path / "etc" / "datalayer" / "environment-contents.json"
    monkeypatch.setattr(builds, "ENVIRONMENT_CONTENTS_MANIFEST", str(target))
    build = _build()
    build.entries[0].source_uri = "https://x/it's?a=1&b='2'"

    subprocess.run(["sh", "-c", manifest_command(build)], check=True)  # noqa: S603, S607

    written = json.loads(target.read_text())
    assert written["entries"][0]["source"] == "https://x/it's?a=1&b='2'"
    assert written == environment_contents_manifest(build)


@pytest.mark.skipif(
    shutil.which("curl") is None or shutil.which("sha256sum") is None, reason="needs curl"
)
def test_a_wrong_checksum_fails_the_fetch_in_a_real_shell(tmp_path):
    """`sha256sum -c` verifies: the file arrives, the digest is checked, the build stops."""
    source = tmp_path / "iris.csv"
    source.write_bytes(b"sepal,petal\n1,2\n")
    destination = tmp_path / "out" / "iris.csv"
    right = BuildEntry(
        source_uri=source.as_uri(),
        destination_path=str(destination),
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    wrong = right.model_copy(update={"sha256": "0" * 64})

    def run(command: str) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", "-c", command], capture_output=True, text=True)  # noqa: S603, S607

    assert run(fetch_command(right)).returncode == 0
    assert destination.read_bytes() == source.read_bytes()
    failed = run(fetch_command(wrong))
    assert failed.returncode != 0
    assert "FAILED" in failed.stdout + failed.stderr


# --- Datalayer: a Dockerfile fragment --------------------------------------------


def test_datalayer_answers_the_fragment_as_text_and_writes_it_where_asked(tmp_path):
    path = tmp_path / "build" / "contents.Dockerfile"

    artifact = build_artifact(_build("datalayer"), output_path=path)

    assert artifact.provider == "datalayer"
    assert artifact.reference == str(path)
    assert artifact.manifest_path == ENVIRONMENT_CONTENTS_MANIFEST
    assert artifact.dockerfile == dockerfile_fragment(_build("datalayer"))
    assert path.read_text() == artifact.dockerfile
    assert [entry.sha256 for entry in artifact.entries] == [SHA_A, SHA_B]

    unwritten = build_artifact(_build("datalayer"))
    assert unwritten.reference == "dockerfile://python-science-env"
    assert unwritten.dockerfile == artifact.dockerfile


# --- E2B: the Dockerfile the CLI builds from ---------------------------------------


def test_e2b_writes_the_dockerfile_the_cli_consumes_and_names_its_path(tmp_path):
    path = tmp_path / "e2b.Dockerfile"

    artifact = build_artifact(_build("e2b"), output_path=path)

    assert artifact.provider == "e2b"
    assert artifact.reference == str(path)
    text = path.read_text()
    assert text.startswith("FROM e2bdev/code-interpreter:latest\n")
    assert sum("sha256sum -c" in line for line in text.splitlines() if line.startswith("RUN")) == 2
    assert ENVIRONMENT_CONTENTS_MANIFEST in text

    with pytest.raises(SandboxConfigurationError, match="e2b template build"):
        build_artifact(_build("e2b"))
    custom = build_artifact(_build("e2b", base_image="my/template:1"), output_path=path)
    assert custom.dockerfile.startswith("FROM my/template:1\n")


# --- Modal: an Image with the same commands -------------------------------------------


class _FakeImage:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.apt: list[str] = []
        self.commands: list[str] = []
        self.object_id = None

    def apt_install(self, *packages):
        self.apt.extend(packages)
        return self

    def run_commands(self, *commands):
        self.commands.extend(commands)
        return self


@pytest.fixture
def fake_modal(monkeypatch):
    module = types.ModuleType("modal")
    made: list[_FakeImage] = []

    def debian_slim(**_):
        made.append(_FakeImage("debian_slim"))
        return made[-1]

    def from_registry(tag, **_):
        made.append(_FakeImage(tag))
        return made[-1]

    module.Image = SimpleNamespace(debian_slim=debian_slim, from_registry=from_registry)
    monkeypatch.setitem(sys.modules, "modal", module)
    return made


def test_modal_builds_an_image_of_the_same_verified_fetches(fake_modal):
    artifact = build_artifact(_build("modal"))

    assert artifact.provider == "modal"
    assert artifact.reference == "modal://python-science-env"
    (image,) = fake_modal
    assert artifact.image is image
    assert image.origin == "debian_slim"
    assert "curl" in image.apt
    assert image.commands == build_commands(_build("modal"))
    assert sum("sha256sum -c" in command for command in image.commands) == 2
    assert SHA_A in image.commands[0] and SHA_B in image.commands[1]
    assert ENVIRONMENT_CONTENTS_MANIFEST in image.commands[-1]
    # The SDK object stays out of the serialized artifact.
    assert "image" not in artifact.model_dump()


def test_modal_extends_the_base_image_it_was_given(fake_modal):
    build_artifact(_build("modal", base_image="datalayer/python-science:1"))
    (image,) = fake_modal
    assert image.origin == "datalayer/python-science:1"


def test_modal_is_imported_only_when_a_modal_build_is_made(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    build_artifact(_build("datalayer"))  # no modal needed
    with pytest.raises(SandboxConfigurationError, match="pip install modal"):
        build_artifact(_build("modal"))


# --- Daytona: an Image and a snapshot -------------------------------------------------


@pytest.fixture
def fake_daytona(monkeypatch):
    module = types.ModuleType("daytona")
    made: list[_FakeImage] = []
    snapshots: list = []

    def debian_slim(version):
        made.append(_FakeImage(f"debian_slim:{version}"))
        return made[-1]

    def base(tag):
        made.append(_FakeImage(tag))
        return made[-1]

    class Daytona:
        def __init__(self, config=None):
            self.snapshot = SimpleNamespace(create=lambda params, **_: snapshots.append(params))

    module.Image = SimpleNamespace(debian_slim=debian_slim, base=base)
    module.CreateSnapshotParams = lambda **values: SimpleNamespace(**values)
    module.Daytona = Daytona
    monkeypatch.setitem(sys.modules, "daytona", module)
    return SimpleNamespace(images=made, snapshots=snapshots, Daytona=Daytona)


def test_daytona_builds_an_image_and_creates_the_snapshot(fake_daytona):
    artifact = build_artifact(_build("daytona"))

    assert artifact.provider == "daytona"
    assert artifact.reference == "python-science-env-contents"
    (image,) = fake_daytona.images
    assert image.origin == "debian_slim:3.12"
    assert image.commands == build_commands(_build("daytona"))
    assert sum("sha256sum -c" in command for command in image.commands) == 2
    (snapshot,) = fake_daytona.snapshots
    assert snapshot.name == "python-science-env-contents"
    assert snapshot.image is image


def test_daytona_takes_the_client_the_snapshot_name_and_the_base_image_it_is_given(fake_daytona):
    created: list = []
    client = SimpleNamespace(
        snapshot=SimpleNamespace(create=lambda params, **_: created.append(params))
    )

    artifact = build_artifact(
        _build("daytona", base_image="datalayer/python-science:1"),
        snapshot_name="science-2026-08",
        daytona_client=client,
    )

    assert artifact.reference == "science-2026-08"
    assert fake_daytona.images[0].origin == "datalayer/python-science:1"
    assert [params.name for params in created] == ["science-2026-08"]
    assert fake_daytona.snapshots == []


# --- Reading it back from inside a sandbox ----------------------------------------------


class _Sandbox:
    """Runs the probe here, the way the contract suite's fakes do."""

    def run_code(self, code, timeout=None):
        import contextlib
        import io

        del timeout
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, "<sandbox>", "exec"), {"__name__": "__main__"})  # noqa: S102
        lines = out.getvalue().splitlines()
        return SimpleNamespace(
            execution_ok=True,
            execution_error=None,
            code_error=None,
            logs=SimpleNamespace(stdout=[SimpleNamespace(line=line) for line in lines]),
        )


def test_what_an_artifact_carries_is_read_back_from_inside(tmp_path):
    manifest_path = tmp_path / "environment-contents.json"

    assert installed_environment_contents(_Sandbox(), path=str(manifest_path)) is None

    manifest_path.write_text(json.dumps(environment_contents_manifest(_build())))
    found = installed_environment_contents(_Sandbox(), path=str(manifest_path))
    assert found == environment_contents_manifest(_build())
    assert [entry["path"] for entry in found["entries"]] == [
        "/opt/datalayer/contents/iris.csv",
        "/opt/datalayer/contents/model-card.md",
    ]


def test_a_build_entry_needs_a_real_digest():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BuildEntry(source_uri="https://x/y", destination_path="/y", sha256="not-a-digest")
    with pytest.raises(ValidationError):
        EnvironmentBuild(environment="e", provider="kaggle")
