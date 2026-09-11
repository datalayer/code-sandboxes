# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The builder interface, its neutral types, and the shared files step."""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any

import pytest
import yaml

from code_sandboxes.environments import errors
from code_sandboxes.environments.builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
    CapabilityReport,
    CapabilitySet,
    CheckResult,
    EnvironmentBuilder,
    ValidationResult,
    builder_contract_violations,
    get_builder,
)
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.files import build_entries, files_step
from code_sandboxes.environments.spec import Environment, parse_environment

DIGEST = "sha256:" + "a" * 64
SHA = "b" * 64

ENVIRONMENT = (
    "metadata: {name: geo}\n"
    "spec:\n"
    '  language: {version: "3.13"}\n'
    '  base: {ref: datalayer/python-cpu, channel: "2026.09"}\n'
    "  files:\n"
    "    - path: /home/datalayer/content/notes.md\n"
    "      contentRef: blob://environments/notes.md\n"
    f"      sha256: {SHA}\n"
)


def environment(**spec: Any) -> Environment:
    data = yaml.safe_load(ENVIRONMENT)
    data["spec"].update(spec)
    return parse_environment(data)


def artifact(**fields: Any) -> ArtifactReference:
    values = {
        "variant": "datalayer",
        "immutable_reference": f"registry.example/environments/u/1/geo@{DIGEST}",
        "provider_artifact_id": DIGEST,
        "contract_version": "sandbox-contract/v1",
    }
    values.update(fields)
    return ArtifactReference(**values)


class FakeBuilder:
    """Every operation, answering the least a real builder would."""

    variant = "datalayer"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.deleted: list[ArtifactReference] = []

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variant="datalayer", build_sources=("packages",), package_managers=("uv", "pip")
        )

    def validate(self, environment: Environment, lock_text: str | None = None) -> CapabilityReport:
        return CapabilityReport(variant="datalayer")

    def build(self, request: BuildRequest) -> ArtifactReference:
        return artifact()

    def inspect(self, reference: ArtifactReference) -> ArtifactMetadata:
        return ArtifactMetadata(reference=reference, size_bytes=1)

    def smoke_test(self, reference: ArtifactReference) -> ValidationResult:
        return ValidationResult(contract_version="sandbox-contract/v1")

    def resolve(self, version_ref: str) -> ArtifactReference:
        return artifact()

    def exists(self, reference: ArtifactReference) -> bool:
        return reference not in self.deleted

    def delete(self, reference: ArtifactReference) -> None:
        self.deleted.append(reference)


def test_a_complete_builder_honors_the_interface() -> None:
    builder = FakeBuilder()
    assert builder_contract_violations(builder, variant="datalayer") == []
    assert isinstance(builder, EnvironmentBuilder)
    reference = builder.build(
        BuildRequest(
            environment_uid="env-1",
            version=1,
            build_uid="build-1",
            owner_uid="user-1",
            variant="datalayer",
            environment=environment(),
            lock_text="numpy==2.2.0 --hash=sha256:" + SHA,
            lock_digest="sha256:" + SHA,
            resolved_base=f"datalayer/python-cpu@{DIGEST}",
        )
    )
    assert builder.exists(reference)
    builder.delete(reference)
    assert not builder.exists(reference)


def test_what_a_builder_lacks_is_named() -> None:
    class Incomplete(FakeBuilder):
        delete = None  # type: ignore[assignment]

    class Misdeclared(FakeBuilder):
        variant = "modal"

    class Mismatched(FakeBuilder):
        def capabilities(self) -> CapabilitySet:
            return CapabilitySet(variant="e2b", build_sources=(), package_managers=())

    assert builder_contract_violations(Incomplete()) == ["no `delete`"]
    assert builder_contract_violations(Misdeclared(), variant="datalayer") == [
        "`variant` is 'modal', asked for 'datalayer'"
    ]
    assert builder_contract_violations(Mismatched()) == [
        "`capabilities()` describes 'e2b', the builder is 'datalayer'"
    ]


def test_a_variant_that_is_not_one_has_no_builder() -> None:
    with pytest.raises(EnvironmentsError) as refused:
        get_builder("kaggle")
    assert refused.value.code is errors.CAPABILITY_UNSUPPORTED


def test_a_variant_whose_builder_is_not_shipped_is_unsupported_not_an_import_error() -> None:
    with pytest.raises(EnvironmentsError) as refused:
        get_builder("Modal")
    assert refused.value.code is errors.CAPABILITY_UNSUPPORTED
    assert refused.value.detail == {"variant": "modal"}


def test_a_builder_is_loaded_by_its_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("code_sandboxes.environments.adapters.datalayer")
    module.Builder = FakeBuilder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    builder = get_builder("DATALAYER", registry="registry.example")
    assert isinstance(builder, FakeBuilder)
    assert builder.options == {"registry": "registry.example"}


def test_a_builder_that_breaks_the_interface_is_not_handed_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken(FakeBuilder):
        variant = "e2b"

    module = types.ModuleType("code_sandboxes.environments.adapters.datalayer")
    module.Builder = Broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError, match="asked for 'datalayer'"):
        get_builder("datalayer")


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"immutable_reference": "registry.example/geo:v1"}, "by its digest"),
        ({"variant": "modal", "immutable_reference": "geo-image"}, "by its image id"),
        ({"variant": "kaggle"}, "is not a variant"),
        (
            {"mutable_alias": f"registry.example/environments/u/1/geo@{DIGEST}"},
            "cannot be the same",
        ),
    ],
)
def test_an_artifact_is_referenced_by_what_cannot_move(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        artifact(**fields)


def test_managed_variants_keep_the_reference_their_provider_gives() -> None:
    assert artifact(variant="modal", immutable_reference="im-123").immutable_reference == "im-123"
    assert artifact(variant="e2b", immutable_reference="dl-acme/geo-v7:9f3c").variant == "e2b"


def test_a_build_starts_from_a_base_pinned_by_digest() -> None:
    with pytest.raises(ValueError, match="resolved to a digest"):
        BuildRequest(
            environment_uid="env-1",
            version=1,
            build_uid="build-1",
            owner_uid="user-1",
            variant="datalayer",
            environment=environment(),
            lock_text="",
            lock_digest="sha256:" + SHA,
            resolved_base="datalayer/python-cpu:2026.09",
        )


def test_only_gating_checks_decide_a_validation() -> None:
    result = ValidationResult(
        contract_version="sandbox-contract/v1",
        checks=[
            CheckResult(id="conformance:4", name="kernel", passed=True),
            CheckResult(id="conformance:12", name="throughput", passed=False, gating=False),
        ],
    )
    assert result.passed and result.failures == []
    result.checks.append(CheckResult(id="conformance:6", name="filesystem", passed=False))
    assert not result.passed
    assert [check.id for check in result.failures] == ["conformance:6"]


def test_a_report_with_findings_is_unsupported() -> None:
    assert CapabilityReport(variant="modal").supported
    report = CapabilityReport(
        variant="modal",
        findings=[CapabilityFinding(code="DL_ENV_CAPABILITY_UNSUPPORTED", message="VOLUME")],
    )
    assert not report.supported


def test_the_files_step_is_the_verified_fetch_every_variant_shares() -> None:
    commands = files_step(
        environment(), variant="modal", source_of=lambda entry: "https://signed.example/notes.md"
    )
    assert (
        "curl -fsSL https://signed.example/notes.md -o /home/datalayer/content/notes.md"
        in commands[0]
    )
    assert f'echo "{SHA}  /home/datalayer/content/notes.md" | sha256sum -c' in commands[0]
    assert "environment-contents.json" in commands[-1]
    assert files_step(environment(files=[]), variant="datalayer") == []


def test_a_file_is_not_baked_without_its_digest() -> None:
    unverified = environment(
        files=[{"path": "/home/datalayer/content/a", "contentRef": "blob://a"}]
    )
    with pytest.raises(EnvironmentsError) as refused:
        build_entries(unverified)
    assert refused.value.code is errors.SPEC_INVALID
    with pytest.raises(ValueError):
        files_step(environment(), variant="kaggle")


def test_the_neutral_modules_import_no_provider_sdk() -> None:
    """What a service may import must not drag a provider in."""
    code = (
        "import sys\n"
        "for name in ('modal', 'daytona', 'e2b', 'e2b_code_interpreter', 'kaggle'):\n"
        "    sys.modules[name] = None\n"
        "import code_sandboxes.environments\n"
        "import code_sandboxes.environments.conformance\n"
        "import code_sandboxes.environments.files\n"
        "import code_sandboxes.environments.schema\n"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
