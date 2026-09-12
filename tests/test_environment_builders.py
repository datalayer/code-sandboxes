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

from code_sandboxes.environments import builders, errors
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


def test_every_variant_ships_a_builder_that_honours_the_interface() -> None:
    """Since E2-06: a spec is answered for on every variant, whatever half of
    that variant's builder has landed. `test_environment_managed_builders.py`
    is where each one's answers live."""
    from code_sandboxes.environments.spec import VARIANTS

    for variant in VARIANTS:
        assert builder_contract_violations(get_builder(variant), variant=variant) == [], variant


def test_a_variant_whose_builder_is_not_shipped_is_unsupported_not_an_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release that drops an adapter answers "this release cannot", not a
    traceback about a module a caller has never heard of."""
    monkeypatch.setitem(
        builders.BUILDER_MODULES, "modal", "code_sandboxes.environments.adapters.nowhere"
    )
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
        "import code_sandboxes.environments.redact\n"
        "import code_sandboxes.environments.schema\n"
    )
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr


class TestLaunchingAnArtifact:
    """One neutral reference, translated into the argument its variant takes (E2-02).

    A caller holding an Environment's artifact should not have to know that
    E2B calls it a template, Daytona a snapshot and Modal an image id.
    """

    def an_artifact(self, variant: str, reference: str) -> builders.ArtifactReference:
        return builders.ArtifactReference(
            variant=variant,
            immutable_reference=reference,
            provider_artifact_id=reference,
            contract_version="sandbox-contract/v1",
        )

    def test_each_managed_variant_gets_its_own_argument(self) -> None:
        assert builders.launch_arguments(self.an_artifact("e2b", "dl/geo:bld-1")) == {
            "template": "dl/geo:bld-1"
        }
        assert builders.launch_arguments(self.an_artifact("daytona", "snap-1")) == {
            "snapshot": "snap-1"
        }
        assert builders.launch_arguments(self.an_artifact("modal", "im-123")) == {
            "image_id": "im-123"
        }

    def test_a_mapping_is_read_either_way_round(self) -> None:
        """What a route answers is camelCase; what a model holds is snake."""
        assert builders.launch_arguments(
            {
                "variant": "e2b",
                "immutableReference": "dl/geo:bld-2",
                "providerArtifactId": "bld-2",
                "contractVersion": "sandbox-contract/v1",
            }
        ) == {"template": "dl/geo:bld-2"}

    def test_the_datalayer_variant_is_launched_by_naming_its_version(self) -> None:
        """A client passing a digest could name somebody else's artifact."""
        from code_sandboxes.environments.errors import EnvironmentsError

        artifact = self.an_artifact(
            "datalayer", "123456789012.dkr.ecr.us-east-1.amazonaws.com/e@sha256:" + "a" * 64
        )
        with pytest.raises(EnvironmentsError) as raised:
            builders.launch_arguments(artifact)
        assert raised.value.code.code == "DL_ENV_CAPABILITY_UNSUPPORTED"
        assert "only Runtimes may call" in raised.value.message

    def test_every_variant_with_an_artifact_has_an_argument(self) -> None:
        """A variant added without a line in LAUNCH_ARGUMENTS launches nothing."""
        from code_sandboxes.environments.spec import VARIANTS

        assert set(VARIANTS) - {"datalayer"} == set(builders.LAUNCH_ARGUMENTS)

    def test_something_that_is_not_an_artifact_is_refused(self) -> None:
        from code_sandboxes.environments.errors import EnvironmentsError

        with pytest.raises(EnvironmentsError) as raised:
            builders.launch_arguments("dl/geo:bld-1")
        assert raised.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_create_hands_the_artifact_to_the_variant(self) -> None:
        """`Sandbox.create(artifact=…)` reaches the adapter's own argument."""
        from code_sandboxes.base import Sandbox

        sandbox = Sandbox.create(variant="e2b", artifact=self.an_artifact("e2b", "dl/geo:bld-1"))
        assert sandbox._template == "dl/geo:bld-1"

    def test_what_the_caller_passed_wins_over_the_artifact(self) -> None:
        from code_sandboxes.base import Sandbox

        sandbox = Sandbox.create(
            variant="daytona",
            artifact=self.an_artifact("daytona", "snap-1"),
            snapshot="snap-chosen",
        )
        assert sandbox._snapshot == "snap-chosen"

    def test_a_modal_artifact_reaches_its_image_id(self) -> None:
        from code_sandboxes.base import Sandbox

        sandbox = Sandbox.create(variant="modal", artifact=self.an_artifact("modal", "im-123"))
        assert sandbox._image_id == "im-123"
