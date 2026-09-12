# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The one interface every variant's Environment builder implements.

A builder turns a version of an Environment — its spec and its lock — into
one immutable artifact on one variant, and answers for that artifact
afterwards: what it is, whether it still exists, whether it passes the
contract. The types below say nothing about any provider, which is what lets
a service hold them: the provider-specific builders live in
:mod:`code_sandboxes.environments.adapters` and load only through
:func:`get_builder`, the way ``manage.get_manager`` loads a manager.

An artifact is referenced by what the provider cannot change underneath it —
an OCI digest, an E2B build id, a Daytona snapshot id, a Modal image id —
never by a name or a tag, which every provider lets move.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import normalize_variant
from .errors import CAPABILITY_UNSUPPORTED, SPEC_INVALID, EnvironmentsError
from .spec import VARIANTS, Environment

__all__ = [
    "BUILDER_MODULES",
    "LAUNCH_ARGUMENTS",
    "ArtifactMetadata",
    "ArtifactReference",
    "BuildRequest",
    "CapabilityFinding",
    "CapabilityReport",
    "CapabilitySet",
    "CheckResult",
    "EnvironmentBuilder",
    "ValidationResult",
    "builder_contract_violations",
    "get_builder",
    "launch_arguments",
]

_OCI_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CapabilitySet(_Model):
    """What a variant's builder can do, before any spec is looked at."""

    variant: str
    build_sources: tuple[str, ...]
    package_managers: tuple[str, ...]
    #: Dockerfile instructions this variant's builder cannot honor.
    forbidden_instructions: tuple[str, ...] = ()
    supports_gpu: bool = False
    #: Where artifacts can be built; empty when the variant has no regions.
    regions: tuple[str, ...] = ()
    max_build_seconds: int | None = None
    max_artifact_bytes: int | None = None


class CapabilityFinding(_Model):
    code: str
    message: str
    field: str | None = None


class CapabilityReport(_Model):
    """Whether one variant can build one spec, and why not."""

    variant: str
    findings: list[CapabilityFinding] = Field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.findings


class ArtifactReference(_Model):
    """One immutable artifact of one version, on one variant."""

    variant: str
    #: What a sandbox is launched from: never a mutable name or tag.
    immutable_reference: str
    provider_artifact_id: str
    region: str | None = None
    size_class: str | None = None
    #: A name kept for people and dashboards; never launched from.
    mutable_alias: str | None = None
    #: A non-secret fingerprint of the provider account the artifact lives in.
    provider_account: str | None = None
    contract_version: str
    architecture: str = "linux/amd64"

    @model_validator(mode="after")
    def _immutable(self) -> ArtifactReference:
        if self.variant not in VARIANTS:
            raise ValueError(f"{self.variant!r} is not a variant")
        if self.variant == "datalayer" and not _OCI_DIGEST.search(self.immutable_reference):
            raise ValueError("a Datalayer artifact is referenced by its digest, `...@sha256:<hex>`")
        if self.variant == "modal" and not self.immutable_reference.startswith("im-"):
            raise ValueError("a Modal artifact is referenced by its image id, `im-...`")
        if self.mutable_alias is not None and self.mutable_alias == self.immutable_reference:
            raise ValueError("the mutable alias and the immutable reference cannot be the same")
        return self


class ArtifactMetadata(_Model):
    reference: ArtifactReference
    size_bytes: int | None = None
    created_at: str | None = None
    provider_state: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class CheckResult(_Model):
    """One check of a validation: the conformance suite's, or the doctor's."""

    id: str
    name: str
    passed: bool
    #: Whether a failure blocks the version; the extended tier records without gating.
    gating: bool = True
    detail: str | None = None
    data: dict[str, object] = Field(default_factory=dict)


class ValidationResult(_Model):
    contract_version: str
    checks: list[CheckResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks if check.gating)

    @property
    def failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.gating and not check.passed]


class BuildRequest(_Model):
    """Everything one variant's build of one version needs, and nothing secret."""

    environment_uid: str
    version: int = Field(ge=1)
    build_uid: str
    owner_uid: str
    variant: str
    environment: Environment
    #: The resolved lock, as text, and its digest: every variant builds from the same one.
    lock_text: str
    lock_digest: str
    #: The base, pinned by digest for this variant.
    resolved_base: str
    region: str | None = None
    size_class: str = "small"
    #: Build secret ids; the values are fetched by the step that needs them.
    build_secret_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> BuildRequest:
        if self.variant not in VARIANTS:
            raise ValueError(f"{self.variant!r} is not a variant")
        if "@sha256:" not in self.resolved_base:
            raise ValueError("the base is resolved to a digest before a build starts")
        return self


@runtime_checkable
class EnvironmentBuilder(Protocol):
    """The eight operations of a variant's builder."""

    variant: str

    def capabilities(self) -> CapabilitySet: ...

    def validate(
        self, environment: Environment, lock_text: str | None = None
    ) -> CapabilityReport: ...

    def build(self, request: BuildRequest) -> ArtifactReference: ...

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata: ...

    def smoke_test(self, artifact: ArtifactReference) -> ValidationResult: ...

    def resolve(self, version_ref: str) -> ArtifactReference: ...

    def exists(self, artifact: ArtifactReference) -> bool: ...

    def delete(self, artifact: ArtifactReference) -> None: ...


#: What launching an artifact means to each variant that has one to launch
#: (PLAN_ENV.md E2-02). `datalayer` is not here on purpose: see
#: :func:`launch_arguments`. A variant added to `VARIANTS` without a line here
#: refuses to launch an artifact, and `test_environment_builders.py` says so.
LAUNCH_ARGUMENTS: dict[str, str] = {
    "e2b": "template",
    "daytona": "snapshot",
    "modal": "image_id",
}


def launch_arguments(artifact: ArtifactReference | Mapping[str, Any]) -> dict[str, Any]:
    """What launching this artifact means to its variant (PLAN_ENV.md E2-02).

    One neutral reference in, one variant's own argument out: E2B launches a
    template build, Daytona a snapshot id, Modal an image id. A caller that
    has an artifact should not have to know which.

    The Datalayer variant is deliberately not here. A sandbox of it is started
    by the platform, which resolves the version's artifact itself and hands the
    digest to the Operator over a route only Runtimes may call (E1-10, E1-11):
    a client passing a digest could name an artifact of somebody else's
    environment, so it names the environment and the version instead.
    """
    reference = artifact if isinstance(artifact, ArtifactReference) else None
    if reference is None:
        if not isinstance(artifact, Mapping):
            raise EnvironmentsError(
                SPEC_INVALID,
                "an artifact is an ArtifactReference, or the mapping of one",
                detail={"artifact": str(artifact)[:120]},
            )
        reference = ArtifactReference.model_validate(
            {to_snake(name): value for name, value in artifact.items()}
        )
    variant = reference.variant
    if variant == "datalayer":
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            "A Datalayer sandbox is launched by naming the environment and the version: the "
            "platform resolves the artifact itself, and hands its digest to the Operator over "
            "a route only Runtimes may call",
            detail={"variant": variant, "reference": reference.immutable_reference},
        )
    argument = LAUNCH_ARGUMENTS.get(variant)
    if argument is None:
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"the {variant} variant launches no artifact of an Environment",
            detail={"variant": variant},
        )
    return {argument: reference.immutable_reference}


def to_snake(name: str) -> str:
    """`immutableReference` as `immutable_reference`: a mapping read either way."""
    out: list[str] = []
    for character in str(name):
        if character.isupper():
            out.append("_")
            out.append(character.lower())
        else:
            out.append(character)
    return "".join(out)


#: Where each variant's builder lives, imported only when that variant is asked for.
BUILDER_MODULES: dict[str, str] = {
    variant: f"code_sandboxes.environments.adapters.{variant}" for variant in VARIANTS
}

_OPERATIONS = (
    "capabilities",
    "validate",
    "build",
    "inspect",
    "smoke_test",
    "resolve",
    "exists",
    "delete",
)


def get_builder(variant: str, **options: object) -> EnvironmentBuilder:
    """The builder of a variant, its provider SDK imported only now.

    A variant with no builder yet is ``DL_ENV_CAPABILITY_UNSUPPORTED``, not
    an import error: a caller asked for something this release cannot do.
    """
    normalized = normalize_variant(variant)
    module_name = BUILDER_MODULES.get(normalized)
    if module_name is None:
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"no Environment builder for variant {variant!r}; the variants are "
            + ", ".join(VARIANTS),
            detail={"variant": variant},
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"the {normalized} Environment builder is not available in this release",
            detail={"variant": normalized},
        ) from None
    builder = module.Builder(**options)
    violations = builder_contract_violations(builder, variant=normalized)
    if violations:
        raise TypeError(
            f"{module_name}.Builder breaks the builder interface: " + "; ".join(violations)
        )
    return builder


def builder_contract_violations(builder: object, *, variant: str | None = None) -> list[str]:
    """What a builder lacks to be one; empty when it has everything."""
    violations = [
        f"no `{operation}`"
        for operation in _OPERATIONS
        if not callable(getattr(builder, operation, None))
    ]
    declared = getattr(builder, "variant", None)
    if declared not in VARIANTS:
        violations.append(f"`variant` is {declared!r}, not one of {', '.join(VARIANTS)}")
    elif variant is not None and declared != variant:
        violations.append(f"`variant` is {declared!r}, asked for {variant!r}")
    if not violations:
        capabilities = builder.capabilities()  # type: ignore[attr-defined]
        if not isinstance(capabilities, CapabilitySet):
            violations.append("`capabilities()` does not return a CapabilitySet")
        elif capabilities.variant != declared:
            violations.append(
                f"`capabilities()` describes {capabilities.variant!r}, the builder is {declared!r}"
            )
    return violations
