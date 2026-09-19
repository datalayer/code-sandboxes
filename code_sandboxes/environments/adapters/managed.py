# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""What the managed variants can and cannot honour (PLAN_ENV.md §6, E2-06).

Every requested variant answers **before anything is queued**: a spec that
E2B cannot run, or that asks Modal for something its builder does not
implement, is said so at `validate` rather than discovered by a build that
fails ten minutes later. The answers are the constraints of section 6's
table, one place each, so the message a person reads names the provider and
what to do about it.

What is here is each managed variant's *capability half* — `capabilities` and
`validate` — plus the eight operations of the builder interface, of which the
ones that touch a provider refuse by name until their item lands (E2-03 for
E2B, E2-04 for Daytona, E2-05 for Modal). That order is deliberate: Runtimes
answers `validate` for every variant and installs **no** provider extra
(D-7), so an adapter must be importable, and useful, with no SDK present.
An SDK imported at module scope would turn `validate` into a 500.

@module code_sandboxes.environments.adapters.managed
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from ..builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
    CapabilityReport,
    CapabilitySet,
    ValidationResult,
)
from ..errors import CAPABILITY_UNSUPPORTED, SPEC_INVALID, EnvironmentsError
from ..spec import GPU_SIZE_CLASSES, Environment, EnvironmentSpec

__all__ = ["ManagedBuilder"]


class ManagedBuilder:
    """The half of a managed builder that needs no provider (§6, E2-06).

    Subclasses say what their provider is and what it cannot do; everything
    that reaches the provider refuses by name until the item that builds it
    lands, so a caller is never told "no builder" when the real answer is
    "not this spec".
    """

    variant = ""
    #: The item that will implement the provider half.
    item = ""
    #: What the provider is called in a message a person reads.
    title = ""
    #: Whether this provider runs a GPU at all (D-20, E2-17).
    gpu = False
    #: Whether this provider has a mechanism to inject a build secret into
    #: exactly the step that names it, without baking it into the image or
    #: an intermediate layer (E0-04, E3-05). Modal does (a `Secret` scoped to
    #: the build steps that name it); E2B and Daytona do not — E0-04's spike
    #: found only a registry login for each, never an arbitrary named
    #: secret — so they refuse a spec naming one, rather than silently drop
    #: it or bake it in.
    supports_build_secrets = True
    #: The build sources it will accept in this phase.
    build_sources: tuple[str, ...] = ("packages",)
    #: When `dependencyFile` is among `build_sources`, the `sourceFormat`s the
    #: variant's own build actually installs. A conda source (E3-02) installs
    #: with `micromamba`; a `pyproject`/`requirements` dependencyFile is not
    #: built for a managed variant yet (E3-01), so it is not listed here.
    dependency_formats: tuple[str, ...] = ()
    package_managers: tuple[str, ...] = ("uv", "pip")
    #: Dockerfile instructions its own builder does not implement (§6, E3-03).
    #: Checked against a `dockerfile`-sourced spec at `validate`, before
    #: anything is queued — the point being to refuse an instruction a variant
    #: would otherwise drop, rather than hand back an image quietly missing
    #: whatever it asked for.
    forbidden_instructions: tuple[str, ...] = ()
    #: Where its artifacts live; empty when the variant is regionless.
    regions: tuple[str, ...] = ()
    max_build_seconds: int | None = 45 * 60
    max_artifact_bytes: int | None = None

    def __init__(
        self,
        *,
        log: Callable[[str], None] | None = None,
        credential: Any = None,
        **options: Any,
    ) -> None:
        self._log = log or (lambda _line: None)
        self._credential = credential
        self._options = dict(options)

    # -- what it can do -------------------------------------------------------

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variant=self.variant,
            build_sources=self.build_sources,
            package_managers=self.package_managers,
            forbidden_instructions=self.forbidden_instructions,
            supports_gpu=self.gpu,
            regions=self.regions,
            max_build_seconds=self.max_build_seconds,
            max_artifact_bytes=self.max_artifact_bytes,
        )

    def validate(self, environment: Environment, lock_text: str | None = None) -> CapabilityReport:
        """Whether this variant can build this spec, said before anything is queued."""
        findings: list[CapabilityFinding] = list(self._shared_findings(environment))
        findings.extend(self._own_findings(environment, lock_text))
        return CapabilityReport(variant=self.variant, findings=findings)

    def _shared_findings(self, environment: Environment) -> list[CapabilityFinding]:
        """What every managed variant answers the same way."""
        spec = environment.spec
        findings: list[CapabilityFinding] = []
        if spec.build.source not in self.build_sources:
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=(
                        f"`{spec.build.source}` is not built for {self.title} in this phase; "
                        f"it builds {', '.join(self.build_sources)}"
                    ),
                    field="spec.build.source",
                )
            )
        elif spec.build.source == "dependencyFile":
            findings.extend(self._dependency_file_findings(spec))
        elif spec.build.source == "dockerfile":
            findings.extend(self._dockerfile_findings(spec))
        if spec.packages.python.manager not in self.package_managers:
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=(
                        f"`{spec.packages.python.manager}` is not resolved for {self.title} yet; "
                        f"it takes {', '.join(self.package_managers)}"
                    ),
                    field="spec.packages.python.manager",
                )
            )
        if spec.platform.architecture != "linux/amd64":
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=f"{self.title} builds `linux/amd64`, which is the sandbox contract's",
                    field="spec.platform.architecture",
                )
            )
        if not self.gpu:
            findings.extend(self._gpu_findings(spec))
        if self.regions:
            asked = [
                region
                for region in spec.compatibility.regions
                if region and region not in self.regions
            ]
            if asked:
                findings.append(
                    CapabilityFinding(
                        code=CAPABILITY_UNSUPPORTED.code,
                        message=(
                            f"{self.title} has no region {asked[0]!r}; it builds in "
                            + ", ".join(self.regions)
                        ),
                        field="spec.compatibility.regions",
                    )
                )
        if not self.supports_build_secrets and spec.build_secrets:
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=(
                        f"{self.title} has no per-step secret mechanism E0-04 could find — only "
                        "a registry login, never an arbitrary named secret — so `buildSecrets` "
                        "cannot be injected without baking it into the image. Build datalayer or "
                        "modal, which mount one per step, or drop the secret"
                    ),
                    field="spec.buildSecrets",
                )
            )
        return findings

    def _gpu_findings(self, spec: EnvironmentSpec) -> list[CapabilityFinding]:
        """A variant with no GPU refuses either way a GPU is asked for. A GPU
        is asked two ways, and the spec's own validation couples them; a
        `validate` can be reached before that, so both are read here rather
        than trusting the coupling."""
        if spec.resources.size_class in GPU_SIZE_CLASSES:
            return [
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=(
                        f"{self.title} has no GPU, so `{spec.resources.size_class}` cannot be "
                        f"built for it. Drop {self.variant} from the variants, or build "
                        "the GPU classes for modal or daytona, which run them on their "
                        "own hardware"
                    ),
                    field="spec.resources.sizeClass",
                )
            ]
        if spec.resources.accelerator != "none":
            return [
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=(
                        f"{self.title} has no GPU, so an accelerator cannot be built for it. "
                        f"Drop {self.variant} from the variants, or build the GPU classes for "
                        "modal or daytona, which run them on their own hardware"
                    ),
                    field="spec.resources.accelerator",
                )
            ]
        return []

    def _dependency_file_findings(self, spec: EnvironmentSpec) -> list[CapabilityFinding]:
        """A `dependencyFile` this variant accepts still only builds the
        `sourceFormat`s it has an install step for (E3-01, E3-02): a conda
        file installs with `micromamba`, but a `pyproject` or `requirements`
        one is not built for a managed variant yet."""
        source_format = spec.build.dependency_file.source_format
        if source_format in self.dependency_formats:
            return []
        return [
            CapabilityFinding(
                code=CAPABILITY_UNSUPPORTED.code,
                message=(
                    f"a `{source_format}` dependency file is not built for "
                    f"{self.title} yet; it builds "
                    f"{', '.join(self.dependency_formats) or 'no dependency file'}"
                ),
                field="spec.build.dependencyFile.sourceFormat",
            )
        ]

    def _own_findings(
        self, environment: Environment, lock_text: str | None
    ) -> list[CapabilityFinding]:
        """What only this provider refuses. Nothing, unless a subclass says so."""
        return []

    # -- what needs the provider ----------------------------------------------

    def _not_built(self, operation: str) -> EnvironmentsError:
        return EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"The {self.title} builder cannot {operation} yet: {self.item} has not landed. "
            f"What it can already do is answer whether a spec is buildable here",
            detail={"variant": self.variant, "operation": operation, "missing": self.item},
        )

    def build(self, request: BuildRequest) -> ArtifactReference:
        raise self._not_built("build")

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        raise self._not_built("inspect an artifact")

    def smoke_test(
        self,
        artifact: ArtifactReference,
        *,
        environment: Any = None,
        lock_text: str | None = None,
        secret_values: Sequence[str] = (),
    ) -> ValidationResult:
        """Launch the artifact and run Appendix B's core tier in it (E2-03/04/05).

        `environment` and `lock_text` are what the core tier needs and an
        artifact does not carry: the Python version the spec asks for and the
        packages its lock pins. The caller has both — it read them to build —
        and passes them rather than making every adapter fetch them again.
        They are keyword-only and optional so a caller that has neither still
        type-checks, and an adapter that needs them says so itself.
        """
        raise self._not_built("smoke-test an artifact")

    def resolve(self, version_ref: str) -> ArtifactReference:
        raise self._not_built("resolve a reference")

    def exists(self, artifact: ArtifactReference) -> bool:
        raise self._not_built("say whether an artifact exists")

    def delete(self, artifact: ArtifactReference) -> None:
        raise self._not_built("delete an artifact")

    # -- helpers for the subclasses -------------------------------------------

    def _dockerfile_findings(self, spec: Any) -> list[CapabilityFinding]:
        """Every instruction in the spec's Dockerfile this variant would not honour.

        The contract's own refusals are `spec.py`'s to make and apply to every
        variant alike; this is the narrower question of what *this* builder
        does with a Dockerfile it accepts. Each is named with its line, since
        a Dockerfile is somebody's file.
        """
        if not self.forbidden_instructions:
            return []
        dockerfile = getattr(spec.build, "dockerfile", None)
        content = getattr(dockerfile, "content", "") or ""
        if not content.strip():
            return []
        from ..contract import parse_dockerfile

        refused = set(self.forbidden_instructions)
        return [
            CapabilityFinding(
                code=CAPABILITY_UNSUPPORTED.code,
                message=(
                    f"line {instruction.line}: {self.title} does not implement "
                    f"`{instruction.keyword}`"
                ),
                field="spec.build.dockerfile.content",
            )
            for instruction in parse_dockerfile(content)
            if instruction.keyword in refused
        ]

    @staticmethod
    def _spec_finding(message: str, field: str) -> CapabilityFinding:
        return CapabilityFinding(code=SPEC_INVALID.code, message=message, field=field)
