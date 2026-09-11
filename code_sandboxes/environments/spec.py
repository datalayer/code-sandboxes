# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The canonical Environment specification, ``environments.datalayer.io/v1alpha1``.

A user writes one of these; it is resolved once into a lock and compiled into
a build request for every variant. Users never write four provider
definitions. The models below are the specification — the JSON Schema is
exported from them, and the TypeScript types are generated from that — and
:func:`spec_findings` holds every rule a model's types cannot express.

The spec carries no runtime secret. Build secrets are referenced by id and
injected into the build step that needs them; anything in ``env`` that looks
like a credential is refused, with ``buildSecrets`` named as the place for it.

``spec_digest`` is the sha256 of the RFC 8785 serialization of ``spec`` with
its defaults written out, so two specs that mean the same thing hash the
same. It is one half of the build cache key; the lock digest and the resolved
base digest are the others.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

from .bases import APPROVED_BASES, ApprovedBase
from .canonical import canonical_digest
from .contract import SANDBOX_CONTRACT_V1, SUPPORTED_CONTRACTS
from .errors import CAPABILITY_UNSUPPORTED, SPEC_INVALID, EnvironmentsError, ErrorCode
from .lifecycle import VersionState

__all__ = [
    "API_VERSION",
    "BUILD_SOURCES",
    "GPU_SIZE_CLASSES",
    "KIND",
    "SIZE_CLASSES",
    "SUPPORTED_BUILD_SOURCES",
    "VARIANTS",
    "Accelerator",
    "ArtifactStatus",
    "Base",
    "BuildSecret",
    "BuildSpec",
    "Commands",
    "Compatibility",
    "Environment",
    "EnvironmentSpec",
    "FileEntry",
    "Language",
    "LockStatus",
    "Metadata",
    "Packages",
    "Platform",
    "PythonPackages",
    "ResourceHints",
    "Resources",
    "SpecFinding",
    "SystemPackages",
    "VariantSet",
    "VersionStatus",
    "parse_environment",
    "spec_digest",
    "spec_findings",
    "validate_environment",
]

API_VERSION = "environments.datalayer.io/v1alpha1"
KIND = "Environment"

#: The Code Sandbox variants an Environment builds for, as ``SandboxVariant`` spells them.
VARIANTS: tuple[str, ...] = ("datalayer", "e2b", "daytona", "modal")

#: The size classes a version runs on (PLAN_ENV.md, D-4); the rate table
#: that prices them lives with the services.
SIZE_CLASSES: tuple[str, ...] = ("small", "medium", "large", "gpu-small", "gpu-large")
GPU_SIZE_CLASSES: tuple[str, ...] = ("gpu-small", "gpu-large")

BUILD_SOURCES: tuple[str, ...] = ("packages", "dependencyFile", "dockerfile", "image")
#: What builds today; the other sources come with dependency files, Dockerfiles
#: and image imports.
SUPPORTED_BUILD_SOURCES: tuple[str, ...] = ("packages",)
SUPPORTED_PACKAGE_MANAGERS: tuple[str, ...] = ("uv", "pip")

RESERVED_NAME_PREFIXES: tuple[str, ...] = ("datalayer-", "dl-", "kube-", "system-")
MAX_NAME_LENGTH = 63
MAX_FILES = 32
MAX_FILE_BYTES = 1024 * 1024
MAX_FILES_TOTAL_BYTES = 8 * 1024 * 1024
MAX_POST_INSTALL_COMMANDS = 32
FILE_CONTENT_SCHEMES: tuple[str, ...] = ("blob", "https", "s3")

_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_APT_PACKAGE = re.compile(r"^[a-z0-9][a-z0-9+.-]+(=[A-Za-z0-9.+~:-]+)?$")
_REGION = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_CREDENTIAL_NAME = re.compile(
    r"(^|_)(token|secret|password|passwd|pwd|apikey|api_key|accesskey|access_key|"
    r"privatekey|private_key|credential|credentials)($|_)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = re.compile(
    r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}"
    r"|sk-[A-Za-z0-9_-]{20,}|xox[abpr]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+|dlsec_[0-9A-Za-z]+"
)
_URL_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]*@", re.IGNORECASE)


class _Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class Metadata(_Model):
    #: Unique per owner: a DNS-1123 label, without a reserved prefix.
    name: str
    title: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class Language(_Model):
    name: Literal["python"] = "python"
    #: ``major.minor``; must be what the base provides, never silently replaced.
    version: str = Field(pattern=r"^3\.\d+$")


class Base(_Model):
    ref: str
    channel: str


class Platform(_Model):
    architecture: Literal["linux/amd64"] = "linux/amd64"


class PythonPackages(_Model):
    manager: Literal["uv", "pip", "conda"] = "uv"
    dependencies: list[str] = Field(default_factory=list)
    #: Merged under Datalayer's protected constraints, which win.
    constraints: list[str] = Field(default_factory=list)
    indexes: list[str] = Field(default_factory=lambda: ["https://pypi.org/simple"])


class SystemPackages(_Model):
    apt: list[str] = Field(default_factory=list)


class Packages(_Model):
    python: PythonPackages = Field(default_factory=PythonPackages)
    system: SystemPackages = Field(default_factory=SystemPackages)


class FileEntry(_Model):
    path: str
    content_ref: str
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class Commands(_Model):
    post_install: list[str] = Field(default_factory=list)


class BuildSecret(_Model):
    id: str = Field(pattern=r"^dlsec_[0-9A-Za-z]+$")
    mount_as: Literal["env", "file"] = "env"
    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class Accelerator(_Model):
    type: str
    count: int = Field(default=1, ge=1)
    cuda: str | None = None


class ResourceHints(_Model):
    cpu: float | None = Field(default=None, gt=0)
    memory_gi: float | None = Field(default=None, gt=0)
    disk_gi: float | None = Field(default=None, gt=0)


class Resources(_Model):
    size_class: str = "small"
    accelerator: Literal["none"] | Accelerator = "none"
    hints: ResourceHints = Field(default_factory=ResourceHints)


class VariantSet(_Model):
    required: list[str] = Field(default_factory=lambda: ["datalayer"])
    optional: list[str] = Field(default_factory=list)


class Compatibility(_Model):
    variants: VariantSet = Field(default_factory=VariantSet)
    regions: list[str] = Field(default_factory=list)


class BuildSpec(_Model):
    source: Literal["packages", "dependencyFile", "dockerfile", "image"] = "packages"


class EnvironmentSpec(_Model):
    contract: str = SANDBOX_CONTRACT_V1.version
    language: Language
    base: Base
    platform: Platform = Field(default_factory=Platform)
    packages: Packages = Field(default_factory=Packages)
    files: list[FileEntry] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    commands: Commands = Field(default_factory=Commands)
    build_secrets: list[BuildSecret] = Field(default_factory=list)
    resources: Resources = Field(default_factory=Resources)
    compatibility: Compatibility = Field(default_factory=Compatibility)
    build: BuildSpec = Field(default_factory=BuildSpec)


class Environment(_Model):
    api_version: Literal["environments.datalayer.io/v1alpha1"] = API_VERSION
    kind: Literal["Environment"] = KIND
    metadata: Metadata
    spec: EnvironmentSpec


# --- The status a version carries: system-owned, never user-authored -------------


class LockStatus(_Model):
    ref: str
    digest: str
    resolved_at: str
    python_version: str
    package_count: int = Field(ge=0)


class ArtifactStatus(_Model):
    ref: str
    state: str


class VersionStatus(_Model):
    version: int = Field(ge=1)
    spec_digest: str
    lock: LockStatus | None = None
    resolved_bases: dict[str, str] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactStatus] = Field(default_factory=dict)
    burning_rate: float | None = None
    state: VersionState = VersionState.DRAFT


# --- Rules the types cannot hold ------------------------------------------------


@dataclass(frozen=True)
class SpecFinding:
    """One problem with a spec: where, what, and under which code."""

    field: str
    message: str
    code: ErrorCode = SPEC_INVALID

    def to_dict(self) -> dict[str, str]:
        return {"field": self.field, "message": self.message, "code": self.code.code}


def _requirement_problem(text: str) -> str | None:
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:  # pragma: no cover - packaging ships with pip and jupyter
        return None
    try:
        Requirement(text)
    except InvalidRequirement as error:
        return str(error)
    return None


def _index_findings(field: str, url: str) -> list[SpecFinding]:
    if _URL_CREDENTIALS.match(url):
        return [
            SpecFinding(
                field,
                "carries a credential in its URL; reference the credential in `buildSecrets`",
            )
        ]
    if not url.startswith("https://"):
        return [SpecFinding(field, f"`{url}` is not an https URL")]
    return []


def spec_findings(
    environment: Environment, *, bases: Mapping[str, ApprovedBase] = APPROVED_BASES
) -> list[SpecFinding]:
    """Every rule of the specification the environment breaks."""
    findings: list[SpecFinding] = []
    spec = environment.spec
    name = environment.metadata.name

    if len(name) > MAX_NAME_LENGTH or not _NAME.match(name):
        findings.append(
            SpecFinding(
                "metadata.name",
                f"`{name}` is not a DNS-1123 label: lowercase letters, digits and `-`, "
                f"at most {MAX_NAME_LENGTH} characters",
            )
        )
    for prefix in RESERVED_NAME_PREFIXES:
        if name.startswith(prefix):
            findings.append(SpecFinding("metadata.name", f"the `{prefix}` prefix is reserved"))

    if spec.contract not in SUPPORTED_CONTRACTS:
        findings.append(
            SpecFinding(
                "spec.contract",
                f"`{spec.contract}` is not a supported contract; supported: "
                + ", ".join(SUPPORTED_CONTRACTS),
                CAPABILITY_UNSUPPORTED,
            )
        )

    base = bases.get(spec.base.ref)
    if base is None:
        findings.append(
            SpecFinding(
                "spec.base.ref",
                f"`{spec.base.ref}` is not an approved base; approved: " + ", ".join(bases),
            )
        )
    elif spec.language.version not in base.python_versions:
        findings.append(
            SpecFinding(
                "spec.language.version",
                f"Python {spec.language.version} is not what `{base.ref}` provides "
                f"({', '.join(base.python_versions)}); "
                "it is validated against the base, not replaced",
            )
        )

    if spec.build.source not in SUPPORTED_BUILD_SOURCES:
        findings.append(
            SpecFinding(
                "spec.build.source",
                f"`{spec.build.source}` is not buildable yet; supported: "
                + ", ".join(SUPPORTED_BUILD_SOURCES),
                CAPABILITY_UNSUPPORTED,
            )
        )

    python = spec.packages.python
    if python.manager not in SUPPORTED_PACKAGE_MANAGERS:
        findings.append(
            SpecFinding(
                "spec.packages.python.manager",
                f"`{python.manager}` is not supported yet; supported: "
                + ", ".join(SUPPORTED_PACKAGE_MANAGERS),
                CAPABILITY_UNSUPPORTED,
            )
        )
    for group in ("dependencies", "constraints"):
        for index, requirement in enumerate(getattr(python, group)):
            problem = _requirement_problem(requirement)
            if problem:
                findings.append(
                    SpecFinding(
                        f"spec.packages.python.{group}[{index}]", f"`{requirement}`: {problem}"
                    )
                )
    for index, url in enumerate(python.indexes):
        findings.extend(_index_findings(f"spec.packages.python.indexes[{index}]", url))
    for index, package in enumerate(spec.packages.system.apt):
        if not _APT_PACKAGE.match(package):
            findings.append(
                SpecFinding(
                    f"spec.packages.system.apt[{index}]",
                    f"`{package}` is not an apt package name, optionally `=version`",
                )
            )

    if len(spec.files) > MAX_FILES:
        findings.append(SpecFinding("spec.files", f"at most {MAX_FILES} files are baked in"))
    total = 0
    seen_paths: set[str] = set()
    reserved = SANDBOX_CONTRACT_V1.reserved_path
    for index, entry in enumerate(spec.files):
        field = f"spec.files[{index}]"
        normalized = posixpath.normpath(entry.path)
        if (
            not posixpath.isabs(entry.path)
            or normalized != entry.path.rstrip("/")
            or ".." in entry.path.split("/")
        ):
            findings.append(
                SpecFinding(f"{field}.path", f"`{entry.path}` is not an absolute, normalized path")
            )
        if normalized == reserved or normalized.startswith(reserved + "/"):
            findings.append(SpecFinding(f"{field}.path", f"`{reserved}` is reserved for Datalayer"))
        if normalized in seen_paths:
            findings.append(SpecFinding(f"{field}.path", f"`{entry.path}` is listed twice"))
        seen_paths.add(normalized)
        scheme = entry.content_ref.split("://", 1)[0] if "://" in entry.content_ref else ""
        if scheme not in FILE_CONTENT_SCHEMES:
            findings.append(
                SpecFinding(
                    f"{field}.contentRef",
                    "must be a "
                    + ", ".join(f"`{s}://`" for s in FILE_CONTENT_SCHEMES)
                    + " reference",
                )
            )
        if entry.size_bytes is not None:
            total += entry.size_bytes
            if entry.size_bytes > MAX_FILE_BYTES:
                findings.append(
                    SpecFinding(
                        f"{field}.sizeBytes", f"a baked file is at most {MAX_FILE_BYTES} bytes"
                    )
                )
    if total > MAX_FILES_TOTAL_BYTES:
        findings.append(
            SpecFinding("spec.files", f"baked files total at most {MAX_FILES_TOTAL_BYTES} bytes")
        )

    for key, value in spec.env.items():
        field = f"spec.env.{key}"
        if not _ENV_NAME.match(key):
            findings.append(SpecFinding(field, f"`{key}` is not an environment variable name"))
        if (
            _CREDENTIAL_NAME.search(key)
            or _CREDENTIAL_VALUE.search(value)
            or _URL_CREDENTIALS.match(value)
        ):
            findings.append(
                SpecFinding(
                    field,
                    "looks like a credential; `env` is not secret, "
                    "so reference it in `buildSecrets`",
                )
            )

    if len(spec.commands.post_install) > MAX_POST_INSTALL_COMMANDS:
        findings.append(
            SpecFinding(
                "spec.commands.postInstall", f"at most {MAX_POST_INSTALL_COMMANDS} commands"
            )
        )
    for index, command in enumerate(spec.commands.post_install):
        if not command.strip():
            findings.append(SpecFinding(f"spec.commands.postInstall[{index}]", "is empty"))

    for attribute, label in (("id", "id"), ("name", "name")):
        values = [getattr(secret, attribute) for secret in spec.build_secrets]
        if len(values) != len(set(values)):
            findings.append(SpecFinding("spec.buildSecrets", f"a secret {label} is listed twice"))

    resources = spec.resources
    if resources.size_class not in SIZE_CLASSES:
        findings.append(
            SpecFinding(
                "spec.resources.sizeClass",
                f"`{resources.size_class}` is not a size class; the classes are "
                + ", ".join(SIZE_CLASSES),
            )
        )
    gpu_class = resources.size_class in GPU_SIZE_CLASSES
    wants_gpu = resources.accelerator != "none"
    if gpu_class and not wants_gpu:
        findings.append(
            SpecFinding(
                "spec.resources.accelerator", f"`{resources.size_class}` needs an accelerator"
            )
        )
    if wants_gpu and not gpu_class:
        findings.append(
            SpecFinding(
                "spec.resources.sizeClass",
                "an accelerator needs a GPU size class: " + ", ".join(GPU_SIZE_CLASSES),
            )
        )
    if gpu_class and base is not None and not base.accelerator:
        findings.append(
            SpecFinding(
                "spec.base.ref", f"`{base.ref}` has no CUDA; a GPU size class needs a CUDA base"
            )
        )

    variants = spec.compatibility.variants
    if not variants.required:
        findings.append(
            SpecFinding("spec.compatibility.variants.required", "at least one variant is required")
        )
    for group in ("required", "optional"):
        members = getattr(variants, group)
        for variant in members:
            if variant not in VARIANTS:
                findings.append(
                    SpecFinding(
                        f"spec.compatibility.variants.{group}",
                        f"`{variant}` is not a variant; the variants are " + ", ".join(VARIANTS),
                    )
                )
        if len(members) != len(set(members)):
            findings.append(
                SpecFinding(f"spec.compatibility.variants.{group}", "a variant is listed twice")
            )
    both = sorted(set(variants.required) & set(variants.optional))
    if both:
        findings.append(
            SpecFinding(
                "spec.compatibility.variants",
                "a variant is either required or optional, not both: " + ", ".join(both),
            )
        )
    for index, region in enumerate(spec.compatibility.regions):
        if not _REGION.match(region):
            findings.append(
                SpecFinding(
                    f"spec.compatibility.regions[{index}]", f"`{region}` is not a region name"
                )
            )
    return findings


def _loc(location: tuple[Any, ...]) -> str:
    field = ""
    for part in location:
        if isinstance(part, int):
            field += f"[{part}]"
        else:
            field += ("." if field else "") + str(part)
    return field


def parse_environment(document: Mapping[str, Any] | str | Environment) -> Environment:
    """An Environment from a mapping, or from YAML or JSON text.

    A document the models refuse is ``DL_ENV_SPEC_INVALID``, every problem
    listed with the field it is in, as the document spells the field.
    """
    if isinstance(document, Environment):
        return document
    data: Any = document
    if isinstance(document, str):
        try:
            import yaml
        except ImportError:
            data = json.loads(document)
        else:
            data = yaml.safe_load(document)
    try:
        return Environment.model_validate(data)
    except ValidationError as error:
        findings = [
            SpecFinding(_loc(item["loc"]) or "(document)", item["msg"]).to_dict()
            for item in error.errors()
        ]
        first = findings[0]
        raise EnvironmentsError(
            SPEC_INVALID, f"{first['field']}: {first['message']}", detail={"findings": findings}
        ) from None


def validate_environment(
    document: Mapping[str, Any] | str | Environment,
    *,
    bases: Mapping[str, ApprovedBase] = APPROVED_BASES,
) -> Environment:
    """The Environment, or the error its findings amount to.

    A spec with any invalid field is ``DL_ENV_SPEC_INVALID``; one that is
    valid but asks for what cannot be built yet is
    ``DL_ENV_CAPABILITY_UNSUPPORTED``. Either way every finding is listed.
    """
    environment = parse_environment(document)
    findings = spec_findings(environment, bases=bases)
    if findings:
        invalid = [finding for finding in findings if finding.code is SPEC_INVALID]
        code = SPEC_INVALID if invalid else CAPABILITY_UNSUPPORTED
        first = (invalid or findings)[0]
        raise EnvironmentsError(
            code,
            f"{first.field}: {first.message}",
            detail={"findings": [finding.to_dict() for finding in findings]},
        )
    return environment


def spec_digest(environment: Environment | EnvironmentSpec) -> str:
    """``sha256:<hex>`` of the canonical spec, defaults included; metadata excluded."""
    spec = environment.spec if isinstance(environment, Environment) else environment
    return canonical_digest(spec.model_dump(by_alias=True, mode="json"))
