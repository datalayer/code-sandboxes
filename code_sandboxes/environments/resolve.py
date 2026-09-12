# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Resolution: one lock per version, and every variant builds from it.

A version is resolved once (PLAN_ENV.md §5, D-9). The lock it produces is a
``uv pip compile --generate-hashes`` document: every Python package pinned,
with hashes, plus the apt versions and the protected pins recorded as comments
so one document says everything a build installs. Every variant of that
version then builds from that one lock, which is why four sandboxes of the
same version have the same packages.

Two things shape the resolution:

- **Datalayer's protected constraints** — the kernel stack a sandbox needs to
  connect at all, in ``constraints/sandbox-contract-v1.txt``. They are merged
  *over* the user's requirements: a requirement that contradicts one is refused
  with the range that is supported, rather than resolved into a sandbox that
  builds and then never connects.
- **The base**, which is resolved from ``spec.base`` to a digest per requested
  variant and recorded in ``status.resolvedBases``. Resolution itself runs
  inside the Datalayer base (D-9), so the interpreter that resolves is the
  interpreter the artifact will have.

Where the solve runs is the :class:`ResolveRunner`'s business:

- :class:`BuildkitResolveRunner` is D-9's: a BuildKit solve ``FROM`` the
  resolved base digest, which gives resolution the builder's isolation and its
  egress allowlist, and which pins apt versions against the snapshot mirror at
  the channel's date.
- :class:`LocalResolveRunner` runs ``uv`` where it is called, for ``plane
  local`` and for tests. It is *not* the base image, so it refuses to pin apt
  packages rather than write versions from the wrong distribution into a lock.

@module code_sandboxes.environments.resolve
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol

from .bases import APPROVED_BASES, ApprovedBase, resolve_base
from .errors import (
    CAPABILITY_UNSUPPORTED,
    PACKAGE_NOT_FOUND,
    PROTECTED_PACKAGE,
    PROVIDER_ERROR,
    RESOLVE_CONFLICT,
    SPEC_INVALID,
    EnvironmentsError,
)
from .spec import Environment, parse_environment

__all__ = [
    "APT_PIN_PREFIX",
    "CONSTRAINTS_PATH",
    "LOCK_FORMAT",
    "PROTECTED_PIN_PREFIX",
    "BuildkitResolveRunner",
    "LocalResolveRunner",
    "MergedRequirements",
    "ProtectedPin",
    "ResolveOutcome",
    "ResolveRequest",
    "ResolveRunner",
    "apt_pins",
    "apt_pins_in",
    "lock_document",
    "locked_versions",
    "merge_requirements",
    "parse_resolver_failure",
    "protected_pins",
    "resolve_bases",
    "resolve_environment",
]

#: What the lock is, as the version records it.
LOCK_FORMAT = "uv-pip-compile"

#: Datalayer's protected constraints, as the base channels also carry them at
#: ``/opt/datalayer/constraints/sandbox-contract-v1.txt`` (E1-05).
CONSTRAINTS_PATH = Path(__file__).parent / "constraints" / "sandbox-contract-v1.txt"

#: How an apt pin is written in the lock. A comment, so every reader of a
#: ``pip`` requirements file — the CLI's diff included — ignores it.
APT_PIN_PREFIX = "# datalayer-apt: "

#: How a protected pin is recorded in the lock.
PROTECTED_PIN_PREFIX = "# datalayer-protected: "

_COMMENT = re.compile(r"(?:^|\s+)#")
_HASH_OPTION = re.compile(r"\s--hash=\S+")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# -- The protected constraints -------------------------------------------------


@dataclass(frozen=True)
class ProtectedPin:
    """One package Datalayer pins, whatever the user asked for."""

    name: str
    """The canonical name, as :func:`packaging.utils.canonicalize_name` writes it."""

    requirement: str
    """The constraint as the file spells it: ``ipykernel==7.3.0``."""

    version: str | None
    """The version it pins, when it pins exactly one."""

    @property
    def supported(self) -> str:
        """The range a user requirement has to agree with."""
        from packaging.requirements import Requirement

        return str(Requirement(self.requirement).specifier)

    def permits(self, specifier: str) -> bool:
        """Whether ``specifier`` — a user's — admits the version pinned here.

        A pin with no single version (a range of Datalayer's own) is taken to
        agree with anything: only an exact pin can be contradicted.
        """
        from packaging.specifiers import InvalidSpecifier, SpecifierSet

        if not self.version:
            return True
        if not specifier:
            return True
        try:
            return SpecifierSet(specifier).contains(self.version, prereleases=True)
        except InvalidSpecifier:
            return False


@lru_cache(maxsize=4)
def _pins_of(text: str) -> tuple[ProtectedPin, ...]:
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    pins: list[ProtectedPin] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:  # pragma: no cover - the file is ours
            raise ValueError(f"{line!r} is not a requirement: {error}") from error
        pinned = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator in ("==", "===")
        ]
        pins.append(
            ProtectedPin(
                name=canonicalize_name(requirement.name),
                requirement=line,
                version=pinned[0] if len(pinned) == 1 else None,
            )
        )
    return tuple(pins)


def protected_pins(text: str | None = None) -> tuple[ProtectedPin, ...]:
    """Datalayer's protected constraints, from the file or from ``text``."""
    return _pins_of(text if text is not None else CONSTRAINTS_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class MergedRequirements:
    """What the solve is asked for, once Datalayer's pins are merged over the user's."""

    requirements: tuple[str, ...]
    constraints: tuple[str, ...]
    notes: tuple[str, ...]
    """What the merge did, for the build log and the lock's header."""


def merge_requirements(
    dependencies: Sequence[str],
    constraints: Sequence[str] = (),
    pins: Sequence[ProtectedPin] | None = None,
) -> MergedRequirements:
    """The user's requirements with Datalayer's protected pins merged over them.

    A user requirement — or a user constraint — on a protected package is kept
    only when it agrees with the pin. One that does not is
    ``DL_ENV_PROTECTED_PACKAGE``, naming what was asked for and what is
    supported, because the alternative is a build that succeeds and a sandbox
    that never connects.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    table = {pin.name: pin for pin in (pins if pins is not None else protected_pins())}
    kept: list[str] = []
    user_constraints: list[str] = []
    notes: list[str] = []
    for field_name, lines, out in (
        ("spec.packages.python.dependencies", dependencies, kept),
        ("spec.packages.python.constraints", constraints, user_constraints),
    ):
        for line in lines:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                requirement = Requirement(text)
            except InvalidRequirement:
                # The spec's own rules report an unreadable requirement with
                # its field; the solve is not the place to discover it.
                raise EnvironmentsError(
                    SPEC_INVALID,
                    f"`{text}` is not a requirement",
                    detail={"field": field_name, "requirement": text},
                ) from None
            pin = table.get(canonicalize_name(requirement.name))
            if pin is None:
                out.append(text)
                continue
            if not pin.permits(str(requirement.specifier)):
                raise EnvironmentsError(
                    PROTECTED_PACKAGE,
                    f"`{requirement.name}` is pinned by Datalayer to "
                    f"`{pin.supported}`, which `{text}` excludes: remove it, or ask for "
                    "a version inside that range",
                    detail={
                        "field": field_name,
                        "package": pin.name,
                        "requested": str(requirement.specifier) or "any",
                        "supported": pin.supported,
                    },
                )
            notes.append(
                f"{requirement.name} is Datalayer's to pin; `{text}` agrees and is dropped"
            )
    return MergedRequirements(
        requirements=tuple(kept),
        constraints=tuple(user_constraints) + tuple(pin.requirement for pin in table.values()),
        notes=tuple(notes),
    )


# -- What a runner is asked, and what it answers -------------------------------


@dataclass(frozen=True)
class ResolveRequest:
    """One solve: what to resolve, for which interpreter, from where."""

    python_version: str
    requirements: tuple[str, ...]
    constraints: tuple[str, ...]
    indexes: tuple[str, ...]
    apt: tuple[str, ...] = ()
    base_reference: str = ""
    """The base the solve runs inside, pinned by digest (D-9)."""

    registry_auth: Mapping[str, str] | None = None
    """What the registry needs to be read, when the runner pulls the base."""


@dataclass
class ResolveOutcome:
    """A solve's answer: the pinned Python set, and the apt versions with it."""

    lock_text: str
    apt_pins: dict[str, str] = field(default_factory=dict)
    apt_source: str = ""
    """The mirror the apt versions were pinned against, for the lock's header."""


class ResolveRunner(Protocol):
    """Where a solve runs."""

    name: str

    def solve(
        self, request: ResolveRequest, log: Callable[[str], None] | None = None
    ) -> ResolveOutcome: ...


# -- Reading uv's refusals -----------------------------------------------------


def _one_paragraph(output: str) -> str:
    """uv's error as one line: its box drawing gone and its wrapping undone."""
    lines: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        for marker in ("× ", "╰─▶ ", "help: ", "┃ ", "│ "):
            if line.startswith(marker):
                line = line[len(marker) :]
        if line:
            lines.append(line)
    return " ".join(lines)


_MENTION = re.compile(
    r"(?:you require|depends on) (?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<specifier>(?:[=<>!~]=?[^\s,]+)(?:,[=<>!~]=?[^\s,]+)*)?"
)


def _requirements_mentioned(text: str) -> list[tuple[str, str]]:
    """Every requirement uv's reasoning names, in the order it names them.

    uv explains a refusal as a chain — "because A depends on B and you require
    A, we can conclude that you require B. And because you require C…" — so
    the requirements that matter are the ones it says are required, and the
    pair that conflicts is the package it says two incompatible things about.
    """
    return [
        # A specifier never ends in a dot, so a trailing one is the sentence's.
        (match.group("name"), (match.group("specifier") or "").strip().rstrip("."))
        for match in _MENTION.finditer(text)
    ]


def _conflicting_pair(mentions: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The two requirements on one package that cannot both hold."""
    from packaging.utils import canonicalize_name

    seen: dict[str, list[str]] = {}
    for name, specifier in mentions:
        if not specifier:
            continue
        seen.setdefault(canonicalize_name(name), []).append(f"{name}{specifier}")
    for asked in seen.values():
        unique = list(dict.fromkeys(asked))
        if len(unique) > 1:
            return unique[-2], unique[-1]
    written = list(dict.fromkeys(f"{name}{specifier}" for name, specifier in mentions if specifier))
    return (written[-2], written[-1]) if len(written) > 1 else None


def parse_resolver_failure(
    output: str, pins: Sequence[ProtectedPin] | None = None
) -> EnvironmentsError:
    """uv's refusal as one of section 10's codes.

    Read from what uv actually writes (0.12): ``No solution found when
    resolving dependencies`` followed by its reasoning, wrapped over several
    lines and drawn in a box. A failure that is not a resolution failure at all
    — the index unreachable, uv missing — is ``DL_ENV_PROVIDER_ERROR``, which
    is retryable, because nothing about the version is wrong.
    """
    from packaging.utils import canonicalize_name

    table = {pin.name: pin for pin in (pins if pins is not None else protected_pins())}
    text = _one_paragraph(output)
    detail: dict[str, Any] = {"output": text[:2000]}
    mentions = _requirements_mentioned(text)

    missing = re.search(r"Because (\S+) was not found in the package registry", text)
    if missing:
        name = missing.group(1)
        pin = table.get(canonicalize_name(name))
        if pin is not None:
            return EnvironmentsError(
                PROTECTED_PACKAGE,
                f"`{pin.requirement}` is Datalayer's pin for `{name}` and no index serves it: "
                "it is the copy in the base image, so nothing may ask for another",
                detail={**detail, "package": pin.name, "supported": pin.supported},
            )
        return EnvironmentsError(
            PACKAGE_NOT_FOUND,
            f"`{name}` is not in any index this environment may read",
            detail={**detail, "package": name},
        )

    # A protected package in the reasoning is the reason, whatever else the
    # chain mentions: Datalayer's pin is the one requirement nobody may bend.
    for name, specifier in mentions:
        pin = table.get(canonicalize_name(name))
        if pin is not None and not pin.permits(specifier):
            return EnvironmentsError(
                PROTECTED_PACKAGE,
                f"`{name}` is pinned by Datalayer to `{pin.supported}`, and this version "
                f"asks for `{name}{specifier}`: remove it, or ask for a version inside "
                "that range",
                detail={
                    **detail,
                    "package": pin.name,
                    "requested": specifier or "any",
                    "supported": pin.supported,
                },
            )

    pair = _conflicting_pair(mentions) if "No solution found" in text else None
    if pair:
        return EnvironmentsError(
            RESOLVE_CONFLICT,
            f"`{pair[0]}` and `{pair[1]}` cannot both be satisfied: relax one of them",
            detail={**detail, "conflict": [pair[0], pair[1]]},
        )
    if "No solution found" in text:
        return EnvironmentsError(
            RESOLVE_CONFLICT,
            "The dependencies cannot be satisfied together: " + text[:400],
            detail=detail,
        )
    return EnvironmentsError(
        PROVIDER_ERROR,
        "The resolver did not finish: " + text[:400],
        detail=detail,
    )


# -- Running the solve ---------------------------------------------------------


def _index_options(indexes: Sequence[str]) -> list[str]:
    options: list[str] = []
    for position, index in enumerate(indexes):
        options.extend(["--index-url" if position == 0 else "--extra-index-url", index])
    return options


class LocalResolveRunner:
    """``uv`` where this runs: for ``plane local`` and for tests.

    Not the base image, so what it can honestly answer is the Python set
    alone. An environment with apt packages is refused rather than locked with
    versions read from whichever distribution happens to be running the
    resolver.
    """

    name = "local"

    def __init__(self, uv: str | None = None, timeout: float = 600.0) -> None:
        # `None` means "find it"; an empty string means "there is none", which
        # is how a test says so without hiding uv from the process.
        self._uv = (shutil.which("uv") or "") if uv is None else uv
        self._timeout = timeout

    def solve(
        self, request: ResolveRequest, log: Callable[[str], None] | None = None
    ) -> ResolveOutcome:
        say = log or (lambda _line: None)
        if not self._uv:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No `uv` to resolve with: the local resolver needs it on the PATH",
                detail={"missing": "uv", "runner": self.name},
            )
        if request.apt:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "The local resolver cannot pin apt packages: it is not the base image. "
                "Resolve on the build pool, which pins them against the snapshot mirror",
                detail={
                    "runner": self.name,
                    "apt": list(request.apt),
                    "seam": "BuildkitResolveRunner",
                    "item": "E1-06",
                },
            )
        with tempfile.TemporaryDirectory(prefix="dl-resolve-") as directory:
            root = Path(directory)
            requirements = root / "requirements.in"
            requirements.write_text("\n".join(request.requirements) + "\n", encoding="utf-8")
            constraints = root / "constraints.txt"
            constraints.write_text("\n".join(request.constraints) + "\n", encoding="utf-8")
            command = [
                self._uv,
                "pip",
                "compile",
                "--quiet",
                "--no-header",
                "--generate-hashes",
                "--python-version",
                request.python_version,
                "--constraint",
                str(constraints),
                *_index_options(request.indexes),
                str(requirements),
            ]
            say(
                f"Resolving {len(request.requirements)} requirements with uv "
                f"for Python {request.python_version}"
            )
            try:
                finished = subprocess.run(  # noqa: S603 - the argv is built here
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as expired:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The resolver did not finish within {self._timeout:.0f}s",
                    detail={"runner": self.name, "timeout": self._timeout},
                ) from expired
        if finished.returncode != 0:
            for line in (finished.stderr or "").splitlines():
                say(line)
            raise parse_resolver_failure(finished.stderr or finished.stdout or "")
        return ResolveOutcome(lock_text=finished.stdout)


class BuildkitResolveRunner:
    """D-9's solve: ``uv pip compile`` inside the resolved base, under BuildKit.

    The solve gets the builder's isolation, its egress allowlist and the exact
    interpreter the artifact will have, and it is where apt versions are pinned
    against ``snapshot.ubuntu.com`` or ``snapshot.debian.org`` at the channel's
    date. It needs ``buildctl`` and a reachable ``buildkitd``, neither of which
    exists until the build pool is deployed (E1-06), so until then it refuses
    by name rather than resolving somewhere else.
    """

    name = "buildkit"

    def __init__(
        self,
        buildctl: str | None = None,
        address: str | None = None,
        apt_snapshot: str = "",
        timeout: float = 900.0,
    ) -> None:
        self._buildctl = (shutil.which("buildctl") or "") if buildctl is None else buildctl
        self._address = address or ""
        self._apt_snapshot = apt_snapshot
        self._timeout = timeout

    def dockerfile(self, request: ResolveRequest) -> str:
        """The solve, as the frontend reads it: resolve, then pin apt, then export both."""
        lines = [
            f"FROM {request.base_reference} AS solve",
            "USER root",
            "WORKDIR /solve",
            "COPY requirements.in constraints.txt ./",
            "RUN --mount=type=cache,target=/root/.cache/uv "
            f"uv pip compile --quiet --no-header --generate-hashes "
            f"--python-version {request.python_version} --constraint constraints.txt "
            + " ".join(_index_options(request.indexes))
            + " requirements.in -o /solve/lock.txt",
        ]
        if request.apt:
            # `--simulate` names every package apt would install, transitive
            # ones included, each with the exact version the mirror serves at
            # the channel's date (D-9).
            snapshot = self._apt_snapshot
            if snapshot:
                codename = "$(. /etc/os-release; echo $VERSION_CODENAME)"
                lines.append(
                    f"RUN printf 'deb {snapshot} {codename} main\\n' "
                    "> /etc/apt/sources.list.d/datalayer-snapshot.list"
                )
            lines.append(
                "RUN apt-get update -qq && apt-get install --simulate --no-install-recommends "
                + " ".join(request.apt)
                + " > /solve/apt.txt"
            )
        lines.extend(
            [
                "FROM scratch",
                "COPY --from=solve /solve/lock.txt /lock.txt",
            ]
        )
        if request.apt:
            lines.append("COPY --from=solve /solve/apt.txt /apt.txt")
        return "\n".join(lines) + "\n"

    def solve(
        self, request: ResolveRequest, log: Callable[[str], None] | None = None
    ) -> ResolveOutcome:
        say = log or (lambda _line: None)
        if not self._buildctl:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No `buildctl` to resolve with: the build pool is not deployed here",
                detail={"missing": "buildctl", "runner": self.name, "item": "E1-06"},
            )
        if "@sha256:" not in request.base_reference:
            raise EnvironmentsError(
                SPEC_INVALID,
                "The base is resolved to a digest before anything is solved",
                detail={"base": request.base_reference},
            )
        with tempfile.TemporaryDirectory(prefix="dl-solve-") as directory:
            root = Path(directory)
            (root / "requirements.in").write_text(
                "\n".join(request.requirements) + "\n", encoding="utf-8"
            )
            (root / "constraints.txt").write_text(
                "\n".join(request.constraints) + "\n", encoding="utf-8"
            )
            (root / "Dockerfile").write_text(self.dockerfile(request), encoding="utf-8")
            out = root / "out"
            command = [
                self._buildctl,
                *(["--addr", self._address] if self._address else []),
                "build",
                "--frontend",
                "dockerfile.v0",
                "--local",
                f"context={root}",
                "--local",
                f"dockerfile={root}",
                "--output",
                f"type=local,dest={out}",
            ]
            say(f"Solving the lock in {request.base_reference}")
            try:
                finished = subprocess.run(  # noqa: S603 - the argv is built here
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    env=self._environment(request),
                )
            except subprocess.TimeoutExpired as expired:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The solve did not finish within {self._timeout:.0f}s",
                    detail={"runner": self.name, "timeout": self._timeout},
                ) from expired
            for line in (finished.stderr or "").splitlines():
                say(line)
            if finished.returncode != 0:
                raise parse_resolver_failure(finished.stderr or finished.stdout or "")
            lock = (out / "lock.txt").read_text(encoding="utf-8")
            pins, source = {}, ""
            apt_file = out / "apt.txt"
            if request.apt and apt_file.exists():
                pins = apt_pins(apt_file.read_text(encoding="utf-8"))
                source = self._apt_snapshot or "the base channel's mirror"
        return ResolveOutcome(lock_text=lock, apt_pins=pins, apt_source=source)

    def _environment(self, request: ResolveRequest) -> dict[str, str] | None:
        """The registry credential, as client-side auth: `buildkitd` holds none (D-17)."""
        import os

        auth = dict(request.registry_auth or {})
        if not auth:
            return None
        return {**os.environ, **{str(key): str(value) for key, value in auth.items()}}


def apt_pins(simulation: str) -> dict[str, str]:
    """The versions ``apt-get install --simulate`` says it would install."""
    pins: dict[str, str] = {}
    for raw in simulation.splitlines():
        line = raw.strip()
        if not line.startswith("Inst "):
            continue
        rest = line[len("Inst ") :]
        name = rest.split(" ", 1)[0]
        opened = rest.find("(")
        if opened == -1:
            continue
        version = rest[opened + 1 :].split(" ", 1)[0].rstrip(")")
        if name and version:
            pins[name] = version
    return pins


# -- The lock ------------------------------------------------------------------


def locked_versions(lock_text: str) -> dict[str, str]:
    """The version a lock pins for each package, by canonical name."""
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    joined: list[str] = []
    pending = ""
    for raw in lock_text.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        joined.append(pending + line)
        pending = ""
    if pending:
        joined.append(pending)
    versions: dict[str, str] = {}
    for line in joined:
        text = _COMMENT.split(line, maxsplit=1)[0]
        text = _HASH_OPTION.sub(" ", " " + text).strip()
        if not text or text.startswith("-"):
            continue
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            continue
        pinned = [
            specifier.version
            for specifier in requirement.specifier
            if specifier.operator in ("==", "===")
        ]
        if len(pinned) == 1:
            versions[canonicalize_name(requirement.name)] = pinned[0]
    return versions


def apt_pins_in(lock_text: str) -> dict[str, str]:
    """The apt versions a lock records, by package.

    The builder installs exactly these (E1-07): the lock is the one document
    that says what a build installs, apt included, so a build never asks a
    mirror what the newest version is.
    """
    pins: dict[str, str] = {}
    for raw in lock_text.splitlines():
        line = raw.strip()
        if not line.startswith(APT_PIN_PREFIX.strip()):
            continue
        pin = line[len(APT_PIN_PREFIX.strip()) :].strip()
        name, _, version = pin.partition("=")
        if name and version:
            pins[name.strip()] = version.strip()
    return pins


def lock_document(
    outcome: ResolveOutcome,
    *,
    python_version: str,
    base_reference: str,
    merged: MergedRequirements,
    resolved_at: datetime | None = None,
) -> dict[str, Any]:
    """The stored lock: its text, its digest, and what a reader needs from it.

    The apt pins and the protected pins are written as comments above uv's
    output, so the document is still a requirements file — ``pip install -r``
    reads it, and so does every tool that only knows that format — while
    saying everything the build installs.
    """
    when = (resolved_at or _utcnow()).replace(microsecond=0).isoformat()
    header = [
        "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.",
        f"# resolved-at: {when}",
        f"# python: {python_version}",
        f"# base: {base_reference}",
    ]
    for pin in outcome.apt_pins.items():
        header.append(f"{APT_PIN_PREFIX}{pin[0]}={pin[1]}")
    if outcome.apt_pins and outcome.apt_source:
        header.append(f"# datalayer-apt-source: {outcome.apt_source}")
    for constraint in merged.constraints:
        header.append(f"{PROTECTED_PIN_PREFIX}{constraint}")
    text = "\n".join(header) + "\n" + outcome.lock_text.lstrip("\n")
    if not text.endswith("\n"):
        text += "\n"
    packages = locked_versions(text)
    return {
        "digest": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "format": LOCK_FORMAT,
        "content": text,
        "python_version": python_version,
        "package_count": len(packages),
    }


def resolve_bases(
    environment: Environment,
    variants: Sequence[str],
    bases: dict[str, ApprovedBase] = APPROVED_BASES,
) -> dict[str, str]:
    """Each variant's base, pinned by digest: ``<repository>@sha256:…`` (D-9, §4)."""
    base = bases.get(environment.spec.base.ref)
    repository = base.repository if base is not None else environment.spec.base.ref
    resolved: dict[str, str] = {}
    for variant in variants:
        digest = resolve_base(
            environment.spec.base.ref, environment.spec.base.channel, variant, bases
        )
        resolved[variant] = f"{repository}@{digest}"
    return resolved


def resolve_environment(
    *,
    spec: Mapping[str, Any] | str | Environment,
    variants: Sequence[str],
    credential: Any = None,
    log: Callable[[str], None] | None = None,
    runner: ResolveRunner | None = None,
    bases: dict[str, ApprovedBase] = APPROVED_BASES,
    resolved_at: datetime | None = None,
) -> dict[str, Any]:
    """A version's lock, and the base each variant builds from.

    This is the resolver seam of ``EnvironmentBuildWorkflow`` (E1-03): it takes
    the version's spec and the variants the build was asked for, and answers
    the document the workflow stores once, by digest.

    Parameters
    ----------
    spec
        The version's specification, as a mapping, as YAML or JSON, or parsed.
    variants
        Every variant this version must resolve a base for.
    credential
        The build credential, whose registry auth the BuildKit runner needs to
        pull the base. Unused by the local runner.
    log
        Where the solve's output goes, line by line: the build's log.
    runner
        Where the solve runs. D-9's BuildKit solve by default.
    bases
        The approved bases, injected by tests and by a plane whose channel is
        published somewhere else.
    resolved_at
        The moment the lock records. Now by default.

    Returns
    -------
    dict
        ``digest``, ``format``, ``content``, ``python_version``,
        ``package_count`` and ``resolved_bases``.

    Raises
    ------
    EnvironmentsError
        Everything a person can act on: a spec that cannot be resolved, a
        requirement that contradicts a protected pin, a conflict, a package no
        index has, a base channel nobody published.
    """
    say = log or (lambda _line: None)
    environment = parse_environment(spec)
    python = environment.spec.packages.python
    if environment.spec.build.source != "packages":
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            f"`{environment.spec.build.source}` is not resolved yet: only `packages` is, "
            "in this phase",
            detail={"field": "spec.build.source", "source": environment.spec.build.source},
        )
    if python.manager == "conda":
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            "Conda environments resolve through their own solver, which is not built yet: "
            "use `uv` or `pip` syntax",
            detail={"field": "spec.packages.python.manager", "manager": "conda"},
        )
    wanted = sorted({str(variant) for variant in variants} or {"datalayer"})
    resolved_bases = resolve_bases(environment, wanted, bases)
    merged = merge_requirements(python.dependencies, python.constraints)
    for note in merged.notes:
        say(note)
    # Resolution runs in the Datalayer base when it is one of the variants —
    # its interpreter is the one the artifact will have (D-9) — and otherwise
    # in the first base asked for, which is the same image on this channel.
    solving_in = resolved_bases.get("datalayer") or next(iter(resolved_bases.values()))
    request = ResolveRequest(
        python_version=environment.spec.language.version,
        requirements=merged.requirements,
        constraints=merged.constraints,
        indexes=tuple(python.indexes),
        apt=tuple(environment.spec.packages.system.apt),
        base_reference=solving_in,
        registry_auth=_registry_auth(credential),
    )
    outcome = (runner or BuildkitResolveRunner()).solve(request, say)
    document = lock_document(
        outcome,
        python_version=environment.spec.language.version,
        base_reference=solving_in,
        merged=merged,
        resolved_at=resolved_at,
    )
    say(
        f"Locked {document['package_count']} packages"
        + (f" and {len(outcome.apt_pins)} apt packages" if outcome.apt_pins else "")
        + f" as {document['digest']}"
    )
    return {**document, "resolved_bases": resolved_bases}


def _registry_auth(credential: Any) -> Mapping[str, str] | None:
    """The credential's registry auth, however it carries it.

    The build credential is minted by the workflow (E1-03) and its shape is
    durable's; a resolver that insisted on one type would be coupled to it.
    """
    if credential is None:
        return None
    for attribute in ("registry_auth", "environment", "env"):
        value = getattr(credential, attribute, None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
    return None
