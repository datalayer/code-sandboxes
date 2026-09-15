# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Conda resolution: an ``environment.yml`` to one explicit lock (E3-02).

A ``build.source: dependencyFile`` version whose ``sourceFormat`` is ``conda``
brings a conda ``environment.yml`` — the form advanced users already have for a
project that pulls, say, ``gdal`` from ``conda-forge`` rather than from a wheel.
It is resolved the way every other source is (PLAN_ENV.md §5, D-9): **once**,
into one lock, and every variant of the version then builds from that one lock,
so four sandboxes of the same version carry the same packages.

Two decisions this item makes, both recorded here rather than left implicit:

- **micromamba, not conda-lock.** The solve runs ``micromamba`` — the same tool
  the Datalayer, E2B and Daytona builders install the lock with, and the tool
  Modal's ``micromamba_install`` wraps — so the interpreter that resolves is the
  one that installs, with no second solver's opinion in between. The lock it
  produces is a conda **explicit** file (``@EXPLICIT``): one ``package-url#hash``
  line per package, which ``micromamba create --file`` installs without
  re-solving. That is the "explicit lock with hashes" this box asks for.
- **The protected pins still apply to the pip layer.** A conda environment needs
  the same kernel stack every sandbox needs to connect (``ipykernel`` and its
  siblings, ``constraints/sandbox-contract-v1.txt``). They are merged *over* the
  ``pip:`` section of the ``environment.yml`` exactly as :func:`merge_requirements`
  merges them over a ``packages`` source's dependencies — a pip requirement that
  contradicts a protected pin is refused with the range that is supported, and
  every pin is forced in whether or not the environment named it, because a
  ``--constraint`` alone never installs what nothing else already depends on.

Where the solve runs is the :class:`CondaResolveRunner`'s business, mirroring
:mod:`code_sandboxes.environments.resolve`:

- :class:`BuildkitCondaResolveRunner` is D-9's: a BuildKit solve ``FROM`` the
  resolved base digest, which gives resolution the builder's isolation and its
  egress allowlist.
- :class:`MicromambaResolveRunner` runs ``micromamba`` where it is called, for
  ``plane local`` and for tests.

@module code_sandboxes.environments.resolve_conda
"""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .errors import (
    CAPABILITY_UNSUPPORTED,
    PACKAGE_NOT_FOUND,
    PROVIDER_ERROR,
    RESOLVE_CONFLICT,
    SPEC_INVALID,
    EnvironmentsError,
)
from .resolve import (
    PROTECTED_PIN_PREFIX,
    WHEELHOUSE_IMAGE_PATH,
    WHEELHOUSE_PATH,
    MergedRequirements,
    ProtectedPin,
    merge_requirements,
)

__all__ = [
    "CONDA_LOCK_FORMAT",
    "BuildkitCondaResolveRunner",
    "CondaEnvironment",
    "CondaResolveOutcome",
    "CondaResolveRequest",
    "CondaResolveRunner",
    "MicromambaResolveRunner",
    "conda_lock_document",
    "conda_lock_protected_pins",
    "explicit_lock_packages",
    "is_conda_lock",
    "merge_conda_pip",
    "parse_conda_environment",
    "parse_conda_failure",
    "resolve_conda_environment",
]

#: What a conda lock is, as the version records it: a conda **explicit** file,
#: the ``@EXPLICIT`` form ``micromamba create --file`` installs without solving.
CONDA_LOCK_FORMAT = "conda-explicit"

#: The platform a Phase-2 artifact is built for (D-9): every variant is
#: ``linux/amd64``, so the solve is for ``linux-64`` in conda's own naming.
CONDA_PLATFORM = "linux-64"

#: How the protected pip pins are recorded in the lock's header, the same
#: prefix :func:`code_sandboxes.environments.resolve.lock_document` uses, so a
#: reader of either lock finds Datalayer's pins the same way.
_EXPLICIT_MARKER = "@EXPLICIT"
_PIP_SECTION_KEY = "pip"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# -- Reading the environment.yml ----------------------------------------------


@dataclass(frozen=True)
class CondaEnvironment:
    """An ``environment.yml``, as the resolver reads it."""

    name: str
    channels: tuple[str, ...]
    conda_dependencies: tuple[str, ...]
    pip_dependencies: tuple[str, ...]


def parse_conda_environment(text: str) -> CondaEnvironment:
    """One ``environment.yml``'s channels, conda packages and pip packages.

    A conda ``dependencies`` list mixes plain conda specs with a single
    ``{"pip": [...]}`` mapping for the pip layer; both are separated here so the
    protected pins merge over the pip layer alone (:func:`merge_conda_pip`) and
    the conda layer is passed through untouched. Anything that is not a conda
    spec or the one pip mapping — a nested list, a bare number — is refused with
    its position, because a solve is not the place to discover a malformed file.
    """
    import yaml

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise EnvironmentsError(
            SPEC_INVALID,
            f"the environment.yml is not valid YAML: {error}",
            detail={"field": "spec.build.dependencyFile.content"},
        ) from error
    if not isinstance(document, Mapping):
        raise EnvironmentsError(
            SPEC_INVALID,
            "the environment.yml is empty or not a mapping",
            detail={"field": "spec.build.dependencyFile.content"},
        )
    raw_dependencies = document.get("dependencies")
    if raw_dependencies is None:
        raise EnvironmentsError(
            SPEC_INVALID,
            "the environment.yml names no `dependencies`",
            detail={"field": "spec.build.dependencyFile.content"},
        )
    if not isinstance(raw_dependencies, Sequence) or isinstance(raw_dependencies, (str, bytes)):
        raise EnvironmentsError(
            SPEC_INVALID,
            "`dependencies` in the environment.yml is not a list",
            detail={"field": "spec.build.dependencyFile.content.dependencies"},
        )
    conda: list[str] = []
    pip: list[str] = []
    seen_pip = False
    for index, entry in enumerate(raw_dependencies):
        field_name = f"spec.build.dependencyFile.content.dependencies[{index}]"
        if isinstance(entry, str):
            spec = entry.strip()
            if spec:
                conda.append(spec)
            continue
        if isinstance(entry, Mapping) and set(entry) == {_PIP_SECTION_KEY}:
            if seen_pip:
                raise EnvironmentsError(
                    SPEC_INVALID,
                    "the environment.yml names more than one `pip:` section",
                    detail={"field": field_name},
                )
            seen_pip = True
            pip.extend(_pip_requirements(entry[_PIP_SECTION_KEY], field_name))
            continue
        raise EnvironmentsError(
            SPEC_INVALID,
            "a `dependencies` entry in the environment.yml is neither a conda "
            "spec nor a single `pip:` section",
            detail={"field": field_name},
        )
    channels = _channels(document)
    return CondaEnvironment(
        name=str(document.get("name") or "environment"),
        channels=channels,
        conda_dependencies=tuple(conda),
        pip_dependencies=tuple(pip),
    )


def _pip_requirements(entries: Any, field_name: str) -> list[str]:
    """The strings of a ``pip:`` section, or a refusal naming its position."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise EnvironmentsError(
            SPEC_INVALID,
            "the `pip:` section in the environment.yml is not a list",
            detail={"field": f"{field_name}.pip"},
        )
    requirements: list[str] = []
    for pip_index, requirement in enumerate(entries):
        if not isinstance(requirement, str):
            raise EnvironmentsError(
                SPEC_INVALID,
                "a `pip:` entry in the environment.yml is not a string",
                detail={"field": f"{field_name}.pip[{pip_index}]"},
            )
        spec = requirement.strip()
        if spec:
            requirements.append(spec)
    return requirements


def _channels(document: Mapping[str, Any]) -> tuple[str, ...]:
    raw = document.get("channels")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise EnvironmentsError(
            SPEC_INVALID,
            "`channels` in the environment.yml is not a list",
            detail={"field": "spec.build.dependencyFile.content.channels"},
        )
    channels: list[str] = []
    for index, channel in enumerate(raw):
        if not isinstance(channel, str):
            raise EnvironmentsError(
                SPEC_INVALID,
                "a `channels` entry in the environment.yml is not a string",
                detail={"field": f"spec.build.dependencyFile.content.channels[{index}]"},
            )
        text = channel.strip()
        if text:
            channels.append(text)
    return tuple(channels)


def merge_conda_pip(
    environment: CondaEnvironment,
    pins: Sequence[ProtectedPin] | None = None,
) -> MergedRequirements:
    """Datalayer's protected pins merged over the environment's ``pip:`` layer.

    The conda layer is Datalayer's to leave alone — the kernel stack is a pip
    concern — but the pip layer is merged exactly as a ``packages`` source's
    dependencies are (:func:`merge_requirements`): a pip requirement that
    contradicts a protected pin is refused, and every pin is forced in whether
    or not the environment named it. The one that has no index — the
    jupyter-server fork wheel (E1-04) — is satisfied from the wheelhouse the
    same ``--find-links`` reaches in the pip step below.
    """
    return merge_requirements(environment.pip_dependencies, pins=pins)


def rendered_environment(
    environment: CondaEnvironment,
    merged: MergedRequirements,
    *,
    python_version: str,
) -> str:
    """The ``environment.yml`` the solve is actually given.

    The user's conda packages and channels, with two things settled: the
    interpreter is pinned to the base's ``python`` (D-9, never silently
    replaced), and the ``pip:`` section is the merged one — the user's pip
    requirements with the protected pins forced over them.
    """
    import yaml

    conda_without_python = [
        spec for spec in environment.conda_dependencies if _conda_package_name(spec) != "python"
    ]
    dependencies: list[Any] = [f"python={python_version}", *conda_without_python]
    pip_layer = list(merged.requirements)
    if pip_layer:
        dependencies.append({_PIP_SECTION_KEY: pip_layer})
    document = {
        "name": environment.name,
        "channels": list(environment.channels) or ["conda-forge"],
        "dependencies": dependencies,
    }
    return yaml.safe_dump(document, sort_keys=False, default_flow_style=False)


def _conda_package_name(spec: str) -> str:
    """The package a conda spec names, lowercased: ``python`` from
    ``python=3.13``, ``python >=3.11`` or ``python[version='3.13']``."""
    return re.split(r"[\s=<>!~\[]", spec.strip(), maxsplit=1)[0].strip().lower()


# -- What a runner is asked, and what it answers ------------------------------


@dataclass(frozen=True)
class CondaResolveRequest:
    """One conda solve: what to resolve, for which interpreter, from where."""

    environment_yml: str
    """The rendered ``environment.yml`` — interpreter pinned, pins merged."""

    python_version: str
    platform: str = CONDA_PLATFORM
    base_reference: str = ""
    """The base the solve runs inside, pinned by digest (D-9)."""

    registry_auth: Mapping[str, str] | None = None
    """What the registry needs to be read, when the runner pulls the base."""


@dataclass
class CondaResolveOutcome:
    """A conda solve's answer: the explicit lock, verbatim from ``micromamba``."""

    lock_text: str


class CondaResolveRunner(Protocol):
    """Where a conda solve runs."""

    name: str

    def solve(
        self, request: CondaResolveRequest, log: Callable[[str], None] | None = None
    ) -> CondaResolveOutcome: ...


# -- Reading micromamba's refusals --------------------------------------------

_NOTHING_PROVIDES = re.compile(
    r"nothing provides (?:requested )?(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
)
_PACKAGE_NOT_FOUND = re.compile(
    r"(?:package|libmamba).*?(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*) is not available"
)


def parse_conda_failure(output: str) -> EnvironmentsError:
    """``micromamba``'s refusal as one of section 10's codes.

    A package no channel serves, and an unsatisfiable set of packages, are the
    version's own to fix and are reported as such. Anything else — the channel
    unreachable, ``micromamba`` missing — is ``DL_ENV_PROVIDER_ERROR``, which is
    retryable, because nothing about the version is wrong.
    """
    text = " ".join(line.strip() for line in output.splitlines() if line.strip())
    detail: dict[str, Any] = {"output": text[:2000]}

    provides = _NOTHING_PROVIDES.search(text)
    if provides:
        name = provides.group("name")
        return EnvironmentsError(
            PACKAGE_NOT_FOUND,
            f"`{name}` is not in any channel this environment may read",
            detail={**detail, "package": name},
        )
    unavailable = _PACKAGE_NOT_FOUND.search(text)
    if unavailable:
        name = unavailable.group("name")
        return EnvironmentsError(
            PACKAGE_NOT_FOUND,
            f"`{name}` is not in any channel this environment may read",
            detail={**detail, "package": name},
        )
    lowered = text.lower()
    if (
        "could not solve" in lowered
        or "unsolvable" in lowered
        or "encountered problems while solving" in lowered
        or "no solution" in lowered
    ):
        return EnvironmentsError(
            RESOLVE_CONFLICT,
            "The conda dependencies cannot be satisfied together: " + text[:400],
            detail=detail,
        )
    return EnvironmentsError(
        PROVIDER_ERROR,
        "The conda resolver did not finish: " + text[:400],
        detail=detail,
    )


# -- Running the solve --------------------------------------------------------


class MicromambaResolveRunner:
    """``micromamba`` where this runs: for ``plane local`` and for tests.

    Creates the environment the ``environment.yml`` asks for into a scratch
    prefix and exports it as an explicit lock. The prefix is thrown away; only
    the lock is kept, which is the one thing every variant then builds from.
    """

    name = "micromamba"

    def __init__(self, micromamba: str | None = None, timeout: float = 900.0) -> None:
        # `None` means "find it"; an empty string means "there is none", which
        # is how a test says so without hiding micromamba from the process.
        self._micromamba = (shutil.which("micromamba") or "") if micromamba is None else micromamba
        self._timeout = timeout

    def solve(
        self, request: CondaResolveRequest, log: Callable[[str], None] | None = None
    ) -> CondaResolveOutcome:
        say = log or (lambda _line: None)
        if not self._micromamba:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No `micromamba` to resolve with: the local conda resolver needs it on the PATH",
                detail={"missing": "micromamba", "runner": self.name},
            )
        with tempfile.TemporaryDirectory(prefix="dl-conda-") as directory:
            root = Path(directory)
            spec_file = root / "environment.yml"
            spec_file.write_text(request.environment_yml, encoding="utf-8")
            prefix = root / "prefix"
            create = [
                self._micromamba,
                "create",
                "--yes",
                "--prefix",
                str(prefix),
                "--platform",
                request.platform,
                "--file",
                str(spec_file),
            ]
            say(f"Resolving the conda environment for {request.platform}")
            try:
                created = subprocess.run(  # noqa: S603 - the argv is built here
                    create,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                    check=False,
                    env={**_os_environ(), "PIP_FIND_LINKS": str(WHEELHOUSE_PATH)},
                )
            except subprocess.TimeoutExpired as expired:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The conda solve did not finish within {self._timeout:.0f}s",
                    detail={"runner": self.name, "timeout": self._timeout},
                ) from expired
            if created.returncode != 0:
                for line in (created.stderr or "").splitlines():
                    say(line)
                raise parse_conda_failure(created.stderr or created.stdout or "")
            export = subprocess.run(  # noqa: S603 - the argv is built here
                [self._micromamba, "env", "export", "--explicit", "--prefix", str(prefix)],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
            if export.returncode != 0:
                for line in (export.stderr or "").splitlines():
                    say(line)
                raise parse_conda_failure(export.stderr or export.stdout or "")
        return CondaResolveOutcome(lock_text=export.stdout)


class BuildkitCondaResolveRunner:
    """D-9's conda solve: ``micromamba`` inside the resolved base, under BuildKit.

    The solve gets the builder's isolation, its egress allowlist and the exact
    interpreter the artifact will have. It needs ``buildctl`` and a reachable
    ``buildkitd``, neither of which exists until the build pool is deployed
    (E1-06), so until then it refuses by name rather than resolving elsewhere.
    """

    name = "buildkit-conda"

    def __init__(
        self,
        buildctl: str | None = None,
        address: str | None = None,
        tlscert: str | None = None,
        tlskey: str | None = None,
        tlscacert: str | None = None,
        timeout: float = 1200.0,
    ) -> None:
        self._buildctl = (shutil.which("buildctl") or "") if buildctl is None else buildctl
        self._tlscert = tlscert or ""
        self._tlskey = tlskey or ""
        self._tlscacert = tlscacert or ""
        self._address = address or ""
        self._timeout = timeout

    def _tls_options(self) -> list[str]:
        if not (self._tlscert and self._tlskey and self._tlscacert):
            return []
        return [
            f"--tlscert={self._tlscert}",
            f"--tlskey={self._tlskey}",
            f"--tlscacert={self._tlscacert}",
        ]

    def dockerfile(self, request: CondaResolveRequest) -> str:
        """The solve, as the frontend reads it: create the env, then export it.

        The ``environment.yml`` is a spec field a user wrote, so its path is the
        only thing that reaches the ``RUN`` line — its content is a file copied
        into the context, never interpolated into a shell command — and the
        wheelhouse is brought along for the one protected pin no index has
        (E1-04), reached through ``PIP_FIND_LINKS`` the same way the local
        runner reaches it.
        """
        return "\n".join(
            [
                f"FROM {request.base_reference} AS solve",
                "USER root",
                "WORKDIR /solve",
                "COPY environment.yml ./environment.yml",
                "ENV PIP_FIND_LINKS=" + shlex.quote(WHEELHOUSE_IMAGE_PATH),
                "RUN --mount=type=cache,target=/opt/conda/pkgs "
                "micromamba create --yes --prefix /solve/prefix "
                f"--platform {shlex.quote(request.platform)} --file environment.yml",
                "RUN micromamba env export --explicit --prefix /solve/prefix > /solve/lock.txt",
                "FROM scratch",
                "COPY --from=solve /solve/lock.txt /lock.txt",
            ]
        ) + "\n"

    def solve(
        self, request: CondaResolveRequest, log: Callable[[str], None] | None = None
    ) -> CondaResolveOutcome:
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
        with tempfile.TemporaryDirectory(prefix="dl-conda-solve-") as directory:
            root = Path(directory)
            (root / "environment.yml").write_text(request.environment_yml, encoding="utf-8")
            (root / "Dockerfile").write_text(self.dockerfile(request), encoding="utf-8")
            out = root / "out"
            command = [
                self._buildctl,
                *(["--addr", self._address] if self._address else []),
                *self._tls_options(),
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
            say(f"Solving the conda lock in {request.base_reference}")
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
                    f"The conda solve did not finish within {self._timeout:.0f}s",
                    detail={"runner": self.name, "timeout": self._timeout},
                ) from expired
            for line in (finished.stderr or "").splitlines():
                say(line)
            if finished.returncode != 0:
                raise parse_conda_failure(finished.stderr or finished.stdout or "")
            lock = (out / "lock.txt").read_text(encoding="utf-8")
        return CondaResolveOutcome(lock_text=lock)

    def _environment(self, request: CondaResolveRequest) -> dict[str, str] | None:
        auth = dict(request.registry_auth or {})
        if not auth:
            return None
        return {**_os_environ(), **{str(key): str(value) for key, value in auth.items()}}


def _os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


# -- The lock -----------------------------------------------------------------


def explicit_lock_packages(lock_text: str) -> list[str]:
    """Every package an explicit lock installs, one URL per line.

    An ``@EXPLICIT`` file is comments, the ``@EXPLICIT`` marker, and then one
    ``https://…/pkg.conda#hash`` line per package; the URLs are what a build
    installs and what this counts.
    """
    packages: list[str] = []
    for raw in lock_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line == _EXPLICIT_MARKER:
            continue
        packages.append(line)
    return packages


def is_conda_lock(lock_text: str | None) -> bool:
    """Whether a lock is a conda explicit lock, and not the pip one.

    A conda lock carries the ``@EXPLICIT`` marker; a pip lock never does. A
    builder reads this to install with ``micromamba`` rather than ``uv pip
    sync`` — the one signal that travels with the lock text itself, so a
    builder handed only :attr:`BuildRequest.lock_text` still knows which it is.
    """
    if not lock_text:
        return False
    return any(line.strip() == _EXPLICIT_MARKER for line in lock_text.splitlines())


def conda_lock_protected_pins(lock_text: str) -> list[str]:
    """The pip requirements a conda lock's header records as Datalayer's pins.

    :func:`conda_lock_document` writes the protected pip pins as
    ``# datalayer-protected: <req>`` lines above the ``@EXPLICIT`` body. A
    builder installs the conda layer from the body and then this pip layer, so
    the kernel stack (E1-04) is present the same way it is for every source.
    """
    prefix = PROTECTED_PIN_PREFIX.strip()
    pins: list[str] = []
    for raw in lock_text.splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            requirement = line[len(prefix) :].strip()
            if requirement:
                pins.append(requirement)
    return pins


def conda_lock_document(
    outcome: CondaResolveOutcome,
    *,
    python_version: str,
    base_reference: str,
    merged: MergedRequirements,
    platform: str = CONDA_PLATFORM,
    resolved_at: datetime | None = None,
) -> dict[str, Any]:
    """The stored conda lock: its text, its digest, and what a reader needs.

    The protected pins are written as comments above the explicit lock, the
    same ``# datalayer-protected:`` lines the pip lock carries, so the one
    document says the whole of what a build installs — the conda packages by
    URL and hash, and the pip pins Datalayer forced over the pip layer — while
    staying a file ``micromamba create --file`` reads unchanged.
    """
    when = (resolved_at or _utcnow()).replace(microsecond=0).isoformat()
    header = [
        "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.",
        f"# resolved-at: {when}",
        f"# python: {python_version}",
        f"# platform: {platform}",
        f"# base: {base_reference}",
    ]
    for constraint in merged.constraints:
        header.append(f"{PROTECTED_PIN_PREFIX}{constraint}")
    body = outcome.lock_text.lstrip("\n")
    if _EXPLICIT_MARKER not in {line.strip() for line in body.splitlines()}:
        raise EnvironmentsError(
            PROVIDER_ERROR,
            "micromamba did not produce an explicit lock (no @EXPLICIT marker)",
            detail={"format": CONDA_LOCK_FORMAT},
        )
    text = "\n".join(header) + "\n" + body
    if not text.endswith("\n"):
        text += "\n"
    packages = explicit_lock_packages(text)
    return {
        "digest": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "format": CONDA_LOCK_FORMAT,
        "content": text,
        "python_version": python_version,
        "package_count": len(packages),
    }


def resolve_conda_environment(
    *,
    environment_yml: str,
    python_version: str,
    resolved_bases: Mapping[str, str],
    platform: str = CONDA_PLATFORM,
    credential: Any = None,
    log: Callable[[str], None] | None = None,
    runner: CondaResolveRunner | None = None,
    resolved_at: datetime | None = None,
) -> dict[str, Any]:
    """A conda version's lock, from its ``environment.yml`` (E3-02).

    This is the conda seam of :func:`code_sandboxes.environments.resolve.resolve_environment`:
    it takes the ``environment.yml`` a ``dependencyFile`` source carries and the
    bases already resolved for the wanted variants, and answers the same lock
    document every other source answers — ``digest``, ``format``, ``content``,
    ``python_version``, ``package_count`` — so the workflow stores it the same way.

    Raises
    ------
    EnvironmentsError
        Everything a person can act on: a malformed ``environment.yml``, a pip
        requirement that contradicts a protected pin, a conflict, or a package
        no channel serves.
    """
    say = log or (lambda _line: None)
    environment = parse_conda_environment(environment_yml)
    merged = merge_conda_pip(environment)
    for note in merged.notes:
        say(note)
    rendered = rendered_environment(environment, merged, python_version=python_version)
    solving_in = resolved_bases.get("datalayer") or next(iter(resolved_bases.values()))
    request = CondaResolveRequest(
        environment_yml=rendered,
        python_version=python_version,
        platform=platform,
        base_reference=solving_in,
        registry_auth=_registry_auth(credential),
    )
    outcome = (runner or BuildkitCondaResolveRunner()).solve(request, say)
    document = conda_lock_document(
        outcome,
        python_version=python_version,
        base_reference=solving_in,
        merged=merged,
        platform=platform,
        resolved_at=resolved_at,
    )
    say(f"Locked {document['package_count']} conda packages as {document['digest']}")
    return {**document, "resolved_bases": dict(resolved_bases)}


def _registry_auth(credential: Any) -> Mapping[str, str] | None:
    """The credential's registry auth, however it carries it — as
    :func:`code_sandboxes.environments.resolve._registry_auth` reads it, kept
    in step so both resolvers accept the one credential shape durable mints."""
    if credential is None:
        return None
    for attribute in ("registry_auth", "environment", "env"):
        value = getattr(credential, attribute, None)
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
    return None
