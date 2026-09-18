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
    WHEELHOUSE_IMAGE_PATH,
    WHEELHOUSE_PATH,
    MergedRequirements,
    ProtectedPin,
    buildkit_proxy,
    buildkit_proxy_options,
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
    "conda_lock_pip_requirements",
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

#: The marker the ``@EXPLICIT`` conda lock body opens with, and the key the
#: ``environment.yml`` names its pip layer under.
_EXPLICIT_MARKER = "@EXPLICIT"
_PIP_SECTION_KEY = "pip"

#: How the whole pip layer is recorded in the lock's header — the user's pip
#: requirements *and* the protected pins Datalayer forces over them, pinned to
#: the versions the solve resolved — one ``# datalayer-pip: <req>`` line each,
#: above the ``@EXPLICIT`` body. A builder installs the conda layer from the
#: body and then this pip layer, so the whole of what a version resolved to is
#: in the one document and nothing resolved in the solve is lost from the build.
CONDA_PIP_PREFIX = "# datalayer-pip: "

#: The ``micromamba`` the conda solve and every conda build use, pinned so the
#: tool that resolves is the tool that installs (E3-02): the approved base bakes
#: uv, the wheelhouse and the doctor, but not micromamba, so it is brought in
#: here rather than assumed. A BuildKit build copies the binary from this image;
#: a builder driving an SDK installs the same pinned release.
MICROMAMBA_VERSION = "2.0.5"
MICROMAMBA_IMAGE = f"mambaorg/micromamba:{MICROMAMBA_VERSION}"
MICROMAMBA_BINARY = "/usr/local/bin/micromamba"

#: A channel URL that carries a credential in its userinfo — the same shape
#: :func:`code_sandboxes.environments.spec._index_findings` refuses in an index
#: URL, so a token is caught the same way whichever field names it.
_URL_CREDENTIALS = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]*@", re.IGNORECASE)


def micromamba_bootstrap_dockerfile_line() -> str:
    """The Dockerfile line that brings the pinned micromamba into a build.

    ``COPY --from`` the pinned micromamba image, so the binary is present and
    reproducible without a network fetch inside the build itself. Used by the
    resolver's own solve image and by the Datalayer (BuildKit) builder.
    """
    return f"COPY --from={MICROMAMBA_IMAGE} /bin/micromamba {MICROMAMBA_BINARY}"


def micromamba_bootstrap_command() -> str:
    """The shell command that installs the pinned micromamba into a build.

    For a builder that drives an SDK (E2B, Daytona) rather than emitting a
    Dockerfile: the same pinned release ``COPY --from`` brings, fetched into
    ``/usr/local/bin`` so a later ``micromamba install`` finds it on the PATH.
    """
    return (
        f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/{MICROMAMBA_VERSION} "
        "| tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba"
    )


#: Where an approved Datalayer base keeps its interpreter: `python` on the
#: PATH is `/opt/conda/bin/python` (E1-05), and that is the environment a
#: sandbox's kernel runs in.
CONDA_PREFIX = "/opt/conda"


def micromamba_install_command(lock_path: str, *, micromamba: str = "micromamba") -> str:
    """The command that installs an explicit lock into the base's own interpreter.

    **Into `/opt/conda`, by name.** The base sets no `MAMBA_ROOT_PREFIX`, and
    left to itself `micromamba install --name base` makes a *new* `base` under
    `~/.local/share/mamba` — for a Datalayer base, inside the very home the
    runtime mounts a person's content over, and nowhere `python` looks. The
    first real conda build (2026-09-18) installed all 81 packages there, and
    then failed its own `postInstall` with `No module named 'osgeo'`. Both the
    root and the target prefix are named, so nothing depends on which user the
    step happens to run as.
    """
    return (
        f"{micromamba} install --yes --root-prefix {CONDA_PREFIX} "
        f"--prefix {CONDA_PREFIX} --file {lock_path}"
    )


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
            if _URL_CREDENTIALS.match(text):
                # The same refusal `spec.packages.python.indexes` gives a
                # credential-bearing index URL: a token in the channel is
                # copied into the solve context and can reappear in the
                # explicit lock, so it belongs in the build's secrets, not here.
                raise EnvironmentsError(
                    SPEC_INVALID,
                    "a `channels` entry carries a credential in its URL; reference the "
                    "credential in `buildSecrets`",
                    detail={"field": f"spec.build.dependencyFile.content.channels[{index}]"},
                )
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
    """A conda solve's answer: the explicit lock, and the pip layer it resolved.

    ``lock_text`` is the ``@EXPLICIT`` conda lock, verbatim from ``micromamba``.
    ``pip_lock`` is the pip layer the same solve installed — the user's pip
    requirements and the protected pins forced over them — pinned to the
    versions it resolved, read back from ``micromamba env export`` so the
    artifact carries the whole of what the solve produced, not the conda layer
    alone (E3-02). A runner that cannot read the prefix back leaves it empty,
    and :func:`conda_lock_document` falls back to the merged requirements.
    """

    lock_text: str
    pip_lock: tuple[str, ...] = ()


def pip_requirements_from_env_yaml(text: str) -> tuple[str, ...]:
    """The pip layer a ``micromamba env export`` names, pinned, in order.

    A conda ``env export`` (the YAML form, not ``--explicit``) lists the pip
    packages it installed under a single ``{"pip": [...]}`` entry of its
    ``dependencies``, each ``name==version`` — cleanly separated from the conda
    packages, which are their own strings. This reads that section back, so the
    solve's resolved pip versions become the lock's pip layer. A malformed or
    pip-less export is an empty layer, never a raised error: the explicit lock
    is what a solve is judged by, and its own marker is checked elsewhere.
    """
    import yaml

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return ()
    if not isinstance(document, Mapping):
        return ()
    dependencies = document.get("dependencies")
    if not isinstance(dependencies, Sequence) or isinstance(dependencies, (str, bytes)):
        return ()
    requirements: list[str] = []
    for entry in dependencies:
        if isinstance(entry, Mapping) and _PIP_SECTION_KEY in entry:
            for requirement in entry[_PIP_SECTION_KEY] or []:
                if isinstance(requirement, str) and requirement.strip():
                    requirements.append(requirement.strip())
    return tuple(requirements)


#: What the solved prefix's own interpreter is asked, to list the pip layer
#: whole. Standard library only, and it runs inside the prefix.
#:
#: ``micromamba env export`` names the pip packages the *file* asked for and
#: nothing they pulled in: the first real solve (2026-09-18) exported 7 pip
#: pins where pip had installed several dozen distributions, so every build
#: would have resolved ``tornado``, ``pyzmq``, ``traitlets`` and the rest
#: afresh — on each variant, on whichever day it ran — under a lock that
#: claimed to say the whole of what a build installs. A distribution records
#: who installed it in its own ``INSTALLER`` file, which is what tells the pip
#: layer from the conda packages that also carry Python metadata.
PIP_LAYER_SCRIPT = """\
import importlib.metadata as metadata

layer = {}
for distribution in metadata.distributions():
    installer = (distribution.read_text("INSTALLER") or "").strip().lower()
    name = distribution.metadata["Name"]
    if installer in ("pip", "uv") and name:
        layer[name.lower().replace("_", "-")] = distribution.version
for name in sorted(layer):
    print(f"{name}=={layer[name]}")
"""


def pip_requirements_from_listing(text: str) -> tuple[str, ...]:
    """The pip layer :data:`PIP_LAYER_SCRIPT` printed: ``name==version`` lines.

    Anything else on a line — a warning an interpreter wrote to the same
    stream — is dropped rather than installed.
    """
    requirements: list[str] = []
    for line in text.splitlines():
        entry = line.strip()
        name, separator, version = entry.partition("==")
        if separator and name and version and " " not in entry:
            requirements.append(entry)
    return tuple(requirements)


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
            export = self._export(
                [self._micromamba, "env", "export", "--explicit", "--prefix", str(prefix)],
                say,
            )
            # The pip layer the same solve installed, pinned, read back from the
            # YAML export's own `pip:` section (E3-02): the explicit export above
            # carries conda packages alone, so without this the user's resolved
            # pip requirements would be absent from the artifact.
            pip_export = self._export(
                [self._micromamba, "env", "export", "--prefix", str(prefix)],
                say,
            )
            # Every distribution pip installed, the transitive ones included:
            # the export above names only what the file asked for.
            listing = self._export(
                [str(prefix / "bin" / "python"), "-c", PIP_LAYER_SCRIPT],
                say,
            )
        return CondaResolveOutcome(
            lock_text=export.stdout,
            pip_lock=pip_requirements_from_listing(listing.stdout)
            or pip_requirements_from_env_yaml(pip_export.stdout),
        )

    def _export(
        self, argv: list[str], say: Callable[[str], None]
    ) -> subprocess.CompletedProcess[str]:
        """One ``micromamba env export``, its timeout handled the same as the solve.

        The ``create`` above and both exports share the one refusal so a timeout
        anywhere becomes ``DL_ENV_PROVIDER_ERROR`` rather than a raw
        :class:`subprocess.TimeoutExpired` a caller cannot classify or retry.
        """
        try:
            result = subprocess.run(  # noqa: S603 - the argv is built here
                argv,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"The conda solve did not finish within {self._timeout:.0f}s",
                detail={"runner": self.name, "timeout": self._timeout},
            ) from expired
        if result.returncode != 0:
            for line in (result.stderr or "").splitlines():
                say(line)
            raise parse_conda_failure(result.stderr or result.stdout or "")
        return result


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
        proxy: str | None = None,
    ) -> None:
        self._buildctl = (shutil.which("buildctl") or "") if buildctl is None else buildctl
        #: The build pool's egress proxy (E1-06), or `DATALAYER_BUILDKIT_PROXY`.
        self._proxy = buildkit_proxy(proxy)
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
        runner reaches it. ``micromamba`` is copied in from its pinned image
        first, because the approved base bakes uv and the wheelhouse but not it.
        The explicit lock and the pip layer are both exported, so the artifact
        carries the whole of what the solve resolved, not the conda layer alone.
        """
        micromamba = shlex.quote(MICROMAMBA_BINARY)
        return (
            "\n".join(
                [
                    f"FROM {request.base_reference} AS solve",
                    "USER root",
                    "WORKDIR /solve",
                    micromamba_bootstrap_dockerfile_line(),
                    "COPY environment.yml ./environment.yml",
                    "ENV PIP_FIND_LINKS=" + shlex.quote(WHEELHOUSE_IMAGE_PATH),
                    "RUN --mount=type=cache,target=/opt/conda/pkgs "
                    f"{micromamba} create --yes --prefix /solve/prefix "
                    f"--platform {shlex.quote(request.platform)} --file environment.yml",
                    f"RUN {micromamba} env export --explicit "
                    "--prefix /solve/prefix > /solve/lock.txt",
                    f"RUN {micromamba} env export --prefix /solve/prefix > /solve/pip-env.yml",
                    # The pip layer whole, from the prefix's own interpreter:
                    # a file of this package's, never a spec field.
                    "COPY pip_layer.py ./pip_layer.py",
                    "RUN /solve/prefix/bin/python pip_layer.py > /solve/pip-lock.txt",
                    "FROM scratch",
                    "COPY --from=solve /solve/lock.txt /lock.txt",
                    "COPY --from=solve /solve/pip-env.yml /pip-env.yml",
                    "COPY --from=solve /solve/pip-lock.txt /pip-lock.txt",
                ]
            )
            + "\n"
        )

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
            (root / "pip_layer.py").write_text(PIP_LAYER_SCRIPT, encoding="utf-8")
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
                *buildkit_proxy_options(self._proxy),
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
            pip_env = out / "pip-env.yml"
            pip_listing = out / "pip-lock.txt"
            pip_lock = (
                pip_requirements_from_listing(pip_listing.read_text(encoding="utf-8"))
                if pip_listing.exists()
                else ()
            ) or (
                pip_requirements_from_env_yaml(pip_env.read_text(encoding="utf-8"))
                if pip_env.exists()
                else ()
            )
        return CondaResolveOutcome(lock_text=lock, pip_lock=pip_lock)

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


#: `name-version-build.conda` (or `.tar.bz2`), as the last segment of an
#: explicit lock's URL. A conda name may itself contain `-`, so the version and
#: the build are the last two dash-separated fields, never the first two.
_EXPLICIT_PACKAGE = re.compile(
    r"/(?P<name>[^/]+)-(?P<version>[^-/]+)-(?P<build>[^-/]+)\.(?:conda|tar\.bz2)(?:#.*)?$"
)


def conda_lock_python_packages(lock_text: str) -> dict[str, str]:
    """The Python distributions an explicit lock installs, by name, with their versions.

    Only the ones a build string marks as Python packages — `py313h…` for a
    compiled one, `pyh…`/`pyhd8ed…` for a noarch one. A conda lock is mostly
    libraries with no Python metadata at all (`libgdal-core`, `proj`, `openssl`),
    and Appendix B check 5 asks the interpreter for a distribution's version:
    handing it `proj` would fail a check with nothing wrong to report.
    """
    found: dict[str, str] = {}
    for url in explicit_lock_packages(lock_text):
        match = _EXPLICIT_PACKAGE.search(url)
        if match and match["build"].startswith("py"):
            found[match["name"].lower()] = match["version"]
    return found


def conda_expected_packages(environment_yml: str, lock_text: str) -> dict[str, str]:
    """What an `environment.yml` names at its top level, pinned to what its lock resolved.

    Check 5's question, for a conda source (E3-02): the file's own `pip:`
    requirements, and the conda packages it names that are Python
    distributions. The interpreter is left out — check 3 asks about it, and
    `python` is not a distribution the interpreter reports about itself.
    Empty for a file that cannot be read: the spec's own validation refuses
    one long before a build, and a smoke test is not where to say so again.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.utils import canonicalize_name

    try:
        environment = parse_conda_environment(environment_yml)
    except EnvironmentsError:
        return {}
    expected: dict[str, str] = {}
    pythons = conda_lock_python_packages(lock_text)
    for spec in environment.conda_dependencies:
        name = _conda_package_name(spec)
        if name != "python" and name in pythons:
            expected[canonicalize_name(name)] = pythons[name]
    pinned: dict[str, str] = {}
    for requirement in conda_lock_pip_requirements(lock_text):
        name, separator, version = requirement.partition("==")
        if separator:
            pinned[canonicalize_name(name.strip())] = version.strip()
    for text in environment.pip_dependencies:
        try:
            name = canonicalize_name(Requirement(text).name)
        except InvalidRequirement:
            continue
        if name in pinned:
            expected[name] = pinned[name]
    return expected


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


def conda_lock_pip_requirements(lock_text: str) -> list[str]:
    """The whole pip layer a conda lock's header records, in order.

    :func:`conda_lock_document` writes the pip layer the solve resolved as
    ``# datalayer-pip: <req>`` lines above the ``@EXPLICIT`` body: the user's
    own pip requirements and the protected pins Datalayer forced over them,
    pinned to the versions the solve produced. A builder installs the conda
    layer from the body and then this pip layer, so the whole of what the
    version resolved to is built and nothing the solve installed is lost (E3-02).
    """
    prefix = CONDA_PIP_PREFIX.strip()
    requirements: list[str] = []
    for raw in lock_text.splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            requirement = line[len(prefix) :].strip()
            if requirement:
                requirements.append(requirement)
    return requirements


def conda_lock_document(
    outcome: CondaResolveOutcome,
    *,
    python_version: str,
    base_reference: str,
    merged: MergedRequirements,
    platform: str = CONDA_PLATFORM,
) -> dict[str, Any]:
    """The stored conda lock: its text, its digest, and what a reader needs.

    The pip layer the solve resolved is written as comments above the explicit
    lock — the ``# datalayer-pip:`` lines a builder installs after the conda
    packages — so the one document says the whole of what a build installs: the
    conda packages by URL and hash, and the user's pip requirements with the
    protected pins Datalayer forced over them, pinned to the versions the solve
    produced. It stays a file ``micromamba create --file`` reads unchanged.
    The pip layer is the solve's own (``outcome.pip_lock``) when the runner
    could read the prefix back, and the merged requirements otherwise, so it is
    always complete rather than the protected pins alone.
    """
    header = [
        "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.",
        f"# python: {python_version}",
        f"# platform: {platform}",
        f"# base: {base_reference}",
    ]
    pip_layer = list(outcome.pip_lock) or list(merged.requirements)
    for requirement in pip_layer:
        header.append(f"{CONDA_PIP_PREFIX}{requirement}")
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
