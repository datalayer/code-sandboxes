# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Daytona variant's Environment builder (PLAN_ENV.md §6, §11.3, E2-04, E2-06).

Daytona's artifact is a **snapshot**, and two of its properties shape
everything here:

- **A snapshot bakes in CPU, memory, disk and GPU**, so a resource change is a
  new artifact for the same version rather than a different launch of the same
  one — which is why the artifact table has a region column and why the
  adapter refuses resources beside a snapshot at launch (correction 13).
- **A snapshot is region-scoped**, so `compatibility.regions` is load-bearing:
  an artifact built in one region cannot be launched in another.

It also rejects `latest`, `lts` and `stable` as the tag of a snapshot's source
image, and prefers a digest — which is what the Datalayer base channel
resolves to anyway (D-9).

**The base is pulled through a private registry entry made for this build**
(D-17, D-18): Daytona's own ECR option (`from_aws_registry`-equivalent) takes
only a standing role, which is not what a per-build credential is, so the
base-reader session's login token is registered as a Docker registry entry
instead — `AWS` and the token, the same shape E2B's `from_image` takes
(E0-04). The entry is organization-wide and never returns its password once
written, so it is named uniquely per build and deleted in a `finally`,
whether the build succeeded or not: no standing credential is left in an
account Datalayer does not control (D-18).

**No `useradd`/`set_user` dance, unlike E2B (E2-03).** E0-04 found Daytona
*honours* the image's `USER 1000:100`, `HOME` and `WORKDIR` — the Datalayer
base already carries all three (E1-05) — so this builder only needs `USER
root` around the steps that must run as root, mirroring the Datalayer
builder's own Dockerfile discipline, never a synthetic account.

**Neither the doctor nor the wheelhouse is copied in, unlike E2B.** The
Datalayer base already bakes `/opt/datalayer/bin/datalayer-sandbox` and
`WHEELHOUSE_IMAGE_PATH` (E1-05) — confirmed live, 2026-09-13: a spike that
added this package's own bundled wheelhouse at the same path came back with
*both* files in the built snapshot, the base's own wheel alongside the
freshly-copied one, because a directory `COPY` merges into what is already
there rather than replacing it. Copying either again would only duplicate
bytes the base already has, so this builder copies the lock and nothing
else, the same as the Datalayer builder's own `dockerfile()` does for a
`packages` build source.

**The entrypoint is set explicitly, every build.** Daytona defaults an
unset one to `sleep infinity` (§4.1, §11.3 item 3), which is alive but is
not PID 1 correctly reaping children or forwarding `SIGTERM` (the contract's
own "Signals" row) — the Datalayer base itself bakes no `ENTRYPOINT` of its
own (the platform supplies the runtime pod's command instead, E0-04), so
Daytona's default would otherwise be exactly what runs. `tini -- sleep
infinity` is set instead: long-running, and a real PID 1.

**The doctor is not run at build time, unlike the Datalayer and E2B
builders.** Found live, 2026-09-13, with a real, hash-verified, 310-package
lock (not the trivial ones earlier live builds used): `datalayer-sandbox
doctor --json` failed its `init` row deterministically, twice, with `"pid1":
"python3"` and an orphaned zombie neither reaped. A build-time `RUN` step
does not execute as the final image's own entrypoint — it runs inside
Daytona's own build agent's exec, whose PID 1 is that agent's own `python3`
process, not `tini` or whatever the finished snapshot actually starts as.
Every other row — `uid`, `gid`, `workdir`, the kernel stack, all of it —
passed correctly in that same build; only `init` cannot be truthfully
answered from inside a build step, for any provider whose build runs one
command at a time this way, not only Daytona's. Running it here anyway
would make an artifact refuse to build for a reason that says nothing about
what it does once actually launched. A real answer needs a launched
sandbox, which is a smoke test's job (E1-14), not this builder's.

**A GPU is baked into the snapshot** (E2-17, D-20), from the spec's own
`resources.accelerator`: its `type` is one of Daytona's GPUs, checked at
`validate` so a GPU Daytona does not offer is refused before any build, and
its `count` is the number of them. CPU and memory come from the spec's hints,
or Daytona's own default for that GPU; the disk from the hints, or enough for
the CUDA base. A sandbox of a GPU snapshot is ephemeral, which Daytona
requires of every GPU sandbox (`DaytonaSandbox` reads the snapshot's `gpu`).

@module code_sandboxes.environments.adapters.daytona
"""

from __future__ import annotations

import shlex
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..accounts import provider_account
from ..builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
    ValidationResult,
)
from ..contract import SANDBOX_CONTRACT_V1, pin_dockerfile_base
from ..errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    EnvironmentsError,
)
from ..files import files_step
from ..resolve import WHEELHOUSE_IMAGE_PATH, apt_pins_in
from ..resolve_conda import (
    conda_lock_pip_requirements,
    is_conda_lock,
    micromamba_bootstrap_command,
    micromamba_install_command,
)
from ..spec import GPU_SIZE_CLASSES, Environment, EnvironmentSpec
from .managed import ManagedBuilder

__all__ = ["DAYTONA_GPUS", "Builder", "daytona_gpu"]

#: Tags Daytona refuses for a snapshot's source image: each moves.
MOVING_TAGS = ("latest", "lts", "stable")

_LOCK_PATH = "/opt/datalayer/lock.txt"
_CONTENT_DIR = "/home/datalayer/content"

#: A long-running PID 1 (§11.3 item 3, contract's "Entrypoint" and "Signals"
#: rows): the Datalayer base bakes none of its own, and Daytona's own default
#: (`sleep infinity` with no `tini`) does not reap children or forward
#: `SIGTERM`.
_CONTRACT_ENTRYPOINT = ["tini", "--", "sleep", "infinity"]

#: The CPU classes' resources (D-4), duplicated here rather than imported:
#: this package publishes to PyPI and `datalayer_common.size_classes` — the
#: canonical table — does not (found while implementing this item). Kept in
#: step with that table by hand; disk is this adapter's own choice, since D-4
#: prices CPU and memory only. GPU classes are refused in `_own_findings`
#: instead of guessed at here (E2-17).
_CPU_RESOURCES: dict[str, dict[str, int]] = {
    "small": {"cpu": 1, "memory": 2, "disk": 10},
    "medium": {"cpu": 4, "memory": 8, "disk": 20},
    "large": {"cpu": 8, "memory": 16, "disk": 40},
}


#: The GPUs Daytona offers, as its SDK's `GpuType` names them (E2-17). Spelled
#: out rather than read from the SDK, because `validate` runs where the SDK may
#: not be installed; a test holds the two together.
DAYTONA_GPUS: tuple[str, ...] = ("H100", "H200", "RTX-PRO-6000", "RTX-4090", "RTX-5090")

#: A GPU snapshot's disk when the spec gives no hint: the CUDA base alone is
#: about 15 GB, and Daytona's own default would not hold it with room to work.
_GPU_DISK_GI = 50


def daytona_gpu(accelerator_type: str) -> str | None:
    """Daytona's name for an accelerator type, or `None` when Daytona has no such GPU.

    `h100`, `H100` and `rtx_4090` name what Daytona calls `H100` and
    `RTX-4090`; a `T4` or an `A100-80GB` is Modal's vocabulary, not Daytona's.
    """
    name = accelerator_type.strip().upper().replace("_", "-")
    return name if name in DAYTONA_GPUS else None


def _pip_lock_command(*, authored: bool) -> str:
    """Install the pip lock: `sync` to it, or `install` it over an authored Dockerfile.

    `sync` would remove what the Dockerfile installed and the lock does not
    name (E3-03, `DOCKERFILE_COVERAGE`).
    """
    verb, target = ("install", "-r ") if authored else ("sync", "")
    return (
        f"uv pip {verb} --system --require-hashes "
        f"--find-links {WHEELHOUSE_IMAGE_PATH} {target}{_LOCK_PATH}"
    )


def _daytona_sdk() -> Any:
    try:
        import daytona
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise EnvironmentsError(
            PROVIDER_ERROR,
            "No `daytona` SDK to build a snapshot with: install `code-sandboxes[daytona]`",
            detail={"missing": "daytona"},
        ) from error
    return daytona


def _daytona_registry_sdk() -> Any:
    try:
        import daytona_api_client
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise EnvironmentsError(
            PROVIDER_ERROR,
            "No `daytona` SDK to register a base's registry with: "
            "install `code-sandboxes[daytona]`",
            detail={"missing": "daytona_api_client"},
        ) from error
    return daytona_api_client


class Builder(ManagedBuilder):
    """Daytona: the capability half, and the build (E2-04)."""

    variant = "daytona"
    item = "E2-04"
    title = "Daytona"
    #: A `packages` list, or a dependency file: a conda `environment.yml`
    #: installed with `micromamba` (E3-02), and a `requirements.txt` or a
    #: `pyproject.toml` with its lock (E3-01), both of which resolve to the
    #: very pip lock a `packages` list does — `build` tells a conda lock from
    #: a pip one and nothing finer, so nothing here is format-specific.
    build_sources = ("packages", "dependencyFile", "dockerfile")
    #: None beyond the contract's own (E3-03): `Image.from_dockerfile` keeps
    #: the Dockerfile text as it is and Daytona builds it on a real Docker
    #: builder, so the grammar it accepts is Docker's. Checked in the SDK on
    #: 2026-09-17.
    forbidden_instructions = ()
    dependency_formats = ("requirements", "pyproject", "conda")
    #: Daytona runs GPUs, on its own hardware and the owner's account (E2-17),
    #: baked into the snapshot with the rest of its resources.
    gpu = True
    #: E0-04's spike found only a registry login for the private base, never
    #: a per-step arbitrary named secret (E3-05): `buildSecrets` is refused.
    supports_build_secrets = False
    #: The targets the owner's organization may build in. Empty until E2-04
    #: reads them from the organization: refusing a region nobody has listed
    #: would refuse every region.
    regions = ()
    #: 37.4 s for the section 4.1 example in E0-04, plus the wait for the
    #: snapshot to reach `Active`, which is asynchronous.
    max_build_seconds = 30 * 60

    def __init__(
        self,
        *,
        log: Callable[[str], None] | None = None,
        credential: Any = None,
        daytona_sdk: Any = None,
        registry_sdk: Any = None,
    ) -> None:
        super().__init__(log=log, credential=credential)
        self._daytona_sdk = daytona_sdk or _daytona_sdk
        self._registry_sdk = registry_sdk or _daytona_registry_sdk
        self._client_instance: Any = None

    def _provider_secrets(self) -> dict[str, str]:
        """The owner's Daytona secrets the build credential carries (D-8, E2-01).

        `BuildCredential.provider_secrets`, the same way the E2B builder
        reads it — an owner's own `DAYTONA_API_KEY`, never logged, held for
        this build alone.
        """
        secrets = getattr(self._credential, "provider_secrets", None)
        return dict(secrets) if secrets else {}

    def _client(self, sdk: Any) -> Any:
        """The owner's own Daytona client (D-8), built once and reused.

        An owner authenticates with `DAYTONA_API_KEY` **or** the
        `DAYTONA_JWT_TOKEN`/`DAYTONA_ORGANIZATION_ID` pair — `accounts.py`'s
        own `CREDENTIAL_VARIABLES` already lists both, and the JWT form was
        missed here in the first version of this code (found in review):
        a JWT-authenticated owner fell through to the worker's own ambient
        credentials, or failed outright, rather than building in their own
        organization. With neither on the credential, the SDK's own
        constructor falls back to the ambient environment — fine for a
        single-owner worker or a test, wrong for a real multi-owner one.
        """
        if self._client_instance is None:
            secrets = self._provider_secrets()
            api_key = secrets.get("DAYTONA_API_KEY") or None
            jwt_token = secrets.get("DAYTONA_JWT_TOKEN") or None
            organization_id = secrets.get("DAYTONA_ORGANIZATION_ID") or None
            if api_key:
                config = sdk.DaytonaConfig(api_key=api_key)
            elif jwt_token and organization_id:
                config = sdk.DaytonaConfig(jwt_token=jwt_token, organization_id=organization_id)
            else:
                config = None
            self._client_instance = sdk.Daytona(config) if config is not None else sdk.Daytona()
        return self._client_instance

    def _own_findings(
        self, environment: Environment, lock_text: str | None
    ) -> list[CapabilityFinding]:
        findings: list[CapabilityFinding] = []
        tag = environment.spec.base.channel.strip().lower()
        if tag in MOVING_TAGS:
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        f"Daytona refuses `{tag}` as the tag of a snapshot's source image, "
                        "because it moves: name a dated channel"
                    ),
                    field="spec.base.channel",
                )
            )
        # A snapshot carries the machine it was built for, so two regions are
        # two artifacts. Said here so a spec asking for several knows it is
        # asking for several builds, not one.
        if len(environment.spec.compatibility.regions) > 1:
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        "A Daytona snapshot is region-scoped, so "
                        f"{len(environment.spec.compatibility.regions)} regions are "
                        f"{len(environment.spec.compatibility.regions)} artifacts of this version, "
                        "each built and stored separately"
                    ),
                    field="spec.compatibility.regions",
                )
            )
        findings.extend(self._accelerator_findings(environment.spec))
        # `spec.buildSecrets` needs no check of its own here: `supports_build_secrets
        # = False` above (E3-05, merged since this branch started) makes
        # `ManagedBuilder._own_findings` refuse it before this method is
        # even reached — E0-04's spike found only a registry login for the
        # private base, never an arbitrary named secret, the same gap E2B
        # has.
        return findings

    # -- Building -------------------------------------------------------------

    def build(self, request: BuildRequest) -> ArtifactReference:
        """Build a snapshot from the resolved lock, and keep its id.

        The base is pulled through a registry entry made for this build
        alone (D-17, D-18, see the module docstring); `USER root` brackets
        the steps that need it, the same discipline the Datalayer builder's
        own Dockerfile keeps, because Daytona honours the base's `USER`,
        `HOME` and `WORKDIR` rather than overriding them the way E2B does
        (E0-04).
        """
        spec = request.environment.spec
        sdk = self._daytona_sdk()
        client = self._client(sdk)
        name = f"dl-{request.environment.metadata.name}-v{request.version}-{request.build_uid}"

        registry_id = self._register_base_pull(client, request.resolved_base)
        try:
            with tempfile.TemporaryDirectory(prefix="dl-daytona-build-") as scratch:
                lock_file = Path(scratch) / "lock.txt"
                lock_file.write_text(request.lock_text, encoding="utf-8")

                authored = spec.build.dockerfile if spec.build.source == "dockerfile" else None
                image = self._starting_image(sdk, request, authored, Path(scratch))
                # `env` before anything installs, the same order the
                # Datalayer and E2B builders keep: a package that compiles
                # against a library found through an env var behaves
                # differently without it.
                if spec.env:
                    image = image.env(dict(spec.env))
                image = image.dockerfile_commands(["USER root"])
                apt = apt_pins_in(request.lock_text)
                if apt:
                    pinned = " ".join(f"{pkg}={apt[pkg]}" for pkg in sorted(apt))
                    image = image.run_commands(
                        "apt-get update -qq && apt-get install -y --no-install-recommends "
                        f"{pinned} && rm -rf /var/lib/apt/lists/*"
                    )
                # Neither the doctor nor the wheelhouse is copied: the
                # Datalayer base already bakes both (E1-05), confirmed live,
                # 2026-09-13 — a `datalayer-sandbox` COPY would have landed
                # on top of one already there, and a wheelhouse `COPY`
                # merges into the base's own directory rather than
                # replacing it, so copying this package's bundled
                # wheelhouse again would only duplicate what `uv pip sync`
                # can already reach at `WHEELHOUSE_IMAGE_PATH`. Only the
                # lock is genuinely per-build.
                image = image.add_local_file(str(lock_file), _LOCK_PATH)
                if is_conda_lock(request.lock_text):
                    # A conda source (E3-02): `micromamba install --file`
                    # reads the `@EXPLICIT` lock without re-solving, and the
                    # pip layer the solve resolved — the user's pip
                    # requirements and the protected pins over them — comes
                    # from the lock's own `# datalayer-pip:` header, so the
                    # kernel stack (E1-04) and everything the solve installed is
                    # present the same as for a pip source. micromamba is
                    # installed first: the approved base bakes uv but not it.
                    image = image.run_commands(micromamba_bootstrap_command())
                    image = image.run_commands(micromamba_install_command(_LOCK_PATH))
                    pip_requirements = conda_lock_pip_requirements(request.lock_text)
                    if pip_requirements:
                        requirements = " ".join(shlex.quote(req) for req in pip_requirements)
                        image = image.run_commands(
                            "pip install --no-cache-dir "
                            f"--find-links {WHEELHOUSE_IMAGE_PATH} {requirements}"
                        )
                else:
                    # `uv` is not installed here: the approved base already
                    # bakes it (E1-05, `resolve.py`'s own `bootstrap_uv`
                    # docstring — "an approved Datalayer base already has it
                    # baked in"). Reinstalling it added an extra un-hashed
                    # network fetch outside the resolved lock for no reason
                    # (found in review).
                    #
                    # Packages install as root, the same reason the
                    # Datalayer and E2B builders give: a user install lands
                    # under the content directory's own home, which the
                    # runtime mounts over.
                    image = image.run_commands(_pip_lock_command(authored=authored is not None))
                image = image.dockerfile_commands([f"USER 1000:100\nWORKDIR {_CONTENT_DIR}"])
                for command in files_step(request.environment, variant=self.variant):
                    image = image.run_commands(command)
                for command in spec.commands.post_install:
                    image = image.run_commands(command)
                # The doctor is not run at build time here, unlike the
                # Datalayer and E2B builders — found live, 2026-09-13, with
                # a real, 310-package lock: a Daytona build-time `RUN` step
                # does not execute as the final image's own PID 1, only
                # inside its own ephemeral exec, whose PID 1 was
                # consistently `python3` (Daytona's own build agent) across
                # two separate builds. That agent does not reap orphans,
                # so `doctor`'s own `init` row — "PID 1 reaps orphans" —
                # fails deterministically on any build with enough steps to
                # leave one behind, for a reason that says nothing about
                # the snapshot's real, running identity. Every other row
                # (uid, gid, workdir, the kernel stack, …) passed correctly
                # in that same build; only `init` cannot be answered here.
                # A real answer needs a launched sandbox, which is a smoke
                # test's job (E1-14), not this builder's.
                image = image.workdir(_CONTENT_DIR)

                logged: list[str] = []

                def on_logs(line: str) -> None:
                    logged.append(line)
                    self._log(line)

                resources = self._resources(sdk, request.size_class, spec)
                # A snapshot is region-scoped, and the region that scopes it is
                # Daytona's, not this platform's. `validate` already refuses
                # more than one, so the first is the only one.
                declared = list(request.environment.spec.compatibility.regions)
                region = declared[0] if declared else None
                try:
                    snapshot = client.snapshot.create(
                        sdk.CreateSnapshotParams(
                            name=name,
                            image=image,
                            resources=resources,
                            entrypoint=_CONTRACT_ENTRYPOINT,
                            # Only a region Daytona knows. `request.region` is
                            # *Datalayer's* — `r1` — and sending it answered
                            # "Region not found" on the first real Daytona
                            # build (2026-09-17). The owner names a Daytona
                            # region in `compatibility.regions`; with none,
                            # the field is left out and the account's own
                            # default decides, which is what every snapshot
                            # in the owner's account already has.
                            **({"region_id": region} if region else {}),
                        ),
                        on_logs=on_logs,
                        timeout=self.max_build_seconds,
                    )
                except Exception as error:
                    raise EnvironmentsError(
                        BUILD_FAILED,
                        f"The Daytona build failed: {error}",
                        detail={"variant": self.variant, "name": name, "log": logged[-20:]},
                    ) from error
        finally:
            # No standing credential is left in an account Datalayer does
            # not control, whether the build above succeeded or not (D-18).
            self._unregister_base_pull(client, registry_id)

        return ArtifactReference(
            variant=self.variant,
            immutable_reference=snapshot.id,
            provider_artifact_id=snapshot.id,
            region=request.region,
            size_class=request.size_class,
            # The name kept for people and dashboards; never launched from —
            # a snapshot id cannot move, but a name can be reused once its
            # snapshot is deleted (E0-04).
            mutable_alias=snapshot.name,
            provider_account=provider_account(self.variant, self._provider_secrets()) or None,
            contract_version=spec.contract or SANDBOX_CONTRACT_V1.version,
        )

    @staticmethod
    def _starting_image(sdk: Any, request: BuildRequest, authored: Any, scratch: Path) -> Any:
        """The image the chain starts from: the base, or the author's Dockerfile on it.

        A Dockerfile source (E3-03) starts from the author's own Dockerfile,
        its base pinned to the digest the resolver chose — which is also what
        the build's registry entry lets Daytona pull. The contract's steps are
        chained after it either way.
        """
        if authored is None:
            return sdk.Image.base(request.resolved_base)
        dockerfile = scratch / "Dockerfile"
        dockerfile.write_text(
            pin_dockerfile_base(authored.content, request.resolved_base), encoding="utf-8"
        )
        return sdk.Image.from_dockerfile(str(dockerfile))

    @staticmethod
    def _accelerator_findings(spec: EnvironmentSpec) -> list[CapabilityFinding]:
        """A GPU Daytona does not offer, refused before any build (E2-17)."""
        accelerator = spec.resources.accelerator
        if accelerator == "none" or daytona_gpu(accelerator.type):
            return []
        return [
            CapabilityFinding(
                code="DL_ENV_CAPABILITY_UNSUPPORTED",
                message=(
                    f"Daytona has no GPU called `{accelerator.type}`; it offers "
                    + ", ".join(DAYTONA_GPUS)
                ),
                field="spec.resources.accelerator.type",
            )
        ]

    def _resources(self, sdk: Any, size_class: str, spec: EnvironmentSpec) -> Any:
        """What the snapshot bakes in: a class's CPU, or the spec's GPU (§11.3 item 4, E2-17)."""
        accelerator = spec.resources.accelerator
        if size_class not in GPU_SIZE_CLASSES or accelerator == "none":
            shape = _CPU_RESOURCES.get(size_class, _CPU_RESOURCES["small"])
            return sdk.Resources(cpu=shape["cpu"], memory=shape["memory"], disk=shape["disk"])
        # D-4 gives the GPU classes no CPU or memory of their own (E4-11 has
        # them, for Datalayer's nodes): the GPU is what the class is for, and
        # Daytona sizes the rest of the machine for it unless the spec hints.
        hints = spec.resources.hints
        return sdk.Resources(
            cpu=round(hints.cpu) if hints.cpu else None,
            memory=round(hints.memory_gi) if hints.memory_gi else None,
            disk=round(hints.disk_gi) if hints.disk_gi else _GPU_DISK_GI,
            gpu=accelerator.count,
            gpu_type=sdk.GpuType(daytona_gpu(accelerator.type)),
        )

    def _register_base_pull(self, client: Any, resolved_base: str) -> str | None:
        """A private registry entry for this build's base pull (D-17, D-18).

        Returns the entry's **id**, not the name given it: `delete_registry`
        (and `get_registry`) take the id (found live, 2026-09-13 — a first
        attempt deleted by name and got `NotFoundException`, the registry
        left behind until a second call read it back by name to find its
        id). `None` when the credential carries no registry login: the base
        is then whatever the ambient Daytona organization can already
        reach, the same fallback the client itself takes with no
        `DAYTONA_API_KEY`.
        """
        registry_host = str(getattr(self._credential, "registry", "") or "")
        username = str(getattr(self._credential, "username", "") or "")
        password = str(getattr(self._credential, "password", "") or "")
        if not (registry_host and username and password):
            return None
        registry_sdk = self._registry_sdk()
        # Unique per build: the entry is organization-wide, so two builds at
        # once must not collide, and a name in use is refused (E0-04).
        name = f"dl-build-{uuid.uuid4().hex[:20]}"
        # `Daytona` exposes `.snapshot`, `.secret` and `.volume` as public
        # services, all built from this one client's `_api_client` — but not
        # a registry service, so the same `_api_client` is reached for
        # directly, confirmed live to work the same way (2026-09-13).
        api = registry_sdk.DockerRegistryApi(client._api_client)
        try:
            created = api.create_registry(
                registry_sdk.CreateDockerRegistry(
                    name=name, url=registry_host, username=username, password=password
                )
            )
        except Exception as error:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"Daytona could not be given a pull credential for the base: {error}",
                detail={"variant": self.variant},
            ) from error
        return str(created.id)

    def _unregister_base_pull(self, client: Any, registry_id: str | None) -> None:
        if not registry_id:
            return
        registry_sdk = self._registry_sdk()
        api = registry_sdk.DockerRegistryApi(client._api_client)
        try:
            api.delete_registry(registry_id)
        except Exception as error:
            # Best-effort: the build's own result matters more than this
            # cleanup, and a left-behind entry is the owner's organization's
            # to remove by hand — logged, never raised over a build result.
            self._log(f"Could not delete the Daytona registry entry {registry_id}: {error}")

    # -- Reading the registry ---------------------------------------------------

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        sdk = self._daytona_sdk()
        client = self._client(sdk)
        try:
            snapshot = client.snapshot.get(artifact.provider_artifact_id)
        except sdk.DaytonaNotFoundError as error:
            raise EnvironmentsError(
                ARTIFACT_MISSING,
                f"`{artifact.immutable_reference}` is not a Daytona snapshot",
                detail={"variant": self.variant, "reference": artifact.immutable_reference},
            ) from error
        except Exception as error:
            raise self._provider_error("read the snapshot", error) from error
        created = getattr(snapshot, "created_at", None)
        state = getattr(snapshot, "state", None)
        return ArtifactMetadata(
            reference=artifact,
            created_at=created.isoformat() if hasattr(created, "isoformat") else None,
            provider_state=str(getattr(state, "value", state) or "") or None,
            labels={"name": str(getattr(snapshot, "name", ""))},
        )

    def exists(self, artifact: ArtifactReference) -> bool:
        sdk = self._daytona_sdk()
        client = self._client(sdk)
        try:
            client.snapshot.get(artifact.provider_artifact_id)
        except sdk.DaytonaNotFoundError:
            return False
        except Exception as error:
            raise self._provider_error("ask whether the snapshot exists", error) from error
        return True

    def delete(self, artifact: ArtifactReference) -> None:
        """Remove the snapshot, by id; one already gone is removed (E2-18).

        Deleting what is gone is a success because the collector deletes
        first and marks second (E1-17): a sweep that died between the two
        deletes again tomorrow, and must not be refused for having worked.

        **By id, never by name.** The SDK takes either, and a name is reused
        once its snapshot is deleted (E0-04) — deleting by name could remove
        a later build's snapshot that inherited it.
        """
        sdk = self._daytona_sdk()
        client = self._client(sdk)
        snapshot = artifact.provider_artifact_id or artifact.immutable_reference
        try:
            client.snapshot.delete(snapshot)
        except sdk.DaytonaNotFoundError:
            self._log(f"The Daytona snapshot {snapshot} was already gone")
            return
        except Exception as error:
            raise self._provider_error("delete the snapshot", error) from error
        self._log(f"Deleted the Daytona snapshot {snapshot}")

    def smoke_test(
        self,
        artifact: ArtifactReference,
        *,
        environment: Any = None,
        lock_text: str | None = None,
        secret_values: Sequence[str] = (),
    ) -> ValidationResult:
        """Launch the snapshot and run Appendix B's core tier in it (E2-04).

        This box's own `Done when` asks for exactly this — "a sandbox launched
        from its id passes the core tier" — and until 2026-09-17 it refused
        through `ManagedBuilder`, so **no Daytona build could reach
        `succeeded`**: the workflow calls this step, the snapshot was built and
        live at the provider, and the build was recorded failed.

        **Launched by id, never by name** (E0-04): a Daytona sandbox record
        keeps the snapshot's *name*, and a name is republished, so only the id
        says which artifact actually ran.

        **Restarted by stopping and starting the sandbox**, not the kernel
        (E0-04 again): Daytona's own daemon is PID 1 here, so there is no
        kernel to restart, and check 8 means the thing that survives a real
        restart.

        The sandbox is deleted whether the tier passed or not — a smoke test
        that leaves a sandbox running bills the owner for a check.
        """
        if environment is None:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "A Daytona smoke test needs the version's spec: the core tier "
                "asks for the Python version it declared and the packages its "
                "lock pinned, and an artifact carries neither",
                detail={"variant": self.variant},
            )
        from ..conformance import expected_packages, run_accelerator_check, run_core_tier

        snapshot = artifact.provider_artifact_id or artifact.immutable_reference
        sandbox = self._smoke_test_sandbox(snapshot)
        self._log(f"Launching {snapshot} to smoke-test it")
        try:
            sandbox.start()
            result = run_core_tier(
                sandbox,
                python_version=environment.spec.language.version,
                expected_packages=expected_packages(environment, lock_text or ""),
                secret_values=tuple(secret_values),
                restart=lambda: self._restart(sandbox),
            )
            accelerator = environment.spec.resources.accelerator
            if accelerator != "none":
                # A GPU version is only the version its spec describes when
                # its GPUs are visible and its CUDA is the spec's (check 11,
                # E2-17): the core tier alone passes on a machine with none.
                result.checks.append(
                    run_accelerator_check(sandbox, cuda=accelerator.cuda, count=accelerator.count)
                )
            return result
        except EnvironmentsError:
            raise
        except Exception as error:
            raise self._provider_error("smoke-test the snapshot", error) from error
        finally:
            try:
                sandbox.stop()
            except Exception as error:
                self._log(f"The smoke-test sandbox could not be stopped: {error}")

    def _smoke_test_sandbox(self, snapshot: str) -> Any:
        """A sandbox of this build's own snapshot, deleted when it stops."""
        from ...daytona_sandbox import DaytonaSandbox

        secrets = self._provider_secrets()
        return DaytonaSandbox(
            api_key=secrets.get("DAYTONA_API_KEY"),
            organization_id=secrets.get("DAYTONA_ORGANIZATION_ID"),
            snapshot=snapshot,
            delete_on_stop=True,
        )

    @staticmethod
    def _restart(sandbox: Any) -> None:
        """Check 8's restart, as Daytona can do it.

        Its daemon is PID 1, so there is no kernel to restart: the sandbox
        itself is stopped and started, which is the stronger version of the
        same question.
        """
        sandbox.stop()
        sandbox.start()

    def _provider_error(self, what: str, error: BaseException) -> EnvironmentsError:
        return EnvironmentsError(
            PROVIDER_ERROR,
            f"Daytona could not {what}: {error}",
            detail={"variant": self.variant},
        )
