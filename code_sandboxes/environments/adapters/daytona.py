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

**GPU classes are not built here.** `gpu = True` on this builder is a true
capability (Daytona's own hardware runs one, D-20), and `validate` leaves a
GPU size class buildable rather than refusing it — the constraint table of
section 6 has nothing against it, and an existing test
(`test_a_gpu_spec_is_buildable_on_modal_and_daytona`) already pins that
answer. What actually stops a GPU build today is upstream: the CUDA base
channel is E2-17's to publish, and `bases.py` has no digest to resolve
`python-cuda` to yet, so a `BuildRequest` for one cannot be constructed in
practice. `build()` itself still guards it explicitly, refusing plainly
rather than baking an unsourced guess at a GPU type and count into a
snapshot, in case that ever changes before the real numbers do (section 11.3
item 7).

@module code_sandboxes.environments.adapters.daytona
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..accounts import provider_account
from ..builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
)
from ..contract import SANDBOX_CONTRACT_V1
from ..errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    EnvironmentsError,
)
from ..files import files_step
from ..resolve import WHEELHOUSE_IMAGE_PATH, apt_pins_in
from ..spec import GPU_SIZE_CLASSES, Environment
from .managed import ManagedBuilder

__all__ = ["Builder"]

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
    #: Daytona runs GPUs, on its own hardware and the owner's account (E2-17).
    #: This builder does not build one yet: see `_own_findings`.
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
        # A GPU class is left buildable here on purpose (`gpu = True`,
        # D-20): the CUDA base E2-17 has not published yet, so a GPU
        # `BuildRequest` cannot reach `build()` in practice — `bases.py`'s
        # own resolver has no digest to resolve `python-cuda` to, and
        # refuses first. `build()` itself still guards it explicitly (see
        # its own docstring), so a spec that somehow got a `resolved_base`
        # anyway is refused plainly rather than baking an unsourced guess
        # at a GPU type and count into a snapshot.
        #
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
        if request.size_class in GPU_SIZE_CLASSES:
            # `validate` leaves a GPU class buildable (`gpu = True`, D-20):
            # `bases.py` has no CUDA digest to resolve yet, so this cannot
            # be reached in practice — refused plainly here rather than
            # baking an unsourced guess at a GPU type and count into a
            # snapshot (E2-17 is what will give this real numbers).
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                f"Daytona runs `{request.size_class}` on its own GPUs, but the CUDA base "
                "and the GPU resource shape this needs are E2-17's, not built yet",
                detail={"variant": self.variant, "missing": "E2-17"},
            )
        sdk = self._daytona_sdk()
        client = self._client(sdk)
        name = f"dl-{request.environment.metadata.name}-v{request.version}-{request.build_uid}"

        registry_id = self._register_base_pull(client, request.resolved_base)
        try:
            with tempfile.TemporaryDirectory(prefix="dl-daytona-build-") as scratch:
                lock_file = Path(scratch) / "lock.txt"
                lock_file.write_text(request.lock_text, encoding="utf-8")

                image = sdk.Image.base(request.resolved_base)
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
                image = (
                    image.add_local_file(str(lock_file), _LOCK_PATH)
                    # `uv` is not installed here: the approved base already
                    # bakes it (E1-05, `resolve.py`'s own `bootstrap_uv`
                    # docstring — "an approved Datalayer base already has it
                    # baked in"), and this phase's `build_sources` is
                    # `("packages",)` only, so every build starts from that
                    # base. Reinstalling it added an extra un-hashed network
                    # fetch outside the resolved lock for no reason (found in
                    # review) — matching the Datalayer builder's own
                    # `dockerfile()`, which installs `uv` only for the
                    # `image` source, not implemented for this variant yet.
                    #
                    # Packages install as root, the same reason the
                    # Datalayer and E2B builders give: a user install lands
                    # under the content directory's own home, which the
                    # runtime mounts over.
                    .run_commands(
                        "uv pip sync --system --require-hashes "
                        f"--find-links {WHEELHOUSE_IMAGE_PATH} {_LOCK_PATH}"
                    )
                )
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

                resources = self._resources(sdk, request.size_class)
                try:
                    snapshot = client.snapshot.create(
                        sdk.CreateSnapshotParams(
                            name=name,
                            image=image,
                            resources=resources,
                            entrypoint=_CONTRACT_ENTRYPOINT,
                            region_id=request.region,
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

    def _resources(self, sdk: Any, size_class: str) -> Any:
        """The CPU resources a size class bakes into the snapshot (§11.3 item 4)."""
        shape = _CPU_RESOURCES.get(size_class, _CPU_RESOURCES["small"])
        return sdk.Resources(cpu=shape["cpu"], memory=shape["memory"], disk=shape["disk"])

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

    def _provider_error(self, what: str, error: BaseException) -> EnvironmentsError:
        return EnvironmentsError(
            PROVIDER_ERROR,
            f"Daytona could not {what}: {error}",
            detail={"variant": self.variant},
        )
