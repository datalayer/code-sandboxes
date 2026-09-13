# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Datalayer variant's Environment builder (PLAN_ENV.md, E1-07).

One version, one lock, one image: this builder turns a :class:`BuildRequest`
into an OCI image in the owner's ECR repository, and hands back the digest —
never the tag, which is only there so a person can find the build again.

What it is careful about, and why:

- **The Dockerfile is generated from the lock**, never from the spec's loose
  dependency list. ``uv pip sync --require-hashes`` installs exactly what was
  resolved (§5, D-9), and the apt versions come from the lock's own record of
  them, so a build a month later installs what the first one did.
- **``env`` is set before anything installs**, because a source build reads it
  — a package that compiles against a library found through ``CFLAGS`` behaves
  differently without it.
- **``postInstall`` runs as uid 1000 with no network**, so a command that would
  fetch something has to fail at build time rather than produce an artifact
  whose contents depend on when it ran.
- **The push is by digest, with attestations.** ``--attest type=sbom`` and
  ``--attest type=provenance,mode=max`` (D-11), under the operability tag
  ``v<n>-<build_uid>`` (D-10) so a retried build of the same version cannot
  collide with the attempt before it. The reference kept is
  ``containerimage.digest`` from ``buildctl``'s metadata.
- **It holds no AWS credential of its own.** The registry token is minted per
  build by the workflow (D-17) and handed here; ``buildkitd`` never sees an AWS
  key.

What it deliberately does not do: the smoke test. For this variant, the trial
sandbox is launched through Runtimes' internal route and checked there
(E1-14, E0-09), because only the platform can start a Datalayer sandbox.

@module code_sandboxes.environments.adapters.datalayer
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from ..contract import SANDBOX_CONTRACT_V1
from ..errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    SPEC_INVALID,
    EnvironmentsError,
)
from ..files import files_step
from ..resolve import WHEELHOUSE_IMAGE_PATH, apt_pins_in, locked_versions
from ..spec import Environment

__all__ = ["ECR_ENVIRONMENT_PREFIX", "Builder", "owner_repository"]

#: Where an owner's environments live in ECR (D-10).
ECR_ENVIRONMENT_PREFIX = "environments/u/"

#: The size classes a Datalayer sandbox runs (D-4). The GPU ones stay in the
#: specification and run on Modal or Daytona until E4-11 (D-20).
CPU_SIZE_CLASSES = ("small", "medium", "large")

#: What a build may take, per build (E1-06's limits are the pool's).
DEFAULT_MAX_BUILD_SECONDS = 45 * 60

#: The largest image this variant pushes: a pull of one is charged as egress
#: and waited for at every cold start.
DEFAULT_MAX_ARTIFACT_BYTES = 20 * 1024**3


def owner_repository(owner_uid: str, environment_name: str) -> str:
    """The owner's repository for one environment: ``environments/u/<owner>/<name>``."""
    if not owner_uid or not environment_name:
        raise ValueError("an owner and an environment name make the repository")
    return f"{ECR_ENVIRONMENT_PREFIX}{owner_uid}/{environment_name}"


class Builder:
    """The Datalayer variant: BuildKit into the owner's ECR repository.

    Parameters
    ----------
    log
        Where the build's output goes, line by line. The workflow's log writer,
        which redacts the credential before anything is stored (D-15).
    credential
        The registry credential for this build: ``registry``, ``username`` and
        ``password`` (D-17). Minted per build, held by the worker, never in a
        step result.
    buildctl
        The ``buildctl`` to run. Found on the PATH by default; an empty string
        means there is none, which is what a deployment without the build pool
        looks like (E1-06).
    address
        ``buildkitd``'s address, when it is not the default socket.
    region
        The registry's region, for the ECR API calls.
    ecr
        The ECR client. A boto3 ``ecr`` client by default, made on first use so
        importing this module needs no credentials.
    run
        How a subprocess is run, so a test can watch the argv without a daemon.
    """

    variant = "datalayer"

    def __init__(
        self,
        *,
        log: Callable[[str], None] | None = None,
        credential: Any = None,
        buildctl: str | None = None,
        address: str | None = None,
        region: str | None = None,
        ecr: Any = None,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        max_build_seconds: int = DEFAULT_MAX_BUILD_SECONDS,
    ) -> None:
        self._log = log or (lambda _line: None)
        self._credential = credential
        self._buildctl = (shutil.which("buildctl") or "") if buildctl is None else buildctl
        self._address = address or os.environ.get("DATALAYER_BUILDKIT_ADDR", "").strip()
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._ecr = ecr
        self._run = run or subprocess.run
        self._max_build_seconds = max_build_seconds

    # -- What this variant can do ---------------------------------------------

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            variant=self.variant,
            # `dependencyFile` resolves the same way `packages` does (E3-01);
            # `image` resolves from an imported reference instead of an
            # approved base (E3-04). Neither changes how this builder itself
            # builds, once the resolver has done its part.
            build_sources=("packages", "dependencyFile", "image"),
            package_managers=("uv", "pip"),
            # Nothing is forbidden by the builder itself: BuildKit is the
            # reference implementation of a Dockerfile, and what the contract
            # forbids `check_dockerfile` already refuses.
            forbidden_instructions=(),
            # No Datalayer plane has a GPU node (D-20): a GPU class runs on
            # Modal or Daytona until E4-11.
            supports_gpu=False,
            regions=("r1",),
            max_build_seconds=self._max_build_seconds,
            max_artifact_bytes=DEFAULT_MAX_ARTIFACT_BYTES,
        )

    def validate(self, environment: Environment, lock_text: str | None = None) -> CapabilityReport:
        """Whether this variant can build this spec, before anything is queued."""
        findings: list[CapabilityFinding] = []
        source = environment.spec.build.source
        if source not in ("packages", "dependencyFile", "image"):
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=f"the Datalayer builder does not build `{source}` yet",
                    field="spec.build.source",
                )
            )
        if environment.spec.packages.python.manager == "conda":
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message="conda environments are resolved by their own solver, "
                    "which is not built yet",
                    field="spec.packages.python.manager",
                )
            )
        size_class = environment.spec.resources.size_class
        if size_class not in CPU_SIZE_CLASSES:
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message=f"`{size_class}` has no placement on a Datalayer node: "
                    "build for modal or daytona, which run a GPU class on their own "
                    "hardware (D-20)",
                    field="spec.resources.sizeClass",
                )
            )
        if environment.spec.platform.architecture != "linux/amd64":
            findings.append(
                CapabilityFinding(
                    code=CAPABILITY_UNSUPPORTED.code,
                    message="the sandbox contract is `linux/amd64`",
                    field="spec.platform.architecture",
                )
            )
        if lock_text is not None and not locked_versions(lock_text):
            findings.append(
                CapabilityFinding(
                    code=SPEC_INVALID.code,
                    message="the lock pins no package: the build installs from the lock, "
                    "so an empty one builds nothing",
                )
            )
        return CapabilityReport(variant=self.variant, findings=findings)

    # -- The build ------------------------------------------------------------

    def dockerfile(self, request: BuildRequest) -> str:
        """The build, as a Dockerfile, from the lock.

        Ordered so each step can only see what it should: the environment
        first, then the system packages at the versions the lock names, then
        the locked Python set, then the files, then the owner's commands with
        no network.
        """
        spec = request.environment.spec
        apt = apt_pins_in(request.lock_text)
        lines = [
            "# syntax=docker/dockerfile:1.7",
            f"# Generated by Datalayer for {request.environment.metadata.name} "
            f"v{request.version}, build {request.build_uid}.",
            f"# Lock: {request.lock_digest}",
            f"FROM {request.resolved_base}",
            f'LABEL io.datalayer.environment="{request.environment.metadata.name}" \\',
            f'      io.datalayer.environment.uid="{request.environment_uid}" \\',
            f'      io.datalayer.version="{request.version}" \\',
            f'      io.datalayer.build="{request.build_uid}" \\',
            f'      io.datalayer.lock="{request.lock_digest}" \\',
            f'      io.datalayer.contract="{spec.contract}" \\',
            f'      io.datalayer.size-class="{request.size_class}"',
        ]
        # `env` before anything installs: a source build reads it.
        for name in sorted(spec.env):
            lines.append(f"ENV {name}={shlex.quote(spec.env[name])}")
        if apt:
            # Pinned, and from the lock: `apt-get install <name>` a month later
            # is a different artifact.
            pinned = " ".join(f"{name}={apt[name]}" for name in sorted(apt))
            lines.extend(
                [
                    "USER root",
                    "RUN --mount=type=cache,target=/var/cache/apt,sharing=locked "
                    "apt-get update -qq \\\n"
                    f"    && apt-get install -y --no-install-recommends {pinned} \\\n"
                    "    && rm -rf /var/lib/apt/lists/*",
                ]
            )
        lines.extend(
            [
                "USER root",
                "COPY lock.txt /opt/datalayer/lock.txt",
                # `sync` and not `install`: the artifact holds the lock's set,
                # and `--require-hashes` means every byte was the resolved one.
                # `--find-links` for what no index has — a protected pin's
                # own wheel, the fork's local version above all (E1-04).
                "RUN --mount=type=cache,target=/root/.cache/uv "
                f"uv pip sync --system --require-hashes --find-links {WHEELHOUSE_IMAGE_PATH} "
                "/opt/datalayer/lock.txt",
            ]
        )
        if request.build_secret_ids:
            # Mounted for the step that needs it and nowhere else, so nothing
            # reaches a layer (§4.1, D-11).
            for secret_id in request.build_secret_ids:
                lines.append(
                    f"RUN --mount=type=secret,id={secret_id} test -s /run/secrets/{secret_id}"
                )
        baked = files_step(request.environment, variant=self.variant)
        if baked:
            lines.append("USER 1000:100")
            lines.append("WORKDIR /home/datalayer/content")
            for command in baked:
                lines.append(f"RUN {command}")
        if spec.commands.post_install:
            lines.append("USER 1000:100")
            lines.append("WORKDIR /home/datalayer/content")
            for command in spec.commands.post_install:
                # No network: a command that fetches something makes an
                # artifact whose contents depend on the day it was built.
                lines.append(f"RUN --network=none {command}")
        lines.extend(
            [
                "USER 1000:100",
                "WORKDIR /home/datalayer/content",
                # The contract's own check, in the image, at build time.
                "RUN /opt/datalayer/bin/datalayer-sandbox doctor --json > /tmp/doctor.json",
            ]
        )
        return "\n".join(lines) + "\n"

    def build(self, request: BuildRequest) -> ArtifactReference:
        """Build and push the version, and keep the digest.

        The repository is created if it is missing, with immutable tags, so a
        first build of a new environment needs nothing prepared.
        """
        if not self._buildctl:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No `buildctl` to build with: the build pool is not deployed here",
                detail={"missing": "buildctl", "variant": self.variant, "item": "E1-06"},
            )
        if "@sha256:" not in request.resolved_base:
            raise EnvironmentsError(
                SPEC_INVALID,
                "The base is resolved to a digest before a build starts",
                detail={"base": request.resolved_base},
            )
        repository = owner_repository(request.owner_uid, request.environment.metadata.name)
        registry = self._registry_host()
        self._ensure_repository(repository)
        self._ensure_cache_repository(f"environments/cache/u/{request.owner_uid}")
        tag = f"v{request.version}-{request.build_uid}"
        reference = f"{registry}/{repository}:{tag}"
        with tempfile.TemporaryDirectory(prefix="dl-build-") as directory:
            root = Path(directory)
            (root / "Dockerfile").write_text(self.dockerfile(request), encoding="utf-8")
            (root / "lock.txt").write_text(request.lock_text, encoding="utf-8")
            metadata = root / "metadata.json"
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
                f"type=image,name={reference},push=true,oci-mediatypes=true",
                # The supply chain is part of the artifact (D-11). `--attest`
                # is `docker buildx`'s flag, not `buildctl`'s — a bare
                # `buildctl` (which never goes through buildx) takes the same
                # request as a dockerfile.v0 frontend option: found live on
                # 2026-09-12, the first real build this adapter ever drove,
                # where `--attest type=sbom` failed before anything else did
                # ("flag provided but not defined: -attest").
                "--opt",
                "attest:sbom=",
                "--opt",
                "attest:provenance=mode=max",
                "--metadata-file",
                str(metadata),
                *self._cache_options(request),
            ]
            self._log(f"Building {reference} from {request.resolved_base}")
            finished = self._invoke(command, timeout=self._max_build_seconds)
            if finished.returncode != 0:
                raise EnvironmentsError(
                    BUILD_FAILED,
                    "The build failed; its log says where",
                    detail={
                        "variant": self.variant,
                        "reference": reference,
                        "exit": finished.returncode,
                    },
                )
            digest = self._digest_of(metadata)
        return ArtifactReference(
            variant=self.variant,
            immutable_reference=f"{registry}/{repository}@{digest}",
            provider_artifact_id=digest,
            region=request.region,
            size_class=request.size_class,
            mutable_alias=reference,
            provider_account=registry.split(".", 1)[0] or None,
            contract_version=request.environment.spec.contract or SANDBOX_CONTRACT_V1.version,
        )

    def smoke_test(self, artifact: ArtifactReference) -> ValidationResult:
        """Not this builder's: only the platform starts a Datalayer sandbox.

        The workflow launches a trial through Runtimes' internal route and runs
        the doctor and the core tier in it (E1-14, E0-09). A builder that
        answered here would be a second way of starting a runtime.
        """
        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            "A Datalayer artifact is smoke-tested by launching it through Runtimes, "
            "not by its builder",
            detail={"variant": self.variant, "seam": "smoke_test", "item": "E1-14"},
        )

    # -- Reading the registry -------------------------------------------------

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        """The image as the registry has it: its size and when it was pushed."""
        repository, digest = self._parts(artifact)
        images = self._describe(repository, digest)
        if not images:
            raise EnvironmentsError(
                ARTIFACT_MISSING,
                f"`{artifact.immutable_reference}` is not in the registry",
                detail={"variant": self.variant, "reference": artifact.immutable_reference},
            )
        image = images[0]
        pushed = image.get("imagePushedAt")
        return ArtifactMetadata(
            reference=artifact,
            size_bytes=int(image["imageSizeInBytes"]) if image.get("imageSizeInBytes") else None,
            created_at=pushed.isoformat() if hasattr(pushed, "isoformat") else pushed,
            provider_state=str(image.get("imageManifestMediaType") or "") or None,
            labels={"tags": ",".join(image.get("imageTags") or [])},
        )

    def resolve(self, version_ref: str) -> ArtifactReference:
        """The artifact a tag points at, as a digest.

        ``version_ref`` is ``<repository>:<tag>``: what a person has, and the
        one thing a launch may never use, because a tag can be moved.
        """
        repository, _, tag = version_ref.rpartition(":")
        if not repository or not tag:
            raise EnvironmentsError(
                SPEC_INVALID,
                f"`{version_ref}` is not `<repository>:<tag>`",
                detail={"reference": version_ref},
            )
        registry = self._registry_host()
        repository = repository.removeprefix(f"{registry}/")
        images = self._describe(repository, tag=tag)
        if not images:
            raise EnvironmentsError(
                ARTIFACT_MISSING,
                f"`{version_ref}` is not in the registry",
                detail={"variant": self.variant, "reference": version_ref},
            )
        digest = str(images[0]["imageDigest"])
        return ArtifactReference(
            variant=self.variant,
            immutable_reference=f"{registry}/{repository}@{digest}",
            provider_artifact_id=digest,
            mutable_alias=f"{registry}/{repository}:{tag}",
            provider_account=registry.split(".", 1)[0] or None,
            contract_version=SANDBOX_CONTRACT_V1.version,
        )

    def exists(self, artifact: ArtifactReference) -> bool:
        repository, digest = self._parts(artifact)
        return bool(self._describe(repository, digest))

    def delete(self, artifact: ArtifactReference) -> None:
        """Delete the image. Deleting one that is gone is a success."""
        repository, digest = self._parts(artifact)
        try:
            self._client().batch_delete_image(
                repositoryName=repository, imageIds=[{"imageDigest": digest}]
            )
        except Exception as error:
            if self._is_missing(error):
                return
            raise self._provider_error("delete the image", error) from error

    # -- The registry, underneath ---------------------------------------------

    def _client(self) -> Any:
        if self._ecr is None:
            try:
                import boto3
            except ImportError as error:
                raise EnvironmentsError(
                    CAPABILITY_UNSUPPORTED,
                    "No AWS SDK to reach the registry with: install "
                    "`code-sandboxes[environments-builder]`",
                    detail={"missing": "boto3", "variant": self.variant},
                ) from error
            self._ecr = boto3.client("ecr", region_name=self._region)
        return self._ecr

    def _registry_host(self) -> str:
        registry = str(getattr(self._credential, "registry", "") or "")
        if not registry:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No registry credential for this build: one is minted per build (D-17)",
                detail={"missing": "credential", "variant": self.variant, "item": "E1-06"},
            )
        return registry

    def _parts(self, artifact: ArtifactReference) -> tuple[str, str]:
        """The repository and digest of an artifact this builder made."""
        reference, _, digest = artifact.immutable_reference.partition("@")
        registry = self._registry_host()
        return reference.removeprefix(f"{registry}/"), digest

    def _describe(
        self, repository: str, digest: str | None = None, tag: str | None = None
    ) -> list[Mapping[str, Any]]:
        image_id: dict[str, str | None] = {"imageDigest": digest} if digest else {"imageTag": tag}
        try:
            answer = self._client().describe_images(repositoryName=repository, imageIds=[image_id])
        except Exception as error:
            if self._is_missing(error):
                return []
            raise self._provider_error("read the registry", error) from error
        return list(answer.get("imageDetails") or [])

    def _ensure_repository(self, repository: str) -> None:
        """Create the owner's repository if it is missing, with immutable tags (D-10)."""
        self._ensure_ecr_repository(repository, mutability="IMMUTABLE")

    def _ensure_cache_repository(self, repository: str) -> None:
        """Create the owner's cache repository if it is missing, with mutable tags.

        Every repository is immutable except this one: a build re-exports its
        cache under the same tag every time, which an immutable one refuses
        (E0-05). Terraform never creates it — its own comment says so, "created
        by the builder" — and nothing here ever had either, found live on
        2026-09-12: the first real build got all the way to a real push, then
        failed exporting its cache with a plain 404, the repository never
        having existed to push to.
        """
        self._ensure_ecr_repository(repository, mutability="MUTABLE")

    def _ensure_ecr_repository(self, repository: str, *, mutability: str) -> None:
        client = self._client()
        try:
            client.describe_repositories(repositoryNames=[repository])
            return
        except Exception as error:
            if not self._is_missing(error):
                raise self._provider_error("read the repository", error) from error
        self._log(f"Creating {repository}")
        try:
            client.create_repository(
                repositoryName=repository,
                imageTagMutability=mutability,
                imageScanningConfiguration={"scanOnPush": True},
                encryptionConfiguration={"encryptionType": "KMS"},
            )
        except Exception as error:
            if self._already_exists(error):
                # Another build of the same environment, or the same owner's
                # cache, got there first.
                return
            raise self._provider_error("create the repository", error) from error

    def _cache_options(self, request: BuildRequest) -> list[str]:
        """The owner's cache, imported and exported: never another owner's (D-12)."""
        registry = self._registry_host()
        cache = f"{registry}/environments/cache/u/{request.owner_uid}:{self.variant}"
        return [
            "--import-cache",
            f"type=registry,ref={cache}",
            "--export-cache",
            f"type=registry,ref={cache},mode=max,image-manifest=true,oci-mediatypes=true",
        ]

    def _invoke(self, command: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        try:
            finished = self._run(
                list(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=self._environment(),
            )
        except subprocess.TimeoutExpired as expired:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"The build did not finish within {timeout:.0f}s",
                detail={"variant": self.variant, "timeout": timeout},
            ) from expired
        for line in (finished.stderr or "").splitlines():
            self._log(line)
        return finished

    def _environment(self) -> dict[str, str] | None:
        """What ``buildctl`` needs to push, as client-side registry auth (D-17).

        The credential is written into a docker config of this build's own, so
        nothing is added to the worker's, and ``buildkitd`` still holds no AWS
        key.
        """
        registry = str(getattr(self._credential, "registry", "") or "")
        username = str(getattr(self._credential, "username", "") or "")
        password = str(getattr(self._credential, "password", "") or "")
        if not (registry and username and password):
            return None
        directory = Path(tempfile.mkdtemp(prefix="dl-docker-"))
        import base64

        auth = base64.b64encode(f"{username}:{password}".encode()).decode()
        (directory / "config.json").write_text(
            json.dumps({"auths": {registry: {"auth": auth}}}), encoding="utf-8"
        )
        directory.chmod(0o700)
        (directory / "config.json").chmod(0o600)
        return {**os.environ, "DOCKER_CONFIG": str(directory)}

    def _digest_of(self, metadata: Path) -> str:
        """``containerimage.digest``, which is the only reference kept."""
        try:
            written = json.loads(metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EnvironmentsError(
                BUILD_FAILED,
                "The build wrote no metadata, so its digest is not known",
                detail={"variant": self.variant},
            ) from error
        digest = str(written.get("containerimage.digest") or "")
        if not digest.startswith("sha256:"):
            raise EnvironmentsError(
                BUILD_FAILED,
                "The build recorded no image digest",
                detail={"variant": self.variant, "metadata": sorted(written)},
            )
        return digest

    @staticmethod
    def _is_missing(error: BaseException) -> bool:
        name: Any = getattr(getattr(error, "response", {}), "get", lambda *_: {})("Error", {})
        code = str((name or {}).get("Code") or "")
        return code in {
            "RepositoryNotFoundException",
            "ImageNotFoundException",
            "RepositoryPolicyNotFoundException",
        } or type(error).__name__ in {"RepositoryNotFoundException", "ImageNotFoundException"}

    @staticmethod
    def _already_exists(error: BaseException) -> bool:
        name: Any = getattr(getattr(error, "response", {}), "get", lambda *_: {})("Error", {})
        return (
            str((name or {}).get("Code") or "") == "RepositoryAlreadyExistsException"
            or type(error).__name__ == "RepositoryAlreadyExistsException"
        )

    def _provider_error(self, what: str, error: BaseException) -> EnvironmentsError:
        return EnvironmentsError(
            PROVIDER_ERROR,
            f"The registry could not {what}: {error}",
            detail={"variant": self.variant},
        )
