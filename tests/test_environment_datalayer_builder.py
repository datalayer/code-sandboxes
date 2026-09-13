# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Datalayer builder: the Dockerfile from the lock, and the push by digest (E1-07).

Nothing here reaches AWS or a daemon: the ECR client is a double that records
what it was asked, and `buildctl` is a function that writes the metadata file a
real one writes. What is checked is what the build *is* — the Dockerfile
generated for the section 4.1 example, the argv that pushes it, and the
reference kept afterwards.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from code_sandboxes.environments.adapters.datalayer import Builder, owner_repository
from code_sandboxes.environments.builders import ArtifactReference, BuildRequest
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.spec import parse_environment

REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
OWNER = "01k0wner000000000000000000"
BASE = "environments/base/python-cpu@sha256:" + "bb" * 32
DIGEST = "sha256:" + "aa" * 32
LOCK = (
    "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.\n"
    "# python: 3.13\n"
    "# datalayer-apt: gdal-bin=3.8.4+dfsg-3build2\n"
    "# datalayer-apt: libgdal34=3.8.4+dfsg-3build2\n"
    "# datalayer-protected: ipykernel==7.3.0\n"
    "geopandas==1.1.1 \\\n    --hash=sha256:" + "cd" * 32 + "\n"
    "rasterio==1.4.3 \\\n    --hash=sha256:" + "ef" * 32 + "\n"
)
LOCK_DIGEST = "sha256:" + "dd" * 32


class Credential:
    """The build's registry credential, as the workflow mints one (D-17)."""

    registry = REGISTRY
    username = "AWS"
    password = "ecr-login-token-7c1f0a9b2e4d4a55b8e1d0c3"


class FakeEcr:
    """The ECR API, as much of it as this builder uses."""

    def __init__(self, *, repositories: set[str] | None = None, images=None) -> None:
        self.repositories = set(repositories or set())
        self.images = images if images is not None else {}
        self.created: list[dict] = []
        self.deleted: list[dict] = []

    def describe_repositories(self, repositoryNames):  # noqa: N803 - boto3's spelling
        missing = [name for name in repositoryNames if name not in self.repositories]
        if missing:
            raise self._missing("RepositoryNotFoundException")
        return {"repositories": [{"repositoryName": name} for name in repositoryNames]}

    def create_repository(self, **kwargs):
        self.created.append(kwargs)
        self.repositories.add(kwargs["repositoryName"])
        return {"repository": {"repositoryName": kwargs["repositoryName"]}}

    def describe_images(self, repositoryName, imageIds):  # noqa: N803 - boto3's spelling
        if repositoryName not in self.repositories:
            raise self._missing("RepositoryNotFoundException")
        wanted = imageIds[0]
        key = wanted.get("imageDigest") or wanted.get("imageTag")
        found = self.images.get((repositoryName, key))
        if not found:
            raise self._missing("ImageNotFoundException")
        return {"imageDetails": [found]}

    def batch_delete_image(self, repositoryName, imageIds):  # noqa: N803 - boto3's spelling
        self.deleted.append({"repository": repositoryName, "ids": imageIds})
        for wanted in imageIds:
            self.images.pop((repositoryName, wanted.get("imageDigest")), None)
        return {"imageIds": imageIds, "failures": []}

    @staticmethod
    def _missing(code: str) -> Exception:
        error = Exception(code)
        error.response = {"Error": {"Code": code}}
        return error


def a_request(**changes) -> BuildRequest:
    """The section 4.1 example as a build of version 3, attempt `bld-1`."""
    spec = {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "geospatial-analysis", "title": "Geospatial analysis"},
        "spec": {
            "language": {"name": "python", "version": "3.13"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "packages": {
                "python": {
                    "manager": "uv",
                    "dependencies": ["geopandas==1.1.1", "rasterio==1.4.3"],
                },
                "system": {"apt": ["gdal-bin"]},
            },
            "env": {"GDAL_DATA": "/usr/share/gdal"},
            "commands": {"postInstall": ["python -c 'import geopandas'"]},
            "resources": {"sizeClass": "medium"},
            "compatibility": {"variants": {"required": ["datalayer"]}},
            **changes.pop("spec", {}),
        },
    }
    fields = {
        "environment_uid": "01k0env0000000000000000000",
        "version": 3,
        "build_uid": "bld-1",
        "owner_uid": OWNER,
        "variant": "datalayer",
        "environment": parse_environment(spec),
        "lock_text": LOCK,
        "lock_digest": LOCK_DIGEST,
        "resolved_base": BASE,
        "region": "r1",
        "size_class": "medium",
    }
    fields.update(changes)
    return BuildRequest(**fields)


def a_builder(**changes) -> Builder:
    options = {
        "credential": Credential(),
        "buildctl": "/usr/bin/buildctl",
        "ecr": FakeEcr(),
        "log": lambda _line: None,
    }
    options.update(changes)
    return Builder(**options)


class Buildctl:
    """A `buildctl` that writes the metadata a successful build writes."""

    def __init__(self, *, digest: str | None = DIGEST, returncode: int = 0) -> None:
        self.digest = digest
        self.returncode = returncode
        self.argv: list[str] = []
        self.context: Path | None = None
        self.env: dict[str, str] | None = None

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        self.argv = list(argv)
        self.env = kwargs.get("env")
        for position, value in enumerate(self.argv):
            if value.startswith("context="):
                self.context = Path(value.removeprefix("context="))
            if value == "--metadata-file" and self.digest is not None:
                Path(self.argv[position + 1]).write_text(
                    json.dumps({"containerimage.digest": self.digest}), encoding="utf-8"
                )
        return subprocess.CompletedProcess(self.argv, self.returncode, "", "solved\n")


# -- The Dockerfile ------------------------------------------------------------


class TestTheDockerfileItGenerates:
    def test_it_is_what_the_section_4_1_example_builds(self) -> None:
        """The snapshot. Every line of it is a decision; read the diff, not just the failure."""
        dockerfile = a_builder().dockerfile(a_request())
        assert dockerfile == (
            "# syntax=docker/dockerfile:1.7\n"
            "# Generated by Datalayer for geospatial-analysis v3, build bld-1.\n"
            f"# Lock: {LOCK_DIGEST}\n"
            f"FROM {BASE}\n"
            'LABEL io.datalayer.environment="geospatial-analysis" \\\n'
            '      io.datalayer.environment.uid="01k0env0000000000000000000" \\\n'
            '      io.datalayer.version="3" \\\n'
            '      io.datalayer.build="bld-1" \\\n'
            f'      io.datalayer.lock="{LOCK_DIGEST}" \\\n'
            '      io.datalayer.contract="sandbox-contract/v1" \\\n'
            '      io.datalayer.size-class="medium"\n'
            "ENV GDAL_DATA=/usr/share/gdal\n"
            "USER root\n"
            "RUN --mount=type=cache,target=/var/cache/apt,sharing=locked apt-get update -qq \\\n"
            "    && apt-get install -y --no-install-recommends "
            "gdal-bin=3.8.4+dfsg-3build2 libgdal34=3.8.4+dfsg-3build2 \\\n"
            "    && rm -rf /var/lib/apt/lists/*\n"
            "USER root\n"
            "COPY lock.txt /opt/datalayer/lock.txt\n"
            "RUN --mount=type=cache,target=/root/.cache/uv "
            "uv pip sync --system --require-hashes --find-links /opt/datalayer/wheelhouse "
            "/opt/datalayer/lock.txt\n"
            "USER 1000:100\n"
            "WORKDIR /home/datalayer/content\n"
            "RUN --network=none python -c 'import geopandas'\n"
            "USER 1000:100\n"
            "WORKDIR /home/datalayer/content\n"
            "RUN /opt/datalayer/bin/datalayer-sandbox doctor --json > /tmp/doctor.json\n"
        )

    def test_the_apt_versions_come_from_the_lock_and_not_from_the_spec(self) -> None:
        # The spec asks for `gdal-bin`; the lock says which one, and names the
        # transitive package too, so a rebuild installs the same two.
        dockerfile = a_builder().dockerfile(a_request())
        assert "gdal-bin=3.8.4+dfsg-3build2" in dockerfile
        assert "libgdal34=3.8.4+dfsg-3build2" in dockerfile
        assert "apt-get install -y --no-install-recommends gdal-bin " not in dockerfile

    def test_the_environment_is_set_before_anything_installs(self) -> None:
        dockerfile = a_builder().dockerfile(a_request())
        assert dockerfile.index("ENV GDAL_DATA") < dockerfile.index("apt-get install")
        assert dockerfile.index("ENV GDAL_DATA") < dockerfile.index("uv pip sync")

    def test_post_install_runs_as_the_user_with_no_network(self) -> None:
        dockerfile = a_builder().dockerfile(a_request())
        commands = dockerfile.splitlines()
        run = next(line for line in commands if "import geopandas" in line)
        assert run.startswith("RUN --network=none ")
        assert commands[commands.index(run) - 2] == "USER 1000:100"

    def test_it_installs_from_the_lock_with_hashes(self) -> None:
        dockerfile = a_builder().dockerfile(a_request())
        assert (
            "uv pip sync --system --require-hashes --find-links /opt/datalayer/wheelhouse "
            "/opt/datalayer/lock.txt" in dockerfile
        )
        # Never the loose list: that is the whole point of resolving once.
        assert "geopandas==1.1.1" not in dockerfile

    def test_a_build_secret_is_mounted_for_its_step_alone(self) -> None:
        request = a_request(build_secret_ids=("dlsec_abc",))
        dockerfile = a_builder().dockerfile(request)
        assert "--mount=type=secret,id=dlsec_abc" in dockerfile
        # Nowhere else, and never in a layer.
        assert dockerfile.count("dlsec_abc") == 2

    def test_it_ends_by_running_the_contracts_own_check(self) -> None:
        assert (
            a_builder()
            .dockerfile(a_request())
            .rstrip()
            .endswith("RUN /opt/datalayer/bin/datalayer-sandbox doctor --json > /tmp/doctor.json")
        )


# -- What it refuses before building -------------------------------------------


class TestWhatItSaysItCannotBuild:
    def test_a_gpu_class_points_at_the_providers_that_run_one(self) -> None:
        request = a_request(spec={"resources": {"sizeClass": "gpu-small"}})
        report = a_builder().validate(request.environment)
        assert not report.supported
        assert any("modal or daytona" in finding.message for finding in report.findings)
        assert all(finding.field == "spec.resources.sizeClass" for finding in report.findings)

    def test_a_form_that_is_not_built_yet_is_a_finding(self) -> None:
        request = a_request(spec={"build": {"source": "dockerfile"}})
        report = a_builder().validate(request.environment)
        assert [finding.field for finding in report.findings] == ["spec.build.source"]

    def test_a_dependency_file_source_is_supported(self) -> None:
        """E3-01 resolves it the same way `packages` does; this builder never
        refused it on its own — until this box, its own `validate` still did."""
        request = a_request(
            spec={
                "packages": {},
                "build": {
                    "source": "dependencyFile",
                    "dependencyFile": {
                        "sourceFormat": "requirements",
                        "content": "geopandas==1.1.1\n",
                    },
                },
            }
        )
        report = a_builder().validate(request.environment, lock_text=LOCK)
        assert report.supported

    def test_an_image_source_is_supported(self) -> None:
        request = a_request(
            spec={
                "packages": {},
                "build": {
                    "source": "image",
                    "image": {"reference": "python:3.12-slim-bookworm"},
                },
            }
        )
        report = a_builder().validate(request.environment, lock_text=LOCK)
        assert report.supported

    def test_an_empty_lock_is_a_finding_because_the_build_installs_from_it(self) -> None:
        report = a_builder().validate(a_request().environment, lock_text="# nothing\n")
        assert any(finding.code == "DL_ENV_SPEC_INVALID" for finding in report.findings)

    def test_the_section_4_1_example_is_supported(self) -> None:
        assert a_builder().validate(a_request().environment, lock_text=LOCK).supported

    def test_this_variant_has_no_gpu_yet(self) -> None:
        capabilities = a_builder().capabilities()
        assert capabilities.supports_gpu is False
        assert capabilities.build_sources == ("packages", "dependencyFile", "image")
        assert capabilities.package_managers == ("uv", "pip")


# -- The push ------------------------------------------------------------------


class TestBuildingAndPushing:
    def test_it_pushes_by_digest_with_its_attestations(self) -> None:
        buildctl = Buildctl()
        builder = a_builder(run=buildctl)
        artifact = builder.build(a_request())
        argv = " ".join(buildctl.argv)
        repository = owner_repository(OWNER, "geospatial-analysis")
        assert f"type=image,name={REGISTRY}/{repository}:v3-bld-1,push=true" in argv
        # `--opt attest:...`, not `--attest`: that flag is `docker buildx`'s,
        # and a bare `buildctl` refuses it outright (found live, 2026-09-12).
        assert "--opt attest:sbom=" in argv
        assert "--opt attest:provenance=mode=max" in argv
        # The digest is what is kept; the tag is only there to find the build.
        assert artifact.immutable_reference == f"{REGISTRY}/{repository}@{DIGEST}"
        assert artifact.provider_artifact_id == DIGEST
        assert artifact.mutable_alias == f"{REGISTRY}/{repository}:v3-bld-1"
        assert artifact.region == "r1"
        assert artifact.size_class == "medium"
        assert artifact.contract_version == "sandbox-contract/v1"

    def test_the_context_holds_the_lock_the_build_installs(self) -> None:
        buildctl = Buildctl()
        a_builder(run=buildctl).build(a_request())
        assert buildctl.context is not None
        # The directory is gone by now; what matters is that it was the context
        # and that the argv named the Dockerfile beside it.
        assert f"dockerfile={buildctl.context}" in buildctl.argv

    def test_a_retried_build_of_the_same_version_pushes_a_tag_of_its_own(self) -> None:
        first, second = Buildctl(), Buildctl(digest="sha256:" + "ee" * 32)
        a_builder(run=first).build(a_request())
        a_builder(run=second).build(a_request(build_uid="bld-2"))
        tags = [
            part
            for argv in (first.argv, second.argv)
            for part in argv
            if part.startswith("type=image")
        ]
        assert "v3-bld-1" in tags[0] and "v3-bld-2" in tags[1]
        assert tags[0] != tags[1]

    def test_the_owners_cache_is_imported_and_exported_and_is_only_theirs(self) -> None:
        buildctl = Buildctl()
        a_builder(run=buildctl).build(a_request())
        argv = " ".join(buildctl.argv)
        cache = f"{REGISTRY}/environments/cache/u/{OWNER}:datalayer"
        assert f"--import-cache type=registry,ref={cache}" in argv
        assert f"type=registry,ref={cache},mode=max,image-manifest=true" in argv

    def test_the_repository_is_created_with_immutable_tags_when_it_is_missing(self) -> None:
        ecr = FakeEcr()
        a_builder(run=Buildctl(), ecr=ecr).build(a_request())
        assert ecr.created[0]["repositoryName"] == owner_repository(OWNER, "geospatial-analysis")
        assert ecr.created[0]["imageTagMutability"] == "IMMUTABLE"
        assert ecr.created[0]["imageScanningConfiguration"] == {"scanOnPush": True}
        assert ecr.created[0]["encryptionConfiguration"] == {"encryptionType": "KMS"}

    def test_the_cache_repository_is_created_with_mutable_tags_when_it_is_missing(self) -> None:
        """Terraform never creates it — its own comment says "created by the

        builder" — and until this, nothing here ever had either: the first
        real build got all the way to a real push and only then failed
        exporting its cache, a plain 404 against a repository that was never
        created (found live, 2026-09-12).
        """
        ecr = FakeEcr()
        a_builder(run=Buildctl(), ecr=ecr).build(a_request())
        cache = [
            item
            for item in ecr.created
            if item["repositoryName"] == f"environments/cache/u/{OWNER}"
        ]
        assert len(cache) == 1
        assert cache[0]["imageTagMutability"] == "MUTABLE"
        assert cache[0]["imageScanningConfiguration"] == {"scanOnPush": True}
        assert cache[0]["encryptionConfiguration"] == {"encryptionType": "KMS"}

    def test_an_existing_repository_is_left_as_it_is(self) -> None:
        ecr = FakeEcr(
            repositories={
                owner_repository(OWNER, "geospatial-analysis"),
                f"environments/cache/u/{OWNER}",
            }
        )
        a_builder(run=Buildctl(), ecr=ecr).build(a_request())
        assert ecr.created == []

    def test_the_credential_reaches_buildctl_and_not_the_workers_own_config(self) -> None:
        buildctl = Buildctl()
        a_builder(run=buildctl).build(a_request())
        assert buildctl.env is not None
        config = Path(buildctl.env["DOCKER_CONFIG"]) / "config.json"
        written = json.loads(config.read_text(encoding="utf-8"))
        assert REGISTRY in written["auths"]
        # Readable by this build alone.
        assert oct(config.stat().st_mode)[-3:] == "600"

    def test_a_failed_build_is_that_code_and_says_to_read_the_log(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(run=Buildctl(returncode=1, digest=None)).build(a_request())
        assert raised.value.code.code == "DL_ENV_BUILD_FAILED"

    def test_a_build_that_recorded_no_digest_is_not_an_artifact(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(run=Buildctl(digest=None)).build(a_request())
        assert raised.value.code.code == "DL_ENV_BUILD_FAILED"
        assert "digest" in raised.value.message

    def test_it_names_the_build_pool_when_there_is_no_buildctl(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(buildctl="").build(a_request())
        assert raised.value.detail["item"] == "E1-06"

    def test_a_base_that_is_not_pinned_by_digest_is_not_even_a_request(self) -> None:
        # `BuildRequest` refuses it, so no build can start from a tag: the
        # builder's own check below it is the belt to that brace.
        with pytest.raises(ValueError, match="resolved to a digest"):
            a_request(resolved_base="datalayer/python-cpu:2026.09")

    def test_a_build_with_no_credential_says_which_item_mints_one(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            Builder(buildctl="/usr/bin/buildctl", ecr=FakeEcr(), run=Buildctl()).build(a_request())
        assert raised.value.detail["missing"] == "credential"


# -- Reading the registry ------------------------------------------------------


def an_artifact(digest: str = DIGEST) -> ArtifactReference:
    repository = owner_repository(OWNER, "geospatial-analysis")
    return ArtifactReference(
        variant="datalayer",
        immutable_reference=f"{REGISTRY}/{repository}@{digest}",
        provider_artifact_id=digest,
        contract_version="sandbox-contract/v1",
    )


class TestReadingTheRegistry:
    def repository(self) -> str:
        return owner_repository(OWNER, "geospatial-analysis")

    def an_ecr(self) -> FakeEcr:
        return FakeEcr(
            repositories={self.repository()},
            images={
                (self.repository(), DIGEST): {
                    "imageDigest": DIGEST,
                    "imageSizeInBytes": 116_178_944,
                    "imageTags": ["v3-bld-1"],
                    "imageManifestMediaType": "application/vnd.oci.image.manifest.v1+json",
                },
                (self.repository(), "v3-bld-1"): {
                    "imageDigest": DIGEST,
                    "imageSizeInBytes": 116_178_944,
                    "imageTags": ["v3-bld-1"],
                },
            },
        )

    def test_inspect_reads_the_size_the_registry_holds(self) -> None:
        metadata = a_builder(ecr=self.an_ecr()).inspect(an_artifact())
        assert metadata.size_bytes == 116_178_944
        assert metadata.labels["tags"] == "v3-bld-1"

    def test_inspect_says_so_when_the_image_is_gone(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(ecr=self.an_ecr()).inspect(an_artifact("sha256:" + "99" * 32))
        assert raised.value.code.code == "DL_ENV_ARTIFACT_MISSING"

    def test_resolve_turns_a_tag_into_a_digest(self) -> None:
        artifact = a_builder(ecr=self.an_ecr()).resolve(f"{REGISTRY}/{self.repository()}:v3-bld-1")
        assert artifact.immutable_reference.endswith(f"@{DIGEST}")
        assert artifact.mutable_alias.endswith(":v3-bld-1")

    def test_resolve_refuses_something_that_is_not_a_tag(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(ecr=self.an_ecr()).resolve("no-tag-here")
        assert raised.value.code.code == "DL_ENV_SPEC_INVALID"

    def test_exists_turns_false_after_delete(self) -> None:
        builder = a_builder(ecr=self.an_ecr())
        assert builder.exists(an_artifact()) is True
        builder.delete(an_artifact())
        assert builder.exists(an_artifact()) is False

    def test_deleting_what_is_already_gone_is_a_success(self) -> None:
        builder = a_builder(ecr=FakeEcr(repositories={self.repository()}))
        builder.delete(an_artifact())  # no raise

    def test_a_smoke_test_is_runtimes_business_for_this_variant(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            a_builder().smoke_test(an_artifact())
        assert raised.value.detail["item"] == "E1-14"


def test_the_builder_is_loaded_by_variant_and_keeps_the_interface() -> None:
    from code_sandboxes.environments.builders import builder_contract_violations, get_builder

    builder = get_builder("datalayer", log=print, credential=Credential(), buildctl="")
    assert builder_contract_violations(builder, variant="datalayer") == []
