# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Daytona builder: a snapshot from the lock, and its id kept (E2-04).

Nothing here reaches Daytona: the `daytona`/`daytona_api_client` SDKs are
doubles that record what they were asked, the way `datalayer.py`'s own tests
record `buildctl`'s argv without a daemon, and `test_environment_e2b_builder.py`
records a `Template` chain without a real build. What is checked is what the
declarative image *is* — the base by digest, `env` before any install, apt
packages from the lock, `USER root` bracketing the install steps, no doctor or
wheelhouse copied (the Datalayer base already bakes both, unlike E2B's), an
explicit entrypoint and resources every build, and the registry entry this
build's own base pull needs, made and torn down around it.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

import pytest

from code_sandboxes.environments.adapters.daytona import DAYTONA_GPUS, Builder, daytona_gpu
from code_sandboxes.environments.builders import ArtifactReference, BuildRequest
from code_sandboxes.environments.errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    EnvironmentsError,
)
from code_sandboxes.environments.spec import parse_environment

OWNER = "01k0wner000000000000000000"
BASE = "environments/base/python-cpu@sha256:" + "bb" * 32
LOCK = (
    "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.\n"
    "# python: 3.13\n"
    "# datalayer-protected: ipykernel==7.3.0\n"
    "geopandas==1.1.1 \\\n    --hash=sha256:" + "cd" * 32 + "\n"
)
APT_LOCK = LOCK + "# datalayer-apt: gdal-bin=3.8.4+dfsg-3build2\n"
LOCK_DIGEST = "sha256:" + "dd" * 32

#: A conda explicit lock (E3-02) and its `dependencyFile` source.
CONDA_LOCK = (
    "# Resolved by Datalayer (PLAN_ENV.md D-9). Do not edit: a change makes a new version.\n"
    "# python: 3.13\n"
    "# platform: linux-64\n"
    "# datalayer-pip: ipykernel==7.3.0\n"
    "@EXPLICIT\n"
    "https://conda.anaconda.org/conda-forge/linux-64/gdal-3.8.4-py313.conda#" + "ab" * 32 + "\n"
)
CONDA_SPEC = {
    "build": {
        "source": "dependencyFile",
        "dependencyFile": {
            "sourceFormat": "conda",
            "content": "name: geo\nchannels: [conda-forge]\ndependencies: [python=3.13, gdal]\n",
        },
    },
}


class Credential:
    """The build's owner secrets, as the workflow mints them (D-8, D-17, E2-01)."""

    provider_secrets: ClassVar[dict[str, str]] = {"DAYTONA_API_KEY": "owners-daytona-key"}
    registry: ClassVar[str] = "773842031886.dkr.ecr.us-east-1.amazonaws.com"
    username: ClassVar[str] = "AWS"
    password: ClassVar[str] = "ecr-token"


class JwtCredential:
    """An owner authenticated by JWT instead of an API key (D-8) — the other
    form `accounts.py`'s own `CREDENTIAL_VARIABLES` lists for Daytona."""

    provider_secrets: ClassVar[dict[str, str]] = {
        "DAYTONA_JWT_TOKEN": "owners-jwt-token",
        "DAYTONA_ORGANIZATION_ID": "org-42",
    }


class NoRegistryCredential:
    """A credential with the owner's own key, but no base-reader login (D-18)."""

    provider_secrets: ClassVar[dict[str, str]] = {"DAYTONA_API_KEY": "owners-daytona-key"}


class Call:
    """One method call a fake recorded: its name, arguments and keywords."""

    def __init__(
        self, name: str, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs or {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Call({self.name!r}, {self.args!r}, {self.kwargs!r})"


class FakeImage:
    """The declarative `Image` chain: every call recorded, in order, on one list."""

    def __init__(self, calls: list[Call]) -> None:
        self.calls = calls

    @classmethod
    def base(cls, ref: str) -> FakeImage:
        return cls([Call("base", (ref,))])

    @classmethod
    def from_dockerfile(cls, path: str) -> FakeImage:
        """The text is read now: the builder's scratch file is gone by the time a test looks."""
        return cls([Call("from_dockerfile", (path,), {"content": Path(path).read_text()})])

    def env(self, env_vars: dict[str, str]) -> FakeImage:
        self.calls.append(Call("env", (env_vars,)))
        return self

    def dockerfile_commands(self, lines: list[str]) -> FakeImage:
        self.calls.append(Call("dockerfile_commands", (lines,)))
        return self

    def run_commands(self, *commands: str) -> FakeImage:
        self.calls.append(Call("run_commands", commands))
        return self

    def add_local_file(self, local_path: str, remote_path: str) -> FakeImage:
        content = Path(local_path).read_bytes()
        self.calls.append(Call("add_local_file", (local_path, remote_path), {"content": content}))
        return self

    def add_local_dir(self, local_path: str, remote_path: str) -> FakeImage:
        self.calls.append(Call("add_local_dir", (local_path, remote_path)))
        return self

    def workdir(self, path: str) -> FakeImage:
        self.calls.append(Call("workdir", (path,)))
        return self


class FakeResources:
    def __init__(
        self,
        *,
        cpu: int | None = None,
        memory: int | None = None,
        disk: int | None = None,
        gpu: int | None = None,
        gpu_type: Any = None,
    ) -> None:
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.gpu = gpu
        self.gpu_type = gpu_type


#: The SDK's `GpuType`, by value, as the builder asks for one.
FakeGpuType = enum.Enum("GpuType", {name.replace("-", "_"): name for name in DAYTONA_GPUS})


class FakeCreateSnapshotParams:
    def __init__(
        self,
        *,
        name: str,
        image: Any,
        resources: Any = None,
        entrypoint: list[str] | None = None,
        region_id: str | None = None,
        sandbox_class: Any = None,
    ) -> None:
        self.name = name
        self.image = image
        self.resources = resources
        self.entrypoint = entrypoint
        self.region_id = region_id
        self.sandbox_class = sandbox_class


class FakeDaytonaConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeDaytonaNotFoundError(Exception):
    """Stands in for `daytona.DaytonaNotFoundError`."""


class FakeSnapshot:
    def __init__(
        self,
        *,
        id: str = "snp-123",  # noqa: A002 - mirrors the SDK's own `Snapshot.id` field
        name: str = "dl-geospatial-analysis-v3-bld-1",
        created_at: Any = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc),
        state: Any = "ACTIVE",
    ) -> None:
        self.id = id
        self.name = name
        self.created_at = created_at
        self.state = state


class FakeSnapshotService:
    def __init__(
        self,
        *,
        create_result: FakeSnapshot | None = None,
        create_error: Exception | None = None,
        get_results: dict[str, FakeSnapshot] | None = None,
        get_errors: dict[str, Exception] | None = None,
        delete_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.create_calls: list[Call] = []
        self.get_calls: list[Call] = []
        self.delete_calls: list[Call] = []
        self._delete_errors = delete_errors or {}
        self._create_result = create_result
        self._create_error = create_error
        self._get_results = get_results or {}
        self._get_errors = get_errors or {}

    def create(
        self, params: FakeCreateSnapshotParams, *, on_logs: Any = None, timeout: Any = None
    ) -> FakeSnapshot:
        self.create_calls.append(
            Call("create", (params,), {"on_logs": on_logs, "timeout": timeout})
        )
        if on_logs:
            on_logs(f"Creating snapshot {params.name} (SnapshotState.PENDING)")
        if self._create_error:
            raise self._create_error
        return self._create_result or FakeSnapshot(name=params.name)

    def get(self, name_or_id: str) -> FakeSnapshot:
        self.get_calls.append(Call("get", (name_or_id,)))
        if name_or_id in self._get_errors:
            raise self._get_errors[name_or_id]
        if name_or_id in self._get_results:
            return self._get_results[name_or_id]
        raise FakeDaytonaNotFoundError(f"no such snapshot {name_or_id}")

    def delete(self, snapshot: Any) -> None:
        """As the SDK's: an id or a name, and a missing one is not found."""
        self.delete_calls.append(Call("delete", (snapshot,)))
        if snapshot in self._delete_errors:
            raise self._delete_errors[snapshot]
        if snapshot not in self._get_results:
            raise FakeDaytonaNotFoundError(f"no such snapshot {snapshot}")
        del self._get_results[snapshot]


class FakeDaytonaClient:
    def __init__(self, *, snapshot_service: FakeSnapshotService | None = None) -> None:
        #: A sentinel, never a real API client: `_register_base_pull` reaches
        #: it only to hand it, opaquely, to `DockerRegistryApi`.
        self._api_client = object()
        self.snapshot = snapshot_service or FakeSnapshotService()


class _DaytonaFactory:
    """Stands in for the `Daytona` class itself: calling it records the config
    it was given and hands back the one client every test inspects."""

    def __init__(self, module: FakeDaytonaModule) -> None:
        self._module = module

    def __call__(self, config: FakeDaytonaConfig | None = None) -> FakeDaytonaClient:
        self._module.daytona_calls.append(Call("Daytona", (config,)))
        return self._module.client


class FakeDaytonaModule:
    """Stands in for `import daytona`."""

    def __init__(self, *, client: FakeDaytonaClient | None = None) -> None:
        self.client = client or FakeDaytonaClient()
        self.daytona_calls: list[Call] = []
        self.DaytonaNotFoundError = FakeDaytonaNotFoundError
        self.Image = FakeImage
        self.Resources = FakeResources
        self.GpuType = FakeGpuType
        self.CreateSnapshotParams = FakeCreateSnapshotParams
        self.DaytonaConfig = FakeDaytonaConfig
        self.Daytona = _DaytonaFactory(self)


class FakeCreatedRegistry:
    def __init__(self, id: str = "reg-abc123") -> None:  # noqa: A002 - mirrors `DockerRegistry.id`
        self.id = id


class FakeCreateDockerRegistry:
    def __init__(self, *, name: str, url: str, username: str, password: str) -> None:
        self.name = name
        self.url = url
        self.username = username
        self.password = password


class FakeRegistryClient:
    def __init__(
        self,
        *,
        create_result: FakeCreatedRegistry | None = None,
        create_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.create_calls: list[FakeCreateDockerRegistry] = []
        self.delete_calls: list[str] = []
        self._create_result = create_result
        self._create_error = create_error
        self._delete_error = delete_error

    def create_registry(self, payload: FakeCreateDockerRegistry) -> FakeCreatedRegistry:
        self.create_calls.append(payload)
        if self._create_error:
            raise self._create_error
        return self._create_result or FakeCreatedRegistry()

    def delete_registry(self, registry_id: str) -> None:
        self.delete_calls.append(registry_id)
        if self._delete_error:
            raise self._delete_error


class _DockerRegistryApiFactory:
    """Stands in for the `DockerRegistryApi` class itself: calling it records
    the `_api_client` it was given and hands back the one client every test
    inspects."""

    def __init__(self, sdk: FakeRegistrySdk) -> None:
        self._sdk = sdk

    def __call__(self, api_client: Any) -> FakeRegistryClient:
        self._sdk.api_client_calls.append(api_client)
        return self._sdk.client


class FakeRegistrySdk:
    """Stands in for `import daytona_api_client`."""

    def __init__(self, *, client: FakeRegistryClient | None = None) -> None:
        self.client = client or FakeRegistryClient()
        self.api_client_calls: list[Any] = []
        self.CreateDockerRegistry = FakeCreateDockerRegistry
        self.DockerRegistryApi = _DockerRegistryApiFactory(self)


def a_request(**changes: Any) -> BuildRequest:
    spec = {
        "apiVersion": "environments.datalayer.io/v1alpha1",
        "kind": "Environment",
        "metadata": {"name": "geospatial-analysis", "title": "Geospatial analysis"},
        "spec": {
            "language": {"name": "python", "version": "3.13"},
            "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
            "packages": {"python": {"manager": "uv", "dependencies": ["geopandas==1.1.1"]}},
            "env": {"GDAL_DATA": "/usr/share/gdal"},
            "commands": {"postInstall": ["python -c 'import geopandas'"]},
            "resources": {"sizeClass": "medium"},
            "compatibility": {"variants": {"required": ["daytona"]}},
            **changes.pop("spec", {}),
        },
    }
    fields = {
        "environment_uid": "01k0env0000000000000000000",
        "version": 3,
        "build_uid": "bld-1",
        "owner_uid": OWNER,
        "variant": "daytona",
        "environment": parse_environment(spec),
        "lock_text": LOCK,
        "lock_digest": LOCK_DIGEST,
        "resolved_base": BASE,
        "region": "us",
        "size_class": "medium",
    }
    fields.update(changes)
    return BuildRequest(**fields)


def a_builder(
    *,
    daytona: FakeDaytonaModule | None = None,
    registry: FakeRegistrySdk | None = None,
    **changes: Any,
) -> Builder:
    options: dict[str, Any] = {
        "daytona_sdk": lambda: daytona or FakeDaytonaModule(),
        "registry_sdk": lambda: registry or FakeRegistrySdk(),
        "credential": Credential(),
    }
    options.update(changes)
    return Builder(**options)


def an_artifact(**changes: Any) -> ArtifactReference:
    fields = {
        "variant": "daytona",
        "immutable_reference": "snp-123",
        "provider_artifact_id": "snp-123",
        "region": "us",
        "size_class": "medium",
        "mutable_alias": "dl-geospatial-analysis-v3-bld-1",
        "contract_version": "sandbox-contract/v1",
    }
    fields.update(changes)
    return ArtifactReference(**fields)


def calls_named(image: FakeImage, name: str) -> list[Call]:
    return [call for call in image.calls if call.name == name]


class TestBuildingASnapshot:
    def test_it_starts_from_the_resolved_base(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        [snapshot_call] = daytona.client.snapshot.create_calls
        image = snapshot_call.args[0].image
        assert image.calls[0].name == "base"
        assert image.calls[0].args[0] == BASE

    def test_env_is_set_before_any_install_step(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        names = [call.name for call in image.calls]
        install = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "run_commands" and "uv pip sync" in call.args[0]
        )
        assert names.index("env") < install

    def test_apt_packages_from_the_lock_are_installed(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request(lock_text=APT_LOCK))
        image = daytona.client.snapshot.create_calls[0].args[0].image
        installs = [
            call for call in calls_named(image, "run_commands") if "apt-get install" in call.args[0]
        ]
        assert len(installs) == 1
        assert "gdal-bin=3.8.4+dfsg-3build2" in installs[0].args[0]

    def test_no_apt_step_when_the_lock_pins_none(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        assert not any(
            "apt-get install" in call.args[0] for call in calls_named(image, "run_commands")
        )

    def test_neither_the_doctor_nor_the_wheelhouse_is_copied(self) -> None:
        """The Datalayer base already bakes both (E1-05, found live 2026-09-13):
        copying either again would only duplicate what is already there."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        assert calls_named(image, "add_local_dir") == []
        [lock_copy] = calls_named(image, "add_local_file")
        assert lock_copy.args[1] == "/opt/datalayer/lock.txt"

    def test_uv_is_not_reinstalled_the_base_already_has_it(self) -> None:
        """`resolve.py`'s own `bootstrap_uv` docstring: an approved base
        already bakes `uv` (E1-05) — reinstalling it added an un-hashed
        network fetch outside the resolved lock for no reason (found in
        review)."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        assert not any(
            "pip install" in call.args[0] and "uv==" in call.args[0]
            for call in calls_named(image, "run_commands")
        )

    def test_the_lock_copied_in_is_the_requests_own(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request(lock_text=LOCK))
        image = daytona.client.snapshot.create_calls[0].args[0].image
        [lock_copy] = calls_named(image, "add_local_file")
        assert lock_copy.kwargs["content"] == LOCK.encode("utf-8")

    def test_uv_pip_sync_reaches_the_bases_own_shared_wheelhouse(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        [sync] = [
            call for call in calls_named(image, "run_commands") if "uv pip sync" in call.args[0]
        ]
        assert "--require-hashes" in sync.args[0]
        assert "--find-links /opt/datalayer/wheelhouse" in sync.args[0]

    def test_a_conda_lock_installs_with_micromamba_and_then_the_pip_pins(self) -> None:
        """A conda source (E3-02): micromamba is bootstrapped, `micromamba
        install --file` reads the `@EXPLICIT` lock, and the pip layer the solve
        resolved follows — never the pip-lock `uv pip sync`."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request(lock_text=CONDA_LOCK, spec=CONDA_SPEC))
        image = daytona.client.snapshot.create_calls[0].args[0].image
        runs = calls_named(image, "run_commands")
        bootstrap = next(i for i, call in enumerate(runs) if "micro.mamba.pm" in call.args[0])
        micromamba = next(i for i, call in enumerate(runs) if "micromamba install" in call.args[0])
        pip = next(i for i, call in enumerate(runs) if "ipykernel==7.3.0" in call.args[0])
        assert bootstrap < micromamba < pip
        assert not any("uv pip sync" in call.args[0] for call in runs)
        # The interpreter a sandbox runs, not a new `base` under the content
        # home (2026-09-18): the same command the Datalayer builder emits.
        assert "--root-prefix /opt/conda --prefix /opt/conda" in runs[micromamba].args[0]
        assert "--name base" not in runs[micromamba].args[0]

    def test_user_root_brackets_the_install_steps(self) -> None:
        """Daytona honours the base's `USER`, unlike E2B (E0-04): no synthetic
        account, just `USER root` around what needs it."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        root_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "dockerfile_commands" and call.args[0] == ["USER root"]
        )
        contract_user_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "dockerfile_commands" and call.args[0][0].startswith("USER 1000:100")
        )
        sync_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "run_commands" and "uv pip sync" in call.args[0]
        )
        assert root_at < sync_at < contract_user_at

    def test_post_install_runs_after_user_is_restored(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        contract_user_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "dockerfile_commands" and call.args[0][0].startswith("USER 1000:100")
        )
        post_install_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "run_commands" and "import geopandas" in call.args[0]
        )
        assert contract_user_at < post_install_at

    def test_the_doctor_check_is_not_run_at_build_time(self) -> None:
        """A build-time `RUN` step's own PID 1 is Daytona's build agent, not
        the finished snapshot's — found live, 2026-09-13: `init` failed
        deterministically for a reason unrelated to the shipped artifact.
        A real answer needs a launched sandbox (E1-14), not this builder."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        assert not any("doctor" in call.args[0] for call in calls_named(image, "run_commands"))

    def test_the_chain_ends_with_workdir(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        image = daytona.client.snapshot.create_calls[0].args[0].image
        assert image.calls[-1].name == "workdir"
        assert image.calls[-1].args[0] == "/home/datalayer"

    def test_the_entrypoint_is_always_set(self) -> None:
        """Daytona's own default, unset, is `sleep infinity` with no PID 1 (§11.3
        item 3): a real one is set every build."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        params = daytona.client.snapshot.create_calls[0].args[0]
        assert params.entrypoint == ["tini", "--", "sleep", "infinity"]

    @pytest.mark.parametrize(
        "size_class,cpu,memory,disk",
        [("small", 1, 2, 10), ("medium", 4, 8, 20), ("large", 8, 16, 40)],
    )
    def test_resources_come_from_the_size_class(
        self, size_class: str, cpu: int, memory: int, disk: int
    ) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request(size_class=size_class))
        resources = daytona.client.snapshot.create_calls[0].args[0].resources
        assert (resources.cpu, resources.memory, resources.disk) == (cpu, memory, disk)

    def test_the_region_sent_is_daytonas_own_not_this_platforms(self) -> None:
        """`request.region` is Datalayer's — `r1` — and Daytona answered
        "Region not found" for it on the first real build (2026-09-17).

        The owner names a Daytona region in `compatibility.regions`; that is
        the one that scopes the snapshot.
        """
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(
            a_request(region="r1", spec={"compatibility": {"regions": ["eu"]}})
        )
        params = daytona.client.snapshot.create_calls[0].args[0]
        assert params.region_id == "eu"

    def test_with_no_region_named_the_account_default_decides(self) -> None:
        """The field is left out rather than filled with something Daytona
        does not know — which is what every snapshot in a real account has."""
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request(region="r1"))
        params = daytona.client.snapshot.create_calls[0].args[0]
        assert params.region_id is None

    def test_the_snapshot_is_named_after_the_environment_and_version(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(a_request())
        params = daytona.client.snapshot.create_calls[0].args[0]
        assert params.name == "dl-geospatial-analysis-v3-bld-1"

    def test_build_logs_reach_the_log(self) -> None:
        logged: list[str] = []
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona, log=logged.append).build(a_request())
        assert any("Creating snapshot" in line for line in logged)

    def test_the_artifact_is_the_snapshot_id(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    create_result=FakeSnapshot(id="snp-999", name="dl-geospatial-analysis-v3-bld-1")
                )
            )
        )
        artifact = a_builder(daytona=daytona).build(a_request())
        assert artifact.variant == "daytona"
        assert artifact.immutable_reference == "snp-999"
        assert artifact.provider_artifact_id == "snp-999"
        assert artifact.mutable_alias == "dl-geospatial-analysis-v3-bld-1"
        assert artifact.region == "us"
        assert artifact.size_class == "medium"
        assert artifact.contract_version

    def test_provider_account_is_populated_from_the_credential(self) -> None:
        daytona = FakeDaytonaModule()
        artifact = a_builder(daytona=daytona).build(a_request())
        assert artifact.provider_account and artifact.provider_account.startswith("daytona:")

    def test_provider_account_is_none_with_no_credential(self) -> None:
        daytona = FakeDaytonaModule()
        artifact = a_builder(daytona=daytona, credential=None).build(a_request())
        assert artifact.provider_account is None

    def test_a_build_failure_is_reported_with_the_log(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(create_error=RuntimeError("quota exceeded"))
            )
        )
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(daytona=daytona).build(a_request())
        assert raised.value.code.code == BUILD_FAILED.code
        assert "quota exceeded" in str(raised.value)
        assert raised.value.detail["log"]


class TestTheBuildsOwnRegistryEntry:
    def test_a_registry_entry_is_made_for_the_build(self) -> None:
        registry = FakeRegistrySdk()
        a_builder(registry=registry).build(a_request())
        [created] = registry.client.create_calls
        assert created.url == Credential.registry
        assert created.username == "AWS"
        assert created.password == "ecr-token"

    def test_it_is_deleted_by_id_not_by_the_name_it_was_given(self) -> None:
        """`delete_registry` takes the id, not the name (found live, 2026-09-13:
        a first attempt deleted by name and got `NotFoundException`)."""
        registry = FakeRegistrySdk(
            client=FakeRegistryClient(create_result=FakeCreatedRegistry(id="reg-xyz"))
        )
        a_builder(registry=registry).build(a_request())
        assert registry.client.delete_calls == ["reg-xyz"]
        [created] = registry.client.create_calls
        assert created.name != "reg-xyz"

    def test_it_is_deleted_even_when_the_build_fails(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(create_error=RuntimeError("no"))
            )
        )
        registry = FakeRegistrySdk(
            client=FakeRegistryClient(create_result=FakeCreatedRegistry(id="reg-xyz"))
        )
        with pytest.raises(EnvironmentsError):
            a_builder(daytona=daytona, registry=registry).build(a_request())
        assert registry.client.delete_calls == ["reg-xyz"]

    def test_no_registry_entry_with_no_pull_credential(self) -> None:
        registry = FakeRegistrySdk()
        a_builder(registry=registry, credential=NoRegistryCredential()).build(a_request())
        assert registry.client.create_calls == []
        assert registry.client.delete_calls == []

    def test_a_registry_create_failure_is_a_provider_error(self) -> None:
        registry = FakeRegistrySdk(
            client=FakeRegistryClient(create_error=RuntimeError("forbidden"))
        )
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(registry=registry).build(a_request())
        assert raised.value.code.code == PROVIDER_ERROR.code

    def test_a_registry_delete_failure_does_not_hide_a_successful_build(self) -> None:
        """Best-effort cleanup: the build's own result matters more (logged, not raised)."""
        logged: list[str] = []
        registry = FakeRegistrySdk(
            client=FakeRegistryClient(delete_error=RuntimeError("already gone"))
        )
        artifact = a_builder(registry=registry, log=logged.append).build(a_request())
        assert artifact.immutable_reference
        assert any("Could not delete the Daytona registry entry" in line for line in logged)


class TestAGpuSnapshot:
    """E2-17: the spec's GPU is baked into the snapshot with its other resources."""

    GPU: ClassVar[dict[str, Any]] = {
        "sizeClass": "gpu-large",
        "accelerator": {"type": "h100", "count": 2, "cuda": "12.8"},
    }

    def gpu_request(self, **resources: Any) -> BuildRequest:
        return a_request(
            spec={
                "base": {"ref": "datalayer/python-cuda", "channel": "2026.09"},
                "resources": {**self.GPU, **resources},
            },
            size_class="gpu-large",
        )

    def test_the_gpu_type_and_count_are_the_specs(self) -> None:
        daytona = FakeDaytonaModule()
        a_builder(daytona=daytona).build(self.gpu_request())
        resources = daytona.client.snapshot.create_calls[0].args[0].resources
        assert (resources.gpu, resources.gpu_type) == (2, FakeGpuType("H100"))
        # D-4 gives a GPU class no CPU or memory: Daytona sizes the machine
        # for the GPU, and the disk holds the CUDA base.
        assert (resources.cpu, resources.memory, resources.disk) == (None, None, 50)

    def test_the_hints_size_the_rest_of_the_machine(self) -> None:
        daytona = FakeDaytonaModule()
        request = self.gpu_request(hints={"cpu": 8, "memoryGi": 64, "diskGi": 120})
        a_builder(daytona=daytona).build(request)
        resources = daytona.client.snapshot.create_calls[0].args[0].resources
        assert (resources.cpu, resources.memory, resources.disk, resources.gpu) == (8, 64, 120, 2)

    def test_a_gpu_daytona_does_not_offer_is_refused_before_any_build(self) -> None:
        request = a_request(
            spec={
                "base": {"ref": "datalayer/python-cuda", "channel": "2026.09"},
                "resources": {"sizeClass": "gpu-large", "accelerator": {"type": "A100-80GB"}},
            }
        )
        report = a_builder().validate(request.environment)
        assert report.supported is False
        [finding] = [item for item in report.findings if "A100-80GB" in item.message]
        assert finding.field == "spec.resources.accelerator.type"
        assert all(name in finding.message for name in DAYTONA_GPUS)

    def test_a_name_is_read_the_way_people_write_it(self) -> None:
        assert [daytona_gpu(name) for name in ("h100", "RTX_4090", " rtx-pro-6000 ", "T4")] == [
            "H100",
            "RTX-4090",
            "RTX-PRO-6000",
            None,
        ]

    def test_the_names_are_the_sdks(self) -> None:
        """Spelled out because `validate` runs where the SDK may not be; held to it here."""
        sdk = pytest.importorskip("daytona")
        offered = {g.value for g in sdk.GpuType if not g.value.lower().startswith("unknown")}
        assert set(DAYTONA_GPUS) == offered


class TestWhatDaytonaCannotBuildYet:
    def test_a_build_secret_is_refused_before_anything_is_queued(self) -> None:
        """Daytona has no per-step secret mechanism E0-04 could find (found
        in review: this chain consumed no build secret at all, and nothing
        said so)."""
        spec = {
            "apiVersion": "environments.datalayer.io/v1alpha1",
            "kind": "Environment",
            "metadata": {"name": "geo"},
            "spec": {
                "language": {"version": "3.13"},
                "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
                "buildSecrets": [{"id": "dlsec_pypitoken1", "name": "PYPI_TOKEN"}],
            },
        }
        report = a_builder().validate(parse_environment(spec))
        assert report.supported is False
        assert "spec.buildSecrets" in [finding.field for finding in report.findings]


class TestReadingTheRegistry:
    def test_inspect_reads_the_snapshot_by_id(self) -> None:
        snapshot = FakeSnapshot(id="snp-123", name="dl-geo-v3", state="ACTIVE")
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(get_results={"snp-123": snapshot})
            )
        )
        metadata = a_builder(daytona=daytona).inspect(an_artifact(provider_artifact_id="snp-123"))
        assert metadata.provider_state == "ACTIVE"
        assert metadata.labels["name"] == "dl-geo-v3"
        assert metadata.created_at == "2026-09-13T12:00:00+00:00"

    def test_inspect_refuses_artifact_missing_once_the_snapshot_is_gone(self) -> None:
        daytona = FakeDaytonaModule()
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(daytona=daytona).inspect(an_artifact(provider_artifact_id="snp-gone"))
        assert raised.value.code.code == ARTIFACT_MISSING.code

    def test_inspect_maps_other_failures_to_a_provider_error(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    get_errors={"snp-123": RuntimeError("timeout")}
                )
            )
        )
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(daytona=daytona).inspect(an_artifact(provider_artifact_id="snp-123"))
        assert raised.value.code.code == PROVIDER_ERROR.code

    def test_exists_true_when_the_snapshot_is_there(self) -> None:
        snapshot = FakeSnapshot(id="snp-123")
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(get_results={"snp-123": snapshot})
            )
        )
        assert (
            a_builder(daytona=daytona).exists(an_artifact(provider_artifact_id="snp-123")) is True
        )

    def test_exists_false_once_the_snapshot_is_gone(self) -> None:
        daytona = FakeDaytonaModule()
        assert (
            a_builder(daytona=daytona).exists(an_artifact(provider_artifact_id="snp-gone")) is False
        )

    def test_exists_maps_other_failures_to_a_provider_error(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    get_errors={"snp-123": RuntimeError("timeout")}
                )
            )
        )
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(daytona=daytona).exists(an_artifact(provider_artifact_id="snp-123"))
        assert raised.value.code.code == PROVIDER_ERROR.code

    def test_inspect_and_exists_pass_the_owners_key_too(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    get_results={"snp-123": FakeSnapshot(id="snp-123")}
                )
            )
        )
        a_builder(daytona=daytona).inspect(an_artifact(provider_artifact_id="snp-123"))
        [config] = [call.args[0] for call in daytona.daytona_calls]
        assert config.kwargs["api_key"] == "owners-daytona-key"

    def test_a_jwt_authenticated_owner_is_recognized_too(self) -> None:
        """`accounts.py`'s own `CREDENTIAL_VARIABLES` lists both auth forms
        for Daytona; only the API key was read at first (found in review)."""
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    get_results={"snp-123": FakeSnapshot(id="snp-123")}
                )
            )
        )
        a_builder(daytona=daytona, credential=JwtCredential()).inspect(
            an_artifact(provider_artifact_id="snp-123")
        )
        [config] = [call.args[0] for call in daytona.daytona_calls]
        assert config.kwargs["jwt_token"] == "owners-jwt-token"
        assert config.kwargs["organization_id"] == "org-42"
        assert "api_key" not in config.kwargs

    def test_the_client_is_built_once_and_reused(self) -> None:
        daytona = FakeDaytonaModule(
            client=FakeDaytonaClient(
                snapshot_service=FakeSnapshotService(
                    get_results={"snp-999": FakeSnapshot(id="snp-999")}
                )
            )
        )
        builder = a_builder(daytona=daytona)
        builder.build(a_request())
        builder.exists(an_artifact(provider_artifact_id="snp-999"))
        assert len(daytona.daytona_calls) == 1


class TestCancellingABuild:
    """E2-18: the build step lets a cancelled build's thread finish unheard, so
    a snapshot Daytona goes on building would be recorded by nobody. Found by
    E2-14's drill on 2026-09-18: `dl-backfill-drill-v2-…` kept building after
    its build was cancelled."""

    NAME = "dl-geospatial-analysis-v3-bld-1"

    def test_a_snapshot_being_made_is_deleted_by_id(self) -> None:
        service = FakeSnapshotService(
            get_results={
                self.NAME: FakeSnapshot(id="snp-9", name=self.NAME),
                "snp-9": FakeSnapshot(id="snp-9", name=self.NAME),
            }
        )
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        builder.cancel(a_request())
        assert [call.args for call in service.get_calls] == [(self.NAME,)]
        assert [call.args for call in service.delete_calls] == [("snp-9",)]

    def test_one_made_after_the_cancel_is_deleted_by_the_build(self) -> None:
        """Not made yet when the cancel came: the build deletes it the moment
        Daytona hands it back, and does not answer with it."""
        made = FakeSnapshot(id="snp-late", name=self.NAME)
        service = FakeSnapshotService(create_result=made, get_results={"snp-late": made})
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        builder.cancel(a_request())
        assert service.delete_calls == []
        with pytest.raises(EnvironmentsError) as raised:
            builder.build(a_request())
        assert raised.value.code.code == BUILD_FAILED.code
        assert "cancelled" in raised.value.message
        assert [call.args for call in service.delete_calls] == [("snp-late",)]

    def test_a_snapshot_daytona_refused_is_deleted_by_the_build(self) -> None:
        """Daytona keeps a failed snapshot, in `error`, under the build's name:
        one over its 20 GB limit was left there (E2-17, 2026-09-18)."""
        failed = FakeSnapshot(id="snp-err", name=self.NAME, state="error")
        service = FakeSnapshotService(
            create_error=RuntimeError(
                "Snapshot size (28.66GB) exceeds maximum allowed size of 20GB"
            ),
            get_results={self.NAME: failed, "snp-err": failed},
        )
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        with pytest.raises(EnvironmentsError) as raised:
            builder.build(a_request())
        assert raised.value.code.code == BUILD_FAILED.code
        assert "20GB" in raised.value.message
        assert [call.args for call in service.delete_calls] == [("snp-err",)]

    def test_another_build_of_the_same_builder_is_not_touched(self) -> None:
        builder = a_builder()
        builder.cancel(a_request(build_uid="bld-other"))
        assert builder.build(a_request()).provider_artifact_id


class TestDeletingASnapshot:
    """E2-18: retention and a failed build both need a snapshot to go."""

    def test_exists_is_false_after_delete(self) -> None:
        service = FakeSnapshotService(get_results={"snp-123": FakeSnapshot(id="snp-123")})
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        artifact = an_artifact(provider_artifact_id="snp-123")
        assert builder.exists(artifact) is True
        builder.delete(artifact)
        assert builder.exists(artifact) is False

    def test_it_deletes_by_id_never_by_the_name(self) -> None:
        """A name is reused once its snapshot is deleted (E0-04), so deleting
        by name could remove a later build's snapshot that inherited it."""
        service = FakeSnapshotService(get_results={"snp-123": FakeSnapshot(id="snp-123")})
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        builder.delete(an_artifact(provider_artifact_id="snp-123", mutable_alias="dl-geo-v3"))
        assert [call.args for call in service.delete_calls] == [("snp-123",)]

    def test_deleting_what_is_already_gone_is_a_success(self) -> None:
        """The collector deletes first and marks second (E1-17): a sweep
        that died in between deletes again, and must not be refused for it."""
        service = FakeSnapshotService(get_results={"snp-123": FakeSnapshot(id="snp-123")})
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        artifact = an_artifact(provider_artifact_id="snp-123")
        builder.delete(artifact)
        builder.delete(artifact)
        assert len(service.delete_calls) == 2

    def test_any_other_failure_is_a_provider_error_and_not_a_success(self) -> None:
        service = FakeSnapshotService(
            get_results={"snp-123": FakeSnapshot(id="snp-123")},
            delete_errors={"snp-123": RuntimeError("the snapshot is in use")},
        )
        builder = a_builder(
            daytona=FakeDaytonaModule(client=FakeDaytonaClient(snapshot_service=service))
        )
        with pytest.raises(EnvironmentsError) as raised:
            builder.delete(an_artifact(provider_artifact_id="snp-123"))
        assert raised.value.code.code == PROVIDER_ERROR.code
        assert "in use" in raised.value.message


class TestSmokeTestingASnapshot:
    """E2-04's own `Done when`: a sandbox launched from its id passes the core tier.

    It refused through `ManagedBuilder` until 2026-09-17, and the build
    workflow calls this step — so no Daytona build could reach `succeeded`:
    the snapshot was built and live at the provider, and the build was
    recorded failed.
    """

    def _environment(self):
        return parse_environment(a_request().environment.model_dump(by_alias=True))

    def test_it_launches_by_id_and_runs_the_core_tier(self, monkeypatch) -> None:
        """A Daytona sandbox record keeps the snapshot's *name*, and a name is
        republished, so only the id says which artifact ran (E0-04)."""
        made: dict = {}
        ran: dict = {}

        class FakeSandbox:
            def __init__(self, **kwargs):
                made.update(kwargs)
                self.events: list[str] = []

            def start(self):
                self.events.append("start")

            def stop(self):
                self.events.append("stop")

        builder = a_builder()
        monkeypatch.setattr(
            "code_sandboxes.daytona_sandbox.DaytonaSandbox", FakeSandbox, raising=False
        )
        monkeypatch.setattr(
            "code_sandboxes.environments.conformance.run_core_tier",
            lambda sandbox, **kwargs: ran.update(kwargs) or "the-result",
        )

        answer = builder.smoke_test(
            an_artifact(
                variant="daytona",
                immutable_reference="snap-1",
                provider_artifact_id="snap-1",
            ),
            environment=self._environment(),
            lock_text="",
        )
        assert answer == "the-result"
        assert made["snapshot"] == "snap-1"
        # A smoke test that leaves a sandbox running bills the owner for a check.
        assert made["delete_on_stop"] is True
        assert "restart" in ran and ran["python_version"]

    def test_a_gpu_version_also_passes_check_eleven(self, monkeypatch) -> None:
        """E2-17: the core tier alone passes on a machine with no GPU, so a GPU
        version's smoke test adds check 11, gating, with the spec's CUDA and count."""
        from code_sandboxes.environments.builders import CheckResult, ValidationResult

        asked: dict = {}

        class FakeSandbox:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

        monkeypatch.setattr(
            "code_sandboxes.daytona_sandbox.DaytonaSandbox", FakeSandbox, raising=False
        )
        monkeypatch.setattr(
            "code_sandboxes.environments.conformance.run_core_tier",
            lambda sandbox, **kwargs: ValidationResult(contract_version="sandbox-contract/v1"),
        )
        monkeypatch.setattr(
            "code_sandboxes.environments.conformance.run_accelerator_check",
            lambda sandbox, **kwargs: asked.update(kwargs)
            or CheckResult(id="conformance:11", name="gpu", passed=False, gating=True),
        )
        request = TestAGpuSnapshot().gpu_request()
        answer = a_builder().smoke_test(
            an_artifact(
                variant="daytona", immutable_reference="snap-gpu", provider_artifact_id="snap-gpu"
            ),
            environment=request.environment,
            lock_text="",
        )
        assert asked == {"cuda": "12.8", "count": 2}
        assert [check.id for check in answer.checks] == ["conformance:11"]
        assert answer.passed is False

    def test_the_sandbox_is_deleted_even_when_the_tier_raises(self, monkeypatch) -> None:
        events: list[str] = []

        class FakeSandbox:
            def __init__(self, **_kwargs):
                pass

            def start(self):
                events.append("start")

            def stop(self):
                events.append("stop")

        monkeypatch.setattr(
            "code_sandboxes.daytona_sandbox.DaytonaSandbox", FakeSandbox, raising=False
        )
        monkeypatch.setattr(
            "code_sandboxes.environments.conformance.run_core_tier",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("the tier blew up")),
        )
        with pytest.raises(EnvironmentsError):
            a_builder().smoke_test(
                an_artifact(
                    variant="daytona", immutable_reference="snap-1", provider_artifact_id="snap-1"
                ),
                environment=self._environment(),
                lock_text="",
            )
        assert events == ["start", "stop"]

    def test_without_a_spec_it_says_what_it_needs(self) -> None:
        """The core tier asks for the Python version and the pinned packages,
        and an artifact carries neither."""
        with pytest.raises(EnvironmentsError) as raised:
            a_builder().smoke_test(
                an_artifact(
                    variant="daytona", immutable_reference="snap-1", provider_artifact_id="snap-1"
                )
            )
        assert raised.value.code is CAPABILITY_UNSUPPORTED
        assert "needs the version's spec" in str(raised.value)
