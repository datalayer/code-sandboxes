# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Modal builder: an image from the lock, and its id kept (E2-05).

Nothing here reaches Modal: the `modal` SDK is a double that records what it
was asked, the way `test_environment_daytona_builder.py` records the Daytona
SDK's calls without a real snapshot. What is checked is what the image
chain *is* — pulled through a Secret made for this build, `env` before any
install, apt packages from the lock, no `USER` line (Modal ignores it), no
doctor or wheelhouse copied (the Datalayer base already bakes both), the
pinned image builder version, an entrypoint that execs its arguments, and
the secret cleaned up afterward through the SDK's own background loop.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest

from code_sandboxes.environments.adapters.modal import Builder
from code_sandboxes.environments.builders import ArtifactReference, BuildRequest
from code_sandboxes.environments.errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    BUILD_SECRET_UNAVAILABLE,
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
    "# datalayer-protected: ipykernel==7.3.0\n"
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

    provider_secrets: ClassVar[dict[str, str]] = {
        "MODAL_TOKEN_ID": "owners-modal-token-id",
        "MODAL_TOKEN_SECRET": "owners-modal-token-secret",
    }
    username: ClassVar[str] = "AKIA-owners-access-key"
    password: ClassVar[str] = "owners-secret-key"


class NoRegistryCredential:
    """A credential with the owner's own token, but no base-reader login (D-18)."""

    provider_secrets: ClassVar[dict[str, str]] = {
        "MODAL_TOKEN_ID": "owners-modal-token-id",
        "MODAL_TOKEN_SECRET": "owners-modal-token-secret",
    }


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


class FakeNotFoundError(Exception):
    """Stands in for `modal.exception.NotFoundError`."""


class FakeImage:
    """The image chain: every call recorded, in order, on one list."""

    def __init__(self, calls: list[Call], *, build_error: Exception | None = None) -> None:
        self.calls = calls
        self.object_id: str | None = None
        self.published: list[str] = []
        self._build_error = build_error

    def env(self, env_vars: dict[str, str]) -> FakeImage:
        self.calls.append(Call("env", (env_vars,)))
        return self

    def apt_install(self, *packages: str) -> FakeImage:
        self.calls.append(Call("apt_install", packages))
        return self

    def add_local_file(self, local_path: str, remote_path: str, *, copy: bool = False) -> FakeImage:
        content = Path(local_path).read_bytes()
        self.calls.append(
            Call("add_local_file", (local_path, remote_path), {"copy": copy, "content": content})
        )
        return self

    def run_commands(self, *commands: str, secrets: Any = None) -> FakeImage:
        kwargs = {} if secrets is None else {"secrets": list(secrets)}
        self.calls.append(Call("run_commands", commands, kwargs))
        return self

    def micromamba_install(self, *, spec_file: str) -> FakeImage:
        self.calls.append(Call("micromamba_install", (), {"spec_file": spec_file}))
        return self

    def pip_install(self, *packages: str, find_links: str | None = None) -> FakeImage:
        self.calls.append(Call("pip_install", packages, {"find_links": find_links}))
        return self

    def workdir(self, path: str) -> FakeImage:
        self.calls.append(Call("workdir", (path,)))
        return self

    def entrypoint(self, commands: list[str]) -> FakeImage:
        self.calls.append(Call("entrypoint", (commands,)))
        return self

    def build(self, app: Any) -> FakeImage:
        self.calls.append(Call("build", (app,)))
        # Simulates Modal's own `enable_output()` printing straight to
        # stdout, so the adapter's `contextlib.redirect_stdout` capture is
        # exercised the same way it is against the real SDK.
        print(f"Built image im-built-{id(self):x}")  # noqa: T201
        if self._build_error:
            raise self._build_error
        self.object_id = f"im-built-{id(self):x}"
        return self

    def publish(self, name: str) -> None:
        self.published.append(name)


class FakeImageFactory:
    """Stands in for `modal.Image`: its two entry points, `from_aws_ecr` and `from_id`."""

    def __init__(
        self,
        *,
        build_error: Exception | None = None,
        existing_ids: set[str] | None = None,
        get_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.from_aws_ecr_calls: list[Call] = []
        self.from_id_calls: list[Call] = []
        #: Every image `from_aws_ecr` created, in call order.
        self.created: list[FakeImage] = []
        self._build_error = build_error
        self._existing_ids = existing_ids if existing_ids is not None else set()
        self._get_errors = get_errors or {}

    def from_aws_ecr(self, tag: str, secret: Any = None, **kwargs: Any) -> FakeImage:
        self.from_aws_ecr_calls.append(Call("from_aws_ecr", (tag,), {"secret": secret, **kwargs}))
        image = FakeImage([Call("from_aws_ecr", (tag,))], build_error=self._build_error)
        self.created.append(image)
        return image

    def from_id(self, image_id: str, *, client: Any = None) -> FakeImage:
        self.from_id_calls.append(Call("from_id", (image_id,), {"client": client}))
        if image_id in self._get_errors:
            raise self._get_errors[image_id]
        if image_id not in self._existing_ids:
            raise FakeNotFoundError(f"no such image {image_id}")
        found = FakeImage([])
        found.object_id = image_id
        return found


class FakeSecret:
    def __init__(
        self,
        env_dict: dict[str, str],
        *,
        object_id: str = "st-fake123",
        hydrate_error: Exception | None = None,
    ) -> None:
        self.env_dict = env_dict
        self.object_id = object_id
        self.hydrate_calls: list[Any] = []
        self._hydrate_error = hydrate_error

    def hydrate(self, *, client: Any = None) -> None:
        self.hydrate_calls.append(client)
        if self._hydrate_error:
            raise self._hydrate_error


class FakeSecretFactory:
    """Stands in for `modal.Secret`."""

    def __init__(self, *, hydrate_error: Exception | None = None) -> None:
        self.from_dict_calls: list[FakeSecret] = []
        self._hydrate_error = hydrate_error

    def from_dict(self, env_dict: dict[str, str]) -> FakeSecret:
        secret = FakeSecret(env_dict, hydrate_error=self._hydrate_error)
        self.from_dict_calls.append(secret)
        return secret


class FakeStub:
    """Stands in for the raw gRPC stub `client.stub` — its own method names
    mirror the real RPCs, so they are assigned as attributes rather than
    `def`-ined, the same way `test_environment_daytona_builder.py` handles
    the same clash for `DockerRegistryApi`."""

    def __init__(self, *, delete_error: Exception | None = None) -> None:
        self.secret_delete_calls: list[Any] = []
        self._delete_error = delete_error
        self.SecretDelete = self._secret_delete

    async def _secret_delete(self, request: Any) -> None:
        self.secret_delete_calls.append(request)
        if self._delete_error:
            raise self._delete_error


class FakeClient:
    def __init__(self, *, delete_error: Exception | None = None) -> None:
        self.stub = FakeStub(delete_error=delete_error)


class FakeClientFactory:
    """Stands in for `modal.Client`: its two entry points."""

    def __init__(self, client: FakeClient, *, error: Exception | None = None) -> None:
        self._client = client
        self._error = error
        self.from_credentials_calls: list[tuple[str, str]] = []
        self.from_env_calls = 0

    def from_credentials(self, token_id: str, token_secret: str) -> FakeClient:
        self.from_credentials_calls.append((token_id, token_secret))
        if self._error:
            raise self._error
        return self._client

    def from_env(self) -> FakeClient:
        self.from_env_calls += 1
        if self._error:
            raise self._error
        return self._client


class FakeAppFactory:
    """Stands in for `modal.App`."""

    def __init__(self, app: Any) -> None:
        self._app = app
        self.lookup_calls: list[Call] = []

    def lookup(
        self,
        name: str,
        *,
        client: Any = None,
        environment_name: str | None = None,
        create_if_missing: bool = False,
    ) -> Any:
        self.lookup_calls.append(
            Call("lookup", (name,), {"client": client, "create_if_missing": create_if_missing})
        )
        return self._app


def _fake_enable_output() -> Any:
    import contextlib

    @contextlib.contextmanager
    def _cm() -> Any:
        yield

    return _cm()


class FakeModalModule:
    """Stands in for `import modal`."""

    def __init__(
        self,
        *,
        client: FakeClient | None = None,
        client_error: Exception | None = None,
        build_error: Exception | None = None,
        secret_hydrate_error: Exception | None = None,
        existing_ids: set[str] | None = None,
        get_errors: dict[str, Exception] | None = None,
    ) -> None:
        self.client = client or FakeClient()
        self.app = object()
        self.Client = FakeClientFactory(self.client, error=client_error)
        self.App = FakeAppFactory(self.app)
        self.Secret = FakeSecretFactory(hydrate_error=secret_hydrate_error)
        self.Image = FakeImageFactory(
            build_error=build_error, existing_ids=existing_ids, get_errors=get_errors
        )
        self.exception = type("_Exc", (), {"NotFoundError": FakeNotFoundError})()
        self.enable_output = _fake_enable_output


class FakeApiPb2:
    """Stands in for `modal_proto.api_pb2`: `SecretDeleteRequest` mirrors the
    real message type's own name, assigned rather than `def`-ined for the
    same reason as `FakeStub.SecretDelete`."""

    def __init__(self) -> None:
        self.SecretDeleteRequest = self._secret_delete_request

    def _secret_delete_request(self, *, secret_id: str) -> Call:
        return Call("SecretDeleteRequest", (), {"secret_id": secret_id})


class FakeSynchronizer:
    def wrap(self, fn: Any) -> Any:
        def _run(*args: Any, **kwargs: Any) -> Any:
            return asyncio.run(fn(*args, **kwargs))

        return _run


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
            "compatibility": {"variants": {"required": ["modal"]}},
            **changes.pop("spec", {}),
        },
    }
    fields = {
        "environment_uid": "01k0env0000000000000000000",
        "version": 3,
        "build_uid": "bld-1",
        "owner_uid": OWNER,
        "variant": "modal",
        "environment": parse_environment(spec),
        "lock_text": LOCK,
        "lock_digest": LOCK_DIGEST,
        "resolved_base": BASE,
        "region": None,
        "size_class": "medium",
    }
    fields.update(changes)
    return BuildRequest(**fields)


def a_builder(*, modal: FakeModalModule | None = None, **changes: Any) -> Builder:
    options: dict[str, Any] = {
        "modal_sdk": lambda: modal or FakeModalModule(),
        "modal_internals": lambda: (FakeSynchronizer(), FakeApiPb2()),
        "credential": Credential(),
    }
    options.update(changes)
    return Builder(**options)


def an_artifact(**changes: Any) -> ArtifactReference:
    fields = {
        "variant": "modal",
        "immutable_reference": "im-abc123",
        "provider_artifact_id": "im-abc123",
        "size_class": "medium",
        "mutable_alias": "dl-geospatial-analysis-v3-bld-1",
        "contract_version": "sandbox-contract/v1",
    }
    fields.update(changes)
    return ArtifactReference(**fields)


def calls_named(image: FakeImage, name: str) -> list[Call]:
    return [call for call in image.calls if call.name == name]


def run_commands_containing(image: FakeImage, needle: str) -> list[Call]:
    """`run_commands` takes several commands in one call (`build()` passes
    the `pip install uv` and `uv pip sync` lines together): search every arg
    of every such call, not just the first."""
    return [
        call
        for call in calls_named(image, "run_commands")
        if any(needle in str(command) for command in call.args)
    ]


@pytest.fixture(autouse=True)
def _restore_image_builder_version() -> Any:
    """`build()` mutates the process environment (see the module docstring)."""
    previous = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    yield
    if previous is None:
        os.environ.pop("MODAL_IMAGE_BUILDER_VERSION", None)
    else:
        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = previous


class TestBuildingAnImage:
    def test_it_pulls_the_base_through_ecr(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [call] = modal.Image.from_aws_ecr_calls
        assert call.args[0] == BASE

    def test_the_ecr_secret_carries_the_credential(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [secret] = modal.Secret.from_dict_calls
        assert secret.env_dict["AWS_ACCESS_KEY_ID"] == Credential.username
        assert secret.env_dict["AWS_SECRET_ACCESS_KEY"] == Credential.password
        assert secret.env_dict["AWS_REGION"] == "us-east-1"

    def test_no_secret_with_no_pull_credential(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal, credential=NoRegistryCredential()).build(a_request())
        assert modal.Secret.from_dict_calls == []
        [call] = modal.Image.from_aws_ecr_calls
        assert call.kwargs["secret"] is None

    def test_apt_packages_from_the_lock_are_installed(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request(lock_text=APT_LOCK))
        [image] = modal.Image.created
        [install] = calls_named(image, "apt_install")
        assert "gdal-bin=3.8.4+dfsg-3build2" in install.args

    def test_no_apt_step_when_the_lock_pins_none(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        assert calls_named(image, "apt_install") == []

    def test_env_is_set_before_apt_and_uv(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request(lock_text=APT_LOCK))
        [image] = modal.Image.created
        names = [call.name for call in image.calls]
        assert names.index("env") < names.index("apt_install")
        sync_at = next(
            i
            for i, call in enumerate(image.calls)
            if call.name == "run_commands" and any("uv pip sync" in str(arg) for arg in call.args)
        )
        assert names.index("env") < sync_at

    def test_neither_the_doctor_nor_the_wheelhouse_is_copied(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        # The only two files baked in are the lock and the entrypoint
        # script — neither the doctor nor the wheelhouse.
        copies = calls_named(image, "add_local_file")
        assert {call.args[1] for call in copies} == {
            "/opt/datalayer/lock.txt",
            "/opt/datalayer/bin/entrypoint.sh",
        }
        [lock_copy] = [call for call in copies if call.args[1] == "/opt/datalayer/lock.txt"]
        assert lock_copy.kwargs["copy"] is True

    def test_the_lock_copied_in_is_the_requests_own(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request(lock_text=LOCK))
        [image] = modal.Image.created
        [lock_copy] = [
            call
            for call in calls_named(image, "add_local_file")
            if call.args[1] == "/opt/datalayer/lock.txt"
        ]
        assert lock_copy.kwargs["content"] == LOCK.encode("utf-8")

    def test_uv_pip_sync_reaches_the_bases_own_shared_wheelhouse(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        [sync_call] = run_commands_containing(image, "uv pip sync")
        [sync] = [arg for arg in sync_call.args if "uv pip sync" in arg]
        assert "--require-hashes" in sync
        assert "--find-links /opt/datalayer/wheelhouse" in sync

    def test_a_conda_lock_installs_with_micromamba_and_then_the_pip_pins(self) -> None:
        """A conda source (E3-02): Modal's own `micromamba_install` reads the
        `@EXPLICIT` lock, and `pip_install` layers the protected pip pins the
        resolver forced — never the pip-lock `uv pip sync`."""
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request(lock_text=CONDA_LOCK, spec=CONDA_SPEC))
        [image] = modal.Image.created
        [mamba] = calls_named(image, "micromamba_install")
        assert mamba.kwargs["spec_file"] == "/opt/datalayer/lock.txt"
        [pip] = calls_named(image, "pip_install")
        assert "ipykernel==7.3.0" in pip.args
        mamba_at = image.calls.index(mamba)
        pip_at = image.calls.index(pip)
        assert mamba_at < pip_at
        assert not run_commands_containing(image, "uv pip sync")

    def test_no_user_line_is_ever_emitted(self) -> None:
        """Modal ignores `USER` entirely (found live): writing one would be
        dead code, so this builder never calls `dockerfile_commands` at all."""
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        assert all("USER" not in str(call.args) for call in image.calls)

    def test_the_doctor_check_is_not_run_at_build_time(self) -> None:
        """A build-time check would run as root, not `1000:100` (see the
        module docstring): it belongs to a live launch instead."""
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        assert not any("doctor" in call.args[0] for call in calls_named(image, "run_commands"))

    def test_the_chain_ends_with_workdir_then_entrypoint(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        # `build` is appended by `FakeImage.build` itself; the two before it
        # are the chain's own last words.
        names = [call.name for call in image.calls]
        assert names[-3:] == ["workdir", "entrypoint", "build"]

    def test_the_entrypoint_execs_its_arguments(self) -> None:
        """A bare script path, nothing for `entrypoint()`'s own Dockerfile
        rendering to mis-escape (found in review of the first version of
        this code, which used an inline, unquoted `$0 $@` forwarder)."""
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        [entrypoint_call] = calls_named(image, "entrypoint")
        assert entrypoint_call.args[0] == ["/opt/datalayer/bin/entrypoint.sh"]

    def test_the_entrypoint_script_itself_quotes_its_forwarding(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        [script_copy] = [
            call
            for call in calls_named(image, "add_local_file")
            if call.args[1] == "/opt/datalayer/bin/entrypoint.sh"
        ]
        assert script_copy.kwargs["content"] == (
            b'#!/bin/sh\nif [ "$#" -eq 0 ]; then exec sleep infinity; fi\nexec "$@"\n'
        )
        assert any(
            "chmod +x /opt/datalayer/bin/entrypoint.sh" in str(call.args)
            for call in calls_named(image, "run_commands")
        )

    def test_the_entrypoint_stays_alive_with_no_command_at_all(self) -> None:
        """`ModalSandbox.start()` — the actual launcher — creates the
        sandbox with no command args and execs into it separately; found
        live, 2026-09-13: `exec "$@"` alone is a no-op with nothing to
        expand, so the container exited before that first real exec
        arrived."""
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        [script_copy] = [
            call
            for call in calls_named(image, "add_local_file")
            if call.args[1] == "/opt/datalayer/bin/entrypoint.sh"
        ]
        script = script_copy.kwargs["content"].decode("utf-8")
        assert "sleep infinity" in script

    def test_the_image_builder_version_is_pinned(self) -> None:
        modal = FakeModalModule()
        os.environ.pop("MODAL_IMAGE_BUILDER_VERSION", None)
        a_builder(modal=modal).build(a_request())
        assert os.environ["MODAL_IMAGE_BUILDER_VERSION"] == "2025.06"

    def test_the_app_is_looked_up_by_environment_name(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [lookup] = modal.App.lookup_calls
        assert lookup.args[0] == "dl-geospatial-analysis"
        assert lookup.kwargs["create_if_missing"] is True

    def test_build_logs_reach_the_log(self) -> None:
        logged: list[str] = []
        modal = FakeModalModule()
        a_builder(modal=modal, log=logged.append).build(a_request())
        assert any("Built image" in line for line in logged)

    def test_a_build_failure_is_reported_with_the_log(self) -> None:
        modal = FakeModalModule(build_error=RuntimeError("aiohttp wheel build failed"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).build(a_request())
        assert raised.value.code.code == BUILD_FAILED.code
        assert "aiohttp wheel build failed" in str(raised.value)
        assert raised.value.detail["log"]

    def test_the_image_is_published_for_operability(self) -> None:
        modal = FakeModalModule()
        artifact = a_builder(modal=modal).build(a_request())
        [image] = modal.Image.created
        assert image.published == [artifact.mutable_alias]

    def test_the_artifact_is_the_image_id(self) -> None:
        modal = FakeModalModule()
        artifact = a_builder(modal=modal).build(a_request())
        assert artifact.variant == "modal"
        assert artifact.immutable_reference.startswith("im-built-")
        assert artifact.provider_artifact_id == artifact.immutable_reference
        assert artifact.mutable_alias == "dl-geospatial-analysis-v3-bld-1"
        assert artifact.size_class == "medium"
        assert artifact.contract_version

    def test_provider_account_is_populated_from_the_credential(self) -> None:
        modal = FakeModalModule()
        artifact = a_builder(modal=modal).build(a_request())
        assert artifact.provider_account and artifact.provider_account.startswith("modal:")

    def test_provider_account_is_none_with_no_credential(self) -> None:
        modal = FakeModalModule()
        artifact = a_builder(modal=modal, credential=None).build(a_request())
        assert artifact.provider_account is None


class TestTheEcrSecretIsCleanedUp:
    def test_the_secret_is_deleted_after_a_successful_build(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal).build(a_request())
        [secret] = modal.Secret.from_dict_calls
        [deleted] = modal.client.stub.secret_delete_calls
        assert deleted.kwargs["secret_id"] == secret.object_id

    def test_the_secret_is_deleted_even_when_the_build_fails(self) -> None:
        modal = FakeModalModule(build_error=RuntimeError("no"))
        with pytest.raises(EnvironmentsError):
            a_builder(modal=modal).build(a_request())
        assert len(modal.client.stub.secret_delete_calls) == 1

    def test_no_secret_to_delete_with_no_pull_credential(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal, credential=NoRegistryCredential()).build(a_request())
        assert modal.client.stub.secret_delete_calls == []

    def test_a_secret_delete_failure_does_not_hide_a_successful_build(self) -> None:
        logged: list[str] = []
        modal = FakeModalModule(client=FakeClient(delete_error=RuntimeError("already gone")))
        artifact = a_builder(modal=modal, log=logged.append).build(a_request())
        assert artifact.immutable_reference
        assert any("Could not delete the Modal secret" in line for line in logged)

    def test_a_partially_hydrated_secret_is_still_cleaned_up_on_failure(self) -> None:
        """`hydrate` is the RPC that creates the app-owned remote secret: if
        it creates the secret and this process then observes an error on
        the same call, the credential must not be left behind just because
        nothing local ever confirmed success (found in review)."""
        modal = FakeModalModule(secret_hydrate_error=RuntimeError("timeout"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).build(a_request())
        assert raised.value.code.code == PROVIDER_ERROR.code
        [secret] = modal.Secret.from_dict_calls
        [deleted] = modal.client.stub.secret_delete_calls
        assert deleted.kwargs["secret_id"] == secret.object_id


class TestWhatModalCannotBuildYet:
    def test_a_gpu_size_class_is_refused_at_build_time_naming_e2_17(self) -> None:
        modal = FakeModalModule()
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).build(a_request(size_class="gpu-large"))
        assert raised.value.code.code == CAPABILITY_UNSUPPORTED.code
        assert raised.value.detail["missing"] == "E2-17"
        # Refused before any provider is touched.
        assert modal.Secret.from_dict_calls == []
        assert modal.Image.from_aws_ecr_calls == []


class TestABuildSecret:
    """E3-05: a secret is attached to the `run_commands` steps that name it."""

    SECRET_ID = "dlsec_01J9BUILDSECRET0000000000"
    VALUE = "tiles-licence-value-8f3a"

    def a_secret_request(self, mount_as: str = "env") -> BuildRequest:
        return a_request(
            spec={
                "buildSecrets": [{"id": self.SECRET_ID, "name": "TILES_KEY", "mountAs": mount_as}],
                "commands": {
                    "postInstall": [
                        "python -c 'import geopandas'",
                        "python unpack_tiles.py --key-env TILES_KEY",
                    ]
                },
            },
            build_secret_ids=(self.SECRET_ID,),
        )

    def resolver(self, asked: list[tuple[str, str]] | None = None) -> Any:
        def resolve_secret(secret: Any, *, owner_uid: str) -> str:
            if asked is not None:
                asked.append((secret.id, owner_uid))
            return self.VALUE

        return resolve_secret

    def test_it_reaches_only_the_command_that_names_it(self) -> None:
        modal = FakeModalModule()
        asked: list[tuple[str, str]] = []
        a_builder(modal=modal, resolve_secret=self.resolver(asked)).build(self.a_secret_request())
        assert asked == [(self.SECRET_ID, OWNER)]
        [build_secret] = [s for s in modal.Secret.from_dict_calls if "TILES_KEY" in s.env_dict]
        assert build_secret.env_dict == {"TILES_KEY": self.VALUE}
        assert build_secret.hydrate_calls == [modal.client]
        [image] = modal.Image.created
        [named] = run_commands_containing(image, "unpack_tiles.py")
        assert named.kwargs == {"secrets": [build_secret]}
        [unnamed] = run_commands_containing(image, "import geopandas")
        assert unnamed.kwargs == {}
        # Never on the install steps either.
        [install] = run_commands_containing(image, "uv pip sync")
        assert install.kwargs == {}

    def test_it_is_deleted_after_the_build_like_the_base_secret(self) -> None:
        modal = FakeModalModule()
        a_builder(modal=modal, resolve_secret=self.resolver()).build(self.a_secret_request())
        assert len(modal.client.stub.secret_delete_calls) == 2

    def test_it_is_deleted_even_when_the_build_fails(self) -> None:
        modal = FakeModalModule(build_error=RuntimeError("step exited 1"))
        with pytest.raises(EnvironmentsError):
            a_builder(modal=modal, resolve_secret=self.resolver()).build(self.a_secret_request())
        assert len(modal.client.stub.secret_delete_calls) == 2

    def test_a_failed_build_never_carries_its_value(self) -> None:
        modal = FakeModalModule(build_error=RuntimeError(f"unpack_tiles.py printed {self.VALUE}"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal, resolve_secret=self.resolver()).build(self.a_secret_request())
        assert self.VALUE not in str(raised.value)
        assert self.VALUE not in repr(raised.value.detail)
        assert raised.value.__cause__ is None

    def test_one_iam_will_not_give_stops_the_build_before_modal(self) -> None:
        modal = FakeModalModule()

        def refused(secret: Any, *, owner_uid: str) -> str:
            raise EnvironmentsError(BUILD_SECRET_UNAVAILABLE, "IAM refused")

        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal, resolve_secret=refused).build(self.a_secret_request())
        assert raised.value.code is BUILD_SECRET_UNAVAILABLE
        assert modal.Secret.from_dict_calls == []
        assert modal.Image.from_aws_ecr_calls == []

    def test_a_file_mounted_one_is_refused_before_anything_is_queued(self) -> None:
        report = a_builder().validate(self.a_secret_request("file").environment)
        assert report.supported is False
        assert "spec.buildSecrets[0].mountAs" in [finding.field for finding in report.findings]

    def test_an_env_one_is_buildable(self) -> None:
        report = a_builder().validate(self.a_secret_request().environment)
        assert "spec.buildSecrets" not in " ".join(finding.field for finding in report.findings)

    def test_a_file_mounted_one_is_refused_at_build_time_too(self) -> None:
        modal = FakeModalModule()
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal, resolve_secret=self.resolver()).build(
                self.a_secret_request("file")
            )
        assert raised.value.code.code == CAPABILITY_UNSUPPORTED.code
        assert modal.Secret.from_dict_calls == []


class TestReadingTheRegistry:
    def test_inspect_reads_the_image_by_id(self) -> None:
        modal = FakeModalModule(existing_ids={"im-abc123"})
        metadata = a_builder(modal=modal).inspect(an_artifact(provider_artifact_id="im-abc123"))
        assert metadata.labels["image_id"] == "im-abc123"

    def test_inspect_refuses_artifact_missing_once_the_image_is_gone(self) -> None:
        modal = FakeModalModule()
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).inspect(an_artifact(provider_artifact_id="im-gone"))
        assert raised.value.code.code == ARTIFACT_MISSING.code

    def test_inspect_maps_other_failures_to_a_provider_error(self) -> None:
        modal = FakeModalModule(get_errors={"im-abc123": RuntimeError("timeout")})
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).inspect(an_artifact(provider_artifact_id="im-abc123"))
        assert raised.value.code.code == PROVIDER_ERROR.code

    def test_exists_true_when_the_image_is_there(self) -> None:
        modal = FakeModalModule(existing_ids={"im-abc123"})
        assert a_builder(modal=modal).exists(an_artifact(provider_artifact_id="im-abc123")) is True

    def test_exists_false_once_the_image_is_gone(self) -> None:
        modal = FakeModalModule()
        assert a_builder(modal=modal).exists(an_artifact(provider_artifact_id="im-gone")) is False

    def test_exists_maps_other_failures_to_a_provider_error(self) -> None:
        modal = FakeModalModule(get_errors={"im-abc123": RuntimeError("timeout")})
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).exists(an_artifact(provider_artifact_id="im-abc123"))
        assert raised.value.code.code == PROVIDER_ERROR.code

    def test_inspect_and_exists_use_the_owners_credential_too(self) -> None:
        modal = FakeModalModule(existing_ids={"im-abc123"})
        a_builder(modal=modal).inspect(an_artifact(provider_artifact_id="im-abc123"))
        assert modal.Client.from_credentials_calls == [
            ("owners-modal-token-id", "owners-modal-token-secret")
        ]

    def test_the_client_is_built_once_and_reused(self) -> None:
        modal = FakeModalModule(existing_ids={"im-abc123"})
        builder = a_builder(modal=modal)
        builder.exists(an_artifact(provider_artifact_id="im-abc123"))
        builder.inspect(an_artifact(provider_artifact_id="im-abc123"))
        assert len(modal.Client.from_credentials_calls) == 1

    def test_an_authentication_failure_is_a_provider_error_not_a_raw_exception(self) -> None:
        """`_client` is called before `build`/`inspect`/`exists`'s own `try`
        (found in review): an auth failure used to escape unmapped."""
        modal = FakeModalModule(client_error=RuntimeError("bad token"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(modal=modal).exists(an_artifact(provider_artifact_id="im-abc123"))
        assert raised.value.code.code == PROVIDER_ERROR.code
