# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Contents attachment contract, kept by every provider adapter.

One suite, four providers. What the Contents service may ask of a sandbox —
a volume, the shared filesystem, a bucket, files to fetch, or nothing but a
credential — is the same list everywhere, and what each provider answers has
to be honest about what it can do: Datalayer's Operator makes the mounts
before the pod exists; Daytona, E2B and Modal mount a volume of their own only
when the sandbox is created and cannot mount the shared filesystem at all;
nobody mounts a bucket without a driver on the node.

The fakes below are not mocks answering canned strings. Each one runs the
code the adapter sends it, in this process, in the shape its SDK would answer
in — so a materialized file really is written and really is checked, and a
manifest really lands in a directory, under a `HOME` this suite owns.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import traceback
import types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from code_sandboxes import contents
from code_sandboxes.base import Sandbox
from code_sandboxes.client import CodeSandboxClient
from code_sandboxes.contents import (
    BRIDGE_CONNECT_FAILED,
    BRIDGE_NOT_PREPARED,
    CHECKSUM_MISMATCH,
    CREDENTIAL_DELIVERY_UNSUPPORTED,
    DELIVERY_UNSUPPORTED,
    FILESYSTEM_PRIMITIVES,
    FUSE_UNAVAILABLE,
    LOCAL_BRIDGE_MOUNT,
    LOCAL_BRIDGE_NOT_A_MOUNT,
    LOCAL_BRIDGE_UNSUPPORTED,
    MANIFEST_ENV,
    MOUNT_MISSING,
    MOUNT_NEEDS_RESTART,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    URL_ENV,
    ContentAttachmentError,
    ContentAttachmentSpec,
    ContentManifest,
    ready,
)
from code_sandboxes.datalayer_sandbox import DatalayerSandbox
from code_sandboxes.daytona_sandbox import DaytonaSandbox
from code_sandboxes.e2b_sandbox import E2BSandbox
from code_sandboxes.modal_sandbox import ModalSandbox
from code_sandboxes.models import SandboxConfig

PROVIDERS = ("datalayer", "daytona", "e2b", "modal")
#: The providers whose SDK mounts a volume of its own, and only at creation.
NATIVE_VOLUME_PROVIDERS = ("daytona", "e2b", "modal")

TOKEN = "short-lived-sandbox-token"
CONTENTS_URL = "https://contents.example.test"


# --- Running the code for real -------------------------------------------


class _Kernel:
    """One namespace that executes what an adapter sends, here."""

    def __init__(self) -> None:
        self.namespace: dict = {"__name__": "__main__"}
        self.snippets: list[str] = []

    def run(self, code: str) -> SimpleNamespace:
        self.snippets.append(code)
        out, err, error = io.StringIO(), io.StringIO(), None
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<sandbox>", "exec"), self.namespace)  # noqa: S102
        except BaseException as raised:  # the fakes report, they never raise
            error = SimpleNamespace(
                name=type(raised).__name__, value=str(raised), traceback=traceback.format_exc()
            )
        return SimpleNamespace(stdout=out.getvalue(), stderr=err.getvalue(), error=error)


class _Volumes:
    """The volumes a provider has, and what it mounted where."""

    def __init__(self) -> None:
        self.ids: set[str] = {"vol-a"}
        self.mounted: list[tuple[str, str]] = []

    def exists(self, volume_id: str) -> bool:
        return volume_id in self.ids

    def mount(self, mount_path: str, volume_id: str) -> None:
        """What the provider does at creation: the path appears."""
        if volume_id not in self.ids:
            raise LookupError(f"no volume {volume_id!r}")
        Path(mount_path).mkdir(parents=True, exist_ok=True)
        self.mounted.append((mount_path, volume_id))


class _Served:
    """The signed URLs a manifest carries, answered here instead of by S3."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fetches: list[str] = []

    def serve(self, name: str, payload: bytes) -> str:
        url = f"https://signed.example.test/{name}?X-Amz-Signature=abc"
        self.files[url] = payload
        return url

    def urlopen(self, url, timeout=None):
        del timeout
        self.fetches.append(url)
        if url not in self.files:
            raise urllib.error.URLError(f"no such object: {url}")
        return io.BytesIO(self.files[url])


# --- The providers -----------------------------------------------------------


class _Harness:
    """A provider, wired to a fake of its SDK so that `start()` works."""

    provider: str
    native_volumes: bool
    bucket_code: str

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.volumes = _Volumes()
        self.creation_envs: list[dict] = []
        self.install(monkeypatch)
        self.sandbox: Sandbox = self.make_sandbox()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raise NotImplementedError

    def make_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError

    def rebuild(self, **kwargs) -> Sandbox:
        """A fresh sandbox of the same provider, created with these arguments —
        `features=["fuse"]` for an environment that exposes FUSE."""
        self.sandbox = self.make_sandbox(**kwargs)
        return self.sandbox

    def mounts(self) -> list[tuple[str, str]]:
        """What the RUNNING provider sandbox was created with."""
        raise NotImplementedError

    def path(self, *parts: str) -> str:
        return str(self.root.joinpath(*parts))

    def provision(self, mount_path: str) -> None:
        """What the Datalayer Operator does before the pod starts."""
        Path(mount_path).mkdir(parents=True, exist_ok=True)


class _DatalayerHarness(_Harness):
    provider = "datalayer"
    native_volumes = False
    bucket_code = ""  # the Operator mounts buckets; nothing is refused

    def install(self, monkeypatch):  # noqa: C901 - one fake SDK, in one place
        harness = self

        class Runtime:
            def __init__(self):
                self.kernel = _Kernel()
                self.stopped = False

            def start(self):
                return None

            def stop(self):
                self.stopped = True

            def execute(self, code, timeout=None):
                del timeout
                reply = self.kernel.run(code)
                error = None
                if reply.error is not None:
                    error = {
                        "ename": reply.error.name,
                        "evalue": reply.error.value,
                        "traceback": [reply.error.traceback],
                    }
                return SimpleNamespace(
                    stdout=reply.stdout, stderr=reply.stderr, result=None, error=error
                )

            def get_variable(self, name):
                return self.kernel.namespace[name]

            def set_variable(self, name, value):
                self.kernel.namespace[name] = value

        class AgentClient:
            def __init__(self, api_key=None, urls=None):
                del api_key, urls

            def create_runtime(self, name, environment, time_reservation, snapshot_name=None):
                del name, environment, time_reservation, snapshot_name
                harness.runtimes.append(Runtime())
                return harness.runtimes[-1]

        self.runtimes: list = []
        for name in ("datalayer_core", "datalayer_core.utils", "datalayer_core.utils.urls"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        client_module = types.ModuleType("agent_runtimes.client")
        client_module.AgentClient = AgentClient
        inner = types.ModuleType("agent_runtimes.client.agent_client")
        inner.DEFAULT_TIME_RESERVATION = 10
        monkeypatch.setitem(sys.modules, "agent_runtimes", types.ModuleType("agent_runtimes"))
        monkeypatch.setitem(sys.modules, "agent_runtimes.client", client_module)
        monkeypatch.setitem(sys.modules, "agent_runtimes.client.agent_client", inner)

    def make_sandbox(self, **kwargs):
        return DatalayerSandbox(config=SandboxConfig(timeout=10.0), **kwargs)

    def mounts(self):
        return []


class _DaytonaHarness(_Harness):
    provider = "daytona"
    native_volumes = True
    bucket_code = DELIVERY_UNSUPPORTED

    def install(self, monkeypatch):  # noqa: C901 - one fake SDK, in one place
        harness = self

        class Interpreter:
            def __init__(self):
                self.kernel = _Kernel()

            def create_context(self, cwd=None):
                return SimpleNamespace(id="ctx", cwd=cwd)

            def run_code(self, code, *, context=None, on_stdout=None, on_stderr=None, **_):
                del context
                reply = self.kernel.run(code)
                if on_stdout and reply.stdout:
                    on_stdout(SimpleNamespace(output=reply.stdout))
                if on_stderr and reply.stderr:
                    on_stderr(SimpleNamespace(output=reply.stderr))
                return SimpleNamespace(stdout=reply.stdout, stderr=reply.stderr, error=reply.error)

        class ProviderSandbox:
            def __init__(self, params):
                self.id = "sbx-daytona"
                self.params = params
                self.code_interpreter = Interpreter()
                self.deleted = False
                for mount in getattr(params, "volumes", None) or []:
                    harness.volumes.mount(mount.mount_path, mount.volume_id)
                harness.creation_envs.append(dict(getattr(params, "env_vars", None) or {}))

            def delete(self):
                self.deleted = True

            def stop(self):
                return None

        class Daytona:
            def __init__(self, config=None):
                del config

            def create(self, params):
                return ProviderSandbox(params)

        module = types.ModuleType("daytona")
        module.Daytona = Daytona
        module.DaytonaConfig = lambda **values: SimpleNamespace(**values)
        module.CreateSandboxFromSnapshotParams = lambda **values: SimpleNamespace(**values)
        module.CreateSandboxFromImageParams = lambda **values: SimpleNamespace(**values)
        module.VolumeMount = lambda **values: SimpleNamespace(subpath=None, **values)
        monkeypatch.setitem(sys.modules, "daytona", module)

    def make_sandbox(self, **kwargs):
        return DaytonaSandbox(config=SandboxConfig(timeout=10.0), **kwargs)

    def mounts(self):
        volumes = getattr(self.sandbox._sandbox.params, "volumes", None) or []
        return [(mount.mount_path, mount.volume_id) for mount in volumes]


class _E2BHarness(_Harness):
    provider = "e2b"
    native_volumes = True
    bucket_code = DELIVERY_UNSUPPORTED

    def install(self, monkeypatch):
        harness = self

        class ProviderSandbox:
            def __init__(self, **params):
                self.params = params
                self.sandbox_id = "sbx-e2b"
                self.kernel = _Kernel()
                self.killed = False
                for mount_path, volume_id in params.get("volume_mounts", {}).items():
                    harness.volumes.mount(mount_path, volume_id)
                harness.creation_envs.append(dict(params.get("envs") or {}))

            def run_code(self, code, on_stdout=None, on_stderr=None, **_):
                reply = self.kernel.run(code)
                stdout = reply.stdout.splitlines()
                stderr = reply.stderr.splitlines()
                for line in stdout:
                    if on_stdout:
                        on_stdout(SimpleNamespace(line=line, timestamp=1_700_000_000_000))
                for line in stderr:
                    if on_stderr:
                        on_stderr(SimpleNamespace(line=line, timestamp=1_700_000_000_000))
                return SimpleNamespace(
                    results=[],
                    logs=SimpleNamespace(stdout=stdout, stderr=stderr),
                    error=reply.error,
                )

            def kill(self, **_):
                self.killed = True

        class SandboxFactory:
            @staticmethod
            def create(**params):
                return ProviderSandbox(**params)

        module = types.ModuleType("e2b_code_interpreter")
        module.Sandbox = SandboxFactory
        monkeypatch.setitem(sys.modules, "e2b_code_interpreter", module)

    def make_sandbox(self, **kwargs):
        return E2BSandbox(config=SandboxConfig(timeout=10.0), **kwargs)

    def mounts(self):
        return list(self.sandbox._sandbox.params.get("volume_mounts", {}).items())


class _ModalHarness(_Harness):
    provider = "modal"
    native_volumes = True
    bucket_code = CREDENTIAL_DELIVERY_UNSUPPORTED

    def install(self, monkeypatch):  # noqa: C901 - one fake SDK, in one place
        harness = self

        class Stream:
            def __init__(self, text):
                self._text = text

            def read(self):
                return self._text

        class Process:
            def __init__(self, stdout, stderr, returncode):
                self.stdout = Stream(stdout)
                self.stderr = Stream(stderr)
                self.returncode = returncode

            def wait(self):
                return None

        class ProviderSandbox:
            object_id = "sb-modal"

            def __init__(self, kwargs):
                self.kwargs = kwargs
                self.kernel = _Kernel()
                for mount_path, volume in kwargs.get("volumes", {}).items():
                    harness.volumes.mount(mount_path, volume.name)

            def exec(self, *args, timeout=None):
                del timeout
                if "-u" in args:
                    # The session driver: this fake has none, and the adapter
                    # falls back to one process per snippet, as it would.
                    raise RuntimeError("no session driver here")
                reply = self.kernel.run(args[-1])
                if reply.error is None:
                    return Process(reply.stdout, reply.stderr, 0)
                return Process(reply.stdout, reply.stderr + reply.error.traceback, 1)

            def terminate(self):
                return None

            def detach(self):
                return None

        class Image:
            def pip_install(self, *_):
                return self

        module = types.ModuleType("modal")
        module.App = SimpleNamespace(lookup=lambda name, create_if_missing=False: object())
        module.Image = SimpleNamespace(debian_slim=lambda python_version=None: Image())
        module.Secret = SimpleNamespace(
            from_dict=lambda values: harness.creation_envs.append(dict(values))
        )
        module.Volume = SimpleNamespace(
            from_name=lambda name, **_: SimpleNamespace(name=name),
        )
        module.Sandbox = SimpleNamespace(create=lambda **kwargs: ProviderSandbox(kwargs))
        monkeypatch.setitem(sys.modules, "modal", module)

    def make_sandbox(self, **kwargs):
        return ModalSandbox(config=SandboxConfig(timeout=10.0, max_lifetime=30.0), **kwargs)

    def mounts(self):
        volumes = self.sandbox._sandbox.kwargs.get("volumes", {})
        return [(path, volume.name) for path, volume in volumes.items()]


_HARNESSES = {
    "datalayer": _DatalayerHarness,
    "daytona": _DaytonaHarness,
    "e2b": _E2BHarness,
    "modal": _ModalHarness,
}


@pytest.fixture(params=PROVIDERS)
def harness(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """One provider, in a home of its own, with nothing left in the environment.

    The fakes execute in this process, so the manifest is written under a
    `HOME` that is this test's, the candidate directories are under it too,
    and the variables the install exports are taken back out afterwards.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        contents,
        "MANIFEST_DIRECTORIES",
        (str(tmp_path / "etc" / "datalayer"), str(home / ".datalayer")),
    )
    for name in (MANIFEST_ENV, URL_ENV, TOKEN_ENV, TOKEN_FILE_ENV):
        monkeypatch.delenv(name, raising=False)
    return _HARNESSES[request.param](tmp_path, monkeypatch)


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> _Served:
    store = _Served()
    monkeypatch.setattr(urllib.request, "urlopen", store.urlopen)
    return store


def _attachment(harness: _Harness, uid: str, **fields) -> dict:
    return {
        "uid": uid,
        "source_uid": f"source-{uid}",
        "sandbox_uid": "sandbox-1",
        "sandbox_provider": harness.provider,
        **fields,
    }


def _manifest(harness: _Harness, *attachments: dict, token: str | None = TOKEN) -> ContentManifest:
    return ContentManifest(
        sandbox_uid="sandbox-1",
        sandbox_provider=harness.provider,
        generated_at="2026-08-26T09:00:00Z",
        attachments=list(attachments),
        contents_url=CONTENTS_URL,
        token=token,
    )


def _volume_attachment(harness: _Harness, uid: str = "att-volume", **fields) -> dict:
    """A volume of the provider's own — or, on Datalayer, a mount the Operator made."""
    mount_path = harness.path("mnt", "vol-a")
    if not harness.native_volumes:
        harness.provision(mount_path)
    return _attachment(
        harness,
        uid,
        delivery="mount",
        mount_path=mount_path,
        provider_resource_id="vol-a",
        **fields,
    )


def _materialize_attachment(
    harness: _Harness, served: _Served, uid: str = "att-files", **fields
) -> tuple[dict, Path, bytes]:
    payload = b"a,b\n1,2\n"
    url = served.serve("report.csv", payload)
    mount_path = harness.path("data")
    spec = _attachment(
        harness,
        uid,
        delivery="materialize",
        mount_path=mount_path,
        materialize=[
            {
                "source_url": url,
                "path": "report.csv",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        ],
        **fields,
    )
    return spec, Path(mount_path) / "report.csv", payload


def _client(harness: _Harness) -> CodeSandboxClient:
    return CodeSandboxClient(harness.sandbox, owns_sandbox=True)


# --- Capabilities ------------------------------------------------------------


def test_capabilities_agree_with_the_contents_matrix(harness):
    """What the service's SUPPORT table says, said again by the adapter.

    Datalayer mounts the shared filesystem, a bucket and a person's own
    folder, because the Operator and Clouder's CSI do it on the node. The
    others mount a volume of their own and nothing else: no shared
    filesystem, no bucket without a credential leaving Contents, no bridge.
    """
    capabilities = harness.sandbox.content_capabilities()

    assert capabilities.provider == harness.provider
    assert capabilities.client
    assert capabilities.materialize
    assert capabilities.mount
    assert capabilities.filesystem_primitives == list(FILESYSTEM_PRIMITIVES)
    if harness.provider == "datalayer":
        assert capabilities.bucket_mount
        assert capabilities.local_bridge_mount.supported
        assert capabilities.local_bridge_mount.read_only
        assert capabilities.local_bridge_mount.read_write
    else:
        assert not capabilities.bucket_mount
        assert not capabilities.local_bridge_mount.supported
    # The client answers with the same, before anything is started.
    assert _client(harness).content_capabilities() == capabilities
    assert not harness.sandbox.is_started


# --- The client delivery -----------------------------------------------------


def test_a_client_attachment_is_ready_on_attach(harness):
    client = _client(harness)

    manifest = _manifest(harness, _attachment(harness, "att-client", delivery="client"))

    prepared = client.attach(manifest)

    assert [(item.uid, item.status, item.capabilities) for item in prepared] == [
        ("att-client", "ready", ["client"])
    ]
    assert client.attachment_status("att-client") == prepared[0]
    assert client.is_started


# --- Mounts ------------------------------------------------------------------


def test_the_shared_filesystem_mounts_only_where_there_is_one(harness):
    """A `mount` with no provider volume behind it is the Datalayer shared fs."""
    client = _client(harness)
    mount_path = harness.path("mnt", "shared")
    manifest = _manifest(
        harness, _attachment(harness, "att-shared", delivery="mount", mount_path=mount_path)
    )

    if harness.provider == "datalayer":
        harness.provision(mount_path)
        prepared = client.attach(manifest)
        assert prepared[0].status == "ready"
        assert prepared[0].capabilities == ["mount"]
        assert prepared[0].mount_path == mount_path
        return

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(manifest)

    assert raised.value.uid == "att-shared"
    assert raised.value.code == DELIVERY_UNSUPPORTED
    assert raised.value.attachments[0].status == "failed"
    assert "shared filesystem" in (raised.value.attachments[0].detail or "")
    # The sandbox is left running: whether it is still worth having is the
    # caller's decision.
    assert client.is_started
    assert client.attachment_status("att-shared").status == "failed"


def test_a_mount_the_operator_did_not_make_is_missing(harness):
    if harness.provider != "datalayer":
        pytest.skip("only Datalayer has an Operator making mounts")
    client = _client(harness)
    manifest = _manifest(
        harness,
        _attachment(harness, "att-absent", delivery="mount", mount_path=harness.path("mnt", "no")),
    )

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(manifest)

    assert raised.value.code == MOUNT_MISSING
    assert client.is_started


def test_a_volume_mount_asked_of_a_running_sandbox_needs_a_restart(harness):
    if not harness.native_volumes:
        pytest.skip("Datalayer mounts are the Operator's, made before the pod")
    client = _client(harness)
    client.start()
    manifest = _manifest(harness, _volume_attachment(harness))

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(manifest)

    assert raised.value.code == MOUNT_NEEDS_RESTART
    assert client.is_started
    assert harness.mounts() == []
    # The request is kept: a restart honours it.
    client.restart()
    again = client.reconcile_contents(manifest)
    assert again[0].status == "ready"
    assert harness.mounts() == [(harness.path("mnt", "vol-a"), "vol-a")]


def test_a_volume_mount_asked_before_start_is_made_at_creation(harness):
    client = _client(harness)
    mount_path = harness.path("mnt", "vol-a")

    prepared = client.attach(_manifest(harness, _volume_attachment(harness)))

    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == ["mount"]
    assert prepared[0].provider_resource_id == "vol-a"
    assert Path(mount_path).is_dir()
    if harness.native_volumes:
        assert harness.mounts() == [(mount_path, "vol-a")]


def test_an_optional_attachment_that_cannot_be_honoured_is_degraded(harness):
    """Optional means the sandbox is still fit to run without it."""
    client = _client(harness)
    if harness.native_volumes:
        spec = _attachment(
            harness,
            "att-optional",
            delivery="mount",
            mount_path=harness.path("mnt", "shared"),
            required=False,
        )
        expected = DELIVERY_UNSUPPORTED
    else:
        spec = _attachment(
            harness,
            "att-optional",
            delivery="mount",
            mount_path=harness.path("mnt", "never-made"),
            required=False,
        )
        expected = MOUNT_MISSING

    prepared = client.attach(_manifest(harness, spec))

    assert prepared[0].status == "degraded"
    assert prepared[0].error_code == expected


def test_a_bucket_mount_is_refused_without_a_provider_mechanism(harness):
    """A bucket as a filesystem needs a driver on the node, or a credential
    handed to the provider — and the credential never leaves Contents."""
    client = _client(harness)
    mount_path = harness.path("mnt", "bucket")
    manifest = _manifest(
        harness,
        _attachment(
            harness, "att-bucket", delivery="mount", access_mode="mount", mount_path=mount_path
        ),
    )

    if harness.provider == "datalayer":
        harness.provision(mount_path)
        assert client.attach(manifest)[0].status == "ready"
        return

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(manifest)

    assert raised.value.code == harness.bucket_code
    assert not harness.volumes.mounted


# --- Materializing -----------------------------------------------------------


def test_a_materialize_entry_is_written_and_verified(harness, served):
    client = _client(harness)
    spec, target, payload = _materialize_attachment(harness, served)

    prepared = client.attach(_manifest(harness, spec))

    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == ["materialize"]
    assert target.read_bytes() == payload
    assert not target.with_suffix(".csv.part").exists()
    assert served.fetches == [spec["materialize"][0]["source_url"]]


def test_a_materialize_entry_with_the_wrong_digest_fails_and_leaves_no_file(harness, served):
    client = _client(harness)
    spec, target, _payload = _materialize_attachment(harness, served)
    spec["materialize"][0]["sha256"] = "0" * 64

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == CHECKSUM_MISMATCH
    assert not target.exists()
    assert not target.with_suffix(".csv.part").exists()


def test_a_materialize_entry_that_cannot_be_fetched_is_reported(harness, served):
    client = _client(harness)
    spec, target, _payload = _materialize_attachment(harness, served)
    spec["materialize"][0]["source_url"] = "https://signed.example.test/expired"

    prepared = client.prepare_contents(_manifest(harness, spec))

    assert prepared[0].status == "failed"
    assert prepared[0].error_code == contents.FETCH_FAILED
    assert not target.exists()


# --- Restart and reconcile -----------------------------------------------------


def test_restart_then_reconcile_returns_the_same_set_without_duplicate_mounts(harness, served):
    client = _client(harness)
    files, target, payload = _materialize_attachment(harness, served)
    manifest = _manifest(
        harness,
        _attachment(harness, "att-client", delivery="client"),
        _volume_attachment(harness),
        files,
    )
    first = client.attach(manifest)
    assert [item.status for item in first] == ["ready"] * 3

    client.restart()
    # A fresh sandbox has a fresh filesystem; the fakes run here, so the
    # file the previous one materialized is taken away by hand.
    target.unlink()

    again = client.reconcile_contents(manifest)

    assert [(item.uid, item.status, item.capabilities) for item in again] == [
        (item.uid, item.status, item.capabilities) for item in first
    ]
    assert target.read_bytes() == payload
    assert len(served.fetches) == 2
    if harness.native_volumes:
        mounts = harness.mounts()
        assert mounts == [(harness.path("mnt", "vol-a"), "vol-a")]
        assert len({path for path, _ in mounts}) == len(mounts)

    # And once more, with everything in place: nothing is fetched again.
    third = client.reconcile_contents(manifest)
    assert [item.status for item in third] == ["ready"] * 3
    assert len(served.fetches) == 2
    assert "already present" in (third[2].detail or "")


def test_prepare_fetches_afresh_where_reconcile_leaves_a_good_file_alone(harness, served):
    client = _client(harness)
    files, _target, _payload = _materialize_attachment(harness, served)
    manifest = _manifest(harness, files)

    client.attach(manifest)
    client.prepare_contents(manifest)
    assert len(served.fetches) == 2
    client.reconcile_contents(manifest)
    assert len(served.fetches) == 2


# --- Detaching ---------------------------------------------------------------


def test_detach_removes_materialized_files_and_never_deletes_the_volume(harness, served):
    client = _client(harness)
    files, target, _payload = _materialize_attachment(harness, served)
    volume = _volume_attachment(harness)
    client.attach(_manifest(harness, volume, files))
    assert target.exists()

    client.detach("att-files")
    client.detach("att-volume")

    assert not target.exists()
    assert client.attachment_status("att-files") is None
    assert client.attachment_status("att-volume") is None
    # The volume is the provider's and the mount point is the Operator's:
    # detaching forgets the request and touches neither.
    assert harness.volumes.exists("vol-a")
    assert Path(volume["mount_path"]).exists()
    # Detaching what is already gone is not an error.
    client.detach("att-files")
    if harness.native_volumes:
        client.restart()
        assert harness.mounts() == []


# --- The manifest inside ---------------------------------------------------------


def test_the_manifest_written_into_the_sandbox_carries_no_token(harness):
    client = _client(harness)
    manifest = _manifest(harness, _attachment(harness, "att-client", delivery="client"))

    client.attach(manifest)

    location = client.contents_location
    assert location is not None
    text = Path(location.manifest_path).read_text()
    assert TOKEN not in text
    written = json.loads(text)
    assert "token" not in written
    assert written["contents_url"] == CONTENTS_URL
    assert written["sandbox_provider"] == harness.provider
    assert [item["uid"] for item in written["attachments"]] == ["att-client"]
    # The token has a file of its own that only the owner can read.
    assert location.token_path is not None
    token_file = Path(location.token_path)
    assert token_file.read_text() == TOKEN
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    # And the kernel's environment names all of it.
    assert os.environ[MANIFEST_ENV] == location.manifest_path
    assert os.environ[URL_ENV] == CONTENTS_URL
    assert os.environ[TOKEN_ENV] == TOKEN
    assert os.environ[TOKEN_FILE_ENV] == location.token_path


def test_the_environment_is_given_to_the_provider_at_creation(harness):
    """Daytona, E2B and Modal take their environment when the sandbox is
    made, so a process that is not the kernel's child sees it too."""
    client = _client(harness)
    manifest = _manifest(harness, _attachment(harness, "att-client", delivery="client"))

    client.attach(manifest)

    configured = harness.sandbox.config.env_vars
    assert configured[URL_ENV] == CONTENTS_URL
    assert configured[TOKEN_ENV] == TOKEN
    assert configured[MANIFEST_ENV].endswith("/contents.json")
    assert configured[TOKEN_FILE_ENV].endswith("/contents.token")
    if harness.native_volumes:
        assert harness.creation_envs, "the environment never reached the provider"
        assert harness.creation_envs[-1][URL_ENV] == CONTENTS_URL
        assert harness.creation_envs[-1][TOKEN_ENV] == TOKEN


def test_a_manifest_without_credentials_leaves_no_token_behind(harness):
    client = _client(harness)
    manifest = _manifest(harness, _attachment(harness, "att-client", delivery="client"), token=None)

    client.attach(manifest)

    location = client.contents_location
    assert location.token_path is None
    assert not (Path(location.directory) / "contents.token").exists()
    assert TOKEN_ENV not in os.environ
    assert TOKEN_FILE_ENV not in os.environ
    assert os.environ[MANIFEST_ENV] == location.manifest_path


def test_a_later_manifest_without_credentials_takes_the_token_back(harness):
    """A token the manifest no longer carries is nowhere in the sandbox:
    not in the kernel's environment, not in a file, not in what the provider
    is handed at the next creation."""
    client = _client(harness)
    attachment = _attachment(harness, "att-client", delivery="client")
    client.attach(_manifest(harness, attachment))
    first = client.contents_location
    assert first.token_path is not None and Path(first.token_path).exists()

    client.attach(_manifest(harness, attachment, token=None))

    location = client.contents_location
    assert location.token_path is None
    assert not Path(first.token_path).exists()
    assert TOKEN_ENV not in os.environ
    assert TOKEN_FILE_ENV not in os.environ
    assert os.environ[MANIFEST_ENV] == location.manifest_path
    configured = harness.sandbox.config.env_vars
    assert TOKEN_ENV not in configured
    assert TOKEN_FILE_ENV not in configured
    assert configured[MANIFEST_ENV].endswith("/contents.json")


def test_configuring_contents_replaces_what_an_earlier_manifest_exported(harness):
    sandbox = harness.sandbox
    attachment = _attachment(harness, "att-client", delivery="client")
    sandbox.configure_contents(_manifest(harness, attachment))
    assert sandbox.config.env_vars[TOKEN_ENV] == TOKEN

    sandbox.configure_contents(_manifest(harness, attachment, token=None))

    assert TOKEN_ENV not in sandbox.config.env_vars
    assert TOKEN_FILE_ENV not in sandbox.config.env_vars
    assert sandbox.config.env_vars[URL_ENV] == CONTENTS_URL


def test_detaching_files_under_a_mount_path_keeps_the_mount_request():
    """A mount request is keyed by path; another attachment sharing the
    path is not the mount, and forgetting it must not forget the mount."""
    from code_sandboxes.contents import CreationTimeMounts

    common = {"source_uid": "source", "sandbox_uid": "sandbox-1", "sandbox_provider": "e2b"}
    mounts = CreationTimeMounts()
    volume = ContentAttachmentSpec(
        uid="att-volume",
        delivery="mount",
        mount_path="/data",
        provider_resource_id="vol-a",
        **common,
    )
    files = ContentAttachmentSpec(
        uid="att-files", delivery="materialize", mount_path="/data", **common
    )
    assert mounts.request(volume)

    mounts.forget(files)
    assert mounts.requested == {"/data": "vol-a"}

    mounts.forget(volume)
    assert mounts.requested == {}


def test_a_sandbox_that_has_not_started_cannot_prepare(harness):
    from code_sandboxes.exceptions import SandboxNotStartedError

    with pytest.raises(SandboxNotStartedError):
        harness.sandbox.prepare_contents(
            _manifest(harness, _attachment(harness, "att-client", delivery="client"))
        )


# --- Environment contents: a checkout at a pinned revision, python access ----


REVISION = "5098cee2a638c56c311aca0c18987e407fe127fd"
OTHER_COMMIT = "0000000000000000000000000000000000000000"
GIT_URL = "https://github.com/jakevdp/sklearn_tutorial.git"
#: What the fake `git archive --format=tar <sha>` answers: the digest a
#: RuntimeContent pins is the digest of that stream.
ARCHIVE_DIGEST = hashlib.sha256(b"tar:" + REVISION.encode()).hexdigest()

_FAKE_GIT = '''#!{python}
"""A `git` that does what the checkout snippet asks, here, and writes it down.

`clone --no-checkout` makes an empty `.git`; `fetch` records the sha it was
asked for — or fails when told to; `checkout --detach` writes HEAD, and a
working file; `rev-parse HEAD` reads HEAD back; `archive` streams bytes a
digest can be taken of. FAKE_GIT_HEAD makes the checkout land on another
commit than the one asked for, FAKE_GIT_FAIL=fetch makes the fetch fail.
"""
import os
import pathlib
import sys

args = sys.argv[1:]
with open(os.environ["FAKE_GIT_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
cwd = None
if args[:1] == ["-C"]:
    cwd, args = args[1], args[2:]
command = args[0]
if command == "clone":
    url, target = args[-2], args[-1]
    git = pathlib.Path(target) / ".git"
    git.mkdir(parents=True)
    (git / "url").write_text(url)
elif command == "fetch":
    if os.environ.get("FAKE_GIT_FAIL") == "fetch":
        sys.stderr.write("fatal: could not read from remote repository\\n")
        sys.exit(128)
    (pathlib.Path(cwd) / ".git" / "FETCHED").write_text(args[-1])
elif command == "checkout":
    head = os.environ.get("FAKE_GIT_HEAD") or args[-1]
    (pathlib.Path(cwd) / ".git" / "HEAD").write_text(head)
    (pathlib.Path(cwd) / "README.md").write_text("checkout of " + head)
elif command == "rev-parse":
    sys.stdout.write((pathlib.Path(cwd) / ".git" / "HEAD").read_text() + "\\n")
elif command == "archive":
    sys.stdout.buffer.write(b"tar:" + args[-1].encode())
else:
    sys.stderr.write("fake git: unknown command " + command + "\\n")
    sys.exit(1)
'''


class _FakeGit:
    def __init__(self, log: Path) -> None:
        self.log = log

    def commands(self) -> list[str]:
        if not self.log.exists():
            return []
        return [line for line in self.log.read_text().splitlines() if line]

    def clones(self) -> list[str]:
        return [line for line in self.commands() if line.startswith("clone ")]


@pytest.fixture
def fake_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeGit:
    """A `git` on PATH that the snippet inside the sandbox finds first."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    script = binaries / "git"
    script.write_text(_FAKE_GIT.format(python=sys.executable))
    script.chmod(0o755)
    log = tmp_path / "git.log"
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_GIT_LOG", str(log))
    monkeypatch.delenv("FAKE_GIT_HEAD", raising=False)
    monkeypatch.delenv("FAKE_GIT_FAIL", raising=False)
    return _FakeGit(log)


def _checkout_attachment(
    harness: _Harness, uid: str = "env-git", *, sha256: str | None = ARCHIVE_DIGEST, **fields
) -> tuple[dict, Path]:
    """An Environment's git content, at the path the Environment declared."""
    mount_path = harness.path("home", "tutorials", "sklearn")
    spec = _attachment(
        harness,
        uid,
        delivery="environment",
        mount_path=mount_path,
        provider_resource_id="01M0YX0MXYRP0Q29YWYRE5THZE",
        materialize=[
            {"git_url": GIT_URL, "revision": REVISION, "path": mount_path, "sha256": sha256}
        ],
        **fields,
    )
    return spec, Path(mount_path)


def _bucket_attachment(harness: _Harness, uid: str = "env-s3", **fields) -> tuple[dict, Path]:
    """An Environment's bucket, python access at the path the Environment declared."""
    mount_path = harness.path("home", "datasets")
    spec = _attachment(
        harness,
        uid,
        delivery="environment",
        mount_path=mount_path,
        provider_resource_id="01M0YX0MXYTZK4R5RH9SSXQEG3",
        materialize=[
            {
                "bucket": "datalayer-datasets-prod",
                "region": "us-east-1",
                "prefix": "public/",
                "path": mount_path,
            }
        ],
        **fields,
    )
    return spec, Path(mount_path)


def test_a_git_entry_is_cloned_at_its_pinned_revision_and_detached(harness, fake_git):
    """The declared path holds exactly the pinned commit, checked out detached."""
    client = _client(harness)
    spec, target = _checkout_attachment(harness)

    prepared = client.attach(_manifest(harness, spec))

    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == ["materialize"]
    # The path is the one the Environment declared — unchanged.
    assert prepared[0].mount_path == str(target)
    assert prepared[0].provider_resource_id == "01M0YX0MXYRP0Q29YWYRE5THZE"
    assert (target / ".git" / "HEAD").read_text() == REVISION
    assert (target / "README.md").read_text() == f"checkout of {REVISION}"
    assert not Path(str(target) + ".part").exists()
    commands = fake_git.commands()
    part = str(target) + ".part"
    assert f"clone --no-checkout {GIT_URL} {part}" in commands
    assert f"-C {part} fetch --depth 1 origin {REVISION}" in commands
    assert f"-C {part} checkout --detach {REVISION}" in commands
    assert f"-C {part} rev-parse HEAD" in commands
    assert f"-C {part} archive --format=tar {REVISION}" in commands
    assert "checkout" in (prepared[0].detail or "")


def test_a_checkout_whose_archive_digest_does_not_match_leaves_nothing(harness, fake_git):
    client = _client(harness)
    spec, target = _checkout_attachment(harness, sha256="0" * 64)

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == CHECKSUM_MISMATCH
    assert not target.exists()
    assert not Path(str(target) + ".part").exists()


def test_a_checkout_that_lands_on_another_commit_is_a_failed_fetch(harness, fake_git, monkeypatch):
    """HEAD is verified after the checkout; a sha that is not the pinned one is refused."""
    monkeypatch.setenv("FAKE_GIT_HEAD", OTHER_COMMIT)
    client = _client(harness)
    spec, target = _checkout_attachment(harness, sha256=None)

    prepared = client.prepare_contents(_manifest(harness, spec))

    assert prepared[0].status == "failed"
    assert prepared[0].error_code == contents.FETCH_FAILED
    assert OTHER_COMMIT in (prepared[0].detail or "")
    assert not target.exists()
    assert not Path(str(target) + ".part").exists()


def test_a_fetch_that_fails_is_reported_and_leaves_nothing(harness, fake_git, monkeypatch):
    monkeypatch.setenv("FAKE_GIT_FAIL", "fetch")
    client = _client(harness)
    spec, target = _checkout_attachment(harness, required=False)

    prepared = client.prepare_contents(_manifest(harness, spec))

    assert prepared[0].status == "degraded"
    assert prepared[0].error_code == contents.FETCH_FAILED
    assert "could not read from remote" in (prepared[0].detail or "")
    assert not target.exists()
    assert not Path(str(target) + ".part").exists()


def test_reconcile_does_not_reclone_a_checkout_at_the_right_revision(harness, fake_git):
    client = _client(harness)
    spec, target = _checkout_attachment(harness)
    manifest = _manifest(harness, spec)

    client.attach(manifest)
    assert len(fake_git.clones()) == 1

    again = client.reconcile_contents(manifest)
    assert again[0].status == "ready"
    assert "already present" in (again[0].detail or "")
    assert len(fake_git.clones()) == 1
    assert (target / ".git" / "HEAD").read_text() == REVISION

    # A checkout that drifted is put back at the pinned revision.
    (target / ".git" / "HEAD").write_text(OTHER_COMMIT)
    repaired = client.reconcile_contents(manifest)
    assert repaired[0].status == "ready"
    assert len(fake_git.clones()) == 2
    assert (target / ".git" / "HEAD").read_text() == REVISION

    # Prepare, unlike reconcile, delivers afresh.
    client.prepare_contents(manifest)
    assert len(fake_git.clones()) == 3


def test_a_bucket_entry_is_python_access_and_fetches_nothing(harness, served, fake_git):
    """No bytes, no key: the manifest inside names the bucket for the Contents client."""
    client = _client(harness)
    spec, declared = _bucket_attachment(harness)

    prepared = client.attach(_manifest(harness, spec))

    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == ["python"]
    assert prepared[0].mount_path == str(declared)
    assert "nothing fetched" in (prepared[0].detail or "")
    assert served.fetches == []
    assert fake_git.commands() == []
    assert not declared.exists()
    written = json.loads(Path(client.contents_location.manifest_path).read_text())
    (entry,) = written["attachments"][0]["materialize"]
    assert entry["bucket"] == "datalayer-datasets-prod"
    assert entry["region"] == "us-east-1"
    assert entry["prefix"] == "public/"
    assert entry["path"] == str(declared)
    assert entry["source_url"] is None
    # No credential of the bucket's anywhere in the sandbox.
    assert "AKIA" not in Path(client.contents_location.manifest_path).read_text()


def test_an_environment_attachment_without_entries_is_the_platform_s_mount(harness):
    """On Datalayer the Operator mounted it; elsewhere nothing says how, and it is refused."""
    client = _client(harness)
    mount_path = harness.path("home", "models")
    spec = _attachment(
        harness,
        "env-nfs",
        delivery="environment",
        mount_path=mount_path,
        provider_resource_id="01M0YX0MXYD8YZVMJW016KTB4M",
    )

    if harness.provider == "datalayer":
        harness.provision(mount_path)
        prepared = client.attach(_manifest(harness, spec))
        assert prepared[0].status == "ready"
        assert prepared[0].capabilities == ["mount"]
        return

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))
    assert raised.value.code == DELIVERY_UNSUPPORTED
    assert not harness.volumes.mounted


def test_detach_removes_a_checkout_and_leaves_a_bucket_alone(harness, fake_git):
    client = _client(harness)
    checkout, target = _checkout_attachment(harness)
    bucket, declared = _bucket_attachment(harness)
    client.attach(_manifest(harness, checkout, bucket))
    assert (target / ".git").is_dir()

    client.detach("env-git")
    client.detach("env-s3")

    assert not target.exists()
    assert not declared.exists()
    assert client.attachment_status("env-git") is None
    assert client.attachment_status("env-s3") is None


def test_a_materialize_entry_is_exactly_one_form():
    from pydantic import ValidationError

    from code_sandboxes.contents import MaterializeEntry

    assert MaterializeEntry(source_url="https://x/y", path="/a").form == "file"
    assert MaterializeEntry(git_url=GIT_URL, revision=REVISION, path="/a").form == "git"
    assert MaterializeEntry(bucket="b", region="us-east-1", path="/a").form == "s3"
    with pytest.raises(ValidationError):
        MaterializeEntry(path="/a")
    with pytest.raises(ValidationError):
        MaterializeEntry(source_url="https://x/y", git_url=GIT_URL, revision=REVISION, path="/a")
    with pytest.raises(ValidationError, match="revision"):
        MaterializeEntry(git_url=GIT_URL, path="/a")


# --- Local bridges: a person's own folder, mounted -------------------------------
#
# Supported PER ENVIRONMENT, never per provider. On Daytona, E2B and Modal the
# sandbox mounts the bridge itself, with a FUSE filesystem the adapter starts
# inside it — so only an environment advertising `fuse` can; everywhere else
# the answer is a refusal that offers Synchronize, a copy called a copy. On
# Datalayer the Operator renders a CSI volume and the adapter only looks
# whether it is mounted. On every provider, `ready` stands only if the path
# is a MOUNTPOINT: a copy is never reported as a mount.

BRIDGE_TOKEN = "bridge-mount-token-xyz"  # a fake
RELAY_URL = "wss://relay.example.test/bridges/br-1"


def _bridge_attachment(
    harness: _Harness, uid: str = "att-bridge", mode: str = "ro", **fields
) -> dict:
    mount_path = harness.path("mnt", "laptop")
    return _attachment(
        harness,
        uid,
        delivery="local-bridge",
        mount_path=mount_path,
        mode=mode,
        bridge={
            "bridge_uid": "br-1",
            "relay_url": RELAY_URL,
            "mount_token": BRIDGE_TOKEN,
            "mount_path": mount_path,
            "mode": mode,
        },
        **fields,
    )


class _Launcher:
    """The bridge mount process, faked: the adapter's snippet starts it with
    `subprocess.Popen` inside the sandbox and reads its first line; here the
    fakes run in this process, so `Popen` is this, and the line is what the
    test decided the launcher would say. A `connected` answer makes the mount
    path a mountpoint, as FUSE would."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.answer: dict = {"status": "connected", "mode": "ro"}
        self.launches: list[tuple[list[str], dict]] = []
        self.runs: list[list[str]] = []
        self.mountpoints: set[str] = set()
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda args, **kwargs: self.runs.append(list(args)) or SimpleNamespace(returncode=0),
        )
        monkeypatch.setattr(os.path, "ismount", lambda path: str(path) in self.mountpoints)

    def popen(self, args, **kwargs):
        self.launches.append((list(args), kwargs))
        reader, writer = os.pipe()
        os.write(writer, (json.dumps(self.answer) + "\n").encode("utf-8"))
        os.close(writer)
        if self.answer.get("status") == "connected":
            self.mountpoints.add(args[args.index("--mount-path") + 1])
        return SimpleNamespace(
            stdout=os.fdopen(reader, "rb"), pid=4242, poll=lambda: None, kill=lambda: None
        )

    @property
    def argv(self) -> list[str]:
        return self.launches[-1][0]


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> _Launcher:
    return _Launcher(monkeypatch)


def _with_fuse(harness: _Harness) -> Sandbox:
    """The same provider, in an environment that exposes FUSE."""
    if harness.provider == "datalayer":
        pytest.skip("on Datalayer the Operator mounts the bridge; nothing runs in the sandbox")
    return harness.rebuild(features=["fuse"])


def test_local_bridge_support_is_per_environment_not_per_provider(harness):
    if harness.provider == "datalayer":
        pytest.skip("Datalayer mounts through Clouder's CSI on every environment")
    sandbox_type = type(harness.sandbox)

    # What the provider ships declares no `fuse`: the stock images do not
    # expose /dev/fuse, and nothing is claimed that is not so.
    shipped = sandbox_type.list_environments()
    assert shipped
    assert all(environment.features == [] for environment in shipped)
    assert all("features" in (environment.metadata or {}) for environment in shipped)
    assert not harness.sandbox.content_capabilities().local_bridge_mount.supported

    # An environment that does — a template built for it — is the unit of
    # support, and the sandbox created with it says so.
    bridged = harness.rebuild(features=["fuse"]).content_capabilities().local_bridge_mount
    assert bridged.supported
    assert bridged.required_features == ["fuse"]
    assert bridged.read_only and bridged.read_write
    assert bridged.reconnect and bridged.cleanup
    # By environment name too, when the catalog names it.
    harness.rebuild()
    fused = [
        environment.model_copy(
            update={"metadata": {**(environment.metadata or {}), "features": ["fuse"]}}
        )
        for environment in shipped
    ]
    original = sandbox_type.list_environments
    try:
        sandbox_type.list_environments = classmethod(lambda cls: fused)
        harness.sandbox.config.environment = shipped[0].name
        assert harness.sandbox.content_capabilities().local_bridge_mount.supported
        harness.sandbox.config.environment = "no-such-environment"
        assert not harness.sandbox.content_capabilities().local_bridge_mount.supported
    finally:
        sandbox_type.list_environments = original


def test_the_catalog_carries_each_environment_s_features():
    from code_sandboxes.providers import provider_catalog

    entries = {entry["name"]: entry for entry in provider_catalog({"DAYTONA_API_KEY": "k"})}
    environments = entries["daytona"]["environments"]
    assert environments
    assert all(environment["features"] == [] for environment in environments)


def test_a_local_bridge_is_refused_where_the_environment_lacks_fuse(harness, launcher):
    if harness.provider == "datalayer":
        pytest.skip("Datalayer mounts through Clouder's CSI on every environment")
    client = _client(harness)
    spec = _bridge_attachment(harness)

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == LOCAL_BRIDGE_UNSUPPORTED
    detail = raised.value.attachments[0].detail or ""
    assert harness.provider in detail
    assert harness.sandbox.config.environment in detail
    assert "'fuse'" in detail
    assert "Synchronize" in detail and "datalayer contents sync" in detail
    # Refused before anything ran: no mount process, no copy made in its place.
    assert launcher.launches == []
    assert not Path(spec["mount_path"]).exists()
    # Optional: degraded, the sandbox still fit to run.
    degraded = client.prepare_contents(
        _manifest(harness, _bridge_attachment(harness, "att-opt", required=False))
    )
    assert (degraded[0].status, degraded[0].error_code) == ("degraded", LOCAL_BRIDGE_UNSUPPORTED)


def test_a_local_bridge_mount_is_started_where_the_environment_has_fuse(harness, launcher):
    _with_fuse(harness)
    client = _client(harness)
    spec = _bridge_attachment(harness, mode="rw")
    launcher.answer = {"status": "connected", "mode": "rw"}

    prepared = client.attach(_manifest(harness, spec))

    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == [LOCAL_BRIDGE_MOUNT]
    assert prepared[0].mount_path == spec["mount_path"]

    # Started inside the sandbox: this interpreter, the module written
    # there, the relay, the path and the mode on the command line — and the
    # token NOT on it, in a file only the owner can read.
    argv = launcher.argv
    assert argv[0] == sys.executable
    assert argv[1].endswith("/bridge_mount.py")
    assert Path(argv[1]).read_text() == (
        Path(contents.__file__).with_name("bridge_mount.py").read_text()
    )
    assert argv[argv.index("--relay-url") + 1] == RELAY_URL
    assert argv[argv.index("--mount-path") + 1] == spec["mount_path"]
    assert argv[argv.index("--mode") + 1] == "rw"
    assert BRIDGE_TOKEN not in " ".join(argv)
    token_file = Path(argv[argv.index("--token-file") + 1])
    assert token_file.read_text() == BRIDGE_TOKEN
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert launcher.launches[-1][1]["start_new_session"] is True
    assert Path(spec["mount_path"]).is_dir()

    # A reconcile finds the mount there and starts nothing.
    again = client.reconcile_contents(_manifest(harness, spec))
    assert again[0].status == "ready"
    assert "already mounted" in (again[0].detail or "")
    assert len(launcher.launches) == 1

    # Detaching unmounts and forgets the token; the person's folder is theirs.
    client.detach("att-bridge")
    assert any(spec["mount_path"] in run and "-u" in run for run in launcher.runs)
    assert not token_file.exists()
    assert client.attachment_status("att-bridge") is None


@pytest.mark.parametrize(
    ("answer", "expected_code", "expected_words"),
    [
        (
            {
                "status": "failed",
                "error": "BRIDGE_CONNECT_FAILED",
                "state": "expired",
                "detail": "relay said no",
            },
            BRIDGE_CONNECT_FAILED,
            ("relay said no", "expired"),
        ),
        (
            {"status": "failed", "error": "FUSE_UNAVAILABLE", "detail": "/dev/fuse is absent"},
            FUSE_UNAVAILABLE,
            ("/dev/fuse",),
        ),
    ],
)
def test_a_local_bridge_that_does_not_connect_is_failed(
    harness, launcher, answer, expected_code, expected_words
):
    _with_fuse(harness)
    client = _client(harness)
    spec = _bridge_attachment(harness)
    launcher.answer = answer

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == expected_code
    detail = raised.value.attachments[0].detail or ""
    assert all(word in detail for word in expected_words)
    # The token is not left lying around after a failed start.
    argv = launcher.argv
    assert not Path(argv[argv.index("--token-file") + 1]).exists()
    assert client.is_started


def test_a_local_bridge_needs_a_prepared_bridge_session(harness, launcher):
    _with_fuse(harness)
    client = _client(harness)
    spec = _bridge_attachment(harness)
    del spec["bridge"]

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == BRIDGE_NOT_PREPARED
    assert launcher.launches == []


def test_a_datalayer_local_bridge_is_ready_only_when_the_csi_volume_is_mounted(harness, launcher):
    if harness.provider != "datalayer":
        pytest.skip("only Datalayer has an Operator rendering CSI volumes")
    client = _client(harness)
    spec = _bridge_attachment(harness)
    manifest = _manifest(harness, spec)

    # Nothing there: the Operator did not render the volume.
    with pytest.raises(ContentAttachmentError) as absent:
        client.attach(manifest)
    assert absent.value.code == MOUNT_MISSING

    # A directory, but no mount behind it: not the person's folder.
    harness.provision(spec["mount_path"])
    with pytest.raises(ContentAttachmentError) as unmounted:
        client.attach(manifest)
    assert unmounted.value.code == MOUNT_MISSING
    assert "mountpoint" in (unmounted.value.attachments[0].detail or "")

    # The CSI driver bound the bridge filesystem there.
    launcher.mountpoints.add(spec["mount_path"])
    prepared = client.reconcile_contents(manifest)
    assert prepared[0].status == "ready"
    assert prepared[0].capabilities == [LOCAL_BRIDGE_MOUNT]
    # Nothing was started in the sandbox: the mount is the node's.
    assert launcher.launches == []


# --- Mutation: a copy reported as a mount (must be caught) ---------------------


def _materializing_copy(harness: _Harness):
    """The mutation: an adapter that copies the folder in and calls it a mount."""

    def prepare(self, spec, *, reconcile):
        del reconcile
        target = Path(spec.mount_path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "notes.txt").write_bytes(b"a faithful copy\n")
        return ready(spec, capabilities=[LOCAL_BRIDGE_MOUNT], detail="mounted (not really)")

    return prepare


def test_a_copy_reported_as_a_local_bridge_mount_is_caught(harness, launcher, monkeypatch):
    if harness.provider != "datalayer":
        harness.rebuild(features=["fuse"])
    client = _client(harness)
    spec = _bridge_attachment(harness)
    monkeypatch.setattr(
        type(harness.sandbox), "_prepare_local_bridge", _materializing_copy(harness)
    )

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == LOCAL_BRIDGE_NOT_A_MOUNT
    assert "not a mountpoint" in (raised.value.attachments[0].detail or "")
    assert (Path(spec["mount_path"]) / "notes.txt").exists()  # the copy is there —
    assert client.attachment_status("att-bridge").status == "failed"  # and is not a mount


def test_a_local_bridge_claiming_any_other_capability_is_caught(harness, launcher, monkeypatch):
    """Even a real mountpoint is not a bridge if the adapter says `materialize`."""
    if harness.provider != "datalayer":
        harness.rebuild(features=["fuse"])
    client = _client(harness)
    spec = _bridge_attachment(harness)
    launcher.mountpoints.add(spec["mount_path"])
    monkeypatch.setattr(
        type(harness.sandbox),
        "_prepare_local_bridge",
        lambda self, spec, *, reconcile: ready(spec, capabilities=["materialize"]),
    )

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(_manifest(harness, spec))

    assert raised.value.code == LOCAL_BRIDGE_NOT_A_MOUNT
    assert "a copy is never reported as a mount" in (raised.value.attachments[0].detail or "")


# --- The fuse probe -----------------------------------------------------------------


def test_the_fuse_probe_reports_what_the_sandbox_has(harness):
    harness.sandbox.start()

    answer = contents.probe_fuse(harness.sandbox)

    fusepy_here = importlib.util.find_spec("fuse") is not None
    assert answer["fusepy"] is fusepy_here
    assert answer["device"] is os.path.exists("/dev/fuse")
    assert answer["ok"] is (fusepy_here and os.path.exists("/dev/fuse"))


SESSION_KEY = "ab" * 32


def test_a_bridge_s_secrets_go_to_their_own_files_and_never_into_the_manifest(harness, launcher):
    """The mount token and the session key are handed to the mount process
    through files only the owner can read; the manifest JSON, readable to
    anything in the sandbox, names the bridge and carries neither."""
    _with_fuse(harness)
    client = _client(harness)
    spec = _bridge_attachment(harness)
    spec["bridge"]["session_key"] = SESSION_KEY

    client.attach(_manifest(harness, spec))

    written = Path(client.contents_location.manifest_path).read_text()
    assert BRIDGE_TOKEN not in written
    assert SESSION_KEY not in written
    assert TOKEN not in written
    assert json.loads(written)["attachments"][0]["bridge"]["bridge_uid"] == "br-1"
    argv = launcher.argv
    assert argv[argv.index("--bridge-uid") + 1] == "br-1"
    assert SESSION_KEY not in " ".join(argv)
    session_file = Path(argv[argv.index("--session-key-file") + 1])
    assert session_file.read_text() == SESSION_KEY
    assert stat.S_IMODE(session_file.stat().st_mode) == 0o600

    # Detaching forgets the session key with the token.
    client.detach("att-bridge")
    assert not session_file.exists()


def test_a_failed_start_forgets_the_session_key_with_the_token(harness, launcher):
    _with_fuse(harness)
    client = _client(harness)
    spec = _bridge_attachment(harness)
    spec["bridge"]["session_key"] = SESSION_KEY
    launcher.answer = {"status": "failed", "error": "BRIDGE_CONNECT_FAILED", "detail": "no relay"}

    with pytest.raises(ContentAttachmentError):
        client.attach(_manifest(harness, spec))

    argv = launcher.argv
    assert not Path(argv[argv.index("--token-file") + 1]).exists()
    assert not Path(argv[argv.index("--session-key-file") + 1]).exists()
