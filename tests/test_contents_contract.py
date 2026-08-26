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
import io
import json
import os
import stat
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
    CHECKSUM_MISMATCH,
    CREDENTIAL_DELIVERY_UNSUPPORTED,
    DELIVERY_UNSUPPORTED,
    FILESYSTEM_PRIMITIVES,
    MANIFEST_ENV,
    MOUNT_MISSING,
    MOUNT_NEEDS_RESTART,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    URL_ENV,
    ContentAttachmentError,
    ContentManifest,
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

    def make_sandbox(self) -> Sandbox:
        raise NotImplementedError

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

    def make_sandbox(self):
        return DatalayerSandbox(config=SandboxConfig(timeout=10.0))

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

    def make_sandbox(self):
        return DaytonaSandbox(config=SandboxConfig(timeout=10.0))

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

    def make_sandbox(self):
        return E2BSandbox(config=SandboxConfig(timeout=10.0))

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

    def make_sandbox(self):
        return ModalSandbox(config=SandboxConfig(timeout=10.0, max_lifetime=30.0))

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


def test_a_sandbox_that_has_not_started_cannot_prepare(harness):
    from code_sandboxes.exceptions import SandboxNotStartedError

    with pytest.raises(SandboxNotStartedError):
        harness.sandbox.prepare_contents(
            _manifest(harness, _attachment(harness, "att-client", delivery="client"))
        )
