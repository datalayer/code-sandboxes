# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The client's Contents surface, over the in-process variant.

`CodeSandboxClient` is what `agent-runtimes` and `datalayer-runtimes` hand a
manifest to, so the order of things — configure, start, install the manifest,
prepare — and the strict verb, `attach`, live there. The eval variant has no
mounts and fetches nothing, which is exactly what makes it the right place to
prove the shape: everything here is what every provider inherits.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from code_sandboxes import contents
from code_sandboxes.client import CodeSandboxClient
from code_sandboxes.contents import (
    DELIVERY_UNSUPPORTED,
    MANIFEST_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    URL_ENV,
    ContentAttachmentError,
    ContentAttachmentSpec,
    ContentManifest,
    PreparedAttachment,
)
from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.exceptions import SandboxNotStartedError

TOKEN = "short-lived-sandbox-token"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home of this test's own, and an environment left as it was found."""
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
    return home


@pytest.fixture
def client(home) -> CodeSandboxClient:
    client = CodeSandboxClient(EvalSandbox(), owns_sandbox=True)
    yield client
    client.close()


def _attachment(uid: str, **fields) -> dict:
    return {
        "uid": uid,
        "source_uid": f"source-{uid}",
        "sandbox_uid": "sandbox-1",
        "sandbox_provider": "eval",
        **fields,
    }


def _manifest(*attachments: dict, token: str | None = TOKEN) -> ContentManifest:
    return ContentManifest(
        sandbox_uid="sandbox-1",
        sandbox_provider="eval",
        generated_at="2026-08-26T09:00:00Z",
        attachments=list(attachments),
        contents_url="https://contents.example.test",
        token=token,
    )


def test_a_client_attachment_is_ready_and_the_sandbox_is_started(client):
    assert not client.is_started

    prepared = client.attach(_manifest(_attachment("att-client", delivery="client")))

    assert prepared == [
        PreparedAttachment(uid="att-client", status="ready", capabilities=["client"])
    ]
    assert client.is_started
    assert client.attachment_status("att-client") == prepared[0]


def test_the_manifest_is_installed_and_the_environment_names_it(client, home):
    manifest = _manifest(_attachment("att-client", delivery="client"))

    client.attach(manifest)

    location = client.contents_location
    assert location is not None
    # The first candidate directory that can be written wins; here it can.
    assert location.directory == contents.MANIFEST_DIRECTORIES[0]
    assert location.manifest_path == str(Path(location.directory) / "contents.json")
    assert location.directory != str(home / ".datalayer")
    written = json.loads(Path(location.manifest_path).read_text())
    assert written["sandbox_uid"] == "sandbox-1"
    assert written["contents_url"] == "https://contents.example.test"
    assert [item["uid"] for item in written["attachments"]] == ["att-client"]
    assert os.environ[MANIFEST_ENV] == location.manifest_path
    assert os.environ[URL_ENV] == "https://contents.example.test"


def test_the_token_is_kept_out_of_the_manifest_and_in_a_file_of_its_own(client):
    client.attach(_manifest(_attachment("att-client", delivery="client")))

    location = client.contents_location
    text = Path(location.manifest_path).read_text()
    assert TOKEN not in text
    assert "token" not in json.loads(text)
    token_file = Path(location.token_path)
    assert token_file.read_text() == TOKEN
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert os.environ[TOKEN_ENV] == TOKEN
    assert os.environ[TOKEN_FILE_ENV] == location.token_path


def test_the_manifest_falls_back_to_home_when_etc_cannot_be_written(client, home, tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root can write anywhere; the fallback cannot be provoked")
    blocked = tmp_path / "etc" / "datalayer"
    blocked.mkdir(parents=True)
    blocked.chmod(0o500)
    try:
        client.attach(_manifest(_attachment("att-client", delivery="client")))
    finally:
        blocked.chmod(0o700)

    assert client.contents_location.directory == str(home / ".datalayer")
    assert not (blocked / "contents.json").exists()


def test_attach_refuses_a_required_attachment_the_provider_cannot_honour(client):
    """The eval variant mounts nothing, and says so rather than pretending."""
    manifest = _manifest(
        _attachment("att-client", delivery="client"),
        _attachment("att-mount", delivery="mount", mount_path="/mnt/data"),
    )

    with pytest.raises(ContentAttachmentError) as raised:
        client.attach(manifest)

    assert raised.value.uid == "att-mount"
    assert raised.value.code == DELIVERY_UNSUPPORTED
    assert [item.status for item in raised.value.attachments] == ["ready", "failed"]
    # Left running — the caller decides whether to stop it.
    assert client.is_started
    assert client.attachment_status("att-mount").status == "failed"


def test_an_optional_attachment_is_degraded_and_attach_does_not_raise(client):
    prepared = client.attach(
        _manifest(_attachment("att-mount", delivery="mount", mount_path="/mnt", required=False))
    )

    assert prepared[0].status == "degraded"
    assert prepared[0].error_code == DELIVERY_UNSUPPORTED


def test_materializing_over_a_bare_kernel_is_refused_by_default(client):
    """The base class fetches nothing; an adapter has to say it can."""
    prepared = client.prepare_contents(
        _manifest(
            _attachment(
                "att-files",
                delivery="materialize",
                mount_path="/data",
                materialize=[{"source_url": "https://signed.example.test/x", "path": "x"}],
            )
        )
    )

    assert prepared[0].status == "failed"
    assert prepared[0].error_code == DELIVERY_UNSUPPORTED


def test_reconcile_answers_the_same_as_prepare(client):
    manifest = _manifest(
        _attachment("att-client", delivery="client"),
        _attachment("att-mount", delivery="mount", mount_path="/mnt", required=False),
    )

    first = client.prepare_contents(manifest)
    again = client.reconcile_contents(manifest)

    assert again == first


def test_a_second_attach_rewrites_the_manifest(client):
    """Retrying, or attaching one more thing, ends in the same place."""
    client.attach(_manifest(_attachment("att-one", delivery="client")))
    first = client.contents_location

    client.attach(
        _manifest(
            _attachment("att-one", delivery="client"), _attachment("att-two", delivery="client")
        )
    )

    assert client.contents_location == first
    written = json.loads(Path(first.manifest_path).read_text())
    assert [item["uid"] for item in written["attachments"]] == ["att-one", "att-two"]


def test_detach_forgets_the_attachment(client):
    client.attach(_manifest(_attachment("att-client", delivery="client")))

    client.detach("att-client")

    assert client.attachment_status("att-client") is None
    # And again, without complaint.
    client.detach("att-client")


def test_the_capabilities_of_a_bare_kernel(client):
    capabilities = client.content_capabilities()

    assert capabilities.provider == "eval"
    assert capabilities.client
    assert not capabilities.mount
    assert not capabilities.bucket_mount
    assert not capabilities.materialize
    assert not capabilities.local_bridge_mount.supported
    assert "upload" in capabilities.filesystem_primitives


def test_configure_puts_the_environment_in_the_configuration(home):
    """What a provider that takes its environment at creation would get."""
    sandbox = EvalSandbox()

    sandbox.configure_contents(_manifest(_attachment("att-client", delivery="client")))

    # Before the sandbox exists nothing can be tried, so the canonical
    # directory is named; the install corrects the kernel if it moved.
    canonical = contents.MANIFEST_DIRECTORIES[0]
    assert sandbox.config.env_vars[MANIFEST_ENV] == f"{canonical}/contents.json"
    assert sandbox.config.env_vars[TOKEN_FILE_ENV] == f"{canonical}/contents.token"
    assert sandbox.config.env_vars[TOKEN_ENV] == TOKEN
    assert sandbox.config.env_vars[URL_ENV] == "https://contents.example.test"


def test_a_sandbox_that_has_not_started_cannot_prepare():
    with pytest.raises(SandboxNotStartedError):
        EvalSandbox().prepare_contents(_manifest(_attachment("att-client", delivery="client")))


def test_a_manifest_from_the_contents_service_validates_with_its_extra_fields():
    """The service's `ContentAttachment` carries more than the sandbox acts on."""
    manifest = ContentManifest.model_validate(
        {
            "contract_version": "v1",
            "sandbox_uid": "runtime-1",
            "sandbox_provider": "datalayer",
            "generated_at": "2026-08-26T09:00:00Z",
            "attachments": [
                {
                    "uid": "attachment-1",
                    "source_uid": "volume-1",
                    "revision_uid": None,
                    "sandbox_uid": "runtime-1",
                    "sandbox_provider": "datalayer",
                    "mode": "rw",
                    "mount_path": "/mnt/volume-1",
                    "delivery": "mount",
                    "required": True,
                    "access_mode": None,
                    "fallback_reason": None,
                    "filesystem_primitives": ["list", "stat", "read", "write"],
                    "provider_resource_id": "pvc-volume-1",
                    "capabilities": ["mount"],
                    "status": "requested",
                    "token_audience": "sandbox:runtime-1",
                    "expires_at": "2026-08-26T10:00:00Z",
                    "limits": {"bytes": None},
                    "cleanup_policy": "revoke",
                    "created_at": "2026-08-26T09:00:00Z",
                    "error": None,
                }
            ],
        }
    )

    spec = manifest.attachments[0]
    assert isinstance(spec, ContentAttachmentSpec)
    assert spec.is_volume_mount
    assert not spec.is_bucket_mount
    assert spec.materialize == []
    assert manifest.token is None
    assert manifest.attachment("attachment-1") is spec
    assert manifest.attachment("absent") is None
    assert "token" not in repr(manifest)


def test_a_bucket_mount_is_told_apart_from_a_volume_mount():
    bucket = ContentAttachmentSpec(
        **_attachment("b", delivery="mount", access_mode="mount", mount_path="/mnt/bucket")
    )
    assert bucket.is_bucket_mount
    assert not bucket.is_volume_mount
