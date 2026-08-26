# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Contents attachments, the same on every provider.

The Contents service decides WHAT a sandbox gets — a volume, a home folder, a
dataset revision, a bucket — and writes it down as a manifest of attachments.
What this module owns is the other half: how a sandbox on a given provider
HONOURS each attachment, and the one vocabulary the answer is given in, so
that a caller reading a `PreparedAttachment` does not need to know whether a
Daytona volume or a Clouder CSI mount is behind it.

The models mirror the service's contract (`ContentAttachment`,
`ContentAttachmentManifest`) field for field where the sandbox side needs
them, and ignore the rest: a manifest fetched from the service validates here
unchanged.

Three things are provider-neutral enough to live here rather than in an
adapter, because every provider this package drives has a Python kernel and
nothing else is needed for them:

- :func:`install_manifest` writes the manifest INTO the sandbox and exports
  the environment variables a Contents client inside it reads. The token is
  never written into the JSON — it goes to its own file, mode 0600 — and it
  is the short-lived sandbox credential the manifest carries, never a user's
  key.
- :func:`materialize` fetches the entries of a `materialize` attachment
  inside the sandbox, from the signed URLs the manifest carries, so the bytes
  never pass through the host that runs this package.
- :class:`CreationTimeMounts` keeps the volume mounts a provider only honours
  when the sandbox is created, for the adapters — Daytona, E2B, Modal — whose
  SDKs take mounts as a creation parameter and nothing afterwards.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import SandboxError, SandboxExecutionError

if TYPE_CHECKING:
    from .base import Sandbox

__all__ = [
    "CHECKSUM_MISMATCH",
    "CREDENTIAL_DELIVERY_UNSUPPORTED",
    "DELIVERY_UNSUPPORTED",
    "FETCH_FAILED",
    "FILESYSTEM_PRIMITIVES",
    "MANIFEST_DIRECTORIES",
    "MANIFEST_ENV",
    "MOUNT_MISSING",
    "MOUNT_NEEDS_RESTART",
    "MOUNT_PATH_MISSING",
    "TOKEN_ENV",
    "TOKEN_FILE_ENV",
    "URL_ENV",
    "ContentAttachmentError",
    "ContentAttachmentSpec",
    "ContentCapabilities",
    "ContentManifest",
    "CreationTimeMounts",
    "LocalBridgeCapability",
    "ManifestLocation",
    "MaterializeEntry",
    "PreparedAttachment",
    "contents_environment",
    "install_manifest",
    "materialize",
    "not_ready",
    "path_exists",
    "probe",
    "ready",
    "remove_materialized",
    "unsupported",
]

Delivery = Literal["mount", "local-bridge", "materialize", "client", "environment"]
AccessMode = Literal["mount", "python", "object-client"]
PreparedStatus = Literal["ready", "degraded", "failed"]

#: What a sandbox may do to an attachment's files through this package's own
#: filesystem, whichever provider is underneath: the `SandboxFilesystem` of the
#: base class offers all of these over a bare kernel.
FILESYSTEM_PRIMITIVES: tuple[str, ...] = (
    "list",
    "stat",
    "read",
    "write",
    "mkdir",
    "remove",
    "upload",
    "download",
)

# --- What a prepared attachment can say went wrong -------------------------

#: The provider has no way to deliver the attachment the way it asks to be.
DELIVERY_UNSUPPORTED = "DELIVERY_UNSUPPORTED"
#: A mount the provider only takes at creation, asked of a running sandbox.
MOUNT_NEEDS_RESTART = "MOUNT_NEEDS_RESTART"
#: The mount path is not there — the Operator (or the provider) did not make it.
MOUNT_MISSING = "MOUNT_MISSING"
#: A mount or a relative materialization with nowhere to go.
MOUNT_PATH_MISSING = "MOUNT_PATH_MISSING"
#: Delivering the attachment would need a credential to leave Contents.
CREDENTIAL_DELIVERY_UNSUPPORTED = "CREDENTIAL_DELIVERY_UNSUPPORTED"
#: A materialized file did not match the digest the manifest gave for it.
CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
#: A materialized file could not be fetched from its signed URL.
FETCH_FAILED = "FETCH_FAILED"

# --- Where the manifest goes inside the sandbox ----------------------------

#: Tried in order; the first directory that can be written wins. `/etc` is
#: for a sandbox whose kernel runs as root or was given the directory; a home
#: directory is where everything else can write.
MANIFEST_DIRECTORIES: tuple[str, ...] = ("/etc/datalayer", "~/.datalayer")
MANIFEST_FILENAME = "contents.json"
TOKEN_FILENAME = "contents.token"  # noqa: S105 - a file NAME

MANIFEST_ENV = "DATALAYER_CONTENTS_MANIFEST"
URL_ENV = "DATALAYER_CONTENTS_URL"
TOKEN_ENV = "DATALAYER_CONTENTS_TOKEN"  # noqa: S105 - the NAME of a variable
TOKEN_FILE_ENV = "DATALAYER_CONTENTS_TOKEN_FILE"  # noqa: S105

#: The line a snippet run inside the sandbox answers on, and the only line
#: :func:`probe` reads. Long and specific: the code's own output must not be
#: mistaken for an answer.
_MARKER = "__code_sandboxes_contents__:"


# --- Models ------------------------------------------------------------------


class MaterializeEntry(BaseModel):
    """One file of a `materialize` attachment.

    Attributes:
        source_url: A signed URL the sandbox may fetch the bytes from. Signed
            by Contents for this attachment, and short-lived: it is not a
            credential a notebook can reuse for anything else.
        path: Where the file goes — absolute, or relative to the attachment's
            `mount_path`.
        sha256: The digest the fetched file must have, when Contents knows it.
        size: The size in bytes, when known.
    """

    model_config = ConfigDict(extra="ignore")

    source_url: str
    path: str
    sha256: str | None = None
    size: int | None = None


class ContentAttachmentSpec(BaseModel):
    """One attachment of the manifest: what the sandbox is to be given.

    Mirrors the `ContentAttachment` of the Contents contract, keeping the
    fields the sandbox side acts on. `delivery` is what is dispatched on;
    `access_mode` is set only for a Cloud Storage source and, together with
    `delivery="mount"`, names a BUCKET mount; `provider_resource_id` is set
    only for a volume and is what a provider-native mount is made from.
    """

    model_config = ConfigDict(extra="ignore")

    uid: str
    source_uid: str
    revision_uid: str | None = None
    sandbox_uid: str
    sandbox_provider: str
    mode: Literal["ro", "rw"] = "ro"
    mount_path: str | None = None
    delivery: Delivery = "mount"
    required: bool = True
    access_mode: AccessMode | None = None
    filesystem_primitives: list[str] = Field(default_factory=list)
    provider_resource_id: str | None = None
    status: str = "requested"
    materialize: list[MaterializeEntry] = Field(default_factory=list)

    @property
    def is_bucket_mount(self) -> bool:
        """A Cloud Storage source asked for as a filesystem."""
        return self.delivery == "mount" and self.access_mode == "mount"

    @property
    def is_volume_mount(self) -> bool:
        """A mount a provider can make from one of its own volumes."""
        return (
            self.delivery == "mount"
            and not self.is_bucket_mount
            and bool(self.provider_resource_id)
        )


class ContentManifest(BaseModel):
    """Everything a sandbox is to be given, and how to reach Contents.

    `contents_url` and `token` are the SANDBOX's credentials to Contents:
    minted for `sandbox_uid`, short-lived, scoped to these attachments. A
    user's API key never travels here. The token is kept out of reprs, and
    :func:`install_manifest` keeps it out of the JSON written into the
    sandbox.
    """

    model_config = ConfigDict(extra="ignore")

    contract_version: Literal["v1"] = "v1"
    sandbox_uid: str
    sandbox_provider: str
    generated_at: str
    attachments: list[ContentAttachmentSpec] = Field(default_factory=list)
    contents_url: str | None = None
    token: str | None = Field(default=None, repr=False)

    def attachment(self, uid: str) -> ContentAttachmentSpec | None:
        """The attachment with this uid, or None."""
        for spec in self.attachments:
            if spec.uid == uid:
                return spec
        return None


class PreparedAttachment(BaseModel):
    """What became of one attachment on this sandbox.

    `ready` means the attachment is usable now. `degraded` means an OPTIONAL
    attachment could not be honoured, and the sandbox is still fit to run.
    `failed` means a required one could not be, and `error_code` says why in
    a word a caller can act on; `detail` says it in a sentence.
    """

    model_config = ConfigDict(extra="forbid")

    uid: str
    status: PreparedStatus
    mount_path: str | None = None
    provider_resource_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    detail: str | None = None
    error_code: str | None = None


class LocalBridgeCapability(BaseModel):
    """Whether a person's own folder can be bridged into the sandbox.

    A local bridge is a mount made by a driver on the node — Clouder's CSI on
    Datalayer — and the fields say what that driver does: the roots it will
    bridge (empty for no restriction), whether it can be read-only or
    read-write, whether it comes back after the connection dropped, and
    whether it is taken away cleanly when the sandbox goes.
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool = False
    required_features: list[str] = Field(default_factory=list)
    allowed_roots: list[str] = Field(default_factory=list)
    read_only: bool = False
    read_write: bool = False
    reconnect: bool = False
    cleanup: bool = False


class ContentCapabilities(BaseModel):
    """What a provider can do with an attachment, before one is asked for.

    `mount` is a provider-native volume mount, made from an attachment's
    `provider_resource_id`. `bucket_mount` is a bucket as a filesystem, which
    needs a driver on the node — or a credential handed to the provider,
    which is refused. `materialize` is fetching files into the sandbox from
    signed URLs. `client` is the sandbox reaching Contents itself, which
    every provider with a network can do.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    mount: bool = False
    bucket_mount: bool = False
    materialize: bool = False
    client: bool = True
    local_bridge_mount: LocalBridgeCapability = Field(default_factory=LocalBridgeCapability)
    filesystem_primitives: list[str] = Field(default_factory=lambda: list(FILESYSTEM_PRIMITIVES))


class ContentAttachmentError(SandboxError):
    """A required attachment is not ready, and the caller asked to be told.

    `uid` and `code` are the first attachment that is not ready and why;
    `attachments` is everything that was prepared, so the caller can see the
    rest — and the sandbox is left running, because whether to stop it is
    the caller's decision, not this package's.
    """

    def __init__(
        self,
        uid: str,
        code: str,
        message: str | None = None,
        attachments: list[PreparedAttachment] | None = None,
    ):
        self.uid = uid
        self.code = code
        self.attachments = list(attachments or [])
        super().__init__(message or f"Content attachment {uid} is not ready: {code}")


@dataclass(frozen=True)
class ManifestLocation:
    """Where :func:`install_manifest` put the manifest inside the sandbox."""

    directory: str
    manifest_path: str
    token_path: str | None


# --- Answers ----------------------------------------------------------------


def ready(
    spec: ContentAttachmentSpec,
    *,
    capabilities: list[str],
    detail: str | None = None,
) -> PreparedAttachment:
    """The attachment is usable now."""
    return PreparedAttachment(
        uid=spec.uid,
        status="ready",
        mount_path=spec.mount_path,
        provider_resource_id=spec.provider_resource_id,
        capabilities=list(capabilities),
        detail=detail,
    )


def not_ready(spec: ContentAttachmentSpec, code: str, detail: str) -> PreparedAttachment:
    """The attachment could not be honoured: failed if required, else degraded."""
    return PreparedAttachment(
        uid=spec.uid,
        status="failed" if spec.required else "degraded",
        mount_path=spec.mount_path,
        provider_resource_id=spec.provider_resource_id,
        detail=detail,
        error_code=code,
    )


def unsupported(
    spec: ContentAttachmentSpec, provider: str, detail: str | None = None
) -> PreparedAttachment:
    """The provider has no way to deliver this attachment as asked."""
    return not_ready(
        spec,
        DELIVERY_UNSUPPORTED,
        detail or f"{provider} sandboxes cannot deliver an attachment as {spec.delivery!r}",
    )


# --- Talking to the sandbox ---------------------------------------------------


def probe(sandbox: Sandbox, code: str, *, timeout: float | None = None) -> dict[str, Any]:
    """Run `code` in the sandbox and read back the one line it answers on.

    The snippets of this module write their answer as JSON behind
    :data:`_MARKER`, on stdout, because stdout is the one channel every
    provider has: a kernel that cannot hand a variable back — Modal's session
    process — can still print. A snippet that raised is an execution error
    here, not a `failed` attachment: it is this package's code that broke,
    not the attachment.
    """
    execution = sandbox.run_code(code, timeout=timeout)
    if not execution.execution_ok:
        raise SandboxExecutionError(
            "SandboxError", execution.execution_error or "Sandbox execution failed"
        )
    if execution.code_error is not None:
        raise SandboxExecutionError(
            execution.code_error.name, execution.code_error.value, execution.code_error.traceback
        )
    for message in reversed(execution.logs.stdout):
        at = message.line.find(_MARKER)
        if at >= 0:
            return json.loads(message.line[at + len(_MARKER) :])
    raise SandboxExecutionError(
        "ContentsProbeError", "the sandbox answered without the report it was asked for"
    )


def _answer(expression: str) -> str:
    """Code writing the JSON of `expression` on its own line of stdout.

    Under prefixed names, and dropped afterwards: the namespace is the one
    the caller keeps working in.
    """
    return (
        "import json as _cs_answer_json, sys as _cs_answer_sys\n"
        f"_cs_answer_sys.stdout.write({_MARKER!r} + _cs_answer_json.dumps({expression}) + '\\n')\n"
        "_cs_answer_sys.stdout.flush()\n"
        "del _cs_answer_json, _cs_answer_sys\n"
    )


def path_exists(sandbox: Sandbox, path: str) -> bool:
    """Whether `path` exists inside the sandbox."""
    code = (
        "import os as _cs_os\n"
        f"_cs_exists = _cs_os.path.exists({path!r})\n"
        + _answer("{'exists': _cs_exists}")
        + "del _cs_os, _cs_exists\n"
    )
    return bool(probe(sandbox, code)["exists"])


def contents_environment(
    manifest: ContentManifest, location: ManifestLocation | None = None
) -> dict[str, str]:
    """The environment a Contents client inside the sandbox reads.

    Without a `location` — before the sandbox exists, when a provider takes
    its environment at creation — the canonical paths are named; the write
    that happens once the sandbox runs corrects the kernel's environment if
    the manifest had to go elsewhere.
    """
    if location is None:
        directory = MANIFEST_DIRECTORIES[0]
        location = ManifestLocation(
            directory=directory,
            manifest_path=posixpath.join(directory, MANIFEST_FILENAME),
            token_path=posixpath.join(directory, TOKEN_FILENAME) if manifest.token else None,
        )
    environment = {MANIFEST_ENV: location.manifest_path}
    if manifest.contents_url:
        environment[URL_ENV] = manifest.contents_url
    if manifest.token and location.token_path:
        environment[TOKEN_ENV] = manifest.token
        environment[TOKEN_FILE_ENV] = location.token_path
    return environment


def install_manifest(
    sandbox: Sandbox,
    manifest: ContentManifest,
    *,
    directories: tuple[str, ...] | None = None,
) -> ManifestLocation:
    """Write the manifest into the sandbox and export where it is.

    The JSON carries everything but the token. The token — when there is one
    — goes to a file of its own that only the owner can read, and the
    environment names both. Rewritten on every call, so a retry, a reconcile
    or a manifest with one more attachment all end in the same state.
    """
    payload = {
        "manifest": manifest.model_dump(mode="json", exclude={"token"}),
        "token": manifest.token,
        "url": manifest.contents_url,
        "directories": list(directories or MANIFEST_DIRECTORIES),
        "names": [MANIFEST_FILENAME, TOKEN_FILENAME],
        "env": [MANIFEST_ENV, URL_ENV, TOKEN_ENV, TOKEN_FILE_ENV],
    }
    code = (
        "import json as _cs_json, os as _cs_os\n"
        f"_cs_payload = _cs_json.loads({json.dumps(payload)!r})\n"
        "_cs_chosen = None\n"
        "for _cs_candidate in _cs_payload['directories']:\n"
        "    _cs_directory = _cs_os.path.expanduser(_cs_candidate)\n"
        "    try:\n"
        "        _cs_os.makedirs(_cs_directory, exist_ok=True)\n"
        "        _cs_part = _cs_os.path.join(_cs_directory, _cs_payload['names'][0] + '.part')\n"
        "        with open(_cs_part, 'w') as _cs_handle:\n"
        "            _cs_json.dump(_cs_payload['manifest'], _cs_handle, indent=2)\n"
        "        _cs_manifest_path = _cs_os.path.join(_cs_directory, _cs_payload['names'][0])\n"
        "        _cs_os.replace(_cs_part, _cs_manifest_path)\n"
        "        _cs_chosen = _cs_directory\n"
        "        break\n"
        "    except OSError:\n"
        "        continue\n"
        "if _cs_chosen is None:\n"
        "    raise OSError('none of ' + ', '.join(_cs_payload['directories']) + ' is writable')\n"
        "_cs_token_path = None\n"
        "if _cs_payload['token'] is not None:\n"
        "    _cs_token_path = _cs_os.path.join(_cs_chosen, _cs_payload['names'][1])\n"
        "    _cs_fd = _cs_os.open(_cs_token_path, _cs_os.O_WRONLY | _cs_os.O_CREAT "
        "| _cs_os.O_TRUNC, 0o600)\n"
        "    try:\n"
        "        _cs_os.write(_cs_fd, _cs_payload['token'].encode('utf-8'))\n"
        "    finally:\n"
        "        _cs_os.close(_cs_fd)\n"
        "    _cs_os.chmod(_cs_token_path, 0o600)\n"
        "_cs_env = _cs_payload['env']\n"
        "_cs_os.environ[_cs_env[0]] = _cs_manifest_path\n"
        "if _cs_payload['url']:\n"
        "    _cs_os.environ[_cs_env[1]] = _cs_payload['url']\n"
        "if _cs_token_path is not None:\n"
        "    _cs_os.environ[_cs_env[2]] = _cs_payload['token']\n"
        "    _cs_os.environ[_cs_env[3]] = _cs_token_path\n"
        + _answer(
            "{'directory': _cs_chosen, 'manifest_path': _cs_manifest_path, "
            "'token_path': _cs_token_path}"
        )
        + "del _cs_json, _cs_os, _cs_payload, _cs_chosen, _cs_candidate, _cs_directory, "
        "_cs_manifest_path, _cs_token_path, _cs_env\n"
    )
    answer = probe(sandbox, code)
    return ManifestLocation(
        directory=answer["directory"],
        manifest_path=answer["manifest_path"],
        token_path=answer.get("token_path"),
    )


# --- Materializing ------------------------------------------------------------


def _destinations(spec: ContentAttachmentSpec) -> list[tuple[MaterializeEntry, str]]:
    """Each entry and the absolute path it lands on inside the sandbox."""
    resolved: list[tuple[MaterializeEntry, str]] = []
    for entry in spec.materialize:
        if posixpath.isabs(entry.path):
            resolved.append((entry, posixpath.normpath(entry.path)))
        elif spec.mount_path:
            resolved.append(
                (entry, posixpath.normpath(posixpath.join(spec.mount_path, entry.path)))
            )
        else:
            raise ValueError(entry.path)
    return resolved


def materialize(
    sandbox: Sandbox,
    spec: ContentAttachmentSpec,
    *,
    reconcile: bool,
    timeout: float | None = None,
) -> PreparedAttachment:
    """Fetch the files of a `materialize` attachment inside the sandbox.

    Each file is fetched from its signed URL by the sandbox itself, written
    beside its destination and moved into place only once its digest — when
    the manifest gave one — has been checked, so a half-fetched or corrupt
    file is never what a notebook opens. With `reconcile`, a file that is
    already there with the right digest is left alone; without it, every
    file is fetched afresh.
    """
    try:
        destinations = _destinations(spec)
    except ValueError as error:
        return not_ready(
            spec,
            MOUNT_PATH_MISSING,
            f"materialize entry {error.args[0]!r} is relative and the attachment has no mount_path",
        )
    if not destinations:
        return ready(spec, capabilities=["materialize"], detail="nothing to materialize")

    payload = {
        "reconcile": reconcile,
        "timeout": timeout or 300,
        "entries": [
            {"path": path, "source_url": entry.source_url, "sha256": entry.sha256}
            for entry, path in destinations
        ],
    }
    code = (
        "import hashlib as _cs_hashlib, json as _cs_json, os as _cs_os\n"
        "import urllib.request as _cs_request\n"
        f"_cs_payload = _cs_json.loads({json.dumps(payload)!r})\n"
        "def _cs_digest(path):\n"
        "    digest = _cs_hashlib.sha256()\n"
        "    with open(path, 'rb') as handle:\n"
        "        for chunk in iter(lambda: handle.read(1 << 20), b''):\n"
        "            digest.update(chunk)\n"
        "    return digest.hexdigest()\n"
        "_cs_report = []\n"
        "for _cs_entry in _cs_payload['entries']:\n"
        "    _cs_path = _cs_entry['path']\n"
        "    _cs_item = {'path': _cs_path, 'status': 'written'}\n"
        "    try:\n"
        "        _cs_wanted = _cs_entry.get('sha256')\n"
        "        if (\n"
        "            _cs_payload['reconcile']\n"
        "            and _cs_os.path.isfile(_cs_path)\n"
        "            and (_cs_wanted is None or _cs_digest(_cs_path) == _cs_wanted)\n"
        "        ):\n"
        "            _cs_item['status'] = 'present'\n"
        "        else:\n"
        "            _cs_os.makedirs(_cs_os.path.dirname(_cs_path) or '/', exist_ok=True)\n"
        "            _cs_part = _cs_path + '.part'\n"
        "            with _cs_request.urlopen(\n"
        "                _cs_entry['source_url'], timeout=_cs_payload['timeout']\n"
        "            ) as _cs_source, open(_cs_part, 'wb') as _cs_target:\n"
        "                for _cs_chunk in iter(lambda: _cs_source.read(1 << 20), b''):\n"
        "                    _cs_target.write(_cs_chunk)\n"
        "            if _cs_wanted is not None and _cs_digest(_cs_part) != _cs_wanted:\n"
        "                _cs_os.remove(_cs_part)\n"
        "                raise ValueError('CHECKSUM_MISMATCH')\n"
        "            _cs_os.replace(_cs_part, _cs_path)\n"
        "        _cs_item['size'] = _cs_os.path.getsize(_cs_path)\n"
        "    except Exception as _cs_error:\n"
        "        _cs_item['status'] = 'failed'\n"
        "        _cs_item['error'] = (\n"
        "            'CHECKSUM_MISMATCH'\n"
        "            if str(_cs_error) == 'CHECKSUM_MISMATCH'\n"
        "            else 'FETCH_FAILED'\n"
        "        )\n"
        "        _cs_item['detail'] = type(_cs_error).__name__ + ': ' + str(_cs_error)\n"
        "    _cs_report.append(_cs_item)\n"
        + _answer("{'entries': _cs_report}")
        + "del _cs_hashlib, _cs_json, _cs_os, _cs_request, _cs_payload, _cs_digest, _cs_report\n"
    )
    answer = probe(sandbox, code, timeout=(timeout or 300) * max(1, len(destinations)))
    failures = [item for item in answer["entries"] if item.get("status") == "failed"]
    if failures:
        first = failures[0]
        return not_ready(
            spec,
            first.get("error") or FETCH_FAILED,
            "; ".join(
                f"{item['path']}: {item.get('detail', item.get('error'))}" for item in failures
            ),
        )
    present = sum(1 for item in answer["entries"] if item.get("status") == "present")
    written = len(answer["entries"]) - present
    return ready(
        spec,
        capabilities=["materialize"],
        detail=f"{written} file(s) materialized, {present} already present",
    )


def remove_materialized(sandbox: Sandbox, spec: ContentAttachmentSpec) -> None:
    """Remove the files a `materialize` attachment put in the sandbox.

    Files only — the directory above them may be a mount, or shared with
    something else — and never anything at the source: the source is the
    service's, and detaching is not deleting.
    """
    try:
        paths = [path for _entry, path in _destinations(spec)]
    except ValueError:
        return
    if not paths:
        return
    code = (
        "import json as _cs_json, os as _cs_os\n"
        f"_cs_paths = _cs_json.loads({json.dumps(paths)!r})\n"
        "_cs_removed = []\n"
        "for _cs_path in _cs_paths:\n"
        "    for _cs_candidate in (_cs_path, _cs_path + '.part'):\n"
        "        try:\n"
        "            _cs_os.remove(_cs_candidate)\n"
        "            _cs_removed.append(_cs_candidate)\n"
        "        except OSError:\n"
        "            pass\n"
        + _answer("{'removed': _cs_removed}")
        + "del _cs_json, _cs_os, _cs_paths, _cs_removed\n"
    )
    probe(sandbox, code)


# --- Mounts a provider only takes at creation --------------------------------


class CreationTimeMounts:
    """The volume mounts of a provider whose SDK takes them at creation.

    Daytona, E2B and Modal all mount a volume as a parameter of creating the
    sandbox and offer nothing for a sandbox already running. So a request is
    RECORDED here, keyed by mount path — asking twice for the same path is one
    mount, not two — and read by the adapter when it creates; what the running
    sandbox was actually created with is kept apart, so that a request made
    after creation is answered honestly: it needs a restart.
    """

    def __init__(self) -> None:
        #: mount path → provider volume id, to be honoured at the next creation.
        self.requested: dict[str, str] = {}
        #: What the running sandbox was created with.
        self.mounted: dict[str, str] = {}

    def request(self, spec: ContentAttachmentSpec) -> bool:
        """Record a volume mount for creation; False when this is not one."""
        if not spec.is_volume_mount or not spec.mount_path or not spec.provider_resource_id:
            return False
        self.requested[spec.mount_path] = spec.provider_resource_id
        return True

    def request_all(self, manifest: ContentManifest) -> None:
        for spec in manifest.attachments:
            self.request(spec)

    def forget(self, spec: ContentAttachmentSpec) -> None:
        """Forget the request. The volume itself is never touched."""
        if spec.mount_path:
            self.requested.pop(spec.mount_path, None)

    def created(self) -> None:
        """The sandbox was just created with everything requested."""
        self.mounted = dict(self.requested)

    def stopped(self) -> None:
        self.mounted = {}

    def prepare(
        self,
        sandbox: Sandbox,
        spec: ContentAttachmentSpec,
        *,
        provider: str,
        bucket_code: str = DELIVERY_UNSUPPORTED,
    ) -> PreparedAttachment:
        """Answer for a `mount` attachment on a running sandbox."""
        if spec.is_bucket_mount:
            return not_ready(
                spec,
                bucket_code,
                f"{provider} sandboxes cannot mount a bucket from Contents: "
                "the bucket credential would have to leave Contents for the provider",
            )
        if not spec.provider_resource_id:
            return unsupported(
                spec,
                provider,
                f"{provider} sandboxes cannot mount the Datalayer shared filesystem; "
                "a volume of the provider's own is mounted by provider_resource_id",
            )
        if not spec.mount_path:
            return not_ready(spec, MOUNT_PATH_MISSING, "a mount needs a mount_path")
        if self.mounted.get(spec.mount_path) == spec.provider_resource_id:
            if path_exists(sandbox, spec.mount_path):
                return ready(spec, capabilities=["mount"])
            return not_ready(
                spec,
                MOUNT_MISSING,
                f"{spec.mount_path} was requested at creation but is not there",
            )
        self.request(spec)
        return not_ready(
            spec,
            MOUNT_NEEDS_RESTART,
            f"{provider} mounts a volume only when the sandbox is created; "
            f"{spec.mount_path} is recorded for the next start",
        )
