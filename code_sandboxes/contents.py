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
- :func:`materialize` delivers the entries of a `materialize` attachment
  inside the sandbox: a file is fetched from the signed URL the manifest
  carries, a git entry is cloned at its pinned revision, and a bucket entry
  is not fetched at all — it is written into the manifest for the Contents
  client inside to open. The bytes never pass through the host that runs
  this package, and the sandbox never holds a bucket key.
- :class:`CreationTimeMounts` keeps the volume mounts a provider only honours
  when the sandbox is created, for the adapters — Daytona, E2B, Modal — whose
  SDKs take mounts as a creation parameter and nothing afterwards.
"""

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import SandboxError, SandboxExecutionError

if TYPE_CHECKING:
    from .base import Sandbox

__all__ = [
    "BRIDGE_CONNECT_FAILED",
    "BRIDGE_NOT_PREPARED",
    "CHECKSUM_MISMATCH",
    "CREDENTIAL_DELIVERY_UNSUPPORTED",
    "DELIVERY_UNSUPPORTED",
    "FETCH_FAILED",
    "FILESYSTEM_PRIMITIVES",
    "FUSE_FEATURE",
    "FUSE_UNAVAILABLE",
    "LOCAL_BRIDGE_MOUNT",
    "LOCAL_BRIDGE_NOT_A_MOUNT",
    "LOCAL_BRIDGE_UNSUPPORTED",
    "MANIFEST_DIRECTORIES",
    "MANIFEST_ENV",
    "MOUNT_MISSING",
    "MOUNT_NEEDS_RESTART",
    "MOUNT_PATH_MISSING",
    "SYNCHRONIZE_HINT",
    "TOKEN_ENV",
    "TOKEN_FILE_ENV",
    "URL_ENV",
    "ContentAttachmentError",
    "ContentAttachmentSpec",
    "ContentCapabilities",
    "ContentManifest",
    "CreationTimeMounts",
    "LocalBridgeCapability",
    "LocalBridgeSpec",
    "ManifestLocation",
    "MaterializeEntry",
    "MaterializeForm",
    "PreparedAttachment",
    "contents_environment",
    "environment_features",
    "install_manifest",
    "local_bridge_capability",
    "local_bridge_unsupported",
    "materialize",
    "not_ready",
    "path_exists",
    "path_is_mountpoint",
    "prepare_local_bridge",
    "probe",
    "probe_fuse",
    "ready",
    "remove_materialized",
    "start_bridge_mount",
    "stop_bridge_mount",
    "unsupported",
]

Delivery = Literal["mount", "local-bridge", "materialize", "client", "environment"]
AccessMode = Literal["mount", "python", "object-client"]
PreparedStatus = Literal["ready", "degraded", "failed"]
#: How one `materialize` entry is delivered: a `file` fetched from a signed
#: URL, a `git` checkout at a pinned revision, or `s3` python access.
MaterializeForm = Literal["file", "git", "s3"]

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
#: The environment the sandbox runs in cannot sustain the bridge filesystem.
LOCAL_BRIDGE_UNSUPPORTED = "LOCAL_BRIDGE_UNSUPPORTED"
#: The bridge filesystem could not reach the relay, or the relay refused it.
BRIDGE_CONNECT_FAILED = "BRIDGE_CONNECT_FAILED"
#: A `local-bridge` attachment that carries no bridge session to mount.
BRIDGE_NOT_PREPARED = "BRIDGE_NOT_PREPARED"
#: The environment claimed FUSE and the sandbox turned out not to have it.
FUSE_UNAVAILABLE = "FUSE_UNAVAILABLE"
#: A local bridge answered `ready` without a mount behind it. Never reported
#: as a mount: a copy is not a bridge, and this is what catches one.
LOCAL_BRIDGE_NOT_A_MOUNT = "LOCAL_BRIDGE_NOT_A_MOUNT"

#: The environment feature that means the bridge filesystem can run: fusepy
#: and `/dev/fuse` inside the sandbox.
FUSE_FEATURE = "fuse"
#: The capability a prepared local bridge carries, and the only one it may.
LOCAL_BRIDGE_MOUNT = "local-bridge-mount"
#: What to offer instead of a mount, wherever one is refused.
SYNCHRONIZE_HINT = (
    "Synchronize keeps a copy of the folder in step instead "
    "(`datalayer contents sync`); it is a copy, and is reported as one"
)

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

#: Every variable a manifest may export — and so every one an install must
#: take back when the manifest no longer says it.
CONTENTS_ENVIRONMENT_NAMES: tuple[str, ...] = (MANIFEST_ENV, URL_ENV, TOKEN_ENV, TOKEN_FILE_ENV)

#: What a manifest carries that must never be written into the manifest
#: JSON inside the sandbox: the sandbox credential, and a bridge's mount
#: token and session key.
MANIFEST_SECRETS: dict[str, Any] = {
    "token": True,
    "attachments": {"__all__": {"bridge": {"mount_token", "session_key"}}},
}

#: The line a snippet run inside the sandbox answers on, and the only line
#: :func:`probe` reads. Long and specific: the code's own output must not be
#: mistaken for an answer.
_MARKER = "__code_sandboxes_contents__:"


# --- Models ------------------------------------------------------------------


class MaterializeEntry(BaseModel):
    """One entry of a `materialize` attachment, in one of three forms.

    The FILE form — `source_url`, `path`, `sha256?`, `size?` — is a file the
    sandbox fetches from a signed URL. The GIT form — `git_url`, `revision`,
    `path`, `sha256?` — is a repository cloned INSIDE the sandbox at exactly
    that revision, and never at any other: a tutorial that changes under a
    person is not the tutorial they were promised. The S3 form — `bucket`,
    `region`, `prefix?`, `path` — delivers PYTHON ACCESS and nothing else: no
    bytes are fetched, the sandbox gets no bucket key, and the entry is
    written into the manifest so that `datalayer.contents` inside the
    sandbox opens it through the Contents service.

    Attributes:
        source_url: A signed URL the sandbox may fetch the bytes from. Signed
            by Contents for this attachment, and short-lived: it is not a
            credential a notebook can reuse for anything else.
        path: Where the entry goes — absolute, or relative to the
            attachment's `mount_path`. For a bucket it is the declared path
            the Environment named, kept as declared, even though nothing is
            mounted there.
        sha256: The digest the delivered bytes must have, when known: of the
            file, or of `git archive --format=tar <revision>`.
        size: The size in bytes, when known (file form).
        git_url: The repository to clone (git form).
        revision: The full commit sha to check out (git form).
        bucket: The bucket name (s3 form).
        region: The bucket's region (s3 form).
        prefix: A prefix inside the bucket, when the content is a subpath.
    """

    model_config = ConfigDict(extra="ignore")

    path: str
    source_url: str | None = None
    sha256: str | None = None
    size: int | None = None
    git_url: str | None = None
    revision: str | None = None
    bucket: str | None = None
    region: str | None = None
    prefix: str | None = None

    @model_validator(mode="after")
    def _one_form(self) -> MaterializeEntry:
        forms = [
            name
            for name, given in (
                ("file", self.source_url),
                ("git", self.git_url),
                ("s3", self.bucket),
            )
            if given
        ]
        if len(forms) != 1:
            raise ValueError(
                "a materialize entry is exactly one of a source_url (file), "
                "a git_url (git) or a bucket (s3)"
            )
        if forms[0] == "git" and not self.revision:
            raise ValueError("a git materialize entry needs the revision to check out")
        return self

    @property
    def form(self) -> MaterializeForm:
        if self.git_url:
            return "git"
        if self.bucket:
            return "s3"
        return "file"


class LocalBridgeSpec(BaseModel):
    """The bridge session a `local-bridge` attachment mounts through.

    What Contents answers on `POST /attachments/{uid}/prepare` for a
    `local-bridge` attachment, under `bridge`: the session's identity, the
    relay to connect to (`wss://host/bridges/{bridge_uid}`), and the mount
    token — the sandbox side's credential to that one session, short-lived
    and kept out of reprs, the manifest JSON and every command line.
    """

    model_config = ConfigDict(extra="ignore")

    bridge_uid: str
    relay_url: str
    mount_token: str = Field(repr=False)
    mount_path: str | None = None
    mode: Literal["ro", "rw"] = "ro"
    state: str | None = None
    #: The key that seals the frames end to end, so the relay forwards what
    #: it cannot read. Handed to each end on its own path; a secret, like
    #: the token, and kept out of the same places.
    session_key: str | None = Field(default=None, repr=False)


class ContentAttachmentSpec(BaseModel):
    """One attachment of the manifest: what the sandbox is to be given.

    Mirrors the `ContentAttachment` of the Contents contract, keeping the
    fields the sandbox side acts on. `delivery` is what is dispatched on;
    `access_mode` is set only for a Cloud Storage source and, together with
    `delivery="mount"`, names a BUCKET mount; `provider_resource_id` is set
    only for a volume and is what a provider-native mount is made from;
    `bridge` is set only for a `local-bridge` and is the session it mounts.
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
    bridge: LocalBridgeSpec | None = None

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


def local_bridge_capability(features: list[str] | tuple[str, ...]) -> LocalBridgeCapability:
    """What an ENVIRONMENT with these features can do with a local bridge.

    Per environment, not per provider: the same provider ships images that
    expose FUSE and images that do not, and the bridge filesystem runs only
    where `fuse` is among the features. Where it runs it does everything —
    `ro` and `rw`, reconnection while the token is good, and a clean
    unmount when the sandbox goes.
    """
    if FUSE_FEATURE in features:
        return LocalBridgeCapability(
            supported=True,
            required_features=[FUSE_FEATURE],
            read_only=True,
            read_write=True,
            reconnect=True,
            cleanup=True,
        )
    return LocalBridgeCapability(supported=False, required_features=[FUSE_FEATURE])


def environment_features(
    environments: Any, environment: str | None, explicit: list[str] | None = None
) -> list[str]:
    """The features of the environment a sandbox was created with.

    `explicit` wins when the caller passed the features along with the rest
    of the environment's metadata (the platform does, from its catalog);
    otherwise the environment is looked up by name in `environments` — the
    provider's `list_environments()` — and a name it does not list has no
    features: it may run, but nothing is claimed of it.
    """
    if explicit is not None:
        return [str(feature) for feature in explicit]
    for candidate in environments or []:
        if getattr(candidate, "name", None) == environment:
            return list(getattr(candidate, "features", None) or [])
    return []


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


def path_is_mountpoint(sandbox: Sandbox, path: str) -> bool:
    """Whether `path` is a MOUNT inside the sandbox, not merely a directory.

    A directory with files in it is what a copy looks like; a mountpoint is
    what a mount looks like. The difference is the whole of what a local
    bridge promises, so it is checked rather than assumed.
    """
    code = (
        "import os as _cs_os\n"
        f"_cs_mounted = _cs_os.path.ismount({path!r})\n"
        + _answer("{'mounted': _cs_mounted}")
        + "del _cs_os, _cs_mounted\n"
    )
    return bool(probe(sandbox, code)["mounted"])


def probe_fuse(sandbox: Sandbox) -> dict[str, Any]:
    """What the sandbox has of the `fuse` feature: fusepy and `/dev/fuse`.

    This is the probe behind the feature an environment advertises: a
    provider template that preinstalls fusepy and exposes the device answers
    `ok`, and one that does not is one on which no bridge mount is started.
    """
    code = (
        "import os as _cs_os, shutil as _cs_shutil\n"
        "try:\n"
        "    import fuse as _cs_fuse\n"
        "    _cs_fusepy = hasattr(_cs_fuse, 'FUSE')\n"
        "    del _cs_fuse\n"
        "except Exception:\n"
        "    _cs_fusepy = False\n"
        "_cs_device = _cs_os.path.exists('/dev/fuse')\n"
        "_cs_fusermount = _cs_shutil.which('fusermount3') or _cs_shutil.which('fusermount')\n"
        + _answer(
            "{'fusepy': _cs_fusepy, 'device': _cs_device, 'fusermount': _cs_fusermount, "
            "'ok': bool(_cs_fusepy and _cs_device)}"
        )
        + "del _cs_os, _cs_shutil, _cs_fusepy, _cs_device, _cs_fusermount\n"
    )
    return probe(sandbox, code)


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
    or a manifest with one more attachment all end in the same state — and
    a manifest WITHOUT a token, after one with, takes the token back: the
    variables are unset and the file is removed, so a credential the manifest
    no longer carries is nowhere in the sandbox either.
    """
    payload = {
        # The sandbox credential and a bridge's secrets stay out of the JSON:
        # the token goes to its own file, the bridge's to files of their own
        # when the mount is started, and neither is readable at the manifest.
        "manifest": manifest.model_dump(mode="json", exclude=MANIFEST_SECRETS),
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
        "if _cs_payload['token'] is None:\n"
        "    for _cs_candidate in _cs_payload['directories']:\n"
        "        _cs_stale = _cs_os.path.join(_cs_os.path.expanduser(_cs_candidate), "
        "_cs_payload['names'][1])\n"
        "        try:\n"
        "            _cs_os.remove(_cs_stale)\n"
        "        except OSError:\n"
        "            pass\n"
        "else:\n"
        "    _cs_token_path = _cs_os.path.join(_cs_chosen, _cs_payload['names'][1])\n"
        "    _cs_fd = _cs_os.open(_cs_token_path, _cs_os.O_WRONLY | _cs_os.O_CREAT "
        "| _cs_os.O_TRUNC, 0o600)\n"
        "    try:\n"
        "        _cs_os.write(_cs_fd, _cs_payload['token'].encode('utf-8'))\n"
        "    finally:\n"
        "        _cs_os.close(_cs_fd)\n"
        "    _cs_os.chmod(_cs_token_path, 0o600)\n"
        "_cs_env = _cs_payload['env']\n"
        "for _cs_name in _cs_env:\n"
        "    _cs_os.environ.pop(_cs_name, None)\n"
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
        "_cs_manifest_path, _cs_token_path, _cs_env, _cs_name\n"
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


#: The code that runs INSIDE the sandbox to deliver every entry of one
#: attachment, whatever its form. One snippet rather than three, so that the
#: report — one item per entry, in order — is answered on one line.
#:
#: A file is fetched beside its destination and moved into place only once
#: its digest has been checked. A git entry is cloned into a `.part`
#: directory beside its destination — `clone --no-checkout`, `fetch --depth 1
#: origin <sha>`, `checkout --detach <sha>` — HEAD is read back and must be
#: the sha, the archive digest is compared when one was given, and only then
#: is the directory moved into place: a failure removes the `.part` and
#: leaves nothing, never a partial checkout. A bucket entry does nothing
#: here: python access is the manifest already installed, and there is no
#: key in the sandbox to fetch with.
_MATERIALIZE_SNIPPET = (
    "import hashlib as _cs_hashlib, json as _cs_json, os as _cs_os, shutil as _cs_shutil\n"
    "import subprocess as _cs_subprocess, urllib.request as _cs_request\n"
    "_cs_payload = _cs_json.loads(__CS_PAYLOAD__)\n"
    "def _cs_digest(path):\n"
    "    digest = _cs_hashlib.sha256()\n"
    "    with open(path, 'rb') as handle:\n"
    "        for chunk in iter(lambda: handle.read(1 << 20), b''):\n"
    "            digest.update(chunk)\n"
    "    return digest.hexdigest()\n"
    "def _cs_git(*args, cwd=None):\n"
    "    done = _cs_subprocess.run(\n"
    "        ['git', *args], cwd=cwd, capture_output=True,\n"
    "        timeout=_cs_payload['timeout'], check=False,\n"
    "    )\n"
    "    if done.returncode != 0:\n"
    "        raise RuntimeError(\n"
    "            'git ' + ' '.join(args) + ' failed: '\n"
    "            + done.stderr.decode('utf-8', 'replace').strip()\n"
    "        )\n"
    "    return done.stdout\n"
    "def _cs_head(path):\n"
    "    return _cs_git('-C', path, 'rev-parse', 'HEAD').decode('utf-8', 'replace').strip()\n"
    "def _cs_archive_digest(path, revision):\n"
    "    return _cs_hashlib.sha256(\n"
    "        _cs_git('-C', path, 'archive', '--format=tar', revision)\n"
    "    ).hexdigest()\n"
    "def _cs_checkout_is(path, entry):\n"
    "    if not _cs_os.path.isdir(_cs_os.path.join(path, '.git')):\n"
    "        return False\n"
    "    try:\n"
    "        if _cs_head(path) != entry['revision']:\n"
    "            return False\n"
    "        wanted = entry.get('sha256')\n"
    "        return wanted is None or _cs_archive_digest(path, entry['revision']) == wanted\n"
    "    except Exception:\n"
    "        return False\n"
    "def _cs_file(entry, path, item):\n"
    "    wanted = entry.get('sha256')\n"
    "    if (\n"
    "        _cs_payload['reconcile']\n"
    "        and _cs_os.path.isfile(path)\n"
    "        and (wanted is None or _cs_digest(path) == wanted)\n"
    "    ):\n"
    "        item['status'] = 'present'\n"
    "    else:\n"
    "        _cs_os.makedirs(_cs_os.path.dirname(path) or '/', exist_ok=True)\n"
    "        part = path + '.part'\n"
    "        try:\n"
    "            with _cs_request.urlopen(\n"
    "                entry['source_url'], timeout=_cs_payload['timeout']\n"
    "            ) as source, open(part, 'wb') as target:\n"
    "                for chunk in iter(lambda: source.read(1 << 20), b''):\n"
    "                    target.write(chunk)\n"
    "        except BaseException:\n"
    "            # A download cut short leaves nothing: the next attempt starts\n"
    "            # clean rather than beside a stale partial file.\n"
    "            try:\n"
    "                _cs_os.remove(part)\n"
    "            except OSError:\n"
    "                pass\n"
    "            raise\n"
    "        if wanted is not None and _cs_digest(part) != wanted:\n"
    "            _cs_os.remove(part)\n"
    "            raise ValueError('CHECKSUM_MISMATCH')\n"
    "        _cs_os.replace(part, path)\n"
    "    item['size'] = _cs_os.path.getsize(path)\n"
    "def _cs_checkout(entry, path, item):\n"
    "    revision = entry['revision']\n"
    "    if _cs_payload['reconcile'] and _cs_checkout_is(path, entry):\n"
    "        item['status'] = 'present'\n"
    "        item['revision'] = revision\n"
    "        return\n"
    "    _cs_os.makedirs(_cs_os.path.dirname(path) or '/', exist_ok=True)\n"
    "    part = path + '.part'\n"
    "    _cs_shutil.rmtree(part, ignore_errors=True)\n"
    "    try:\n"
    "        _cs_git('clone', '--no-checkout', entry['git_url'], part)\n"
    "        _cs_git('-C', part, 'fetch', '--depth', '1', 'origin', revision)\n"
    "        _cs_git('-C', part, 'checkout', '--detach', revision)\n"
    "        head = _cs_head(part)\n"
    "        if head != revision:\n"
    "            raise RuntimeError(\n"
    "                'HEAD is ' + head + ' after checking out ' + revision\n"
    "            )\n"
    "        wanted = entry.get('sha256')\n"
    "        if wanted is not None and _cs_archive_digest(part, revision) != wanted:\n"
    "            raise ValueError('CHECKSUM_MISMATCH')\n"
    "    except BaseException:\n"
    "        _cs_shutil.rmtree(part, ignore_errors=True)\n"
    "        raise\n"
    "    if _cs_os.path.isdir(path) and not _cs_os.path.islink(path):\n"
    "        _cs_shutil.rmtree(path)\n"
    "    elif _cs_os.path.lexists(path):\n"
    "        _cs_os.remove(path)\n"
    "    _cs_os.replace(part, path)\n"
    "    item['revision'] = revision\n"
    "_cs_report = []\n"
    "for _cs_entry in _cs_payload['entries']:\n"
    "    _cs_path = _cs_entry['path']\n"
    "    _cs_item = {'path': _cs_path, 'form': _cs_entry['form'], 'status': 'written'}\n"
    "    try:\n"
    "        if _cs_entry['form'] == 'git':\n"
    "            _cs_checkout(_cs_entry, _cs_path, _cs_item)\n"
    "        elif _cs_entry['form'] == 's3':\n"
    "            _cs_item['status'] = 'python'\n"
    "        else:\n"
    "            _cs_file(_cs_entry, _cs_path, _cs_item)\n"
    "    except Exception as _cs_error:\n"
    "        _cs_item['status'] = 'failed'\n"
    "        _cs_item['error'] = (\n"
    "            'CHECKSUM_MISMATCH'\n"
    "            if str(_cs_error) == 'CHECKSUM_MISMATCH'\n"
    "            else 'FETCH_FAILED'\n"
    "        )\n"
    "        _cs_item['detail'] = type(_cs_error).__name__ + ': ' + str(_cs_error)\n"
    "    _cs_report.append(_cs_item)\n"
)

_MATERIALIZE_CLEANUP = (
    "del _cs_hashlib, _cs_json, _cs_os, _cs_shutil, _cs_subprocess, _cs_request, "
    "_cs_payload, _cs_digest, _cs_git, _cs_head, _cs_archive_digest, _cs_checkout_is, "
    "_cs_file, _cs_checkout, _cs_report\n"
)


def _materialize_payload(
    destinations: list[tuple[MaterializeEntry, str]], *, reconcile: bool, timeout: float
) -> dict[str, Any]:
    entries = []
    for entry, path in destinations:
        item: dict[str, Any] = {"path": path, "form": entry.form, "sha256": entry.sha256}
        if entry.form == "file":
            item["source_url"] = entry.source_url
        elif entry.form == "git":
            item.update(git_url=entry.git_url, revision=entry.revision)
        else:
            item.update(bucket=entry.bucket, region=entry.region, prefix=entry.prefix)
        entries.append(item)
    return {"reconcile": reconcile, "timeout": timeout, "entries": entries}


def materialize(
    sandbox: Sandbox,
    spec: ContentAttachmentSpec,
    *,
    reconcile: bool,
    timeout: float | None = None,
) -> PreparedAttachment:
    """Deliver the entries of a `materialize` attachment inside the sandbox.

    A file is fetched from its signed URL by the sandbox itself, written
    beside its destination and moved into place only once its digest — when
    the manifest gave one — has been checked, so a half-fetched or corrupt
    file is never what a notebook opens. A git entry is cloned at its pinned
    revision, verified, and moved into place the same way — or removed
    entirely: there is no such thing as a partial checkout here. A bucket
    entry is python access: nothing is fetched, and the manifest already
    installed is what the Contents client inside opens it with.

    With `reconcile`, a file that is already there with the right digest,
    and a checkout already at the right revision, are left alone; without
    it, every entry is delivered afresh.
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

    capabilities: list[str] = []
    forms = {entry.form for entry, _path in destinations}
    if forms & {"file", "git"}:
        capabilities.append("materialize")
    if "s3" in forms:
        capabilities.append("python")
    payload = _materialize_payload(destinations, reconcile=reconcile, timeout=timeout or 300)
    if forms == {"s3"}:
        # Nothing to run: python access is the manifest, and it is installed.
        return ready(
            spec,
            capabilities=capabilities,
            detail=(
                f"{len(destinations)} bucket(s) reachable through the Contents service "
                "from code; nothing fetched, no bucket credential in the sandbox"
            ),
        )
    code = (
        _MATERIALIZE_SNIPPET.replace("__CS_PAYLOAD__", repr(json.dumps(payload)))
        + _answer("{'entries': _cs_report}")
        + _MATERIALIZE_CLEANUP
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
    python = sum(1 for item in answer["entries"] if item.get("status") == "python")
    checkouts = sum(1 for item in answer["entries"] if item.get("form") == "git")
    written = len(answer["entries"]) - present - python
    detail = f"{written} entry(ies) materialized, {present} already present"
    if checkouts:
        detail += f", {checkouts} git checkout(s) at their pinned revision"
    if python:
        detail += f", {python} bucket(s) reachable from code through Contents"
    return ready(spec, capabilities=capabilities, detail=detail)


def remove_materialized(sandbox: Sandbox, spec: ContentAttachmentSpec) -> None:
    """Remove what a `materialize` attachment put in the sandbox.

    Files, and the checkout directories this package made — never the
    directory above a file, which may be a mount or shared with something
    else — and never anything at the source: the source is the service's,
    and detaching is not deleting. A bucket entry put nothing there.
    """
    try:
        destinations = _destinations(spec)
    except ValueError:
        return
    targets = [
        {"path": path, "tree": entry.form == "git"}
        for entry, path in destinations
        if entry.form != "s3"
    ]
    if not targets:
        return
    code = (
        "import json as _cs_json, os as _cs_os, shutil as _cs_shutil\n"
        f"_cs_targets = _cs_json.loads({json.dumps(targets)!r})\n"
        "_cs_removed = []\n"
        "for _cs_target in _cs_targets:\n"
        "    for _cs_candidate in (_cs_target['path'], _cs_target['path'] + '.part'):\n"
        "        try:\n"
        "            if _cs_target['tree'] and _cs_os.path.isdir(_cs_candidate) "
        "and not _cs_os.path.islink(_cs_candidate):\n"
        "                _cs_shutil.rmtree(_cs_candidate)\n"
        "            else:\n"
        "                _cs_os.remove(_cs_candidate)\n"
        "            _cs_removed.append(_cs_candidate)\n"
        "        except OSError:\n"
        "            pass\n"
        + _answer("{'removed': _cs_removed}")
        + "del _cs_json, _cs_os, _cs_shutil, _cs_targets, _cs_removed\n"
    )
    probe(sandbox, code)


# --- A person's own folder, bridged in -----------------------------------------


def local_bridge_unsupported(
    spec: ContentAttachmentSpec,
    *,
    provider: str,
    environment: str | None,
    features: list[str],
) -> PreparedAttachment:
    """The environment cannot run the bridge filesystem: say so, and what to do instead.

    Never a copy dressed as a mount. The answer names the environment and
    the provider, what the environment would need, and Synchronize — the
    copy workflow, called what it is.
    """
    have = ", ".join(features) if features else "none"
    return not_ready(
        spec,
        LOCAL_BRIDGE_UNSUPPORTED,
        f"environment {environment or '?'} on {provider} cannot mount a local folder: "
        f"a local bridge needs the {FUSE_FEATURE!r} feature (fusepy and /dev/fuse in "
        f"the sandbox) and the environment advertises {have}. {SYNCHRONIZE_HINT}",
    )


#: Where the bridge filesystem module is written inside the sandbox, beside
#: the manifest, and what its files are called.
BRIDGE_MODULE_FILENAME = "bridge_mount.py"


def _bridge_module_source() -> str:
    """The sandbox-side module, as text, to be written where it will run."""
    from pathlib import Path

    return Path(__file__).with_name(BRIDGE_MODULE_FILENAME).read_text(encoding="utf-8")


#: The code that runs INSIDE the sandbox to start the bridge filesystem.
#:
#: It writes the module and the token — the token to a file only the owner
#: can read, never onto a command line — starts the mount as a process of its
#: own session, so it outlives the cell, and waits for the one line the
#: launcher answers on its stdout: `connected`, or `failed` with why. A
#: launcher that says nothing in time is a failure too, and is killed. On a
#: reconcile, a path that is already a mountpoint is left as it is.
_BRIDGE_MOUNT_SNIPPET = (
    "import json as _cs_json, os as _cs_os, select as _cs_select, subprocess as _cs_subprocess\n"
    "import sys as _cs_sys, time as _cs_time\n"
    "_cs_payload = _cs_json.loads(__CS_PAYLOAD__)\n"
    "_cs_report = {}\n"
    "_cs_mount_path = _cs_payload['mount_path']\n"
    "if _cs_payload['reconcile'] and _cs_os.path.ismount(_cs_mount_path):\n"
    "    _cs_report = {'status': 'connected', 'already_mounted': True}\n"
    "    _cs_status_path = None\n"
    "    for _cs_candidate in _cs_payload['directories']:\n"
    "        _cs_candidate_status = _cs_os.path.join(\n"
    "            _cs_os.path.expanduser(_cs_candidate), _cs_payload['names']['status'])\n"
    "        if _cs_os.path.isfile(_cs_candidate_status):\n"
    "            _cs_status_path = _cs_candidate_status\n"
    "            break\n"
    "    if _cs_status_path:\n"
    "        try:\n"
    "            with open(_cs_status_path) as _cs_handle:\n"
    "                _cs_report['state'] = _cs_json.load(_cs_handle).get('state')\n"
    "        except Exception:\n"
    "            pass\n"
    "else:\n"
    "    _cs_chosen = None\n"
    "    for _cs_candidate in _cs_payload['directories']:\n"
    "        _cs_directory = _cs_os.path.expanduser(_cs_candidate)\n"
    "        try:\n"
    "            _cs_os.makedirs(_cs_directory, exist_ok=True)\n"
    "            _cs_module = _cs_os.path.join(_cs_directory, _cs_payload['names']['module'])\n"
    "            with open(_cs_module + '.part', 'w') as _cs_handle:\n"
    "                _cs_handle.write(_cs_payload['module_source'])\n"
    "            _cs_os.replace(_cs_module + '.part', _cs_module)\n"
    "            _cs_chosen = _cs_directory\n"
    "            break\n"
    "        except OSError:\n"
    "            continue\n"
    "    if _cs_chosen is None:\n"
    "        raise OSError('none of ' + ', '.join(_cs_payload['directories']) + ' is writable')\n"
    "    _cs_token_path = _cs_os.path.join(_cs_chosen, _cs_payload['names']['token'])\n"
    "    _cs_fd = _cs_os.open(_cs_token_path, _cs_os.O_WRONLY | _cs_os.O_CREAT "
    "| _cs_os.O_TRUNC, 0o600)\n"
    "    try:\n"
    "        _cs_os.write(_cs_fd, _cs_payload['mount_token'].encode('utf-8'))\n"
    "    finally:\n"
    "        _cs_os.close(_cs_fd)\n"
    "    _cs_os.chmod(_cs_token_path, 0o600)\n"
    "    _cs_session_path = None\n"
    "    if _cs_payload.get('session_key'):\n"
    "        _cs_session_path = _cs_os.path.join(_cs_chosen, _cs_payload['names']['session'])\n"
    "        _cs_fd = _cs_os.open(_cs_session_path, _cs_os.O_WRONLY | _cs_os.O_CREAT "
    "| _cs_os.O_TRUNC, 0o600)\n"
    "        try:\n"
    "            _cs_os.write(_cs_fd, _cs_payload['session_key'].encode('utf-8'))\n"
    "        finally:\n"
    "            _cs_os.close(_cs_fd)\n"
    "        _cs_os.chmod(_cs_session_path, 0o600)\n"
    "    _cs_status = _cs_os.path.join(_cs_chosen, _cs_payload['names']['status'])\n"
    "    _cs_log = open(_cs_os.path.join(_cs_chosen, _cs_payload['names']['log']), 'ab')\n"
    "    _cs_os.makedirs(_cs_mount_path, exist_ok=True)\n"
    "    _cs_argv = [_cs_sys.executable, _cs_module,\n"
    "                '--relay-url', _cs_payload['relay_url'],\n"
    "                '--mount-path', _cs_mount_path,\n"
    "                '--mode', _cs_payload['mode'],\n"
    "                '--token-file', _cs_token_path,\n"
    "                '--status-file', _cs_status]\n"
    "    if _cs_payload.get('bridge_uid'):\n"
    "        _cs_argv += ['--bridge-uid', _cs_payload['bridge_uid']]\n"
    "    if _cs_session_path:\n"
    "        _cs_argv += ['--session-key-file', _cs_session_path]\n"
    "    _cs_process = _cs_subprocess.Popen(\n"
    "        _cs_argv,\n"
    "        stdout=_cs_subprocess.PIPE, stderr=_cs_log, stdin=_cs_subprocess.DEVNULL,\n"
    "        start_new_session=True, close_fds=True,\n"
    "    )\n"
    "    _cs_log.close()\n"
    "    _cs_line = b''\n"
    "    _cs_deadline = _cs_time.monotonic() + _cs_payload['timeout']\n"
    "    while _cs_time.monotonic() < _cs_deadline:\n"
    "        _cs_ready, _, _ = _cs_select.select([_cs_process.stdout], [], [], 0.2)\n"
    "        if _cs_ready:\n"
    "            _cs_line = _cs_process.stdout.readline()\n"
    "            break\n"
    "        if _cs_process.poll() is not None:\n"
    "            _cs_line = _cs_process.stdout.readline()\n"
    "            break\n"
    "    if _cs_line.strip():\n"
    "        try:\n"
    "            _cs_report = _cs_json.loads(_cs_line.decode('utf-8', 'replace'))\n"
    "        except ValueError:\n"
    "            _cs_report = {'status': 'failed', 'error': 'BRIDGE_CONNECT_FAILED',\n"
    "                          'detail': 'unreadable answer: ' + repr(_cs_line[:200])}\n"
    "    else:\n"
    "        _cs_report = {'status': 'failed', 'error': 'BRIDGE_CONNECT_FAILED',\n"
    "                      'detail': 'no answer from the bridge mount within '\n"
    "                      + str(_cs_payload['timeout']) + 's'}\n"
    "    if _cs_report.get('status') != 'connected':\n"
    "        try:\n"
    "            _cs_process.kill()\n"
    "        except Exception:\n"
    "            pass\n"
    "        for _cs_secret in (_cs_token_path, _cs_session_path):\n"
    "            try:\n"
    "                if _cs_secret:\n"
    "                    _cs_os.remove(_cs_secret)\n"
    "            except OSError:\n"
    "                pass\n"
    "    _cs_report['pid'] = _cs_process.pid\n"
    "    _cs_report['module'] = _cs_module\n"
)

_BRIDGE_MOUNT_CLEANUP = (
    "for _cs_name in ('_cs_json', '_cs_os', '_cs_select', '_cs_subprocess', '_cs_sys', "
    "'_cs_time', '_cs_payload', '_cs_report', '_cs_mount_path', '_cs_status_path', "
    "'_cs_candidate', '_cs_candidate_status', '_cs_handle', '_cs_chosen', '_cs_directory', "
    "'_cs_module', '_cs_token_path', '_cs_session_path', '_cs_secret', '_cs_argv', '_cs_fd', "
    "'_cs_status', '_cs_log', '_cs_process', '_cs_line', '_cs_deadline', '_cs_ready'):\n"
    "    globals().pop(_cs_name, None)\n"
    "del _cs_name\n"
)


def _bridge_names(spec: ContentAttachmentSpec) -> dict[str, str]:
    uid = spec.bridge.bridge_uid if spec.bridge else spec.uid
    return {
        "module": BRIDGE_MODULE_FILENAME,
        "token": f"bridge-{uid}.token",
        "session": f"bridge-{uid}.session",
        "status": f"bridge-{uid}.status",
        "log": f"bridge-{uid}.log",
    }


def start_bridge_mount(
    sandbox: Sandbox,
    spec: ContentAttachmentSpec,
    *,
    reconcile: bool,
    timeout: float = 60.0,
    directories: tuple[str, ...] | None = None,
) -> PreparedAttachment:
    """Start the bridge filesystem inside the sandbox and say whether it connected.

    `ready` only on `connected` — the launcher's word that the tunnel is up
    and the mount is being made — and `failed` otherwise, with the
    launcher's own reason (`BRIDGE_CONNECT_FAILED`, `FUSE_UNAVAILABLE`).
    The caller (the base class) checks afterwards that the path really is a
    mountpoint; this reports what the launcher said.
    """
    bridge = spec.bridge
    if bridge is None:
        return not_ready(
            spec,
            BRIDGE_NOT_PREPARED,
            "the attachment carries no bridge session: Contents prepares one "
            "(relay, mount token) before the sandbox can mount it",
        )
    mount_path = spec.mount_path or bridge.mount_path
    if not mount_path:
        return not_ready(spec, MOUNT_PATH_MISSING, "a local bridge needs a mount_path")
    payload = {
        "reconcile": reconcile,
        "timeout": timeout,
        "directories": list(directories or MANIFEST_DIRECTORIES),
        "names": _bridge_names(spec),
        "module_source": _bridge_module_source(),
        "bridge_uid": bridge.bridge_uid,
        "relay_url": bridge.relay_url,
        "mount_token": bridge.mount_token,
        "session_key": bridge.session_key,
        "mount_path": mount_path,
        "mode": bridge.mode or spec.mode,
    }
    code = (
        _BRIDGE_MOUNT_SNIPPET.replace("__CS_PAYLOAD__", repr(json.dumps(payload)))
        + _answer("_cs_report")
        + _BRIDGE_MOUNT_CLEANUP
    )
    answer = probe(sandbox, code, timeout=timeout + 15)
    if answer.get("status") == "connected":
        detail = "already mounted" if answer.get("already_mounted") else "bridge mount connected"
        if answer.get("state") and answer["state"] != "connected":
            return not_ready(
                spec,
                BRIDGE_CONNECT_FAILED,
                f"the bridge mount at {mount_path} is {answer['state']}",
            )
        return ready(spec, capabilities=[LOCAL_BRIDGE_MOUNT], detail=detail)
    code_name = str(answer.get("error") or BRIDGE_CONNECT_FAILED)
    detail = str(answer.get("detail") or "the bridge mount did not connect")
    if answer.get("state"):
        detail += f" (bridge {answer['state']})"
    return not_ready(spec, code_name, detail)


def stop_bridge_mount(
    sandbox: Sandbox,
    spec: ContentAttachmentSpec,
    *,
    directories: tuple[str, ...] | None = None,
) -> None:
    """Unmount the bridge filesystem and end its process; forget the token.

    Detaching takes away what was delivered — the mount — and nothing at
    the source: the person's folder is theirs, and unmounting is not
    deleting.
    """
    mount_path = spec.mount_path or (spec.bridge.mount_path if spec.bridge else None)
    if not mount_path:
        return
    payload = {
        "mount_path": mount_path,
        "directories": list(directories or MANIFEST_DIRECTORIES),
        "names": _bridge_names(spec),
    }
    code = (
        "import json as _cs_json, os as _cs_os, shutil as _cs_shutil, signal as _cs_signal\n"
        "import subprocess as _cs_subprocess\n"
        f"_cs_payload = _cs_json.loads({json.dumps(payload)!r})\n"
        "_cs_done = {'unmounted': False, 'killed': False}\n"
        "for _cs_command in (['fusermount3', '-u', '-z'], ['fusermount', '-u', '-z'], "
        "['umount', '-l']):\n"
        "    if _cs_shutil.which(_cs_command[0]) is None:\n"
        "        continue\n"
        "    _cs_result = _cs_subprocess.run(_cs_command + [_cs_payload['mount_path']], "
        "capture_output=True, check=False, timeout=30)\n"
        "    if _cs_result.returncode == 0:\n"
        "        _cs_done['unmounted'] = True\n"
        "        break\n"
        "for _cs_candidate in _cs_payload['directories']:\n"
        "    _cs_directory = _cs_os.path.expanduser(_cs_candidate)\n"
        "    _cs_status = _cs_os.path.join(_cs_directory, _cs_payload['names']['status'])\n"
        "    if _cs_os.path.isfile(_cs_status):\n"
        "        try:\n"
        "            with open(_cs_status) as _cs_handle:\n"
        "                _cs_pid = int(_cs_json.load(_cs_handle).get('pid') or 0)\n"
        "            if _cs_pid > 0:\n"
        "                _cs_os.kill(_cs_pid, _cs_signal.SIGTERM)\n"
        "                _cs_done['killed'] = True\n"
        "        except (OSError, ValueError):\n"
        "            pass\n"
        "    for _cs_name in ('token', 'session', 'status'):\n"
        "        try:\n"
        "            _cs_os.remove(\n"
        "                _cs_os.path.join(_cs_directory, _cs_payload['names'][_cs_name]))\n"
        "        except OSError:\n"
        "            pass\n"
        + _answer("_cs_done")
        + "for _cs_name in ('_cs_json', '_cs_os', '_cs_shutil', '_cs_signal', '_cs_subprocess', "
        "'_cs_payload', '_cs_done', '_cs_command', '_cs_result', '_cs_candidate', "
        "'_cs_directory', '_cs_status', '_cs_handle', '_cs_pid'):\n"
        "    globals().pop(_cs_name, None)\n"
        "del _cs_name\n"
    )
    probe(sandbox, code)


def prepare_local_bridge(
    sandbox: Sandbox,
    spec: ContentAttachmentSpec,
    *,
    provider: str,
    environment: str | None,
    features: list[str],
    reconcile: bool,
) -> PreparedAttachment:
    """A `local-bridge` attachment on a provider whose sandbox mounts it itself.

    Refused, with Synchronize offered, unless the environment advertises
    `fuse`; started inside the sandbox otherwise, and `ready` only once the
    launcher says the tunnel is connected.
    """
    if not local_bridge_capability(features).supported:
        return local_bridge_unsupported(
            spec, provider=provider, environment=environment, features=features
        )
    return start_bridge_mount(sandbox, spec, reconcile=reconcile)


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
        """Forget the request. The volume itself is never touched.

        Only a volume mount was recorded, so only one is forgotten: files
        materialized under the same path are another attachment, and
        detaching them must not cancel the mount.
        """
        if spec.is_volume_mount and spec.mount_path:
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
        """Answer for a `mount` or `environment` attachment on a running sandbox."""
        if spec.delivery == "environment":
            # The platform mounts an Environment content; this provider has
            # no such mount, and the manifest gave no entries saying how the
            # declared path is to be honoured here. Not a restart matter.
            return unsupported(
                spec,
                provider,
                f"{provider} sandboxes cannot mount an Environment content; at this "
                "provider a git content is a materialized checkout and a bucket is "
                "python access, and the manifest carried neither for "
                f"{spec.mount_path or spec.uid}",
            )
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
