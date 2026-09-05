# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Building an Environment's `contents:` manifest into a provider artifact.

An Environment may carry a build manifest: immutable files — a dataset
sample, a model card, a tutorial notebook — each named by where it comes
from, where it goes and what its sha256 must be. Those files are not
attached at launch; they are BAKED into the artifact the sandbox starts
from, so they are there before the kernel is, on every provider: the
Datalayer runtime image, a Daytona snapshot, an E2B template, a Modal image.

The build is the same everywhere: one verified fetch per entry —

    curl -fsSL <source> -o <destination> && echo "<sha256>  <destination>" | sha256sum -c

— and a checksum that does not match FAILS THE BUILD, said in the artifact's
own words (`sha256sum -c`), never a file quietly left in place with the
wrong bytes. Then one more file, :data:`ENVIRONMENT_CONTENTS_MANIFEST`,
listing `{path, sha256, source}` for every entry, so that what an artifact
carries can be read back from inside the sandbox — by attachment
diagnostics, by a Contents client, by a person — without trusting anything
but the artifact itself.

What differs is what each provider calls an artifact:

- **Datalayer** takes a Dockerfile fragment — the `RUN` lines — returned as
  text and written to a path the caller names, to be appended to the runtime
  image's Dockerfile.
- **Modal** takes a `modal.Image`, built here with `run_commands` of the
  same fetches; the SDK is imported when the build is made, not before.
- **Daytona** takes an `Image` of the Daytona SDK's builder, with the same
  commands, and a snapshot created from it; the snapshot name is the
  reference.
- **E2B** has no builder this package can rely on offline: the Dockerfile
  the `e2b template build` CLI consumes is written, and its path is the
  reference.

Nothing here touches a provider's network except the Daytona snapshot
creation, and that only through a client the caller may hand in.
"""

from __future__ import annotations

import json
import posixpath
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import SandboxConfigurationError

if TYPE_CHECKING:
    from .base import Sandbox

__all__ = [
    "ENVIRONMENT_CONTENTS_MANIFEST",
    "BuildEntry",
    "BuildProvider",
    "BuiltArtifact",
    "EnvironmentBuild",
    "build_artifact",
    "build_commands",
    "dockerfile_fragment",
    "environment_contents_manifest",
    "installed_environment_contents",
]

BuildProvider = Literal["datalayer", "daytona", "e2b", "modal"]

#: Where every artifact lists what it carries: `{path, sha256, source}` per
#: entry. Beside the attachment manifest a running sandbox is given, and
#: read the same way.
ENVIRONMENT_CONTENTS_MANIFEST = "/etc/datalayer/environment-contents.json"

#: What the Datalayer fragment and the E2B Dockerfile both need from the
#: image they extend: a shell, `curl`, and `sha256sum` (coreutils).
_REQUIRED_TOOLS = ("sh", "curl", "sha256sum")


# --- Models ------------------------------------------------------------------


class BuildEntry(BaseModel):
    """One immutable file of an Environment's `contents:` build manifest."""

    model_config = ConfigDict(extra="ignore")

    source_uri: str
    destination_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = None


class EnvironmentBuild(BaseModel):
    """An Environment's build manifest, for one provider."""

    model_config = ConfigDict(extra="ignore")

    environment: str
    provider: BuildProvider
    entries: list[BuildEntry] = Field(default_factory=list)
    #: The image the artifact extends, when the provider takes one. Left to
    #: the provider's own default otherwise.
    base_image: str | None = None


class BuiltArtifact(BaseModel):
    """What the build produced, and how the provider names it.

    `reference` is what the provider is asked for at launch — an image, a
    snapshot, a Dockerfile path. `dockerfile` is the text, for the providers
    whose artifact is text; `image` is the SDK object, for those whose
    artifact is one. `manifest_path` is where, inside a sandbox started from
    the artifact, the list of what it carries is found.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    provider: BuildProvider
    reference: str
    manifest_path: str = ENVIRONMENT_CONTENTS_MANIFEST
    entries: list[BuildEntry] = Field(default_factory=list)
    dockerfile: str | None = None
    image: Any = Field(default=None, exclude=True)


# --- The commands, the same on every provider --------------------------------


def _absolute(path: str) -> str:
    if not posixpath.isabs(path):
        raise SandboxConfigurationError(
            f"a build entry's destination must be absolute; {path!r} is not"
        )
    return posixpath.normpath(path)


def environment_contents_manifest(build: EnvironmentBuild) -> dict[str, Any]:
    """What :data:`ENVIRONMENT_CONTENTS_MANIFEST` says inside the artifact."""
    return {
        "contract_version": "v1",
        "environment": build.environment,
        "provider": build.provider,
        "entries": [
            {
                "path": _absolute(entry.destination_path),
                "sha256": entry.sha256,
                "source": entry.source_uri,
                **({"size": entry.size_bytes} if entry.size_bytes is not None else {}),
            }
            for entry in build.entries
        ],
    }


def fetch_command(entry: BuildEntry) -> str:
    """Fetch one entry and verify it, or fail.

    `curl -f` fails on an HTTP error rather than saving the error page;
    `sha256sum -c` fails when the digest is not the one the manifest gave.
    Either failure fails the build: the file is not silently wrong.
    """
    destination = _absolute(entry.destination_path)
    directory = posixpath.dirname(destination) or "/"
    return (
        f"mkdir -p {shlex.quote(directory)}"
        f" && curl -fsSL {shlex.quote(entry.source_uri)} -o {shlex.quote(destination)}"
        f' && echo "{entry.sha256}  {destination}" | sha256sum -c'
    )


def manifest_command(build: EnvironmentBuild) -> str:
    """Write :data:`ENVIRONMENT_CONTENTS_MANIFEST` into the artifact."""
    directory = posixpath.dirname(ENVIRONMENT_CONTENTS_MANIFEST)
    # One line: a Dockerfile `RUN` cannot carry a raw newline.
    text = json.dumps(environment_contents_manifest(build), sort_keys=True)
    return (
        f"mkdir -p {shlex.quote(directory)}"
        f" && printf '%s\\n' {shlex.quote(text)} > {shlex.quote(ENVIRONMENT_CONTENTS_MANIFEST)}"
    )


def build_commands(build: EnvironmentBuild) -> list[str]:
    """Every shell command of the build, in order: the fetches, then the manifest."""
    commands = [fetch_command(entry) for entry in build.entries]
    commands.append(manifest_command(build))
    return commands


def dockerfile_fragment(build: EnvironmentBuild, *, base_image: str | None = None) -> str:
    """The `RUN` lines of the build, as a Dockerfile fragment.

    With `base_image`, a complete Dockerfile — a `FROM` line first — which is
    what E2B's CLI consumes; without it, the fragment a runtime image's
    Dockerfile appends. Either way the requirement on the image is stated in
    the fragment: `curl` and `sha256sum` must be there, and a missing
    checksum FAILS the build.
    """
    lines = [
        f"# Environment contents of {build.environment} for {build.provider}, "
        "built by code_sandboxes.",
        "# Each file is fetched and verified; a checksum that does not match fails the build",
        f"# (`sha256sum -c`). Needs {', '.join(_REQUIRED_TOOLS)} in the image.",
    ]
    if base_image:
        lines.insert(0, f"FROM {base_image}")
    for command in build_commands(build):
        lines.append(f"RUN {command}")
    return "\n".join(lines) + "\n"


# --- Per provider ---------------------------------------------------------------


def _write(text: str, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path)


def _build_datalayer(build: EnvironmentBuild, output_path: str | Path | None) -> BuiltArtifact:
    text = dockerfile_fragment(build, base_image=build.base_image)
    reference = _write(text, output_path) if output_path else f"dockerfile://{build.environment}"
    return BuiltArtifact(
        provider="datalayer", reference=reference, entries=list(build.entries), dockerfile=text
    )


def _build_e2b(build: EnvironmentBuild, output_path: str | Path | None) -> BuiltArtifact:
    if not output_path:
        raise SandboxConfigurationError(
            "an E2B template is built by the `e2b template build` CLI from a Dockerfile; "
            "name the path to write it to"
        )
    text = dockerfile_fragment(
        build, base_image=build.base_image or "e2bdev/code-interpreter:latest"
    )
    return BuiltArtifact(
        provider="e2b",
        reference=_write(text, output_path),
        entries=list(build.entries),
        dockerfile=text,
    )


def _build_modal(build: EnvironmentBuild) -> BuiltArtifact:
    try:
        import modal
    except ImportError as exc:
        raise SandboxConfigurationError(
            "modal is required to build a Modal image. Install it with: pip install modal"
        ) from exc
    if build.base_image:
        image = modal.Image.from_registry(build.base_image)
    else:
        image = modal.Image.debian_slim()
    # The image needs curl; debian_slim has no curl of its own.
    image = image.apt_install("curl", "ca-certificates")
    image = image.run_commands(*build_commands(build))
    reference = getattr(image, "object_id", None) or f"modal://{build.environment}"
    return BuiltArtifact(
        provider="modal", reference=str(reference), entries=list(build.entries), image=image
    )


def _build_daytona(
    build: EnvironmentBuild, *, snapshot_name: str | None, daytona_client: Any | None
) -> BuiltArtifact:
    try:
        import daytona
    except ImportError as exc:
        raise SandboxConfigurationError(
            "daytona is required to build a Daytona snapshot. Install it with: pip install daytona"
        ) from exc
    if build.base_image:
        image = daytona.Image.base(build.base_image)
    else:
        image = daytona.Image.debian_slim("3.12")
    image = image.run_commands(*build_commands(build))
    name = snapshot_name or f"{build.environment}-contents"
    client = daytona_client if daytona_client is not None else daytona.Daytona()
    params = daytona.CreateSnapshotParams(name=name, image=image)
    client.snapshot.create(params)
    return BuiltArtifact(
        provider="daytona", reference=name, entries=list(build.entries), image=image
    )


def build_artifact(
    build: EnvironmentBuild,
    *,
    output_path: str | Path | None = None,
    snapshot_name: str | None = None,
    daytona_client: Any | None = None,
) -> BuiltArtifact:
    """Build the Environment's contents into an artifact of its provider.

    Args:
        build: The manifest and the provider.
        output_path: Where to write the Dockerfile — required for E2B, whose
            CLI builds from it; optional for Datalayer, whose fragment is
            returned as text either way.
        snapshot_name: The Daytona snapshot to create; the Environment's
            name with `-contents` by default.
        daytona_client: A `daytona.Daytona` to create the snapshot with; one
            is made from the environment's credentials by default.
    """
    if build.provider == "datalayer":
        return _build_datalayer(build, output_path)
    if build.provider == "e2b":
        return _build_e2b(build, output_path)
    if build.provider == "modal":
        return _build_modal(build)
    if build.provider == "daytona":
        return _build_daytona(build, snapshot_name=snapshot_name, daytona_client=daytona_client)
    raise SandboxConfigurationError(f"no builder for provider {build.provider!r}")


# --- Reading it back ------------------------------------------------------------


def installed_environment_contents(
    sandbox: Sandbox, *, path: str = ENVIRONMENT_CONTENTS_MANIFEST
) -> dict[str, Any] | None:
    """What the artifact a running sandbox started from says it carries.

    Read from inside the sandbox — the file is the artifact's, not this
    host's — and None when the artifact carries no such manifest, which is
    what an image built without Environment contents looks like.
    """
    from .contents import _answer, probe

    code = (
        "import json as _cs_json, os as _cs_os\n"
        f"_cs_path = {path!r}\n"
        "_cs_found = None\n"
        "if _cs_os.path.isfile(_cs_path):\n"
        "    with open(_cs_path) as _cs_handle:\n"
        "        _cs_found = _cs_json.load(_cs_handle)\n"
        + _answer("{'manifest': _cs_found}")
        + "del _cs_json, _cs_os, _cs_path, _cs_found\n"
    )
    return probe(sandbox, code)["manifest"]
