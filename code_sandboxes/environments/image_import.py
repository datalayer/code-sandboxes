# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Importing an existing OCI image as an Environment's base (PLAN_ENV.md, E3-04).

`build.source: image` skips the Datalayer base entirely: the reference names
the image, resolution pins it to a digest the same way a base is pinned
(D-9), and the contract layer — Datalayer's protected pins, installed the
same way `merge_requirements` already installs them over a `packages` or
`dependencyFile` source — is added on top by the ordinary resolve and build
path. Nothing downstream needs to know a build started from an import rather
than an approved base; the only difference is where the `FROM` digest comes
from.

**Public registries only, for now.** A per-organization allowlist and
credentials for a private registry are the rest of this box (E3-04's second
half), left for E3-06's organization policy document to hold — the
`policy.py` docstring already says as much about the scan threshold, and a
registry allowlist is the same shape of decision. `DEFAULT_ALLOWED_REGISTRIES`
below is a bootstrap standing in for that document before it exists, not a
policy of its own: when E3-06 lands, an organization's own list replaces this
one rather than adding to it.

@module code_sandboxes.environments.image_import
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .errors import POLICY_DENIED, PROVIDER_ERROR, SPEC_INVALID, EnvironmentsError

__all__ = [
    "DEFAULT_ALLOWED_REGISTRIES",
    "ImageReference",
    "image_registry_allowed",
    "parse_image_reference",
    "refuse_unless_allowed",
    "resolve_image_digest",
]

#: Registries recognized as public sources until E3-06's organization policy
#: document can name its own. `docker.io` is what an unqualified reference
#: (`python:3.12-slim-bookworm`) means; the rest are the registries the
#: section 11 example set and the plan's own `python:3.12-slim-bookworm`
#: instance already assume reachable with no credential.
DEFAULT_ALLOWED_REGISTRIES: tuple[str, ...] = (
    "docker.io",
    "ghcr.io",
    "quay.io",
    "gcr.io",
    "registry.k8s.io",
    "public.ecr.aws",
)

#: What a bare reference with no registry host means (Docker's own default).
DOCKER_HUB = "docker.io"
#: Docker Hub's real API host: `docker.io` itself answers no registry API.
_DOCKER_HUB_API_HOST = "registry-1.docker.io"
#: Docker Hub's anonymous token realm, for a public repository's pull scope.
_DOCKER_HUB_AUTH = "https://auth.docker.io/token"

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
#: A `WWW-Authenticate` challenge's `key="value"` pairs, in any order.
_CHALLENGE_FIELD = re.compile(r'(\w+)="([^"]*)"')
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


@dataclass(frozen=True)
class ImageReference:
    """A parsed `<registry>/<repository>[:tag|@digest]`, Docker's own way."""

    registry: str
    repository: str
    tag: str = ""
    digest: str = ""

    @property
    def pinned(self) -> bool:
        """Whether this reference already names a digest, needing no resolve."""
        return bool(self.digest)

    def with_digest(self, digest: str) -> ImageReference:
        return ImageReference(self.registry, self.repository, self.tag, digest)

    def __str__(self) -> str:
        suffix = f"@{self.digest}" if self.digest else f":{self.tag or 'latest'}"
        return f"{self.registry}/{self.repository}{suffix}"


def parse_image_reference(reference: str) -> ImageReference:
    """`ghcr.io/owner/repo:tag`, `python:3.12-slim-bookworm`, or one pinned by digest.

    The one thing this must get right is telling a registry host from the
    first path segment of a Docker Hub repository — `nginx/nginx-prometheus-
    exporter` names an organization, `ghcr.io/owner/repo` a registry — which is
    exactly the rule Docker's own reference parser uses: the first segment is
    a host only if it has a dot, a port, or is `localhost`.
    """
    text = str(reference or "").strip()
    if not text or any(character.isspace() for character in text):
        # A reference is embedded straight into a Dockerfile `FROM` line
        # (`resolve_image_base`, `BuildkitResolveRunner.dockerfile`): an
        # embedded newline would end that line and start another Dockerfile
        # instruction of the author's choosing. Whitespace is never legal in
        # a real reference anyway, so refusing it here is free.
        raise EnvironmentsError(
            SPEC_INVALID,
            "an image reference names an image, with no whitespace in it",
            detail={"field": "spec.build.image.reference", "reference": reference},
        )
    digest = ""
    if "@" in text:
        text, _, digest = text.partition("@")
        if not _DIGEST.match(digest):
            raise EnvironmentsError(
                SPEC_INVALID,
                f"{digest!r} is not a sha256 digest",
                detail={"field": "spec.build.image.reference", "reference": reference},
            )
    tag = ""
    path = text
    last_segment = text.rsplit("/", 1)[-1]
    if ":" in last_segment:
        path, _, tag = text.rpartition(":")
    if not path:
        raise EnvironmentsError(
            SPEC_INVALID,
            "an image reference names a repository",
            detail={"field": "spec.build.image.reference", "reference": reference},
        )
    first, sep, rest = path.partition("/")
    if sep and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = DOCKER_HUB, path
    if registry == DOCKER_HUB and "/" not in repository:
        repository = f"library/{repository}"
    if not repository:
        raise EnvironmentsError(
            SPEC_INVALID,
            "an image reference names a repository",
            detail={"field": "spec.build.image.reference", "reference": reference},
        )
    if not tag and not digest:
        tag = "latest"
    return ImageReference(registry=registry, repository=repository, tag=tag, digest=digest)


def image_registry_allowed(
    image: ImageReference, allowlist: tuple[str, ...] = DEFAULT_ALLOWED_REGISTRIES
) -> bool:
    return image.registry in allowlist


def refuse_unless_allowed(
    image: ImageReference,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWED_REGISTRIES,
    *,
    has_credential: bool = False,
) -> None:
    """Refuse an unlisted registry, unless a credential was referenced for it.

    A registry named alongside `credentialSecretId` (E3-04's private half) is
    presumed a deliberate private one, its governance E3-06's own allowlist
    to hold once it exists — this bootstrap list is only ever the *public*
    default, never the only door. What resolving that credential into a real
    one to pull with is E3-05's mechanism, not built for anything yet; a spec
    that clears this check but has no real credential fails honestly at the
    registry instead, the same as an approved base with no digest published.
    """
    if has_credential or image_registry_allowed(image, allowlist):
        return
    raise EnvironmentsError(
        POLICY_DENIED,
        f"{image.registry} is not an allowed registry for an imported image "
        f"(the allowed ones are {', '.join(allowlist)}, until an organization "
        "policy names its own, or reference a credential for a private one)",
        detail={"field": "spec.build.image.reference", "registry": image.registry},
    )


def _api_host(registry: str) -> str:
    return _DOCKER_HUB_API_HOST if registry == DOCKER_HUB else registry


def _challenge(header: str) -> dict[str, str]:
    return dict(_CHALLENGE_FIELD.findall(header or ""))


def resolve_image_digest(
    image: ImageReference,
    *,
    transport: Any = None,
    timeout: float = 10.0,
) -> str:
    """The reference's digest, over the registry's own v2 API — no credential,
    since only a public registry reaches this (§ this module's own docstring).

    Already pinned by digest, this is free: an operator who wrote `@sha256:…`
    meant exactly that image, and nothing is asked of the registry. A tag is
    resolved with one anonymous manifest request, following the registry's own
    `WWW-Authenticate` challenge for the one-time pull token every public
    registry answers with no credential (D-9's base digests are pinned the
    same way, just from a table Datalayer keeps instead of asked live).
    """
    if image.pinned:
        return image.digest
    try:
        import httpx
    except ImportError as error:
        from .errors import CAPABILITY_UNSUPPORTED

        raise EnvironmentsError(
            CAPABILITY_UNSUPPORTED,
            "No HTTP client to resolve an imported image's digest with: install "
            "`code-sandboxes[environments-builder]`",
            detail={"missing": "httpx"},
        ) from error
    url = f"https://{_api_host(image.registry)}/v2/{image.repository}/manifests/{image.tag}"
    headers = {"Accept": _MANIFEST_ACCEPT}
    with httpx.Client(transport=transport, timeout=timeout) as client:
        try:
            response = client.get(url, headers=headers)
            if response.status_code == 401:
                challenge = _challenge(response.headers.get("www-authenticate", ""))
                realm = challenge.get("realm")
                if not realm:
                    raise EnvironmentsError(
                        PROVIDER_ERROR,
                        f"{image.registry} refused an anonymous manifest request with no "
                        "way to ask for a token",
                        detail={"registry": image.registry, "status": 401},
                    )
                token_response = client.get(
                    realm,
                    params={
                        key: value
                        for key, value in (
                            ("service", challenge.get("service")),
                            ("scope", challenge.get("scope")),
                        )
                        if value
                    },
                )
                token_response.raise_for_status()
                token = str(token_response.json().get("token") or "")
                response = client.get(url, headers={**headers, "Authorization": f"Bearer {token}"})
        except httpx.HTTPError as error:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"{image.registry} could not be reached to resolve {image}: {error}",
                detail={"registry": image.registry, "reference": str(image)},
            ) from error
    if response.status_code == 404:
        raise EnvironmentsError(
            PROVIDER_ERROR,
            f"{image} does not exist at {image.registry}, or is private",
            detail={"registry": image.registry, "reference": str(image), "status": 404},
        )
    if response.status_code != 200:
        raise EnvironmentsError(
            PROVIDER_ERROR,
            f"{image.registry} answered {response.status_code} resolving {image}",
            detail={
                "registry": image.registry,
                "reference": str(image),
                "status": response.status_code,
            },
        )
    digest = response.headers.get("docker-content-digest", "")
    if not digest:
        # A registry that does not set the header still answers a manifest
        # whose digest is defined as the hash of exactly these bytes (OCI
        # image spec): the same number every compliant registry would have
        # put in the header.
        digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
    return digest
