# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The scan and the signature an artifact needs before anything runs it.

Between a build and a sandbox there are two gates (PLAN_ENV.md D-11, E1-08,
E1-09):

1. **The scan.** ECR scans on push; this waits for that scan with a time
   bound, reads the findings with the reader principal, and hands them to
   :mod:`.policy`, which decides. A blocking decision is
   ``DL_ENV_SCAN_BLOCKED``, naming the findings a person can act on, and the
   decision record is stored on the artifact so it can be re-read later
   without asking the scanner again.
2. **The signature.** cosign signs the digest with the KMS key as soon as the
   scan passes and **before** the smoke test, because the Operator starts no
   unsigned ``environments/`` image — the smoke test's own trial sandbox
   included. A replayed attestation finds the signature that is already there
   rather than pushing a second one, which immutable tags would refuse
   anyway.

Order matters and is not an implementation detail: signing first would put a
signature on an artifact the scan then blocks, and a signature is what the
Operator reads as "Datalayer vouches for this".

Everything outward is injectable — the ECR client, how cosign is run, the
clock — so the whole chain is tested without AWS and without a daemon.

@module code_sandboxes.environments.attest
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .errors import (
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    SCAN_BLOCKED,
    EnvironmentsError,
)
from .policy import (
    DEFAULT_POLICY,
    UNRESTRICTED_POLICY,
    EnvironmentsPolicy,
    PolicyDecision,
    ScanPolicy,
    decide,
    findings_of,
    refuse_unlicensed,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SCAN_STATUSES",
    "AttestationResult",
    "Attestor",
    "attest_artifact",
    "signature_tag",
]

#: What ECR says about a scan. `ACTIVE` and `COMPLETE` both mean it finished:
#: enhanced scanning says `ACTIVE`, basic says `COMPLETE`.
SCAN_STATUSES = {
    "done": ("COMPLETE", "ACTIVE"),
    "waiting": ("IN_PROGRESS", "PENDING", "SCAN_ELIGIBILITY_EXPIRED"),
    "refused": ("FAILED", "UNSUPPORTED_IMAGE", "FINDINGS_UNAVAILABLE"),
}

#: The platform an Environment image is built for: the image of an index
#: whose scan is read.
_SCANNED_PLATFORM = ("linux", "amd64")

#: What a registry answers for an image index, and for a single image.
_INDEX_MEDIA_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
_MANIFEST_MEDIA_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)

#: How long a scan is waited for, and how often it is asked about.
DEFAULT_SCAN_TIMEOUT_SECONDS = 15 * 60
DEFAULT_SCAN_INTERVAL_SECONDS = 10.0

#: A whole digest, not merely something that starts with `sha256:` — found on
#: PR #27's Copilot review: `sha256:bad` used to pass the check that reached
#: the scan and the signature, both keyed on this string.
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def signature_tag(digest: str) -> str:
    """Where cosign keeps the signature of a digest: `sha256-<hex>.sig`.

    cosign's own convention, and the reason a signature needs no repository of
    its own: it is an image beside the one it signs, in the same repository, at
    a tag derived from the digest. A replay looks for exactly this tag.
    """
    text = str(digest or "").strip()
    if not text.startswith("sha256:"):
        raise ValueError("a signature is kept beside a digest, `sha256:<hex>`")
    return text.replace(":", "-", 1) + ".sig"


#: Where a CycloneDX component keeps its licence, in the order they are read.
_CYCLONEDX_LICENSE_KEYS = ("id", "name")
#: Where an SPDX package keeps its licence. `licenseConcluded` is what the
#: tool decided; `licenseDeclared` is what the package claimed. BuildKit
#: writes SPDX, so this is the one that matters in practice.
_SPDX_LICENSE_KEYS = ("licenseConcluded", "licenseDeclared")
#: What SPDX writes when it could not tell, which is not a licence.
_SPDX_UNKNOWN = frozenset({"NOASSERTION", "NONE", ""})


def _spdx_license_pairs(document: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every SPDX package with a licence, as `(package, licence)`.

    `licenseConcluded` is what the tool decided and `licenseDeclared` what the
    package claimed, so the concluded one is read first and the declared one
    only when it said nothing.
    """
    found: list[tuple[str, str]] = []
    for package in document.get("packages") or ():
        if not isinstance(package, Mapping):
            continue
        name = str(package.get("name") or "").strip()
        for key in _SPDX_LICENSE_KEYS:
            value = str(package.get(key) or "").strip()
            if value and value.upper() not in _SPDX_UNKNOWN:
                # Kept even with no name: `licenses_of` reads only the licence
                # half, and dropping an unnamed package's licence there would
                # be exactly the silent behavior change this refactor must
                # not make. `licenses_by_package`'s own callers decide what an
                # empty package name means for them.
                found.append((name, value))
                break
    return found


def _cyclonedx_license_pairs(document: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every CycloneDX component with a licence, as `(component, licence)`."""
    found: list[tuple[str, str]] = []
    for component in document.get("components") or ():
        if not isinstance(component, Mapping):
            continue
        name = str(component.get("name") or "").strip()
        for entry in component.get("licenses") or ():
            if not isinstance(entry, Mapping):
                continue
            licence = entry.get("license")
            value = ""
            if isinstance(licence, Mapping):
                for key in _CYCLONEDX_LICENSE_KEYS:
                    value = str(licence.get(key) or "").strip()
                    if value:
                        break
            if not value:
                value = str(entry.get("expression") or "").strip()
            if value:
                # Kept even with no component name; see the SPDX reader's own
                # note above.
                found.append((name, value))
    return found


def licenses_by_package(document: Any) -> list[tuple[str, str]]:
    """Every `(package, licence)` an SBOM names, package first (E3-06).

    The same two shapes `licenses_of` reads, kept attributed rather than
    flattened: a policy that denies a licence has to name the package that
    carries it, which `licenses_of`'s own deduplicated set of licence strings
    alone cannot answer. Insertion order, not sorted — the order the SBOM's
    own packages/components came in, which is what a refusal naming "the
    first one" should mean deterministically for the same document.
    """
    if not isinstance(document, Mapping):
        return []
    return _spdx_license_pairs(document) + _cyclonedx_license_pairs(document)


def licenses_of(document: Any) -> list[str]:
    """Every licence an SBOM names, deduplicated and sorted.

    Reads both shapes the ecosystem writes: SPDX, which is what BuildKit's
    `attest:sbom=` produces, and CycloneDX. A document in neither shape, or one
    that names nothing, answers an empty list rather than raising: a
    publication's licence list is worth having and never worth failing a build
    over.

    This is what a published version's snapshot carries (D-12, E2-16). Until
    it did, `licenses` came from the scan summary — and the registry's scanner
    reports vulnerabilities, not licences, so every publication froze an empty
    list beside an SBOM reference.
    """
    if not isinstance(document, Mapping):
        return []
    return sorted({licence for _package, licence in licenses_by_package(document)})


@dataclass(frozen=True)
class AttestationResult:
    """What the workflow stores about an artifact once both gates have passed."""

    decision: PolicyDecision
    signature_ref: str
    sbom_ref: str = ""
    provenance_ref: str = ""
    size_bytes: int | None = None
    signed_now: bool = True
    """False when a replay found the signature that was already there."""
    licenses: tuple[str, ...] = ()
    """What the SBOM named, frozen onto the artifact for a publication to carry."""

    def body(self) -> dict[str, Any]:
        """The mapping `activities_environments.attest` answers."""
        return {
            "scan_summary": self.decision.body(),
            "sbom_ref": self.sbom_ref,
            "provenance_ref": self.provenance_ref,
            "signature_ref": self.signature_ref,
            "size_bytes": self.size_bytes,
            "signed_now": self.signed_now,
            "licenses": list(self.licenses),
        }


class Attestor:
    """The two gates, against one registry.

    Parameters
    ----------
    ecr
        The ECR client. A boto3 `ecr` client by default, made on first use.
    cosign
        The `cosign` to run; an empty string means there is none, which is
        what a worker image without it looks like.
    key
        The KMS key cosign signs with, as cosign names one:
        `awskms:///alias/datalayer-environments` or an arn.
    run
        How a subprocess is run, so a test watches the argv.
    sleep, now
        The clock, injected so a test waits for nothing.
    policy
        What the findings have to say. The owner's default unless an
        organization tightened it (E3-06).
    registry_auth
        What cosign reads the repository with, to sign it (D-17): the same
        `{"DOCKER_CONFIG": <dir>}` shape the resolver and the builder already
        take off a `BuildCredential`, the one attribute name every caller
        reads regardless of what kind of credential it holds. cosign has no
        AWS credential chain of its own for ECR, unlike the boto3 client the
        scan is read with — found live, 2026-09-14: with none, `cosign sign`
        reached the registry anonymously and was refused, `401 Unauthorized`,
        on every real artifact this pipeline ever tried to sign.
    """

    def __init__(
        self,
        *,
        ecr: Any = None,
        cosign: str | None = None,
        key: str = "",
        region: str | None = None,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        sleep: Callable[[float], None] | None = None,
        now: Callable[[], float] | None = None,
        policy: ScanPolicy = DEFAULT_POLICY,
        scan_timeout_seconds: float = DEFAULT_SCAN_TIMEOUT_SECONDS,
        scan_interval_seconds: float = DEFAULT_SCAN_INTERVAL_SECONDS,
        log: Callable[[str], None] | None = None,
        registry_auth: Mapping[str, str] | None = None,
    ) -> None:
        self._ecr = ecr
        self._cosign = (shutil.which("cosign") or "") if cosign is None else cosign
        self._key = key or os.environ.get("DATALAYER_ENVIRONMENTS_KMS_KEY", "").strip()
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._run = run or subprocess.run
        self._sleep = sleep or time.sleep
        self._now = now or time.monotonic
        self._policy = policy
        self._timeout = scan_timeout_seconds
        self._interval = scan_interval_seconds
        self._log = log or (lambda _line: None)
        self._registry_auth = dict(registry_auth or {})

    # -- the scan -------------------------------------------------------------

    def scan(self, *, repository: str, digest: str) -> PolicyDecision:
        """Wait for the scan of this digest, read it, and decide (E1-08).

        A scan that never finishes inside the bound is `DL_ENV_PROVIDER_ERROR`
        — retryable, because nothing about the version is wrong — and so is a
        scanner that refuses the image. A scan that finished is decided, and a
        blocking decision raises `DL_ENV_SCAN_BLOCKED`.

        An artifact that is an image index is decided by the scan of its
        linux/amd64 image: see `_scanned_digest`.
        """
        deadline = self._now() + self._timeout
        scanned = self._scanned_digest(repository=repository, digest=digest)
        where = {"repository": repository, "digest": digest, "scannedDigest": scanned}
        waited = 0
        while True:
            status, described = self._describe(repository=repository, digest=scanned)
            if status in SCAN_STATUSES["done"]:
                break
            if status in SCAN_STATUSES["refused"]:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The registry did not scan this artifact: {status}",
                    detail={**where, "scanStatus": status},
                )
            if self._now() >= deadline:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The scan of this artifact did not finish within {self._timeout:.0f}s "
                    f"(last status {status or 'unknown'})",
                    detail={**where, "scanStatus": status},
                )
            waited += 1
            if waited == 1:
                self._log(f"Waiting for the scan of {scanned}")
            self._sleep(self._interval)
        findings = findings_of(self._findings(described))
        decision = decide(
            findings,
            policy=self._policy,
            scanner=str(described.get("scanner") or "ECR"),
            scanned_at=_moment(described.get("completed_at")),
            scan_status=status,
        )
        self._log(
            f"Scan of {digest}: {decision.decision}"
            + (f" — {decision.said()}" if not decision.passed else "")
        )
        return decision

    def _scanned_digest(self, *, repository: str, digest: str) -> str:
        """The digest whose scan decides this artifact: its own, or its linux/amd64 image's.

        The Datalayer builder pushes with SBOM and provenance attestations, so
        what it records is an OCI image index: the image, and an attestation
        manifest beside it. Enhanced scanning scans the image and answers
        `UNSUPPORTED_IMAGE` for the index (found live on r1, 2026-09-14), so the
        scan read is that of the image the index names for linux/amd64. The
        signature stays on the index, which is what a pod pulls.
        """
        try:
            answer = self._client().batch_get_image(
                repositoryName=repository,
                imageIds=[{"imageDigest": digest}],
                acceptedMediaTypes=[*_INDEX_MEDIA_TYPES, *_MANIFEST_MEDIA_TYPES],
            )
        except Exception as error:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"The manifest of {digest} could not be read: {error}",
                detail={"repository": repository, "digest": digest},
            ) from error
        images = answer.get("images") or []
        if not images:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"The registry has no image {digest}",
                detail={"repository": repository, "digest": digest},
            )
        try:
            manifest = json.loads(images[0].get("imageManifest") or "{}")
        except ValueError:
            manifest = {}
        media_type = str(images[0].get("imageManifestMediaType") or manifest.get("mediaType") or "")
        if media_type not in _INDEX_MEDIA_TYPES and "manifests" not in manifest:
            return digest
        os_name, architecture = _SCANNED_PLATFORM
        for entry in manifest.get("manifests") or []:
            annotations = entry.get("annotations") or {}
            if annotations.get("vnd.docker.reference.type") == "attestation-manifest":
                continue
            platform = entry.get("platform") or {}
            child = str(entry.get("digest") or "")
            if (
                platform.get("os") == os_name
                and platform.get("architecture") == architecture
                and _DIGEST.match(child)
            ):
                self._log(
                    f"Reading the scan of {child}, the {os_name}/{architecture} image of {digest}"
                )
                return child
        raise EnvironmentsError(
            PROVIDER_ERROR,
            f"The image index {digest} holds no {os_name}/{architecture} image to scan",
            detail={"repository": repository, "digest": digest},
        )

    #: A hard cap on pages read, so a registry that never stops paginating
    #: cannot wedge a build forever — this many pages is already far past any
    #: real image's own finding count (E1-08, found live 2026-09-14: see
    #: `_describe`'s own docstring for why a page limit is not optional).
    _MAX_FINDING_PAGES = 50

    def _describe(self, *, repository: str, digest: str) -> tuple[str, dict[str, Any]]:
        """Every `DescribeImageScanFindings` page, as one status and one answer.

        ECR paginates enhanced findings — `nextToken`, not a field this call
        can ask to skip — and a single unpaginated call answers only its first
        page. Found live, 2026-09-14: the first real artifact this pipeline
        scanned had 1,547 enhanced findings across many pages and 31 of them
        critical, and a single-page read passed a decision that should have
        blocked, because none of those 31 were in the one page it happened to
        see. Every page is read and merged before `decide` ever runs.
        """
        pages = 0
        status = ""
        enhanced: list[Any] = []
        basic: list[Any] = []
        completed_at: Any = None
        token: str | None = None
        while True:
            try:
                kwargs: dict[str, Any] = {
                    "repositoryName": repository,
                    "imageId": {"imageDigest": digest},
                }
                if token:
                    kwargs["nextToken"] = token
                answer = self._client().describe_image_scan_findings(**kwargs)
            except Exception as error:
                if _is_in_progress(error):
                    return "IN_PROGRESS", {}
                if _is_missing(error):
                    raise EnvironmentsError(
                        PROVIDER_ERROR,
                        f"The registry has no scan for {digest}: {error}",
                        detail={"repository": repository, "digest": digest},
                    ) from error
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The scan of {digest} could not be read: {error}",
                    detail={"repository": repository, "digest": digest},
                ) from error
            status = str(((answer.get("imageScanStatus") or {}).get("status")) or "")
            findings = answer.get("imageScanFindings") or {}
            enhanced.extend(findings.get("enhancedFindings") or [])
            basic.extend(findings.get("findings") or [])
            completed_at = findings.get("imageScanCompletedAt") or completed_at
            pages += 1
            token = answer.get("nextToken")
            if not token or pages >= self._MAX_FINDING_PAGES:
                if token:
                    self._log(
                        f"Stopped reading {digest}'s scan after {pages} pages, "
                        "with more findings still unread"
                    )
                break
        return status, {
            "findings": {"enhancedFindings": enhanced, "findings": basic},
            "completed_at": completed_at,
            "scanner": "ECR enhanced" if enhanced else "ECR basic",
        }

    @staticmethod
    def _findings(described: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
        findings = described.get("findings") or {}
        enhanced = findings.get("enhancedFindings")
        if isinstance(enhanced, list) and enhanced:
            return enhanced
        basic = findings.get("findings")
        return basic if isinstance(basic, list) else []

    # -- the signature --------------------------------------------------------

    def can_sign(self) -> None:
        """Refuse now if nothing here could sign, whatever the scan says.

        Checked before the scan, not after: an artifact nobody can sign can
        never be used — the Operator starts no unsigned `environments/` image
        (D-11) — so waiting a quarter of an hour for a scan first would spend
        the wait to reach the same refusal, and would read in the log as the
        scan being the problem.
        """
        if not self._key:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No signing key: the artifact cannot be signed, and the Operator starts no "
                "unsigned environments/ image",
                detail={"missing": "DATALAYER_ENVIRONMENTS_KMS_KEY", "item": "E1-06"},
            )
        if not self._cosign:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "No `cosign` to sign with: the worker image does not carry it",
                detail={"missing": "cosign", "item": "E1-06"},
            )

    def sign(self, *, registry: str, repository: str, digest: str) -> tuple[str, bool]:
        """Sign the digest with the KMS key, unless it is signed already (E1-09).

        Answers the signature's own reference — the digest itself, which is
        what `cosign verify --key <key>` takes, resolving the signature's
        real location on its own — and whether this call made it.

        A replay finds the existing signature by asking cosign whether *this*
        key has already signed it, the same check the Operator itself will
        make before starting a pod (`can_verify`): cosign 3.1.3 stores a
        signature as an OCI 1.1 referrer, not the classic `sha256-<hex>.sig`
        sidecar tag a first version of this method assumed (found live,
        2026-09-14, twice — `--registry-referrers-mode` only ever governed
        *reading* referrers, never where `sign` writes one, so it changed
        nothing the first time either). Signing twice would not error the way
        it would have under the old tag (an OCI referrer is not an immutable
        tag to collide with), but it would leave two things claiming to be
        Datalayer's word on the same artifact, which `can_verify` first
        avoids.
        """
        self.can_sign()
        reference = f"{registry}/{repository}@{digest}"
        if self.can_verify(reference):
            self._log(f"{digest} is signed already")
            return reference, False
        command = [
            self._cosign,
            "sign",
            "--yes",
            # Never the public transparency log: the digest and the
            # repository of a private environment are nobody else's to read
            # (matching clouder's own deploy-check, the one place this was
            # already right — this adapter never had it). Found live,
            # 2026-09-13: without this, `cosign sign` reaches for the public
            # Rekor service by default and prompts for consent to publish an
            # immutable record of the artifact — the wrong default for a
            # private environment, and one a non-interactive worker cannot
            # even answer. The matching verify-side omission is fixed in
            # `datalayer_operator.services.environment_signatures`.
            "--tlog-upload=false",
            # cosign 3.1.3 defaults `--use-signing-config` to `true`: a
            # TUF-provided signing config now names the service URLs,
            # including a transparency log, and `--tlog-upload=false` alone
            # no longer overrides that — cosign refuses the combination
            # outright, "not supported with --signing-config or
            # --use-signing-config" (found live, 2026-09-14, the first real
            # sign this worker's own pinned cosign ever ran). Turning the
            # signing config off restores the plain, flag-driven behavior
            # `--tlog-upload=false` already asks for, and needs no network
            # call of its own to fetch one.
            "--use-signing-config=false",
            "--key",
            self._key,
            reference,
        ]
        self._log(f"Signing {digest} with the KMS key")
        finished = self._invoke(command)
        if finished.returncode != 0:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                "cosign did not sign the artifact; its output says why",
                detail={"digest": digest, "exit": finished.returncode},
            )
        return reference, True

    def can_verify(self, reference: str) -> bool:
        """Whether cosign already finds a signature by this key on `reference`.

        Wherever cosign itself keeps a signature — the OCI 1.1 referrer it
        defaults to, or the legacy sidecar tag an older registry might still
        need — is cosign's own business to resolve; asking it directly, the
        same check the Operator makes before starting a pod, is simpler and
        more honest than tracking cosign's own storage choices here too.
        """
        finished = self._invoke(
            [self._cosign, "verify", "--insecure-ignore-tlog=true", "--key", self._key, reference]
        )
        return finished.returncode == 0

    def _invoke(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        # `registry_auth` — same shape as the resolver's own — puts cosign's
        # `DOCKER_CONFIG` beside the rest of this process's environment,
        # never replacing it: `--key` alone still needs AWS's own chain to
        # reach the KMS key, which this does not touch.
        env = {**os.environ, **self._registry_auth} if self._registry_auth else None
        try:
            finished = self._run(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=self._timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as expired:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"cosign did not finish within {self._timeout:.0f}s",
                detail={"timeout": self._timeout},
            ) from expired
        for line in (finished.stderr or "").splitlines():
            self._log(line)
        return finished

    # -- both, in order -------------------------------------------------------

    def attest(
        self,
        *,
        registry: str,
        repository: str,
        digest: str,
        size_bytes: int | None = None,
        sbom_ref: str = "",
        provenance_ref: str = "",
        sbom: Any = None,
        environments_policy: EnvironmentsPolicy = UNRESTRICTED_POLICY,
    ) -> AttestationResult:
        """Scan, then sign: the order the Operator's check depends on (D-11).

        `sbom`, when the caller has the document, is read for the licences a
        publication carries; the registry's scanner reports vulnerabilities
        and never licences, so there is nowhere else they come from.

        **Refused before signing, never after (E3-06).** A licence an
        organization's own policy denies is checked against the same `sbom`,
        naming the package that carries it, and raised before `sign` runs —
        an artifact whose licence policy denies it is never signed, the same
        as one whose scan blocks it: a signature is Datalayer's word that an
        artifact may run, and this is the second of the two words that go
        into it.
        """
        self.can_sign()
        decision = self.scan(repository=repository, digest=digest)
        if not decision.passed:
            raise EnvironmentsError(
                SCAN_BLOCKED,
                f"The scan blocks this artifact: {decision.said()}",
                detail={
                    "repository": repository,
                    "digest": digest,
                    "decision": decision.body(),
                    "blocking": [finding.id for finding in decision.blocking],
                },
            )
        pairs = licenses_by_package(sbom)
        refuse_unlicensed(pairs, environments_policy)
        signature, signed_now = self.sign(registry=registry, repository=repository, digest=digest)
        return AttestationResult(
            decision=decision,
            signature_ref=signature,
            sbom_ref=sbom_ref or f"{registry}/{repository}@{digest}.sbom",
            provenance_ref=provenance_ref or f"{registry}/{repository}@{digest}.att",
            size_bytes=size_bytes
            if size_bytes is not None
            else self.size_of(repository=repository, digest=digest),
            signed_now=signed_now,
            licenses=tuple(sorted({licence for _package, licence in pairs})),
        )

    def size_of(self, *, repository: str, digest: str) -> int | None:
        """What the registry says the artifact weighs, or None.

        Nobody hands the size down: the builder answers a reference, not a
        weight, so until this asked the registry `size_bytes` was always None
        — and with it `environments.artifact.bytes`, the series section 14
        tracks the artifact size in, which had no point in it on 2026-09-16
        although artifacts had been recorded. The registry has known all
        along; the scan is read from the same client.

        Never a reason to fail an attestation: a size that could not be read
        is a missing number on a dashboard, and the artifact is still signed.
        """
        try:
            images = self._client().describe_images(
                repositoryName=repository, imageIds=[{"imageDigest": digest}]
            )["imageDetails"]
        except Exception as error:
            self._log(f"The artifact's size could not be read: {error}")
            return None
        size = (images[0] or {}).get("imageSizeInBytes") if images else None
        return int(size) if size else None

    def _client(self) -> Any:
        if self._ecr is None:
            try:
                import boto3
            except ImportError as error:
                raise EnvironmentsError(
                    CAPABILITY_UNSUPPORTED,
                    "No AWS SDK to read the scan with: install "
                    "`code-sandboxes[environments-builder]`",
                    detail={"missing": "boto3"},
                ) from error
            self._ecr = boto3.client("ecr", region_name=self._region)
        return self._ecr


def attest_artifact(
    *,
    artifact: Any,
    credential: Any = None,
    policy: ScanPolicy = DEFAULT_POLICY,
    environments_policy: EnvironmentsPolicy = UNRESTRICTED_POLICY,
    log: Callable[[str], None] | None = None,
    size_bytes: int | None = None,
    sbom: Any = None,
    attestor: Attestor | None = None,
) -> dict[str, Any]:
    """The `attest` seam of `EnvironmentBuildWorkflow` (E1-08, E1-09).

    Takes the artifact the builder recorded and the build's credential, and
    answers the mapping the workflow stores: the scan's decision, the
    signature, the SBOM and provenance references, and the size.

    `environments_policy` is the caller's organization's own narrowing of
    what licence a build may carry (E3-06); `sbom`, when the caller has the
    document, is what it is checked against — nobody fetches one here. A
    caller with neither passes nothing through, and nothing is refused: the
    platform default is unrestricted on licences, the same as every other
    dimension of this policy until an organization writes one.
    """
    variant = str(getattr(artifact, "variant", "") or "")
    reference = str(getattr(artifact, "immutable_reference", "") or "")
    if variant and variant != "datalayer":
        # D-11 is about the Datalayer artifact: it lives in this platform's
        # registry, the scanner reads it there, cosign signs that digest, and
        # the Operator refuses to start what is unsigned. A managed artifact
        # is none of those things — it lives in the owner's own provider
        # account (D-8) and is referenced the way that provider names it: a
        # Daytona snapshot uuid, an E2B build id, a Modal `im-…`. There is no
        # digest in a repository to scan or sign, and no Operator starting it.
        #
        # Attesting one anyway is what the first real Daytona build did, and
        # it failed *after* the snapshot was built — the work done, the
        # artifact live at the provider, and the build recorded as failed
        # (2026-09-17).
        #
        # Publishing is not weakened by this: E2-15's gate requires the
        # **Datalayer** artifact to have passed its scan, and that one is
        # still attested here.
        if log:
            log(f"{variant} artifacts are not attested: {reference} is not a digest in a registry")
        return {
            "scan_summary": {},
            "sbom_ref": "",
            "provenance_ref": "",
            "signature_ref": "",
            "size_bytes": size_bytes,
            "signed_now": False,
            "licenses": [],
        }
    registry, _, rest = reference.partition("/")
    repository, _, digest = rest.partition("@")
    if not (registry and repository and _DIGEST.match(digest)):
        raise EnvironmentsError(
            PROVIDER_ERROR,
            f"`{reference}` is not a digest in a repository, so it cannot be attested",
            detail={"reference": reference, "variant": variant},
        )
    use = attestor or Attestor(
        key=str(getattr(credential, "signing_key", "") or ""),
        policy=policy,
        log=log,
        registry_auth=getattr(credential, "registry_auth", None),
    )
    return use.attest(
        registry=registry,
        repository=repository,
        digest=digest,
        size_bytes=size_bytes,
        sbom=sbom,
        environments_policy=environments_policy,
    ).body()


def _moment(value: Any) -> str:
    if value is None:
        return ""
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, Mapping):
        return str((response.get("Error") or {}).get("Code") or "")
    return type(error).__name__


def _is_missing(error: BaseException) -> bool:
    return _error_code(error) in {
        "ImageNotFoundException",
        "RepositoryNotFoundException",
    }


def _is_in_progress(error: BaseException) -> bool:
    """ECR answers a scan that has not finished as an error, not a status.

    `ScanNotFoundException` is one of those answers: enhanced scanning starts
    after the push, so an image pushed a moment ago has no scan yet (found
    live on r1, 2026-09-14, the first build to read its image's scan rather
    than its index's). It is waited for like a running scan, within the same
    bound.
    """
    return _error_code(error) in {
        "ScanInProgressException",
        "ScanNotFoundException",
        "LimitExceededException",
    }
