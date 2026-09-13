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
from .policy import DEFAULT_POLICY, PolicyDecision, ScanPolicy, decide, findings_of

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

    def body(self) -> dict[str, Any]:
        """The mapping `activities_environments.attest` answers."""
        return {
            "scan_summary": self.decision.body(),
            "sbom_ref": self.sbom_ref,
            "provenance_ref": self.provenance_ref,
            "signature_ref": self.signature_ref,
            "size_bytes": self.size_bytes,
            "signed_now": self.signed_now,
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

    # -- the scan -------------------------------------------------------------

    def scan(self, *, repository: str, digest: str) -> PolicyDecision:
        """Wait for the scan of this digest, read it, and decide (E1-08).

        A scan that never finishes inside the bound is `DL_ENV_PROVIDER_ERROR`
        — retryable, because nothing about the version is wrong — and so is a
        scanner that refuses the image. A scan that finished is decided, and a
        blocking decision raises `DL_ENV_SCAN_BLOCKED`.
        """
        deadline = self._now() + self._timeout
        waited = 0
        while True:
            status, described = self._describe(repository=repository, digest=digest)
            if status in SCAN_STATUSES["done"]:
                break
            if status in SCAN_STATUSES["refused"]:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The registry did not scan this artifact: {status}",
                    detail={"repository": repository, "digest": digest, "scanStatus": status},
                )
            if self._now() >= deadline:
                raise EnvironmentsError(
                    PROVIDER_ERROR,
                    f"The scan of this artifact did not finish within {self._timeout:.0f}s "
                    f"(last status {status or 'unknown'})",
                    detail={"repository": repository, "digest": digest, "scanStatus": status},
                )
            waited += 1
            if waited == 1:
                self._log(f"Waiting for the scan of {digest}")
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

    def _describe(self, *, repository: str, digest: str) -> tuple[str, dict[str, Any]]:
        """One `DescribeImageScanFindings`, as a status and what it answered."""
        try:
            answer = self._client().describe_image_scan_findings(
                repositoryName=repository, imageId={"imageDigest": digest}
            )
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
        return status, {
            "findings": findings,
            "completed_at": findings.get("imageScanCompletedAt"),
            "scanner": "ECR enhanced" if findings.get("enhancedFindings") else "ECR basic",
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

        Answers the signature's reference and whether this call made it. A
        replay finds the existing signature: pushing a second one under the
        same tag is what immutable tags refuse, and it would be a second thing
        claiming to be Datalayer's word on the same artifact.
        """
        self.can_sign()
        tag = signature_tag(digest)
        reference = f"{registry}/{repository}@{digest}"
        if self._signature_exists(repository=repository, tag=tag):
            self._log(f"{digest} is signed already, under {tag}")
            return f"{registry}/{repository}:{tag}", False
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
        return f"{registry}/{repository}:{tag}", True

    def _signature_exists(self, *, repository: str, tag: str) -> bool:
        try:
            answer = self._client().describe_images(
                repositoryName=repository, imageIds=[{"imageTag": tag}]
            )
        except Exception as error:
            if _is_missing(error):
                return False
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"Whether {tag} is already signed could not be read: {error}",
                detail={"repository": repository, "tag": tag},
            ) from error
        return bool(answer.get("imageDetails"))

    def _invoke(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            finished = self._run(
                list(command), capture_output=True, text=True, check=False, timeout=self._timeout
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
    ) -> AttestationResult:
        """Scan, then sign: the order the Operator's check depends on (D-11)."""
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
        signature, signed_now = self.sign(registry=registry, repository=repository, digest=digest)
        return AttestationResult(
            decision=decision,
            signature_ref=signature,
            sbom_ref=sbom_ref or f"{registry}/{repository}@{digest}.sbom",
            provenance_ref=provenance_ref or f"{registry}/{repository}@{digest}.att",
            size_bytes=size_bytes,
            signed_now=signed_now,
        )

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
    log: Callable[[str], None] | None = None,
    size_bytes: int | None = None,
    attestor: Attestor | None = None,
) -> dict[str, Any]:
    """The `attest` seam of `EnvironmentBuildWorkflow` (E1-08, E1-09).

    Takes the artifact the builder recorded and the build's credential, and
    answers the mapping the workflow stores: the scan's decision, the
    signature, the SBOM and provenance references, and the size.
    """
    reference = str(getattr(artifact, "immutable_reference", "") or "")
    registry, _, rest = reference.partition("/")
    repository, _, digest = rest.partition("@")
    if not (registry and repository and _DIGEST.match(digest)):
        raise EnvironmentsError(
            PROVIDER_ERROR,
            f"`{reference}` is not a digest in a repository, so it cannot be attested",
            detail={"reference": reference},
        )
    use = attestor or Attestor(
        key=str(getattr(credential, "signing_key", "") or ""),
        policy=policy,
        log=log,
    )
    return use.attest(
        registry=registry,
        repository=repository,
        digest=digest,
        size_bytes=size_bytes,
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
        "ScanNotFoundException",
    }


def _is_in_progress(error: BaseException) -> bool:
    """ECR answers a scan that has not finished as an error, not a status."""
    return _error_code(error) in {"ScanInProgressException", "LimitExceededException"}
