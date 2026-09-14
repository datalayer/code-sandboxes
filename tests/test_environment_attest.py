# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The two gates between a build and a sandbox (PLAN_ENV.md E1-08, E1-09, D-11).

The scan decides whether an artifact may be used, and the signature is
Datalayer's word that it was decided. Nothing here reaches AWS or runs cosign:
the ECR client is a double answering recorded findings, and cosign is a
function whose argv is read.

The findings below are shaped as ECR's `DescribeImageScanFindings` answers
them — `enhancedFindings` with `packageVulnerabilityDetails`, and the older
`findings` with `attributes` — because the rule being tested is what those
rows amount to, and a paraphrase of them would test the paraphrase.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from code_sandboxes.environments.attest import Attestor, attest_artifact, signature_tag
from code_sandboxes.environments.builders import ArtifactReference
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.policy import (
    DEFAULT_POLICY,
    Finding,
    ScanPolicy,
    decide,
    findings_of,
    refuse_if_blocked,
)

REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
REPOSITORY = "environments/u/01k0wner000000000000000000/geo"
DIGEST = "sha256:" + "aa" * 32
KEY = "awskms:///alias/datalayer-environments"


def enhanced(
    identifier: str,
    severity: str,
    *,
    package: str = "libxml2",
    version: str = "2.9.14",
    fixed: str = "",
) -> dict:
    """One `enhancedFindings` row, as ECR's enhanced scanning answers it."""
    return {
        "severity": severity,
        "title": identifier,
        "packageVulnerabilityDetails": {
            "vulnerabilityId": identifier,
            "sourceUrl": f"https://nvd.nist.gov/vuln/detail/{identifier}",
            "vulnerablePackages": [
                {
                    "name": package,
                    "version": version,
                    **({"fixedInVersion": fixed} if fixed else {}),
                }
            ],
        },
    }


def basic(identifier: str, severity: str, *, package: str = "openssl") -> dict:
    """One older `findings` row, which reports no fixed version."""
    return {
        "name": identifier,
        "severity": severity,
        "uri": f"https://security.example/{identifier}",
        "attributes": [
            {"key": "package_name", "value": package},
            {"key": "package_version", "value": "3.0.2"},
        ],
    }


class FakeEcr:
    """The two ECR calls the attestor makes, and nothing else."""

    def __init__(
        self,
        *,
        statuses=("ACTIVE",),
        findings=(),
        enhanced_findings=True,
        signed: set[str] | None = None,
        manifest: dict | None = None,
        pages: list[list[dict]] | None = None,
    ) -> None:
        self.statuses = list(statuses)
        self.findings = list(findings)
        #: Each entry is one `describe_image_scan_findings` page's own
        #: findings, read in order and merged by `_describe` — overrides
        #: `findings` when given, to test reading more than one page.
        self.pages = pages
        self.enhanced_findings = enhanced_findings
        self.signed = set(signed or set())
        self.manifest = manifest
        self.asked = 0
        self.scanned: list[str] = []
        self.described_tags: list[str] = []

    def batch_get_image(self, repositoryName, imageIds, acceptedMediaTypes):  # noqa: N803 - boto3's spelling
        manifest = self.manifest or {
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "layers": [],
        }
        return {
            "images": [
                {
                    "imageId": imageIds[0],
                    "imageManifestMediaType": manifest["mediaType"],
                    "imageManifest": json.dumps(manifest),
                }
            ]
        }

    def describe_image_scan_findings(self, repositoryName, imageId, nextToken=None):  # noqa: N803 - boto3's spelling
        self.asked += 1
        self.scanned.append(imageId["imageDigest"])
        status = self.statuses[min(self.asked, len(self.statuses)) - 1]
        if status == "RAISES_IN_PROGRESS":
            error = Exception("ScanInProgressException")
            error.response = {"Error": {"Code": "ScanInProgressException"}}
            raise error
        if status == "SCAN_NOT_FOUND":
            error = Exception("ScanNotFoundException")
            error.response = {"Error": {"Code": "ScanNotFoundException"}}
            raise error
        if status == "MISSING":
            error = Exception("ImageNotFoundException")
            error.response = {"Error": {"Code": "ImageNotFoundException"}}
            raise error
        key = "enhancedFindings" if self.enhanced_findings else "findings"
        if self.pages is not None:
            index = int(nextToken or "0")
            answer = {
                "imageScanStatus": {"status": status},
                "imageScanFindings": {
                    key: list(self.pages[index]),
                    "imageScanCompletedAt": "2026-09-12T09:00:00Z",
                },
            }
            if index + 1 < len(self.pages):
                answer["nextToken"] = str(index + 1)
            return answer
        return {
            "imageScanStatus": {"status": status},
            "imageScanFindings": {
                key: list(self.findings),
                "imageScanCompletedAt": "2026-09-12T09:00:00Z",
            },
        }

    def describe_images(self, repositoryName, imageIds):  # noqa: N803 - boto3's spelling
        tag = imageIds[0].get("imageTag") or ""
        self.described_tags.append(tag)
        if tag in self.signed:
            return {"imageDetails": [{"imageTags": [tag]}]}
        error = Exception("ImageNotFoundException")
        error.response = {"Error": {"Code": "ImageNotFoundException"}}
        raise error


class Cosign:
    """A cosign whose argv is read, and which can refuse."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.argv: list[str] = []

    def __call__(self, argv, **_kwargs) -> subprocess.CompletedProcess[str]:
        self.argv = list(argv)
        return subprocess.CompletedProcess(self.argv, self.returncode, "", "Pushing signature\n")


def an_attestor(**changes) -> Attestor:
    options = {
        "ecr": FakeEcr(),
        "cosign": "/usr/bin/cosign",
        "key": KEY,
        "run": Cosign(),
        "sleep": lambda _seconds: None,
        "now": lambda: 0.0,
    }
    options.update(changes)
    return Attestor(**options)


# -- what the findings amount to ------------------------------------------------


class TestThePolicy:
    def test_a_critical_finding_with_a_fix_blocks(self) -> None:
        found = findings_of([enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")])
        decision = decide(found)
        assert decision.decision == "blocked"
        assert [finding.id for finding in decision.blocking] == ["CVE-2026-1234"]
        # And the refusal names what to do about it.
        assert "fixed in 2.9.15" in decision.said()
        assert "libxml2 2.9.14" in decision.said()

    def test_a_critical_finding_nothing_fixes_is_recorded_and_does_not_block(self) -> None:
        """Blocking it would leave the owner nothing to do but wait."""
        decision = decide(findings_of([enhanced("CVE-2026-9999", "CRITICAL")]))
        assert decision.decision == "pass"
        assert decision.recorded_unfixable == 1
        assert decision.body()["critical"] == 1
        assert decision.body()["blocking"] == []

    def test_a_high_finding_with_a_fix_does_not_block_under_the_default(self) -> None:
        decision = decide(findings_of([enhanced("CVE-2026-5678", "HIGH", fixed="3.0.3")]))
        assert decision.decision == "pass"
        assert decision.body()["counts"] == {"HIGH": 1}

    def test_an_organization_can_tighten_it(self) -> None:
        strict = ScanPolicy(blocks_at="HIGH", only_fixable=False)
        decision = decide(findings_of([enhanced("CVE-2026-9999", "HIGH")]), policy=strict)
        assert decision.decision == "blocked"
        assert decision.body()["policy"] == {"blocksAt": "HIGH", "onlyFixable": False}

    def test_an_allowed_advisory_is_recorded_as_allowed_rather_than_missed(self) -> None:
        policy = ScanPolicy(allowed=("CVE-2026-1234",))
        decision = decide(
            findings_of([enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")]), policy=policy
        )
        assert decision.decision == "pass"
        assert decision.body()["policy"]["allowed"] == ["CVE-2026-1234"]
        # The finding is still counted: an allowance hides nothing.
        assert decision.body()["counts"] == {"CRITICAL": 1}

    def test_the_default_policy_allows_the_base_channels_esm_locked_ffmpeg_advisories(
        self,
    ) -> None:
        """`datalayer/python-cpu:2026.09`'s own five (E1-05, E1-08, 2026-09-14):
        Inspector marks each fixable, but the fix is an Ubuntu ESM package
        version a plain `apt-get upgrade` cannot reach. Reviewed and allowed,
        not silently missed — still counted, and any other critical still
        blocks."""
        allowed = (
            "CVE-2024-35366",
            "CVE-2024-35367",
            "CVE-2024-35368",
            "CVE-2026-40962",
            "CVE-2025-57052",
        )
        for cve in allowed:
            decision = decide(
                findings_of([enhanced(cve, "CRITICAL", package="ffmpeg", fixed="7:6.1.1-esm")])
            )
            assert decision.decision == "pass", cve
            assert decision.body()["policy"]["allowed"] == list(allowed)
        # A critical outside that fixed list still blocks under the default.
        decision = decide(findings_of([enhanced("CVE-2026-9999", "CRITICAL", fixed="1.2.3")]))
        assert decision.decision == "blocked"

    def test_the_older_findings_shape_is_read_too(self) -> None:
        found = findings_of([basic("CVE-2026-1111", "CRITICAL")])
        assert [(finding.id, finding.package, finding.fixable) for finding in found] == [
            ("CVE-2026-1111", "openssl", False)
        ]
        # Basic scanning knows no fixed version, so under the default nothing
        # it reports blocks — and the decision says how many were recorded.
        decision = decide(found)
        assert (decision.decision, decision.recorded_unfixable) == ("pass", 1)

    def test_a_severity_nobody_knows_is_kept_and_does_not_block(self) -> None:
        decision = decide(findings_of([enhanced("CVE-2026-2222", "UNTRIAGED", fixed="1.0")]))
        assert decision.decision == "pass"
        assert decision.body()["counts"] == {"INFORMATIONAL": 1}

    def test_the_record_holds_what_it_was_decided_from(self) -> None:
        decision = decide(
            findings_of([enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")]),
            scanner="ECR enhanced",
            scanned_at="2026-09-12T09:00:00Z",
            scan_status="ACTIVE",
        )
        body = decision.body()
        assert body["scanner"] == "ECR enhanced"
        assert body["scannedAt"] == "2026-09-12T09:00:00Z"
        assert body["scanStatus"] == "ACTIVE"
        assert body["blocking"][0]["fixedVersion"] == "2.9.15"

    def test_refusing_names_the_cve_and_carries_the_record(self) -> None:
        decision = decide(findings_of([enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")]))
        with pytest.raises(EnvironmentsError) as raised:
            refuse_if_blocked(decision, reference=f"{REGISTRY}/{REPOSITORY}@{DIGEST}")
        assert raised.value.code.code == "DL_ENV_SCAN_BLOCKED"
        assert "CVE-2026-1234" in raised.value.message
        assert raised.value.detail["blocking"] == ["CVE-2026-1234"]
        assert raised.value.detail["decision"]["decision"] == "blocked"

    def test_a_passing_decision_refuses_nothing(self) -> None:
        refuse_if_blocked(decide([Finding(id="CVE-1", severity="LOW")]))  # no raise


# -- waiting for the scan -------------------------------------------------------


class TestTheScan:
    def test_a_finished_scan_is_read_and_decided(self) -> None:
        ecr = FakeEcr(findings=[enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")])
        decision = an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert decision.decision == "blocked"
        assert ecr.asked == 1

    def test_it_waits_while_the_scan_is_in_progress(self) -> None:
        ecr = FakeEcr(statuses=("IN_PROGRESS", "IN_PROGRESS", "ACTIVE"), findings=[])
        waits: list[float] = []
        decision = an_attestor(ecr=ecr, sleep=waits.append).scan(
            repository=REPOSITORY, digest=DIGEST
        )
        assert decision.passed
        assert ecr.asked == 3 and waits == [10.0, 10.0]

    def test_a_scan_that_answers_with_an_error_while_it_runs_is_waited_for_too(self) -> None:
        ecr = FakeEcr(statuses=("RAISES_IN_PROGRESS", "ACTIVE"))
        assert an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST).passed

    def test_a_scan_that_never_finishes_is_retryable(self) -> None:
        ecr = FakeEcr(statuses=("IN_PROGRESS",))
        clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr, now=lambda: next(clock)).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert raised.value.code.retry.value != "no"

    def test_a_scanner_that_refuses_the_image_says_so(self) -> None:
        ecr = FakeEcr(statuses=("UNSUPPORTED_IMAGE",))
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.detail["scanStatus"] == "UNSUPPORTED_IMAGE"

    def test_an_image_the_registry_has_no_scan_for_is_named(self) -> None:
        ecr = FakeEcr(statuses=("MISSING",))
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert raised.value.detail["digest"] == DIGEST

    def test_every_page_of_findings_is_read_before_deciding(self) -> None:
        """Found live, 2026-09-14: a single unpaginated call answered only its
        first page, and a real image with 1,547 findings across many pages
        passed a decision that should have blocked — none of its 31 critical
        findings were on the one page a single call happened to see. Every
        page is now read and merged before `decide` runs."""
        page_1 = [enhanced("CVE-2026-1111", "LOW")]
        page_2 = [enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")]
        page_3 = [enhanced("CVE-2026-2222", "MEDIUM")]
        ecr = FakeEcr(pages=[page_1, page_2, page_3])
        decision = an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert decision.decision == "blocked"
        assert [finding.id for finding in decision.blocking] == ["CVE-2026-1234"]
        assert decision.body()["counts"] == {"LOW": 1, "CRITICAL": 1, "MEDIUM": 1}
        assert ecr.asked == 3

    def test_reading_pages_stops_at_the_cap_rather_than_paginating_forever(self) -> None:
        pages = [
            [enhanced(f"CVE-2026-{n:04d}", "LOW")] for n in range(Attestor._MAX_FINDING_PAGES + 5)
        ]
        ecr = FakeEcr(pages=pages)
        decision = an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert decision.decision == "pass"
        assert ecr.asked == Attestor._MAX_FINDING_PAGES

    IMAGE = "sha256:" + "bb" * 32
    ATTESTATION = "sha256:" + "cc" * 32

    def an_index(self, *entries: dict) -> dict:
        return {"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": list(entries)}

    def attestation_entry(self) -> dict:
        return {
            "digest": self.ATTESTATION,
            "platform": {"os": "unknown", "architecture": "unknown"},
            "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
        }

    def test_an_index_is_decided_by_the_scan_of_its_linux_amd64_image(self) -> None:
        """The builder pushes with attestations, so what it records is an
        index, and the scanner answers `UNSUPPORTED_IMAGE` for an index (found
        live on r1, 2026-09-14)."""
        ecr = FakeEcr(
            manifest=self.an_index(
                self.attestation_entry(),
                {"digest": self.IMAGE, "platform": {"os": "linux", "architecture": "amd64"}},
            ),
            findings=[enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")],
        )
        decision = an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert ecr.scanned == [self.IMAGE]
        assert decision.decision == "blocked"

    def test_an_index_with_no_linux_amd64_image_is_named(self) -> None:
        ecr = FakeEcr(manifest=self.an_index(self.attestation_entry()))
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert "linux/amd64" in raised.value.message
        assert ecr.scanned == []

    def test_an_image_pushed_a_moment_ago_is_waited_for_until_its_scan_exists(self) -> None:
        """Enhanced scanning starts after the push: the first answers for a
        new image are `ScanNotFoundException` (found live on r1, 2026-09-14)."""
        ecr = FakeEcr(statuses=("SCAN_NOT_FOUND", "SCAN_NOT_FOUND", "ACTIVE"))
        waits: list[float] = []
        decision = an_attestor(ecr=ecr, sleep=waits.append).scan(
            repository=REPOSITORY, digest=DIGEST
        )
        assert decision.passed
        assert ecr.asked == 3 and waits == [10.0, 10.0]

    def test_a_scan_that_never_appears_is_retryable(self) -> None:
        ecr = FakeEcr(statuses=("SCAN_NOT_FOUND",))
        clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr, now=lambda: next(clock)).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert raised.value.code.retry.value != "no"

    def test_a_single_image_is_scanned_by_its_own_digest(self) -> None:
        ecr = FakeEcr()
        an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert ecr.scanned == [DIGEST]

    def test_a_refused_scan_names_both_the_artifact_and_the_image_read(self) -> None:
        ecr = FakeEcr(
            statuses=("UNSUPPORTED_IMAGE",),
            manifest=self.an_index(
                {"digest": self.IMAGE, "platform": {"os": "linux", "architecture": "amd64"}}
            ),
        )
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(ecr=ecr).scan(repository=REPOSITORY, digest=DIGEST)
        assert raised.value.detail["digest"] == DIGEST
        assert raised.value.detail["scannedDigest"] == self.IMAGE


# -- the signature --------------------------------------------------------------


class TestTheSignature:
    def test_it_signs_the_digest_with_the_kms_key(self) -> None:
        cosign = Cosign()
        reference, signed_now = an_attestor(run=cosign).sign(
            registry=REGISTRY, repository=REPOSITORY, digest=DIGEST
        )
        assert cosign.argv == [
            "/usr/bin/cosign",
            "sign",
            "--yes",
            # Never the public transparency log: found live 2026-09-13,
            # a bare `cosign sign` reaches for the public Rekor service by
            # default and prompts for consent to publish an immutable
            # record — the wrong default for a private environment.
            "--tlog-upload=false",
            # cosign 3.1.3 defaults to a TUF-provided signing config that
            # `--tlog-upload=false` alone can no longer override (found live
            # 2026-09-14): turned off so the plain flag is honored again.
            "--use-signing-config=false",
            "--key",
            KEY,
            f"{REGISTRY}/{REPOSITORY}@{DIGEST}",
        ]
        assert reference == f"{REGISTRY}/{REPOSITORY}:{signature_tag(DIGEST)}"
        assert signed_now is True

    def test_a_replay_finds_the_signature_instead_of_pushing_a_second(self) -> None:
        """Immutable tags would refuse the second, and two signatures are two words."""
        ecr = FakeEcr(signed={signature_tag(DIGEST)})
        cosign = Cosign()
        reference, signed_now = an_attestor(ecr=ecr, run=cosign).sign(
            registry=REGISTRY, repository=REPOSITORY, digest=DIGEST
        )
        assert signed_now is False
        assert cosign.argv == []
        assert reference.endswith(signature_tag(DIGEST))

    def test_cosign_refusing_is_a_provider_error(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(run=Cosign(returncode=1)).sign(
                registry=REGISTRY, repository=REPOSITORY, digest=DIGEST
            )
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_no_key_is_said_by_name_rather_than_signing_nothing(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(key="").sign(registry=REGISTRY, repository=REPOSITORY, digest=DIGEST)
        assert raised.value.detail["missing"] == "DATALAYER_ENVIRONMENTS_KMS_KEY"

    def test_no_cosign_is_said_by_name(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            an_attestor(cosign="").sign(registry=REGISTRY, repository=REPOSITORY, digest=DIGEST)
        assert raised.value.detail["missing"] == "cosign"

    def test_the_signature_sits_beside_the_image_at_cosigns_own_tag(self) -> None:
        assert signature_tag(DIGEST) == "sha256-" + "aa" * 32 + ".sig"
        with pytest.raises(ValueError):
            signature_tag("not-a-digest")


# -- both, in order -------------------------------------------------------------


class TestAttestingAnArtifact:
    def an_artifact(self) -> ArtifactReference:
        return ArtifactReference(
            variant="datalayer",
            immutable_reference=f"{REGISTRY}/{REPOSITORY}@{DIGEST}",
            provider_artifact_id=DIGEST,
            contract_version="sandbox-contract/v1",
        )

    def test_a_clean_artifact_is_scanned_then_signed(self) -> None:
        cosign = Cosign()
        attestor = an_attestor(run=cosign)
        answer = attest_artifact(
            artifact=self.an_artifact(), size_bytes=116_183_040, attestor=attestor
        )
        assert answer["scan_summary"]["decision"] == "pass"
        assert answer["signature_ref"].endswith(signature_tag(DIGEST))
        assert answer["sbom_ref"].endswith(".sbom")
        assert answer["provenance_ref"].endswith(".att")
        assert answer["size_bytes"] == 116_183_040
        assert cosign.argv, "a clean artifact is signed"

    def test_a_blocked_artifact_is_never_signed(self) -> None:
        """The order is the point: a signature is Datalayer's word on it."""
        cosign = Cosign()
        ecr = FakeEcr(findings=[enhanced("CVE-2026-1234", "CRITICAL", fixed="2.9.15")])
        with pytest.raises(EnvironmentsError) as raised:
            attest_artifact(artifact=self.an_artifact(), attestor=an_attestor(ecr=ecr, run=cosign))
        assert raised.value.code.code == "DL_ENV_SCAN_BLOCKED"
        assert "CVE-2026-1234" in raised.value.message
        assert cosign.argv == [], "a blocked artifact must not be signed"

    def test_a_reference_that_is_not_a_digest_cannot_be_attested(self) -> None:
        artifact = ArtifactReference(
            variant="modal",
            immutable_reference="im-1234567890",
            provider_artifact_id="im-1234567890",
            contract_version="sandbox-contract/v1",
        )
        with pytest.raises(EnvironmentsError) as raised:
            attest_artifact(artifact=artifact, attestor=an_attestor())
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_a_malformed_digest_cannot_be_attested_either(self) -> None:
        """`sha256:bad` starts with `sha256:` too: only a whole one is
        accepted (found on PR #27's Copilot review). `e2b` rather than
        `datalayer`, whose own model validator already refuses a malformed
        digest before this function ever sees it — `attest_artifact` takes
        `artifact: Any`, so nothing guarantees every caller went through it."""
        artifact = ArtifactReference(
            variant="e2b",
            immutable_reference=f"{REGISTRY}/{REPOSITORY}@sha256:bad",
            provider_artifact_id="sha256:bad",
            contract_version="sandbox-contract/v1",
        )
        with pytest.raises(EnvironmentsError) as raised:
            attest_artifact(artifact=artifact, attestor=an_attestor())
        assert raised.value.code.code == "DL_ENV_PROVIDER_ERROR"

    def test_the_policy_it_was_decided_under_is_part_of_the_record(self) -> None:
        answer = attest_artifact(artifact=self.an_artifact(), attestor=an_attestor())
        assert answer["scan_summary"]["policy"] == DEFAULT_POLICY.body()


def test_nothing_reaches_a_registry_when_nothing_could_sign() -> None:
    """The order that keeps a refusal cheap and honest (E1-08, E1-09).

    An artifact nobody can sign can never be used, so waiting a quarter of an
    hour for its scan first would spend the wait to reach the same refusal —
    and would read in the log as the scan being the problem.
    """
    ecr = FakeEcr()
    artifact = ArtifactReference(
        variant="datalayer",
        immutable_reference=f"{REGISTRY}/{REPOSITORY}@{DIGEST}",
        provider_artifact_id=DIGEST,
        contract_version="sandbox-contract/v1",
    )
    with pytest.raises(EnvironmentsError) as raised:
        attest_artifact(artifact=artifact, attestor=an_attestor(ecr=ecr, key=""))
    assert raised.value.detail["missing"] == "DATALAYER_ENVIRONMENTS_KMS_KEY"
    assert ecr.asked == 0, "the registry was asked before anything could have been signed"
