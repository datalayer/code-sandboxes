# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Whether an artifact's scan lets it be used (PLAN_ENV.md, E1-08, D-11).

A build is scanned before anything runs from it, and something has to decide
what the findings mean. That decision is here, and it is pure: the findings in,
a record out, so what blocked a build can be re-read a month later without
asking the scanner again.

The default, taken by the owner on 2026-09-11: **a critical finding that has a
fixed version blocks**. Both halves matter.

- *Critical* alone would block on a `HIGH` nobody can act on, and a severity
  floor that blocks everything makes the scan a formality people turn off.
- *With a fixed version* is the part that makes the refusal actionable: it
  names the package, the version installed and the version to ask for. A
  critical finding with no fix available blocks nothing, because blocking it
  would leave the owner with nothing to do but wait — it is recorded, and the
  decision says how many were recorded that way.

An organization tightens this in E3-06; the shape is here so that when it
does, only the threshold changes.

@module code_sandboxes.environments.policy
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .errors import SCAN_BLOCKED, EnvironmentsError

__all__ = [
    "DEFAULT_POLICY",
    "SEVERITIES",
    "Finding",
    "PolicyDecision",
    "ScanPolicy",
    "decide",
    "findings_of",
    "refuse_if_blocked",
]

#: The severities a scanner reports, weakest first. Anything it reports that is
#: not one of these — `UNTRIAGED`, `UNDEFINED` — counts as `INFORMATIONAL`:
#: unknown is not a reason to block, and not a reason to lose the finding.
SEVERITIES: tuple[str, ...] = (
    "INFORMATIONAL",
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
)


def _rank(severity: str) -> int:
    name = str(severity or "").strip().upper()
    return SEVERITIES.index(name) if name in SEVERITIES else 0


@dataclass(frozen=True)
class Finding:
    """One thing a scan found, as every scanner can say it."""

    id: str
    """The advisory: `CVE-2026-1234`, `GHSA-…`."""

    severity: str
    package: str = ""
    installed_version: str = ""
    fixed_version: str = ""
    """The version that fixes it, when the scanner knows one."""

    uri: str = ""

    @property
    def fixable(self) -> bool:
        return bool(self.fixed_version.strip())

    def body(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "package": self.package,
            "installedVersion": self.installed_version,
            "fixedVersion": self.fixed_version,
            **({"uri": self.uri} if self.uri else {}),
        }

    def said(self) -> str:
        """The finding as a person reads it in a refusal."""
        where = f" in {self.package} {self.installed_version}".rstrip()
        fix = f"; fixed in {self.fixed_version}" if self.fixable else "; no fix available"
        return f"{self.id} ({self.severity}){where}{fix}"


@dataclass(frozen=True)
class ScanPolicy:
    """What a scan has to say for an artifact to be used.

    `blocks_at` is the weakest severity that blocks, and `only_fixable` says
    whether a finding with no fix available blocks with it. The default is
    `CRITICAL` and `True`: the owner's decision of 2026-09-11.
    """

    blocks_at: str = "CRITICAL"
    only_fixable: bool = True
    #: Advisories this owner has looked at and accepted, by id. An allowance is
    #: a decision somebody made; it is recorded in the decision so the next
    #: reader sees it was allowed rather than missed.
    allowed: tuple[str, ...] = ()

    def blocks(self, finding: Finding) -> bool:
        if finding.id in self.allowed:
            return False
        if _rank(finding.severity) < _rank(self.blocks_at):
            return False
        return finding.fixable or not self.only_fixable

    def body(self) -> dict[str, Any]:
        return {
            "blocksAt": self.blocks_at,
            "onlyFixable": self.only_fixable,
            **({"allowed": list(self.allowed)} if self.allowed else {}),
        }


#: What every owner is scanned against until an organization says otherwise.
DEFAULT_POLICY = ScanPolicy()


@dataclass(frozen=True)
class PolicyDecision:
    """Why an artifact was used, or was not: the record stored in `scan_summary`.

    Everything needed to re-read the decision without the scanner: what was
    scanned, what the threshold was, what was found by severity, and which
    findings blocked it.
    """

    decision: str
    """`pass` or `blocked`."""

    policy: ScanPolicy
    counts: Mapping[str, int]
    blocking: Sequence[Finding] = field(default_factory=tuple)
    recorded_unfixable: int = 0
    """Findings at or above the threshold that nothing fixes yet."""

    scanner: str = ""
    scanned_at: str = ""
    scan_status: str = ""

    @property
    def passed(self) -> bool:
        return self.decision == "pass"

    def body(self) -> dict[str, Any]:
        """The record, as the artifact stores it and the CLI prints it."""
        return {
            "decision": self.decision,
            "policy": self.policy.body(),
            "counts": {name: int(count) for name, count in self.counts.items() if count},
            "critical": int(self.counts.get("CRITICAL", 0)),
            "blocking": [finding.body() for finding in self.blocking],
            "unfixable": self.recorded_unfixable,
            **({"scanner": self.scanner} if self.scanner else {}),
            **({"scannedAt": self.scanned_at} if self.scanned_at else {}),
            **({"scanStatus": self.scan_status} if self.scan_status else {}),
        }

    def said(self) -> str:
        """Why the build stopped, naming the findings a person has to act on."""
        if self.passed:
            return "the scan passed"
        named = "; ".join(finding.said() for finding in self.blocking[:5])
        more = len(self.blocking) - 5
        return named + (f"; and {more} more" if more > 0 else "")


def findings_of(reported: Iterable[Mapping[str, Any]]) -> list[Finding]:
    """ECR's `DescribeImageScanFindings` rows as findings (E1-08).

    Enhanced scanning answers `enhancedFindings`, whose shape differs from
    basic scanning's `findings`; both are read here, because an operator
    switching a repository between them should not change what blocks.
    """
    read: list[Finding] = []
    for row in reported:
        if not isinstance(row, Mapping):
            continue
        vulnerability = row.get("packageVulnerabilityDetails")
        if isinstance(vulnerability, Mapping):
            # Enhanced: one finding, possibly several packages.
            packages = [
                package
                for package in (vulnerability.get("vulnerablePackages") or [])
                if isinstance(package, Mapping)
            ] or [{}]
            for package in packages:
                read.append(
                    Finding(
                        id=str(vulnerability.get("vulnerabilityId") or row.get("title") or ""),
                        severity=str(row.get("severity") or "").upper(),
                        package=str(package.get("name") or ""),
                        installed_version=str(package.get("version") or ""),
                        fixed_version=str(package.get("fixedInVersion") or ""),
                        uri=str(vulnerability.get("sourceUrl") or ""),
                    )
                )
            continue
        attributes = {
            str(item.get("key")): str(item.get("value"))
            for item in (row.get("attributes") or [])
            if isinstance(item, Mapping)
        }
        read.append(
            Finding(
                id=str(row.get("name") or ""),
                severity=str(row.get("severity") or "").upper(),
                package=attributes.get("package_name", ""),
                installed_version=attributes.get("package_version", ""),
                # Basic scanning reports no fixed version. A finding nobody
                # can act on does not block under the default policy, and
                # saying so is the honest reading of a scanner that cannot
                # answer the question.
                fixed_version="",
                uri=str(row.get("uri") or ""),
            )
        )
    return [finding for finding in read if finding.id]


def decide(
    findings: Iterable[Finding],
    *,
    policy: ScanPolicy = DEFAULT_POLICY,
    scanner: str = "",
    scanned_at: str = "",
    scan_status: str = "",
) -> PolicyDecision:
    """What the findings amount to, under this policy."""
    read = list(findings)
    counts: dict[str, int] = dict.fromkeys(SEVERITIES, 0)
    for finding in read:
        name = str(finding.severity or "").upper()
        counts[name if name in counts else "INFORMATIONAL"] += 1
    blocking = [finding for finding in read if policy.blocks(finding)]
    at_threshold = [
        finding for finding in read if _rank(finding.severity) >= _rank(policy.blocks_at)
    ]
    unfixable = sum(1 for finding in at_threshold if not finding.fixable)
    return PolicyDecision(
        decision="blocked" if blocking else "pass",
        policy=policy,
        counts=counts,
        blocking=tuple(blocking),
        recorded_unfixable=unfixable,
        scanner=scanner,
        scanned_at=scanned_at,
        scan_status=scan_status,
    )


def refuse_if_blocked(decision: PolicyDecision, *, reference: str = "") -> None:
    """Raise `DL_ENV_SCAN_BLOCKED` when the decision blocked, naming the findings."""
    if decision.passed:
        return
    raise EnvironmentsError(
        SCAN_BLOCKED,
        f"The scan blocks this artifact: {decision.said()}",
        detail={
            **({"reference": reference} if reference else {}),
            "decision": decision.body(),
            "blocking": [finding.id for finding in decision.blocking],
        },
    )
