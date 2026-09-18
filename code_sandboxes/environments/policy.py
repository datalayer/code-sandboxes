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

from .errors import POLICY_DENIED, SCAN_BLOCKED, EnvironmentsError

__all__ = [
    "DEFAULT_POLICY",
    "SEVERITIES",
    "EnvironmentsPolicy",
    "Finding",
    "PolicyDecision",
    "ScanPolicy",
    "decide",
    "environments_policy_from_rules",
    "findings_of",
    "refuse_if_blocked",
    "refuse_unlicensed",
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


#: Advisories allowed on the Datalayer base channel itself (D-9), reviewed and
#: accepted 2026-09-14 rather than silently missed: all five are in `ffmpeg`
#: (`CVE-2024-35366`, `CVE-2024-35367`, `CVE-2024-35368`, `CVE-2026-40962`) and
#: `libcjson1` (`CVE-2025-57052`, pulled in only by `librist4`, itself only
#: needed by `ffmpeg`), on `datalayer/python-cpu:2026.09`. Amazon Inspector
#: reports each as fixable, but the fix is an Ubuntu ESM (Extended Security
#: Maintenance) package version — `7:6.1.1-3ubuntu5+esm5` and the like — not
#: reachable by a plain `apt-get upgrade`, only by a Pro subscription token
#: this deployment does not hold. `ffmpeg` has no other installed package
#: depending on it (`apt-cache rdepends --installed`), so it is kept for what
#: it is: an image-wide convenience (matplotlib's `FFMpegWriter` and the like),
#: not something the sandbox contract's own doctor checks for. Revisit once
#: Ubuntu ships a non-ESM fix, or the channel drops `ffmpeg`.
_BASE_CHANNEL_ALLOWED: tuple[str, ...] = (
    "CVE-2024-35366",
    "CVE-2024-35367",
    "CVE-2024-35368",
    "CVE-2026-40962",
    "CVE-2025-57052",
)

#: What every owner is scanned against until an organization says otherwise.
DEFAULT_POLICY = ScanPolicy(allowed=_BASE_CHANNEL_ALLOWED)


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


@dataclass(frozen=True)
class EnvironmentsPolicy:
    """An organization's own narrowing of what a build may do (E3-06).

    Every allowlist is ``None`` until an organization writes one, meaning
    that dimension is unrestricted beyond the platform's own defaults — the
    approved bases, the public registries a spec's own validation already
    checks, and nothing at all for indexes, packages or licences. Setting a
    list, even an empty one, is a real policy: an organization that writes
    ``allowed_licenses=()`` has denied every licence, and this does not
    second-guess a policy that strict — the refusal it produces names
    exactly why.

    This narrows; it never widens. A base, index, registry, package or
    licence the platform already refuses stays refused whatever an
    organization allows — the same rule `refuse_widening` already keeps for
    the gateway's own policy layers (`iam/datalayer_iam/services/
    mcp_policies.py`), applied here to a different set of rules.
    """

    allowed_bases: tuple[str, ...] | None = None
    allowed_indexes: tuple[str, ...] | None = None
    allowed_registries: tuple[str, ...] | None = None
    allowed_packages: tuple[str, ...] | None = None
    allowed_licenses: tuple[str, ...] | None = None
    scan: ScanPolicy = DEFAULT_POLICY
    #: The policy document's own version, carried into E1-08's decision
    #: record so a build made under one policy is not misread once an
    #: organization changes it — the version answers "which policy" without
    #: needing IAM asked again.
    version: int | None = None

    def allows(self, dimension: str, value: str) -> bool:
        """Whether `value` clears this policy's allowlist for `dimension`.

        `True` when the dimension is unrestricted (`None`) or the value is on
        the list; the comparison is exact, not a prefix or a host match — an
        index or a registry is compared the way `image_registry_allowed`
        already compares one, and a base or a package by its own name.
        """
        allowed = getattr(self, f"allowed_{dimension}")
        return allowed is None or value in allowed


#: No organization has written a policy: every dimension unrestricted, the
#: platform's own default scan threshold. The reading a caller with no
#: policy document gets, and what every check below is a no-op against.
UNRESTRICTED_POLICY = EnvironmentsPolicy()

#: What `environments_policy_from_rules` reads a list-shaped field as. Kept
#: beside `ScanPolicy`'s own fields so the two are validated the same way.
_LIST_FIELDS: tuple[str, ...] = (
    "allowedBases",
    "allowedIndexes",
    "allowedRegistries",
    "allowedPackages",
    "allowedLicenses",
)


def environments_policy_from_rules(
    rules: Mapping[str, Any] | None, *, version: int | None = None
) -> EnvironmentsPolicy:
    """The `environments` section of an organization's MCP policy document,
    as the shape this module checks against (E3-06).

    `rules` is the raw object IAM stores under the policy's own
    ``environments`` key — camelCase, the same convention every other rule in
    `iam/datalayer_iam/services/mcp_policies.py` already uses. Absent, or not
    an object, answers :data:`UNRESTRICTED_POLICY`: an organization that has
    not written this section has narrowed nothing, the same reading every
    other rule in that module gives an unset one.

    Never raises. A caller holding an organization's policy already trusts
    IAM to have validated it at the write (`mcp_policies.validate_rules`);
    asking twice, differently, would let the two readings disagree about
    what a stored policy means.
    """
    if not isinstance(rules, Mapping):
        return UNRESTRICTED_POLICY
    lists: dict[str, tuple[str, ...] | None] = {}
    for camel in _LIST_FIELDS:
        value = rules.get(camel)
        snake = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in camel
        )  # allowedBases -> allowed_bases
        lists[snake] = (
            tuple(str(item) for item in value) if isinstance(value, (list, tuple)) else None
        )
    scan_rules = rules.get("scan") if isinstance(rules.get("scan"), Mapping) else {}
    blocks_at = str(scan_rules.get("blocksAt") or DEFAULT_POLICY.blocks_at).upper()
    scan = ScanPolicy(
        blocks_at=blocks_at if blocks_at in SEVERITIES else DEFAULT_POLICY.blocks_at,
        only_fixable=bool(scan_rules.get("onlyFixable", DEFAULT_POLICY.only_fixable)),
        allowed=tuple(str(item) for item in (scan_rules.get("allowed") or ()))
        or DEFAULT_POLICY.allowed,
    )
    return EnvironmentsPolicy(scan=scan, version=version, **lists)


def refuse_unlicensed(
    pairs: Sequence[tuple[str, str]], policy: EnvironmentsPolicy = UNRESTRICTED_POLICY
) -> None:
    """Raise `DL_ENV_POLICY_DENIED` on the first licence this policy does not
    allow, naming the package that carries it (E3-06).

    `pairs` is `attest.licenses_by_package`'s own output: `(package,
    licence)`, in the SBOM's own order, so the same artifact always names the
    same first offender. A no-op when `policy.allowed_licenses` is unset —
    every artifact today, until an organization writes one.
    """
    if policy.allowed_licenses is None:
        return
    for package, licence in pairs:
        if not policy.allows("licenses", licence):
            raise EnvironmentsError(
                POLICY_DENIED,
                f"`{licence}` ({package or 'an unnamed package'}) is not an allowed "
                f"licence (the allowed ones are {', '.join(policy.allowed_licenses) or '(none)'})",
                detail={
                    "field": "licenses",
                    "package": package,
                    "license": licence,
                    "allowed": list(policy.allowed_licenses),
                    **({"policyVersion": policy.version} if policy.version is not None else {}),
                },
            )
