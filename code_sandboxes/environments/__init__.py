# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Environments: define once, resolve once, build an immutable artifact per variant.

An **Environment** is the stable object a person names. An **Environment
Version** is immutable once it leaves draft: its spec, the lock resolved from
it, and its build outcome. An **Environment Artifact** is the build output of
one version on one variant — an OCI digest for Datalayer, a build id for E2B,
a snapshot id for Daytona, an image id for Modal.

This package holds what every party to that life shares, and nothing
provider-specific:

- :mod:`.spec` — the canonical specification and its rules;
- :mod:`.resolve` — the lock a version is resolved into, once (D-9);
- :mod:`.policy` — whether an artifact's scan lets it be used (D-11);
- :mod:`.accounts` — whose provider account a managed artifact lives in (D-8);
- :mod:`.attest` — the scan and the signature it needs before anything runs it;
- :mod:`.contract` — ``sandbox-contract/v1`` and the Dockerfile validator;
- :mod:`.lifecycle` — the version states and their legal moves;
- :mod:`.errors` — the error taxonomy every provider error maps into;
- :mod:`.builders` — the builder interface and its neutral types;
- :mod:`.conformance` — the checks an artifact passes before it is ready;
- :mod:`.redact` — a build's log with its secrets taken out before it is stored;
- :mod:`.doctor` — the in-sandbox contract checker.

The builders themselves are in :mod:`.adapters`, loaded by variant.
"""

from __future__ import annotations

from .accounts import account_fingerprints, fingerprint_matches, provider_account
from .attest import Attestor, attest_artifact, signature_tag
from .bases import APPROVED_BASES, ApprovedBase
from .builders import (
    ArtifactReference,
    BuildRequest,
    CapabilityReport,
    CapabilitySet,
    CheckResult,
    EnvironmentBuilder,
    ValidationResult,
    get_builder,
)
from .canonical import canonical_digest, canonical_json
from .contract import (
    CONTRACT_V1,
    SANDBOX_CONTRACT_V1,
    SUPPORTED_CONTRACTS,
    SandboxContract,
    check_dockerfile,
    validate_dockerfile,
)
from .errors import ERROR_CODES, EnvironmentsError, ErrorCode, map_provider_error
from .lifecycle import VersionState, build_outcome, can_transition, transition
from .policy import DEFAULT_POLICY, PolicyDecision, ScanPolicy, decide, refuse_if_blocked
from .resolve import (
    LOCK_FORMAT,
    BuildkitResolveRunner,
    LocalResolveRunner,
    ResolveRunner,
    locked_versions,
    merge_requirements,
    parse_resolver_failure,
    protected_pins,
    resolve_environment,
)
from .spec import (
    API_VERSION,
    VARIANTS,
    Environment,
    EnvironmentSpec,
    parse_environment,
    spec_digest,
    spec_findings,
    validate_environment,
)

__all__ = [
    "API_VERSION",
    "APPROVED_BASES",
    "CONTRACT_V1",
    "DEFAULT_POLICY",
    "ERROR_CODES",
    "LOCK_FORMAT",
    "SANDBOX_CONTRACT_V1",
    "SUPPORTED_CONTRACTS",
    "VARIANTS",
    "ApprovedBase",
    "ArtifactReference",
    "Attestor",
    "BuildRequest",
    "BuildkitResolveRunner",
    "CapabilityReport",
    "CapabilitySet",
    "CheckResult",
    "Environment",
    "EnvironmentBuilder",
    "EnvironmentSpec",
    "EnvironmentsError",
    "ErrorCode",
    "LocalResolveRunner",
    "PolicyDecision",
    "ResolveRunner",
    "SandboxContract",
    "ScanPolicy",
    "ValidationResult",
    "VersionState",
    "account_fingerprints",
    "attest_artifact",
    "build_outcome",
    "can_transition",
    "canonical_digest",
    "canonical_json",
    "check_dockerfile",
    "decide",
    "fingerprint_matches",
    "get_builder",
    "locked_versions",
    "map_provider_error",
    "merge_requirements",
    "parse_environment",
    "parse_resolver_failure",
    "protected_pins",
    "provider_account",
    "refuse_if_blocked",
    "resolve_environment",
    "signature_tag",
    "spec_digest",
    "spec_findings",
    "transition",
    "validate_dockerfile",
    "validate_environment",
]
