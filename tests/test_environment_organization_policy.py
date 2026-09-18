# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""An organization's own narrowing of what a build may do (PLAN_ENV.md E3-06).

`EnvironmentsPolicy` and its reading from an organization's raw MCP policy
document — the shape `iam/datalayer_iam/services/mcp_policies.py` stores under
the policy's own `environments` key. The spec-level checks that read it
(bases, indexes, registries, packages) live with `spec_findings` in
`test_environment_spec.py`; the licence check lives with attestation in
`test_environment_attest.py`, since a licence is only known once the SBOM is.
This file is the shape and the parsing alone.
"""

from __future__ import annotations

import pytest

from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.policy import (
    DEFAULT_POLICY,
    UNRESTRICTED_POLICY,
    EnvironmentsPolicy,
    environments_policy_from_rules,
    refuse_unlicensed,
)


class TestEnvironmentsPolicyFromRules:
    """Reading the `environments` section of a stored MCP policy document."""

    def test_no_rules_is_unrestricted(self) -> None:
        assert environments_policy_from_rules(None) == UNRESTRICTED_POLICY
        assert environments_policy_from_rules({}) == UNRESTRICTED_POLICY

    def test_something_that_is_not_an_object_is_unrestricted_too(self) -> None:
        """A stored policy is trusted, not re-validated here (IAM already
        did, at the write) — but a caller handing this the wrong shape by
        accident must not be refused everything for it."""
        for rules in ("environments", 12, ["a", "list"]):
            assert environments_policy_from_rules(rules) == UNRESTRICTED_POLICY  # type: ignore[arg-type]

    def test_every_allowlist_is_read_by_its_camelcase_name(self) -> None:
        rules = {
            "allowedBases": ["datalayer/python-cpu"],
            "allowedIndexes": ["https://pypi.org/simple"],
            "allowedRegistries": ["docker.io"],
            "allowedPackages": ["numpy", "pandas"],
            "allowedLicenses": ["MIT", "Apache-2.0"],
        }
        policy = environments_policy_from_rules(rules)
        assert policy.allowed_bases == ("datalayer/python-cpu",)
        assert policy.allowed_indexes == ("https://pypi.org/simple",)
        assert policy.allowed_registries == ("docker.io",)
        assert policy.allowed_packages == ("numpy", "pandas")
        assert policy.allowed_licenses == ("MIT", "Apache-2.0")

    def test_a_dimension_not_written_stays_unrestricted(self) -> None:
        """One allowlist narrowed, the rest of the platform's own defaults."""
        policy = environments_policy_from_rules({"allowedLicenses": ["MIT"]})
        assert policy.allowed_licenses == ("MIT",)
        assert policy.allowed_bases is None
        assert policy.allowed_indexes is None
        assert policy.allowed_registries is None
        assert policy.allowed_packages is None

    def test_an_allowlist_written_empty_denies_everything_on_it(self) -> None:
        """A real, if probably unintended, policy — not read the same as unset."""
        policy = environments_policy_from_rules({"allowedLicenses": []})
        assert policy.allowed_licenses == ()
        assert policy.allowed_licenses is not None
        assert policy.allows("licenses", "MIT") is False

    def test_the_scan_threshold_narrows_the_platform_default(self) -> None:
        policy = environments_policy_from_rules(
            {"scan": {"blocksAt": "HIGH", "onlyFixable": False, "allowed": ["CVE-1"]}}
        )
        assert policy.scan.blocks_at == "HIGH"
        assert policy.scan.only_fixable is False
        assert policy.scan.allowed == ("CVE-1",)

    def test_no_scan_section_keeps_the_platform_default(self) -> None:
        policy = environments_policy_from_rules({"allowedLicenses": ["MIT"]})
        assert policy.scan == DEFAULT_POLICY

    def test_an_unrecognized_scan_threshold_falls_back_to_the_default(self) -> None:
        """A stored value outside `SEVERITIES` is not trusted blindly — the
        platform default is the honest reading of a value it cannot place."""
        policy = environments_policy_from_rules({"scan": {"blocksAt": "NOT-A-SEVERITY"}})
        assert policy.scan.blocks_at == DEFAULT_POLICY.blocks_at

    def test_the_version_is_carried_through_for_the_decision_record(self) -> None:
        policy = environments_policy_from_rules({"allowedLicenses": ["MIT"]}, version=7)
        assert policy.version == 7


class TestAllows:
    """`EnvironmentsPolicy.allows`: the one question every check asks it."""

    def test_unrestricted_allows_anything(self) -> None:
        assert UNRESTRICTED_POLICY.allows("bases", "anything") is True
        assert UNRESTRICTED_POLICY.allows("licenses", "GPL-3.0") is True

    def test_a_value_on_the_list_is_allowed(self) -> None:
        policy = EnvironmentsPolicy(allowed_bases=("datalayer/python-cpu",))
        assert policy.allows("bases", "datalayer/python-cpu") is True

    def test_a_value_off_the_list_is_not(self) -> None:
        policy = EnvironmentsPolicy(allowed_bases=("datalayer/python-cpu",))
        assert policy.allows("bases", "datalayer/python-cuda") is False

    def test_the_comparison_is_exact_not_a_prefix_or_a_host(self) -> None:
        """An index is compared the way `image_registry_allowed` already
        compares a registry — the whole value, never a substring of it."""
        policy = EnvironmentsPolicy(allowed_indexes=("https://pypi.org/simple",))
        assert policy.allows("indexes", "https://pypi.org/simple/extra") is False
        assert policy.allows("indexes", "pypi.org") is False


class TestRefuseUnlicensed:
    """`refuse_unlicensed`: E3-06's own licence check, naming the package."""

    def test_unrestricted_refuses_nothing(self) -> None:
        refuse_unlicensed([("gpl-lib", "GPL-3.0")], UNRESTRICTED_POLICY)

    def test_the_default_argument_is_unrestricted_too(self) -> None:
        """A caller passing no policy at all gets today's behaviour: nothing
        about licences was ever refused before this box, and a bare call
        must not start refusing by accident."""
        refuse_unlicensed([("gpl-lib", "GPL-3.0")])

    def test_a_licence_off_the_allowlist_is_refused_naming_the_package(self) -> None:
        policy = EnvironmentsPolicy(allowed_licenses=("MIT",))
        with pytest.raises(EnvironmentsError) as raised:
            refuse_unlicensed([("gpl-lib", "GPL-3.0")], policy)
        assert raised.value.code.code == "DL_ENV_POLICY_DENIED"
        assert "gpl-lib" in raised.value.message
        assert "GPL-3.0" in raised.value.message
        assert raised.value.detail["package"] == "gpl-lib"
        assert raised.value.detail["license"] == "GPL-3.0"

    def test_every_licence_on_the_allowlist_passes(self) -> None:
        policy = EnvironmentsPolicy(allowed_licenses=("MIT", "Apache-2.0"))
        refuse_unlicensed([("gdal", "MIT"), ("requests", "Apache-2.0")], policy)

    def test_the_first_offender_in_sbom_order_is_the_one_named(self) -> None:
        """Deterministic for the same artifact: the SBOM's own order, not
        sorted — a rebuild of the same lock names the same package first."""
        policy = EnvironmentsPolicy(allowed_licenses=("MIT",))
        with pytest.raises(EnvironmentsError) as raised:
            refuse_unlicensed([("first-bad", "GPL-3.0"), ("second-bad", "AGPL-3.0")], policy)
        assert raised.value.detail["package"] == "first-bad"

    def test_an_unnamed_package_is_still_refused_and_says_so(self) -> None:
        policy = EnvironmentsPolicy(allowed_licenses=("MIT",))
        with pytest.raises(EnvironmentsError) as raised:
            refuse_unlicensed([("", "GPL-3.0")], policy)
        assert "an unnamed package" in raised.value.message

    def test_the_policy_version_is_carried_into_the_refusal(self) -> None:
        policy = EnvironmentsPolicy(allowed_licenses=("MIT",), version=4)
        with pytest.raises(EnvironmentsError) as raised:
            refuse_unlicensed([("gpl-lib", "GPL-3.0")], policy)
        assert raised.value.detail["policyVersion"] == 4
