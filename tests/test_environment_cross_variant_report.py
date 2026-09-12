# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""One dependency set everywhere (PLAN_ENV.md E2-08, Appendix B check 5).

Every variant of a version resolves from the *same* lock (section 5), so
Appendix B check 5 — comparing the imported version of every top-level
package with the lock — should agree on every variant that ran it. When it
does not, something about a provider's own base or layer let a different
transitive version in despite the shared lock, and that is worth catching
and naming across variants, not only within the one sandbox that happened to
disagree with itself.

Nothing here starts a sandbox: `package_versions_of` reads what
`conformance.run_core_tier` already recorded from check 5's own
`ValidationResult`, and `drifted_from`/`cross_variant_packages` are pure
functions over what the version's own running record already holds.
"""

from __future__ import annotations

from code_sandboxes.environments.builders import CheckResult, ValidationResult
from code_sandboxes.environments.conformance import (
    cross_variant_packages,
    drifted_from,
    package_versions_of,
)

CONTRACT = "sandbox-contract/v1"


def a_validation(packages: dict) -> ValidationResult:
    """A `ValidationResult` carrying only check 5's own answer, as `_imports` records it."""
    return ValidationResult(
        contract_version=CONTRACT,
        checks=[
            CheckResult(id="conformance:1", name="doctor", passed=True),
            CheckResult(
                id="conformance:5", name="imports", passed=True, data={"packages": packages}
            ),
        ],
    )


def entry(
    version: str | None, *, imported: str | None = "geopandas", error: str | None = None
) -> dict:
    return {"version": version, "imported": imported, "error": error}


class TestPackageVersionsOf:
    def test_it_reads_check_5s_own_versions(self) -> None:
        validation = a_validation({"geopandas": entry("1.1.1"), "rasterio": entry("1.4.3")})
        assert package_versions_of(validation) == {"geopandas": "1.1.1", "rasterio": "1.4.3"}

    def test_a_package_check_5_could_not_resolve_a_version_for_is_left_out(self) -> None:
        """Not installed at all is check 5's own failure, in its own sandbox —
        there is no version here to compare against another variant's."""
        validation = a_validation({"geopandas": entry(None, error="not installed")})
        assert package_versions_of(validation) == {}

    def test_no_check_5_at_all_answers_nothing_rather_than_raising(self) -> None:
        validation = ValidationResult(contract_version=CONTRACT, checks=[])
        assert package_versions_of(validation) == {}


class TestDriftedFrom:
    def test_two_variants_agreeing_drift_nowhere(self) -> None:
        stored = {"e2b": {"geopandas": "1.1.1", "rasterio": "1.4.3"}}
        assert drifted_from(stored, "modal", {"geopandas": "1.1.1", "rasterio": "1.4.3"}) is None

    def test_a_disagreeing_package_names_itself_and_both_versions(self) -> None:
        stored = {"e2b": {"geopandas": "1.1.1"}}
        found = drifted_from(stored, "modal", {"geopandas": "1.1.2"})
        assert found == ("geopandas", "1.1.2", "1.1.1")

    def test_it_never_compares_a_variant_against_its_own_earlier_record(self) -> None:
        """A backfill re-running the same variant is not a disagreement with itself."""
        stored = {"modal": {"geopandas": "1.1.2"}}
        assert drifted_from(stored, "modal", {"geopandas": "1.1.1"}) is None

    def test_no_other_variant_has_built_yet_drifts_nowhere(self) -> None:
        assert drifted_from({}, "datalayer", {"geopandas": "1.1.1"}) is None

    def test_a_package_absent_from_the_other_side_is_not_a_disagreement(self) -> None:
        """A package one variant's lock resolved but never imported (an
        optional extra) is not a drift; only two recorded versions differing is."""
        stored = {"e2b": {"rasterio": "1.4.3"}}
        assert drifted_from(stored, "modal", {"geopandas": "1.1.1"}) is None

    def test_three_variants_the_third_disagrees_with_the_first_not_the_second(self) -> None:
        stored = {"datalayer": {"geopandas": "1.1.1"}, "e2b": {"geopandas": "1.1.1"}}
        found = drifted_from(stored, "modal", {"geopandas": "1.1.2"})
        assert found is not None and found[0] == "geopandas"


class TestCrossVariantPackages:
    def test_it_folds_this_variants_versions_in(self) -> None:
        stored = {"datalayer": {"geopandas": "1.1.1"}}
        merged = cross_variant_packages(stored, "e2b", {"geopandas": "1.1.1", "rasterio": "1.4.3"})
        assert merged == {
            "datalayer": {"geopandas": "1.1.1"},
            "e2b": {"geopandas": "1.1.1", "rasterio": "1.4.3"},
        }

    def test_the_same_variant_built_again_replaces_its_own_entry_not_adds_one(self) -> None:
        stored = {"e2b": {"geopandas": "1.1.1"}}
        merged = cross_variant_packages(stored, "e2b", {"geopandas": "1.1.2"})
        assert merged == {"e2b": {"geopandas": "1.1.2"}}

    def test_the_report_of_four_variants_that_agree_is_identical_however_it_was_built_up(
        self,
    ) -> None:
        """The example's four variants, built one at a time in either order,
        end on the same stored report — the `Done when` this item asks for."""
        versions = {"geopandas": "1.1.1", "rasterio": "1.4.3"}
        forward: dict = {}
        for variant in ("datalayer", "e2b", "daytona", "modal"):
            forward = cross_variant_packages(forward, variant, versions)
        backward: dict = {}
        for variant in ("modal", "daytona", "e2b", "datalayer"):
            backward = cross_variant_packages(backward, variant, versions)
        assert forward == backward
        assert set(forward) == {"datalayer", "e2b", "daytona", "modal"}

    def test_it_does_not_mutate_what_was_stored(self) -> None:
        stored = {"datalayer": {"geopandas": "1.1.1"}}
        cross_variant_packages(stored, "e2b", {"geopandas": "1.1.1"})
        assert stored == {"datalayer": {"geopandas": "1.1.1"}}


class TestTheDriftedModalExample:
    """The scenario the item's `Done when` names directly."""

    def test_a_deliberately_drifted_modal_artifact_is_found_against_the_others(self) -> None:
        stored = {
            "datalayer": {"geopandas": "1.1.1", "rasterio": "1.4.3"},
            "e2b": {"geopandas": "1.1.1", "rasterio": "1.4.3"},
            "daytona": {"geopandas": "1.1.1", "rasterio": "1.4.3"},
        }
        modal_validation = a_validation(
            {"geopandas": entry("1.1.2"), "rasterio": entry("1.4.3")}  # drifted transitively
        )
        modal_versions = package_versions_of(modal_validation)
        found = drifted_from(stored, "modal", modal_versions)
        assert found == ("geopandas", "1.1.2", "1.1.1")

        merged = cross_variant_packages(stored, "modal", modal_versions)
        assert merged["modal"]["geopandas"] == "1.1.2"
        assert all(
            merged[variant]["geopandas"] == "1.1.1" for variant in ("datalayer", "e2b", "daytona")
        )
