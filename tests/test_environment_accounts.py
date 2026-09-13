# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Whose provider account an artifact lives in (PLAN_ENV.md D-8, E2-01).

A managed build runs with the environment owner's provider secret, so its
artifact exists only in that account. A launch has to answer whether the
credentials in this caller's hands are the ones that can see it — a template
id from another team is not a missing template, it is somebody else's.

Nothing here asks a provider anything: a fingerprint is derived from the
credentials, so both sides of the comparison — the build in the worker and
the launch in Runtimes — work it out the same way with no network.
"""

from __future__ import annotations

import pytest

from code_sandboxes.environments.accounts import (
    account_fingerprints,
    fingerprint_matches,
    provider_account,
)

OWNER = {"DAYTONA_API_KEY": "dtn_owner_key", "DAYTONA_ORGANIZATION_ID": "org-42"}
STRANGER = {"DAYTONA_API_KEY": "dtn_other_key", "DAYTONA_ORGANIZATION_ID": "org-99"}


class TestWhatNamesAnAccount:
    def test_the_organization_names_a_daytona_account_and_the_key_stands_behind_it(self) -> None:
        found = account_fingerprints("daytona", OWNER)
        assert found[0] == "daytona:org-42"
        assert found[1].startswith("daytona:cred-")
        # The digest is not the key: nothing here can be turned back into it.
        assert "dtn_owner_key" not in " ".join(found)

    def test_the_workspace_names_a_modal_account(self) -> None:
        found = account_fingerprints("modal", {"MODAL_TOKEN_ID": "ak-1", "MODAL_WORKSPACE": "acme"})
        assert found[0] == "modal:acme"

    def test_a_modal_profile_names_it_when_no_workspace_does(self) -> None:
        found = account_fingerprints("modal", {"MODAL_TOKEN_ID": "ak-1", "MODAL_PROFILE": "acme"})
        assert found[0] == "modal:acme"

    def test_an_e2b_account_is_its_keys_digest_unless_a_team_is_named(self) -> None:
        """E2B's API answers which team a key belongs to only to an access
        token, which a launch does not hold — so the key stands for the team
        until a deployment names it."""
        assert account_fingerprints("e2b", {"E2B_API_KEY": "e2b_abc"})[0].startswith("e2b:cred-")
        named = account_fingerprints("e2b", {"E2B_API_KEY": "e2b_abc", "E2B_TEAM_ID": "team-7"})
        assert named[0] == "e2b:team-7"

    def test_a_modal_token_digests_on_the_id_and_not_the_secret(self) -> None:
        """The half that identifies, not the half that authenticates."""
        one = account_fingerprints("modal", {"MODAL_TOKEN_ID": "ak-1", "MODAL_TOKEN_SECRET": "s1"})
        two = account_fingerprints("modal", {"MODAL_TOKEN_ID": "ak-1", "MODAL_TOKEN_SECRET": "s2"})
        assert one == two

    def test_secrets_that_open_no_account_of_this_variant_name_none(self) -> None:
        assert account_fingerprints("e2b", {"DAYTONA_API_KEY": "dtn"}) == ()
        assert account_fingerprints("daytona", {}) == ()
        assert account_fingerprints("daytona", None) == ()
        # An empty value is not a credential.
        assert account_fingerprints("e2b", {"E2B_API_KEY": "   "}) == ()

    def test_a_variant_with_no_account_rule_names_none(self) -> None:
        assert account_fingerprints("datalayer", {"DATALAYER_API_KEY": "x"}) == ()

    def test_the_one_an_artifact_records_is_the_strongest_name(self) -> None:
        assert provider_account("daytona", OWNER) == "daytona:org-42"
        assert provider_account("e2b", {"E2B_API_KEY": "k"}).startswith("e2b:cred-")
        assert provider_account("e2b", {}) == ""


class TestWhoMayLaunchIt:
    def test_the_owner_may(self) -> None:
        recorded = provider_account("daytona", OWNER)
        assert fingerprint_matches(recorded, OWNER) is True

    def test_another_account_may_not(self) -> None:
        recorded = provider_account("daytona", OWNER)
        assert fingerprint_matches(recorded, STRANGER) is False

    def test_a_rotated_key_in_the_same_organization_still_may(self) -> None:
        """The point of keeping several names: a key is rotated, an
        organization is not."""
        recorded = provider_account("daytona", OWNER)
        rotated = {"DAYTONA_API_KEY": "dtn_rotated", "DAYTONA_ORGANIZATION_ID": "org-42"}
        assert fingerprint_matches(recorded, rotated) is True

    def test_an_artifact_recorded_by_its_key_still_matches_once_the_organization_is_named(
        self,
    ) -> None:
        """An older build recorded the digest; the caller now names the org too."""
        recorded = provider_account("daytona", {"DAYTONA_API_KEY": "dtn_owner_key"})
        assert recorded.startswith("daytona:cred-")
        assert fingerprint_matches(recorded, OWNER) is True

    def test_nothing_recorded_is_not_a_match(self) -> None:
        """Conservative on purpose: a refusal is a retry, a wrong pass is a
        sandbox in an account nobody chose."""
        assert fingerprint_matches("", OWNER) is False
        assert fingerprint_matches("   ", OWNER) is False

    def test_nothing_held_is_not_a_match(self) -> None:
        recorded = provider_account("daytona", OWNER)
        assert fingerprint_matches(recorded, {}) is False
        assert fingerprint_matches(recorded, None) is False
        assert fingerprint_matches(recorded, []) is False

    def test_a_fingerprint_of_another_variant_is_not_a_match(self) -> None:
        assert fingerprint_matches("e2b:team-7", {"E2B_TEAM_ID": "team-7"}) is True
        assert fingerprint_matches("modal:team-7", {"E2B_TEAM_ID": "team-7"}) is False

    def test_the_names_can_be_passed_instead_of_the_secrets(self) -> None:
        """Runtimes works the caller's names out once and compares many
        artifacts against them."""
        held = account_fingerprints("daytona", OWNER)
        assert fingerprint_matches("daytona:org-42", held) is True
        assert fingerprint_matches("daytona:org-99", held) is False


@pytest.mark.parametrize("variant", ["e2b", "daytona", "modal"])
def test_every_managed_variant_has_a_rule(variant: str) -> None:
    """A managed variant with no rule would record no account, and every
    launch of its artifacts would be refused."""
    from code_sandboxes.environments.accounts import CREDENTIAL_VARIABLES

    assert CREDENTIAL_VARIABLES.get(variant), variant
