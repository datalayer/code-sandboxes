# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The provider registry, and the secrets it answers about.

A service asks this registry two questions about somebody else's account:
which providers that account can use, and what it may launch on them. Both
have to be answered from the SAME credentials — the ones passed in, never the
ones the service itself happens to hold in its environment.
"""

from __future__ import annotations

from code_sandboxes.base import Sandbox
from code_sandboxes.models import SandboxEnvironment, SandboxVariant
from code_sandboxes.providers import (
    ProviderRequirement,
    SandboxProvider,
    available_providers,
    get_provider,
    provider_catalog,
)


def _environment(name: str) -> SandboxEnvironment:
    return SandboxEnvironment(name=name, title=name)


def test_the_listing_is_given_the_credentials_it_takes_as_arguments():
    asked: list[dict] = []

    provider = SandboxProvider(
        variant=SandboxVariant.DATALAYER,
        title="Test",
        description="",
        requirements=(ProviderRequirement(env_vars=("TEST_TOKEN",)),),
        environment_secrets=(("token", "TEST_TOKEN"),),
        list_environments=lambda **kwargs: asked.append(kwargs) or [_environment("env")],
    )

    assert provider.environments({"TEST_TOKEN": "abc"}) == [_environment("env")]
    assert asked == [{"token": "abc"}]


def test_a_credential_that_is_not_there_is_not_passed_as_nothing():
    """Omitting the argument lets the SDK fall back; passing None does not."""
    asked: list[dict] = []
    provider = SandboxProvider(
        variant=SandboxVariant.DATALAYER,
        title="Test",
        description="",
        environment_secrets=(("token", "TEST_TOKEN"), ("run_url", "TEST_URL")),
        list_environments=lambda **kwargs: asked.append(kwargs) or [],
    )

    provider.environments({"TEST_TOKEN": "abc"})

    assert asked == [{"token": "abc"}]


def test_a_provider_that_cannot_be_reached_ships_nothing_rather_than_raising():
    def explode(**_kwargs):
        raise RuntimeError("the platform is down")

    provider = SandboxProvider(
        variant=SandboxVariant.DATALAYER,
        title="Test",
        description="",
        list_environments=explode,
    )

    assert provider.environments() == []


def test_datalayer_declares_the_credentials_its_listing_takes():
    """It asks the platform what the ACCOUNT may launch, so it needs one."""
    provider = get_provider("datalayer")

    assert provider.environment_secrets == (
        ("token", "DATALAYER_TOKEN"),
        ("run_url", "DATALAYER_RUN_URL"),
    )


def test_the_catalog_answers_both_questions_from_the_same_secrets(monkeypatch):
    """Enabled and its environments, from the account — not the process.

    A catalog that read `enabled` from the secrets passed in and the
    environments from the environment of the service reported every Datalayer
    account as enabled with nothing to launch.
    """
    seen: list[dict] = []

    def fake_list_environments(cls, variant=SandboxVariant.DATALAYER, **kwargs):
        seen.append({"variant": str(variant), **kwargs})
        return [_environment("ai-agents-env")]

    monkeypatch.setattr(Sandbox, "list_environments", classmethod(fake_list_environments))

    catalog = provider_catalog({"DATALAYER_TOKEN": "account-token"})
    datalayer = next(entry for entry in catalog if entry["name"] == "datalayer")

    assert datalayer["enabled"]
    assert [env["name"] for env in datalayer["environments"]] == ["ai-agents-env"]
    assert {"variant": str(SandboxVariant.DATALAYER), "token": "account-token"} in seen


def test_a_provider_that_is_not_enabled_is_never_asked(monkeypatch):
    """Asking an unusable provider what it ships is a call that fails."""

    def fail_if_called(cls, variant=None, **kwargs):
        raise AssertionError(f"{variant} was asked for environments while disabled")

    monkeypatch.setattr(Sandbox, "list_environments", classmethod(fail_if_called))

    catalog = provider_catalog({})
    datalayer = next(entry for entry in catalog if entry["name"] == "datalayer")

    assert not datalayer["enabled"]
    assert datalayer["environments"] == []


def test_availability_is_read_from_the_secrets_given_not_from_the_process():
    names = {provider.name for provider in available_providers({"DATALAYER_TOKEN": "t"})}

    assert "datalayer" in names
    # Nothing was read from os.environ: a provider with no requirements is
    # available anywhere, and the credentialed ones are not.
    assert "eval" in names
    assert "kaggle" not in names
