# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Resolving a build secret's value from IAM's internal route (PLAN_ENV.md, E3-05).

No network beyond `httpx.MockTransport`: `resolve_build_secret` is what the
Datalayer builder calls to fetch one `BuildSecret`'s value at the moment the
step that names it runs, never earlier and never logged.
"""

from __future__ import annotations

import httpx
import pytest

from code_sandboxes.environments.build_secrets import (
    IAM_API_KEY_VARIABLE,
    IAM_URL_VARIABLE,
    resolve_build_secret,
    resolve_provider_credential,
)
from code_sandboxes.environments.errors import (
    BUILD_SECRET_UNAVAILABLE,
    CAPABILITY_UNSUPPORTED,
    EnvironmentsError,
)
from code_sandboxes.environments.spec import BuildSecret

SECRET = BuildSecret(id="dlsec_01J9BUILDSECRET0000000000", name="PIP_TOKEN")
OWNER = "01k0wner000000000000000000"


def test_it_calls_iams_internal_route_with_the_service_key() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"id": SECRET.id, "name": "PIP_TOKEN", "value": "tok-1"})

    value = resolve_build_secret(
        SECRET,
        owner_uid=OWNER,
        iam_url="https://iam.example.com",
        api_key="durable-to-iam-key",
        transport=httpx.MockTransport(handler),
    )

    assert value == "tok-1"
    assert len(calls) == 1
    request = calls[0]
    assert request.url.path == f"/api/iam/v1/secrets/{SECRET.id}/value"
    assert request.url.params["owner_uid"] == OWNER
    assert request.headers["X-API-Key"] == "durable-to-iam-key"


def test_a_trailing_slash_on_the_iam_url_is_tolerated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": "tok-1"})

    value = resolve_build_secret(
        SECRET,
        owner_uid=OWNER,
        iam_url="https://iam.example.com/",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    assert value == "tok-1"


def test_no_key_refuses_before_any_network_call() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no key to call IAM with: nothing should be sent")

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="",
            transport=httpx.MockTransport(unreachable),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    assert IAM_API_KEY_VARIABLE in raised.value.message
    # A configuration problem: retrying with the same unset key answers the
    # same refusal.
    assert raised.value.retryable is False


def test_no_iam_url_refuses_before_any_network_call() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no URL to call IAM at: nothing should be sent")

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="",
            api_key="k",
            transport=httpx.MockTransport(unreachable),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    assert IAM_URL_VARIABLE in raised.value.message
    assert raised.value.retryable is False


def test_a_404_is_build_secret_unavailable_not_a_bare_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    assert SECRET.id in raised.value.message
    # This owner genuinely has no such secret: retrying answers the same.
    assert raised.value.retryable is False


def test_iam_unreachable_is_build_secret_unavailable() -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(unreachable),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    # A network failure is the usual shape of "IAM is briefly unreachable".
    assert raised.value.retryable is True


def test_a_non_200_that_is_not_404_is_still_build_secret_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    # IAM's own trouble, which a retry may have cleared by the time it runs.
    assert raised.value.retryable is True


def test_malformed_json_is_build_secret_unavailable_and_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    assert raised.value.retryable is True


def test_a_non_object_json_body_is_build_secret_unavailable() -> None:
    """A `200` whose body parses but is not an object: `.get` must never be
    called on it directly, or a list or bare string crashes with `AttributeError`
    instead of raising the taxonomy's own code."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    assert raised.value.retryable is False


def test_an_empty_value_is_refused_rather_than_used() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": ""})

    with pytest.raises(EnvironmentsError) as raised:
        resolve_build_secret(
            SECRET,
            owner_uid=OWNER,
            iam_url="https://iam.example.com",
            api_key="k",
            transport=httpx.MockTransport(handler),
        )
    assert raised.value.code is BUILD_SECRET_UNAVAILABLE
    # The response parsed, but not into a value: retrying an unchanged
    # answer would not help.
    assert raised.value.retryable is False


def test_the_environment_variables_are_read_when_no_argument_is_given(monkeypatch) -> None:
    monkeypatch.setenv(IAM_API_KEY_VARIABLE, "from-env-key")
    monkeypatch.setenv(IAM_URL_VARIABLE, "https://iam.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "from-env-key"
        return httpx.Response(200, json={"value": "tok-1"})

    value = resolve_build_secret(SECRET, owner_uid=OWNER, transport=httpx.MockTransport(handler))
    assert value == "tok-1"


# -- the owner's own provider credential (E2-01, D-8) -------------------------------------------


def _credential_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_the_credential_is_asked_for_by_variant_and_owner() -> None:
    """A build knows whose it is and which variant it is building, never a secret id."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"provider": "daytona", "value": {"DAYTONA_API_KEY": "k"}},
        )

    secrets = resolve_provider_credential(
        "daytona",
        owner_uid="owner-1",
        iam_url="https://iam.example",
        api_key="worker-key",
        transport=_credential_transport(handler),
    )
    assert secrets == {"DAYTONA_API_KEY": "k"}
    (request,) = seen
    assert request.url.path == "/api/iam/v1/secrets/provider/daytona/value"
    assert request.url.params["owner_uid"] == "owner-1"
    # The worker's own key, never a person's token.
    assert request.headers["X-API-Key"] == "worker-key"
    assert "authorization" not in {name.lower() for name in request.headers}


def test_the_account_names_come_through_beside_the_key() -> None:
    """`environments/accounts.py` fingerprints the account from these, so a
    credential is more than the key that opens it."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"value": {"E2B_API_KEY": "k", "E2B_TEAM_ID": "team"}})

    assert resolve_provider_credential(
        "e2b",
        owner_uid="owner-1",
        iam_url="https://iam.example",
        api_key="k",
        transport=_credential_transport(handler),
    ) == {"E2B_API_KEY": "k", "E2B_TEAM_ID": "team"}


def test_an_owner_with_no_credential_is_refused_by_name_and_not_retried() -> None:
    """Never a fallback to whatever keys the worker holds: that is the one
    thing D-8 exists to prevent."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "No credential for that provider"})

    with pytest.raises(EnvironmentsError) as raised:
        resolve_provider_credential(
            "modal",
            owner_uid="owner-1",
            iam_url="https://iam.example",
            api_key="k",
            transport=_credential_transport(handler),
        )
    assert raised.value.code is CAPABILITY_UNSUPPORTED
    assert raised.value.retryable is False
    assert "their own account" in str(raised.value)


def test_iam_being_away_is_retryable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "away"})

    with pytest.raises(EnvironmentsError) as raised:
        resolve_provider_credential(
            "daytona",
            owner_uid="owner-1",
            iam_url="https://iam.example",
            api_key="k",
            transport=_credential_transport(handler),
        )
    assert raised.value.retryable is True


def test_datalayer_is_not_a_variant_a_credential_is_kept_for() -> None:
    """The platform's own builder uses the platform's own registry."""
    with pytest.raises(EnvironmentsError) as raised:
        resolve_provider_credential(
            "datalayer", owner_uid="owner-1", iam_url="https://iam.example", api_key="k"
        )
    assert raised.value.code is CAPABILITY_UNSUPPORTED


def test_a_credential_that_names_nothing_is_refused() -> None:
    for value in ({}, "a string", None, {"": "v"}):

        def handler(_request: httpx.Request, value=value) -> httpx.Response:
            return httpx.Response(200, json={"value": value})

        with pytest.raises(EnvironmentsError) as raised:
            resolve_provider_credential(
                "daytona",
                owner_uid="owner-1",
                iam_url="https://iam.example",
                api_key="k",
                transport=_credential_transport(handler),
            )
        assert raised.value.code is CAPABILITY_UNSUPPORTED, value
