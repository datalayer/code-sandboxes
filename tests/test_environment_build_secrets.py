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
)
from code_sandboxes.environments.errors import BUILD_SECRET_UNAVAILABLE, EnvironmentsError
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


def test_the_environment_variables_are_read_when_no_argument_is_given(monkeypatch) -> None:
    monkeypatch.setenv(IAM_API_KEY_VARIABLE, "from-env-key")
    monkeypatch.setenv(IAM_URL_VARIABLE, "https://iam.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "from-env-key"
        return httpx.Response(200, json={"value": "tok-1"})

    value = resolve_build_secret(SECRET, owner_uid=OWNER, transport=httpx.MockTransport(handler))
    assert value == "tok-1"
