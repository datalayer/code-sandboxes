# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The error taxonomy, checked against its table."""

from __future__ import annotations

import uuid

import pytest

from code_sandboxes.environments import errors
from code_sandboxes.environments.errors import (
    ERROR_CODES,
    EnvironmentsError,
    ProviderErrorRule,
    error_code,
    map_provider_error,
    provider_error_rules,
    register_provider_error,
)

#: PLAN_ENV.md §10: the code, and whether retrying can succeed (None: it depends).
TABLE = [
    ("DL_ENV_SPEC_INVALID", False),
    ("DL_ENV_CAPABILITY_UNSUPPORTED", False),
    ("DL_ENV_RESOLVE_CONFLICT", False),
    ("DL_ENV_PACKAGE_NOT_FOUND", None),
    ("DL_ENV_PROTECTED_PACKAGE", False),
    ("DL_ENV_BUILD_FAILED", False),
    ("DL_ENV_BUILD_TIMEOUT", True),
    ("DL_ENV_QUOTA_EXCEEDED", True),
    ("DL_ENV_POLICY_DENIED", False),
    ("DL_ENV_SCAN_BLOCKED", False),
    ("DL_ENV_SMOKE_TEST_FAILED", False),
    ("DL_ENV_ARTIFACT_MISSING", True),
    ("DL_ENV_PROVIDER_ERROR", True),
]


def test_the_taxonomy_is_exactly_the_table() -> None:
    assert list(ERROR_CODES) == [code for code, _ in TABLE]


@pytest.mark.parametrize(("code", "retryable"), TABLE)
def test_each_code_is_as_retryable_as_the_table_says(code: str, retryable: bool | None) -> None:
    assert error_code(code).retryable is retryable
    assert error_code(code).user_action


def test_an_unknown_code_names_the_codes_that_exist() -> None:
    with pytest.raises(KeyError, match="DL_ENV_SPEC_INVALID"):
        error_code("DL_ENV_NOPE")


def test_a_code_the_occurrence_decides_defaults_to_not_retrying() -> None:
    assert EnvironmentsError(errors.PACKAGE_NOT_FOUND, "numpyy").retryable is False
    assert (
        EnvironmentsError(errors.PACKAGE_NOT_FOUND, "index down", retryable=True).retryable is True
    )


def test_contradicting_what_a_code_decides_is_refused() -> None:
    with pytest.raises(ValueError, match="not retryable"):
        EnvironmentsError(errors.SPEC_INVALID, "bad", retryable=True)


def test_the_api_body_carries_the_code_the_message_and_the_correlation_id() -> None:
    error = EnvironmentsError(
        errors.QUOTA_EXCEEDED, "2 builds already running", detail={"limit": 2}
    )
    assert error.to_body("corr-1") == {
        "code": "DL_ENV_QUOTA_EXCEEDED",
        "message": "2 builds already running",
        "retryable": True,
        "userAction": "Wait or raise the quota",
        "detail": {"limit": 2},
        "correlationId": "corr-1",
    }
    assert "correlationId" not in error.to_body()
    assert str(error) == "DL_ENV_QUOTA_EXCEEDED: 2 builds already running"


class TemplateBuildFailedError(Exception):
    pass


def _variant() -> str:
    """A variant name no other test registers rules for."""
    return f"test-{uuid.uuid4().hex[:8]}"


def test_an_error_no_rule_claims_is_a_provider_error_marked_unmapped() -> None:
    mapped = map_provider_error(_variant(), RuntimeError("the provider fell over"))
    assert mapped.code is errors.PROVIDER_ERROR
    assert mapped.detail["unmapped"] is True
    assert mapped.detail["providerError"] == "the provider fell over"
    assert mapped.detail["providerErrorType"] == "RuntimeError"


def test_a_rule_claims_an_error_by_its_exception_name() -> None:
    variant = _variant()
    register_provider_error(
        variant,
        ProviderErrorRule(errors.BUILD_FAILED, exception_names=("TemplateBuildFailedError",)),
    )
    mapped = map_provider_error(variant, TemplateBuildFailedError("exit status 2"))
    assert mapped.code is errors.BUILD_FAILED
    assert "unmapped" not in mapped.detail


def test_a_rule_claims_an_error_by_its_message() -> None:
    variant = _variant()
    register_provider_error(
        variant,
        ProviderErrorRule(
            errors.PACKAGE_NOT_FOUND, message_pattern=r"no matching distribution", retryable=False
        ),
    )
    mapped = map_provider_error(variant, "ERROR: No matching distribution found for numpyy")
    assert mapped.code is errors.PACKAGE_NOT_FOUND
    assert mapped.retryable is False


def test_both_parts_of_a_rule_must_match() -> None:
    variant = _variant()
    register_provider_error(
        variant,
        ProviderErrorRule(
            errors.QUOTA_EXCEEDED,
            exception_names=("TemplateBuildFailedError",),
            message_pattern="quota",
        ),
    )
    assert (
        map_provider_error(variant, TemplateBuildFailedError("exit status 2")).code
        is errors.PROVIDER_ERROR
    )
    assert (
        map_provider_error(variant, TemplateBuildFailedError("team quota reached")).code
        is errors.QUOTA_EXCEEDED
    )


def test_a_rule_that_names_nothing_claims_nothing() -> None:
    assert not ProviderErrorRule(errors.BUILD_FAILED).matches(RuntimeError("anything"))


def test_a_timeout_is_a_build_timeout_on_every_variant() -> None:
    assert (
        map_provider_error(_variant(), TimeoutError("took too long")).code is errors.BUILD_TIMEOUT
    )


def test_a_variant_is_matched_however_it_is_spelled() -> None:
    variant = _variant()
    register_provider_error(
        variant.upper(), ProviderErrorRule(errors.SCAN_BLOCKED, message_pattern="CVE-")
    )
    assert map_provider_error(f"  {variant} ", "CVE-2026-0001").code is errors.SCAN_BLOCKED
    assert provider_error_rules(variant)[0].code is errors.SCAN_BLOCKED
