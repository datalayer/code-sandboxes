# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""What can go wrong in an Environment's life, in one vocabulary.

Every failure an Environment meets — a spec that does not validate, a
dependency set that cannot be satisfied, a provider that refuses a build — is
reported under one of the codes below, whichever provider it came from. A
caller acts on the code; the provider's own words travel in the detail.

Adapters normalize their provider's errors into these codes with
:func:`register_provider_error`. Anything no rule claims becomes
``DL_ENV_PROVIDER_ERROR`` with the raw message attached and ``unmapped`` set:
an error nobody mapped is how an adapter silently rots, and that flag is what
the alert is raised on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..models import normalize_variant

__all__ = [
    "ARTIFACT_MISSING",
    "BUILD_FAILED",
    "BUILD_TIMEOUT",
    "CAPABILITY_UNSUPPORTED",
    "ERROR_CODES",
    "PACKAGE_NOT_FOUND",
    "POLICY_DENIED",
    "PROTECTED_PACKAGE",
    "PROVIDER_ERROR",
    "QUOTA_EXCEEDED",
    "RESOLVE_CONFLICT",
    "SCAN_BLOCKED",
    "SMOKE_TEST_FAILED",
    "SPEC_INVALID",
    "EnvironmentsError",
    "ErrorCode",
    "ProviderErrorRule",
    "Retry",
    "error_code",
    "map_provider_error",
    "provider_error_rules",
    "register_provider_error",
]


class Retry(str, Enum):
    """Whether sending the same request again can succeed."""

    NO = "no"
    YES = "yes"
    #: Yes, once something outside the request has changed: a quota refilled.
    LATER = "later"
    #: The occurrence decides: a package index that is down, or a package
    #: that does not exist.
    SOMETIMES = "sometimes"


@dataclass(frozen=True)
class ErrorCode:
    """One code of the taxonomy: what it means and what the user does."""

    code: str
    meaning: str
    retry: Retry
    user_action: str

    @property
    def retryable(self) -> bool | None:
        """True or False when the code decides; None when the occurrence does."""
        if self.retry is Retry.SOMETIMES:
            return None
        return self.retry in (Retry.YES, Retry.LATER)


SPEC_INVALID = ErrorCode(
    "DL_ENV_SPEC_INVALID",
    "Schema or field validation failed",
    Retry.NO,
    "Fix the highlighted field",
)
CAPABILITY_UNSUPPORTED = ErrorCode(
    "DL_ENV_CAPABILITY_UNSUPPORTED",
    "A requested variant cannot honor the spec",
    Retry.NO,
    "Remove the instruction or drop the variant",
)
RESOLVE_CONFLICT = ErrorCode(
    "DL_ENV_RESOLVE_CONFLICT",
    "Dependency set unsatisfiable",
    Retry.NO,
    "Relax a pin; the conflicting pair is shown",
)
PACKAGE_NOT_FOUND = ErrorCode(
    "DL_ENV_PACKAGE_NOT_FOUND",
    "Package, version, or index unavailable",
    Retry.SOMETIMES,
    "Check the name, the version, or the index allowlist",
)
PROTECTED_PACKAGE = ErrorCode(
    "DL_ENV_PROTECTED_PACKAGE",
    "Spec pins a package Datalayer reserves",
    Retry.NO,
    "Remove the pin; the supported range is shown",
)
BUILD_FAILED = ErrorCode(
    "DL_ENV_BUILD_FAILED", "Build command exited non-zero", Retry.NO, "Read the build log"
)
BUILD_TIMEOUT = ErrorCode(
    "DL_ENV_BUILD_TIMEOUT", "Exceeded the per-variant time limit", Retry.YES, "Retry or simplify"
)
QUOTA_EXCEEDED = ErrorCode(
    "DL_ENV_QUOTA_EXCEEDED",
    "Concurrency, storage, or spend quota",
    Retry.LATER,
    "Wait or raise the quota",
)
POLICY_DENIED = ErrorCode(
    "DL_ENV_POLICY_DENIED",
    "Organization policy blocked a base, an index, or a license",
    Retry.NO,
    "Contact the organization admin",
)
SCAN_BLOCKED = ErrorCode(
    "DL_ENV_SCAN_BLOCKED",
    "Vulnerability or malware policy blocked the artifact",
    Retry.NO,
    "Upgrade the offending package",
)
SMOKE_TEST_FAILED = ErrorCode(
    "DL_ENV_SMOKE_TEST_FAILED",
    "Artifact violates the sandbox contract",
    Retry.NO,
    "The contract violation is named",
)
ARTIFACT_MISSING = ErrorCode(
    "DL_ENV_ARTIFACT_MISSING",
    "No artifact for the requested variant or region",
    Retry.YES,
    "Build it or pick another variant",
)
PROVIDER_ERROR = ErrorCode(
    "DL_ENV_PROVIDER_ERROR", "Unmapped provider failure", Retry.YES, "Retry; Datalayer is alerted"
)

#: Every code, by its name, in the order of the taxonomy.
ERROR_CODES: dict[str, ErrorCode] = {
    code.code: code
    for code in (
        SPEC_INVALID,
        CAPABILITY_UNSUPPORTED,
        RESOLVE_CONFLICT,
        PACKAGE_NOT_FOUND,
        PROTECTED_PACKAGE,
        BUILD_FAILED,
        BUILD_TIMEOUT,
        QUOTA_EXCEEDED,
        POLICY_DENIED,
        SCAN_BLOCKED,
        SMOKE_TEST_FAILED,
        ARTIFACT_MISSING,
        PROVIDER_ERROR,
    )
}


def error_code(code: str) -> ErrorCode:
    """The code of that name, or an error naming the ones that exist."""
    try:
        return ERROR_CODES[code]
    except KeyError:
        raise KeyError(f"no error code {code!r}; the codes are {', '.join(ERROR_CODES)}") from None


class EnvironmentsError(Exception):
    """A failure of an Environment's life, under one of :data:`ERROR_CODES`.

    ``retryable`` comes from the code, except for a code whose retry depends
    on the occurrence (``PACKAGE_NOT_FOUND``), where the caller says — and
    says ``False`` when it does not know, so nothing loops on a missing
    package. Saying the opposite of what the code decides is a bug, and is
    refused rather than quietly overridden.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        detail: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(f"{code.code}: {message}")
        decided = code.retryable
        if decided is not None and retryable is not None and retryable != decided:
            raise ValueError(f"{code.code} is {'' if decided else 'not '}retryable by definition")
        self.code = code
        self.message = message
        self.detail: dict[str, Any] = dict(detail or {})
        self.retryable: bool = decided if decided is not None else bool(retryable)

    def to_body(self, correlation_id: str | None = None) -> dict[str, Any]:
        """The error as an API body: code, human message, retryability, detail."""
        body: dict[str, Any] = {
            "code": self.code.code,
            "message": self.message,
            "retryable": self.retryable,
            "userAction": self.code.user_action,
            "detail": self.detail,
        }
        if correlation_id:
            body["correlationId"] = correlation_id
        return body


# --- Provider errors ---------------------------------------------------------


@dataclass(frozen=True)
class ProviderErrorRule:
    """How one kind of provider error maps into the taxonomy.

    A rule claims an error by the name of its exception class, by a pattern
    over its message, or by both; a rule with neither claims nothing.
    """

    code: ErrorCode
    exception_names: tuple[str, ...] = ()
    message_pattern: str | None = None
    #: For a code whose retry the occurrence decides.
    retryable: bool | None = None

    def matches(self, error: BaseException | str) -> bool:
        if not self.exception_names and not self.message_pattern:
            return False
        if self.exception_names:
            name = type(error).__name__ if isinstance(error, BaseException) else ""
            if name not in self.exception_names:
                return False
        if self.message_pattern and not re.search(
            self.message_pattern, str(error), flags=re.IGNORECASE
        ):
            return False
        return True


#: Rules every variant falls back on, after its own.
_DEFAULT_RULES: tuple[ProviderErrorRule, ...] = (
    ProviderErrorRule(BUILD_TIMEOUT, exception_names=("TimeoutError",)),
)

_RULES: dict[str, list[ProviderErrorRule]] = {}


def register_provider_error(variant: str, rule: ProviderErrorRule) -> None:
    """Teach the taxonomy one of a provider's errors."""
    _RULES.setdefault(normalize_variant(variant), []).append(rule)


def provider_error_rules(variant: str) -> tuple[ProviderErrorRule, ...]:
    """The rules a variant's errors are mapped by, its own first."""
    return (*_RULES.get(normalize_variant(variant), ()), *_DEFAULT_RULES)


def map_provider_error(variant: str, error: BaseException | str) -> EnvironmentsError:
    """A provider's error, under the code a rule gives it.

    What no rule claims is ``DL_ENV_PROVIDER_ERROR``, with the provider's
    message and exception type in the detail and ``unmapped`` set.
    """
    normalized = normalize_variant(variant)
    text = str(error) or (type(error).__name__ if isinstance(error, BaseException) else "")
    kind = type(error).__name__ if isinstance(error, BaseException) else None
    for rule in provider_error_rules(normalized):
        if rule.matches(error):
            return EnvironmentsError(
                rule.code,
                text,
                detail={"variant": normalized, "providerError": text, "providerErrorType": kind},
                retryable=rule.retryable if rule.code.retryable is None else None,
            )
    return EnvironmentsError(
        PROVIDER_ERROR,
        text,
        detail={
            "variant": normalized,
            "providerError": text,
            "providerErrorType": kind,
            "unmapped": True,
        },
    )
