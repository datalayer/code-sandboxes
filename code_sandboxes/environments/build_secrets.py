# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Resolving one build secret's value from IAM, for the one build step that names it
(PLAN_ENV.md, E3-05).

A :class:`~code_sandboxes.environments.spec.BuildSecret` in an Environment's
spec names an IAM secret by id (``dlsec_...``), never a value: `BuildRequest`
itself carries only the id (`environments/builders.py`'s own docstring —
"Everything one variant's build of one version needs, **and nothing
secret**"). The value is fetched here, by the builder, at the moment the
build step that names it runs — never earlier, never written to a step
result, and never logged.

IAM's ``GET /secrets/{id}/value`` is gated on a service's own key
(``X-API-Key``), the same way ``datalayer_runtimes.authn.require_service_scope``
gates Runtimes' ``/internal/...`` routes for durable: this worker holds
``DATALAYER_DURABLE_IAM_API_KEY``, IAM's own scope is ``build-secrets:read``,
and nothing else may call that route. See ``iam/datalayer_iam/authn.py`` and
its ``BUILD_SECRETS_READ`` scope for the other side of this same contract.

@module code_sandboxes.environments.build_secrets
"""

from __future__ import annotations

import os
from typing import Any

from .errors import BUILD_SECRET_UNAVAILABLE, EnvironmentsError
from .spec import BuildSecret

__all__ = [
    "IAM_API_KEY_VARIABLE",
    "IAM_URL_VARIABLE",
    "resolve_build_secret",
]

#: Where the durable worker's own key to IAM comes from — the same naming
#: convention `datalayer_runtimes.authn.SERVICE_KEY_VARIABLES` uses: holder
#: first, audience second.
IAM_API_KEY_VARIABLE = "DATALAYER_DURABLE_IAM_API_KEY"

#: IAM's own base URL, the way every other internal caller in this worker
#: reads its peers' addresses from the environment.
IAM_URL_VARIABLE = "DATALAYER_IAM_URL"


def resolve_build_secret(
    secret: BuildSecret,
    *,
    owner_uid: str,
    iam_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
    transport: Any = None,
) -> str:
    """The raw value IAM holds for ``secret.id``, owned by ``owner_uid``.

    Raises ``DL_ENV_BUILD_SECRET_UNAVAILABLE`` — retryable, since the usual
    cause is IAM being briefly unreachable — when IAM cannot be reached, when
    the worker holds no key of its own, or when IAM refuses the request
    (unknown id, wrong owner, or a build secret feature not yet configured on
    this deployment). A build secret named in a spec is never silently
    dropped: a caller that gets this exception is expected to fail the build
    step, not build on without it.
    """
    key = api_key if api_key is not None else os.environ.get(IAM_API_KEY_VARIABLE, "")
    if not key:
        # A configuration problem, not a transient one: retrying with the
        # same unset variable answers the same refusal.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"No {IAM_API_KEY_VARIABLE} to ask IAM for {secret.id}'s value with",
            detail={"id": secret.id, "missing": IAM_API_KEY_VARIABLE},
            retryable=False,
        )
    base = (iam_url if iam_url is not None else os.environ.get(IAM_URL_VARIABLE, "")).rstrip("/")
    if not base:
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"No {IAM_URL_VARIABLE} to ask for {secret.id}'s value from",
            detail={"id": secret.id, "missing": IAM_URL_VARIABLE},
            retryable=False,
        )
    try:
        import httpx
    except ImportError as error:
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            "No HTTP client to resolve a build secret with: install "
            "`code-sandboxes[environments-builder]`",
            detail={"id": secret.id, "missing": "httpx"},
            retryable=False,
        ) from error
    url = f"{base}/api/iam/v1/secrets/{secret.id}/value"
    try:
        with httpx.Client(transport=transport, timeout=timeout) as client:
            response = client.get(
                url,
                params={"owner_uid": owner_uid},
                headers={"X-API-Key": key},
            )
    except httpx.HTTPError as error:
        # A network failure: the usual reason IAM cannot be reached is that
        # it is briefly unreachable, so this is worth retrying.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"IAM could not be reached to resolve {secret.id}: {error}",
            detail={"id": secret.id},
            retryable=True,
        ) from error
    if response.status_code == 404:
        # This owner genuinely has no such secret: retrying answers the same.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"{secret.id} does not exist for this owner",
            detail={"id": secret.id, "status": 404},
            retryable=False,
        )
    if response.status_code != 200:
        # Everything else — 5xx, a rate limit, a gateway timeout — is IAM's
        # own trouble, which a retry may have cleared by the time it runs.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"IAM refused to resolve {secret.id}: {response.status_code}",
            detail={"id": secret.id, "status": response.status_code},
            retryable=True,
        )
    try:
        body = response.json()
    except ValueError as error:
        # Malformed JSON is IAM answering incoherently, not this secret's
        # own state — the same class of trouble a 5xx is.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"IAM answered {secret.id} with a body that is not valid JSON",
            detail={"id": secret.id},
            retryable=True,
        ) from error
    value = body.get("value") if isinstance(body, dict) else None
    if not isinstance(value, str) or not value:
        # The response parsed, but not into a secret's value: retrying an
        # unchanged answer would not help.
        raise EnvironmentsError(
            BUILD_SECRET_UNAVAILABLE,
            f"IAM answered {secret.id} with no usable value",
            detail={"id": secret.id},
            retryable=False,
        )
    return value
