# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Whose provider account an artifact lives in (PLAN_ENV.md, D-8, E2-01).

A managed-variant build runs with the **environment owner's** provider
secret — their E2B team, their Daytona organization, their Modal workspace —
and the artifact it produces exists only there. So a launch has to answer a
question the artifact's reference cannot: *are the credentials in this
caller's hands the ones that can see it?* A template id from another team is
not a missing template, it is somebody else's; launching it would either fail
obscurely or, worse, start something unrelated that happens to share a name.

This module answers it without holding a secret and without asking a provider
anything:

- **A fingerprint is derived from the credentials themselves.** The account's
  own identifier when the caller's configuration names one — Daytona's
  organization, Modal's workspace — and otherwise a digest of the credential
  that opens the account. A digest identifies the account exactly (the same
  key always answers the same thing) and reveals nothing.
- **An account answers to several names**, so both sides compute the whole set
  and a match on any one is a match. That is what lets an artifact recorded
  before the organization was configured still launch after it is.
- **No provider call.** Asking E2B who a key belongs to would make a launch
  wait on a third party to decide whether it may proceed, and would need a
  credential scope the launch path does not otherwise need.

The comparison is deliberately conservative: **no fingerprint on either side
is not a match.** An artifact recorded by an older build, or a caller with no
provider secret at all, is refused with `provider_account_mismatch` rather
than let through — the failure of a refusal is a launch somebody has to
retry, and the failure of a pass is a sandbox running in an account nobody
chose.

@module code_sandboxes.environments.accounts
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

__all__ = [
    "ACCOUNT_VARIABLES",
    "CREDENTIAL_VARIABLES",
    "account_fingerprints",
    "fingerprint_matches",
    "provider_account",
]

#: What names an account, per variant, when the caller's configuration says
#: so. Read from the same variables the SDKs read, so a deployment that can
#: reach a provider can also name its account there — nothing extra to set.
ACCOUNT_VARIABLES: dict[str, tuple[str, ...]] = {
    # Daytona takes the organization explicitly, and a JWT credential is only
    # usable with it.
    "daytona": ("DAYTONA_ORGANIZATION_ID",),
    # Modal names the workspace in its profile; `MODAL_WORKSPACE` is what a
    # deployment sets when it holds several.
    "modal": ("MODAL_WORKSPACE", "MODAL_PROFILE"),
    # E2B's API key belongs to a team, and its API answers which only to an
    # access token — which a launch does not hold. A deployment that wants the
    # team in the record sets it; otherwise the credential's digest stands for
    # it.
    "e2b": ("E2B_TEAM_ID",),
}

#: The credential whose digest stands for the account when nothing names it.
#: The first variable present is used, so the pair of a token id and its
#: secret digests on the id — the half that identifies, not the half that
#: authenticates.
CREDENTIAL_VARIABLES: dict[str, tuple[str, ...]] = {
    "daytona": ("DAYTONA_API_KEY", "DAYTONA_JWT_TOKEN"),
    "modal": ("MODAL_TOKEN_ID",),
    "e2b": ("E2B_API_KEY",),
}


def _digest(value: str) -> str:
    """A short, stable, non-reversible name for a credential.

    Sixteen hex characters of sha256: enough that two accounts do not collide,
    short enough to read in a record, and nothing anybody can turn back into
    the key.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def account_fingerprints(variant: str, secrets: Mapping[str, str] | None) -> tuple[str, ...]:
    """Every name the account these secrets open answers to, strongest first.

    `("daytona:org-42", "daytona:cred-1f2e3d4c5b6a7980")` — the organization
    the caller named, and the digest of the key that opens it. A build records
    the first; a launch compares against all of them, so an artifact recorded
    before the organization was configured still launches after it is.

    Empty when the secrets open no account of this variant, which is a
    refusal and not a pass: see the module docstring.
    """
    name = str(variant or "").strip().lower()
    held = {
        str(key): str(value) for key, value in (secrets or {}).items() if str(value or "").strip()
    }
    if not held:
        return ()
    found: list[str] = []
    for variable in ACCOUNT_VARIABLES.get(name, ()):
        value = held.get(variable, "").strip()
        if value:
            found.append(f"{name}:{value}")
            break
    for variable in CREDENTIAL_VARIABLES.get(name, ()):
        value = held.get(variable, "").strip()
        if value:
            found.append(f"{name}:cred-{_digest(value)}")
            break
    return tuple(found)


def provider_account(variant: str, secrets: Mapping[str, str] | None) -> str:
    """The one fingerprint an artifact records: the strongest name available."""
    found = account_fingerprints(variant, secrets)
    return found[0] if found else ""


def fingerprint_matches(recorded: str, held: Sequence[str] | Mapping[str, str] | None) -> bool:
    """Whether an artifact recorded for `recorded` may be launched by `held`.

    `held` is what :func:`account_fingerprints` answered for the caller, or
    the caller's secrets to work it out from. Nothing recorded, or nothing
    held, is **not** a match.
    """
    wanted = str(recorded or "").strip()
    if not wanted:
        return False
    if isinstance(held, Mapping):
        variant = wanted.split(":", 1)[0]
        names: Sequence[str] = account_fingerprints(variant, held)
    else:
        names = list(held or ())
    return wanted in {str(name).strip() for name in names if str(name).strip()}
