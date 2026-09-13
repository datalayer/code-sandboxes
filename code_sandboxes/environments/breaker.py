# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""One provider having a bad day stops being every user's bad day (PLAN_ENV.md §13, E2-10).

When a managed provider starts refusing everything — an outage, an expired
organization quota, a queue nobody is draining — the platform's default
behaviour is the worst one available: it keeps sending builds, and each user
learns about it separately, minutes later, from a message about their own
build. Ten users each get a mysterious failure, the provider gets ten more
requests it cannot serve, and nobody is told the one true thing, which is that
this provider is not building today.

A breaker per provider replaces that with one answer: **optional-variant
builds stop, and the provider reports as degraded.** Required variants are
never stopped — a version cannot become ready without them, so refusing them
would hide the failure rather than contain it, and `datalayer` is the
platform's own and has no third party to be degraded by.

**The state is derived, never stored.** It is counted from the build records
in a rolling window, so every Runtimes replica answers the same thing, a
restart forgets nothing, and there is no cache to go stale in one pod and not
another. That also makes the whole decision a pure function of four numbers,
which is what this module is.

Two arms open it, because the two failures look nothing alike:

- **The error rate**, over a window, once enough builds have been attempted
  for a rate to mean anything. It closes when the newest counted failure is
  older than the cool-down — which is the "half-open" state of a textbook
  breaker: builds are accepted again, and if the provider is still broken the
  next failures re-open it, without anybody scheduling a probe.
- **The queue depth**, which is what a provider that accepts work and never
  finishes it looks like. It closes as the queue drains.

**What counts against a provider** is only what the provider is responsible
for (section 10): an unmapped provider failure and a build that ran out of
time. A build whose `uv pip sync` exited 1 is the spec's fault, and counting
it would let one user's broken requirements close a provider for everybody.

@module code_sandboxes.environments.breaker
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .errors import BUILD_TIMEOUT, PROVIDER_ERROR, EnvironmentsError

__all__ = [
    "PROVIDER_FAILURE_CODES",
    "Health",
    "ProviderLoad",
    "Thresholds",
    "counts_against_provider",
    "decide",
    "load_of",
    "refuse_if_degraded",
]

#: The failures a provider is answerable for (section 10). A
#: `DL_ENV_BUILD_FAILED` is the spec's, and `DL_ENV_SCAN_BLOCKED`,
#: `DL_ENV_SMOKE_TEST_FAILED` and the resolve failures are Datalayer's own
#: decisions about the artifact: none of them says anything about the
#: provider.
PROVIDER_FAILURE_CODES: frozenset[str] = frozenset({PROVIDER_ERROR.code, BUILD_TIMEOUT.code})

#: The statuses of a build that is holding a place in a provider's queue.
OPEN_BUILD_STATUSES: frozenset[str] = frozenset({"queued", "running"})


@dataclass(frozen=True)
class Thresholds:
    """When a provider is treated as degraded, and for how long.

    The defaults are deliberately unexciting: half of at least four builds
    failing on the provider's own account, or eight builds waiting, is not a
    coincidence — and ten minutes is long enough to outlast a deploy and short
    enough that a provider that recovered is used again without anybody
    intervening.
    """

    #: How far back the rate is measured.
    window_seconds: int = 900
    #: Below this many attempts in the window, a rate is noise.
    attempts: int = 4
    error_rate: float = 0.5
    #: Builds waiting on this provider, across owners.
    queue_depth: int = 8
    #: How long a crossed rate keeps the door shut, from the newest failure.
    cool_down_seconds: int = 600

    def since(self, now: datetime) -> datetime:
        return now - timedelta(seconds=self.window_seconds)


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class ProviderLoad:
    """What one provider's recent builds add up to, as the registry counted them."""

    variant: str
    #: Builds of this variant that ended inside the window.
    attempts: int = 0
    #: How many of them ended in a failure the provider is answerable for.
    failures: int = 0
    #: Builds of this variant that have not ended, whenever they started.
    queue_depth: int = 0
    #: When the newest counted failure ended, which is what the cool-down runs from.
    newest_failure_at: datetime | None = None

    @property
    def error_rate(self) -> float:
        return (self.failures / self.attempts) if self.attempts else 0.0


@dataclass(frozen=True)
class Health:
    """Whether this provider takes optional builds right now, and why not."""

    variant: str
    degraded: bool = False
    #: What a person reads. Empty while the provider is healthy.
    reason: str = ""
    error_rate: float = 0.0
    attempts: int = 0
    queue_depth: int = 0
    #: How long to wait before asking again; 0 while it is healthy, and while
    #: the queue is what is wrong, since a queue drains at its own pace.
    retry_after_seconds: int = 0
    #: `error_rate`, `queue_depth`, or empty.
    arm: str = ""

    def as_body(self) -> dict[str, Any]:
        """The shape `/sandbox-providers` and a refusal's detail both carry."""
        body: dict[str, Any] = {"variant": self.variant, "degraded": self.degraded}
        if self.degraded:
            body.update(
                reason=self.reason,
                arm=self.arm,
                errorRate=round(self.error_rate, 3),
                attempts=self.attempts,
                queueDepth=self.queue_depth,
            )
            if self.retry_after_seconds:
                body["retryAfterSeconds"] = self.retry_after_seconds
        return body


def counts_against_provider(error_code: str | None) -> bool:
    """Whether a failure with this code says anything about the provider."""
    return str(error_code or "").strip() in PROVIDER_FAILURE_CODES


def decide(
    load: ProviderLoad, *, thresholds: Thresholds | None = None, now: datetime | None = None
) -> Health:
    """Whether this provider takes optional builds, from what its builds did.

    Pure: the same four numbers always answer the same thing, in Runtimes'
    route and in the durable worker alike.
    """
    limits = thresholds or DEFAULT_THRESHOLDS
    moment = now or datetime.now(timezone.utc)
    rate = load.error_rate
    common = {
        "variant": load.variant,
        "error_rate": rate,
        "attempts": load.attempts,
        "queue_depth": load.queue_depth,
    }
    if load.queue_depth >= limits.queue_depth:
        return Health(
            degraded=True,
            arm="queue_depth",
            reason=(
                f"{load.queue_depth} builds are waiting on {load.variant}, which is at or over "
                f"the limit of {limits.queue_depth}: it is accepting work faster than it "
                "finishes it"
            ),
            **common,
        )
    if load.attempts >= limits.attempts and rate >= limits.error_rate:
        waited = _seconds_since(load.newest_failure_at, moment)
        if waited is None or waited < limits.cool_down_seconds:
            left = limits.cool_down_seconds - (waited or 0)
            return Health(
                degraded=True,
                arm="error_rate",
                reason=(
                    f"{load.failures} of the last {load.attempts} {load.variant} builds failed on "
                    f"the provider's own account ({rate:.0%}), so {load.variant} is not being sent "
                    "more work for now"
                ),
                retry_after_seconds=max(1, int(left)),
                **common,
            )
    return Health(**common)


def _seconds_since(moment: datetime | None, now: datetime) -> float | None:
    if moment is None:
        return None
    when = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds())


def refuse_if_degraded(health: Health, *, variant: str | None = None) -> None:
    """Raise the refusal an optional build of a degraded provider gets.

    `DL_ENV_PROVIDER_ERROR`, which section 10 already calls retryable: the
    build was not refused for anything about the version, and the same request
    works once the provider does.
    """
    if not health.degraded:
        return
    raise EnvironmentsError(
        PROVIDER_ERROR,
        f"{variant or health.variant} is degraded and takes no optional build right now: "
        f"{health.reason}",
        detail={"degraded": health.as_body(), "variant": variant or health.variant},
    )


def load_of(
    variant: str,
    builds: Any,
    *,
    thresholds: Thresholds | None = None,
    now: datetime | None = None,
) -> ProviderLoad:
    """Count one provider's window from build records, however they are shaped.

    Reads the field names the registry stores (`variant`, `status`,
    `error_code`, `finished_at`) and the camel ones its routes answer, so the
    service and a test fixture can both be counted without converting first.
    """
    limits = thresholds or DEFAULT_THRESHOLDS
    moment = now or datetime.now(timezone.utc)
    since = limits.since(moment)
    name = str(variant or "").strip().lower()
    attempts = failures = depth = 0
    newest: datetime | None = None
    for record in builds or []:
        item: Mapping[str, Any] = getattr(record, "value", record)
        if str(_read(item, "variant") or "").strip().lower() != name:
            continue
        status = str(_read(item, "status") or "").strip()
        if status in OPEN_BUILD_STATUSES:
            depth += 1
            continue
        ended = _time(_read(item, "finished_at") or _read(item, "created_at"))
        if ended is None or ended < since:
            continue
        attempts += 1
        if status == "failed" and counts_against_provider(_read(item, "error_code")):
            failures += 1
            if newest is None or ended > newest:
                newest = ended
    return ProviderLoad(
        variant=name,
        attempts=attempts,
        failures=failures,
        queue_depth=depth,
        newest_failure_at=newest,
    )


def _read(item: Mapping[str, Any], name: str) -> Any:
    """A field by its stored name or the camel one a route answers."""
    if name in item:
        return item[name]
    parts = name.split("_")
    camel = parts[0] + "".join(part.title() for part in parts[1:])
    return item.get(camel)


def _time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
