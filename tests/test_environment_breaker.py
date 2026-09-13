# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A provider that cannot build stops taking optional builds (PLAN_ENV.md §13, E2-10).

What the breaker is for is not the failing build — it is the nine after it. A
provider in an outage would otherwise be sent every build that arrives, each
user would learn about the outage from their own failure minutes later, and
nobody would be told the one true thing.

Everything here is the decision, which is a pure function of four numbers
counted from the build records: attempts, failures the provider is answerable
for, queue depth, and when the newest of those failures ended. Nothing is
stored, so a Runtimes replica that just started answers what the others do.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from code_sandboxes.environments.breaker import (
    PROVIDER_FAILURE_CODES,
    Health,
    ProviderLoad,
    Thresholds,
    counts_against_provider,
    decide,
    load_of,
    refuse_if_degraded,
)
from code_sandboxes.environments.errors import ERROR_CODES, EnvironmentsError

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
LIMITS = Thresholds()


def at(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(
    variant: str = "e2b", status: str = "succeeded", *, code: str = "", ago: float = 1
) -> dict:
    return {
        "variant": variant,
        "status": status,
        "error_code": code,
        "finished_at": at(ago),
        "created_at": at(ago + 5),
    }


def failed(variant: str = "e2b", *, code: str = "DL_ENV_PROVIDER_ERROR", ago: float = 1) -> dict:
    return build(variant, "failed", code=code, ago=ago)


# -- what counts against a provider ---------------------------------------------


class TestWhatCounts:
    def test_the_two_codes_the_provider_is_answerable_for(self) -> None:
        assert counts_against_provider("DL_ENV_PROVIDER_ERROR") is True
        assert counts_against_provider("DL_ENV_BUILD_TIMEOUT") is True

    def test_a_spec_that_does_not_build_is_not_the_providers_fault(self) -> None:
        """The one that matters most: one user's broken requirements must not
        close a provider for everybody."""
        assert counts_against_provider("DL_ENV_BUILD_FAILED") is False
        assert counts_against_provider("DL_ENV_RESOLVE_CONFLICT") is False
        assert counts_against_provider("DL_ENV_SPEC_INVALID") is False

    def test_nor_is_a_decision_datalayer_made_about_the_artifact(self) -> None:
        assert counts_against_provider("DL_ENV_SCAN_BLOCKED") is False
        assert counts_against_provider("DL_ENV_SMOKE_TEST_FAILED") is False

    def test_nothing_and_nonsense_count_for_nothing(self) -> None:
        assert counts_against_provider(None) is False
        assert counts_against_provider("") is False
        assert counts_against_provider("DL_ENV_SOMETHING_ELSE") is False

    def test_every_code_that_counts_is_one_of_the_taxonomys(self) -> None:
        """A code outside section 10 would count nothing forever, silently."""
        assert PROVIDER_FAILURE_CODES <= set(ERROR_CODES)
        assert all(ERROR_CODES[code].retryable for code in PROVIDER_FAILURE_CODES)


# -- counting the window --------------------------------------------------------


class TestTheWindow:
    def test_it_counts_one_variants_builds_and_leaves_the_others_alone(self) -> None:
        builds = [failed("e2b"), failed("e2b", ago=3), build("e2b"), failed("modal")]
        load = load_of("e2b", builds, now=NOW)
        assert (load.attempts, load.failures) == (3, 2)
        assert load_of("modal", builds, now=NOW).failures == 1
        assert load_of("daytona", builds, now=NOW) == ProviderLoad(variant="daytona")

    def test_a_build_that_ended_before_the_window_is_history(self) -> None:
        old = failed("e2b", ago=LIMITS.window_seconds / 60 + 1)
        assert load_of("e2b", [old], now=NOW).attempts == 0

    def test_an_open_build_is_queue_depth_and_not_an_attempt(self) -> None:
        """A build that has not ended has failed nothing — and it is exactly
        what a provider that accepts work and never finishes it leaves behind."""
        waiting = [
            {"variant": "e2b", "status": "queued", "created_at": at(400)},
            {"variant": "e2b", "status": "running", "created_at": at(1)},
        ]
        load = load_of("e2b", waiting, now=NOW)
        assert (load.queue_depth, load.attempts, load.failures) == (2, 0, 0)

    def test_a_cancelled_build_says_nothing_about_the_provider(self) -> None:
        load = load_of("e2b", [build("e2b", "cancelled")], now=NOW)
        assert (load.attempts, load.failures) == (1, 0)

    def test_it_reads_the_camel_names_a_route_answers_too(self) -> None:
        answered = [
            {
                "variant": "e2b",
                "status": "failed",
                "errorCode": "DL_ENV_BUILD_TIMEOUT",
                "finishedAt": at(2),
            }
        ]
        assert load_of("e2b", answered, now=NOW).failures == 1

    def test_a_record_with_no_moment_is_not_counted_into_a_window(self) -> None:
        assert load_of("e2b", [{"variant": "e2b", "status": "failed"}], now=NOW).attempts == 0

    def test_the_newest_counted_failure_is_what_the_cool_down_runs_from(self) -> None:
        load = load_of(
            "e2b", [failed("e2b", ago=9), failed("e2b", ago=2), failed("e2b", ago=6)], now=NOW
        )
        assert load.newest_failure_at == NOW - timedelta(minutes=2)

    def test_a_record_the_repository_wrapped_is_read_the_same(self) -> None:
        class Record:
            def __init__(self, value: dict) -> None:
                self.value = value

        assert load_of("e2b", [Record(failed("e2b"))], now=NOW).failures == 1

    def test_nothing_at_all_is_a_provider_with_nothing_against_it(self) -> None:
        assert load_of("e2b", None, now=NOW).error_rate == 0.0
        assert load_of("e2b", [], now=NOW).queue_depth == 0


# -- the decision ---------------------------------------------------------------


class TestTheErrorRateArm:
    def test_half_of_four_opens_it(self) -> None:
        load = load_of("e2b", [failed(), failed(ago=2), build(), build(ago=3)], now=NOW)
        health = decide(load, now=NOW)
        assert health.degraded is True and health.arm == "error_rate"
        assert "2 of the last 4 e2b builds failed" in health.reason
        assert "50%" in health.reason
        assert health.retry_after_seconds == LIMITS.cool_down_seconds - 60

    def test_three_failures_out_of_three_is_not_enough_to_go_on(self) -> None:
        """Under the attempt floor a rate is noise: three builds by one owner
        with one bad afternoon would close a provider for the platform."""
        load = load_of("e2b", [failed(), failed(ago=2), failed(ago=3)], now=NOW)
        assert load.error_rate == 1.0
        assert decide(load, now=NOW).degraded is False

    def test_a_provider_that_mostly_works_stays_open(self) -> None:
        builds = [failed(), *[build(ago=index) for index in range(1, 6)]]
        assert decide(load_of("e2b", builds, now=NOW), now=NOW).degraded is False

    def test_failures_that_are_not_the_providers_never_open_it(self) -> None:
        builds = [failed(code="DL_ENV_BUILD_FAILED", ago=index) for index in range(1, 9)]
        load = load_of("e2b", builds, now=NOW)
        assert (load.attempts, load.failures) == (8, 0)
        assert decide(load, now=NOW).degraded is False

    def test_it_closes_once_the_cool_down_has_passed_since_the_last_failure(self) -> None:
        """The half-open state, without a probe to schedule: builds are taken
        again, and if the provider is still broken the next ones re-open it."""
        minutes = LIMITS.cool_down_seconds / 60
        builds = [failed(ago=minutes - 1), failed(ago=minutes), build(), build(ago=2)]
        load = load_of("e2b", builds, now=NOW)
        assert decide(load, now=NOW).degraded is True
        later = NOW + timedelta(seconds=61)
        # The same window, one minute past the cool-down.
        assert decide(load, now=later).degraded is False

    def test_a_shorter_cool_down_is_all_it_takes_to_wait_less(self) -> None:
        load = load_of("e2b", [failed(ago=5), failed(ago=6), build(), build(ago=2)], now=NOW)
        assert decide(load, thresholds=Thresholds(cool_down_seconds=60), now=NOW).degraded is False


class TestTheQueueDepthArm:
    def test_a_queue_at_the_limit_opens_it_whatever_the_rate(self) -> None:
        waiting = [{"variant": "e2b", "status": "queued", "created_at": at(3)} for _ in range(8)]
        health = decide(load_of("e2b", waiting, now=NOW), now=NOW)
        assert health.degraded is True and health.arm == "queue_depth"
        assert "8 builds are waiting on e2b" in health.reason
        # A queue drains at its own pace: there is no cool-down to quote.
        assert health.retry_after_seconds == 0

    def test_a_queue_under_the_limit_is_just_a_busy_provider(self) -> None:
        waiting = [{"variant": "e2b", "status": "running", "created_at": at(3)} for _ in range(7)]
        assert decide(load_of("e2b", waiting, now=NOW), now=NOW).degraded is False

    def test_it_closes_as_the_queue_drains(self) -> None:
        load = ProviderLoad(variant="e2b", queue_depth=8)
        assert decide(load, now=NOW).degraded is True
        assert decide(ProviderLoad(variant="e2b", queue_depth=7), now=NOW).degraded is False


# -- what a caller is told ------------------------------------------------------


class TestTheRefusal:
    def test_it_is_the_retryable_provider_code_and_names_the_provider(self) -> None:
        load = load_of("e2b", [failed(), failed(ago=2), build(), build(ago=3)], now=NOW)
        health = decide(load, now=NOW)
        with pytest.raises(EnvironmentsError) as refused:
            refuse_if_degraded(health)
        assert refused.value.code.code == "DL_ENV_PROVIDER_ERROR"
        assert refused.value.code.retryable is True, "the same request works once the provider does"
        assert "e2b is degraded and takes no optional build right now" in refused.value.message
        detail = refused.value.detail["degraded"]
        assert (detail["variant"], detail["arm"], detail["attempts"]) == ("e2b", "error_rate", 4)
        assert detail["retryAfterSeconds"] == LIMITS.cool_down_seconds - 60

    def test_a_healthy_provider_refuses_nothing(self) -> None:
        refuse_if_degraded(decide(ProviderLoad(variant="e2b"), now=NOW))

    def test_a_healthy_providers_body_says_only_that(self) -> None:
        assert Health(variant="modal").as_body() == {"variant": "modal", "degraded": False}

    def test_the_body_carries_what_a_dashboard_and_a_person_both_need(self) -> None:
        body = decide(
            load_of("e2b", [failed(), failed(ago=2), build(), build(ago=3)], now=NOW), now=NOW
        ).as_body()
        assert set(body) == {
            "variant",
            "degraded",
            "reason",
            "arm",
            "errorRate",
            "attempts",
            "queueDepth",
            "retryAfterSeconds",
        }
        assert body["errorRate"] == 0.5
