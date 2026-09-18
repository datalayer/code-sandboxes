# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A `dockerfile` source, from validate to both builders (PLAN_ENV.md E3-03).

The author's Dockerfile names its base in `FROM`; that base, named by its
channel, is what the resolver pins and solves in, and what both builders pin
the `FROM` to. The contract's own lines come after the author's, and install
the lock rather than sync to it, so the author's own installs are kept.
"""

from __future__ import annotations

import pytest

from code_sandboxes.environments.bases import ApprovedBase
from code_sandboxes.environments.contract import (
    dockerfile_base,
    dockerfile_findings_for_build,
    pin_dockerfile_base,
)
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.resolve import COVERAGE_PREFIX, resolve_environment
from code_sandboxes.environments.spec import validate_environment

from .test_environment_datalayer_builder import a_builder as a_datalayer_builder
from .test_environment_datalayer_builder import a_request as a_datalayer_request
from .test_environment_daytona_builder import FakeDaytonaModule, calls_named
from .test_environment_daytona_builder import a_builder as a_daytona_builder
from .test_environment_daytona_builder import a_request as a_daytona_request
from .test_environment_resolve import A_LOCK, BASES, RecordedRunner, a_spec

PINNED = "registry.example/environments/base/python-cpu@sha256:" + "aa" * 32

AUTHORED = (
    "# syntax=docker/dockerfile:1\n"
    "FROM datalayer/python-cpu:2026.09 AS build\n"
    "RUN pip install --no-cache-dir rich==14.1.0\n"
    "FROM build\n"
    "COPY --from=build /etc/hostname /tmp/built-from\n"
    "COPY <<EOF /tmp/greeting\n"
    "hello\n"
    "EOF\n"
)


def findings(text: str) -> list[tuple[int, str]]:
    return [(finding.line, finding.message) for finding in dockerfile_findings_for_build(text)]


class TestWhatIsBuiltFromADockerfile:
    def test_a_channel_named_base_a_stage_a_heredoc_and_a_stage_copy_are_all_fine(self) -> None:
        assert findings(AUTHORED) == []

    @pytest.mark.parametrize(
        ("line", "why"),
        [
            ("FROM datalayer/python-cpu", "channel as its tag"),
            ("FROM datalayer/python-cpu@sha256:" + "bb" * 32, "by its channel, not a digest"),
            ("FROM datalayer/python-cpu:1999.01", "has no channel `1999.01`"),
        ],
        ids=["no-tag", "digest", "unknown-channel"],
    )
    def test_the_base_must_be_named_by_one_of_its_channels(self, line: str, why: str) -> None:
        [(number, message)] = findings(line + "\nRUN true\n")
        assert number == 1 and why in message

    def test_two_different_approved_bases_are_refused_on_the_second(self) -> None:
        text = "FROM datalayer/python-cpu:2026.09 AS a\nFROM datalayer/python-cpu:2026.10\n"
        bases = {
            "datalayer/python-cpu": ApprovedBase(
                ref="datalayer/python-cpu",
                python_versions=("3.13",),
                channels={"2026.09": {"datalayer": "sha256:" + "11" * 32}, "2026.10": {}},
            )
        }
        [finding] = dockerfile_findings_for_build(text, bases)
        assert finding.line == 2 and "same base and channel" in finding.message

    @pytest.mark.parametrize("keyword", ["COPY", "ADD"])
    def test_a_file_from_the_build_context_is_refused_until_it_can_be_uploaded(
        self, keyword: str
    ) -> None:
        [(number, message)] = findings(
            f"FROM datalayer/python-cpu:2026.09\n{keyword} app.py /app\n"
        )
        assert number == 2 and "`app.py` from the build context" in message

    def test_add_of_a_url_needs_no_context(self) -> None:
        assert (
            findings("FROM datalayer/python-cpu:2026.09\nADD https://example.org/a.tgz /a\n") == []
        )

    def test_a_different_escape_character_is_refused(self) -> None:
        [(number, message)] = findings("# escape=`\nFROM datalayer/python-cpu:2026.09\n")
        assert number == 1 and "escape" in message


class TestItsBase:
    def test_is_the_approved_base_the_from_names(self) -> None:
        base = dockerfile_base(AUTHORED)
        assert (base.ref, base.channel, base.lines) == ("datalayer/python-cpu", "2026.09", (2,))

    def test_is_refused_naming_the_line_when_it_cannot_be_pinned(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            dockerfile_base("RUN true\nFROM datalayer/python-cpu\n")
        assert raised.value.code.code == "DL_ENV_SPEC_INVALID"
        assert raised.value.message.startswith("line 2:")

    def test_is_pinned_where_it_is_named_and_nowhere_else(self) -> None:
        pinned = pin_dockerfile_base(AUTHORED, PINNED)
        assert pinned.splitlines()[0] == f"FROM {PINNED} AS build"
        # The stage stays a stage, and the author's own lines are untouched.
        assert "FROM build\n" in pinned
        assert "RUN pip install --no-cache-dir rich==14.1.0\n" in pinned
        # The author's directive goes: the build states its own frontend first.
        assert "# syntax=docker/dockerfile:1" not in pinned


def a_dockerfile_spec(content: str = AUTHORED) -> dict[str, object]:
    return a_spec(build={"source": "dockerfile", "dockerfile": {"content": content}})


class TestValidatingIt:
    def test_a_buildable_dockerfile_validates(self) -> None:
        validate_environment(a_dockerfile_spec())

    def test_what_cannot_be_built_is_refused_at_validate_with_its_line(self) -> None:
        with pytest.raises(EnvironmentsError) as raised:
            validate_environment(
                a_dockerfile_spec("FROM datalayer/python-cpu:2026.09\nCOPY a /a\n")
            )
        assert "line 2" in raised.value.message
        assert "build context" in raised.value.message


class TestResolvingIt:
    def test_the_base_is_the_one_the_from_names_not_spec_base(self) -> None:
        """`spec.base` is still required by the schema; the Dockerfile's
        `FROM` is what the build starts from, so it is what is pinned."""
        spec = a_dockerfile_spec()
        spec["spec"]["base"] = {"ref": "datalayer/python-cpu", "channel": "2026.10"}
        runner = RecordedRunner(A_LOCK)
        answer = resolve_environment(spec=spec, variants=["datalayer"], runner=runner, bases=BASES)
        assert answer["resolved_bases"]["datalayer"].endswith("11" * 32)
        assert runner.request is not None and runner.request.base_reference.endswith("11" * 32)

    def test_the_lock_says_what_it_does_not_cover(self) -> None:
        answer = resolve_environment(
            spec=a_dockerfile_spec(),
            variants=["datalayer"],
            runner=RecordedRunner(A_LOCK),
            bases=BASES,
        )
        [line] = [
            text for text in answer["content"].splitlines() if text.startswith(COVERAGE_PREFIX)
        ]
        assert "Dockerfile's own instructions install is not locked" in line

    def test_a_packages_lock_claims_no_such_thing(self) -> None:
        answer = resolve_environment(
            spec=a_spec(), variants=["datalayer"], runner=RecordedRunner(A_LOCK), bases=BASES
        )
        assert COVERAGE_PREFIX not in answer["content"]


class TestTheDatalayerBuilder:
    def request(self):
        return a_datalayer_request(
            spec={"build": {"source": "dockerfile", "dockerfile": {"content": AUTHORED}}}
        )

    def test_accepts_the_source(self) -> None:
        assert "dockerfile" in a_datalayer_builder().capabilities().build_sources

    def test_builds_from_the_authors_dockerfile_pinned_to_the_resolved_base(self) -> None:
        request = self.request()
        text = a_datalayer_builder().dockerfile(request)
        assert f"FROM {request.resolved_base} AS build" in text
        assert "RUN pip install --no-cache-dir rich==14.1.0" in text
        # The author's lines come before the contract's own.
        assert text.index("rich==14.1.0") < text.index("COPY lock.txt")

    def test_installs_the_lock_rather_than_syncing_the_authors_installs_away(self) -> None:
        text = a_datalayer_builder().dockerfile(self.request())
        assert "uv pip install --system --require-hashes" in text
        assert "-r /opt/datalayer/lock.txt" in text
        assert "uv pip sync" not in text

    def test_a_packages_source_still_syncs(self) -> None:
        assert "uv pip sync" in a_datalayer_builder().dockerfile(a_datalayer_request())


class TestTheDaytonaBuilder:
    def image(self, request):
        daytona = FakeDaytonaModule()
        a_daytona_builder(daytona=daytona).build(request)
        return daytona.client.snapshot.create_calls[0].args[0].image

    def test_starts_from_the_authors_dockerfile_pinned_to_the_resolved_base(self) -> None:
        request = a_daytona_request(
            spec={"build": {"source": "dockerfile", "dockerfile": {"content": AUTHORED}}}
        )
        image = self.image(request)
        [start] = calls_named(image, "from_dockerfile")
        assert f"FROM {request.resolved_base} AS build" in start.kwargs["content"]
        assert not calls_named(image, "base")

    def test_installs_the_lock_rather_than_syncing_it(self) -> None:
        request = a_daytona_request(
            spec={"build": {"source": "dockerfile", "dockerfile": {"content": AUTHORED}}}
        )
        runs = [call.args[0] for call in calls_named(self.image(request), "run_commands")]
        assert any(run.startswith("uv pip install --system --require-hashes") for run in runs)
        assert not any("uv pip sync" in run for run in runs)

    def test_a_packages_source_still_starts_from_the_base_and_syncs(self) -> None:
        image = self.image(a_daytona_request())
        assert calls_named(image, "base") and not calls_named(image, "from_dockerfile")
        runs = [call.args[0] for call in calls_named(image, "run_commands")]
        assert any("uv pip sync" in run for run in runs)
