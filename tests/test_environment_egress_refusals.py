# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""A host the build pool's egress proxy refused, named as a finding (E2-19).

The logs under `egress_logs/` are real: `buildctl --progress=plain` against
`moby/buildkit:v0.33.0`, its steps sent through the build pool chart's own
Squid (`ubuntu/squid:6.6`, the chart's `squid.conf`, allowing `pypi.org`
alone), on 2026-09-19. Only the image layers' progress lines are left out.
`micromamba.log` is `micromamba create` 2.3.3 through the same proxy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.resolve import (
    egress_hosts_text,
    egress_refused_hosts,
    parse_resolver_failure,
)
from code_sandboxes.environments.resolve_conda import parse_conda_failure

from .test_environment_datalayer_builder import Buildctl, a_builder, a_request

LOGS = Path(__file__).parent / "egress_logs"


def a_log(name: str) -> str:
    return (LOGS / name).read_text(encoding="utf-8")


class TestTheRefusedHostIsReadFromWhatEachToolWrites:
    def test_pip_names_its_index_a_line_before_it_is_refused(self) -> None:
        # pip's retries say `Tunnel connection failed: 403` with a path only;
        # the host is the index it said it was looking in.
        assert egress_refused_hosts(a_log("buildctl-pip.log")) == ["pypi.example-private.io"]

    def test_uv_names_the_url_four_lines_above_its_tunnel_error(self) -> None:
        assert egress_refused_hosts(a_log("buildctl-uv.log")) == ["github.com"]

    def test_git_and_wget_are_named_and_curl_f_is_left_to_the_log(self) -> None:
        # `curl -f` says only "returned error: 403", which a host answering
        # 403 itself says too: not a finding.
        assert egress_refused_hosts(a_log("buildctl-curl.log"), proxy="http://172.18.0.2:3128") == [
            "github.com"
        ]

    def test_buildkitd_s_own_pull_of_a_base_is_named(self) -> None:
        assert egress_refused_hosts(a_log("buildctl-base.log")) == ["ghcr.io"]

    def test_micromamba_names_the_channel_a_line_above(self) -> None:
        # The issue tracker's URL after the refusal is not what was refused.
        assert egress_refused_hosts(a_log("micromamba.log")) == ["conda.anaconda.org"]

    def test_a_log_with_no_refusal_names_nothing(self) -> None:
        assert (
            egress_refused_hosts(
                "#5 [2/2] RUN pip install requests\n#5 1.2 Successfully installed\n"
            )
            == []
        )

    def test_a_host_s_own_403_over_its_own_tls_is_not_a_refusal(self) -> None:
        log = (
            "#6 [3/4] RUN curl -fsSL https://example.org/private -o /dev/null\n"
            "#6 0.065 curl: (22) The requested URL returned error: 403\n"
        )
        assert egress_refused_hosts(log) == []

    def test_steps_side_by_side_are_read_apart(self) -> None:
        # Step 8's URL is not step 7's refusal.
        log = (
            "#7 [3/4] RUN wget https://blocked.example/a.tar.gz\n"
            "#8 [4/4] RUN curl https://allowed.example/b\n"
            "#8 0.1 ok\n"
            "#7 0.067 Proxy tunneling failed: ForbiddenUnable to establish SSL connection.\n"
        )
        assert egress_refused_hosts(log) == ["blocked.example"]

    def test_the_proxy_itself_is_never_a_refused_host(self) -> None:
        log = (
            "#7 0.066 Connecting to proxy http://10.0.0.5:3128\n"
            "#7 0.067 Proxy tunneling failed: Forbidden\n"
        )
        assert egress_refused_hosts(log, proxy="http://10.0.0.5:3128") == []

    def test_each_host_once_in_the_order_refused(self) -> None:
        log = a_log("buildctl-uv.log") + a_log("buildctl-base.log") + a_log("buildctl-uv.log")
        assert egress_refused_hosts(log) == ["github.com", "ghcr.io"]

    def test_the_hosts_read_as_a_sentence(self) -> None:
        assert egress_hosts_text(["a.io"]) == "`a.io`"
        assert egress_hosts_text(["a.io", "b.io", "c.io"]) == "`a.io`, `b.io` and `c.io`"


class TestARefusalIsAFindingOfTheFailure:
    def test_a_build_refused_a_host_names_it_rather_than_the_log(self) -> None:
        run = Buildctl(returncode=1, digest=None, stderr=a_log("buildctl-uv.log"))
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(run=run).build(a_request())
        error = raised.value
        assert error.code.code == "DL_ENV_BUILD_FAILED"
        assert error.retryable is False
        assert "`github.com`" in error.message
        assert error.detail["refused_hosts"] == ["github.com"]
        assert error.detail["findings"] == [{"kind": "egress_refused", "subject": "github.com"}]

    def test_a_build_that_failed_otherwise_still_says_to_read_the_log(self) -> None:
        run = Buildctl(
            returncode=1, digest=None, stderr=a_log("buildctl-curl.log").split("#7 [4/4]")[0]
        )
        with pytest.raises(EnvironmentsError) as raised:
            a_builder(run=run).build(a_request())
        # git's refusal in step 5 is in this log too: it is named.
        assert raised.value.detail["refused_hosts"] == ["github.com"]
        with pytest.raises(EnvironmentsError) as plain:
            a_builder(run=Buildctl(returncode=1, digest=None, stderr="#5 0.1 exit 1\n")).build(
                a_request()
            )
        assert "findings" not in plain.value.detail
        assert plain.value.message == "The build failed; its log says where"

    def test_a_resolve_refused_an_index_is_not_retried_as_an_outage(self) -> None:
        error = parse_resolver_failure(a_log("buildctl-pip.log"))
        assert error.code.code == "DL_ENV_PACKAGE_NOT_FOUND"
        assert error.retryable is False
        assert "`pypi.example-private.io`" in error.message
        assert error.detail["findings"] == [
            {"kind": "egress_refused", "subject": "pypi.example-private.io"}
        ]

    def test_a_conda_solve_refused_a_channel_names_it(self) -> None:
        error = parse_conda_failure(a_log("micromamba.log"))
        assert error.code.code == "DL_ENV_PACKAGE_NOT_FOUND"
        assert error.retryable is False
        assert error.detail["refused_hosts"] == ["conda.anaconda.org"]

    def test_an_unreachable_index_that_the_proxy_did_not_refuse_is_still_an_outage(self) -> None:
        error = parse_resolver_failure(
            "error: Failed to fetch: `https://pypi.org/simple/x/`\n  Caused by: dns error\n"
        )
        assert error.code.code == "DL_ENV_PROVIDER_ERROR"
