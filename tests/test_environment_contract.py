# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""`sandbox-contract/v1` and the Dockerfile validator: each rule, passing and failing."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_sandboxes.environments import errors
from code_sandboxes.environments.bases import is_approved_repository
from code_sandboxes.environments.conformance import CORE_CHECKS, EXTENDED_CHECKS
from code_sandboxes.environments.contract import (
    SANDBOX_CONTRACT_V1,
    SUPPORTED_CONTRACTS,
    check_dockerfile,
    contract_markdown,
    get_contract,
    main,
    parse_dockerfile,
    validate_dockerfile,
)
from code_sandboxes.environments.doctor.datalayer_sandbox import ROW_IDS
from code_sandboxes.environments.errors import EnvironmentsError

REPOSITORY = Path(__file__).resolve().parents[1]
PAGE = REPOSITORY / "docs" / "docs" / "environments" / "contract.mdx"
DIGEST = "sha256:" + "0" * 64


def test_the_contract_carries_the_identity_the_owner_took() -> None:
    """PLAN_ENV.md, D-6: gid 100 is the one departure from section 3."""
    contract = SANDBOX_CONTRACT_V1
    assert (contract.user, contract.uid, contract.gid) == ("datalayer", 1000, 100)
    assert (contract.home, contract.workdir) == ("/home/datalayer", "/home/datalayer/content")
    assert contract.reserved_path == "/opt/datalayer"
    assert contract.doctor_path == "/opt/datalayer/bin/datalayer-sandbox"
    assert SUPPORTED_CONTRACTS == ("sandbox-contract/v1",)


def test_every_row_is_checked_by_something_that_exists() -> None:
    known = {f"doctor:{row}" for row in ROW_IDS} | {"doctor"}
    known |= {f"conformance:{check}" for check in (*CORE_CHECKS, *EXTENDED_CHECKS)}
    for row in SANDBOX_CONTRACT_V1.rows:
        assert row.checked_by, row.area
        assert set(row.checked_by) <= known, row.area


def test_an_unsupported_contract_is_refused_with_the_supported_ones_named() -> None:
    assert get_contract("sandbox-contract/v1") is SANDBOX_CONTRACT_V1
    with pytest.raises(EnvironmentsError) as refused:
        get_contract("sandbox-contract/v2")
    assert refused.value.code is errors.CAPABILITY_UNSUPPORTED
    assert refused.value.detail["supported"] == ["sandbox-contract/v1"]


PASSING = f"""# syntax=docker/dockerfile:1
FROM datalayer/python-cpu:2026.09 AS base
RUN uv pip install --system geopandas==1.1.1 \\
    rasterio==1.4.3
FROM base
COPY <<EOF /home/datalayer/content/README.md
VOLUME is only text inside a heredoc
EOF
RUN --mount=type=cache,target=/root/.cache pip install numpy
RUN --mount=type=secret,id=token cat /run/secrets/token > /dev/null
RUN --mount=type=bind,from=base,source=/opt,target=/src ls /src
ENV LANG=C.UTF-8
FROM 123456789012.dkr.ecr.us-east-1.amazonaws.com/environments/base/python-cuda@{DIGEST}
"""


def test_a_dockerfile_on_an_approved_base_passes() -> None:
    assert validate_dockerfile(PASSING) == []
    check_dockerfile(PASSING)


@pytest.mark.parametrize(
    ("dockerfile", "line", "message"),
    [
        ("FROM datalayer/python-cpu\nVOLUME /data\n", 2, "VOLUME is refused"),
        ("FROM datalayer/python-cpu\nONBUILD RUN echo\n", 2, "ONBUILD is refused"),
        ("FROM datalayer/python-cpu\nstopsignal SIGINT\n", 2, "STOPSIGNAL is refused"),
        ("FROM python:3.12\n", 1, "not an approved Datalayer base"),
        ("FROM scratch\n", 1, "not an approved Datalayer base"),
        ("ARG BASE=datalayer/python-cpu\nFROM ${BASE}\n", 2, "build argument"),
        ("FROM --platform=linux/arm64 datalayer/python-cpu\n", 1, "linux/arm64"),
        ("FROM datalayer/python-cpu\nRUN --security=insecure make\n", 2, "privileged"),
        ("FROM datalayer/python-cpu\nRUN --network=host curl -s x\n", 2, "host network"),
        (
            "FROM datalayer/python-cpu\n"
            "RUN --mount=type=bind,source=/var/run/docker.sock,"
            "target=/var/run/docker.sock docker ps\n",
            2,
            "Docker socket",
        ),
        (
            "FROM datalayer/python-cpu\nRUN --mount=source=/etc,target=/host-etc ls\n",
            2,
            "host path `/etc`",
        ),
        ("FROM datalayer/python-cpu\nRUN docker run --privileged alpine\n", 2, "privileged mode"),
        ("FROM datalayer/python-cpu\nRUN echo one \\\n  && echo two\nVOLUME /x\n", 4, "VOLUME"),
        ("# escape=`\nFROM datalayer/python-cpu\nRUN echo one `\n  two\nVOLUME /x\n", 5, "VOLUME"),
    ],
)
def test_each_refusal_names_its_line(dockerfile: str, line: int, message: str) -> None:
    findings = validate_dockerfile(dockerfile)
    assert any(
        finding.line == line and message in finding.message for finding in findings
    ), findings


def test_a_refused_dockerfile_is_capability_unsupported_with_every_finding() -> None:
    dockerfile = "FROM python:3.12\nVOLUME /data\nSTOPSIGNAL SIGINT\n"
    with pytest.raises(EnvironmentsError) as refused:
        check_dockerfile(dockerfile)
    assert refused.value.code is errors.CAPABILITY_UNSUPPORTED
    assert refused.value.message.startswith("line 1:")
    assert [finding["line"] for finding in refused.value.detail["findings"]] == [1, 2, 3]


def test_the_parser_joins_continuations_and_skips_comments_inside_them() -> None:
    instructions = parse_dockerfile(
        "FROM datalayer/python-cpu\n"
        "RUN apt-get update \\\n"
        "# a comment inside the continuation\n"
        "    && apt-get install -y gdal-bin\n"
    )
    assert [(item.line, item.keyword) for item in instructions] == [(1, "FROM"), (2, "RUN")]
    assert instructions[1].arguments == "apt-get update && apt-get install -y gdal-bin"


@pytest.mark.parametrize(
    ("image", "approved"),
    [
        ("datalayer/python-cpu", True),
        ("datalayer/python-cpu:2026.09", True),
        (f"ghcr.io/datalayer/python-cuda@{DIGEST}", True),
        ("123456789012.dkr.ecr.us-east-1.amazonaws.com/environments/base/python-cpu:2026.09", True),
        ("localhost:5000/datalayer/python-cpu", True),
        ("python:3.12", False),
        ("datalayer/python-cpu-evil", False),
        ("evil/datalayer-python-cpu", False),
    ],
)
def test_an_approved_base_is_recognized_in_any_registry(image: str, approved: bool) -> None:
    assert is_approved_repository(image) is approved


def test_the_documentation_page_is_generated_from_the_contract(tmp_path: Path) -> None:
    assert PAGE.read_text(encoding="utf-8") == contract_markdown()
    assert main(["--check", str(PAGE)]) == 0
    stale = tmp_path / "contract.mdx"
    stale.write_text(contract_markdown().replace("gid 100", "gid 1000"), encoding="utf-8")
    assert main(["--check", str(stale)]) == 1
