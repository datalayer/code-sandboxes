# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A base channel resolves to the digest its release pushed, and never to one nobody pushed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from code_sandboxes.environments import errors
from code_sandboxes.environments.bases import (
    APPROVED_BASES,
    ApprovedBase,
    BaseChannelUnpublishedError,
    approved_repositories,
    resolve_base,
)
from code_sandboxes.environments.errors import EnvironmentsError
from code_sandboxes.environments.spec import VARIANTS

DIGEST = "sha256:" + "a" * 64


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_2026_09_channel_of_python_cuda_has_no_digest_until_it_is_pushed(
    variant: str,
) -> None:
    """PLAN_ENV.md, E2-17: the CUDA channel is not in ECR yet, so nothing may be made up."""
    ref = "datalayer/python-cuda"
    assert APPROVED_BASES[ref].channels == {"2026.09": {}}
    with pytest.raises(BaseChannelUnpublishedError) as refused:
        resolve_base(ref, "2026.09", variant)
    error = refused.value
    assert isinstance(error, EnvironmentsError)
    assert error.code is errors.ARTIFACT_MISSING
    assert error.retryable is True
    assert error.detail == {
        "reason": "base_channel_unpublished",
        "base": ref,
        "channel": "2026.09",
        "variant": variant,
        "repository": "environments/base/" + ref.rsplit("/", 1)[-1],
    }
    assert "sha256:" not in str(error)
    assert error.to_body()["code"] == "DL_ENV_ARTIFACT_MISSING"


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_2026_09_channel_of_python_cpu_resolves_the_digest_its_release_pushed(
    variant: str,
) -> None:
    """PLAN_ENV.md, E1-05: released 2026-09-12, same digest for every variant."""
    ref = "datalayer/python-cpu"
    digest = "sha256:cd09308a0c5e5adeec7fb5d8d29cf7455e78ba86f5c6c095a79068a280e1254f"
    assert APPROVED_BASES[ref].channels == {"2026.09": dict.fromkeys(VARIANTS, digest)}
    assert resolve_base(ref, "2026.09", variant) == digest


def test_a_published_channel_resolves_to_the_digest_its_release_printed() -> None:
    base = ApprovedBase(
        ref="datalayer/python-cpu",
        python_versions=("3.13",),
        channels={"2026.09": {"datalayer": DIGEST, "Modal": DIGEST}},
    )
    bases = {base.ref: base}
    assert resolve_base("datalayer/python-cpu", "2026.09", "datalayer", bases) == DIGEST
    assert resolve_base("datalayer/python-cpu", "2026.09", " MODAL ", bases) == DIGEST
    with pytest.raises(BaseChannelUnpublishedError) as refused:
        resolve_base("datalayer/python-cpu", "2026.09", "e2b", bases)
    assert refused.value.detail["variant"] == "e2b"


def test_an_unknown_channel_is_the_specs_fault_and_names_the_channels() -> None:
    with pytest.raises(EnvironmentsError) as refused:
        resolve_base("datalayer/python-cpu", "2027.01", "datalayer")
    assert not isinstance(refused.value, BaseChannelUnpublishedError)
    assert refused.value.code is errors.SPEC_INVALID
    assert refused.value.detail["field"] == "spec.base.channel"
    assert refused.value.detail["channels"] == ["2026.09"]


def test_an_unapproved_base_is_the_specs_fault() -> None:
    with pytest.raises(EnvironmentsError) as refused:
        resolve_base("python", "3.13", "datalayer")
    assert refused.value.code is errors.SPEC_INVALID
    assert refused.value.detail["field"] == "spec.base.ref"
    assert refused.value.detail["approved"] == ["datalayer/python-cpu", "datalayer/python-cuda"]


@pytest.mark.parametrize(
    "pinned",
    ["latest", "2026.09", "sha256:abc", "sha256:" + "A" * 64, "python-cpu@sha256:" + "a" * 64],
)
def test_a_channel_pins_nothing_but_a_digest(pinned: str) -> None:
    with pytest.raises(ValidationError, match="not a sha256 digest"):
        ApprovedBase(
            ref="datalayer/python-cpu",
            python_versions=("3.13",),
            channels={"2026.09": {"datalayer": pinned}},
        )


def test_each_base_is_published_under_its_own_repository() -> None:
    assert approved_repositories() == (
        "datalayer/python-cpu",
        "environments/base/python-cpu",
        "datalayer/python-cuda",
        "environments/base/python-cuda",
    )
