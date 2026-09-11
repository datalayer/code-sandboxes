# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The approved bases an Environment may start from.

An Environment names a base by reference and channel — ``datalayer/python-cpu``
at ``2026.09`` — and resolution turns the pair into a digest per variant. Only
the bases below are approved while Environments build from packages; importing
arbitrary images comes later, with the contract injected and verified.

The Python versions are what each base's interpreter is: 3.13, measured by
``datalayer-sandbox doctor`` inside ``jupyter-python:0.1.1``, the image both
channels are built on (3.13.14 on 2026-09-11). Publishing the channels
(PLAN_ENV.md, E1-05) fills ``channels`` with the digests and measures the
versions again against the images it pushes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "APPROVED_BASES",
    "ECR_BASE_PREFIX",
    "ApprovedBase",
    "approved_base",
    "approved_repositories",
    "is_approved_repository",
]

#: Where the bases are published in ECR (PLAN_ENV.md, D-18).
ECR_BASE_PREFIX = "environments/base/"


class ApprovedBase(BaseModel):
    """One approved base."""

    model_config = ConfigDict(frozen=True)

    ref: str
    #: The ``major.minor`` Python versions an Environment on this base may ask for.
    python_versions: tuple[str, ...]
    #: Whether the base carries CUDA, which a GPU size class requires.
    accelerator: bool = False
    #: Channel, then variant, to the digest resolution pins.
    channels: dict[str, dict[str, str]] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        """The base's name without its namespace: ``python-cpu``."""
        return self.ref.rsplit("/", 1)[-1]


APPROVED_BASES: dict[str, ApprovedBase] = {
    base.ref: base
    for base in (
        ApprovedBase(ref="datalayer/python-cpu", python_versions=("3.13",)),
        ApprovedBase(ref="datalayer/python-cuda", python_versions=("3.13",), accelerator=True),
    )
}


def approved_base(ref: str, bases: dict[str, ApprovedBase] = APPROVED_BASES) -> ApprovedBase | None:
    return bases.get(ref)


def approved_repositories(bases: dict[str, ApprovedBase] = APPROVED_BASES) -> tuple[str, ...]:
    """Every repository name an approved base is published under."""
    names: list[str] = []
    for base in bases.values():
        names.extend((base.ref, ECR_BASE_PREFIX + base.name))
    return tuple(names)


def is_approved_repository(image: str, bases: dict[str, ApprovedBase] = APPROVED_BASES) -> bool:
    """Whether an image reference names an approved base, in any registry.

    ``ghcr.io/datalayer/python-cpu@sha256:…`` and
    ``<account>.dkr.ecr.<region>.amazonaws.com/environments/base/python-cpu:2026.09``
    both name ``datalayer/python-cpu``; ``python:3.12`` names nothing approved.
    """
    repository = image.split("@", 1)[0]
    last = repository.rsplit("/", 1)[-1]
    if ":" in last:
        repository = repository[: len(repository) - len(last)] + last.split(":", 1)[0]
    return any(
        repository == name or repository.endswith("/" + name)
        for name in approved_repositories(bases)
    )
