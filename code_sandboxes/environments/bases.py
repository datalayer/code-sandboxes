# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The approved bases an Environment may start from.

An Environment names a base by reference and channel — ``datalayer/python-cpu``
at ``2026.09`` — and resolution turns the pair into a digest per variant. Only
the bases below are approved while Environments build from packages; importing
arbitrary images comes later, with the contract injected and verified.

The Python versions are what each base's interpreter is: 3.13, measured by
``datalayer-sandbox doctor`` inside ``jupyter-python:0.1.1`` (3.13.14 on
2026-09-11) and again inside ``jupyter-python:0.2.0``, the image the ``2026.09``
channel is built on (PLAN_ENV.md, E1-05).

A channel's digests are written here only once its release has pushed the
channel to ECR and printed them. Until then the channel is known and has none,
and resolving it raises :class:`BaseChannelUnpublishedError` rather than answer
with a digest nobody pushed.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import normalize_variant
from .errors import ARTIFACT_MISSING, SPEC_INVALID, EnvironmentsError

__all__ = [
    "APPROVED_BASES",
    "ECR_BASE_PREFIX",
    "ApprovedBase",
    "BaseChannelUnpublishedError",
    "approved_base",
    "approved_repositories",
    "is_approved_repository",
    "resolve_base",
]

#: Where the bases are published in ECR (PLAN_ENV.md, D-18).
ECR_BASE_PREFIX = "environments/base/"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ApprovedBase(BaseModel):
    """One approved base."""

    model_config = ConfigDict(frozen=True)

    ref: str
    #: The ``major.minor`` Python versions an Environment on this base may ask for.
    python_versions: tuple[str, ...]
    #: Whether the base carries CUDA, which a GPU size class requires.
    accelerator: bool = False
    #: Channel, then variant, to the digest resolution pins. A channel with no
    #: variant is approved and not yet published.
    channels: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("channels")
    @classmethod
    def _digests_only(cls, channels: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        for channel, digests in channels.items():
            for variant, digest in digests.items():
                if not _DIGEST.fullmatch(digest):
                    raise ValueError(
                        f"channel {channel!r} pins {variant!r} to {digest!r}, "
                        "which is not a sha256 digest"
                    )
        return {
            channel: {normalize_variant(variant): digest for variant, digest in digests.items()}
            for channel, digests in channels.items()
        }

    @property
    def name(self) -> str:
        """The base's name without its namespace: ``python-cpu``."""
        return self.ref.rsplit("/", 1)[-1]

    @property
    def repository(self) -> str:
        """The ECR repository the base is published to: ``environments/base/python-cpu``."""
        return ECR_BASE_PREFIX + self.name


APPROVED_BASES: dict[str, ApprovedBase] = {
    base.ref: base
    for base in (
        # E1-05: jupyter-python:0.2.1 (the restored fork) plus the contract layer,
        # released 2026-09-12 to environments/base/python-cpu. One image, so every
        # variant pins the same digest until a variant needs a base of its own.
        ApprovedBase(
            ref="datalayer/python-cpu",
            python_versions=("3.13",),
            channels={
                "2026.09": dict.fromkeys(
                    # `.spec.VARIANTS`, spelled out: `spec` imports from this
                    # module, so importing it back here would be circular.
                    ("datalayer", "e2b", "daytona", "modal"),
                    "sha256:cd09308a0c5e5adeec7fb5d8d29cf7455e78ba86f5c6c095a79068a280e1254f",
                )
            },
        ),
        # E2-17: jupyter-python-cuda plus the same layer.
        ApprovedBase(
            ref="datalayer/python-cuda",
            python_versions=("3.13",),
            accelerator=True,
            channels={"2026.09": {}},
        ),
    )
}


class BaseChannelUnpublishedError(EnvironmentsError):
    """An approved channel that has no digest yet for the variant asked.

    Raised under ``DL_ENV_ARTIFACT_MISSING`` with the reason
    ``base_channel_unpublished``, the way D-8 gives its own reason under that
    code: the base the variant builds from is not in the registry. A retry
    succeeds once the channel's release has pushed it and its digest is
    written into :data:`APPROVED_BASES`.
    """

    def __init__(self, base: ApprovedBase, channel: str, variant: str) -> None:
        super().__init__(
            ARTIFACT_MISSING,
            f"`{base.ref}:{channel}` has no published digest for the {variant} variant; "
            f"it resolves once the channel is pushed to `{base.repository}`",
            detail={
                "reason": "base_channel_unpublished",
                "base": base.ref,
                "channel": channel,
                "variant": variant,
                "repository": base.repository,
            },
        )


def approved_base(ref: str, bases: dict[str, ApprovedBase] = APPROVED_BASES) -> ApprovedBase | None:
    return bases.get(ref)


def resolve_base(
    ref: str, channel: str, variant: str, bases: dict[str, ApprovedBase] = APPROVED_BASES
) -> str:
    """The digest ``ref`` at ``channel`` pins for ``variant``.

    A base that is not approved, or a channel it does not have, is the spec's
    fault (``DL_ENV_SPEC_INVALID``). A channel it has, with no digest for the
    variant, raises :class:`BaseChannelUnpublishedError`.
    """
    base = bases.get(ref)
    if base is None:
        raise EnvironmentsError(
            SPEC_INVALID,
            f"`{ref}` is not an approved base; approved: " + ", ".join(bases),
            detail={"field": "spec.base.ref", "base": ref, "approved": list(bases)},
        )
    digests = base.channels.get(channel)
    if digests is None:
        raise EnvironmentsError(
            SPEC_INVALID,
            f"`{ref}` has no channel `{channel}`; channels: "
            + (", ".join(base.channels) or "none"),
            detail={
                "field": "spec.base.channel",
                "base": ref,
                "channel": channel,
                "channels": list(base.channels),
            },
        )
    normalized = normalize_variant(variant)
    digest = digests.get(normalized)
    if digest is None:
        raise BaseChannelUnpublishedError(base, channel, normalized)
    return digest


def approved_repositories(bases: dict[str, ApprovedBase] = APPROVED_BASES) -> tuple[str, ...]:
    """Every repository name an approved base is published under."""
    names: list[str] = []
    for base in bases.values():
        names.extend((base.ref, base.repository))
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
