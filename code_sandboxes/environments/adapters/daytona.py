# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Daytona variant's Environment builder (PLAN_ENV.md §6, §11.3, E2-04, E2-06).

Daytona's artifact is a **snapshot**, and two of its properties shape
everything here:

- **A snapshot bakes in CPU, memory, disk and GPU**, so a resource change is a
  new artifact for the same version rather than a different launch of the same
  one — which is why the artifact table has a region column and why the
  adapter refuses resources beside a snapshot at launch (correction 13).
- **A snapshot is region-scoped**, so `compatibility.regions` is load-bearing:
  an artifact built in one region cannot be launched in another.

It also rejects `latest`, `lts` and `stable` as the tag of a snapshot's source
image, and prefers a digest — which is what the Datalayer base channel
resolves to anyway (D-9).

The build itself is E2-04; until then every operation that reaches Daytona
refuses by name.

@module code_sandboxes.environments.adapters.daytona
"""

from __future__ import annotations

from ..builders import CapabilityFinding
from ..spec import Environment
from .managed import ManagedBuilder

__all__ = ["Builder"]

#: Tags Daytona refuses for a snapshot's source image: each moves.
MOVING_TAGS = ("latest", "lts", "stable")


class Builder(ManagedBuilder):
    """Daytona: the capability half, with the build waiting on E2-04."""

    variant = "daytona"
    item = "E2-04"
    title = "Daytona"
    #: Daytona runs GPUs, on its own hardware and the owner's account (E2-17).
    gpu = True
    #: The targets the owner's organization may build in. Empty until E2-04
    #: reads them from the organization: refusing a region nobody has listed
    #: would refuse every region.
    regions = ()
    #: 37.4 s for the section 4.1 example in E0-04, plus the wait for the
    #: snapshot to reach `Active`, which is asynchronous.
    max_build_seconds = 30 * 60

    def _own_findings(
        self, environment: Environment, lock_text: str | None
    ) -> list[CapabilityFinding]:
        findings: list[CapabilityFinding] = []
        tag = environment.spec.base.channel.strip().lower()
        if tag in MOVING_TAGS:
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        f"Daytona refuses `{tag}` as the tag of a snapshot's source image, "
                        "because it moves: name a dated channel"
                    ),
                    field="spec.base.channel",
                )
            )
        # A snapshot carries the machine it was built for, so two regions are
        # two artifacts. Said here so a spec asking for several knows it is
        # asking for several builds, not one.
        if len(environment.spec.compatibility.regions) > 1:
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        "A Daytona snapshot is region-scoped, so "
                        f"{len(environment.spec.compatibility.regions)} regions are "
                        f"{len(environment.spec.compatibility.regions)} artifacts of this version, "
                        "each built and stored separately"
                    ),
                    field="spec.compatibility.regions",
                )
            )
        return findings
