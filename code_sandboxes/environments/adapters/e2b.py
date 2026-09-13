# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The E2B variant's Environment builder (PLAN_ENV.md §6, §11.2, E2-03, E2-06).

What this release answers is whether a spec can be built for E2B at all —
before anything is queued — and it says no to the two things E2B cannot do:

- **No GPU.** E2B's sandboxes are Firecracker microVMs with no GPU
  passthrough, so a GPU size class is refused here and pointed at Modal and
  Daytona, which run one on their own hardware (D-20, E2-17).
- **No arbitrary base.** A template is compiled to a microVM from a
  Debian-derived image, which is what the Datalayer base channel is; E0-04
  found `code-interpreter-v1` unusable as one, since it runs kernels as root
  in `/home/user`.

The build itself is E2-03: it needs the Datalayer base published in ECR, so
until then every operation that reaches E2B refuses by name.

**The artifact is the build id.** A template name and its tags are mutable
pointers — a rebuild under the same name does not change the template id —
so `namespace/template:<build_id>` is the only reference a launch may use.

@module code_sandboxes.environments.adapters.e2b
"""

from __future__ import annotations

from ..builders import CapabilityFinding
from ..spec import Environment
from .managed import ManagedBuilder

__all__ = ["Builder"]


class Builder(ManagedBuilder):
    """E2B: the capability half, with the build waiting on E2-03."""

    variant = "e2b"
    item = "E2-03"
    title = "E2B"
    #: Firecracker microVMs: no GPU passthrough.
    gpu = False
    #: E2B artifacts are regionless.
    regions = ()
    #: A template build is quicker than an image build: 47 s for the section
    #: 4.1 example in E0-04, 13 to 20 s fully cached.
    max_build_seconds = 20 * 60

    def _own_findings(
        self, environment: Environment, lock_text: str | None
    ) -> list[CapabilityFinding]:
        findings: list[CapabilityFinding] = []
        # E2B ignores the image's USER, WORKDIR, ENV, ENTRYPOINT and CMD and
        # adds a sudo user of its own, so the template ends by setting them
        # (E2-03). A spec that asks for a user of its own would be silently
        # overridden, which is worth saying rather than discovering.
        if environment.spec.env.get("HOME"):
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        "E2B sets the sandbox's HOME itself — it ignores the image's and adds a "
                        "user of its own — so `env.HOME` would not be what a sandbox sees. "
                        "Remove it, or drop e2b from the variants"
                    ),
                    field="spec.env.HOME",
                )
            )
        return findings
