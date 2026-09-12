# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The Modal variant's Environment builder (PLAN_ENV.md §6, §11.4, E2-05, E2-06).

Modal implements its **own** Dockerfile builder, and what it has not
implemented is what this refuses before a build is queued: `ONBUILD`,
`STOPSIGNAL` and `VOLUME` do nothing, `USER` is not honoured the way Docker
honours it, and an `ENTRYPOINT` must exec its arguments. Section 6 gives the
message this answers with, word for word:

    `VOLUME` is not supported by the Modal builder. Remove it, or drop
    `modal` from the optional variants.

Its artifact is the **image id**, `im-…`. A published name is mutable by
design, so a name is worth publishing for operability and is never what a
launch uses; and each chained builder call leaves an intermediate layer with
an id of its own that deleting the image does not delete, which is why
reconciliation counts them (E2-09).

The build itself is E2-05; until then every operation that reaches Modal
refuses by name.

@module code_sandboxes.environments.adapters.modal
"""

from __future__ import annotations

from ..builders import CapabilityFinding
from ..spec import Environment
from .managed import ManagedBuilder

__all__ = ["UNIMPLEMENTED_INSTRUCTIONS", "Builder"]

#: What Modal's own Dockerfile builder does not implement (§6).
UNIMPLEMENTED_INSTRUCTIONS = ("ONBUILD", "STOPSIGNAL", "VOLUME")


class Builder(ManagedBuilder):
    """Modal: the capability half, with the build waiting on E2-05."""

    variant = "modal"
    item = "E2-05"
    title = "Modal"
    #: Modal runs GPUs, in the owner's workspace (E2-17).
    gpu = True
    #: Modal artifacts are regionless.
    regions = ()
    #: 21 s for the section 4.1 example in E0-04, after a 91.2 s base import.
    max_build_seconds = 30 * 60
    forbidden_instructions = UNIMPLEMENTED_INSTRUCTIONS

    def _own_findings(
        self, environment: Environment, lock_text: str | None
    ) -> list[CapabilityFinding]:
        findings: list[CapabilityFinding] = []
        # The commands a spec runs after the install are the one place a
        # `packages` build can name a Dockerfile instruction. Modal would
        # accept the line and do nothing, which is the worst of the three
        # possible answers.
        for index, command in enumerate(environment.spec.commands.post_install):
            first = str(command).strip().split(" ", 1)[0].upper()
            if first in UNIMPLEMENTED_INSTRUCTIONS:
                findings.append(
                    CapabilityFinding(
                        code="DL_ENV_CAPABILITY_UNSUPPORTED",
                        message=(
                            f"`{first}` is not supported by the Modal builder. Remove it, or "
                            "drop `modal` from the optional variants"
                        ),
                        field=f"spec.commands.postInstall[{index}]",
                    )
                )
        return findings
