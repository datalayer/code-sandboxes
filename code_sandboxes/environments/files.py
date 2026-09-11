# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The files step every Environment builder shares.

An Environment may bake small files into its artifact. Every variant bakes
them the same way :mod:`code_sandboxes.builds` already bakes an Environment's
``contents:`` manifest — one verified fetch per file, and a manifest listing
what was baked — so this step is that code, fed from the spec's ``files``,
not a second implementation of it.
"""

from __future__ import annotations

from collections.abc import Callable

from ..builds import BuildEntry, EnvironmentBuild, build_commands
from .errors import SPEC_INVALID, EnvironmentsError
from .spec import VARIANTS, Environment, FileEntry

__all__ = ["build_entries", "files_step"]


def build_entries(
    environment: Environment, *, source_of: Callable[[FileEntry], str] | None = None
) -> list[BuildEntry]:
    """The spec's files as build entries.

    ``source_of`` turns a ``contentRef`` into a URL the build can fetch —
    a presigned URL for a ``blob://`` reference — and defaults to the
    reference itself. A file without its sha256 is refused: the build
    verifies every byte it bakes.
    """
    entries: list[BuildEntry] = []
    for index, entry in enumerate(environment.spec.files):
        if entry.sha256 is None:
            raise EnvironmentsError(
                SPEC_INVALID,
                f"spec.files[{index}]: `{entry.path}` needs its sha256 before it is built",
                detail={"field": f"spec.files[{index}].sha256"},
            )
        entries.append(
            BuildEntry(
                source_uri=source_of(entry) if source_of else entry.content_ref,
                destination_path=entry.path,
                sha256=entry.sha256,
                size_bytes=entry.size_bytes,
            )
        )
    return entries


def files_step(
    environment: Environment,
    *,
    variant: str,
    source_of: Callable[[FileEntry], str] | None = None,
) -> list[str]:
    """The shell commands that bake the spec's files, the same on every variant.

    Empty when the spec bakes nothing, so a builder can always run the step.
    """
    if variant not in VARIANTS:
        raise ValueError(f"{variant!r} is not a variant")
    if not environment.spec.files:
        return []
    build = EnvironmentBuild(
        environment=environment.metadata.name,
        provider=variant,  # type: ignore[arg-type]
        entries=build_entries(environment, source_of=source_of),
    )
    return build_commands(build)
