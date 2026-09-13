# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The E2B variant's Environment builder (PLAN_ENV.md §6, §11.2, E2-03, E2-06).

**Where the template starts from, and why it is not the Datalayer base.**
Section 11.2 says a template starts from the Datalayer base in ECR and adds
"a pinned code-interpreter server layer" — the FastAPI service on port 49999
and the token-less Jupyter Server on 8888 that E2B's own SDK calls
(``run_code``, contexts) actually talk to (E0-04). That server is E2B's own,
proprietary, and not published anywhere this package can vendor it from — it
ships baked into their own ``code-interpreter-v1`` template, whose source is
not public. Rather than guess at a private wire protocol, this builder starts
**from** ``code-interpreter-v1`` (:data:`CODE_INTERPRETER_BASE_TEMPLATE`) —
E0-04's own finding that it "runs kernels as root in ``/home/user``" already
names exactly the two things this builder fixes (``set_user``,
``set_workdir``), the same fix section 11.2 already asks for regardless of
which base a template starts from. What follows is unchanged from the
Datalayer builder's own discipline: the doctor, the wheelhouse and the
constraints file are copied in (built fresh from this package, the same
files the approved base bakes in at E1-05), and ``uv pip sync
--require-hashes`` against the resolved lock reconciles whatever packages
``code-interpreter-v1`` shipped to exactly what the lock pins — the same
reconciliation E1-07 already relies on when a base's own versions differ
from a lock's. This is not the private ECR base's own credential lifecycle
D-18 describes for the other three variants: nothing is pulled from ECR
here, so there is no base-reader session to mint or delete for this one.

**Packages install as root** (E0-04): a user install would land under
``/home/user``, which the runtime mounts over. The baked files and
``postInstall`` run as whichever user `set_user("datalayer")` made the
persistent default, named on no individual call — E2B's `copy`/`run_cmd`
*can* take a `user=` per call, but a user just created mid-build is not one
a per-call `user=` can name yet (found live, see `build`'s own comment);
`set_user` is. `run_cmd` has no per-call network isolation the way
Datalayer's `RUN --network=none` does (E1-07): E2B's builder API exposes
none, so `postInstall` here can reach the network during the build, which
E0-04's spike did not have reason to flag and which this box's own text
does not ask this variant to close.

**What remains before E2-03 ticks.** A live build (2026-09-13, this
session, template and account deleted afterward) ran the whole chain
through and produced a real `build_id` — but `code-interpreter-v1` already
holds an account at uid 1000 or gid 100 of its own, so `datalayer` here
lands on uid 1001, gid 1001, not the contract's `1000:100`, and the content
directory — chowned to the numbers the account was asked for, not the ones
it got — reads as not writable by it. `doctor --json` reports both,
correctly, and fails the build on them, so a real build refuses today
rather than shipping an artifact the contract's own check would not
accept. Getting the numeric identity to match needs either finding what in
`code-interpreter-v1` already holds 1000/100 and moving it, or asking E2B
how they mean a template built on their own base to get a chosen uid:gid —
neither attempted further here after three live builds each cost a real
uid/gid guess and a few minutes: see `build`'s own comment for exactly
what was tried and what broke. Untouched by this: `resolve`, `delete` and
tag-drift reconciliation (section 11.2's own remaining items), still
refusing via the inherited `ManagedBuilder` methods.

**The artifact is the build id.** A template name and its tags are mutable
pointers — a rebuild under the same name does not change the template id —
so ``<team>/<name>:<build_id>`` (E0-04's own verified format: `<name>` alone
launches whatever the newest build is, `<team>/<name>:<build_id>` pins one)
is the only reference a launch may use. The template id itself (needed to
look a build back up without resolving a name first) travels in
`ArtifactReference.mutable_alias`, which is documented as never launched
from — exactly what it is used for here.

**`smoke_test` is deliberately not implemented here.** Nothing on the real
build workflow calls `EnvironmentBuilder.smoke_test` — durable's own
`_SmokeTest` seam (`activities_environments.py`) is the actual orchestration
point, gated on E1-14's still-unbuilt internal trial route, with its own,
different signature (`artifact`, `build_uid`, `given_name`, `log` — no
`python_version` or `expected_packages`, which check 3 and check 5 of the
core tier need and which only the stored lock can answer). Building a
`smoke_test(artifact)` here that cannot get those two values would either
take invented defaults that make check 3 fail every artifact, or silently
skip the checks that name is for — both worse than refusing plainly, which
is what the inherited `ManagedBuilder.smoke_test` already does. A live proof
that the core tier passes belongs to a live *test*, which has the original
`BuildRequest` in scope and calls `Sandbox.create` and
`conformance.run_core_tier` directly, the same way E1-07's own live drills
never went through a `smoke_test` method either.

@module code_sandboxes.environments.adapters.e2b
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
)
from ..contract import SANDBOX_CONTRACT_V1
from ..errors import (
    ARTIFACT_MISSING,
    BUILD_FAILED,
    PROVIDER_ERROR,
    EnvironmentsError,
)
from ..files import files_step
from ..resolve import WHEELHOUSE_PATH
from ..spec import Environment
from .managed import ManagedBuilder

__all__ = ["CODE_INTERPRETER_BASE_TEMPLATE", "Builder"]

#: E2B's own published template that carries the code-interpreter server
#: `run_code` and contexts talk to (E0-04). Not the owner's image, and not
#: pulled from the owner's ECR — this is the provider's own base, the same
#: way `python:3.13-slim-bookworm` is a public base an `image` source names.
CODE_INTERPRETER_BASE_TEMPLATE = "code-interpreter-v1"

#: `uv`, pinned the same way E1-04/E3-04's bootstrap installs it.
_UV_VERSION = "0.12.11"

_DOCTOR_PATH = "/opt/datalayer/bin/datalayer-sandbox"
_WHEELHOUSE_PATH = "/opt/datalayer/wheelhouse"
_LOCK_PATH = "/opt/datalayer/lock.txt"
_CONTENT_DIR = "/home/datalayer/content"

#: The sandbox contract's own user (D-4, §3). `set_user` makes it the
#: template's persistent default; the steps that must run as root instead
#: override it back with their own per-call `user="root"`.
_CONTRACT_USER = "datalayer"


def _e2b_template_cls() -> Any:
    try:
        from e2b import Template
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise EnvironmentsError(
            PROVIDER_ERROR,
            "No `e2b` SDK to build a template with: install `code-sandboxes[e2b]`",
            detail={"missing": "e2b"},
        ) from error
    return Template


class Builder(ManagedBuilder):
    """E2B: the capability half (E2-06) and the build (E2-03)."""

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

    def __init__(
        self,
        *,
        log: Callable[[str], None] | None = None,
        credential: Any = None,
        team: str | None = None,
        template_cls: Any = None,
        zipapp_builder: Callable[[str | Path], Path] | None = None,
    ) -> None:
        super().__init__(log=log, credential=credential)
        #: The owner's team, prefixed on the reference this build's launched
        #: from (`<team>/<name>:<build_id>`, E0-04's own verified format).
        #: Read from the build credential the same way the registry host is
        #: for Datalayer — never a secret itself, an account's own name.
        self._team = team
        self._template_cls = template_cls or _e2b_template_cls
        #: How the doctor zipapp is built, injected in tests so nothing here
        #: needs a real filesystem write to be checked.
        self._zipapp_builder = zipapp_builder or _default_zipapp_builder()

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

    # -- Building -------------------------------------------------------------

    def build(self, request: BuildRequest) -> ArtifactReference:
        """Build a template from the resolved lock, and keep the build id.

        `code-interpreter-v1` is the starting point (see the module
        docstring); the doctor, the wheelhouse and the lock are copied in
        fresh, `uv pip sync --require-hashes` reconciles the base's own
        packages to the lock, and `set_user`/`set_workdir` fix what E2B's
        template ignores (E0-04).
        """
        template_cls = self._template_cls()
        spec = request.environment.spec
        name = f"dl-{request.environment.metadata.name}-v{request.version}"
        tag = f"v{request.version}-{request.build_uid}"
        with tempfile.TemporaryDirectory(prefix="dl-e2b-build-") as scratch:
            root = Path(scratch)
            # `copy`'s source must be relative to a context directory (found
            # live, 2026-09-13: an absolute path is refused outright) — this
            # build's own scratch directory is that context, named explicitly
            # rather than relying on the process's current directory.
            doctor = root / "datalayer-sandbox"
            self._zipapp_builder(doctor)
            wheelhouse = root / "wheelhouse"
            wheelhouse.mkdir()
            for wheel in WHEELHOUSE_PATH.glob("*.whl"):
                (wheelhouse / wheel.name).write_bytes(wheel.read_bytes())
            lock_file = root / "lock.txt"
            lock_file.write_text(request.lock_text, encoding="utf-8")

            chain = (
                template_cls(file_context_path=str(root))
                .from_template(CODE_INTERPRETER_BASE_TEMPLATE)
                # `code-interpreter-v1` sets a persistent DEFAULT USER of its
                # own, `user` (E0-04) — every `run_cmd` inherits it unless
                # given `user=` itself, so root-needing steps below all name
                # `user="root"` explicitly (found live, 2026-09-13: without
                # it, `mkdir /home/datalayer` failed "Permission denied").
                #
                # Three things found live, 2026-09-13, in this exact order,
                # each costing a real build to learn:
                #
                # 1. `useradd`/`groupadd` first, `set_user` after, is the one
                #    ordering that runs the whole chain through without a
                #    crash. `set_user("datalayer")` called **before** a real
                #    account exists — whether nothing had created one yet,
                #    or a later root step tried to renumber whatever
                #    `set_user` itself had just auto-created — left the
                #    account unable to exec anything at all: the very next
                #    step failed `/bin/sh: permission denied`, with nothing
                #    more specific reported. Not explained; only avoided.
                # 2. This `useradd -u 1000 -g 100 ...` is not believed to
                #    take effect: the doctor consistently reports the final
                #    account at uid 1001, gid 1001, not 1000:100 — some
                #    account in `code-interpreter-v1` already holds one or
                #    both numbers. `set_user("datalayer")` is what actually
                #    makes the account usable; this `useradd` line's own
                #    effect could not be confirmed independently of it.
                # 3. Because of (2), this same command's own
                #    `chown -R 1000:100` below chowns the content directory
                #    to numbers the account never actually held, and the
                #    doctor correctly reports the workdir as not writable.
                #    Chowning by name instead (`chown -R datalayer:datalayer`)
                #    would fix that — but only works once the account is
                #    real, i.e. after `set_user`, which is exactly the
                #    ordering (1) found unsafe. Left open rather than
                #    guessed at further; see this method's own "what
                #    remains" note in the module docstring.
                .run_cmd(
                    f"groupadd -g 100 {_CONTRACT_USER} 2>/dev/null; "
                    f"useradd -u 1000 -g 100 -m -d /home/{_CONTRACT_USER} -s /bin/bash "
                    f"{_CONTRACT_USER} 2>/dev/null; "
                    f"mkdir -p {_CONTENT_DIR} && chown -R 1000:100 /home/{_CONTRACT_USER}",
                    user="root",
                )
                .set_user(_CONTRACT_USER)
                .set_workdir(_CONTENT_DIR)
                .copy("datalayer-sandbox", _DOCTOR_PATH, mode=0o755, user="root")
                .copy("wheelhouse", _WHEELHOUSE_PATH, user="root")
                .copy("lock.txt", _LOCK_PATH, user="root")
                .run_cmd(f'pip install --no-cache-dir "uv=={_UV_VERSION}"', user="root")
                # Packages install as root (E0-04): a user install lands
                # under /home/user, which the runtime mounts over.
                .run_cmd(
                    "uv pip sync --system --require-hashes "
                    f"--find-links {_WHEELHOUSE_PATH} {_LOCK_PATH}",
                    user="root",
                )
            )
            if spec.env:
                chain = chain.set_envs(dict(spec.env))
            for command in files_step(request.environment, variant=self.variant):
                chain = chain.run_cmd(command)
            for command in spec.commands.post_install:
                chain = chain.run_cmd(command)
            # The contract's own check, in the image, at build time — the
            # same last layer Datalayer's Dockerfile ends on. `set_user` and
            # `set_workdir` were already set above; repeated here as the
            # chain's own last word, matching section 11.2's own wording
            # ("the template ends with set_user and set_workdir") — a
            # no-op if E2B's build agent already holds them, never wrong.
            chain = (
                chain.run_cmd(f"{_DOCTOR_PATH} doctor --json")
                .set_user(_CONTRACT_USER)
                .set_workdir(_CONTENT_DIR)
            )

            logged: list[str] = []

            def on_build_logs(entry: Any) -> None:
                message = f"[{getattr(entry, 'level', '')}] {getattr(entry, 'message', entry)}"
                logged.append(message)
                self._log(message)

            try:
                info = template_cls.build(chain, name, tags=[tag], on_build_logs=on_build_logs)
            except Exception as error:
                raise EnvironmentsError(
                    BUILD_FAILED,
                    f"The E2B build failed: {error}",
                    detail={"variant": self.variant, "name": name, "log": logged[-20:]},
                ) from error

        namespace = f"{self._team}/" if self._team else ""
        reference = f"{namespace}{info.name}:{info.build_id}"
        return ArtifactReference(
            variant=self.variant,
            immutable_reference=reference,
            provider_artifact_id=info.build_id,
            size_class=request.size_class,
            # The template id, never launched from, kept so `inspect`,
            # `exists` and `delete` need no extra round trip to resolve a
            # name to an id.
            mutable_alias=f"{info.template_id}:{info.name}",
            contract_version=spec.contract or SANDBOX_CONTRACT_V1.version,
        )

    # -- Reading the registry ---------------------------------------------------

    def _ids(self, artifact: ArtifactReference) -> tuple[str, str]:
        """The template id and name this artifact's build belongs to."""
        alias = artifact.mutable_alias or ""
        template_id, _, name = alias.partition(":")
        return template_id, name

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        template_cls = self._template_cls()
        template_id, name = self._ids(artifact)
        tags = template_cls.get_tags(template_id or name)
        found = next(
            (
                tag
                for tag in tags
                if getattr(tag, "build_id", None) == artifact.provider_artifact_id
            ),
            None,
        )
        if found is None:
            raise EnvironmentsError(
                ARTIFACT_MISSING,
                f"`{artifact.immutable_reference}` is not in the template's tags",
                detail={"variant": self.variant, "reference": artifact.immutable_reference},
            )
        return ArtifactMetadata(
            reference=artifact,
            created_at=str(getattr(found, "created_at", "") or "") or None,
            labels={"tag": str(getattr(found, "name", ""))},
        )

    def exists(self, artifact: ArtifactReference) -> bool:
        template_cls = self._template_cls()
        template_id, name = self._ids(artifact)
        if not template_cls.exists(template_id or name):
            return False
        tags = template_cls.get_tags(template_id or name)
        return any(getattr(tag, "build_id", None) == artifact.provider_artifact_id for tag in tags)


def _default_zipapp_builder() -> Callable[[str | Path], Path]:
    from ..doctor.build import build_zipapp

    return build_zipapp
