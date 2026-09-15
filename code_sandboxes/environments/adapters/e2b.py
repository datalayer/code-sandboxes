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

**The build-time identity and locale gaps are closed; a deeper runtime one
was found underneath them.** Three earlier live builds (2026-09-13) landed
`datalayer` at uid 1001, gid 1001, not the contract's `1000:100` —
`code-interpreter-v1` already holds an account at uid 1000 (`user`) and a
group at gid 100 (`users`) of its own, so `useradd -u 1000 -g 100 ...`
silently fell back to the next free numbers instead of erroring. `build`'s
own comment records exactly what was tried; the fix that actually works is
not to create a second account at those numbers but to rename the one
already there — confirmed on a live, build-free probe of
`code-interpreter-v1` — and also close the account's now-orphaned former
private group, which a live build only revealed *after* the identity fix,
when it collided with E2B's own post-build "configuration script"
recreating a default `user` account of its own. A second, separate gap
found the same way: `code-interpreter-v1` sets no locale at all, so the
doctor's own `locale` check failed until `LC_ALL`/`LANG=C.UTF-8` were set
explicitly (the Datalayer base bakes this in at E1-05; Daytona and Modal
inherit it for free by starting from that base, this builder does not).

With both fixed, `doctor --json` passes **at build time** in full — but a
live launch of that same artifact through the real launcher
(`Sandbox.create`, not a hand-rolled build-time check) still fails the core
tier's identity checks: the running kernel is `root`, not `datalayer`, and
its cwd is `/home/user`, not the contract's content directory. The two are
different code paths. `set_user`/`set_workdir` only set the *build chain's*
own persistent default for later `run_cmd`/`copy` calls — they say nothing
about the two `systemd` services that actually execute a launched
sandbox's code, `jupyter.service` and `code-interpreter.service` (the
FastAPI server on 49999 from the module docstring above), and a live probe
of both units' files found neither names a `User=` at all, so both run as
whichever user their own unit's default is: `root`. The cwd is a second,
separate cause layered on top of the first — E2B's own private
`/root/.jupyter/jupyter_server_config.py` (`c.ServerApp.root_dir =
"/home/user"`) and `/root/.server/main.py` (`cwd = request.cwd or
"/home/user"`) hardcode that path for every kernel and context, independent
of which OS account runs the process. Closing this needs either patching
those two `systemd` units (`User=`/`Group=`, plausible, since they are this
build's own files to customize the same way `set_user`/`set_workdir`
already customize the Docker layer) and reconciling the `/home/user` the
FastAPI service still hardcodes with the contract's own content directory —
by a filesystem trick such as a symlink, since the strings themselves live
in E2B's private, unpublished source rather than anything this builder
ships — or a fix from E2B upstream. Not attempted further here: it touches
the actual execution engine every launched sandbox depends on, not a
one-off build step, and getting it wrong risks a template that fails to
start at all rather than one that starts as the wrong user — the same
reasoning `modal_sandbox.py`'s own still-open `setpriv` gap (E2-05) was
left at, for the same reason. Untouched by any of this: `resolve`, `delete`
and tag-drift reconciliation (section 11.2's own remaining items), still
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

import shlex
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..accounts import provider_account
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
from ..resolve import WHEELHOUSE_PATH, apt_pins_in
from ..resolve_conda import conda_lock_protected_pins, is_conda_lock
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

#: The contract's own locale (D-4, §3). The Datalayer base bakes
#: `LC_ALL=C.UTF-8` in (E1-05), and Daytona and Modal both start from it and
#: inherit that for free — `code-interpreter-v1` is not that base and sets no
#: locale at all, so the doctor's own `locale` check failed here (found
#: live, 2026-09-13: `LC_ALL`/`LANG` both unset) until this was set
#: explicitly.
_CONTRACT_LOCALE = "C.UTF-8"


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
    #: A `packages` list and, for conda (E3-02), an `environment.yml`
    #: dependency file installed with `micromamba`.
    build_sources = ("packages", "dependencyFile")
    dependency_formats = ("conda",)
    #: Firecracker microVMs: no GPU passthrough.
    gpu = False
    #: E0-04's spike found only a registry login for the private base, never
    #: a per-step arbitrary named secret (E3-05): `buildSecrets` is refused.
    supports_build_secrets = False
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

    def _provider_secrets(self) -> dict[str, str]:
        """The owner's E2B secrets the build credential carries (D-8, E2-01).

        `BuildCredential.provider_secrets` the same way the Datalayer builder
        reads `.registry`/`.username`/`.password` off it — an owner's own
        `E2B_API_KEY` (and `E2B_TEAM_ID`, when configured), never logged,
        held for this build alone.
        """
        secrets = getattr(self._credential, "provider_secrets", None)
        return dict(secrets) if secrets else {}

    def _api_key(self) -> str | None:
        """The owner's own E2B key, passed to every SDK call that needs one.

        Without it, `Template.build`/`get_tags`/`exists` fall back to the
        ambient `E2B_API_KEY` (found live, 2026-09-13, is what let this
        adapter's own live drill run at all) — fine for a single-owner
        worker or a test, wrong for a real multi-owner one, where a build
        must run in the *environment owner's* team, not whichever team the
        worker process itself happens to be configured for.
        """
        return self._provider_secrets().get("E2B_API_KEY") or None

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
        # `spec.buildSecrets` needs no check of its own here: `supports_build_secrets
        # = False` above (E3-05, merged since this branch started) makes
        # `ManagedBuilder._own_findings` refuse it before this method is
        # even reached — E0-04's spike found only a registry login for the
        # private base, never an arbitrary named secret.
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
                # **The uid/gid gap, found live, is now closed.** The
                # `useradd -u 1000 -g 100 ...` this used to run here always
                # landed the account at uid 1001, gid 1001 — not because
                # `useradd` failed, but because both numbers were already
                # taken: a direct probe of `code-interpreter-v1` (a plain
                # sandbox, no build) found `user:x:1000:1000:...` in
                # `/etc/passwd` and `users:x:100:user` in `/etc/group`, so
                # `-u 1000 -g 100` silently fell back to the next free pair
                # instead of erroring. The fix is not to create a second
                # account at those numbers — it is to rename the one that is
                # already there. `usermod -l datalayer -g 100 -d
                # /home/datalayer -m user` on that same probe produced
                # exactly `uid=1000(datalayer) gid=100(users)`: the existing
                # account, kept at its own uid, moved onto the existing
                # gid-100 group and given the contract's name and home.
                # Read who currently holds uid 1000 rather than hardcoding
                # `user`, so a future `code-interpreter-v1` that ships a
                # different account name — or none at all — still works:
                # renamed if one is there, created fresh (the original
                # `useradd`) if not. `getent group 100` is created first for
                # the same reason, in case a future base has no gid 100 at
                # all.
                #
                # One thing the rename leaves behind: the renamed account's
                # own former *private* group (`user`, gid 1000 on
                # `code-interpreter-v1`), now with no member since its
                # primary gid moved to 100. Found live, 2026-09-13, right
                # after the identity and locale fixes above finally let a
                # build get this far: E2B's own build pipeline runs a
                # "configuration script" of its own *after* this chain,
                # unconditionally recreating a default `user` account —
                # and its `useradd` failed the whole build outright
                # ("group user exists") because that leftover group was
                # still there for it to collide with. Deleting it once
                # nothing needs it anymore (never the gid-100 group itself)
                # is what let E2B's own step proceed.
                .run_cmd(
                    "getent group 100 >/dev/null 2>&1 || groupadd -g 100 users; "
                    "existing_uid_1000=$(getent passwd 1000 | cut -d: -f1); "
                    f'if [ -n "$existing_uid_1000" ]; then '
                    'old_group=$(id -gn "$existing_uid_1000"); '
                    f"usermod -l {_CONTRACT_USER} -g 100 -d /home/{_CONTRACT_USER} -m "
                    '"$existing_uid_1000"; '
                    '[ "$old_group" = "users" ] || groupdel "$old_group" 2>/dev/null || true; '
                    f"else useradd -u 1000 -g 100 -m -d /home/{_CONTRACT_USER} -s /bin/bash "
                    f"{_CONTRACT_USER}; fi; "
                    f"mkdir -p {_CONTENT_DIR} && chown -R 1000:100 /home/{_CONTRACT_USER}",
                    user="root",
                )
                .set_user(_CONTRACT_USER)
                .set_workdir(_CONTENT_DIR)
            )
            # `env` before anything installs, the same order the Datalayer
            # builder keeps (E1-07): a package that compiles against a
            # library found through an env var (`GDAL_DATA` and the like)
            # behaves differently without it. Found in review: this used to
            # run after uv's own install, too late for exactly that case.
            # The contract's own locale goes in unconditionally — merged
            # first, so a spec naming `LC_ALL`/`LANG` itself still wins.
            env = {"LC_ALL": _CONTRACT_LOCALE, "LANG": _CONTRACT_LOCALE}
            env.update(spec.env)
            chain = chain.set_envs(env)
            apt = apt_pins_in(request.lock_text)
            if apt:
                # The lock's own apt versions (D-9), the same pins the
                # Datalayer builder installs — found in review: this chain
                # had never installed them at all, so a spec naming a system
                # package reported as buildable and silently shipped without
                # it.
                pinned = " ".join(f"{name}={apt[name]}" for name in sorted(apt))
                chain = chain.run_cmd(
                    "apt-get update -qq && apt-get install -y --no-install-recommends "
                    f"{pinned} && rm -rf /var/lib/apt/lists/*",
                    user="root",
                )
            chain = (
                chain.copy("datalayer-sandbox", _DOCTOR_PATH, mode=0o755, user="root")
                .copy("wheelhouse", _WHEELHOUSE_PATH, user="root")
                .copy("lock.txt", _LOCK_PATH, user="root")
            )
            if is_conda_lock(request.lock_text):
                # A conda source (E3-02): `micromamba install --file` reads the
                # `@EXPLICIT` lock without re-solving, and the protected pip
                # pins the resolver forced over the pip layer come from the
                # lock's own `# datalayer-protected:` header, so the kernel
                # stack (E1-04) is present the same as for a pip source.
                chain = chain.run_cmd(
                    f"micromamba install --yes --name base --file {_LOCK_PATH}",
                    user="root",
                )
                pins = conda_lock_protected_pins(request.lock_text)
                if pins:
                    requirements = " ".join(shlex.quote(pin) for pin in pins)
                    chain = chain.run_cmd(
                        f"pip install --no-cache-dir --find-links {_WHEELHOUSE_PATH} "
                        f"{requirements}",
                        user="root",
                    )
            else:
                chain = chain.run_cmd(
                    f'pip install --no-cache-dir "uv=={_UV_VERSION}"', user="root"
                ).run_cmd(
                    # Packages install as root (E0-04): a user install lands
                    # under /home/user, which the runtime mounts over.
                    "uv pip sync --system --require-hashes "
                    f"--find-links {_WHEELHOUSE_PATH} {_LOCK_PATH}",
                    user="root",
                )
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

            build_kwargs: dict[str, Any] = {}
            api_key = self._api_key()
            if api_key:
                # The owner's own key (D-8), never the worker's ambient one:
                # found in review — this call took no credential at all
                # before, so a multi-owner worker would have built in
                # whichever team `E2B_API_KEY` happened to name.
                build_kwargs["api_key"] = api_key
            try:
                info = template_cls.build(
                    chain, name, tags=[tag], on_build_logs=on_build_logs, **build_kwargs
                )
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
            # The non-secret fingerprint of the account this build ran in
            # (E2-01) — found in review: this was left unset, so a launch
            # could never tell this artifact's account from another's.
            provider_account=provider_account(self.variant, self._provider_secrets()) or None,
            contract_version=spec.contract or SANDBOX_CONTRACT_V1.version,
        )

    # -- Reading the registry ---------------------------------------------------

    def _ids(self, artifact: ArtifactReference) -> tuple[str, str]:
        """The template id and name this artifact's build belongs to."""
        alias = artifact.mutable_alias or ""
        template_id, _, name = alias.partition(":")
        return template_id, name

    def _get_tags(self, template_cls: Any, id_or_name: str) -> list[Any]:
        """`Template.get_tags`, with the owner's key and the taxonomy's own
        error for anything that is not simply "no such template" (found in
        review: an auth or network failure here used to escape as a raw SDK
        exception, unlike the Datalayer registry adapter's own provider
        calls, which all map into `DL_ENV_PROVIDER_ERROR`)."""
        try:
            return list(template_cls.get_tags(id_or_name, **self._api_key_kwargs()))
        except Exception as error:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"E2B could not be asked for {id_or_name}'s tags: {error}",
                detail={"variant": self.variant, "id_or_name": id_or_name},
            ) from error

    def _api_key_kwargs(self) -> dict[str, Any]:
        api_key = self._api_key()
        return {"api_key": api_key} if api_key else {}

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        template_cls = self._template_cls()
        template_id, name = self._ids(artifact)
        tags = self._get_tags(template_cls, template_id or name)
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
        try:
            found = template_cls.exists(template_id or name, **self._api_key_kwargs())
        except Exception as error:
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"E2B could not be asked whether {template_id or name} exists: {error}",
                detail={"variant": self.variant},
            ) from error
        if not found:
            return False
        tags = self._get_tags(template_cls, template_id or name)
        return any(getattr(tag, "build_id", None) == artifact.provider_artifact_id for tag in tags)


def _default_zipapp_builder() -> Callable[[str | Path], Path]:
    from ..doctor.build import build_zipapp

    return build_zipapp
