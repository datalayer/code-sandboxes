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

**The base is pulled through a Secret made and torn down around this build**
(D-17, D-18): `Image.from_aws_ecr` takes IAM-shaped credentials
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`) as a
`modal.Secret`, and Modal stores it again, server-side, as an app-owned
secret no listing shows — E0-04's own finding, confirmed again live,
2026-09-13. Deleting it needs its id, which needs it hydrated first, and
deleting it at all needs a call this SDK does not expose as a plain method:
see `_delete_secret`.

**`USER` does nothing on Modal — found live, 2026-09-13.** A `USER root` or
`USER 1000:100` `dockerfile_commands()` step logs "Skipping USER
instruction, it is unsupported by Modal container images" and has no effect;
every build step simply runs as whatever Modal's own builder already runs
as (root, in practice — `apt-get install` needs no help to succeed here).
So, unlike Daytona (E2-04) and the Datalayer builder, **no `USER` line is
emitted at all** — writing one would be dead code pretending to do
something it cannot.

**The contract's own doctor check is not run at build time.** Every build
step here runs as root (see above), so `doctor --json` baked into the image
would report `uid: 0`, not the `1000:100` the contract asks for. Section
11.4 item 6 says a live launch is where this gets fixed — "every exec
re-asserts the user" — and, since 2026-09-13, it does:
`code_sandboxes.modal_sandbox.ModalSandbox._start_driver` sets
`os.setgid(100)`/`os.setuid(1000)` inside the driver it starts, in place of
the `setpriv` wrapper section 11.4 names, but only when `self._image_id` is
set — that is, only when the sandbox was launched from a built Environments
artifact, whose image this builder's own recipe (above) always chowns to
`1000:100`, USER line ignored or not. A plain `ModalSandbox` — `debian_slim`,
or anyone else's image, no `image_id` at all — never has this asked of it,
so nothing about general Modal sandbox usage changes. Confirmed live: the
same doctor check that used to report `uid: 0` now reports `1000:100`
against a real launched sandbox. Baking the check into the build would
still only report a *build-time* identity, never provable to match a
*launch-time* one from inside the build itself — this stays a launch-time
check, run by the live drills this item's own tests and E2-11's live matrix
carry out, not a build step.

**Neither the doctor nor the wheelhouse is copied in**, the same finding as
Daytona's: the Datalayer base already bakes both (E1-05), confirmed live —
only the lock is genuinely per-build.

**The image builder version is pinned, every build.** The 2023.12 (legacy)
default installs Modal's own runtime dependencies (`aiohttp`, `grpclib`, …)
from source against the base's Python, and `aiohttp`'s C extension does not
compile on 3.13 (`PyLongObject` lost `ob_digit`) — found live, confirmed
again 2026-09-13, the exact failure E0-04 already recorded. `2025.06` ships
prebuilt for 3.13. Modal reads this from `MODAL_IMAGE_BUILDER_VERSION` in
the process environment, not a per-call argument, so this builder sets it
process-wide before every build — the one place this adapter's own build
reaches outside itself, and worth knowing if a worker ever builds more than
one variant's image concurrently in the same process.

**The entrypoint must stay alive with no command at all, and still exec one
correctly if it is ever given one** (§6, §11.4 item 6) — which took three
live attempts. A quoted `exec "$@"` given inline as
`entrypoint(["/bin/sh", "-c", 'exec "$@"'])` came out of `entrypoint()`'s
own Dockerfile rendering as literal, broken `"exec "$@""` — it does not
escape a quote embedded in an argv element — and the sandbox it produced
shut down within seconds. An unquoted `exec $0 $@` avoided that (nothing
left to escape) but is not a correct forwarder either: unquoted `$@`
word-splits and glob-expands each argument, so a command argument
containing a space — a `python -c "…"` source, above all — arrives split
into several arguments instead of one (found in review). Fixed with a real
script file, not an inline one-liner: `/opt/datalayer/bin/entrypoint.sh`,
baked in with `exec "$@"` as real file *content*, needs no Dockerfile-string
escaping, and forwards correctly. But **the actual launcher,
`code_sandboxes.modal_sandbox.ModalSandbox.start()`, creates the sandbox
with no command args at all** — it execs into the running container
separately, after creation (found live, 2026-09-13, running this exact
builder's own artifact through it, not a hand-rolled `Sandbox.create` call
the way every earlier live check here did): `exec "$@"` with nothing to
expand is a no-op in `sh`, so the script fell straight through to its own
end and the container exited before that first real exec ever reached it.
The entrypoint now execs `sleep infinity` when it is given no arguments,
and `"$@"` when it is — correct either way a caller invokes it.

**A GPU size class is left buildable at `validate()`,** matching an existing
test (`test_a_gpu_spec_is_buildable_on_modal_and_daytona`), and refused at
`build()` instead, naming E2-17 — the same reasoning as Daytona's: the CUDA
base is not published yet, so this cannot be reached in practice, and this
builder does not know the launch-time `gpu=` argument to hand a size class
either (D-20 leaves that to a later item; a GPU is Modal's own launch
option, not an image property, per section 11.4 item 10).

**`spec.buildSecrets` is refused, unlike what Modal's own SDK could do.**
`run_commands`/`apt_install`/`dockerfile_commands` each take a per-step
`secrets=` collection — Modal has a real per-step secret mechanism, unlike
E2B or Daytona. What is missing is not Modal's capability but this branch's
own: `resolve_build_secret` (E3-05) is a separate, not-yet-merged PR, so
there is nothing here yet to turn a `build_secret_id` into a value. Refused
for now, with that reason, rather than silently built without it.

@module code_sandboxes.environments.adapters.modal
"""

from __future__ import annotations

import contextlib
import io
import os
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
    CAPABILITY_UNSUPPORTED,
    PROVIDER_ERROR,
    EnvironmentsError,
)
from ..files import files_step
from ..resolve import WHEELHOUSE_IMAGE_PATH, apt_pins_in
from ..spec import GPU_SIZE_CLASSES, Environment
from .managed import ManagedBuilder

__all__ = ["UNIMPLEMENTED_INSTRUCTIONS", "Builder"]

#: What Modal's own Dockerfile builder does not implement (§6).
UNIMPLEMENTED_INSTRUCTIONS = ("ONBUILD", "STOPSIGNAL", "VOLUME")

#: `uv`, pinned the same way every other builder's bootstrap is (E1-04/E3-04).
_UV_VERSION = "0.12.11"

_LOCK_PATH = "/opt/datalayer/lock.txt"
_CONTENT_DIR = "/home/datalayer/content"

#: The 2023.12 default fails installing Modal's own runtime deps on Python
#: 3.13 (E0-04, confirmed live 2026-09-13): see the module docstring.
_IMAGE_BUILDER_VERSION = "2025.06"

#: A real script file, not an inline one-liner (§6, §11.4 item 6): see the
#: module docstring for why. `entrypoint()` is given this one bare path,
#: nothing in it needing a Dockerfile-string escape.
_ENTRYPOINT_PATH = "/opt/datalayer/bin/entrypoint.sh"
#: `code_sandboxes.modal_sandbox.ModalSandbox.start()` — the actual launcher
#: — creates the sandbox with no command args at all, and execs into it
#: separately afterward (found live, 2026-09-13: `exec "$@"` with nothing to
#: expand is a no-op in `sh`, so the container's own PID 1 fell straight
#: through to the end of the script and exited, and the sandbox was gone
#: before the first real `exec` reached it). `sleep infinity` when nothing
#: is given keeps it alive for that; `exec "$@"` still wins when something
#: is, for a caller that does supply a command directly.
_ENTRYPOINT_SCRIPT = '#!/bin/sh\nif [ "$#" -eq 0 ]; then exec sleep infinity; fi\nexec "$@"\n'


def _modal_sdk() -> Any:
    try:
        import modal
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise EnvironmentsError(
            PROVIDER_ERROR,
            "No `modal` SDK to build an image with: install `code-sandboxes[modal]`",
            detail={"missing": "modal"},
        ) from error
    return modal


def _modal_internals() -> tuple[Any, Any]:
    """`synchronizer` and `api_pb2`, injectable the same way `_modal_sdk` is.

    Neither is reachable off the top-level `modal` package a test's own
    double stands in for — `_delete_secret` needs both, straight from the
    SDK's own internals (see its docstring), so they get their own loader
    rather than becoming an untestable bare import inside that method.
    """
    from modal._utils.async_utils import synchronizer
    from modal_proto import api_pb2

    return synchronizer, api_pb2


class Builder(ManagedBuilder):
    """Modal: the capability half (E2-06) and the build (E2-05)."""

    variant = "modal"
    item = "E2-05"
    title = "Modal"
    #: Modal runs GPUs, in the owner's workspace (E2-17). This builder does
    #: not build one yet: see `build`'s own guard.
    gpu = True
    #: Modal artifacts are regionless.
    regions = ()
    #: 21 s for the section 4.1 example in E0-04, after a 91.2 s base import.
    max_build_seconds = 30 * 60
    forbidden_instructions = UNIMPLEMENTED_INSTRUCTIONS

    def __init__(
        self,
        *,
        log: Callable[[str], None] | None = None,
        credential: Any = None,
        modal_sdk: Any = None,
        modal_internals: Callable[[], tuple[Any, Any]] | None = None,
        image_builder_version: str | None = None,
        region: str | None = None,
    ) -> None:
        super().__init__(log=log, credential=credential)
        self._modal_sdk = modal_sdk or _modal_sdk
        self._modal_internals = modal_internals or _modal_internals
        self._image_builder_version = image_builder_version or _IMAGE_BUILDER_VERSION
        #: The ECR repository's own region (D-16: "ECR repositories live in
        #: `us-east-1`"), never `BuildRequest.region` — Modal's own artifact
        #: is regionless, and a build's region is about where the *result*
        #: is hosted, not where the base it pulls from lives. The same
        #: default the Datalayer builder's own `_region` takes.
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client_instance: Any = None

    def _provider_secrets(self) -> dict[str, str]:
        """The owner's Modal secrets the build credential carries (D-8, E2-01).

        `BuildCredential.provider_secrets`, the same way the E2B and Daytona
        builders read it — an owner's own `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`,
        never logged, held for this build alone.
        """
        secrets = getattr(self._credential, "provider_secrets", None)
        return dict(secrets) if secrets else {}

    def _client(self, sdk: Any) -> Any:
        """The owner's own Modal client (D-8), built once and reused.

        With no token on the credential, the SDK's own `from_env()` falls
        back to the ambient environment — fine for a single-owner worker or
        a test, wrong for a real multi-owner one, the same fallback the E2B
        and Daytona builders document for their own keys.
        """
        if self._client_instance is None:
            secrets = self._provider_secrets()
            token_id = secrets.get("MODAL_TOKEN_ID")
            token_secret = secrets.get("MODAL_TOKEN_SECRET")
            try:
                if token_id and token_secret:
                    self._client_instance = sdk.Client.from_credentials(token_id, token_secret)
                else:
                    self._client_instance = sdk.Client.from_env()
            except Exception as error:
                # `build`/`inspect`/`exists` all call this before their own
                # `try`, so an auth failure here used to escape as a raw
                # Modal exception instead of the adapter's own taxonomy
                # (found in review).
                raise self._provider_error("authenticate", error) from error
        return self._client_instance

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
        if environment.spec.build_secrets:
            # Modal's own per-step `secrets=` mechanism could carry one —
            # unlike E2B or Daytona — but `resolve_build_secret` (E3-05) is
            # a separate, not-yet-merged branch, so there is nothing here
            # yet to turn an id into a value (found in this item's own
            # implementation, not a provider limit).
            ids = ", ".join(secret.id for secret in environment.spec.build_secrets)
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        f"`buildSecrets` ({ids}) needs `resolve_build_secret` (E3-05), not yet "
                        "in this branch — Modal's own per-step secrets could carry one once it "
                        "lands"
                    ),
                    field="spec.buildSecrets",
                )
            )
        return findings

    # -- Building -------------------------------------------------------------

    def build(self, request: BuildRequest) -> ArtifactReference:
        """Build an image from the resolved lock, and keep its id.

        The base is pulled through a Secret made for this build alone
        (D-17, D-18, see the module docstring); no `USER` line is emitted,
        because Modal does not honour one, and the doctor check runs at
        launch, not here, for the same reason.
        """
        spec = request.environment.spec
        if request.size_class in GPU_SIZE_CLASSES:
            # `validate` leaves a GPU class buildable (`gpu = True`, D-20):
            # the CUDA base E2-17 has not published yet, so this cannot be
            # reached in practice — refused plainly here rather than
            # guessing at the launch-time `gpu=` argument this class needs
            # (section 11.4 item 10 is a launch concern, not a build one).
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                f"Modal runs `{request.size_class}` on its own GPUs, but the CUDA base this "
                "needs is E2-17's, not built yet",
                detail={"variant": self.variant, "missing": "E2-17"},
            )
        if request.build_secret_ids or spec.build_secrets:
            # `_own_findings` already refuses this at `validate()` — this
            # guard is `build()`'s own, the same way the GPU one above is,
            # because nothing enforces that `build` is only ever called
            # after a passing `validate` (found in review): a caller
            # handing `build()` an already-resolved request could otherwise
            # get a successful image with a required credential silently
            # missing from it.
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "Modal's build secrets need `resolve_build_secret` (E3-05), not yet in this branch",
                detail={"variant": self.variant, "missing": "E3-05"},
            )
        sdk = self._modal_sdk()
        # Read from the process environment, not a per-call argument (see
        # the module docstring) — set before anything else touches the SDK.
        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = self._image_builder_version
        client = self._client(sdk)
        name = f"dl-{request.environment.metadata.name}-v{request.version}-{request.build_uid}"

        ecr_secret = self._ecr_secret(sdk, client)
        try:
            app = sdk.App.lookup(
                f"dl-{request.environment.metadata.name}", client=client, create_if_missing=True
            )
            image = sdk.Image.from_aws_ecr(request.resolved_base, secret=ecr_secret)
            # `env` before anything installs, the same order every other
            # builder keeps: a package that compiles against a library
            # found through an env var behaves differently without it.
            if spec.env:
                image = image.env(dict(spec.env))
            apt = apt_pins_in(request.lock_text)
            if apt:
                image = image.apt_install(*(f"{pkg}={apt[pkg]}" for pkg in sorted(apt)))
            # `add_local_file` only *records* the local path; it is read
            # only once `image.build(app)` actually runs, well after this
            # call returns — the scratch directory has to outlive the
            # whole build, not just the chain construction, or the build
            # fails with `FileNotFoundError` (found live, 2026-09-13, the
            # first real run of this exact code).
            with tempfile.TemporaryDirectory(prefix="dl-modal-build-") as scratch:
                lock_file = Path(scratch) / "lock.txt"
                lock_file.write_text(request.lock_text, encoding="utf-8")
                entrypoint_file = Path(scratch) / "entrypoint.sh"
                entrypoint_file.write_text(_ENTRYPOINT_SCRIPT, encoding="utf-8")
                # `copy=True`: the default mounts the local file at runtime
                # from this process's own filesystem, which is not what a
                # baked, reusable artifact needs (found live, 2026-09-13).
                image = image.add_local_file(str(lock_file), _LOCK_PATH, copy=True)
                image = image.add_local_file(
                    str(entrypoint_file), _ENTRYPOINT_PATH, copy=True
                ).run_commands(f"chmod +x {_ENTRYPOINT_PATH}")
                image = image.run_commands(
                    f'pip install --no-cache-dir "uv=={_UV_VERSION}"',
                    # Packages install as root: every Modal build step
                    # already runs as root regardless of any `USER` line
                    # (see the module docstring), so this is stating what
                    # is already true rather than asking for it.
                    "uv pip sync --system --require-hashes "
                    f"--find-links {WHEELHOUSE_IMAGE_PATH} {_LOCK_PATH}",
                )
                for command in files_step(request.environment, variant=self.variant):
                    image = image.run_commands(command)
                for command in spec.commands.post_install:
                    image = image.run_commands(command)
                image = image.workdir(_CONTENT_DIR).entrypoint([_ENTRYPOINT_PATH])

                logged: list[str] = []
                buffer = io.StringIO()
                try:
                    # `enable_output()` prints straight to this process's
                    # own stdout — Modal's build API takes no log callback
                    # the way E2B's and Daytona's do — so it is captured
                    # here rather than lost.
                    with contextlib.redirect_stdout(buffer), sdk.enable_output():
                        built = image.build(app)
                except Exception as error:
                    # The buffer is drained even on failure — found in
                    # review of the first version of this code: a build
                    # that raised inside the `with` block above never
                    # reached the line that read it, so a failed build's
                    # own log was silently dropped from the error detail.
                    logged.extend(buffer.getvalue().splitlines())
                    raise EnvironmentsError(
                        BUILD_FAILED,
                        f"The Modal build failed: {error}",
                        detail={"variant": self.variant, "name": name, "log": logged[-20:]},
                    ) from error
                logged.extend(buffer.getvalue().splitlines())
                for line in logged:
                    self._log(line)
            # A name for people and dashboards only (§11.4 item 3): moves
            # on every publish, and is never what a launch uses.
            built.publish(name)
        finally:
            # No standing credential is left in an account Datalayer does
            # not control, whether the build above succeeded or not (D-18).
            self._delete_secret(sdk, client, ecr_secret)

        return ArtifactReference(
            variant=self.variant,
            immutable_reference=built.object_id,
            provider_artifact_id=built.object_id,
            size_class=request.size_class,
            mutable_alias=name,
            provider_account=provider_account(self.variant, self._provider_secrets()) or None,
            contract_version=spec.contract or SANDBOX_CONTRACT_V1.version,
        )

    def _ecr_secret(self, sdk: Any, client: Any) -> Any | None:
        """A Modal Secret carrying this build's own base-reader credential (D-17, D-18).

        `None` when the credential carries no registry login: the base is
        then whatever the ambient workspace can already reach, the same
        fallback `_client` takes with no owner token.
        """
        username = str(getattr(self._credential, "username", "") or "")
        password = str(getattr(self._credential, "password", "") or "")
        if not (username and password):
            return None
        # `from_aws_ecr` wants IAM-shaped credentials, not the docker-login
        # `AWS`/token pair Daytona and E2B take (found live, 2026-09-13):
        # the same D-17 session, read differently.
        secret = sdk.Secret.from_dict(
            {
                "AWS_ACCESS_KEY_ID": username,
                "AWS_SECRET_ACCESS_KEY": password,
                "AWS_REGION": self._region,
            }
        )
        try:
            secret.hydrate(client=client)
        except Exception as error:
            # `hydrate` is the RPC that creates the app-owned remote secret
            # (see the module docstring): if it creates the secret and this
            # process then observes a timeout or another error on the same
            # call, the credential is left in the workspace with nothing
            # local pointing at it unless cleanup is attempted here too
            # (found in review) — best-effort, same as `_delete_secret`
            # always is, and harmless if nothing was actually created.
            self._delete_secret(sdk, client, secret)
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"Modal could not be given a pull credential for the base: {error}",
                detail={"variant": self.variant},
            ) from error
        return secret

    def _delete_secret(self, sdk: Any, client: Any, secret: Any | None) -> None:
        """Delete the app-owned copy Modal stores of this build's own secret.

        Modal exposes no plain method for this — `Secret.objects.delete`
        takes a *name*, and this secret has none (D-18's credential is
        never named or listed) — so the raw `SecretDelete` RPC is reached
        directly. Calling it as a bare coroutine silently does nothing (found
        live, 2026-09-13: no exception, no warning until Python's own GC
        noticed the unawaited coroutine) — it must run on the SDK's own
        background loop, the same one its `stub` is bound to, reached
        through `synchronizer.wrap` the way the SDK's own internals do
        (`modal.secret._SecretManager.delete`).
        """
        if secret is None:
            return
        try:
            synchronizer, api_pb2 = self._modal_internals()

            async def _delete(secret_id: str) -> None:
                await client.stub.SecretDelete(api_pb2.SecretDeleteRequest(secret_id=secret_id))

            synchronizer.wrap(_delete)(secret.object_id)
        except Exception as error:
            # Best-effort: the build's own result matters more than this
            # cleanup, and a left-behind secret is the owner's workspace's
            # to remove by hand — logged, never raised over a build result.
            self._log(f"Could not delete the Modal secret for this build: {error}")

    # -- Reading the registry ---------------------------------------------------
    #
    # `Image.from_id` is a weaker check here than the equivalent call is for
    # E2B or Daytona — found live, 2026-09-13: an id `experimental.image_delete`
    # had already removed still resolved through `from_id` without error,
    # every time, across four different deleted ids. What `image_delete`
    # removes is a layer's own blob (E0-04's own finding, "removes one
    # layer"), not the id-to-metadata record `from_id` reads — so `exists`
    # and `inspect` below correctly catch a malformed id or a real API
    # failure, but cannot yet tell "deleted" from "still there" the way the
    # other three variants can. Recording that plainly rather than letting
    # the code imply a guarantee it does not keep; E2-09's own reconciliation
    # work is where this needs a real answer, if Modal has a better one.

    def inspect(self, artifact: ArtifactReference) -> ArtifactMetadata:
        sdk = self._modal_sdk()
        client = self._client(sdk)
        try:
            image = sdk.Image.from_id(artifact.provider_artifact_id, client=client)
        except sdk.exception.NotFoundError as error:
            raise EnvironmentsError(
                ARTIFACT_MISSING,
                f"`{artifact.immutable_reference}` is not a Modal image",
                detail={"variant": self.variant, "reference": artifact.immutable_reference},
            ) from error
        except Exception as error:
            raise self._provider_error("read the image", error) from error
        return ArtifactMetadata(reference=artifact, labels={"image_id": str(image.object_id)})

    def exists(self, artifact: ArtifactReference) -> bool:
        sdk = self._modal_sdk()
        client = self._client(sdk)
        try:
            sdk.Image.from_id(artifact.provider_artifact_id, client=client)
        except sdk.exception.NotFoundError:
            return False
        except Exception as error:
            raise self._provider_error("ask whether the image exists", error) from error
        return True

    def _provider_error(self, what: str, error: BaseException) -> EnvironmentsError:
        return EnvironmentsError(
            PROVIDER_ERROR,
            f"Modal could not {what}: {error}",
            detail={"variant": self.variant},
        )
