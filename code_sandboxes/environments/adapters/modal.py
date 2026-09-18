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

**A GPU is a launch option on Modal, not part of the image** (section 11.4
item 10, E2-17). A GPU version builds the same image any version does, on the
CUDA base; the spec's `accelerator` names one of Modal's GPUs, checked at
`validate`, and the smoke test launches the image on it (`gpu="T4"`,
`"H100:2"`) and adds check 11 to the core tier.

**A build secret is attached to the `postInstall` steps that name it (E3-05).**
`run_commands` takes a per-step `secrets=` collection, a mechanism E2B and
Daytona lack. Each declared secret is resolved from IAM
(`resolve_build_secret`) before Modal is touched, made into a Modal Secret of
this build's own, attached to exactly the `run_commands` steps whose command
names it (`command_names_secret`), and deleted after the build the way the
base-reader secret is. Its value is redacted from the build's log and from a
failed build's error. Modal hands a secret to a step only as environment
variables, so a `mountAs: file` secret is refused: making it a file would
write it into a layer.

@module code_sandboxes.environments.adapters.modal
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..accounts import provider_account
from ..build_secrets import resolve_build_secret
from ..builders import (
    ArtifactMetadata,
    ArtifactReference,
    BuildRequest,
    CapabilityFinding,
    ValidationResult,
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
from ..redact import redact
from ..resolve import WHEELHOUSE_IMAGE_PATH, apt_pins_in
from ..resolve_conda import conda_lock_pip_requirements, is_conda_lock
from ..spec import BuildSecret, Environment, command_names_secret
from .managed import ManagedBuilder

__all__ = ["MODAL_GPUS", "UNIMPLEMENTED_INSTRUCTIONS", "Builder", "modal_gpu"]

#: What Modal's own Dockerfile builder does not implement (§6).
UNIMPLEMENTED_INSTRUCTIONS = ("ONBUILD", "STOPSIGNAL", "VOLUME")

#: The GPUs a Modal sandbox takes by name (E2-17), as its `gpu=` spells them.
MODAL_GPUS: tuple[str, ...] = (
    "T4",
    "L4",
    "A10G",
    "L40S",
    "A100",
    "A100-40GB",
    "A100-80GB",
    "H100",
    "H200",
    "B200",
)


def modal_gpu(accelerator_type: str, count: int = 1) -> str | None:
    """Modal's `gpu=` for an accelerator, or `None` when Modal has no such GPU.

    `t4` and `a100_80gb` name what Modal calls `T4` and `A100-80GB`, and a
    count above one is Modal's `"H100:2"`. An `RTX-4090` is Daytona's
    vocabulary, not Modal's.
    """
    name = accelerator_type.strip().upper().replace("_", "-")
    if name not in MODAL_GPUS:
        return None
    return f"{name}:{count}" if count > 1 else name


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


def _scrubbed(text: str, values: dict[str, str]) -> str:
    """The text with this build's secret values redacted, or as it was with none (E3-05)."""
    return redact(text, values.values()) if values else text


def _install_packages(image: Any, lock_text: str) -> Any:
    """The package layer for this lock: a conda source (E3-02) installs the
    `@EXPLICIT` lock with Modal's own `micromamba_install` — which brings
    micromamba itself, so no bootstrap is needed here — and layers the pip
    layer the solve resolved (the user's pip requirements and the protected
    pins over them, from the lock's own `# datalayer-pip:` header); a pip
    source runs `uv pip sync`."""
    if is_conda_lock(lock_text):
        image = image.micromamba_install(spec_file=_LOCK_PATH)
        pip_requirements = conda_lock_pip_requirements(lock_text)
        if pip_requirements:
            image = image.pip_install(*pip_requirements, find_links=WHEELHOUSE_IMAGE_PATH)
        return image
    return image.run_commands(
        f'pip install --no-cache-dir "uv=={_UV_VERSION}"',
        # Packages install as root: every Modal build step already runs as
        # root regardless of any `USER` line (see the module docstring), so
        # this is stating what is already true rather than asking for it.
        f"uv pip sync --system --require-hashes --find-links {WHEELHOUSE_IMAGE_PATH} {_LOCK_PATH}",
    )


def _post_install(
    image: Any, commands: list[str], declared: list[BuildSecret], step_secrets: dict[str, Any]
) -> Any:
    """Each `postInstall` command as its own step, with the secrets it names (E3-05)."""
    for command in commands:
        named = [step_secrets[s.id] for s in declared if command_names_secret(command, s)]
        image = image.run_commands(command, secrets=named) if named else image.run_commands(command)
    return image


def _intermediates_of(built: Any) -> tuple[str, ...]:
    """Every layer under a built image, by id, deepest first (E2-05, E2-09).

    Each chained builder call leaves an image of its own, and deleting the
    artifact does not delete them. Modal offers no call that lists an
    account's images, so an intermediate nobody wrote down at build time can
    never be found again — which is why this is recorded rather than
    discovered. Confirmed live on 2026-09-17: a three-call chain built through
    `Image.build` hydrates an `object_id` on every image in `deps()`, not only
    on the last.

    The built image itself is not an intermediate: it is the artifact.
    """
    found: list[str] = []
    seen: set[int] = set()

    def walk(image: Any) -> None:
        if id(image) in seen:
            return
        seen.add(id(image))
        for dependency in getattr(image, "deps", lambda: ())():
            if hasattr(dependency, "deps"):
                walk(dependency)
        object_id = getattr(image, "object_id", None)
        if object_id and image is not built:
            found.append(str(object_id))

    try:
        walk(built)
    except Exception:
        return ()
    return tuple(dict.fromkeys(found))


class Builder(ManagedBuilder):
    """Modal: the capability half (E2-06) and the build (E2-05)."""

    variant = "modal"
    item = "E2-05"
    title = "Modal"
    #: A `packages` list and, for conda (E3-02), an `environment.yml`
    #: dependency file installed with `micromamba_install`.
    build_sources = ("packages", "dependencyFile", "dockerfile")
    dependency_formats = ("conda",)
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
        resolve_secret: Callable[..., str] = resolve_build_secret,
    ) -> None:
        super().__init__(log=log, credential=credential)
        #: How one `BuildSecret`'s value is fetched (E3-05): IAM by default,
        #: the same seam the Datalayer builder takes.
        self._resolve_secret = resolve_secret
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
        for index, secret in enumerate(environment.spec.build_secrets):
            if secret.mount_as == "file":
                findings.append(
                    CapabilityFinding(
                        code="DL_ENV_CAPABILITY_UNSUPPORTED",
                        message=(
                            f"Modal gives `{secret.name}` to a step as an environment variable "
                            "only; a file would be written into a layer. Use `mountAs: env`, or "
                            "drop `modal` from the variants"
                        ),
                        field=f"spec.buildSecrets[{index}].mountAs",
                    )
                )
        accelerator = environment.spec.resources.accelerator
        if accelerator != "none" and modal_gpu(accelerator.type) is None:
            findings.append(
                CapabilityFinding(
                    code="DL_ENV_CAPABILITY_UNSUPPORTED",
                    message=(
                        f"Modal has no GPU called `{accelerator.type}`; it offers "
                        + ", ".join(MODAL_GPUS)
                    ),
                    field="spec.resources.accelerator.type",
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
        # Resolved before Modal is touched: a secret IAM will not give stops
        # the build with nothing to clean up in the owner's workspace.
        declared, values = self._resolved_secrets(request)
        sdk = self._modal_sdk()
        # Read from the process environment, not a per-call argument (see
        # the module docstring) — set before anything else touches the SDK.
        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = self._image_builder_version
        client = self._client(sdk)
        name = f"dl-{request.environment.metadata.name}-v{request.version}-{request.build_uid}"

        ecr_secret = self._ecr_secret(sdk, client)
        step_secrets: dict[str, Any] = {}
        try:
            for secret in declared:
                step_secrets[secret.id] = self._hydrated_secret(
                    sdk, client, {secret.name: values[secret.id]}, keep=step_secrets
                )
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
                image = _install_packages(image, request.lock_text)
                for command in files_step(request.environment, variant=self.variant):
                    image = image.run_commands(command)
                image = _post_install(image, spec.commands.post_install, declared, step_secrets)
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
                    logged.extend(_scrubbed(buffer.getvalue(), values).splitlines())
                    raise EnvironmentsError(
                        BUILD_FAILED,
                        f"The Modal build failed: {_scrubbed(str(error), values)}",
                        detail={"variant": self.variant, "name": name, "log": logged[-20:]},
                    ) from (None if values else error)
                logged.extend(_scrubbed(buffer.getvalue(), values).splitlines())
                for line in logged:
                    self._log(line)
            # A name for people and dashboards only (§11.4 item 3): moves
            # on every publish, and is never what a launch uses.
            built.publish(name)
        finally:
            # No standing credential is left in an account Datalayer does
            # not control, whether the build above succeeded or not (D-18).
            self._delete_secret(sdk, client, ecr_secret)
            for secret in step_secrets.values():
                self._delete_secret(sdk, client, secret)

        return ArtifactReference(
            variant=self.variant,
            immutable_reference=built.object_id,
            provider_artifact_id=built.object_id,
            size_class=request.size_class,
            mutable_alias=name,
            provider_account=provider_account(self.variant, self._provider_secrets()) or None,
            contract_version=spec.contract or SANDBOX_CONTRACT_V1.version,
            intermediates=_intermediates_of(built),
        )

    def _resolved_secrets(self, request: BuildRequest) -> tuple[list[BuildSecret], dict[str, str]]:
        """The build secrets this build attaches, and each one's value from IAM (E3-05).

        `_own_findings` refuses a `mountAs: file` secret at `validate()`. This
        guards it too, the same way `build` guards a GPU class, because
        nothing makes a caller validate first.
        """
        wanted = set(request.build_secret_ids)
        declared = [s for s in request.environment.spec.build_secrets if s.id in wanted]
        if any(secret.mount_as == "file" for secret in declared):
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "Modal gives a build secret to a step as an environment variable only; a "
                "`mountAs: file` secret would be written into a layer",
                detail={"variant": self.variant, "field": "spec.buildSecrets"},
            )
        values = {s.id: self._resolve_secret(s, owner_uid=request.owner_uid) for s in declared}
        return declared, values

    def _ecr_secret(self, sdk: Any, client: Any) -> Any | None:
        """A Modal Secret carrying this build's base-reader session (D-17, D-18).

        `from_aws_ecr` wants the IAM session itself — keys, token and region —
        not the docker-login `AWS`/token pair Daytona and E2B take, which is
        what the credential's `username`/`password` hold. So it reads
        `aws_session`. Reading the login pair as IAM keys is what this did
        until 2026-09-18: the worker minted `AWS`/<ECR token>, this wrote it
        as `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, and AWS answered "The
        security token included in the request is invalid" on the base pull,
        so no Modal build through the worker could ever succeed.

        `None` when the credential carries no session: the base is then
        whatever the ambient workspace can already reach, the same fallback
        `_client` takes with no owner token.
        """
        session = dict(getattr(self._credential, "aws_session", None) or {})
        if not (session.get("AWS_ACCESS_KEY_ID") and session.get("AWS_SECRET_ACCESS_KEY")):
            return None
        secret = sdk.Secret.from_dict(
            {
                **{
                    name: str(session[name])
                    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
                    if session.get(name)
                },
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

    def _hydrated_secret(
        self, sdk: Any, client: Any, env: dict[str, str], *, keep: dict[str, Any]
    ) -> Any:
        """A Modal Secret of this build's own for one build secret (E3-05).

        Created in the owner's workspace like `_ecr_secret`'s, and deleted by
        `build`'s own `finally`. A `hydrate` that fails is cleaned up here,
        since `build` never got it back. The error names no value.
        """
        secret = sdk.Secret.from_dict(env)
        try:
            secret.hydrate(client=client)
        except Exception as error:
            self._delete_secret(sdk, client, secret)
            # `from None`: Modal's own error is not kept, in case it echoes
            # what it was given. Its type is enough to tell failures apart.
            raise EnvironmentsError(
                PROVIDER_ERROR,
                f"Modal could not be given the build secret `{next(iter(env))}` "
                f"({type(error).__name__})",
                detail={"variant": self.variant, "made": len(keep)},
            ) from None
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

    def smoke_test(
        self,
        artifact: ArtifactReference,
        *,
        environment: Any = None,
        lock_text: str | None = None,
        secret_values: Sequence[str] = (),
    ) -> ValidationResult:
        """Launch the image and run Appendix B's core tier in it (E2-05).

        This box's own `Done when` asks for exactly this — "the live test
        launches by image id and passes the core tier" — and it refused
        through `ManagedBuilder`, so **no Modal build could reach
        `succeeded`**: the image was built in the owner's workspace and the
        build recorded failed at this step, the same state Daytona was in
        until 2026-09-17.

        **Launched by image id, never by name**: a published name is mutable
        by design (§6), so only the id says which artifact ran. Launched
        through `ModalSandbox` — the same class a person's launch uses —
        rather than a hand-rolled `Sandbox.create`, because a hand-rolled
        launch is exactly what hid, on 2026-09-13, that every image this
        builder made died the instant it was launched for real.

        **As the owner** (D-8): the sandbox is created with the same client
        the build used, so it runs in the workspace the image is in.

        **Restarted by replacing the sandbox.** Modal has no in-place
        restart, so check 7's restart stops this sandbox and starts another
        from the same image, which is the stronger form of the question.

        The sandbox is terminated whether the tier passed or not.
        """
        if environment is None:
            raise EnvironmentsError(
                CAPABILITY_UNSUPPORTED,
                "A Modal smoke test needs the version's spec: the core tier "
                "asks for the Python version it declared and the packages its "
                "lock pinned, and an artifact carries neither",
                detail={"variant": self.variant},
            )
        from ...modal_sandbox import ModalSandbox
        from ...models import SandboxConfig
        from ..conformance import expected_packages, run_accelerator_check, run_core_tier

        sdk = self._modal_sdk()
        accelerator = environment.spec.resources.accelerator
        gpu = None if accelerator == "none" else modal_gpu(accelerator.type, accelerator.count)
        sandbox = ModalSandbox(
            config=SandboxConfig(name=f"smoke-{artifact.provider_artifact_id}", gpu=gpu),
            app_name=f"dl-{environment.metadata.name}",
            image_id=artifact.provider_artifact_id,
            client=self._client(sdk),
        )
        self._log(f"Launching {artifact.provider_artifact_id} to smoke-test it")
        try:
            sandbox.start()
            result = run_core_tier(
                sandbox,
                python_version=environment.spec.language.version,
                expected_packages=expected_packages(environment, lock_text or ""),
                secret_values=tuple(secret_values),
                restart=lambda: self._restart(sandbox),
            )
            if gpu is not None:
                # The core tier passes on a machine with no GPU: a GPU version
                # is the version its spec describes only when its GPUs are
                # visible and its CUDA is the spec's (check 11, E2-17).
                result.checks.append(
                    run_accelerator_check(sandbox, cuda=accelerator.cuda, count=accelerator.count)
                )
            return result
        except EnvironmentsError:
            raise
        except Exception as error:
            raise self._provider_error("smoke-test the image", error) from error
        finally:
            try:
                sandbox.stop()
            except Exception as error:
                self._log(f"The smoke-test sandbox could not be stopped: {error}")

    @staticmethod
    def _restart(sandbox: Any) -> None:
        """Check 7's restart, as Modal can do it: a new sandbox from the same image."""
        sandbox.stop()
        sandbox.start()

    def delete(self, artifact: ArtifactReference) -> None:
        """Delete the image, and every intermediate layer this build recorded (E2-05, E2-09).

        Deleting the artifact does not delete the layers under it, and Modal
        offers no call that lists an account's images, so what is removed is
        what `build` wrote down — an intermediate nobody recorded can never be
        found again.

        Two answers are outcomes rather than failures, both found live on
        2026-09-17:

        * **Already gone** is success. A replay of a collection must delete the
          same set again with no harm, the same way the Datalayer collector
          treats an artifact that is not there.
        * **Not ours to delete.** The bottom of a chain can be an image the
          workspace does not own — Modal's own `debian_slim` answers
          `PermissionDenied` — and an image somebody else owns was never this
          artifact's to collect. It is logged and stepped over, not raised.
        """
        sdk = self._modal_sdk()
        client = self._client(sdk)
        synchronizer, api_pb2 = self._modal_internals()

        async def _delete(image_id: str) -> None:
            await client.stub.ImageDelete(api_pb2.ImageDeleteRequest(image_id=image_id))

        # The artifact last: an intermediate is only reachable while the
        # record naming it survives, so a half-done collection that has
        # dropped the image would strand them.
        for image_id in (*artifact.intermediates, artifact.provider_artifact_id):
            if not image_id:
                continue
            try:
                # A bare coroutine on Modal's stub silently does nothing, so
                # this runs on the SDK's own loop — see `_delete_secret`.
                synchronizer.wrap(_delete)(image_id)
            except sdk.exception.NotFoundError:
                continue
            except Exception as error:
                if "permission" in str(error).lower():
                    self._log(
                        f"The Modal image {image_id} is not this account's to delete: {error}"
                    )
                    continue
                if image_id == artifact.provider_artifact_id:
                    raise self._provider_error("delete the image", error) from error
                # One layer's refusal does not strand the rest, nor the
                # artifact this was called to collect.
                self._log(f"The Modal intermediate {image_id} could not be deleted: {error}")

    def _provider_error(self, what: str, error: BaseException) -> EnvironmentsError:
        return EnvironmentsError(
            PROVIDER_ERROR,
            f"Modal could not {what}: {error}",
            detail={"variant": self.variant},
        )
