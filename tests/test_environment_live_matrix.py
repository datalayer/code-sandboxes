# Copyright (c) 2025-2026 Datalayer, Inc.
# Datalayer License

"""The managed Environment builders, against the real providers (E2-11).

`test_environment_{e2b,daytona,modal}_builder.py` run each builder against a
double; this builds a real artifact on each real provider from a real,
hash-verified lock — not a trivial one, which every earlier live drill this
plan records used, and which hid the one finding that mattered (E2-04's
build-time doctor check) until a real lock exposed it — launches a sandbox
from that artifact, and runs the formal core tier
(:func:`code_sandboxes.environments.conformance.run_core_tier`) against it.
Everything this test creates is deleted where the provider allows it —
Daytona and Modal both do; E2B's SDK exposes no template-delete call at
all (found live, E2-03's own builder docstring), so a live run of
``TestE2B`` leaves a real template behind every time, for the owner to
clean up from the E2B dashboard.

E2B and Modal are `xfail(strict=False)`: each still has its own
already-documented, unticked gap that the core tier is expected to catch —
not at build time for either any more (both builds now succeed in full),
but at launch: E2B's `jupyter.service`/`code-interpreter.service` run as
root regardless of the build's own `set_user`, and E2B's own private
server hardcodes `/home/user` as the cwd; Modal's own identity gap is
closed (`modal_sandbox.py`'s driver now drops to `1000:100` for a
contract-built artifact), but checks 5/6 (imports, filesystem) still fail
for a separate, unexplained reason specific to a contract-built image
under repeated `exec`. `strict=False` means passing does not fail the run
either — an `XPASS` is exactly the signal that one of those gaps has been
closed and its plan box can tick. Daytona carries no such marker: it is
expected to pass in full, and a failure there is a real regression.

Run on purpose:

    make environments-live    # CODE_SANDBOXES_LIVE=1 pytest -m live \
                              #   tests/test_environment_live_matrix.py

and record the run against the plan's E2-11 box.

This module needs `get_builder` to actually build for `daytona`, `e2b` and
`modal` — E2-04, E2-03 and E2-05, each still its own open PR at the time
this was written. Opened from `main` on purpose, so it starts working for
each variant the moment that variant's PR merges, with no change of its
own; until then, running it live against an unmerged variant fails outright
(`DL_ENV_CAPABILITY_UNSUPPORTED`, "has not landed") rather than `xfail` —
correctly, since that is the true state on `main` today, not a defect this
test is expecting.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator

import pytest

from code_sandboxes.base import Sandbox
from code_sandboxes.environments.builders import BuildRequest, get_builder
from code_sandboxes.environments.conformance import run_core_tier
from code_sandboxes.environments.resolve import LocalResolveRunner, resolve_environment
from code_sandboxes.environments.spec import parse_environment
from code_sandboxes.providers import get_provider

pytestmark = [pytest.mark.live]

#: A tiny, real, pure-Python package: enough for `uv pip sync --require-hashes`
#: to do genuine work without a slow build. The protected kernel stack (D-9)
#: is merged in by the real resolver regardless — this is what makes a real
#: lock different from the trivial ones earlier drills used, and is exactly
#: what surfaced E2-04's build-time doctor finding.
_ENVIRONMENT_SPEC = {
    "apiVersion": "environments.datalayer.io/v1alpha1",
    "kind": "Environment",
    "metadata": {"name": "e2-11-live-matrix"},
    "spec": {
        "language": {"name": "python", "version": "3.13"},
        "base": {"ref": "datalayer/python-cpu", "channel": "2026.09"},
        "packages": {"python": {"manager": "uv", "dependencies": ["packaging==24.2"]}},
    },
}


def _live_requested() -> bool:
    return os.getenv("CODE_SANDBOXES_LIVE") == "1"


def _skip_unless_available(variant: str) -> None:
    if not _live_requested():
        pytest.skip("set CODE_SANDBOXES_LIVE=1 to run against the real providers")
    provider = get_provider(variant)
    if provider is None or not provider.is_available(os.environ):
        missing = ", ".join(
            sorted({v for r in (provider.requirements if provider else ()) for v in r.env_vars})
        )
        pytest.skip(f"{variant}: credentials not in the environment ({missing or 'unknown'})")
    if not (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")):
        pytest.skip(f"{variant}: AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY needed to pull the base")


def _ecr_registry_host() -> str:
    """The owner's own ECR registry host, read from AWS, never hardcoded."""
    import boto3

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = os.environ.get("AWS_REGION", "us-east-1")
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com"


def _ecr_login_password() -> str:
    import shutil

    aws = shutil.which("aws")
    if not aws:
        pytest.skip("the aws CLI is not on PATH; needed to mint the base's pull token")
    region = os.environ.get("AWS_REGION", "us-east-1")
    out = subprocess.run(  # noqa: S603 - a fixed argv, no shell, `aws` resolved above
        [aws, "ecr", "get-login-password", "--region", region],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.strip()


def _resolved_lock() -> tuple[str, dict[str, str]]:
    """A real, complete, hash-verified lock: the protected constraints
    merged in exactly as a real build would get them, resolved locally so
    this needs no BuildKit (D-9's own `LocalResolveRunner`, "for plane local
    and for tests")."""
    outcome = resolve_environment(
        spec=_ENVIRONMENT_SPEC, variants=["daytona"], runner=LocalResolveRunner()
    )
    return outcome["content"], outcome["resolved_bases"]


class _DaytonaCredential:
    def __init__(self, *, registry: str, password: str) -> None:
        self.provider_secrets = {"DAYTONA_API_KEY": os.environ.get("DAYTONA_API_KEY", "")}
        self.registry = registry
        self.username = "AWS"
        self.password = password


class _ModalCredential:
    """The shape durable's `_mint_credential` gives a Modal build: it reads
    `aws_session` (D-18), never `username`/`password` — this used to put the
    AWS keys there, a shape the worker never produces, so this test passed
    while the worker's Modal builds failed their base pull.

    In production `aws_session` is always the `modal_base_reader` IAM user's
    own real key: Modal's own `from_aws_ecr` never reads `AWS_SESSION_TOKEN`
    anywhere in its SDK, so an assumed session fails the same way a wrong key
    would ("The security token included in the request is invalid", found
    live 2026-09-18). This double carries whatever AWS credential the person
    running the live test has ambient — a session token is passed on if one
    is present, but a live Modal run only actually succeeds against a real,
    static key, the same requirement production has.
    """

    def __init__(self) -> None:
        self.provider_secrets = {
            "MODAL_TOKEN_ID": os.environ.get("MODAL_TOKEN_ID", ""),
            "MODAL_TOKEN_SECRET": os.environ.get("MODAL_TOKEN_SECRET", ""),
        }
        self.aws_session = {
            name: os.environ[name]
            for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
            if os.environ.get(name)
        }


def _build_request(variant: str, resolved_base: str, lock_text: str) -> BuildRequest:
    return BuildRequest(
        environment_uid="01k0env0000000000000000000",
        version=1,
        # A per-run nonce, not just the variant (found in review): E2B and
        # Daytona both fold `build_uid` into the artifact/template name, so a
        # constant one would have every nightly run collide with the
        # previous run's own artifact instead of building a fresh one.
        build_uid=f"live-{variant}-{uuid.uuid4().hex[:8]}",
        owner_uid="01k0wner000000000000000000",
        variant=variant,
        environment=parse_environment(_ENVIRONMENT_SPEC),
        lock_text=lock_text,
        lock_digest="sha256:" + "00" * 32,
        resolved_base=resolved_base,
        region="us" if variant == "daytona" else None,
        size_class="small",
    )


def _run_core_tier(sandbox: Sandbox) -> None:
    def restart() -> None:
        sandbox.stop()
        sandbox.start()

    result = run_core_tier(
        sandbox,
        python_version="3.13",
        expected_packages={"packaging": "24.2"},
        restart=restart,
    )
    failures = {check.id: check.detail for check in result.failures}
    assert result.passed, failures


@pytest.fixture(scope="module")
def real_lock() -> Iterator[tuple[str, str]]:
    """The lock text and the bare (registry-less) base reference, resolved
    once and shared by every variant in this module."""
    lock_text, resolved_bases = _resolved_lock()
    yield lock_text, resolved_bases["daytona"]


class TestDaytona:
    """No `xfail`: E2-04's own live drill already proved this passes in full."""

    def test_build_launch_and_the_core_tier(self, real_lock: tuple[str, str]) -> None:
        _skip_unless_available("daytona")
        lock_text, bare_base = real_lock
        registry = _ecr_registry_host()
        resolved_base = f"{registry}/{bare_base}"
        credential = _DaytonaCredential(registry=registry, password=_ecr_login_password())
        builder = get_builder("daytona", credential=credential)
        request = _build_request("daytona", resolved_base, lock_text)

        artifact = builder.build(request)
        try:
            assert builder.exists(artifact)
            sandbox = Sandbox.create(variant="daytona", artifact=artifact, timeout=600)
            sandbox.start()
            try:
                _run_core_tier(sandbox)
            finally:
                sandbox.stop()
        finally:
            import daytona

            daytona.Daytona().snapshot.delete(artifact.provider_artifact_id)


class TestModal:
    """`xfail(strict=False)`: checks 5 (imports) and 6 (filesystem) fail —
    confirmed live, 2026-09-13, the session driver refusing to start
    against a sandbox that reports itself already shutting down, while 1,
    2, 3, 4, 7, 8 and 9 all pass. This used to be attributed to the
    identity gap (checks 1/2 failed too, then) — closed since, live: a
    contract-built artifact's `ModalSandbox` now drops its driver to
    `1000:100` (`_start_driver`, gated on `image_id` so a plain
    `ModalSandbox` is unaffected), and checks 1/2 pass with it. 5/6 turned
    out to be a separate, still-unexplained issue: a plain, non-Environments
    `ModalSandbox` runs several sequential snippets with no trouble at all
    over the same span, so this is specific to a contract-built artifact's
    image under repeated `exec`. An `XPASS` here means that's closed too."""

    @pytest.mark.xfail(
        reason="checks 5 and 6 fail on a session driver restart the sandbox refuses, "
        "specific to a contract-built artifact and not yet root-caused (identity, "
        "checks 1 and 2, is fixed)",
        strict=False,
    )
    def test_build_launch_and_the_core_tier(self, real_lock: tuple[str, str]) -> None:
        _skip_unless_available("modal")
        import modal
        import modal.experimental

        os.environ["MODAL_IMAGE_BUILDER_VERSION"] = "2025.06"
        lock_text, bare_base = real_lock
        resolved_base = f"{_ecr_registry_host()}/{bare_base}"
        credential = _ModalCredential()
        builder = get_builder("modal", credential=credential)
        request = _build_request("modal", resolved_base, lock_text)

        artifact = builder.build(request)
        try:
            assert builder.exists(artifact)
            sandbox = Sandbox.create(variant="modal", artifact=artifact, timeout=600)
            sandbox.start()
            try:
                _run_core_tier(sandbox)
            finally:
                sandbox.stop()
        finally:
            modal.experimental.image_delete(
                artifact.provider_artifact_id, client=modal.Client.from_env()
            )


class TestE2B:
    """`xfail(strict=False)`: the build itself now succeeds in full — the
    uid/gid mismatch this used to refuse on at build time is fixed — but a
    live launch still fails the core tier's identity checks (root, not
    `1000:100`; cwd `/home/user`, not the content directory): the two
    `systemd` services that actually execute a launched sandbox's code
    (`jupyter.service`, `code-interpreter.service`) have no `User=` of
    their own, and E2B's own private server hardcodes `/home/user`,
    independent of anything the build sets. Root-caused and documented in
    full in `adapters/e2b.py`'s own module docstring; not yet fixed there,
    for the same reason `modal_sandbox.py`'s own `setpriv` gap was left
    open before it was closed — this one touches E2B's own private
    execution engine, not a one-off build step. An `XPASS` here means that
    gap closed and E2-03 can tick."""

    @pytest.mark.xfail(
        reason="the build succeeds; a live launch still runs as root because "
        "jupyter.service/code-interpreter.service have no User= and E2B's own "
        "private server hardcodes /home/user as the cwd — see adapters/e2b.py",
        strict=False,
    )
    def test_build_launch_and_the_core_tier(self, real_lock: tuple[str, str]) -> None:
        _skip_unless_available("e2b")
        lock_text, bare_base = real_lock
        resolved_base = f"{_ecr_registry_host()}/{bare_base}"
        builder = get_builder("e2b")
        request = _build_request("e2b", resolved_base, lock_text)

        artifact = builder.build(request)
        try:
            assert builder.exists(artifact)
            sandbox = Sandbox.create(variant="e2b", artifact=artifact, timeout=600)
            sandbox.start()
            try:
                _run_core_tier(sandbox)
            finally:
                sandbox.stop()
        finally:
            pass  # E2B's SDK exposes no template delete call (found live, E2-03).
