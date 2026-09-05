# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""The Contents provider matrix, against the real providers.

`tests/test_contents_contract.py` runs the matrix against fake SDKs. This
runs the same scenario against a Datalayer cluster, Daytona, E2B and Modal
for real — one sandbox each: attach a manifest, materialize a file from a
public URL, restart, reconcile, detach, and read the manifest back. Each
provider is skipped, by name, when its credentials are not in the
environment; nothing here is mocked. Run it on purpose:

    make live-matrix          # CODE_SANDBOXES_LIVE=1 pytest -m live

and record the run against the plan's Milestone 8 box.
"""

from __future__ import annotations

import os
import uuid

import pytest

from code_sandboxes import CodeSandboxClient
from code_sandboxes.contents import ContentManifest
from code_sandboxes.providers import get_provider


# `pytestmark` is a *list*: assigning the filterwarnings mark on its own
# replaced the `live` mark this module already carried, and `-m live` then
# deselected every test — a run that reported `4 deselected` and nothing else.
pytestmark = [
    pytest.mark.live,
    # The client's own deprecations — pydantic's class-based config in its
    # models, the platformdirs migration in its paths — are raised in *this*
    # process by importing and starting the client, and this project promotes
    # warnings to errors. A row whose subject is what happens inside a sandbox
    # died twice on notices about the laptop running the test.
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
    pytest.mark.filterwarnings(
    # The client's own paths module warns about a platformdirs migration on
    # first use, and this project promotes warnings to errors. That warning is
    # the client's, raised in *this* process, and it failed the `datalayer`
    # row of a matrix whose subject is what happens inside the sandbox — a
    # real sandbox was created and the row died on a deprecation notice about
    # where a config file lives on the laptop running the test.
    "ignore:Datalayer is migrating its paths:DeprecationWarning"
),
]

PROVIDERS = ("datalayer", "daytona", "e2b", "modal")

#: A small, stable public file every sandbox can reach, and its digest.
PUBLIC_FILE = (
    "https://raw.githubusercontent.com/jakevdp/sklearn_tutorial/"
    "5098cee2a638c56c311aca0c18987e407fe127fd/LICENSE"
)


def _live_requested() -> bool:
    return os.getenv("CODE_SANDBOXES_LIVE") == "1"


def _skip_unless_available(name: str) -> None:
    if not _live_requested():
        pytest.skip("set CODE_SANDBOXES_LIVE=1 to run against the real providers")
    provider = get_provider(name)
    if provider is None or not provider.is_available(os.environ):
        missing = ", ".join(
            sorted({v for r in (provider.requirements if provider else ()) for v in r.env_vars})
        )
        pytest.skip(
            f"{name}: credentials not in the environment ({missing or 'no requirements known'})"
        )


def _manifest(provider: str, sandbox_uid: str, mount_path: str) -> ContentManifest:
    return ContentManifest(
        sandbox_uid=sandbox_uid,
        sandbox_provider=provider,
        generated_at="2026-08-26T00:00:00Z",
        contents_url=os.getenv("DATALAYER_CONTENTS_URL"),
        token=None,
        attachments=[
            {
                "uid": "att-client",
                "source_uid": "live-client",
                "sandbox_uid": sandbox_uid,
                "sandbox_provider": provider,
                "delivery": "client",
            },
            {
                "uid": "att-files",
                "source_uid": "live-files",
                "sandbox_uid": sandbox_uid,
                "sandbox_provider": provider,
                "delivery": "materialize",
                "mount_path": mount_path,
                "materialize": [{"source_url": PUBLIC_FILE, "path": "LICENSE"}],
            },
        ],
    )


@pytest.mark.parametrize("provider", PROVIDERS)
def test_the_matrix_against_a_live_provider(provider: str) -> None:
    _skip_unless_available(provider)
    sandbox_uid = f"live-{uuid.uuid4().hex[:12]}"
    # A path inside the sandbox this test creates, not on the machine running
    # it: S108 is about insecure local temp files and has nothing to say about
    # where a remote container mounts something.
    mount_path = "/tmp/contents-live"  # noqa: S108
    client = CodeSandboxClient.create(variant=provider)
    try:
        prepared = client.attach(_manifest(provider, sandbox_uid, mount_path))
        assert {item.uid: item.status for item in prepared} == {
            "att-client": "ready",
            "att-files": "ready",
        }

        # The file is there, and is the file.
        digest_of = (
            "import hashlib;"
            f"print(hashlib.sha256(open('{mount_path}/LICENSE','rb').read()).hexdigest())"
        )
        digest = client.run_command(f'python3 -c "{digest_of}"')
        # `run_command` answers a `CommandResult`; the digest is its stdout.
        # This read `str(digest)`, which is the object's repr —
        # `CommandResult(exit_code=0, stdout_len=65, ...)`, 55 characters —
        # so the assertion failed identically on every live provider, and
        # this matrix had never once passed against a real sandbox. The
        # sandboxes were created, attached and read correctly the whole time;
        # the test was measuring the wrong string.
        assert digest.exit_code == 0, digest.stderr
        assert len(digest.stdout.strip().splitlines()[-1]) == 64

        # The manifest inside says what was attached, and carries no token.
        location = client.contents_location
        assert location is not None
        written = client.read_file(location.manifest_path)
        assert "att-files" in written and "token" not in written

        # Restart, reconcile: the same answer, nothing done twice.
        client.restart()
        again = client.reconcile_contents(_manifest(provider, sandbox_uid, mount_path))
        assert {item.uid: item.status for item in again} == {
            "att-client": "ready",
            "att-files": "ready",
        }

        # Detach removes what was materialized and nothing else.
        client.detach("att-files")
        assert client.attachment_status("att-files") is None
        gone = client.run_command(f"test -e {mount_path}/LICENSE && echo present || echo absent")
        # Same mistake as the digest above, second occurrence: the answer is
        # in `.stdout`, and `str()` of the result is its repr.
        assert "absent" in gone.stdout, gone
    finally:
        client.close()
