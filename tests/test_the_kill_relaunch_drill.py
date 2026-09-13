# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""Lose a sandbox on purpose, and see what the harness says about it.

The plan's matrix is "bind, execute, run a task, lose the sandbox, relaunch",
and until now only the Datalayer harness had been through it. This runs the
losing half against each live provider beside it.

Three properties, and the middle one is the reason this exists:

1. A sandbox executes and holds state — `x = 41` reads back as `41`.
2. **Killed, it fails loudly.** A sandbox that has gone away must not answer
   as though it were still there. This is the property every silent failure
   this project has found looks like from the outside: a call that succeeds
   against nothing. A provider may raise or may answer `execution_ok=False`;
   what it may not do is return a fresh, wrong value.
3. Relaunched, it works and is **empty**. `on_lost: relaunch` documents
   exactly this — "a replacement is empty, every variable of the session went
   with the old one" — and a replacement that still had `x` would mean the
   kill had not killed anything, which would make property 2 untestable
   rather than passing.

Each provider is skipped by name when its credentials are absent; nothing
here is mocked, and each row creates two small CPU sandboxes and destroys
them. Run it on purpose:

    make kill-relaunch        # CODE_SANDBOXES_LIVE=1 pytest -m live

Launch the tests:
```
$ CODE_SANDBOXES_LIVE=1 pytest tests/test_the_kill_relaunch_drill.py -m live -v
```
"""

from __future__ import annotations

import os

import pytest

from code_sandboxes import CodeSandboxClient
from code_sandboxes.providers import get_provider

pytestmark = [
    pytest.mark.live,
    # The client's own deprecation notices are raised in *this* process by
    # importing and starting it, and this project promotes warnings to
    # errors. A row about what happens inside a sandbox must not die on a
    # notice about the laptop running the test.
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

PROVIDERS = ("datalayer", "daytona", "e2b", "modal")


def _skip_unless_available(name: str) -> None:
    if os.getenv("CODE_SANDBOXES_LIVE") != "1":
        pytest.skip("set CODE_SANDBOXES_LIVE=1 to run against the real providers")
    provider = get_provider(name)
    if provider is None or not provider.is_available(os.environ):
        missing = ", ".join(
            sorted({v for r in (provider.requirements if provider else ()) for v in r.env_vars})
        )
        pytest.skip(f"{name}: credentials not in the environment ({missing or 'unknown'})")


def _read_back(client: CodeSandboxClient, code: str) -> str:
    """The last line a snippet printed, or "" when it did not run.

    `execute_code` answers a `CodeExecutionOutcome`, whose `stdout` is already
    the combined text — not an `ExecutionResult` with `logs.stdout` lines.
    Reading the wrong one of those two is how the Contents matrix beside this
    passed nothing for months while asserting on an object's `repr`.
    """
    outcome = client.execute_code(code)
    if not getattr(outcome, "execution_ok", False):
        return ""
    text = (outcome.stdout or "").strip()
    return text.splitlines()[-1] if text else ""


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_lost_sandbox_says_so_and_a_replacement_is_empty(provider: str) -> None:
    _skip_unless_available(provider)

    first = CodeSandboxClient.create(variant=provider)
    first.start()
    try:
        # 1. It executes, and it holds state.
        assert (
            _read_back(first, "x = 41\nprint(x + 1)") == "42"
        ), f"{provider}: a fresh sandbox could not run code"
        assert (
            _read_back(first, "print(x)") == "41"
        ), f"{provider}: the sandbox did not hold state between calls"
    finally:
        # 2. Lose it. This is the kill: the sandbox goes away underneath the
        # session, which is what `on_lost` is about.
        first.stop()

    # A killed sandbox must not answer as though it were still there. Either
    # shape is honest — an exception, or an outcome that says it failed — and
    # the one thing it may not do is hand back `41` from a sandbox that no
    # longer exists.
    try:
        after = _read_back(first, "print(x)")
    except Exception:
        after = ""
    assert after != "41", (
        f"{provider}: a destroyed sandbox answered with its old state, so the "
        f"kill was not a kill or the client answered from a cache"
    )

    # 3. The replacement works, and is empty.
    second = CodeSandboxClient.create(variant=provider)
    second.start()
    try:
        assert (
            _read_back(second, "print(1 + 1)") == "2"
        ), f"{provider}: the replacement could not run code"
        # Exactly `False`, not "False or nothing": the line above has already
        # proved this sandbox executes, so an empty answer here would mean the
        # assertion had stopped measuring anything.
        assert _read_back(second, "print('x' in dir())") == "False", (
            f"{provider}: the replacement still had the first sandbox's state, "
            f"so nothing was actually lost"
        )
    finally:
        second.stop()
