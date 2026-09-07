# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""A sandbox that cannot be interrupted, and said it had been.

`Sandbox.interrupt` sets a flag and calls `_do_interrupt`, whose default
"simply sets the interrupt flag" and returns `True`. `datalayer_sandbox` had
no override, so on a Datalayer runtime the interrupt was delivered nowhere
and reported success, and everything above believed it: `agent_runtimes`'
`interrupt_kernel` delegates straight to it, and so does the interrupt
`execute_cell` registers for `tasks/cancel`.

This variant, in detail. The invariants that hold for the whole package —
that no variant inherits the lying default, and that each marks the window
its interrupt is reachable through — are in
`test_every_variant_has_a_real_interrupt.py`.

Measured on prod1 on 2026-09-07: a cell looping for two minutes, cancelled
ten seconds in. `tasks/cancel` and `tasks/get` both answered `cancelled`,
and the next `execute_code` on that session took **60.8 seconds** and came
back empty — where a free kernel answers in about a second. The cell was
still running, and the runtime still being billed.

Launch the tests:
```
$ pytest tests/test_a_datalayer_sandbox_can_be_interrupted.py -v
```
"""

from __future__ import annotations

from code_sandboxes.datalayer_sandbox import DatalayerSandbox


class _Client:
    def __init__(self, answer=True, raises=None):
        self.answer, self.raises, self.calls = answer, raises, 0

    def interrupt(self):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.answer


class _Runtime:
    def __init__(self, client):
        self.sandbox_client = client


def _sandbox(runtime):
    made = DatalayerSandbox.__new__(DatalayerSandbox)
    made._runtime = runtime
    return made


class TestItHasOneAtAll:
    def test_datalayer_overrides_it_like_the_others(self):
        """The assertion that would have caught this: it is about the class,
        not about a call, because the default *succeeds* silently."""
        assert "_do_interrupt" in DatalayerSandbox.__dict__


class TestWhatItDoes:
    def test_it_asks_the_runtime_s_own_client(self):
        client = _Client()
        assert _sandbox(_Runtime(client))._do_interrupt() is True
        assert client.calls == 1

    def test_a_client_that_says_no_is_reported_as_no(self):
        """A cancel that could not reach the kernel is a cancel that failed."""
        assert _sandbox(_Runtime(_Client(answer=False)))._do_interrupt() is False

    def test_no_runtime_is_not_an_interrupt(self):
        assert _sandbox(None)._do_interrupt() is False

    def test_a_runtime_without_a_client_is_not_an_interrupt(self):
        assert _sandbox(_Runtime(None))._do_interrupt() is False

    def test_a_raising_client_is_answered_rather_than_propagated(self):
        """The task layer decides what a failed cancel means; this says so."""
        sandbox = _sandbox(_Runtime(_Client(raises=RuntimeError("kernel gone"))))
        assert sandbox._do_interrupt() is False

    def test_it_never_claims_success_it_did_not_get(self):
        """The whole defect in one line: the default returned True."""
        for runtime in (None, _Runtime(None), _Runtime(_Client(answer=False))):
            assert _sandbox(runtime)._do_interrupt() is not True


class TestTheChainReachesAKernel:
    """The whole path, because each link looked fine alone.

    `tasks/cancel` → the interrupt `execute_cell` registered →
    `CodeSandboxClient.interrupt` → `Sandbox.interrupt` → `_do_interrupt` →
    the runtime's own `sandbox_client`, which `RuntimeService._start` creates
    as a **jupyter-server** client → `POST /api/kernels/{id}/interrupt`.

    Two things are worth pinning here. The client the Datalayer variant
    delegates to wraps a *different* sandbox, so the delegation terminates
    rather than calling back into itself; and the variant at the far end
    implements `_do_interrupt`, so the chain ends at a kernel rather than at
    the base class's `return True`.
    """

    def test_the_client_forwards_to_the_sandbox_it_wraps(self):
        from code_sandboxes.client import CodeSandboxClient

        client = CodeSandboxClient.__new__(CodeSandboxClient)
        asked = []

        class _Sandbox:
            def interrupt(self):
                asked.append(True)
                return True

        client._sandbox = _Sandbox()
        assert client.interrupt() is True
        assert asked == [True], "the client answered without asking the sandbox"

    def test_the_far_end_of_the_delegation_implements_it(self):
        """`_do_interrupt` delegating to a variant that has none would be the
        same silence one level down."""
        from code_sandboxes.jupyter_server_sandbox import JupyterServerSandbox

        assert "_do_interrupt" in JupyterServerSandbox.__dict__

class TestTheInterruptIsReached:
    """`_do_interrupt` was necessary and not sufficient, which the first
    measurement after deploying it said plainly: still 60.5 seconds.

    `Sandbox.interrupt` refuses before it delegates::

        if not self._executing_event.is_set():
            return False
        self._interrupt_requested.set()
        return self._do_interrupt()

    `DatalayerSandbox.run_code` never set that event, so `is_executing` was
    always False and every interrupt was refused at the door — and reported
    as "no code was running" to a caller watching a cell run. Two guards,
    both needed: one that the door opens, one that what is behind it works.
    """

    def test_run_code_says_that_code_is_running(self):
        import threading

        from code_sandboxes.datalayer_sandbox import DatalayerSandbox

        seen = []

        class _Runtime:
            sandbox_client = None

            def execute(self, code, timeout=None):
                seen.append(sandbox._executing_event.is_set())
                raise RuntimeError("stop here; the flag is what is under test")

        sandbox = DatalayerSandbox.__new__(DatalayerSandbox)
        sandbox._started = True
        sandbox._runtime = _Runtime()
        sandbox._executing_event = threading.Event()
        sandbox._interrupt_requested = threading.Event()
        sandbox._connect = lambda: None
        sandbox._sandbox_id = "sb_test"

        class _Config:
            timeout = 30

        sandbox.config = _Config()
        sandbox.run_code("print(1)")

        assert seen == [True], "the sandbox ran code without saying it was running"
        assert not sandbox._executing_event.is_set(), "the flag outlived the execution"
