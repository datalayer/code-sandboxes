# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""A sandbox that cannot be interrupted, and said it had been.

`Sandbox.interrupt` sets a flag and calls `_do_interrupt`, whose default
"simply sets the interrupt flag" and returns `True`. A variant is entitled to
that default only if it *reads* the flag — `google_colab` and `kaggle` poll
`_interrupt_requested` and interrupt cooperatively. `datalayer_sandbox` did
neither: no override, no read. So on a Datalayer runtime the interrupt was
delivered nowhere and reported success, and everything above believed it:
`agent_runtimes`' `interrupt_kernel` delegates straight to it, and so does
the interrupt `execute_cell` registers for `tasks/cancel`.

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


class TestEveryVariantCanBeInterrupted:
    def test_datalayer_overrides_it_like_the_others(self):
        """The assertion that would have caught this: it is about the class,
        not about a call, because the default *succeeds* silently."""
        assert "_do_interrupt" in DatalayerSandbox.__dict__

    def test_no_variant_quietly_acknowledges_an_interrupt(self):
        """Held across the package, so the next variant added without one
        fails here rather than on a runtime somebody is paying for.

        A variant may either override `_do_interrupt` or consult the flag the
        base sets; doing neither means `interrupt()` returns True having done
        nothing. `docker` and `monty` do neither today — pinned rather than
        hidden, so the set cannot grow silently and shrinks visibly when one
        is fixed.
        """
        import importlib
        import inspect
        import pkgutil

        import code_sandboxes
        from code_sandboxes.base import Sandbox

        deaf = set()
        for module in pkgutil.iter_modules(code_sandboxes.__path__):
            if not module.name.endswith("_sandbox"):
                continue
            loaded = importlib.import_module(f"code_sandboxes.{module.name}")
            reads_flag = "_interrupt_requested" in inspect.getsource(loaded)
            for name in dir(loaded):
                value = getattr(loaded, name)
                if (
                    isinstance(value, type)
                    and issubclass(value, Sandbox)
                    and value is not Sandbox
                    and value.__module__ == loaded.__name__
                    and "_do_interrupt" not in value.__dict__
                    and not reads_flag
                ):
                    deaf.add(f"{module.name}.{name}")

        known = {"docker_sandbox.DockerSandbox", "monty_sandbox.MontySandbox"}
        assert deaf - known == set(), f"new variants cannot be interrupted: {deaf - known}"
        assert known - deaf == set(), f"fixed — drop from the pinned set: {known - deaf}"


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
