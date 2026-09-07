# Copyright (c) 2023-2026 Datalayer, Inc.
# Datalayer License

"""Every variant answers for its own interrupt, and can be reached to answer.

`Sandbox.interrupt` is two gates and a delegation::

    if not self._executing_event.is_set():
        return False          # gate one: is anything running?
    self._interrupt_requested.set()
    return self._do_interrupt()   # gate two: can this provider stop it?

A variant has to pass both. Seven did not, and each failure looked like
success from the outside:

* `docker`, `monty` and `google_colab`/`kaggle` had no `_do_interrupt`, so
  the base default ran — set a flag, return `True`, stop nothing. `True`
  means "the interrupt was delivered", and nothing had been.
* `cloudflare`, `coreweave`, `daytona`, `e2b` and `modal` answered honestly
  (`False`, this provider takes no interrupt) but never marked their
  execution window, so `interrupt()` returned at gate one and their answer
  was never reached — and `is_executing` was always False, which every
  status above them believed.

The measurement that made this worth doing: `DatalayerSandbox` had neither,
and adding `_do_interrupt` alone moved prod1 from 60.8 seconds to 60.5 —
a fix that reads as complete and measures as nothing.

Launch the tests:
```
$ pytest tests/test_every_variant_has_a_real_interrupt.py -v
```
"""

from __future__ import annotations

import threading

import pytest

from code_sandboxes.base import Sandbox, marks_execution
from code_sandboxes.jupyter_ingress import interrupt_kernel_client


def variants():
    """Every `Sandbox` subclass the package ships, with the module it is in."""
    import importlib
    import pkgutil

    import code_sandboxes

    for module in pkgutil.iter_modules(code_sandboxes.__path__):
        if not module.name.endswith("_sandbox"):
            continue
        loaded = importlib.import_module(f"code_sandboxes.{module.name}")
        for name in dir(loaded):
            value = getattr(loaded, name)
            if (
                isinstance(value, type)
                and issubclass(value, Sandbox)
                and value is not Sandbox
                and value.__module__ == loaded.__name__
            ):
                yield module.name, name, value


class TestTheWholePackage:
    def test_the_sweep_actually_sees_the_variants(self):
        """A sweep over an empty set passes for the wrong reason."""
        found = {name for _, name, _ in variants()}
        assert len(found) >= 12, f"only found {sorted(found)}"
        assert "DatalayerSandbox" in found

    def test_every_variant_answers_for_itself(self):
        """No exceptions left. The base default is never the right answer,
        because `True` means "delivered" and an inheritor delivers nothing.
        A provider that genuinely cannot be interrupted says so by returning
        `False` from its own override — a different sentence, which lets the
        caller tell "could not be stopped" from "was stopped"."""
        deaf = {
            f"{module}.{name}"
            for module, name, value in variants()
            if "_do_interrupt" not in value.__dict__
        }
        assert deaf == set(), f"these inherit a default that lies: {sorted(deaf)}"

    def test_every_variant_marks_its_execution_window(self):
        """Gate one. Asserted on the decorator rather than a grep for a line
        of source, so a variant that opens the window and forgets to close it
        on one early return cannot pass."""
        mute = {
            f"{module}.{name}"
            for module, name, value in variants()
            if not getattr(value.run_code, "_marks_execution", False)
        }
        assert mute == set(), f"these have an interrupt nothing can reach: {sorted(mute)}"

    def test_an_interrupt_is_never_claimed_by_inheritance(self):
        """The base default still exists for anything outside this package;
        what must not happen is a shipped variant relying on it."""
        assert Sandbox._do_interrupt(None) is True


class TestTheExecutionWindow:
    def _sandbox(self, body):
        class _Sandbox:
            def __init__(self):
                self._executing_event = threading.Event()
                self._interrupt_requested = threading.Event()

            run_code = marks_execution(body)

        return _Sandbox()

    def test_it_is_open_while_the_code_runs(self):
        seen = []
        sandbox = self._sandbox(lambda self: seen.append(self._executing_event.is_set()))
        sandbox.run_code()
        assert seen == [True]

    def test_it_closes_afterwards(self):
        sandbox = self._sandbox(lambda self: None)
        sandbox.run_code()
        assert not sandbox._executing_event.is_set()

    def test_it_closes_even_when_the_run_raises(self):
        """An adapter that raises must not leave the sandbox looking busy:
        every later interrupt would be accepted against work already over."""

        def boom(self):
            raise RuntimeError("the adapter failed")

        sandbox = self._sandbox(boom)
        with pytest.raises(RuntimeError):
            sandbox.run_code()
        assert not sandbox._executing_event.is_set()

    def test_a_stale_request_does_not_stop_the_next_run(self):
        """Cleared on the way in, or the cooperative variants would label a
        fresh run as interrupted because the last one was."""
        sandbox = self._sandbox(lambda self: self._interrupt_requested.is_set())
        sandbox._interrupt_requested.set()
        assert sandbox.run_code() is False

    def test_the_request_survives_the_run_for_the_reader(self):
        """Not cleared on the way out: `google_colab` and `kaggle` read it
        after the call to report the execution as interrupted."""

        def ask(self):
            self._interrupt_requested.set()

        sandbox = self._sandbox(ask)
        sandbox.run_code()
        assert sandbox._interrupt_requested.is_set()


class TestTheSharedKernelInterrupt:
    """`interrupt_kernel_client`, used by docker, colab and kaggle."""

    class _Client:
        def __init__(self, raises=None):
            self.raises, self.calls = raises, 0

        def interrupt(self):
            self.calls += 1
            if self.raises is not None:
                raise self.raises

    def test_it_asks_the_client(self):
        client = self._Client()
        assert interrupt_kernel_client(client, variant="test") is True
        assert client.calls == 1

    def test_no_client_is_not_an_interrupt(self):
        assert interrupt_kernel_client(None, variant="test") is False

    def test_a_raising_client_is_answered_rather_than_propagated(self):
        client = self._Client(raises=RuntimeError("kernel gone"))
        assert interrupt_kernel_client(client, variant="test") is False

    def test_none_from_the_client_is_success(self):
        """`JupyterKernelClient.interrupt` returns `None` and raises on
        failure, so "no exception" is the only signal there is."""
        assert interrupt_kernel_client(self._Client(), variant="test") is True


class TestTheJupyterBackedVariants:
    """docker, colab and kaggle each delegate to the client that runs them."""

    class _Client:
        def __init__(self):
            self.calls = 0

        def interrupt(self):
            self.calls += 1

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("docker_sandbox", "DockerSandbox"),
            ("google_colab_sandbox", "GoogleColabSandbox"),
            ("kaggle_sandbox", "KaggleSandbox"),
        ],
    )
    def test_it_interrupts_through_its_client(self, module, name):
        import importlib

        cls = getattr(importlib.import_module(f"code_sandboxes.{module}"), name)
        sandbox = cls.__new__(cls)
        client = self._Client()
        sandbox._client = client
        sandbox._server_url = "http://localhost:8888"
        sandbox._token = "t"
        assert sandbox._do_interrupt() is True
        assert client.calls == 1, "it answered without asking the kernel"

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("docker_sandbox", "DockerSandbox"),
            ("google_colab_sandbox", "GoogleColabSandbox"),
            ("kaggle_sandbox", "KaggleSandbox"),
        ],
    )
    def test_without_a_client_it_says_no(self, module, name):
        import importlib

        cls = getattr(importlib.import_module(f"code_sandboxes.{module}"), name)
        sandbox = cls.__new__(cls)
        sandbox._client = None
        sandbox._server_url = None
        sandbox._token = None
        assert sandbox._do_interrupt() is False


class TestThePorvidersThatCannotBeInterrupted:
    """Answering `False` is the point: it is a different sentence from `True`.

    These five are not oversights. Cloudflare's bridge, CoreWeave's session
    process, Daytona's and E2B's interpreters and Modal take no interrupt,
    and a timeout is the only stop. What matters is that they say so
    themselves rather than inheriting a default that claims otherwise — and
    that, since they now mark their execution window, the caller reaches that
    answer instead of gate one's identical-looking `False`.
    """

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("cloudflare_sandbox", "CloudflareSandbox"),
            ("coreweave_sandbox", "CoreWeaveSandbox"),
            ("daytona_sandbox", "DaytonaSandbox"),
            ("e2b_sandbox", "E2BSandbox"),
            ("modal_sandbox", "ModalSandbox"),
            ("monty_sandbox", "MontySandbox"),
        ],
    )
    def test_it_refuses_rather_than_pretending(self, module, name):
        import importlib

        cls = getattr(importlib.import_module(f"code_sandboxes.{module}"), name)
        assert cls._do_interrupt(cls.__new__(cls)) is False
