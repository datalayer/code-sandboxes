# Copyright (c) 2023-2025 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

"""A sandbox stays fully accessible from a jupyter kernel client.

The kernel-backed sandboxes hold a ``jupyter_kernel_client.JupyterKernelClient``
and expose it whole through ``Sandbox.kernel_client`` — a caller that needs
the low-level kernel API gets the same client the sandbox uses internally.
That only holds while the two packages agree, and they live in different
distributions: a rename or a signature change on either side breaks sandboxes
at RUNTIME unless something breaks EARLIER. These tests are that something.

Three claims, one test class each:

- The real client satisfies ``ISandboxClient``, the contract the sandboxes
  program against — so it can back any kernel-backed variant.
- Every call a sandbox actually makes on its client binds against the real
  client's signatures — the calls are checked, not just the names.
- The sandboxes expose the client (``kernel_client``), and the stand-ins that
  duck-type it (the Kaggle live session) answer the same calls.
"""

from __future__ import annotations

import inspect

import pytest

jupyter_kernel_client = pytest.importorskip("jupyter_kernel_client")

from jupyter_kernel_client import JupyterKernelClient  # noqa: E402
from jupyter_kernel_client.interfaces import IJupyterKernelClient  # noqa: E402

from code_sandboxes.interfaces import ISandboxClient  # noqa: E402


def _protocol_members(protocol: type) -> list[str]:
    """The names a protocol demands, read from its own annotations/methods."""
    return sorted(
        name
        for name in getattr(protocol, "__protocol_attrs__", [])
        if not name.startswith("_")
    )


def _binds(method, *args, **kwargs) -> None:
    """Assert the real method accepts this exact call shape."""
    signature = inspect.signature(method)
    signature.bind(*args, **kwargs)  # raises TypeError when incompatible


class TestTheRealClientSatisfiesTheSandboxContract:
    """`JupyterKernelClient` is an `ISandboxClient`, member by member."""

    def test_every_member_of_the_contract_exists_on_the_client(self):
        missing = [
            name
            for name in _protocol_members(ISandboxClient)
            if not hasattr(JupyterKernelClient, name)
        ]
        assert missing == [], (
            "The jupyter kernel client no longer offers what the sandboxes "
            f"program against: {missing}"
        )

    def test_the_contract_demands_nothing_the_client_does_not_promise(self):
        """`ISandboxClient` stays a subset of the client's PUBLIC protocol.

        The sandboxes must be drivable through the documented client
        interface alone — never through something private the client happens
        to have today.
        """
        public = set(_protocol_members(IJupyterKernelClient))
        overreach = [
            name for name in _protocol_members(ISandboxClient) if name not in public
        ]
        assert overreach == [], (
            "The sandbox contract asks the client for members outside its "
            f"public protocol: {overreach}"
        )

    def test_the_client_satisfies_its_own_public_protocol(self):
        missing = [
            name
            for name in _protocol_members(IJupyterKernelClient)
            if not hasattr(JupyterKernelClient, name)
        ]
        assert missing == []


class TestTheCallsTheSandboxesMakeBind:
    """Each call a sandbox makes on its client fits the real signature.

    The shapes below are the ones found in the sandbox sources — change a
    call there and this list is the reminder to keep them in step.
    """

    def test_execute_with_code_and_timeout(self):
        _binds(JupyterKernelClient.execute, None, "print(1)", timeout=60.0)

    def test_execute_interactive_with_output_hook(self):
        _binds(
            JupyterKernelClient.execute_interactive,
            None,
            "print(1)",
            output_hook=lambda msg: None,
            timeout=None,
        )

    def test_start_bare_and_with_a_path(self):
        _binds(JupyterKernelClient.start, None)
        _binds(JupyterKernelClient.start, None, path="notebooks/analysis")

    def test_stop_bare_and_keeping_the_kernel(self):
        _binds(JupyterKernelClient.stop, None)
        _binds(JupyterKernelClient.stop, None, shutdown_kernel=False)

    def test_variables_by_name(self):
        _binds(JupyterKernelClient.get_variable, None, "x")
        _binds(JupyterKernelClient.set_variable, None, "x", 1)

    def test_interrupt_bare(self):
        _binds(JupyterKernelClient.interrupt, None)

    def test_identity_and_info_are_readable(self):
        assert isinstance(
            inspect.getattr_static(JupyterKernelClient, "id"), property
        )
        assert isinstance(
            inspect.getattr_static(JupyterKernelClient, "kernel_info"), property
        )


class TestTheSandboxesExposeTheClient:
    """`Sandbox.kernel_client` hands the client out, whole."""

    def test_the_base_declares_the_accessor(self):
        from code_sandboxes import Sandbox

        assert isinstance(
            inspect.getattr_static(Sandbox, "kernel_client"), property
        )

    def test_every_kernel_backed_variant_overrides_it(self):
        # From the package root, as any consumer would: the compatibility
        # promised here is the PUBLIC surface, not the module layout.
        from code_sandboxes import (
            GoogleColabSandbox,
            JupyterServerSandbox,
            KaggleSandbox,
            Sandbox,
        )

        base = inspect.getattr_static(Sandbox, "kernel_client")
        for variant in (JupyterServerSandbox, KaggleSandbox, GoogleColabSandbox):
            own = inspect.getattr_static(variant, "kernel_client")
            assert own is not base, f"{variant.__name__} hides its client"

    def test_the_kaggle_live_session_answers_the_same_calls(self):
        """The live session stands in for the client on the interactive path.

        It duck-types what `KaggleSandbox.run_code` uses — `execute`, `stop`,
        `id`, `get_variable` — with call shapes the real client also accepts,
        so the sandbox code cannot tell the two apart.
        """
        from code_sandboxes.kaggle_live import KaggleLiveSession

        _binds(KaggleLiveSession.execute, None, "print(1)", timeout=60.0)
        _binds(KaggleLiveSession.stop, None)
        _binds(KaggleLiveSession.stop, None, shutdown_kernel=False)
        _binds(KaggleLiveSession.get_variable, None, "x")

        session = KaggleLiveSession(executor=object())
        assert isinstance(session.id, str) and session.id
