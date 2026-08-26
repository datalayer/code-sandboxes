# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Reaching a Datalayer deployment, and saying so when it cannot be reached.

Both things pinned here failed in production the same way: something moved in
a package this one talks to, and what the user was told pointed somewhere
else entirely.
"""

from __future__ import annotations

import inspect

import pytest

from code_sandboxes.datalayer_sandbox import DatalayerSandbox, _urls_for_run
from code_sandboxes.exceptions import SandboxConfigurationError
from code_sandboxes.models import SandboxConfig

#: Importing the SDK warns — about its coming move to platformdirs, about
#: pydantic's class-based config. Neither is what these tests are about, and
#: the suite turns warnings into errors.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_every_service_the_sdk_knows_about_points_at_the_one_origin():
    """A run serves all of its services from a single host.

    Read off the SDK rather than from a list written here: the list went
    stale when `mcp_server_url` was renamed, and every execution died on the
    unexpected keyword — a failure with no visible connection to the rename.
    """
    # The SDK arrives with the `datalayer` extra, which an install without it
    # does not have. With no signature to read there is nothing here to check,
    # so this is a skip and not a failure.
    sdk_urls = pytest.importorskip("datalayer_core.utils.urls")

    urls = _urls_for_run("https://prod1.datalayer.run/")

    services = [
        name
        for name in inspect.signature(sdk_urls.DatalayerURLs.from_environment).parameters
        if name.endswith("_url")
    ]
    assert services, "the SDK declares no service URLs; this test is testing nothing"
    for name in services:
        assert getattr(urls, name) == "https://prod1.datalayer.run"


def test_a_backend_that_cannot_be_imported_says_what_actually_failed(monkeypatch):
    """The reason, not the usual reason.

    The message named the missing package and the command that installs it
    whatever the import error said. With the package installed and one name
    moved inside it, that sent the reader to reinstall a dependency that was
    already there.
    """
    import builtins
    import sys
    import types

    # `start()` reaches for the SDK before it reaches for the backend, so in an
    # install without the `datalayer` extra the SDK is what fails first and the
    # failure under test never happens. Stand in for the SDK: its absence is a
    # different fault with its own message, and not the one pinned here.
    for name in ("datalayer_core", "datalayer_core.utils", "datalayer_core.utils.urls"):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("agent_runtimes"):
            raise ImportError("cannot import name 'DEFAULT_TIME_RESERVATION'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    sandbox = DatalayerSandbox(SandboxConfig())
    with pytest.raises(SandboxConfigurationError) as raised:
        sandbox.start()

    message = str(raised.value)
    assert "DEFAULT_TIME_RESERVATION" in message
    # The old advice is still there for the case where it IS the answer.
    assert "code-sandboxes[datalayer]" in message
