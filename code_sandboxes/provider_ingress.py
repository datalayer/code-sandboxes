# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Execute through a real Jupyter Server exposed by provider ingress."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from .base import Sandbox
from .jupyter_server_sandbox import JupyterServerSandbox
from .models import JupyterServerOptions


@contextlib.contextmanager
def provider_ingress_execution(
    provider_sandbox: Sandbox,
    *,
    direct: bool = False,
    options: JupyterServerOptions | None = None,
) -> Iterator[Sandbox]:
    """Yield the execution sandbox for one cloud-provider container.

    The default prepares a real Jupyter Server inside the already-started
    Daytona, E2B, or Modal sandbox and connects a Jupyter kernel client through
    the provider's authenticated ingress. ``direct=True`` yields the provider
    SDK execution adapter unchanged.

    The caller continues to own ``provider_sandbox`` and must stop it. The
    Jupyter client created here is always closed before returning.
    """
    if direct:
        yield provider_sandbox
        return

    endpoint = provider_sandbox.prepare_jupyter_server(options)
    jupyter = JupyterServerSandbox(
        config=provider_sandbox.config,
        server_url=endpoint.http_url,
        token=endpoint.query.get("token"),
        headers=endpoint.headers,
        reuse_kernel=False,
    )
    jupyter.start()
    try:
        yield jupyter
    finally:
        jupyter.stop()
