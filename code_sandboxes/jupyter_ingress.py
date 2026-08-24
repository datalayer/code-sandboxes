# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Shared commands for Jupyter Servers exposed through provider ingress."""

from __future__ import annotations

import secrets
import shlex

from .models import JupyterServerOptions


def resolved_options(options: JupyterServerOptions | None) -> JupyterServerOptions:
    """Return options carrying a fresh Jupyter token when none was supplied."""
    value = options or JupyterServerOptions()
    if value.token:
        return value
    return value.model_copy(update={"token": secrets.token_urlsafe(32)})


def preparation_command(options: JupyterServerOptions) -> str:
    """A fast, idempotent install check followed by a background server."""
    packages = "jupyter-server ipykernel"
    check = "python -c 'import jupyter_server, ipykernel'"
    if options.install_if_missing:
        install = (
            "python -m pip install --disable-pip-version-check --quiet " + packages
        )
        prerequisite = f"{check} >/dev/null 2>&1 || {install}"
    else:
        prerequisite = check

    token = shlex.quote(options.token or "")
    launch = " ".join(
        [
            "python -m jupyter_server",
            "--ServerApp.ip=0.0.0.0",
            f"--ServerApp.port={options.port}",
            "--ServerApp.port_retries=0",
            "--ServerApp.open_browser=False",
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.allow_origin='*'",
            "--ServerApp.allow_root=True",
            f"--IdentityProvider.token={token}",
            "--ServerApp.password=''",
        ]
    )
    pid_file = f"/tmp/code-sandboxes-jupyter-{options.port}.pid"
    probe = (
        'python -c "import socket; '
        f"socket.create_connection(('127.0.0.1', {options.port}), 1).close()\""
    )
    # Do not return an ingress URL until the socket can accept connections.
    # This is the preliminary cold-start work; a template can later make the
    # import/install branch disappear without changing this contract.
    return (
        f"set -e; {prerequisite}; "
        f"(nohup {launch} >/tmp/code-sandboxes-jupyter.log 2>&1 & "
        f"echo $! >{pid_file}); "
        f"i=0; until {probe} >/dev/null 2>&1; do "
        "i=$((i + 1)); [ $i -lt 120 ] || { "
        "cat /tmp/code-sandboxes-jupyter.log >&2; exit 1; }; sleep 0.25; done"
    )


def websocket_url(http_url: str) -> str:
    """Translate an ingress HTTP URL to the corresponding WebSocket URL."""
    if http_url.startswith("https://"):
        return "wss://" + http_url.removeprefix("https://")
    if http_url.startswith("http://"):
        return "ws://" + http_url.removeprefix("http://")
    raise ValueError(f"Provider returned a non-HTTP ingress URL: {http_url!r}")
