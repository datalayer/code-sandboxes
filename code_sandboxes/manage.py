# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""CRUD management of sandboxes, per variant.

``Sandbox.create`` starts a sandbox to execute code in it; this module is the
other side of a sandbox's life: enumerating the ones that exist, reading one
back, and deleting it — without connecting a kernel to it. Every variant gets
a manager with the same four verbs:

* ``create(**kwargs)`` — bring a sandbox into existence and leave it running
  (detached from this process where the backend allows it).
* ``list()`` — the sandboxes that exist right now, as :class:`SandboxInfo`.
* ``get(sandbox_id)`` — one of them, or ``None``.
* ``update(sandbox_id, **changes)`` — change what the backend can change:
  the container's name (docker), the tags (modal), the capabilities
  (datalayer), the code (kaggle, pushed as a new version).
* ``delete(sandbox_id)`` — remove it; ``True`` when something was deleted.

Not every backend can honour every verb — an ``eval`` sandbox lives and dies
inside this process, a Kaggle batch kernel is created by pushing code. A
manager states what it can do through :attr:`SandboxManager.capabilities`
and raises :class:`SandboxManagementError` for the rest, with the reason in
the message rather than a silent no-op.

Example::

    from code_sandboxes.manage import get_manager

    manager = get_manager("modal")
    for info in manager.list():
        print(info.id, info.status)
    manager.delete("sb-...")
"""

from __future__ import annotations

import contextlib
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from .models import SandboxInfo, SandboxStatus

__all__ = [
    "SandboxManagementError",
    "SandboxManager",
    "get_manager",
    "manageable_variants",
]


class SandboxManagementError(RuntimeError):
    """A management verb the backend cannot honour, with the reason."""


class SandboxManager(ABC):
    """The four CRUD verbs over one sandbox variant."""

    #: The variant this manager speaks for.
    variant: str = ""

    #: Which verbs the backend honours, e.g. ``{"create", "list", "get",
    #: "delete"}``. A verb outside this set raises
    #: :class:`SandboxManagementError` when called.
    capabilities: frozenset[str] = frozenset()

    @abstractmethod
    def list(self) -> list[SandboxInfo]:
        """The sandboxes that exist right now."""

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        """One sandbox, or ``None`` when it does not exist."""
        for info in self.list():
            if info.id == sandbox_id:
                return info
        return None

    @abstractmethod
    def delete(self, sandbox_id: str) -> bool:
        """Remove a sandbox. ``True`` when something was deleted."""

    @abstractmethod
    def create(self, **kwargs: Any) -> SandboxInfo:
        """Bring a sandbox into existence and leave it running."""

    def update(self, sandbox_id: str, **changes: Any) -> SandboxInfo:
        """Change what the backend can change; the sandbox as it now is."""
        raise self._unsupported(
            "update", "this backend has nothing that can be changed in place"
        )

    def _unsupported(self, verb: str, reason: str) -> SandboxManagementError:
        return SandboxManagementError(
            f"The {self.variant} variant does not support {verb}: {reason}"
        )


class _EphemeralManager(SandboxManager):
    """The in-process variants: nothing outlives the interpreter.

    An ``eval`` or ``monty`` sandbox is a Python object of the process that
    made it; there is nothing to enumerate from outside, nothing to delete.
    ``list`` answers the truth — an empty list — and the other verbs say why.
    """

    capabilities = frozenset({"list"})
    _reason = "it runs inside the creating process and dies with it"

    def list(self) -> list[SandboxInfo]:
        return []

    def delete(self, sandbox_id: str) -> bool:
        raise self._unsupported("delete", self._reason)

    def create(self, **kwargs: Any) -> SandboxInfo:
        raise self._unsupported(
            "detached create", self._reason + "; use Sandbox.create() instead"
        )

    def update(self, sandbox_id: str, **changes: Any) -> SandboxInfo:
        raise self._unsupported("update", self._reason)


class EvalSandboxManager(_EphemeralManager):
    variant = "eval"


class MontySandboxManager(_EphemeralManager):
    variant = "monty"


class DockerSandboxManager(SandboxManager):
    """Docker containers labelled as code sandboxes.

    Containers are matched by the ``code-sandboxes`` label that
    ``DockerSandbox`` stamps at start; older containers without the label are
    found by their ``code-sandboxes-jupyter`` image as a fallback.
    """

    variant = "docker"
    capabilities = frozenset({"create", "list", "get", "update", "delete"})

    #: The label DockerSandbox stamps on every container it starts.
    LABEL = "code-sandboxes"

    def __init__(self, docker_client: Any = None, **_: Any) -> None:
        self._docker = docker_client

    def _client(self) -> Any:
        if self._docker is None:
            try:
                import docker
            except ImportError as exc:
                raise SandboxManagementError(
                    "docker package is required: pip install code-sandboxes[docker]"
                ) from exc
            self._docker = docker.from_env()
        return self._docker

    def _containers(self) -> list[Any]:
        client = self._client()
        labelled = client.containers.list(
            all=True, filters={"label": self.LABEL}
        )
        seen = {c.id for c in labelled}
        # Containers from before the label existed: found by their image.
        for container in client.containers.list(all=True):
            if container.id in seen:
                continue
            image_tags = getattr(container.image, "tags", None) or []
            if any("code-sandboxes" in tag for tag in image_tags):
                labelled.append(container)
        return labelled

    @staticmethod
    def _status(container: Any) -> SandboxStatus:
        return {
            "running": SandboxStatus.RUNNING,
            "created": SandboxStatus.STARTING,
            "restarting": SandboxStatus.STARTING,
            "paused": SandboxStatus.STOPPED,
            "exited": SandboxStatus.STOPPED,
            "dead": SandboxStatus.ERROR,
        }.get(getattr(container, "status", ""), SandboxStatus.RUNNING)

    def _info(self, container: Any) -> SandboxInfo:
        attrs = getattr(container, "attrs", {}) or {}
        image_tags = getattr(container.image, "tags", None) or []
        return SandboxInfo(
            id=container.id[:12],
            variant=self.variant,
            status=self._status(container),
            name=container.name,
            metadata={
                "image": image_tags[0] if image_tags else "",
                "created": attrs.get("Created", ""),
            },
        )

    def list(self) -> list[SandboxInfo]:
        return [self._info(c) for c in self._containers()]

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        for container in self._containers():
            if container.id.startswith(sandbox_id) or container.name == sandbox_id:
                return self._info(container)
        return None

    def delete(self, sandbox_id: str) -> bool:
        for container in self._containers():
            if container.id.startswith(sandbox_id) or container.name == sandbox_id:
                container.remove(force=True)
                return True
        return False

    def update(self, sandbox_id: str, name: str | None = None, **_: Any) -> SandboxInfo:
        """Rename the container — the one thing Docker changes in place."""
        if not name:
            raise self._unsupported("update without name=...", "only the name changes")
        for container in self._containers():
            if container.id.startswith(sandbox_id) or container.name == sandbox_id:
                container.rename(name)
                container.reload()
                return self._info(container)
        raise SandboxManagementError(f"No docker sandbox found: {sandbox_id}")

    def create(self, **kwargs: Any) -> SandboxInfo:
        from .docker_sandbox import DockerSandbox

        # auto_remove would erase the container the moment this process lets
        # go of it — the opposite of a detached create.
        sandbox = DockerSandbox(auto_remove=False, **kwargs)
        sandbox.start()
        info = sandbox.info
        if info is None:
            raise SandboxManagementError("The container started without an identity.")
        return info


class JupyterServerSandboxManager(SandboxManager):
    """Kernels of a Jupyter Server, spoken to over its REST API.

    A ``jupyter`` sandbox started with ``server_url`` lives on that server as
    a kernel; this manager enumerates and deletes those kernels. The server
    defaults to ``JUPYTER_SERVER_URL``/``JUPYTER_TOKEN`` from the
    environment, then to ``http://localhost:8888``.
    """

    variant = "jupyter-server"
    capabilities = frozenset({"create", "list", "get", "delete"})

    def __init__(
        self,
        server_url: str | None = None,
        token: str | None = None,
        **_: Any,
    ) -> None:
        self._server_url = (
            server_url
            or os.environ.get("JUPYTER_SERVER_URL")
            or "http://localhost:8888"
        ).rstrip("/")
        self._token = token if token is not None else os.environ.get("JUPYTER_TOKEN")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        import requests

        headers = kwargs.pop("headers", {})
        if self._token:
            headers["Authorization"] = f"token {self._token}"
        try:
            response = requests.request(
                method,
                f"{self._server_url}{path}",
                headers=headers,
                timeout=10,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise SandboxManagementError(
                f"No Jupyter Server answered at {self._server_url}: {exc}"
            ) from exc
        if response.status_code == 403:
            raise SandboxManagementError(
                f"The Jupyter Server at {self._server_url} refused the token; "
                "set JUPYTER_TOKEN or pass token=..."
            )
        return response

    @staticmethod
    def _info(kernel: dict) -> SandboxInfo:
        return SandboxInfo(
            id=kernel.get("id", ""),
            variant="jupyter-server",
            status=(
                SandboxStatus.RUNNING
                if kernel.get("execution_state") != "dead"
                else SandboxStatus.ERROR
            ),
            name=kernel.get("name", ""),
            metadata={
                "execution_state": kernel.get("execution_state", ""),
                "last_activity": kernel.get("last_activity", ""),
                "connections": kernel.get("connections", 0),
            },
        )

    def list(self) -> list[SandboxInfo]:
        response = self._request("GET", "/api/kernels")
        response.raise_for_status()
        return [self._info(k) for k in response.json()]

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        response = self._request("GET", f"/api/kernels/{sandbox_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return self._info(response.json())

    def delete(self, sandbox_id: str) -> bool:
        response = self._request("DELETE", f"/api/kernels/{sandbox_id}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def create(self, kernel_name: str | None = None, **_: Any) -> SandboxInfo:
        payload = {"name": kernel_name} if kernel_name else {}
        response = self._request("POST", "/api/kernels", json=payload)
        response.raise_for_status()
        return self._info(response.json())


class GoogleColabSandboxManager(JupyterServerSandboxManager):
    """Kernels of a Colab runtime, over the same Jupyter REST API.

    The Colab proxy authenticates with its own headers instead of a Jupyter
    token; the runtime URL and proxy token come from ``RUNTIME_URL`` and
    ``RUNTIME_PROXY_TOKEN`` when not passed explicitly.
    """

    variant = "google_colab"

    def __init__(
        self,
        server_url: str | None = None,
        proxy_token: str | None = None,
        **_: Any,
    ) -> None:
        self._colab_url = server_url or os.environ.get("RUNTIME_URL")
        super().__init__(server_url=self._colab_url or "http://unset.invalid", token=None)
        self._proxy_token = proxy_token or os.environ.get("RUNTIME_PROXY_TOKEN")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        # Checked per verb, not at construction: a manager must be buildable
        # to answer what it cannot do.
        if not self._colab_url:
            raise SandboxManagementError(
                "A Colab runtime URL is required: pass server_url=... or set "
                "RUNTIME_URL."
            )
        from .google_colab import (
            COLAB_CLIENT_AGENT_HEADER,
            COLAB_RUNTIME_PROXY_TOKEN_HEADER,
            DEFAULT_COLAB_CLIENT_AGENT,
        )

        headers = kwargs.pop("headers", {})
        headers[COLAB_CLIENT_AGENT_HEADER] = DEFAULT_COLAB_CLIENT_AGENT
        if self._proxy_token:
            headers[COLAB_RUNTIME_PROXY_TOKEN_HEADER] = self._proxy_token
        return super()._request(method, path, headers=headers, **kwargs)

    def _info(self, kernel: dict) -> SandboxInfo:  # type: ignore[override]
        info = super()._info(kernel)
        info.variant = self.variant
        return info


class KaggleSandboxManager(SandboxManager):
    """The user's Kaggle kernels, through the official ``kaggle`` package.

    A batch-mode Kaggle sandbox materialises as a kernel on kaggle.com;
    ``list`` enumerates the user's kernels, ``get`` adds the live run status,
    ``delete`` removes the kernel. ``create`` pushes a batch kernel with the
    given ``code`` — creation on Kaggle *is* a code push.
    """

    variant = "kaggle"
    capabilities = frozenset({"create", "list", "get", "update", "delete"})

    def __init__(self, username: str | None = None, **_: Any) -> None:
        self._username = username
        self._executor: Any = None

    def _get_executor(self) -> Any:
        if self._executor is None:
            from .kaggle_execute import KaggleKernelExecutor

            self._executor = KaggleKernelExecutor(username=self._username)
        return self._executor

    @staticmethod
    def _info(kernel: Any) -> SandboxInfo:
        ref = getattr(kernel, "ref", "") or ""
        return SandboxInfo(
            id=ref,
            variant="kaggle",
            # The list endpoint does not carry the run state; get() does.
            status=SandboxStatus.STOPPED,
            name=getattr(kernel, "title", "") or ref.rsplit("/", 1)[-1],
            metadata={
                "last_run": str(getattr(kernel, "last_run_time", "") or ""),
                "url": f"https://www.kaggle.com/code/{ref}" if ref else "",
            },
        )

    def list(self) -> list[SandboxInfo]:
        executor = self._get_executor()
        kernels = executor.api.kernels_list(mine=True, page_size=50) or []
        return [self._info(k) for k in kernels if k is not None]

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        executor = self._get_executor()
        ref = self._qualify(sandbox_id)
        for info in self.list():
            if info.id == ref:
                # A kernel whose status endpoint fails is still a kernel; the
                # listing answers without the live state.
                with contextlib.suppress(Exception):
                    status = executor.api.kernels_status(ref)
                    state = str(getattr(status, "status", "") or "")
                    info.metadata["run_status"] = state
                    if "RUNNING" in state or "QUEUED" in state:
                        info.status = SandboxStatus.RUNNING
                    elif "ERROR" in state:
                        info.status = SandboxStatus.ERROR
                return info
        return None

    def delete(self, sandbox_id: str) -> bool:
        executor = self._get_executor()
        ref = self._qualify(sandbox_id)
        if not any(info.id == ref for info in self.list()):
            return False
        executor.api.kernels_delete(ref, no_confirm=True)
        return True

    def create(self, code: str = "print('code-sandboxes')", **kwargs: Any) -> SandboxInfo:
        executor = self._get_executor()
        submitted = executor.execute(code, wait=False, download_output=False, **kwargs)
        slug = getattr(submitted, "slug", "")
        return SandboxInfo(
            id=slug,
            variant="kaggle",
            status=SandboxStatus.RUNNING,
            name=slug.rsplit("/", 1)[-1],
            created_at=time.time(),
            metadata={"url": f"https://www.kaggle.com/code/{slug}" if slug else ""},
        )

    def update(self, sandbox_id: str, code: str | None = None, **_: Any) -> SandboxInfo:
        """Push a new version of the kernel's code — how Kaggle updates.

        The kernel's own metadata is pulled and pushed back unchanged; only
        the code file is replaced. The title stays as it is: it is what the
        kernel's slug — its identity — is derived from.
        """
        if code is None:
            raise self._unsupported(
                "update without code=...", "a Kaggle kernel is updated by pushing code"
            )
        import json
        import tempfile
        from pathlib import Path

        executor = self._get_executor()
        ref = self._qualify(sandbox_id)
        folder = Path(tempfile.mkdtemp(prefix="code-sandboxes-kaggle-update-"))
        try:
            executor.api.kernels_pull(ref, str(folder), metadata=True)
        except Exception as exc:
            raise SandboxManagementError(
                f"No kaggle sandbox found: {sandbox_id} ({exc})"
            ) from exc
        metadata = json.loads((folder / "kernel-metadata.json").read_text())
        code_file = folder / (metadata.get("code_file") or "kernel.py")
        if code_file.suffix == ".ipynb":
            code_file.write_text(
                json.dumps(
                    {
                        "nbformat": 4,
                        "nbformat_minor": 5,
                        "metadata": {},
                        "cells": [
                            {
                                "cell_type": "code",
                                "metadata": {},
                                "execution_count": None,
                                "outputs": [],
                                "source": code,
                            }
                        ],
                    }
                )
            )
        else:
            code_file.write_text(code)
        response = executor.api.kernels_push(str(folder))
        error = getattr(response, "error", "") or ""
        if error:
            raise SandboxManagementError(f"Kaggle refused the update: {error}")
        info = self.get(ref)
        if info is None:
            raise SandboxManagementError(
                f"The kernel disappeared while updating: {ref}"
            )
        info.metadata["version"] = getattr(response, "version_number", "")
        return info

    def _qualify(self, sandbox_id: str) -> str:
        """Accept both ``user/slug`` and a bare slug of the current user."""
        if "/" in sandbox_id:
            return sandbox_id
        executor = self._get_executor()
        return f"{executor._resolve_username()}/{sandbox_id}"


class ModalSandboxManager(SandboxManager):
    """Modal sandboxes of the ``code-sandboxes`` app."""

    variant = "modal"
    capabilities = frozenset({"create", "list", "get", "update", "delete"})

    def __init__(self, app_name: str | None = None, **_: Any) -> None:
        from .modal_sandbox import DEFAULT_APP_NAME

        self._app_name = app_name or DEFAULT_APP_NAME

    def _modal(self) -> Any:
        try:
            import modal
        except ImportError as exc:
            raise SandboxManagementError(
                "modal package is required: pip install code-sandboxes[modal]"
            ) from exc
        return modal

    def list(self) -> list[SandboxInfo]:
        modal = self._modal()
        try:
            app = modal.App.lookup(self._app_name)
        except Exception:
            return []
        infos = []
        for sandbox in modal.Sandbox.list(app_id=app.app_id):
            infos.append(
                SandboxInfo(
                    id=sandbox.object_id,
                    variant=self.variant,
                    status=SandboxStatus.RUNNING,
                    name=self._app_name,
                    metadata={"app": self._app_name},
                )
            )
        return infos

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        modal = self._modal()
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception:
            return None
        finished = sandbox.poll()
        return SandboxInfo(
            id=sandbox.object_id,
            variant=self.variant,
            status=SandboxStatus.STOPPED if finished is not None else SandboxStatus.RUNNING,
            name=self._app_name,
            metadata={"returncode": finished},
        )

    def delete(self, sandbox_id: str) -> bool:
        modal = self._modal()
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception:
            return False
        sandbox.terminate()
        return True

    def update(
        self, sandbox_id: str, tags: dict[str, str] | None = None, **_: Any
    ) -> SandboxInfo:
        """Set tags on the sandbox — what Modal changes on a running one."""
        if not tags:
            raise self._unsupported("update without tags=...", "only tags change")
        modal = self._modal()
        try:
            sandbox = modal.Sandbox.from_id(sandbox_id)
        except Exception as exc:
            raise SandboxManagementError(
                f"No modal sandbox found: {sandbox_id}"
            ) from exc
        sandbox.set_tags(tags)
        info = self.get(sandbox_id)
        if info is None:
            raise SandboxManagementError(f"No modal sandbox found: {sandbox_id}")
        info.metadata["tags"] = tags
        return info

    def create(self, **kwargs: Any) -> SandboxInfo:
        from .modal_sandbox import ModalSandbox

        sandbox = ModalSandbox(app_name=self._app_name, **kwargs)
        sandbox.start()
        info = sandbox.info
        if info is None:
            raise SandboxManagementError("The sandbox started without an identity.")
        # The id every other verb answers to is Modal's, not the wrapper's
        # internal one.
        modal_sandbox = sandbox._sandbox
        if modal_sandbox is not None:
            info.id = modal_sandbox.object_id
        # Detach: the Modal sandbox keeps running server-side until its
        # timeout; stop() would terminate it.
        sandbox._sandbox = None
        sandbox._started = False
        return info


class DatalayerSandboxManager(SandboxManager):
    """Runtimes of the Datalayer platform, through ``agent_runtimes``."""

    variant = "datalayer"
    capabilities = frozenset({"create", "list", "get", "update", "delete"})

    def __init__(
        self,
        token: str | None = None,
        run_url: str | None = None,
        **_: Any,
    ) -> None:
        self._token = token
        self._run_url = run_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from agent_runtimes.client import AgentClient
            except ImportError as exc:
                raise SandboxManagementError(
                    "agent_runtimes package is required: "
                    "pip install code-sandboxes[datalayer]"
                ) from exc
            if self._run_url:
                from .datalayer_sandbox import _urls_for_run

                self._client = AgentClient(
                    urls=_urls_for_run(self._run_url), api_key=self._token
                )
            else:
                self._client = AgentClient(api_key=self._token)
        return self._client

    @staticmethod
    def _info(runtime: Any) -> SandboxInfo:
        return SandboxInfo(
            id=getattr(runtime, "uid", "") or "",
            variant="datalayer",
            status=SandboxStatus.RUNNING,
            name=getattr(runtime, "name", "") or "",
            metadata={
                "environment": getattr(runtime, "environment_name", "") or "",
                "pod": getattr(runtime, "pod_name", "") or "",
            },
        )

    def list(self) -> list[SandboxInfo]:
        client = self._get_client()
        return [self._info(r) for r in client.list_runtimes()]

    def get(self, sandbox_id: str) -> SandboxInfo | None:
        client = self._get_client()
        try:
            runtime = client.get_runtime(sandbox_id)
        except Exception:
            return None
        return self._info(runtime) if runtime else None

    def delete(self, sandbox_id: str) -> bool:
        client = self._get_client()
        try:
            client.terminate_runtime(sandbox_id)
        except Exception:
            return False
        return True

    def update(
        self, sandbox_id: str, capabilities: list[str] | None = None, **_: Any
    ) -> SandboxInfo:
        """Update the runtime's capabilities — what the platform changes."""
        if capabilities is None:
            raise self._unsupported(
                "update without capabilities=...", "only the capabilities change"
            )
        client = self._get_client()
        if not client.update_runtime(sandbox_id, capabilities):
            raise SandboxManagementError(
                f"The platform refused the update of runtime {sandbox_id}."
            )
        info = self.get(sandbox_id)
        if info is None:
            raise SandboxManagementError(f"No datalayer sandbox found: {sandbox_id}")
        info.metadata["capabilities"] = ", ".join(capabilities)
        return info

    def create(self, **kwargs: Any) -> SandboxInfo:
        from .datalayer_sandbox import DatalayerSandbox

        sandbox = DatalayerSandbox(token=self._token, run_url=self._run_url, **kwargs)
        sandbox.start()
        info = sandbox.info
        if info is None:
            raise SandboxManagementError("The runtime started without an identity.")
        return info


_MANAGERS: dict[str, type[SandboxManager]] = {
    "eval": EvalSandboxManager,
    "monty": MontySandboxManager,
    "docker": DockerSandboxManager,
    "jupyter-server": JupyterServerSandboxManager,
    "google_colab": GoogleColabSandboxManager,
    "kaggle": KaggleSandboxManager,
    "modal": ModalSandboxManager,
    "datalayer": DatalayerSandboxManager,
}


def manageable_variants() -> list[str]:
    """The variants a manager exists for."""
    return sorted(_MANAGERS)


def get_manager(variant: str, **kwargs: Any) -> SandboxManager:
    """The manager for a variant.

    Args:
        variant: One of :func:`manageable_variants` (``google-colab`` is
            accepted for ``google_colab``).
        **kwargs: Variant-specific connection settings — ``server_url`` /
            ``token`` (jupyter), ``proxy_token`` (google_colab), ``app_name``
            (modal), ``username`` (kaggle), ``token`` / ``run_url``
            (datalayer), ``docker_client`` (docker).

    Returns:
        A :class:`SandboxManager` for the variant.

    Raises:
        ValueError: For an unknown variant.
    """
    normalized = variant.strip().lower().replace("-", "_")
    # Keys carry the variant's own spelling — `jupyter-server` with its dash —
    # and lookups arrive in either form: compare in one normal form.
    manager_class = next(
        (
            cls
            for key, cls in _MANAGERS.items()
            if key.replace("-", "_") == normalized
        ),
        None,
    )
    if manager_class is None:
        raise ValueError(
            f"Unknown sandbox variant: {variant}. "
            "Manageable variants: " + ", ".join(manageable_variants())
        )
    return manager_class(**kwargs)
