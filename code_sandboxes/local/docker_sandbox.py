# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Local Docker-based sandbox implementation.

This sandbox runs a Jupyter Server inside a Docker container and connects
through `jupyter-kernel-client` to execute code.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional
import uuid

import requests

from ..base import Sandbox
from ..exceptions import SandboxConfigurationError, SandboxNotStartedError
from ..models import (
    Context,
    Execution,
    ExecutionError,
    Logs,
    OutputHandler,
    OutputMessage,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
)

DEFAULT_IMAGE = "code-sandboxes-jupyter:latest"
DEFAULT_PORT = 8888


class LocalDockerSandbox(Sandbox):
    """Docker container sandbox using a Jupyter Server backend."""

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        image: Optional[str] = None,
        token: Optional[str] = None,
        host: str = "127.0.0.1",
        container_port: int = DEFAULT_PORT,
        container_name: Optional[str] = None,
        docker_client=None,
        auto_remove: bool = True,
        workdir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(config)
        self._image = image or DEFAULT_IMAGE
        self._token = token or uuid.uuid4().hex
        self._host = host
        self._container_port = container_port
        self._container_name = container_name
        self._docker = docker_client
        self._auto_remove = auto_remove
        self._container = None
        self._client = None
        self._sandbox_id = str(uuid.uuid4())
        self._workdir = workdir
        self._workdir_tmp: Optional[str] = None
        self._server_url: Optional[str] = None
        self._extra_kwargs = kwargs

    @classmethod
    def list_environments(cls) -> list[SandboxEnvironment]:
        return [
            SandboxEnvironment(
                name="local-docker",
                title="Local Docker (Jupyter)",
                language="python",
                owner="local",
                visibility="local",
                burning_rate=0.0,
                metadata={"variant": "local-docker", "image": DEFAULT_IMAGE},
            )
        ]

    def _ensure_docker(self):
        if self._docker is not None:
            return
        try:
            import docker  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SandboxConfigurationError(
                "docker package is required for LocalDockerSandbox. "
                "Install it with: pip install code-sandboxes[docker]"
            ) from exc
        self._docker = docker.from_env()

    def _resolve_workdir(self) -> str:
        if self._workdir:
            return self._workdir
        self._workdir_tmp = tempfile.mkdtemp(prefix="code-sandbox-")
        return self._workdir_tmp

    def _wait_for_server(self, timeout: float = 30.0) -> None:
        if not self._server_url:
            raise SandboxConfigurationError("Server URL not available")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = requests.get(
                    f"{self._server_url}/api/status",
                    params={"token": self._token},
                    timeout=2,
                )
                if response.ok:
                    return
            except Exception:
                time.sleep(0.5)
        raise SandboxConfigurationError("Timed out waiting for Jupyter Server")

    def start(self) -> None:
        if self._started:
            return

        self._ensure_docker()
        try:
            from jupyter_kernel_client import KernelClient
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SandboxConfigurationError(
                "jupyter-kernel-client is required for LocalDockerSandbox. "
                "Install it with: pip install code-sandboxes"
            ) from exc

        workdir = self._resolve_workdir()
        env = {"JUPYTER_TOKEN": self._token}
        env.update(self.config.env_vars)

        mem_limit = self.config.memory_limit
        nano_cpus = None
        if self.config.cpu_limit:
            nano_cpus = int(self.config.cpu_limit * 1_000_000_000)

        ports = {f"{self._container_port}/tcp": (self._host, None)}

        self._container = self._docker.containers.run(
            self._image,
            detach=True,
            environment=env,
            ports=ports,
            volumes={workdir: {"bind": "/workspace", "mode": "rw"}},
            name=self._container_name,
            auto_remove=self._auto_remove,
            mem_limit=mem_limit,
            nano_cpus=nano_cpus,
            **self._extra_kwargs,
        )

        self._container.reload()
        port_info = self._container.attrs["NetworkSettings"]["Ports"].get(
            f"{self._container_port}/tcp"
        )
        if not port_info:
            raise SandboxConfigurationError("Failed to expose Jupyter Server port")

        host_port = port_info[0]["HostPort"]
        self._server_url = f"http://{self._host}:{host_port}"

        self._wait_for_server(timeout=self.config.timeout or 30.0)

        self._client = KernelClient(server_url=self._server_url, token=self._token)
        self._client.start()

        self._default_context = self.create_context("default")
        self._info = SandboxInfo(
            id=self._sandbox_id,
            variant="local-docker",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=self.config.name,
            metadata={
                "image": self._image,
                "server_url": self._server_url,
                "container_id": self._container.id if self._container else None,
            },
            config=self.config,
        )
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return

        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None

        if self._container is not None:
            try:
                self._container.stop()
            except Exception:
                pass
            self._container = None

        if self._workdir_tmp and os.path.isdir(self._workdir_tmp):
            try:
                import shutil

                shutil.rmtree(self._workdir_tmp, ignore_errors=True)
            except Exception:
                pass
            self._workdir_tmp = None

        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[ExecutionError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Execution:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()

        if language != "python":
            raise ValueError(f"LocalDockerSandbox only supports Python, got: {language}")

        if envs:
            env_code = "\n".join(
                f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items()
            )
            code = f"{env_code}\n{code}"

        reply = self._client.execute(code, timeout=timeout or self.config.timeout)

        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        error: Optional[ExecutionError] = None

        current_time = time.time()
        for output in reply.get("outputs", []):
            output_type = output.get("output_type")
            if output_type == "stream":
                name = output.get("name")
                text = output.get("text", "")
                for line in text.splitlines():
                    msg = OutputMessage(line=line, timestamp=current_time, error=name == "stderr")
                    if name == "stderr":
                        stderr_messages.append(msg)
                        if on_stderr:
                            on_stderr(msg)
                    else:
                        stdout_messages.append(msg)
                        if on_stdout:
                            on_stdout(msg)
            elif output_type in ("execute_result", "display_data"):
                result = Result(
                    data=output.get("data", {}),
                    is_main_result=output_type == "execute_result",
                    extra=output.get("metadata", {}),
                )
                results.append(result)
                if on_result:
                    on_result(result)
            elif output_type == "error":
                error = ExecutionError(
                    name=output.get("ename", "Error"),
                    value=output.get("evalue", ""),
                    traceback="\n".join(output.get("traceback", [])),
                )
                if on_error:
                    on_error(error)

        return Execution(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            error=error,
            execution_count=reply.get("execution_count", 0),
            context_id=context.id if context else "default",
        )

    def _get_internal_variable(self, name: str, context: Optional[Context] = None):
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        return self._client.get_variable(name)

    def _set_internal_variable(
        self, name: str, value, context: Optional[Context] = None
    ) -> None:
        if not self._started or self._client is None:
            raise SandboxNotStartedError()
        self._client.set_variable(name, value)
