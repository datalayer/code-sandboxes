# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Datalayer Runtime-based sandbox implementation.

This sandbox uses the Datalayer platform for cloud-based code execution,
providing full isolation and scalable compute resources.
"""

import time
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .filesystem import SandboxFileHandle

from .base import Sandbox
from .contents import (
    FILESYSTEM_PRIMITIVES,
    LOCAL_BRIDGE_MOUNT,
    MOUNT_MISSING,
    MOUNT_PATH_MISSING,
    ContentAttachmentSpec,
    ContentCapabilities,
    LocalBridgeCapability,
    PreparedAttachment,
    not_ready,
    path_exists,
    path_is_mountpoint,
    ready,
)
from .exceptions import (
    SandboxConfigurationError,
    SandboxConnectionError,
    SandboxNotFoundError,
    SandboxNotStartedError,
    SandboxSnapshotError,
)
from .models import (
    CodeError,
    Context,
    ExecutionResult,
    Logs,
    OutputHandler,
    OutputMessage,
    ResourceConfig,
    Result,
    SandboxConfig,
    SandboxEnvironment,
    SandboxInfo,
    SandboxStatus,
    SnapshotInfo,
)


def _urls_for_run(run_url: str):
    """Datalayer service URLs for a deployment served from one origin.

    A run addresses every service under a single host, each on its own path
    prefix, so one URL is enough to reach all of them. `DatalayerURLs` has no
    constructor for that shape — it takes the services one by one — so they are
    filled in here rather than in the SDK.

    Which services those are is read off the SDK rather than written down
    here. A list copied from it goes stale the moment a URL is renamed there,
    and it went stale exactly that way: `mcp_server_url` became
    `jupyter_mcp_server_url` and every execution died on the unexpected
    keyword, far from the rename that caused it. Asking the signature means a
    new service is picked up for free and a renamed one cannot break this.
    """
    import inspect

    from datalayer_core.utils.urls import DatalayerURLs

    base = (run_url or "").rstrip("/")
    services = [
        name
        for name in inspect.signature(DatalayerURLs.from_environment).parameters
        if name.endswith("_url")
    ]
    return DatalayerURLs.from_environment(**dict.fromkeys(services, base))


class DatalayerSandbox(Sandbox):
    """A sandbox using Datalayer Runtime for cloud-based code execution.

    This sandbox provides full isolation, scalable compute (CPU/GPU),
    and supports snapshots for state persistence.

    Example:
        from code_sandboxes import Sandbox

        with Sandbox.create(timeout=60) as sandbox:
            result = sandbox.run_code("print('Hello!')")
            files = sandbox.files.list("/")

        with Sandbox.create(gpu="T4", environment="python-gpu-env") as sandbox:
            sandbox.run_code("import torch; print(torch.cuda.is_available())")

        # With explicit API key
        with Sandbox.create(token="your-api-key") as sandbox:
            sandbox.run_code("x = 1 + 1")

    Attributes:
        client: The Datalayer client instance.
        runtime: The runtime service instance.
    """

    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
        snapshot_name: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Datalayer sandbox.

        Args:
            config: Sandbox configuration.
            token: Datalayer API token. If not provided, uses DATALAYER_API_KEY
                environment variable.
            run_url: Datalayer server URL. If not provided, uses default.
            snapshot_name: Name of snapshot to restore from (optional).
            **kwargs: Additional arguments passed to DatalayerClient.
        """
        super().__init__(config)
        self._token = token
        self._run_url = run_url
        self._snapshot_name = snapshot_name
        self._client = None
        self._runtime = None
        self._sandbox_id = str(uuid.uuid4())
        self._extra_kwargs = kwargs
        # Jupyter's In[n], kept here because the runtime does not report
        # one: a sandbox serves a single client, so its own counter is the
        # session's truth. Without it every cell shows 0, which the read
        # tools render as N/A — an executed cell that looks never-run.
        self._execution_count = 0
        self._end_at: Optional[float] = None

    @property
    def server_url(self) -> Optional[str]:
        """The runtime's Jupyter ingress, once it is running.

        Published because a sandbox is not only something the agent executes
        in — it is a kernel a *person* may want to look at. The notebook and
        document surfaces in a browser build their own connection to the same
        server, and without an address here they had nothing to build it from:
        the runtime would start, appear in the Datalayer console, and leave the
        editors beside the chat connected to nothing.

        Named `server_url` to match `JupyterServerSandbox`, so whatever reads
        one reads the other. `_server_url` is the same value under the name
        older callers probe for.
        """
        runtime = self._runtime
        return getattr(runtime, "ingress", None) if runtime else None

    @property
    def _server_url(self) -> Optional[str]:
        """Alias of {@link server_url}, for callers that probe the private name."""
        return self.server_url

    @property
    def jupyter_token(self) -> Optional[str]:
        """The token that ingress wants.

        Useless apart from the URL and dangerous to confuse with the API key:
        `self._token` authenticates *this process* to Datalayer, and handing it
        to a browser as a Jupyter token would both fail and leak a credential.
        This is the runtime's own, minted for it.
        """
        runtime = self._runtime
        return getattr(runtime, "jupyter_token", None) if runtime else None

    @property
    def kernel_id(self) -> Optional[str]:
        """The kernel the agent is executing in, once there is one.

        Read from the runtime's model rather than kept here: the runtime owns
        the kernel's lifetime and replaces the id when it restarts, and a copy
        would go stale exactly when it mattered — a surface reconnecting to a
        kernel that no longer exists.
        """
        runtime = self._runtime
        if runtime is None:
            return None
        model = getattr(runtime, "model", None)
        return getattr(model, "kernel_id", None) if model else None

    @property
    def client(self):
        """Get the Datalayer client instance."""
        return self._client

    @property
    def runtime(self):
        """Get the runtime service instance."""
        return self._runtime

    @property
    def object_id(self) -> str:
        """Get the sandbox object ID (Modal-style)."""
        return self._sandbox_id

    @classmethod
    def from_id(
        cls,
        sandbox_id: str,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
        **kwargs,
    ) -> "DatalayerSandbox":
        """Retrieve an existing sandbox by its id, connected.

        What :meth:`list_all` does for every runtime, this does for one. It
        used to return an unconnected object with a comment saying it "would
        need agent-runtimes support" — which was there all along, and meant
        every caller wanting an existing runtime wrote the same four lines.

        `sandbox_id` is either the Runtimes `runtime_name` or the runtime's
        `uid`. Both, because `list_all` hands out the uid and `AgentClient`
        looks up by name: a `from_id` that took only one of them would refuse
        the very id this class had just given the caller.

        Raises:
            SandboxNotFoundError: when no such runtime exists, or when the
                lookup cannot be made at all. Answering an object that is not
                connected — the old behaviour — is worse than an error: every
                call on it fails later, somewhere else, with a message about
                whatever it touched first rather than about the wrong id.
        """
        try:
            import datalayer_core.utils.urls  # noqa: F401 - availability check
            from agent_runtimes.client import AgentClient
        except ImportError as error:
            raise SandboxNotFoundError(
                sandbox_id,
                f"Cannot look up sandbox '{sandbox_id}': the Datalayer client "
                f"is not installed. Install code-sandboxes with the "
                f"[datalayer] extra.",
            ) from error

        if run_url:
            client = AgentClient(urls=_urls_for_run(run_url), api_key=token)
        else:
            client = AgentClient(api_key=token)

        runtime = cls._find_runtime(client, sandbox_id)
        if runtime is None:
            raise SandboxNotFoundError(sandbox_id)
        return cls._adopt(runtime, token=token, run_url=run_url, **kwargs)

    @staticmethod
    def _find_runtime(client, sandbox_id: str):
        """The runtime this id names, by name and then by uid.

        By name first because it is one request; the scan is the fallback for
        a uid, which is what `list_all` hands out.
        """
        try:
            return client.get_runtime(sandbox_id)
        except Exception:  # noqa: BLE001 - not a name; try it as a uid
            pass
        try:
            runtimes = client.list_runtimes()
        except Exception:  # noqa: BLE001
            return None
        for runtime in runtimes:
            if sandbox_id in (runtime.uid, runtime.runtime_name):
                return runtime
        return None

    @classmethod
    def _adopt(
        cls,
        runtime,
        *,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
        **kwargs,
    ) -> "DatalayerSandbox":
        """A sandbox around a runtime that is already running.

        Shared with :meth:`list_all` so the two cannot drift — one adopting a
        runtime differently from the other is how `from_id` would come to
        return something that behaves unlike what iteration yields.
        """
        sandbox = cls(token=token, run_url=run_url, **kwargs)
        sandbox._client = getattr(runtime, "_client", None) or sandbox._client
        sandbox._runtime = runtime
        sandbox._sandbox_id = runtime.uid or runtime.runtime_name or str(uuid.uuid4())
        sandbox._started = True
        sandbox._info = SandboxInfo(
            id=sandbox._sandbox_id,
            variant="datalayer",
            status=SandboxStatus.RUNNING,
            created_at=time.time(),
            name=runtime.name,
            metadata={
                "network_policy": sandbox.config.network_policy,
                "allowed_hosts": sandbox.config.allowed_hosts,
            },
        )
        return sandbox

    @classmethod
    def list_all(
        cls,
        tags: Optional[dict[str, str]] = None,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
    ) -> Iterator["DatalayerSandbox"]:
        """List all running sandboxes.

        Similar to Modal's Sandbox.list() method.

        Args:
            tags: Filter sandboxes by tags.
            token: API token for authentication.
            run_url: Datalayer server URL.

        Yields:
            DatalayerSandbox instances.
        """
        try:
            import datalayer_core.utils.urls  # noqa: F401 - availability check
            from agent_runtimes.client import AgentClient
        except ImportError:
            return

        try:
            if run_url:
                urls = _urls_for_run(run_url)
                client = AgentClient(urls=urls, api_key=token)
            else:
                client = AgentClient(api_key=token)

            runtimes = client.list_runtimes()

            for runtime in runtimes:
                sandbox = cls._adopt(runtime, token=token, run_url=run_url)
                sandbox._client = client
                yield sandbox
        except Exception:
            return

    @classmethod
    def list_environments(
        cls,
        token: Optional[str] = None,
        run_url: Optional[str] = None,
    ) -> list[SandboxEnvironment]:
        try:
            import datalayer_core.utils.urls  # noqa: F401 - availability check
            from agent_runtimes.client import AgentClient
        except ImportError:
            return []

        try:
            if run_url:
                urls = _urls_for_run(run_url)
                client = AgentClient(urls=urls, api_key=token)
            else:
                client = AgentClient(api_key=token)

            environments = client.list_environments()
            return [
                SandboxEnvironment(
                    name=env.name,
                    title=env.title,
                    language=env.language,
                    owner=env.owner,
                    visibility=env.visibility,
                    burning_rate=float(env.burning_rate),
                    metadata=env.metadata,
                )
                for env in environments
            ]
        except Exception:
            return []

    def start(self) -> None:
        """Start the sandbox by creating a Datalayer runtime.

        Raises:
            SandboxConfigurationError: If configuration is invalid.
            SandboxConnectionError: If connection to Datalayer fails.
        """
        if self._started:
            return

        try:
            # Import here to avoid hard dependency
            import datalayer_core.utils.urls  # noqa: F401 - availability check
            from agent_runtimes.client import AgentClient
            from agent_runtimes.client.agent_client import DEFAULT_TIME_RESERVATION
        except ImportError as e:
            # What actually failed, not what usually fails.
            #
            # The message used to name the missing package and the command
            # that installs it, whatever the import error said. When the
            # package WAS installed and one of these names had moved, it sent
            # the reader to reinstall a dependency that was already there —
            # the real reason, `cannot import name X`, was thrown away with
            # the exception it was written on.
            raise SandboxConfigurationError(
                f"DatalayerSandbox cannot be used: {e}. "
                "If the package is missing, install it with: "
                "pip install code-sandboxes[datalayer]"
            ) from e

        try:
            # Create client with optional custom URL
            if self._run_url:
                urls = _urls_for_run(self._run_url)
                self._client = AgentClient(urls=urls, api_key=self._token)
            else:
                self._client = AgentClient(api_key=self._token)

            # Calculate time reservation.
            # Default to the platform default (10 minutes) unless max_lifetime is explicitly set.
            default_max_lifetime = SandboxConfig().max_lifetime
            if self.config.max_lifetime != default_max_lifetime:
                lifetime_minutes = int(self.config.max_lifetime / 60)
                time_reservation = max(10, min(lifetime_minutes, 1440))  # Max 24 hours
            else:
                time_reservation = int(DEFAULT_TIME_RESERVATION)

            # Determine environment based on GPU config
            environment = self.config.environment
            if self.config.gpu and "gpu" not in environment.lower():
                # Try to use a GPU environment if GPU is requested
                environment = "python-gpu-env"

            # Build sandbox name
            sandbox_name = self.config.name or f"sandbox-{self._sandbox_id[:8]}"

            # Create the runtime (optionally from snapshot)
            if self._snapshot_name:
                self._runtime = self._client.create_runtime(
                    name=sandbox_name,
                    environment=environment,
                    time_reservation=time_reservation,
                    snapshot_name=self._snapshot_name,
                )
            else:
                self._runtime = self._client.create_runtime(
                    name=sandbox_name,
                    environment=environment,
                    time_reservation=time_reservation,
                )

            # Start the runtime
            if hasattr(self._runtime, "start"):
                self._runtime.start()
            else:  # pragma: no cover - compatibility fallback
                self._runtime._start()

            self._default_context = self.create_context("default")

            # Calculate end time
            self._created_at = time.time()
            self._end_at = self._created_at + self.config.max_lifetime

            # Build resource config
            resources = None
            if self.config.gpu or self.config.cpu_limit or self.config.memory_limit:
                resources = ResourceConfig(
                    cpu=self.config.cpu_limit,
                    memory=self.config.memory_limit // (1024 * 1024)
                    if self.config.memory_limit
                    else None,
                    gpu=self.config.gpu,
                )

            self._info = SandboxInfo(
                id=self._sandbox_id,
                variant="datalayer",
                status=SandboxStatus.RUNNING,
                created_at=self._created_at,
                end_at=self._end_at,
                config=self.config,
                metadata={
                    "network_policy": self.config.network_policy,
                    "allowed_hosts": self.config.allowed_hosts,
                },
                name=sandbox_name,
                resources=resources,
            )
            self._started = True

        except Exception as e:
            url = self._run_url or "default"
            raise SandboxConnectionError(url, str(e)) from e

    def stop(self) -> None:
        """Stop the sandbox and release the Datalayer runtime."""
        if not self._started:
            return

        try:
            if self._runtime:
                if hasattr(self._runtime, "stop"):
                    self._runtime.stop()
                else:  # pragma: no cover - compatibility fallback
                    self._runtime._stop()
        except Exception:
            pass  # Best effort cleanup

        self._runtime = None
        self._client = None
        self._started = False
        if self._info:
            self._info.status = SandboxStatus.STOPPED

    def terminate(self) -> None:
        """Terminate the sandbox. Alias for stop()."""
        self.stop()

    def kill(self) -> None:
        """Kill the sandbox. Alias for stop()."""
        self.stop()

    def set_timeout(self, timeout_seconds: float) -> None:
        """Change the sandbox timeout during runtime.

        Resets the timeout to the new value.

        Args:
            timeout_seconds: New timeout in seconds from now.
        """
        if not self._started:
            raise SandboxNotStartedError()

        self._end_at = time.time() + timeout_seconds
        if self._info:
            self._info.end_at = self._end_at

    def get_info(self) -> SandboxInfo:
        """Retrieve sandbox information.

        Returns:
            SandboxInfo object with current sandbox state.
        """
        if self._info:
            return self._info
        return SandboxInfo(
            id=self._sandbox_id,
            variant="datalayer",
            status=SandboxStatus.PENDING if not self._started else SandboxStatus.RUNNING,
        )

    def wait(self, raise_on_termination: bool = True) -> None:
        """Wait for the sandbox to finish.

        Similar to Modal's wait() method.

        Args:
            raise_on_termination: Whether to raise if sandbox terminates with error.
        """
        # For cloud sandboxes, this would wait for the runtime to complete
        # Currently just a placeholder
        pass

    def _setup_tool_caller(self) -> None:
        """Keep tool calling on the client side for remote sandboxes."""
        return

    # -- Contents attachments ---------------------------------------------
    #
    # On Datalayer the mounts are not this adapter's to make: the Operator
    # reads them off the pod's annotation and mounts them — a volume, a
    # bucket, a person's own folder through Clouder's CSI — before the pod
    # starts. What is left to do here is to LOOK: is the path there, or not.

    def content_capabilities(self) -> ContentCapabilities:
        return ContentCapabilities(
            provider="datalayer",
            mount=True,
            bucket_mount=True,
            materialize=True,
            client=True,
            local_bridge_mount=LocalBridgeCapability(
                supported=True,
                required_features=["clouder-csi"],
                allowed_roots=[],
                read_only=True,
                read_write=True,
                reconnect=True,
                cleanup=True,
            ),
            filesystem_primitives=list(FILESYSTEM_PRIMITIVES),
        )

    def _prepare_mount(self, spec: ContentAttachmentSpec, *, reconcile: bool) -> PreparedAttachment:
        del reconcile
        return self._operator_mount(spec, capability="mount")

    def _prepare_local_bridge(
        self, spec: ContentAttachmentSpec, *, reconcile: bool
    ) -> PreparedAttachment:
        """The Operator rendered a CSI volume for the bridge; is it mounted?

        A mountpoint, not merely a path: the CSI driver binds the bridge
        filesystem into the pod, and a directory that is there without a
        mount behind it is an empty directory the image happened to have —
        or a copy something else made — and neither is the person's folder.
        """
        del reconcile
        mount_path = spec.mount_path or (spec.bridge.mount_path if spec.bridge else None)
        if not mount_path:
            return not_ready(spec, MOUNT_PATH_MISSING, "a local bridge needs a mount_path")
        if path_exists(self, mount_path) and path_is_mountpoint(self, mount_path):
            return ready(spec, capabilities=[LOCAL_BRIDGE_MOUNT])
        return not_ready(
            spec,
            MOUNT_MISSING,
            f"{mount_path} is not a mountpoint in the runtime: the Operator renders a "
            "local bridge as a CSI volume the node driver mounts before the pod "
            "starts, and this one is not mounted",
        )

    def _operator_mount(
        self, spec: ContentAttachmentSpec, *, capability: str
    ) -> PreparedAttachment:
        """Ready when the Operator's mount is there; otherwise, say it is not."""
        if not spec.mount_path:
            return not_ready(spec, MOUNT_PATH_MISSING, "a mount needs a mount_path")
        if path_exists(self, spec.mount_path):
            return ready(spec, capabilities=[capability])
        return not_ready(
            spec,
            MOUNT_MISSING,
            f"{spec.mount_path} is not mounted in the runtime: the Operator makes "
            "mounts from the pod annotation before the pod starts, and this one "
            "was not made",
        )

    def poll(self) -> Optional[int]:
        """Check if the sandbox has finished running.

        Similar to Modal's poll() method.

        Returns:
            None if still running, exit code otherwise.
        """
        if self._started:
            return None
        return 0

    def run_code(
        self,
        code: str,
        language: str = "python",
        context: Optional[Context] = None,
        on_stdout: Optional[OutputHandler[OutputMessage]] = None,
        on_stderr: Optional[OutputHandler[OutputMessage]] = None,
        on_result: Optional[OutputHandler[Result]] = None,
        on_error: Optional[OutputHandler[CodeError]] = None,
        envs: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute code in the Datalayer runtime.

        Args:
            code: The code to execute.
            language: Programming language (default: "python").
            context: Execution context (currently not used, runtime maintains state).
            on_stdout: Callback for stdout messages.
            on_stderr: Callback for stderr messages.
            on_result: Callback for results.
            on_error: Callback for code errors (Python exceptions).
            envs: Environment variables (set before execution).
            timeout: Timeout in seconds.

        Returns:
            Execution result.

        Raises:
            SandboxNotStartedError: If the sandbox hasn't been started.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        started_at = time.time()

        # Set environment variables if provided
        if envs:
            env_code = "\n".join(f"import os; os.environ[{k!r}] = {v!r}" for k, v in envs.items())
            self._runtime.execute(env_code)

        # Execute the code
        execution_timeout = timeout or self.config.timeout
        try:
            response = self._runtime.execute(code, timeout=execution_timeout)
        except Exception as e:
            # Infrastructure failure - couldn't execute the code
            return ExecutionResult(
                results=[],
                logs=Logs(),
                execution_ok=False,
                execution_error=f"Failed to execute code: {e}",
                execution_count=0,
                context_id=context.id if context else "default",
                started_at=started_at,
                completed_at=time.time(),
            )

        # Parse the response
        stdout_messages: list[OutputMessage] = []
        stderr_messages: list[OutputMessage] = []
        results: list[Result] = []
        code_error: Optional[CodeError] = None
        exit_code: Optional[int] = None

        current_time = time.time()

        # Process stdout
        if hasattr(response, "stdout") and response.stdout:
            for line in response.stdout.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=False)
                stdout_messages.append(msg)
                if on_stdout:
                    on_stdout(msg)

        # Process stderr
        if hasattr(response, "stderr") and response.stderr:
            for line in response.stderr.splitlines():
                msg = OutputMessage(line=line, timestamp=current_time, error=True)
                stderr_messages.append(msg)
                if on_stderr:
                    on_stderr(msg)

        # Process results
        if hasattr(response, "result") and response.result is not None:
            result = Result(
                data={"text/plain": str(response.result)},
                is_main_result=True,
            )
            results.append(result)
            if on_result:
                on_result(result)

        # Process display data (rich output)
        if hasattr(response, "display_data") and response.display_data:
            for display in response.display_data:
                result = Result(
                    data=display.get("data", {}),
                    is_main_result=False,
                    extra=display.get("metadata", {}),
                )
                results.append(result)
                if on_result:
                    on_result(result)

        # Process errors (code exceptions)
        if hasattr(response, "error") and response.error:
            ename = response.error.get("ename", "Error")
            evalue = response.error.get("evalue", "")

            # Handle SystemExit specially - extract exit code
            if ename == "SystemExit":
                try:
                    exit_code = int(evalue) if evalue else 0
                except (ValueError, TypeError):
                    exit_code = 1 if evalue else 0
            else:
                code_error = CodeError(
                    name=ename,
                    value=evalue,
                    traceback="\n".join(response.error.get("traceback", [])),
                )
                if on_error:
                    on_error(code_error)

        # Count every execution that reached the runtime, error included —
        # exactly as a Jupyter kernel numbers In[n].
        self._execution_count += 1
        return ExecutionResult(
            results=results,
            logs=Logs(stdout=stdout_messages, stderr=stderr_messages),
            execution_ok=True,
            code_error=code_error,
            exit_code=exit_code,
            execution_count=getattr(response, "execution_count", None) or self._execution_count,
            context_id=context.id if context else "default",
            started_at=started_at,
            completed_at=time.time(),
        )

    def _get_internal_variable(self, name: str, context: Optional[Context] = None) -> Any:
        """Get a variable from the runtime.

        Args:
            name: Variable name.
            context: Context (not used, runtime maintains single namespace).

        Returns:
            The variable value.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        return self._runtime.get_variable(name)

    def _set_internal_variable(
        self, name: str, value: Any, context: Optional[Context] = None
    ) -> None:
        """Set a variable in the runtime.

        Args:
            name: Variable name.
            value: Value to set.
            context: Context (not used, runtime maintains single namespace).
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        self._runtime.set_variable(name, value)

    def create_snapshot(
        self,
        name: str,
        description: str = "",
    ) -> SnapshotInfo:
        """Create a snapshot of the current runtime state.

        Similar to Modal's snapshot_filesystem feature. This allows saving
        the current state of the sandbox for later restoration.

        Args:
            name: Name for the snapshot.
            description: Optional description.

        Returns:
            SnapshotInfo with the snapshot details.

        Raises:
            SandboxNotStartedError: If sandbox is not running.
            SandboxSnapshotError: If snapshot creation fails.
        """
        if not self._started or not self._runtime:
            raise SandboxNotStartedError()

        try:
            snapshot = self._runtime.create_snapshot(name=name, description=description)
            return SnapshotInfo(
                id=snapshot.uid,
                name=name,
                sandbox_id=self._sandbox_id,
                created_at=time.time(),
                description=description,
            )
        except Exception as e:
            raise SandboxSnapshotError("create", str(e)) from e

    def list_snapshots(self) -> list[SnapshotInfo]:
        """List all snapshots.

        Returns:
            List of SnapshotInfo objects.
        """
        if not self._client:
            return []

        try:
            snapshots = self._client.list_snapshots()
            return [
                SnapshotInfo(
                    id=s.uid,
                    name=s.name,
                    sandbox_id="",
                    created_at=getattr(s, "created_at", 0),
                    description=getattr(s, "description", ""),
                )
                for s in snapshots
            ]
        except Exception:
            return []

    def install_packages(
        self, packages: list[str], timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Install Python packages in the runtime.

        Uses pip to install packages.

        Args:
            packages: List of package names to install.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.

        Example:
            sandbox.install_packages(["pandas", "numpy", "matplotlib"])
        """
        # Use %pip magic for better Jupyter integration
        pip_cmd = f"%pip install {' '.join(packages)}"
        return self.run_code(pip_cmd, timeout=timeout or 300)

    def install_requirements(
        self, requirements_path: str, timeout: Optional[float] = None
    ) -> ExecutionResult:
        """Install packages from a requirements file.

        Args:
            requirements_path: Path to requirements.txt file in the sandbox.
            timeout: Timeout in seconds.

        Returns:
            Execution result from the installation.
        """
        pip_cmd = f"%pip install -r {requirements_path}"
        return self.run_code(pip_cmd, timeout=timeout or 300)

    def open_file(self, path: str, mode: str = "r") -> "SandboxFileHandle":
        """Open a file in the sandbox.

        Similar to Modal's sandbox.open() method.

        Args:
            path: Path to the file.
            mode: File mode ('r', 'w', 'rb', 'wb', 'a').

        Returns:
            SandboxFileHandle for file operations.
        """
        from .filesystem import SandboxFileHandle

        return SandboxFileHandle(self, path, mode)
