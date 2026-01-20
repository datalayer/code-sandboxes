# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Command execution for sandboxes."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional

if TYPE_CHECKING:
    from .base import Sandbox


@dataclass
class CommandResult:
    """Result of a command execution.

    Attributes:
        exit_code: The command's exit code.
        stdout: Standard output content.
        stderr: Standard error content.
        duration: Execution duration in seconds.
    """

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0

    @property
    def success(self) -> bool:
        """Whether the command succeeded (exit code 0)."""
        return self.exit_code == 0

    def __repr__(self) -> str:
        return f"CommandResult(exit_code={self.exit_code}, stdout_len={len(self.stdout)}, stderr_len={len(self.stderr)})"


@dataclass
class ProcessHandle:
    """Handle for a running process.

    Similar to Modal's ContainerProcess interface.

    Attributes:
        pid: Process ID (if available).
        command: The command being executed.
    """

    sandbox: "Sandbox"
    command: str
    pid: Optional[int] = None
    _process_var: str = ""
    _completed: bool = False
    _exit_code: Optional[int] = None
    _stdout_buffer: list[str] = field(default_factory=list)
    _stderr_buffer: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._process_var = f"__proc_{id(self)}__"

    @property
    def stdout(self) -> Iterator[str]:
        """Stream stdout lines.

        Yields:
            Lines from stdout as they become available.
        """
        # For synchronous implementation, return buffered output
        for line in self._stdout_buffer:
            yield line

    @property
    def stderr(self) -> Iterator[str]:
        """Stream stderr lines.

        Yields:
            Lines from stderr as they become available.
        """
        for line in self._stderr_buffer:
            yield line

    def read_stdout(self) -> str:
        """Read all stdout content.

        Returns:
            All stdout content as a string.
        """
        return "\n".join(self._stdout_buffer)

    def read_stderr(self) -> str:
        """Read all stderr content.

        Returns:
            All stderr content as a string.
        """
        return "\n".join(self._stderr_buffer)

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for the process to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            The exit code.
        """
        if self._completed:
            return self._exit_code or 0

        self.sandbox.run_code(f"""
{self._process_var}.wait()
__exit_code__ = {self._process_var}.returncode
""")
        self._exit_code = self.sandbox.get_variable("__exit_code__")
        self._completed = True
        return self._exit_code

    def poll(self) -> Optional[int]:
        """Check if the process has completed.

        Returns:
            Exit code if completed, None otherwise.
        """
        if self._completed:
            return self._exit_code

        self.sandbox.run_code(f"""
__poll_result__ = {self._process_var}.poll()
""")
        result = self.sandbox.get_variable("__poll_result__")
        if result is not None:
            self._exit_code = result
            self._completed = True
        return result

    def terminate(self) -> None:
        """Terminate the process."""
        self.sandbox.run_code(f"""
try:
    {self._process_var}.terminate()
except:
    pass
""")
        self._completed = True

    def kill(self) -> None:
        """Kill the process forcefully."""
        self.sandbox.run_code(f"""
try:
    {self._process_var}.kill()
except:
    pass
""")
        self._completed = True

    @property
    def returncode(self) -> Optional[int]:
        """Get the return code if process has completed."""
        if self._completed:
            return self._exit_code
        return self.poll()


class SandboxCommands:
    """Command execution for a sandbox.

    Provides terminal command execution similar to E2B and Modal.

    Example:
        with Sandbox.create() as sandbox:
            # Run a simple command
            result = sandbox.commands.run("ls -la")
            print(result.stdout)

            # Run with streaming
            process = sandbox.commands.exec("python", "-c", "print('hello')")
            for line in process.stdout:
                print(line)
    """

    def __init__(self, sandbox: "Sandbox"):
        """Initialize command operations for a sandbox.

        Args:
            sandbox: The sandbox instance.
        """
        self._sandbox = sandbox

    def run(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        shell: bool = True,
    ) -> CommandResult:
        """Run a command and wait for completion.

        Args:
            command: The command to run.
            cwd: Working directory.
            env: Environment variables.
            timeout: Timeout in seconds.
            shell: Whether to run through shell.

        Returns:
            CommandResult with exit code, stdout, stderr.
        """
        start_time = time.time()

        # Build the subprocess code
        env_str = f", env={{**os.environ, **{env!r}}}" if env else ""
        cwd_str = f", cwd={cwd!r}" if cwd else ""
        timeout_str = f", timeout={timeout}" if timeout else ""

        code = f"""
import subprocess
import os

try:
    __cmd_result__ = subprocess.run(
        {command!r},
        shell={shell},
        capture_output=True,
        text=True{cwd_str}{env_str}{timeout_str}
    )
    __cmd_output__ = {{
        'exit_code': __cmd_result__.returncode,
        'stdout': __cmd_result__.stdout,
        'stderr': __cmd_result__.stderr,
    }}
except subprocess.TimeoutExpired:
    __cmd_output__ = {{
        'exit_code': -1,
        'stdout': '',
        'stderr': 'Command timed out',
    }}
except Exception as e:
    __cmd_output__ = {{
        'exit_code': -1,
        'stdout': '',
        'stderr': str(e),
    }}
"""

        execution = self._sandbox.run_code(code, timeout=timeout)

        if execution.error:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(execution.error),
                duration=time.time() - start_time,
            )

        try:
            result = self._sandbox.get_variable("__cmd_output__")
            return CommandResult(
                exit_code=result["exit_code"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                duration=time.time() - start_time,
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
            )

    def exec(
        self,
        *args: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> ProcessHandle:
        """Execute a command with streaming output.

        Unlike `run()`, this returns immediately with a ProcessHandle
        that can be used to stream output and wait for completion.

        Args:
            *args: Command and arguments.
            cwd: Working directory.
            env: Environment variables.
            timeout: Timeout in seconds.

        Returns:
            ProcessHandle for the running process.
        """
        command = " ".join(args)
        process = ProcessHandle(sandbox=self._sandbox, command=command)

        # Start the subprocess
        env_str = f", env={{**os.environ, **{env!r}}}" if env else ""
        cwd_str = f", cwd={cwd!r}" if cwd else ""

        code = f"""
import subprocess
import os

{process._process_var} = subprocess.Popen(
    {list(args)!r},
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True{cwd_str}{env_str}
)
__proc_pid__ = {process._process_var}.pid
"""

        self._sandbox.run_code(code)
        try:
            process.pid = self._sandbox.get_variable("__proc_pid__")
        except Exception:
            pass

        return process

    def spawn(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ProcessHandle:
        """Spawn a background process.

        The process runs in the background and doesn't block.

        Args:
            command: The command to run.
            cwd: Working directory.
            env: Environment variables.

        Returns:
            ProcessHandle for the background process.
        """
        return self.exec(*command.split(), cwd=cwd, env=env)

    def run_script(
        self,
        script: str,
        interpreter: str = "/bin/bash",
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        """Run a script with the specified interpreter.

        Args:
            script: The script content.
            interpreter: Script interpreter path.
            cwd: Working directory.
            env: Environment variables.
            timeout: Timeout in seconds.

        Returns:
            CommandResult with exit code, stdout, stderr.
        """
        start_time = time.time()

        env_str = f", env={{**os.environ, **{env!r}}}" if env else ""
        cwd_str = f", cwd={cwd!r}" if cwd else ""
        timeout_str = f", timeout={timeout}" if timeout else ""

        code = f"""
import subprocess
import os

try:
    __script_result__ = subprocess.run(
        [{interpreter!r}, '-c', {script!r}],
        capture_output=True,
        text=True{cwd_str}{env_str}{timeout_str}
    )
    __script_output__ = {{
        'exit_code': __script_result__.returncode,
        'stdout': __script_result__.stdout,
        'stderr': __script_result__.stderr,
    }}
except subprocess.TimeoutExpired:
    __script_output__ = {{
        'exit_code': -1,
        'stdout': '',
        'stderr': 'Script execution timed out',
    }}
except Exception as e:
    __script_output__ = {{
        'exit_code': -1,
        'stdout': '',
        'stderr': str(e),
    }}
"""

        execution = self._sandbox.run_code(code, timeout=timeout)

        if execution.error:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(execution.error),
                duration=time.time() - start_time,
            )

        try:
            result = self._sandbox.get_variable("__script_output__")
            return CommandResult(
                exit_code=result["exit_code"],
                stdout=result["stdout"],
                stderr=result["stderr"],
                duration=time.time() - start_time,
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                duration=time.time() - start_time,
            )

    def install_system_packages(
        self,
        packages: list[str],
        package_manager: str = "apt-get",
        timeout: float = 300,
    ) -> CommandResult:
        """Install system packages.

        Args:
            packages: List of package names.
            package_manager: Package manager to use (apt-get, yum, etc.).
            timeout: Timeout in seconds.

        Returns:
            CommandResult from the installation.
        """
        if package_manager == "apt-get":
            cmd = f"apt-get update && apt-get install -y {' '.join(packages)}"
        elif package_manager == "yum":
            cmd = f"yum install -y {' '.join(packages)}"
        elif package_manager == "apk":
            cmd = f"apk add {' '.join(packages)}"
        else:
            cmd = f"{package_manager} install {' '.join(packages)}"

        return self.run(cmd, timeout=timeout)
