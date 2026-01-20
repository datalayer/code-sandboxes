# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Filesystem operations for sandboxes."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from .base import Sandbox


class FileType(str, Enum):
    """Type of file in the sandbox filesystem."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class FileWatchEventType(str, Enum):
    """Types of file watch events."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class FileInfo:
    """Information about a file or directory.

    Attributes:
        name: Name of the file or directory.
        path: Full path to the file.
        type: Type of the entry (file, directory, symlink).
        size: Size in bytes (for files).
        modified: Last modification time (Unix timestamp).
        permissions: File permissions.
    """

    name: str
    path: str
    type: FileType = FileType.FILE
    size: int = 0
    modified: float = 0.0
    permissions: str = ""

    @property
    def is_file(self) -> bool:
        """Check if this is a file."""
        return self.type == FileType.FILE

    @property
    def is_directory(self) -> bool:
        """Check if this is a directory."""
        return self.type == FileType.DIRECTORY

    def __repr__(self) -> str:
        return f"FileInfo(name={self.name!r}, type={self.type.value}, size={self.size})"


@dataclass
class FileWatchEvent:
    """Event from watching a file or directory.

    Attributes:
        event_type: The type of event.
        path: Path to the affected file.
        timestamp: When the event occurred.
        new_path: For rename events, the new path.
    """

    event_type: FileWatchEventType
    path: str
    timestamp: float = 0.0
    new_path: Optional[str] = None


class SandboxFilesystem:
    """Filesystem operations for a sandbox.

    Provides file and directory operations similar to E2B and Modal.

    Example:
        with Sandbox.create() as sandbox:
            # Write a file
            sandbox.files.write("/data/test.txt", "Hello World")

            # Read a file
            content = sandbox.files.read("/data/test.txt")

            # List directory
            for f in sandbox.files.list("/data"):
                print(f.name, f.size)
    """

    def __init__(self, sandbox: "Sandbox"):
        """Initialize filesystem operations for a sandbox.

        Args:
            sandbox: The sandbox instance.
        """
        self._sandbox = sandbox

    def read(self, path: str) -> str:
        """Read a text file from the sandbox.

        Args:
            path: Path to the file.

        Returns:
            File contents as a string.
        """
        execution = self._sandbox.run_code(f"""
with open({path!r}, 'r') as f:
    __file_content__ = f.read()
""")
        if execution.error:
            raise FileNotFoundError(f"Could not read file: {path}")
        return self._sandbox.get_variable("__file_content__")

    def read_bytes(self, path: str) -> bytes:
        """Read a binary file from the sandbox.

        Args:
            path: Path to the file.

        Returns:
            File contents as bytes.
        """
        return self._sandbox._read_file(path)

    def write(self, path: str, content: str, make_dirs: bool = True) -> None:
        """Write a text file to the sandbox.

        Args:
            path: Path to the file.
            content: Content to write.
            make_dirs: Whether to create parent directories.
        """
        if make_dirs:
            dir_path = str(Path(path).parent)
            self._sandbox.run_code(f"""
import os
os.makedirs({dir_path!r}, exist_ok=True)
""")

        self._sandbox.run_code(f"""
with open({path!r}, 'w') as f:
    f.write({content!r})
""")

    def write_bytes(self, path: str, content: bytes, make_dirs: bool = True) -> None:
        """Write a binary file to the sandbox.

        Args:
            path: Path to the file.
            content: Content to write.
            make_dirs: Whether to create parent directories.
        """
        if make_dirs:
            dir_path = str(Path(path).parent)
            self._sandbox.run_code(f"""
import os
os.makedirs({dir_path!r}, exist_ok=True)
""")

        self._sandbox._write_file(path, content)

    def list(self, path: str = "/") -> list[FileInfo]:
        """List contents of a directory.

        Args:
            path: Directory path to list.

        Returns:
            List of FileInfo objects.
        """
        execution = self._sandbox.run_code(f"""
import os
import stat

__dir_contents__ = []
for name in os.listdir({path!r}):
    full_path = os.path.join({path!r}, name)
    try:
        st = os.stat(full_path)
        file_type = 'directory' if stat.S_ISDIR(st.st_mode) else 'file'
        if stat.S_ISLNK(st.st_mode):
            file_type = 'symlink'
        __dir_contents__.append({{
            'name': name,
            'path': full_path,
            'type': file_type,
            'size': st.st_size,
            'modified': st.st_mtime,
            'permissions': oct(st.st_mode)[-3:],
        }})
    except OSError:
        pass
""")
        if execution.error:
            raise FileNotFoundError(f"Could not list directory: {path}")

        contents = self._sandbox.get_variable("__dir_contents__")
        return [
            FileInfo(
                name=f["name"],
                path=f["path"],
                type=FileType(f["type"]),
                size=f["size"],
                modified=f["modified"],
                permissions=f["permissions"],
            )
            for f in contents
        ]

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists.

        Args:
            path: Path to check.

        Returns:
            True if the path exists.
        """
        execution = self._sandbox.run_code(f"""
import os
__path_exists__ = os.path.exists({path!r})
""")
        return self._sandbox.get_variable("__path_exists__")

    def is_file(self, path: str) -> bool:
        """Check if path is a file.

        Args:
            path: Path to check.

        Returns:
            True if the path is a file.
        """
        execution = self._sandbox.run_code(f"""
import os
__is_file__ = os.path.isfile({path!r})
""")
        return self._sandbox.get_variable("__is_file__")

    def is_dir(self, path: str) -> bool:
        """Check if path is a directory.

        Args:
            path: Path to check.

        Returns:
            True if the path is a directory.
        """
        execution = self._sandbox.run_code(f"""
import os
__is_dir__ = os.path.isdir({path!r})
""")
        return self._sandbox.get_variable("__is_dir__")

    def mkdir(self, path: str, parents: bool = True) -> None:
        """Create a directory.

        Args:
            path: Path to create.
            parents: Whether to create parent directories.
        """
        if parents:
            self._sandbox.run_code(f"""
import os
os.makedirs({path!r}, exist_ok=True)
""")
        else:
            self._sandbox.run_code(f"""
import os
os.mkdir({path!r})
""")

    def rm(self, path: str, recursive: bool = False) -> None:
        """Remove a file or directory.

        Args:
            path: Path to remove.
            recursive: Whether to remove directories recursively.
        """
        if recursive:
            self._sandbox.run_code(f"""
import shutil
shutil.rmtree({path!r}, ignore_errors=True)
""")
        else:
            self._sandbox.run_code(f"""
import os
if os.path.isdir({path!r}):
    os.rmdir({path!r})
else:
    os.remove({path!r})
""")

    def copy(self, src: str, dst: str) -> None:
        """Copy a file or directory.

        Args:
            src: Source path.
            dst: Destination path.
        """
        self._sandbox.run_code(f"""
import shutil
import os
if os.path.isdir({src!r}):
    shutil.copytree({src!r}, {dst!r})
else:
    shutil.copy2({src!r}, {dst!r})
""")

    def move(self, src: str, dst: str) -> None:
        """Move a file or directory.

        Args:
            src: Source path.
            dst: Destination path.
        """
        self._sandbox.run_code(f"""
import shutil
shutil.move({src!r}, {dst!r})
""")

    def get_info(self, path: str) -> FileInfo:
        """Get information about a file or directory.

        Args:
            path: Path to get info for.

        Returns:
            FileInfo object.
        """
        execution = self._sandbox.run_code(f"""
import os
import stat

st = os.stat({path!r})
file_type = 'directory' if stat.S_ISDIR(st.st_mode) else 'file'
if stat.S_ISLNK(st.st_mode):
    file_type = 'symlink'
__file_info__ = {{
    'name': os.path.basename({path!r}),
    'path': {path!r},
    'type': file_type,
    'size': st.st_size,
    'modified': st.st_mtime,
    'permissions': oct(st.st_mode)[-3:],
}}
""")
        if execution.error:
            raise FileNotFoundError(f"Could not get info for: {path}")

        info = self._sandbox.get_variable("__file_info__")
        return FileInfo(
            name=info["name"],
            path=info["path"],
            type=FileType(info["type"]),
            size=info["size"],
            modified=info["modified"],
            permissions=info["permissions"],
        )

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file from local filesystem to sandbox.

        Args:
            local_path: Local file path.
            remote_path: Destination path in sandbox.
        """
        self._sandbox.upload_file(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from sandbox to local filesystem.

        Args:
            remote_path: Path in sandbox.
            local_path: Local destination path.
        """
        self._sandbox.download_file(remote_path, local_path)


class SandboxFileHandle:
    """File handle for streaming file operations.

    Similar to Modal's FileIO interface.
    """

    def __init__(
        self,
        sandbox: "Sandbox",
        path: str,
        mode: str = "r",
    ):
        """Initialize a file handle.

        Args:
            sandbox: The sandbox instance.
            path: Path to the file.
            mode: File mode ('r', 'w', 'rb', 'wb', 'a').
        """
        self._sandbox = sandbox
        self._path = path
        self._mode = mode
        self._closed = False
        self._buffer = ""

        # Create the file handle in the sandbox
        self._handle_id = f"__fh_{id(self)}__"
        self._sandbox.run_code(f"""
{self._handle_id} = open({path!r}, {mode!r})
""")

    def write(self, content: Union[str, bytes]) -> int:
        """Write content to the file.

        Args:
            content: Content to write.

        Returns:
            Number of bytes/characters written.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file")

        self._sandbox.run_code(f"""
__write_count__ = {self._handle_id}.write({content!r})
""")
        return self._sandbox.get_variable("__write_count__")

    def read(self, size: int = -1) -> Union[str, bytes]:
        """Read from the file.

        Args:
            size: Number of bytes/characters to read. -1 for all.

        Returns:
            File content.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file")

        self._sandbox.run_code(f"""
__read_content__ = {self._handle_id}.read({size})
""")
        return self._sandbox.get_variable("__read_content__")

    def readline(self) -> Union[str, bytes]:
        """Read a single line from the file.

        Returns:
            A line from the file.
        """
        if self._closed:
            raise ValueError("I/O operation on closed file")

        self._sandbox.run_code(f"""
__read_line__ = {self._handle_id}.readline()
""")
        return self._sandbox.get_variable("__read_line__")

    def flush(self) -> None:
        """Flush the file buffer."""
        if self._closed:
            raise ValueError("I/O operation on closed file")

        self._sandbox.run_code(f"{self._handle_id}.flush()")

    def close(self) -> None:
        """Close the file handle."""
        if not self._closed:
            self._sandbox.run_code(f"{self._handle_id}.close()")
            self._closed = True

    def __enter__(self) -> "SandboxFileHandle":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass
