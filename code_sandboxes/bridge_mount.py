# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A person's own folder, mounted inside the sandbox over the bridge relay.

This module runs INSIDE the sandbox. The adapter copies its source there and
starts it as a background process (see :func:`code_sandboxes.contents.
start_bridge_mount`), so it must stand on its own: the standard library, plus
three things it looks for at run time and does without when they are absent —
`fuse` (fusepy) to mount, a websocket client to reach the relay, and
`datalayer_common.content_bridge` for the frame protocol. None of them is
imported at module level, so the file can always be loaded and asked what is
missing.

What it mounts is a userspace filesystem whose every operation is a round
trip over the bridge: `getattr` is a `stat`, `readdir` is a `list`, `read`
is a `read`. Nothing is cached, here or in the kernel — the mount is made
with `direct_io` and zero attribute timeouts — so a folder that changed on the
person's machine is what the sandbox reads, and a tunnel that dropped makes
every operation fail (`ENOTCONN` while it is down, `EIO` at the moment it
drops) rather than answer with what was last seen. A `ro` mount refuses every
write with `EROFS` before it reaches the wire, on top of the kernel's own
`ro` enforcement.

The tunnel reconnects with backoff for as long as the mount token is good.
A relay that refuses the token — revoked or expired — ends the mount: the
state goes terminal, the filesystem is unmounted, and the process exits.

Protocol client
---------------
The relay forwards binary frames to the trusted client on the person's
machine, and `datalayer_common.content_bridge.BridgeFileSystemClient`
speaks the frame protocol over a transport with `send` and `recv` — here,
one websocket. When the grant carries a `session_key`, the frames are
sealed end to end by a `SecureChannel` the relay cannot open. Where the
module is absent, anything with the eight methods of
:class:`BridgeFileSystem` can be handed to the tunnel through its `connect`
callable — which is also how the tests drive this module without a relay.
"""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import shutil
import stat as stat_module
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any, Protocol

logger = logging.getLogger("datalayer.bridge_mount")

#: The states of a bridge session, as Contents names them.
STATES = ("pending", "connected", "reconnecting", "disconnected", "revoked", "expired")
#: The states there is no coming back from: the mount ends.
TERMINAL_STATES = frozenset({"revoked", "expired"})

#: The delay before the first reconnection attempt, and the most any delay grows to.
DEFAULT_BACKOFF = (1.0, 30.0)

#: What the launcher answers on its first line of stdout, and nothing else.
STATUS_CONNECTED = "connected"
STATUS_FAILED = "failed"

#: Why a mount could not be made, in a word a caller can act on.
FUSE_UNAVAILABLE = "FUSE_UNAVAILABLE"
BRIDGE_CONNECT_FAILED = "BRIDGE_CONNECT_FAILED"
INVALID_MODE = "INVALID_MODE"

#: The environment variable the token may be handed in, instead of a file.
TOKEN_ENV = "DATALAYER_BRIDGE_MOUNT_TOKEN"  # noqa: S105 - the NAME of a variable


# --- The protocol ------------------------------------------------------------


class BridgeFileSystem(Protocol):
    """The filesystem operations of the bridge protocol, as this module uses them.

    Paths are absolute within the bridged root (`/` is the root the person
    chose). An entry — what `stat` answers and `list` yields — is a mapping
    with at least `type` (`file`, `dir` or `symlink`) and `size`; `mtime`
    (seconds since the epoch, or ISO 8601) and `mode` are honoured when
    given. `list` may yield names alone. `unlink` removes a file, and an
    empty directory too where the client has no `rmdir` of its own. A
    missing path raises `FileNotFoundError` (or any error carrying an
    `errno`); a lost connection raises `ConnectionError`.
    """

    def stat(self, path: str) -> Mapping[str, Any]: ...

    def list(self, path: str) -> Iterable[Any]: ...

    def read(self, path: str, offset: int, size: int) -> bytes: ...

    def write(self, path: str, offset: int, data: bytes) -> int: ...

    def mkdir(self, path: str) -> None: ...

    def unlink(self, path: str) -> None: ...

    def rename(self, source: str, target: str) -> None: ...

    def truncate(self, path: str, size: int) -> None: ...


#: Opens a connection to the relay for this token and answers with a client.
#: Raises :class:`BridgeRefusedError` when the relay will not have the token, and
#: anything else when the relay could not be reached this time.
Connector = Callable[[str, str], BridgeFileSystem]


class BridgeRefusedError(Exception):
    """The relay refused the token, and will keep refusing it: `revoked` or `expired`."""

    def __init__(self, state: str, detail: str = ""):
        self.state = state if state in TERMINAL_STATES else "revoked"
        self.detail = detail
        super().__init__(detail or f"the relay refused the mount token: {self.state}")


class BridgeUnavailableError(Exception):
    """The relay could not be reached this time. Worth trying again."""


class BridgeProtocolError(Exception):
    """An operation the other side refused, with the errno it amounts to."""

    def __init__(self, errno_: int, detail: str = ""):
        self.errno = errno_
        super().__init__(detail or os.strerror(errno_))


#: The errors that mean the wire is gone, not that one operation failed.
#: The frame protocol's own channel error — a frame that cannot be
#: understood, one that failed authentication — is turned into one of these
#: by :class:`_RelayClient`, where that protocol is in hand.
CHANNEL_ERRORS: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError, EOFError)


# --- FUSE, when it is there ----------------------------------------------------

try:  # pragma: no cover - which branch runs depends on what is installed
    from fuse import FUSE as _FUSE
    from fuse import FuseOSError, Operations
except ImportError:  # pragma: no cover
    _FUSE = None

    class FuseOSError(OSError):  # type: ignore[no-redef]
        """fusepy's error, so the operations raise the same thing without it."""

        def __init__(self, errno_: int):
            super().__init__(errno_, os.strerror(errno_))

    class Operations:  # type: ignore[no-redef]
        """fusepy's base, so the operations class loads without it."""


def fuse_unavailable_reason() -> str | None:
    """Why a FUSE mount cannot be made here, or None when it can."""
    if _FUSE is None:
        return "fusepy is not installed in the sandbox"
    if not os.path.exists("/dev/fuse"):
        return "/dev/fuse is absent: the sandbox runtime does not expose FUSE"
    return None


def fuse_probe() -> dict[str, Any]:
    """What this sandbox has of what a bridge mount needs: the `fuse` feature."""
    fusermount = shutil.which("fusermount3") or shutil.which("fusermount")
    device = os.path.exists("/dev/fuse")
    return {
        "fusepy": _FUSE is not None,
        "device": device,
        "fusermount": fusermount,
        "ok": _FUSE is not None and device,
    }


# --- The tunnel --------------------------------------------------------------


class BridgeTunnel:
    """The connection to the relay, and the one place its state is kept.

    Operations ask it for the client and get one only while it is
    connected: a tunnel that is reconnecting, disconnected or ended answers
    `ENOTCONN`, never a client that would fail half-way or, worse, a stale
    one. When an operation finds the connection gone it says so with
    :meth:`lost`, and a thread of the tunnel's own tries again with growing
    delays until the relay takes the token back — or refuses it for good.
    """

    def __init__(
        self,
        relay_url: str,
        mount_token: str,
        *,
        connect: Connector | None = None,
        backoff: tuple[float, float] = DEFAULT_BACKOFF,
        sleep: Callable[[float], None] = time.sleep,
        on_state: Callable[[str], None] | None = None,
    ):
        self.relay_url = relay_url
        self._mount_token = mount_token
        self._connect = connect or connect_relay
        self._backoff = backoff
        self._sleep = sleep
        self._on_state = on_state
        self._lock = threading.RLock()
        self._client: BridgeFileSystem | None = None
        self._closed = False
        self._reconnector: threading.Thread | None = None
        self.state: str = "pending"
        #: The delays slept before each reconnection attempt, for the record.
        self.attempts: list[float] = []

    # -- state -------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        with self._lock:
            if state == self.state:
                return
            self.state = state
        logger.info("bridge %s: %s", self.relay_url, state)
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:
                logger.exception("state hook failed for %s", state)

    @property
    def ended(self) -> bool:
        return self.state in TERMINAL_STATES or self._closed

    # -- connecting --------------------------------------------------------

    def open(self) -> None:
        """Connect for the first time. Raises when it cannot; says why."""
        try:
            client = self._connect(self.relay_url, self._mount_token)
        except BridgeRefusedError as refused:
            self._set_state(refused.state)
            raise
        except Exception as error:
            self._set_state("disconnected")
            raise BridgeUnavailableError(f"{type(error).__name__}: {error}") from error
        with self._lock:
            self._client = client
        self._set_state("connected")

    def client(self) -> BridgeFileSystem:
        """The client, while connected. `ENOTCONN` otherwise — never a stale one."""
        with self._lock:
            if self.state == "connected" and self._client is not None:
                return self._client
        raise FuseOSError(errno.ENOTCONN)

    def lost(self, error: BaseException | None = None) -> None:
        """The connection dropped under an operation. Reconnect, in the background."""
        with self._lock:
            if self.ended:
                return
            self._discard_client()
            self._set_state("reconnecting")
            if self._reconnector is None or not self._reconnector.is_alive():
                self._reconnector = threading.Thread(
                    target=self._reconnect, name="bridge-reconnect", daemon=True
                )
                self._reconnector.start()
        if error is not None:
            logger.warning("bridge connection lost: %s: %s", type(error).__name__, error)

    def _reconnect(self) -> None:
        delay = self._backoff[0]
        while not self.ended:
            self.attempts.append(delay)
            self._sleep(delay)
            if self.ended:
                return
            try:
                client = self._connect(self.relay_url, self._mount_token)
            except BridgeRefusedError as refused:
                self._set_state(refused.state)
                return
            except Exception as error:
                logger.info("reconnect failed: %s: %s", type(error).__name__, error)
                delay = min(delay * 2, self._backoff[1])
                continue
            with self._lock:
                if self.ended:
                    _close_quietly(client)
                    return
                self._client = client
            self._set_state("connected")
            return

    def wait_reconnected(self, timeout: float) -> bool:
        """For tests and shutdown: whether the reconnection finished in time."""
        thread = self._reconnector
        if thread is not None:
            thread.join(timeout)
        return self.state == "connected"

    def close(self) -> None:
        """Let go of the connection. The state stays what it was, unless it was live."""
        with self._lock:
            self._closed = True
            self._discard_client()
            if self.state in ("connected", "reconnecting", "pending"):
                self._set_state("disconnected")

    def _discard_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            _close_quietly(client)


def _close_quietly(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception as error:
            logger.debug("closing the connection failed: %s: %s", type(error).__name__, error)


# --- The filesystem ----------------------------------------------------------


def _field(entry: Any, *names: str, default: Any = None) -> Any:
    """The first of `names` the entry carries, as a key or an attribute."""
    for name in names:
        if isinstance(entry, Mapping):
            if name in entry:
                return entry[name]
        elif hasattr(entry, name):
            return getattr(entry, name)
    return default


def _seconds(value: Any) -> float:
    """A timestamp as seconds since the epoch, from what the protocol sent."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return time.time()


def entry_name(entry: Any) -> str:
    """The name of a directory entry, whether it came as a name or a record."""
    if isinstance(entry, str):
        name = entry
    else:
        name = str(_field(entry, "name", "path", default=""))
    return name.rstrip("/").rsplit("/", 1)[-1]


def attributes(entry: Any, *, read_only: bool, uid: int, gid: int) -> dict[str, Any]:
    """A `stat` result for FUSE, from an entry of the protocol.

    The kind decides the mode; the person's own permission bits are not
    carried across, because the sandbox is not the person: what it may do is
    what the bridge allows, `ro` or `rw`, and that is what the bits say.
    """
    kind = str(_field(entry, "type", "kind", default="") or "").lower()
    if not kind:
        if _field(entry, "is_dir", "isdir", default=False):
            kind = "dir"
        elif _field(entry, "is_symlink", default=False):
            kind = "symlink"
        else:
            kind = "file"
    if kind in ("dir", "directory", "folder"):
        mode = stat_module.S_IFDIR | (0o555 if read_only else 0o755)
        nlink = 2
    elif kind in ("symlink", "link"):
        mode = stat_module.S_IFLNK | 0o777
        nlink = 1
    else:
        mode = stat_module.S_IFREG | (0o444 if read_only else 0o644)
        nlink = 1
    size = int(_field(entry, "size", default=0) or 0)
    mtime = _seconds(_field(entry, "mtime", "modified", "modified_at", default=None))
    ctime = _seconds(_field(entry, "ctime", "created", "created_at", default=mtime))
    return {
        "st_mode": mode,
        "st_nlink": nlink,
        "st_size": size,
        "st_uid": uid,
        "st_gid": gid,
        "st_atime": mtime,
        "st_mtime": mtime,
        "st_ctime": ctime,
    }


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_TRUNC | os.O_CREAT


class BridgeOperations(Operations):
    """The FUSE operations, each one a call over the tunnel.

    There is deliberately no state here beyond the tunnel and the mode: no
    attribute cache, no page cache, no directory listing kept from last
    time. What the kernel asks is asked of the person's machine, now, and if
    the machine cannot be asked the answer is an error. The mount options
    (`direct_io`, zero timeouts) keep the kernel from caching on this
    module's behalf.
    """

    def __init__(
        self,
        tunnel: BridgeTunnel,
        *,
        mode: str = "ro",
        uid: int | None = None,
        gid: int | None = None,
    ):
        self.tunnel = tunnel
        self.read_only = mode != "rw"
        # Whose the files look like. Inside a sandbox the mounter *is* the
        # reader, so its own ids are right. On a node agent the mounter is
        # root and the reader is the sandbox's user in another namespace, so
        # the agent says which ids to report — otherwise every file comes
        # back owned by root and the sandbox cannot write to its own folder.
        self.uid = uid if uid is not None else (os.getuid() if hasattr(os, "getuid") else 0)
        self.gid = gid if gid is not None else (os.getgid() if hasattr(os, "getgid") else 0)

    # -- plumbing ----------------------------------------------------------

    def _call(self, operation: str, *args: Any) -> Any:
        client = self.tunnel.client()  # ENOTCONN when there is none
        try:
            return getattr(client, operation)(*args)
        except FuseOSError:
            raise
        except CHANNEL_ERRORS as error:
            # The wire went, under this operation. Nobody is served until it
            # is back — and this operation is not served at all.
            self.tunnel.lost(error)
            raise FuseOSError(errno.EIO) from error
        except BridgeProtocolError as error:
            raise FuseOSError(error.errno) from error
        except OSError as error:
            raise FuseOSError(error.errno or errno.EIO) from error
        except Exception as error:
            logger.warning("%s%r failed: %s: %s", operation, args, type(error).__name__, error)
            raise FuseOSError(errno.EIO) from error

    def _writable(self) -> None:
        if self.read_only:
            raise FuseOSError(errno.EROFS)

    # -- reading -----------------------------------------------------------

    def getattr(self, path: str, fh: Any = None) -> dict[str, Any]:
        entry = self._call("stat", path)
        return attributes(entry, read_only=self.read_only, uid=self.uid, gid=self.gid)

    def readdir(self, path: str, fh: Any = None) -> list[str]:
        names = [entry_name(entry) for entry in self._call("list", path)]
        return [".", ".."] + [name for name in names if name]

    def read(self, path: str, size: int, offset: int, fh: Any = None) -> bytes:
        return bytes(self._call("read", path, offset, size))

    def open(self, path: str, flags: int) -> int:
        if flags & _WRITE_FLAGS:
            self._writable()
        self._call("stat", path)
        return 0

    def access(self, path: str, amode: int) -> int:
        if amode & os.W_OK:
            self._writable()
        self._call("stat", path)
        return 0

    def statfs(self, path: str) -> dict[str, Any]:
        # Nothing meaningful can be said of a folder on somebody's laptop;
        # `df` needs an answer, not a good one.
        return {"f_bsize": 4096, "f_frsize": 4096, "f_blocks": 0, "f_bfree": 0, "f_bavail": 0}

    # -- writing -----------------------------------------------------------

    def write(self, path: str, data: bytes, offset: int, fh: Any = None) -> int:
        self._writable()
        return int(self._call("write", path, offset, bytes(data)))

    def create(self, path: str, mode: int, fi: Any = None) -> int:
        self._writable()
        self._call("write", path, 0, b"")
        return 0

    def truncate(self, path: str, length: int, fh: Any = None) -> None:
        self._writable()
        self._call("truncate", path, length)

    def unlink(self, path: str) -> None:
        self._writable()
        self._call("unlink", path)

    def rmdir(self, path: str) -> None:
        self._writable()
        operation = "rmdir" if hasattr(self.tunnel.client(), "rmdir") else "unlink"
        self._call(operation, path)

    def mkdir(self, path: str, mode: int = 0o755) -> None:
        self._writable()
        self._call("mkdir", path)

    def rename(self, old: str, new: str) -> None:
        self._writable()
        self._call("rename", old, new)

    def chmod(self, path: str, mode: int) -> int:
        self._writable()
        return 0  # the bridge does not carry permission bits; not an error

    def chown(self, path: str, uid: int, gid: int) -> int:
        self._writable()
        return 0

    def utimens(self, path: str, times: Any = None) -> int:
        self._writable()
        return 0

    def flush(self, path: str, fh: Any = None) -> int:
        return 0

    def release(self, path: str, fh: Any = None) -> int:
        return 0

    def fsync(self, path: str, datasync: Any = None, fh: Any = None) -> int:
        return 0


# --- The relay ---------------------------------------------------------------

#: Words a relay's refusal carries, mapped to the state they mean.
_REFUSAL_WORDS = {"revoked": "revoked", "expired": "expired"}
#: Handshake statuses that say the token itself is no good.
_REFUSAL_STATUSES = {401: "expired", 403: "revoked", 404: "revoked", 410: "revoked"}


def refusal_state(text: str | None) -> str | None:
    """The terminal state a relay message names, if it names one."""
    lowered = (text or "").lower()
    for word, state in _REFUSAL_WORDS.items():
        if word in lowered:
            return state
    return None


def control_state(frame: Any) -> str | None:
    """The bridge state a text frame from the relay announces, if any.

    The relay forwards binary frames; a text frame is the relay itself
    speaking — `{"state": "revoked"}` when the person revoked the bridge, and
    the like. Anything that is not such an announcement is None.
    """
    if not isinstance(frame, str):
        return None
    try:
        message = json.loads(frame)
    except ValueError:
        return refusal_state(frame)
    if not isinstance(message, Mapping):
        return None
    state = message.get("state") or message.get("status")
    if isinstance(state, str) and state in STATES:
        return state
    return refusal_state(str(message.get("error") or message.get("reason") or ""))


class _RelayClient:
    """The protocol client over one websocket, with a `close` of its own.

    The protocol's channel errors — a frame the channel cannot open, a
    handshake gone wrong — end the channel, and so are answered as a
    `ConnectionError`: what makes the tunnel reconnect rather than serve.
    """

    def __init__(
        self,
        inner: Any,
        socket: Any,
        channel_errors: tuple[type[BaseException], ...] = (),
    ):
        self._inner = inner
        self._socket = socket
        self._channel_errors = channel_errors

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute) or not self._channel_errors:
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return attribute(*args, **kwargs)
            except self._channel_errors as error:
                raise ConnectionError(f"the channel is over: {error}") from error

        return call

    def close(self) -> None:
        _close_quietly(self._socket)


def _open_websocket(url: str) -> Any:
    """A connected websocket, from whichever client library is installed."""
    try:
        from websockets.exceptions import InvalidStatus
        from websockets.sync.client import connect

        try:
            return connect(url, max_size=None)
        except InvalidStatus as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            state = _REFUSAL_STATUSES.get(int(status or 0))
            if state:
                raise BridgeRefusedError(state, f"the relay answered {status}") from error
            raise
    except ImportError:
        pass
    try:
        import websocket  # websocket-client

        try:
            return websocket.create_connection(url)
        except websocket.WebSocketBadStatusException as error:
            state = _REFUSAL_STATUSES.get(int(getattr(error, "status_code", 0) or 0))
            if state:
                raise BridgeRefusedError(
                    state, f"the relay answered {error.status_code}"
                ) from error
            raise
    except ImportError as error:
        raise BridgeUnavailableError(
            "no websocket client is installed in the sandbox (websockets or websocket-client)"
        ) from error


class _RelayFrames:
    """`send` and `recv` over one websocket: the transport the protocol client takes.

    A text frame in the stream is the relay itself speaking — a state
    announcement — and a terminal one ends the tunnel; anything else that
    goes wrong on the wire is a `ConnectionError`, which is what makes the
    tunnel reconnect rather than serve.
    """

    def __init__(self, socket: Any):
        self._socket = socket

    def send(self, frame: bytes) -> None:
        try:
            self._socket.send(frame)
        except Exception as error:
            raise ConnectionError(str(error)) from error

    def recv(self) -> bytes:
        while True:
            try:
                frame = self._socket.recv()
            except Exception as error:
                state = refusal_state(str(getattr(error, "reason", "") or error))
                if state:
                    raise BridgeRefusedError(state, str(error)) from error
                raise ConnectionError(str(error)) from error
            if isinstance(frame, str):
                state = control_state(frame)
                if state in TERMINAL_STATES:
                    raise BridgeRefusedError(state, frame)
                continue
            return frame


def connect_relay(
    relay_url: str,
    mount_token: str,
    *,
    bridge_uid: str | None = None,
    session_key: str | None = None,
) -> BridgeFileSystem:
    """Open the tunnel: a websocket to the relay, the hello, the channel, the client.

    The first frame is the mount side's hello to the relay — `{"role":
    "mount", "token": …}`, a text frame. When the grant carried a
    `session_key`, the next two frames are the channel handshake — each
    end's public key — after which every frame is sealed end to end and the
    relay forwards what it cannot read. Everything after is the frame
    protocol, which `datalayer_common.content_bridge.BridgeFileSystemClient`
    speaks over the `send` and `recv` of :class:`_RelayFrames`.
    """
    # The protocol lives in `datalayer_core`; `datalayer_common.content_bridge`
    # is a re-export of it. Asking for the re-export first asked every place
    # this runs — a sandbox, and the node agent — to install a *services*
    # package (FastAPI, OpenTelemetry, OpenFGA) for one module that was
    # already there, and the node agent, which has core and not common,
    # failed every local mount with a message written for a sandbox.
    try:
        from datalayer_core.contents_bridge_protocol import (
            BridgeFileSystemClient,
            BridgeProtocolError,
            SecureChannel,
        )
    except ImportError:
        try:
            from datalayer_common.content_bridge import (  # type: ignore[no-redef]
                BridgeFileSystemClient,
                BridgeProtocolError,
                SecureChannel,
            )
        except ImportError as error:
            raise BridgeUnavailableError(
                "the bridge frame protocol is not installed here: it comes with "
                "datalayer-core, and `code_sandboxes[bridge]` asks for it"
            ) from error

    socket = _open_websocket(relay_url)
    transport = _RelayFrames(socket)
    channel = None
    try:
        socket.send(json.dumps({"role": "mount", "token": mount_token}))
        if session_key:
            if not bridge_uid:
                raise BridgeUnavailableError(
                    "a session key was given without the bridge uid it belongs to"
                )
            channel = SecureChannel(role="mount", bridge_uid=bridge_uid, session_key=session_key)
            transport.send(channel.hello())
            channel.establish(transport.recv())
    except BridgeRefusedError:
        _close_quietly(socket)
        raise
    except BridgeUnavailableError:
        _close_quietly(socket)
        raise
    except Exception as error:
        _close_quietly(socket)
        raise BridgeUnavailableError(
            f"the handshake with the relay failed: {type(error).__name__}: {error}"
        ) from error
    return _RelayClient(
        BridgeFileSystemClient(transport, channel=channel), socket, (BridgeProtocolError,)
    )


def relay_connector(bridge_uid: str | None = None, session_key: str | None = None) -> Connector:
    """A connector for one bridge: :func:`connect_relay` with its channel settled."""

    def connect(relay_url: str, mount_token: str) -> BridgeFileSystem:
        return connect_relay(relay_url, mount_token, bridge_uid=bridge_uid, session_key=session_key)

    return connect


# --- Mounting ----------------------------------------------------------------

#: Mounts `operations` at `mount_path` in `mode` and blocks until unmounted.
Mounter = Callable[[BridgeOperations, str, str], None]


def mount_with_fuse(
    operations: BridgeOperations, mount_path: str, mode: str, *, allow_other: bool = False
) -> None:
    """The real thing: fusepy, in the foreground, with no caching anywhere.

    `direct_io` keeps reads and writes out of the page cache, the zero
    timeouts keep attributes and lookups out of the dentry cache, `ro` has
    the kernel refuse writes before they reach the operations, and
    `nothreads` keeps the protocol client — one socket — on one thread.
    """
    if _FUSE is None:
        raise RuntimeError(fuse_unavailable_reason() or "fusepy is not available")
    options: dict[str, Any] = {
        "foreground": True,
        "nothreads": True,
        "direct_io": True,
        "attr_timeout": 0,
        "entry_timeout": 0,
        "negative_timeout": 0,
        "fsname": "datalayer-bridge",
        "subtype": "bridge",
    }
    if mode != "rw":
        options["ro"] = True
    if allow_other:
        # A FUSE mount is the mounting user's alone unless it says otherwise.
        # Inside a sandbox that is the right default; on a node agent the
        # mount is made by root and read by the sandbox's user, who without
        # this is answered `Permission denied` for the folder they asked for.
        # It needs `user_allow_other` in `/etc/fuse.conf`, which the node
        # agent's image sets.
        options["allow_other"] = True
    _FUSE(operations, mount_path, **options)


def unmount(mount_path: str) -> bool:
    """Take the mount down. True when a command was found to do it."""
    commands = [
        ["fusermount3", "-u", "-z", mount_path],
        ["fusermount", "-u", "-z", mount_path],
        ["umount", "-l", mount_path],
    ]
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.run(command, check=False, capture_output=True, timeout=30)  # noqa: S603
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


class _StatusFile:
    """Where the launcher writes the state, for a reconcile to read later."""

    def __init__(self, path: str | None):
        self.path = path

    def write(self, state: str, **extra: Any) -> None:
        if not self.path:
            return
        payload = {"state": state, "at": time.time(), "pid": os.getpid(), **extra}
        part = self.path + ".part"
        try:
            with open(part, "w") as handle:
                json.dump(payload, handle)
            os.replace(part, self.path)
        except OSError:
            logger.warning("could not write the status file %s", self.path)


def _report_stdout(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(payload)) + "\n")
    sys.stdout.flush()


def run_bridge_mount(
    relay_url: str,
    mount_token: str,
    mount_path: str,
    mode: str,
    *,
    bridge_uid: str | None = None,
    session_key: str | None = None,
    connect: Connector | None = None,
    mount: Mounter | None = None,
    status_file: str | None = None,
    report: Callable[[Mapping[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    backoff: tuple[float, float] = DEFAULT_BACKOFF,
    allow_other: bool = False,
    uid: int | None = None,
    gid: int | None = None,
) -> int:
    """Connect, say so, mount, and stay until the mount is gone.

    `bridge_uid` and `session_key` are the grant's: with them the frames
    are sealed end to end; without a key they travel as the relay forwards
    them. `connect` replaces the relay altogether, for tests.

    The first line on stdout is the answer the adapter waits for:
    `{"status": "connected", ...}` once the tunnel is up and the mount is
    about to be made, or `{"status": "failed", "error": ..., "detail": ...}`
    — and nothing before it. Then this blocks in the FUSE loop. The return
    value is the process's exit code: 0 for a mount that was taken down,
    non-zero for one that never came up or was ended by the relay.
    """
    say = report or _report_stdout
    if mode not in ("ro", "rw"):
        say({"status": STATUS_FAILED, "error": INVALID_MODE, "detail": f"mode {mode!r}"})
        return 2
    mounter = mount or (
        partial(mount_with_fuse, allow_other=True) if allow_other else mount_with_fuse
    )
    if mount is None:
        reason = fuse_unavailable_reason()
        if reason:
            say({"status": STATUS_FAILED, "error": FUSE_UNAVAILABLE, "detail": reason})
            return 3
    try:
        os.makedirs(mount_path, exist_ok=True)
    except OSError as error:
        say(
            {
                "status": STATUS_FAILED,
                "error": BRIDGE_CONNECT_FAILED,
                "detail": f"{mount_path} cannot be made: {error}",
            }
        )
        return 3

    status = _StatusFile(status_file)
    mounted = threading.Event()

    def on_state(state: str) -> None:
        status.write(state, mount_path=mount_path, mode=mode)
        if state in TERMINAL_STATES and mounted.is_set():
            # The relay will not have the token again: the mount ends, and
            # the FUSE loop below returns once the kernel lets go.
            unmount(mount_path)

    tunnel = BridgeTunnel(
        relay_url,
        mount_token,
        connect=connect or relay_connector(bridge_uid, session_key),
        sleep=sleep,
        backoff=backoff,
        on_state=on_state,
    )
    try:
        tunnel.open()
    except BridgeRefusedError as refused:
        say(
            {
                "status": STATUS_FAILED,
                "error": BRIDGE_CONNECT_FAILED,
                "state": refused.state,
                "detail": str(refused),
            }
        )
        return 4
    except BridgeUnavailableError as error:
        say({"status": STATUS_FAILED, "error": BRIDGE_CONNECT_FAILED, "detail": str(error)})
        return 4

    say(
        {
            "status": STATUS_CONNECTED,
            "mount_path": mount_path,
            "mode": mode,
            "pid": os.getpid(),
            "status_file": status_file,
        }
    )
    mounted.set()
    try:
        mounter(BridgeOperations(tunnel, mode=mode, uid=uid, gid=gid), mount_path, mode)
    finally:
        ended = tunnel.state
        tunnel.close()
        status.write("unmounted", mount_path=mount_path, mode=mode, ended=ended)
    return 5 if ended in TERMINAL_STATES else 0


def _read_token(args: argparse.Namespace) -> str:
    if args.token_file:
        with open(args.token_file) as handle:
            return handle.read().strip()
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise SystemExit(f"--token-file or {TOKEN_ENV} is required")
    return token


def _read_session_key(args: argparse.Namespace) -> str | None:
    if not args.session_key_file:
        return None
    with open(args.session_key_file) as handle:
        return handle.read().strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="mount a Datalayer local bridge")
    parser.add_argument("--relay-url", required=True)
    parser.add_argument("--mount-path", required=True)
    parser.add_argument("--mode", default="ro", choices=("ro", "rw"))
    parser.add_argument("--bridge-uid", default=None)
    # Neither the token nor the session key travels on the command line:
    # `ps` would show them.
    parser.add_argument("--token-file", default=None)
    parser.add_argument("--session-key-file", default=None)
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--probe", action="store_true", help="answer the fuse probe and exit")
    parser.add_argument(
        "--allow-other",
        action="store_true",
        help="let other users reach the mount; a node agent mounts for a sandbox user",
    )
    parser.add_argument("--uid", type=int, default=None, help="Report files as owned by this uid.")
    parser.add_argument("--gid", type=int, default=None, help="Report files as owned by this gid.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.probe:
        _report_stdout(fuse_probe())
        return 0
    return run_bridge_mount(
        args.relay_url,
        _read_token(args),
        args.mount_path,
        args.mode,
        bridge_uid=args.bridge_uid,
        session_key=_read_session_key(args),
        status_file=args.status_file,
        allow_other=args.allow_other,
        uid=args.uid,
        gid=args.gid,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
