# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The bridge filesystem, driven without a relay, a kernel or FUSE.

What is pinned here is what the mount PROMISES: every operation is a call
over the tunnel and nothing is answered from memory; a `ro` mount refuses
writes before they reach the wire; a tunnel that dropped fails every
operation until it is back — and comes back on its own, with growing delays,
unless the relay refuses the token for good, in which case the mount ends.

The protocol client is a fake over an in-memory tree; the connector is a
fake that fails or refuses on demand; the mounter records what it was given.
The operations class is the real one, and so is the tunnel.
"""

from __future__ import annotations

import errno
import math
import json
import os
import stat
import threading

import pytest

from code_sandboxes import bridge_mount as bm

# --- Fakes -------------------------------------------------------------------


class _Tree:
    """A folder on somebody's laptop, in memory: what the trusted client serves."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {"/notes.txt": b"hello\n", "/data/report.csv": b"a,b\n"}
        self.dirs: set[str] = {"/", "/data", "/empty"}
        self.calls: list[tuple] = []


class _FakeBridgeFS:
    """The eight methods of the protocol, answered from a `_Tree`.

    Every call is recorded, so a test can tell an answer that came over the
    wire from one that did not.
    """

    def __init__(self, tree: _Tree) -> None:
        self.tree = tree
        self.closed = False

    def _note(self, *call) -> None:
        self.tree.calls.append(call)

    def stat(self, path):
        self._note("stat", path)
        if path in self.tree.dirs:
            return {"type": "dir", "size": 0, "mtime": 1_700_000_000}
        if path in self.tree.files:
            return {
                "type": "file",
                "size": len(self.tree.files[path]),
                "mtime": "2026-08-26T09:00:00Z",
            }
        raise FileNotFoundError(errno.ENOENT, path)

    def list(self, path):
        self._note("list", path)
        if path not in self.tree.dirs:
            raise NotADirectoryError(errno.ENOTDIR, path)
        prefix = path.rstrip("/") + "/"
        names = set()
        for candidate in list(self.tree.dirs) + list(self.tree.files):
            if candidate != path and candidate.startswith(prefix):
                names.add(candidate[len(prefix) :].split("/", 1)[0])
        return [{"name": name} for name in sorted(names)]

    def read(self, path, offset, size):
        self._note("read", path, offset, size)
        if path not in self.tree.files:
            raise FileNotFoundError(errno.ENOENT, path)
        return self.tree.files[path][offset : offset + size]

    def write(self, path, offset, data):
        self._note("write", path, offset, data)
        current = bytearray(self.tree.files.get(path, b""))
        current[offset : offset + len(data)] = data
        self.tree.files[path] = bytes(current)
        return len(data)

    def mkdir(self, path):
        self._note("mkdir", path)
        if path in self.tree.dirs:
            raise FileExistsError(errno.EEXIST, path)
        self.tree.dirs.add(path)

    def unlink(self, path):
        self._note("unlink", path)
        if path in self.tree.files:
            del self.tree.files[path]
        elif path in self.tree.dirs:
            self.tree.dirs.discard(path)
        else:
            raise FileNotFoundError(errno.ENOENT, path)

    def rename(self, source, target):
        self._note("rename", source, target)
        self.tree.files[target] = self.tree.files.pop(source)

    def truncate(self, path, size):
        self._note("truncate", path, size)
        self.tree.files[path] = self.tree.files.get(path, b"")[:size]

    def close(self):
        self.closed = True


class _Connector:
    """The relay, as the tunnel sees it: a client, a failure, or a refusal."""

    def __init__(self, tree: _Tree, *, fail_times: int = 0, refuse: str | None = None):
        self.tree = tree
        self.fail_times = fail_times
        self.refuse = refuse
        self.attempts = 0
        self.clients: list[_FakeBridgeFS] = []

    def __call__(self, relay_url, mount_token):
        self.attempts += 1
        assert relay_url.startswith("wss://")
        assert mount_token == "mount-token"
        if self.refuse:
            raise bm.BridgeRefusedError(self.refuse, "relay said no")
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionRefusedError("relay down")
        client = _FakeBridgeFS(self.tree)
        self.clients.append(client)
        return client


class _Dropped(_FakeBridgeFS):
    """A client whose wire goes under the next call."""

    def stat(self, path):
        raise ConnectionResetError("the relay closed the connection")

    read = stat  # type: ignore[assignment]


def _tunnel(connector, **kwargs) -> bm.BridgeTunnel:
    sleeps: list[float] = []
    kwargs.setdefault("sleep", sleeps.append)
    tunnel = bm.BridgeTunnel(
        "wss://relay.test/bridges/br-1", "mount-token", connect=connector, **kwargs
    )
    tunnel.sleeps = sleeps  # type: ignore[attr-defined]
    return tunnel


def _ops(mode: str = "ro") -> tuple[bm.BridgeOperations, _Tree, _Connector]:
    tree = _Tree()
    connector = _Connector(tree)
    tunnel = _tunnel(connector)
    tunnel.open()
    return bm.BridgeOperations(tunnel, mode=mode), tree, connector


# --- Reading -----------------------------------------------------------------


def test_getattr_readdir_and_read_are_calls_over_the_tunnel():
    ops, tree, _ = _ops()

    file_attrs = ops.getattr("/notes.txt")
    dir_attrs = ops.getattr("/data")
    listing = ops.readdir("/", None)
    payload = ops.read("/notes.txt", 3, 1, None)

    assert stat.S_ISREG(file_attrs["st_mode"])
    assert file_attrs["st_size"] == 6
    assert file_attrs["st_mode"] & 0o777 == 0o444  # ro: nobody writes
    assert stat.S_ISDIR(dir_attrs["st_mode"])
    assert dir_attrs["st_nlink"] == 2
    assert listing == [".", "..", "data", "empty", "notes.txt"]
    assert payload == b"ell"
    assert [call[0] for call in tree.calls] == ["stat", "stat", "list", "read"]


def test_a_missing_path_is_enoent_and_not_a_directory_is_enotdir():
    ops, _, _ = _ops()

    with pytest.raises(bm.FuseOSError) as missing:
        ops.getattr("/nope")
    with pytest.raises(bm.FuseOSError) as not_dir:
        ops.readdir("/notes.txt", None)

    assert missing.value.errno == errno.ENOENT
    assert not_dir.value.errno == errno.ENOTDIR


def test_nothing_is_answered_from_memory():
    """Two reads of the same file are two reads over the wire, and a change
    on the person's machine between them is what the second one sees."""
    ops, tree, _ = _ops()

    first = ops.read("/notes.txt", 100, 0, None)
    tree.files["/notes.txt"] = b"changed\n"
    second = ops.read("/notes.txt", 100, 0, None)
    ops.getattr("/notes.txt")
    ops.getattr("/notes.txt")

    assert (first, second) == (b"hello\n", b"changed\n")
    assert [call[0] for call in tree.calls] == ["read", "read", "stat", "stat"]


def test_rw_attributes_are_writable_and_a_timestamp_is_carried():
    ops, _, _ = _ops("rw")
    attrs = ops.getattr("/notes.txt")
    assert attrs["st_mode"] & 0o777 == 0o644
    assert int(attrs["st_mtime"]) == 1_787_734_800  # 2026-08-26T09:00:00Z


# --- Writing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "operation",
    [
        lambda ops: ops.write("/notes.txt", b"x", 0, None),
        lambda ops: ops.create("/new.txt", 0o644),
        lambda ops: ops.truncate("/notes.txt", 0),
        lambda ops: ops.unlink("/notes.txt"),
        lambda ops: ops.rmdir("/empty"),
        lambda ops: ops.mkdir("/made", 0o755),
        lambda ops: ops.rename("/notes.txt", "/moved.txt"),
        lambda ops: ops.open("/notes.txt", os.O_WRONLY),
        lambda ops: ops.access("/notes.txt", os.W_OK),
        lambda ops: ops.chmod("/notes.txt", 0o600),
    ],
)
def test_a_ro_mount_refuses_every_write_before_the_wire(operation):
    ops, tree, _ = _ops("ro")

    with pytest.raises(bm.FuseOSError) as refused:
        operation(ops)

    assert refused.value.errno == errno.EROFS
    assert tree.calls == [], "the refusal never reached the protocol client"
    assert tree.files["/notes.txt"] == b"hello\n"


def test_a_rw_mount_writes_through_to_the_person_s_folder():
    ops, tree, _ = _ops("rw")

    ops.create("/new.txt", 0o644)
    written = ops.write("/new.txt", b"fresh", 0, None)
    ops.mkdir("/made", 0o755)
    ops.rename("/new.txt", "/made/new.txt")
    ops.truncate("/made/new.txt", 2)
    ops.unlink("/notes.txt")
    ops.rmdir("/empty")

    assert written == 5
    assert tree.files == {"/data/report.csv": b"a,b\n", "/made/new.txt": b"fr"}
    assert tree.dirs == {"/", "/data", "/made"}
    assert ops.open("/made/new.txt", os.O_RDWR) == 0


# --- The tunnel ----------------------------------------------------------------


def test_a_dropped_connection_fails_the_operation_and_everything_after_it():
    """`EIO` at the moment the wire goes, `ENOTCONN` while it is down — and
    the protocol client is never asked again until it is back."""
    tree = _Tree()
    connector = _Connector(tree)
    tunnel = _tunnel(connector, sleep=lambda _delay: None)
    tunnel.open()
    ops = bm.BridgeOperations(tunnel, mode="ro")
    connector.fail_times = 10_000  # never comes back in this test
    tunnel._client = _Dropped(tree)  # the wire goes under the next call

    with pytest.raises(bm.FuseOSError) as dropped:
        ops.getattr("/notes.txt")
    assert dropped.value.errno == errno.EIO
    assert tunnel.state == "reconnecting"

    for operation in (
        lambda: ops.getattr("/notes.txt"),
        lambda: ops.readdir("/", None),
        lambda: ops.read("/notes.txt", 10, 0, None),
    ):
        with pytest.raises(bm.FuseOSError) as down:
            operation()
        assert down.value.errno == errno.ENOTCONN
    assert tree.calls == [], "nothing reached the protocol client while the tunnel was down"
    tunnel.close()


def test_the_tunnel_reconnects_with_growing_delays_and_serves_again():
    tree = _Tree()
    connector = _Connector(tree)
    reconnected = threading.Event()
    tunnel = _tunnel(
        connector,
        backoff=(1.0, 4.0),
        on_state=lambda state: reconnected.set() if state == "connected" else None,
    )
    tunnel.open()
    reconnected.clear()
    connector.fail_times = 3  # down for three attempts, then back
    ops = bm.BridgeOperations(tunnel, mode="ro")

    tunnel.lost(ConnectionResetError("gone"))
    assert reconnected.wait(5), "the tunnel did not come back"

    assert tunnel.state == "connected"
    assert tunnel.sleeps == [1.0, 2.0, 4.0, 4.0]  # doubled, then capped
    assert connector.attempts == 1 + 4
    assert ops.read("/notes.txt", 5, 0, None) == b"hello"
    tunnel.close()


@pytest.mark.parametrize("verdict", ["revoked", "expired"])
def test_a_relay_that_refuses_the_token_ends_the_tunnel(verdict):
    tree = _Tree()
    connector = _Connector(tree)
    states: list[str] = []
    ended = threading.Event()

    def on_state(state):
        states.append(state)
        if state in bm.TERMINAL_STATES:
            ended.set()

    tunnel = _tunnel(connector, on_state=on_state)
    tunnel.open()
    connector.refuse = verdict

    tunnel.lost(ConnectionResetError("gone"))
    assert ended.wait(5)

    assert tunnel.state == verdict
    assert states == ["connected", "reconnecting", verdict]
    assert tunnel.ended
    with pytest.raises(bm.FuseOSError) as refused:
        bm.BridgeOperations(tunnel).getattr("/")
    assert refused.value.errno == errno.ENOTCONN
    # No more attempts: the token will not be taken back.
    assert connector.attempts == 2


def test_opening_against_a_refusing_relay_says_which_state():
    connector = _Connector(_Tree(), refuse="expired")
    tunnel = _tunnel(connector)
    with pytest.raises(bm.BridgeRefusedError) as refused:
        tunnel.open()
    assert refused.value.state == "expired"
    assert tunnel.state == "expired"


def test_opening_against_an_unreachable_relay_is_unavailable_not_refused():
    connector = _Connector(_Tree(), fail_times=1)
    tunnel = _tunnel(connector)
    with pytest.raises(bm.BridgeUnavailableError):
        tunnel.open()
    assert tunnel.state == "disconnected"


def test_control_frames_from_the_relay_name_states():
    assert bm.control_state(json.dumps({"state": "revoked"})) == "revoked"
    assert bm.control_state(json.dumps({"error": "token expired"})) == "expired"
    assert bm.control_state(json.dumps({"state": "connected"})) == "connected"
    assert bm.control_state("bridge revoked by owner") == "revoked"
    assert bm.control_state(b"binary frame") is None
    assert bm.control_state(json.dumps({"hello": 1})) is None


# --- Mutation: disconnected serves stale data (must be caught) ---------------


def _assert_disconnected_never_serves(operations_type) -> None:
    """The check every filesystem of this module must pass: once the tunnel
    is down, a read that answered a moment ago must fail, not repeat."""
    tree = _Tree()
    connector = _Connector(tree)
    tunnel = _tunnel(connector, sleep=lambda _delay: None)
    tunnel.open()
    ops = operations_type(tunnel, mode="ro")
    assert ops.read("/notes.txt", 100, 0, None) == b"hello\n"
    assert stat.S_ISREG(ops.getattr("/notes.txt")["st_mode"])
    # The relay never takes the token back — and never comes back either. A
    # finite count is burned through in an instant with a no-op sleep, and a
    # tunnel that reconnected would answer honestly over the wire, which is
    # not what this check is about.
    connector.fail_times = math.inf
    tunnel.lost(ConnectionResetError("gone"))
    for operation in (
        lambda: ops.read("/notes.txt", 100, 0, None),
        lambda: ops.getattr("/notes.txt"),
    ):
        try:
            answer = operation()
        except OSError as error:
            assert error.errno in (errno.EIO, errno.ENOTCONN)
        else:
            raise AssertionError(f"a disconnected mount answered {answer!r} from memory")
    tunnel.close()


class _StaleOperations(bm.BridgeOperations):
    """The mutation: a filesystem that remembers, and answers when it cannot ask."""

    def __init__(self, tunnel, *, mode="ro"):
        super().__init__(tunnel, mode=mode)
        self._reads: dict = {}
        self._attrs: dict = {}

    def read(self, path, size, offset, fh=None):
        try:
            self._reads[path] = super().read(path, size, offset, fh)
        except OSError:
            if path in self._reads:
                return self._reads[path]
            raise
        return self._reads[path]

    def getattr(self, path, fh=None):
        try:
            self._attrs[path] = super().getattr(path, fh)
        except OSError:
            if path in self._attrs:
                return self._attrs[path]
            raise
        return self._attrs[path]


def test_the_real_operations_never_serve_stale_data():
    _assert_disconnected_never_serves(bm.BridgeOperations)


def test_a_filesystem_that_serves_stale_data_while_disconnected_is_caught():
    with pytest.raises(AssertionError, match="from memory"):
        _assert_disconnected_never_serves(_StaleOperations)


# --- The launcher ----------------------------------------------------------------


class _Mounter:
    """Stands in for FUSE: records what it was asked to mount, and returns."""

    def __init__(self) -> None:
        self.mounted: list[tuple] = []
        self.during: list = []

    def __call__(self, operations, mount_path, mode):
        self.mounted.append((mount_path, mode, operations.read_only))
        # While "mounted", the filesystem works.
        self.during.append(operations.readdir("/", None))


def test_the_launcher_answers_connected_first_then_mounts(tmp_path):
    tree = _Tree()
    connector = _Connector(tree)
    mounter = _Mounter()
    said: list[dict] = []
    status_file = tmp_path / "bridge.status"

    code = bm.run_bridge_mount(
        "wss://relay.test/bridges/br-1",
        "mount-token",
        str(tmp_path / "mnt"),
        "ro",
        connect=connector,
        mount=mounter,
        status_file=str(status_file),
        report=said.append,
    )

    assert code == 0
    assert said[0]["status"] == "connected"
    assert said[0]["mount_path"] == str(tmp_path / "mnt")
    assert said[0]["mode"] == "ro"
    assert len(said) == 1, "one line, and nothing else, on stdout"
    assert mounter.mounted == [(str(tmp_path / "mnt"), "ro", True)]
    assert mounter.during == [[".", "..", "data", "empty", "notes.txt"]]
    assert (tmp_path / "mnt").is_dir()
    # The status file tells a later reconcile what became of the mount.
    assert json.loads(status_file.read_text())["state"] == "unmounted"
    assert connector.clients[0].closed


def test_the_launcher_answers_failed_when_the_relay_cannot_be_reached(tmp_path):
    said: list[dict] = []
    mounter = _Mounter()

    code = bm.run_bridge_mount(
        "wss://relay.test/bridges/br-1",
        "mount-token",
        str(tmp_path / "mnt"),
        "rw",
        connect=_Connector(_Tree(), fail_times=1),
        mount=mounter,
        report=said.append,
    )

    assert code != 0
    assert said == [
        {
            "status": "failed",
            "error": "BRIDGE_CONNECT_FAILED",
            "detail": "ConnectionRefusedError: relay down",
        }
    ]
    assert mounter.mounted == []


def test_the_launcher_answers_failed_with_the_state_when_the_relay_refuses(tmp_path):
    said: list[dict] = []
    code = bm.run_bridge_mount(
        "wss://relay.test/bridges/br-1",
        "mount-token",
        str(tmp_path / "mnt"),
        "ro",
        connect=_Connector(_Tree(), refuse="revoked"),
        mount=_Mounter(),
        report=said.append,
    )
    assert code != 0
    assert said[0]["status"] == "failed"
    assert said[0]["error"] == "BRIDGE_CONNECT_FAILED"
    assert said[0]["state"] == "revoked"


def test_the_launcher_refuses_a_mode_it_does_not_know(tmp_path):
    said: list[dict] = []
    assert bm.run_bridge_mount("wss://r", "mount-token", str(tmp_path), "rwx", report=said.append)
    assert said == [{"status": "failed", "error": "INVALID_MODE", "detail": "mode 'rwx'"}]


def test_a_revocation_while_mounted_unmounts_and_exits_non_zero(tmp_path, monkeypatch):
    """The relay takes the token back mid-flight: the tunnel goes terminal,
    the mount is taken down, and the launcher's exit code says so."""
    tree = _Tree()
    connector = _Connector(tree)
    unmounted: list[str] = []
    monkeypatch.setattr(bm, "unmount", lambda path: unmounted.append(path) or True)
    said: list[dict] = []

    def mounter(operations, mount_path, mode):
        # The person revokes the bridge while the sandbox is using it.
        connector.refuse = "revoked"
        operations.tunnel.lost(ConnectionResetError("gone"))
        assert operations.tunnel._reconnector.join(5) is None
        with pytest.raises(bm.FuseOSError):
            operations.getattr("/")

    code = bm.run_bridge_mount(
        "wss://relay.test/bridges/br-1",
        "mount-token",
        str(tmp_path / "mnt"),
        "ro",
        connect=connector,
        mount=mounter,
        report=said.append,
        sleep=lambda _delay: None,
    )

    assert code == 5
    assert unmounted == [str(tmp_path / "mnt")]
    assert said[0]["status"] == "connected"


def test_without_fusepy_the_real_mount_is_refused_up_front(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "_FUSE", None)
    said: list[dict] = []
    code = bm.run_bridge_mount(
        "wss://relay.test/bridges/br-1",
        "mount-token",
        str(tmp_path / "mnt"),
        "ro",
        connect=_Connector(_Tree()),
        report=said.append,
    )
    assert code == 3
    assert said[0]["error"] == "FUSE_UNAVAILABLE"
    assert "fusepy" in said[0]["detail"]


def test_the_probe_says_what_the_sandbox_has(monkeypatch):
    monkeypatch.setattr(bm, "_FUSE", object())
    monkeypatch.setattr(bm.os.path, "exists", lambda path: path == "/dev/fuse")
    assert bm.fuse_probe()["ok"] is True
    monkeypatch.setattr(bm, "_FUSE", None)
    assert bm.fuse_probe()["ok"] is False


def test_the_command_line_takes_the_token_from_a_file_never_argv(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text("mount-token\n")
    seen: dict = {}

    def run(relay_url, mount_token, mount_path, mode, **kwargs):
        seen.update(relay_url=relay_url, mount_token=mount_token, mode=mode, **kwargs)
        return 0

    monkeypatch.setattr(bm, "run_bridge_mount", run)
    assert (
        bm.main(
            [
                "--relay-url",
                "wss://relay.test/bridges/br-1",
                "--mount-path",
                str(tmp_path / "mnt"),
                "--mode",
                "rw",
                "--token-file",
                str(token_file),
                "--status-file",
                str(tmp_path / "s"),
            ]
        )
        == 0
    )
    assert seen["mount_token"] == "mount-token"
    assert seen["mode"] == "rw"
    assert seen["status_file"] == str(tmp_path / "s")
    with pytest.raises(SystemExit):
        bm.main(["--relay-url", "wss://r", "--mount-path", str(tmp_path)])


# --- The relay handshake -------------------------------------------------------


class _Socket:
    """One websocket, faked: what was sent, and what the relay answers."""

    def __init__(self, answers=()):
        self.sent: list = []
        self.answers = list(answers)
        self.closed = False

    def send(self, frame):
        self.sent.append(frame)

    def recv(self):
        if not self.answers:
            raise ConnectionResetError("the relay closed the connection")
        return self.answers.pop(0)

    def close(self):
        self.closed = True


def _protocol_module(monkeypatch, *, established: list):
    """`datalayer_common.content_bridge`, faked: a client that records its
    transport and channel, a channel that records the handshake."""
    import sys
    import types

    module = types.ModuleType("datalayer_common.content_bridge")

    class SecureChannel:
        def __init__(self, *, role, bridge_uid, session_key=b""):
            self.role, self.bridge_uid, self.session_key = role, bridge_uid, session_key

        def hello(self):
            return b"mount-public-key"

        def establish(self, peer_hello):
            established.append((self.role, self.bridge_uid, self.session_key, peer_hello))

    class BridgeFileSystemClient:
        def __init__(self, transport, *, channel=None):
            self.transport, self.channel = transport, channel

        def rmdir(self, path):
            return ("rmdir", path)

    class BridgeProtocolError(Exception):
        pass

    module.SecureChannel = SecureChannel
    module.BridgeFileSystemClient = BridgeFileSystemClient
    module.BridgeProtocolError = BridgeProtocolError
    monkeypatch.setitem(sys.modules, "datalayer_common.content_bridge", module)
    return module


def test_the_relay_handshake_is_the_hello_then_the_channel_keys(monkeypatch):
    established: list = []
    _protocol_module(monkeypatch, established=established)
    socket = _Socket(answers=[b"client-public-key"])
    monkeypatch.setattr(bm, "_open_websocket", lambda url: socket)

    client = bm.connect_relay(
        "wss://relay.test/bridges/br-1", "mount-token", bridge_uid="br-1", session_key="ab" * 32
    )

    # The hello to the relay first — a text frame, role and token — then
    # this end's public key, and the peer's is what the channel derives from.
    assert json.loads(socket.sent[0]) == {"role": "mount", "token": "mount-token"}
    assert socket.sent[1] == b"mount-public-key"
    assert established == [("mount", "br-1", "ab" * 32, b"client-public-key")]
    # The protocol client speaks over the socket, sealed by that channel.
    assert client.channel.bridge_uid == "br-1"
    assert client.transport.send is not None and client.transport.recv is not None
    client.close()
    assert socket.closed


def test_without_a_session_key_the_frames_travel_as_the_relay_forwards_them(monkeypatch):
    established: list = []
    _protocol_module(monkeypatch, established=established)
    socket = _Socket()
    monkeypatch.setattr(bm, "_open_websocket", lambda url: socket)

    client = bm.relay_connector("br-1", None)("wss://relay.test/bridges/br-1", "mount-token")

    assert len(socket.sent) == 1
    assert established == []
    assert client.channel is None


def test_a_session_key_without_the_bridge_uid_is_refused_before_the_wire(monkeypatch):
    _protocol_module(monkeypatch, established=[])
    socket = _Socket()
    monkeypatch.setattr(bm, "_open_websocket", lambda url: socket)
    with pytest.raises(bm.BridgeUnavailableError, match="bridge uid"):
        bm.connect_relay("wss://r", "mount-token", session_key="ab" * 32)
    assert socket.closed


def test_a_relay_that_revokes_during_the_handshake_is_a_refusal(monkeypatch):
    _protocol_module(monkeypatch, established=[])
    socket = _Socket(answers=[json.dumps({"state": "revoked"})])
    monkeypatch.setattr(bm, "_open_websocket", lambda url: socket)
    with pytest.raises(bm.BridgeRefusedError) as refused:
        bm.connect_relay("wss://r", "mount-token", bridge_uid="br-1", session_key="ab" * 32)
    assert refused.value.state == "revoked"
    assert socket.closed


def test_the_protocol_s_own_channel_error_is_a_lost_connection(monkeypatch):
    """A frame the channel cannot authenticate ends the channel: the tunnel
    reconnects, the operation fails with EIO, nothing is served meanwhile."""
    module = _protocol_module(monkeypatch, established=[])
    tree = _Tree()

    class Broken(_FakeBridgeFS):
        def stat(self, path):
            raise module.BridgeProtocolError("frame failed authentication")

    socket = _Socket(answers=[b"client-public-key"])
    monkeypatch.setattr(bm, "_open_websocket", lambda url: socket)
    monkeypatch.setattr(
        module, "BridgeFileSystemClient", lambda transport, channel=None: Broken(tree)
    )
    connector = bm.relay_connector("br-1", "ab" * 32)
    tunnel = _tunnel(connector, sleep=lambda _delay: None)
    tunnel.open()
    ops = bm.BridgeOperations(tunnel, mode="ro")

    with pytest.raises(bm.FuseOSError) as dropped:
        ops.getattr("/notes.txt")
    assert dropped.value.errno == errno.EIO
    assert tunnel.state == "reconnecting"
    assert tree.calls == []
    tunnel.close()


def test_rmdir_is_the_client_s_own_where_it_has_one():
    ops, tree, _connector = _ops("rw")

    ops.rmdir("/empty")
    assert tree.calls[-1] == ("unlink", "/empty")  # the fake has no rmdir

    class WithRmdir(_FakeBridgeFS):
        def rmdir(self, path):
            self._note("rmdir", path)
            self.tree.dirs.discard(path)

    ops.tunnel._client = WithRmdir(tree)
    ops.mkdir("/again", 0o755)
    ops.rmdir("/again")
    assert tree.calls[-1] == ("rmdir", "/again")
    assert "/again" not in tree.dirs


def test_the_command_line_carries_the_bridge_uid_and_reads_the_session_key_from_a_file(
    tmp_path, monkeypatch
):
    (tmp_path / "token").write_text("mount-token\n")
    (tmp_path / "session").write_text("ab" * 32 + "\n")
    seen: dict = {}

    def run(relay_url, mount_token, mount_path, mode, **kwargs):
        seen.update(mount_token=mount_token, **kwargs)
        return 0

    monkeypatch.setattr(bm, "run_bridge_mount", run)
    bm.main(
        [
            "--relay-url", "wss://relay.test/bridges/br-1",
            "--mount-path", str(tmp_path / "mnt"),
            "--bridge-uid", "br-1",
            "--token-file", str(tmp_path / "token"),
            "--session-key-file", str(tmp_path / "session"),
        ]
    )
    assert seen["mount_token"] == "mount-token"
    assert seen["bridge_uid"] == "br-1"
    assert seen["session_key"] == "ab" * 32
