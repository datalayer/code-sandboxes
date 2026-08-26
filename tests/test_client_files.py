# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""The client's filesystem, which is the one a file browser and a terminal use.

A workflow written against a sandbox — list a directory, open a file, save it
back — must not know which provider is underneath. `CodeSandboxClient` is what
`agent-runtimes` and `datalayer-runtimes` are meant to consume, so the
filesystem has to be reachable *there*, not only on the provider object behind
it. These tests use the in-process variant, so what they prove is the shape of
the surface and the behaviour every provider inherits from it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_sandboxes.client import CodeSandboxClient
from code_sandboxes.eval_sandbox import EvalSandbox
from code_sandboxes.filesystem import FileInfo


@pytest.fixture
def client() -> CodeSandboxClient:
    sandbox = CodeSandboxClient(EvalSandbox(), owns_sandbox=True)
    yield sandbox
    sandbox.close()


def test_a_file_written_through_the_client_is_read_back_through_it(
    client, tmp_path: Path
) -> None:
    target = tmp_path / "notes.txt"
    client.write_file(str(target), "hello from the sandbox\n")

    assert client.read_file(str(target)) == "hello from the sandbox\n"
    assert client.read_file(str(target), binary=True) == b"hello from the sandbox\n"


def test_the_directories_above_a_file_are_made_by_default(client, tmp_path: Path) -> None:
    """A browser saving into a new folder should not have to make it first."""
    target = tmp_path / "research" / "deep" / "notes.txt"
    client.write_file(str(target), "written\n")

    assert target.read_text() == "written\n"


def test_listing_says_what_each_entry_is(client, tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    (tmp_path / "report.csv").write_text("a,b\n1,2\n")

    entries = {entry.name: entry for entry in client.list_files(str(tmp_path))}

    assert entries["folder"].is_directory
    assert entries["report.csv"].is_file
    assert entries["report.csv"].size == len("a,b\n1,2\n")


def test_one_entry_can_be_asked_about_on_its_own(client, tmp_path: Path) -> None:
    (tmp_path / "report.csv").write_text("a,b\n")

    info = client.stat_file(str(tmp_path / "report.csv"))

    assert isinstance(info, FileInfo)
    assert info.is_file
    assert info.name == "report.csv"


def test_a_path_that_names_nothing_is_an_error_not_an_empty_answer(
    client, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        client.stat_file(str(tmp_path / "absent.txt"))
    with pytest.raises(FileNotFoundError):
        client.read_file(str(tmp_path / "absent.txt"))


def test_a_large_file_is_read_in_pieces(client, tmp_path: Path) -> None:
    """The shape a caller writes against, whatever the provider can do."""
    payload = b"x" * (3 * 1024)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)

    chunks = list(client.stream_file(str(target), chunk_size=1024))

    assert len(chunks) == 3
    assert b"".join(chunks) == payload


def test_an_empty_file_streams_as_nothing(client, tmp_path: Path) -> None:
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")

    assert list(client.stream_file(str(target))) == []


def test_a_directory_is_made_and_removed_through_the_client(
    client, tmp_path: Path
) -> None:
    folder = tmp_path / "scratch"
    client.make_directory(str(folder))
    assert folder.is_dir()

    (folder / "inside.txt").write_text("x")
    client.delete_file(str(folder), recursive=True)
    assert not folder.exists()


def test_a_file_goes_in_and_comes_out(client, tmp_path: Path) -> None:
    source = tmp_path / "local.txt"
    source.write_text("from the laptop\n")
    inside = tmp_path / "inside" / "copy.txt"
    back = tmp_path / "back.txt"

    client.make_directory(str(inside.parent))
    client.upload_file(str(source), str(inside))
    client.download_file(str(inside), str(back))

    assert back.read_text() == "from the laptop\n"


def test_the_sandbox_starts_itself_when_the_files_are_asked_for(tmp_path: Path) -> None:
    """Asking about files before starting is a mistake worth answering."""
    client = CodeSandboxClient(EvalSandbox(), owns_sandbox=True)
    try:
        assert not client.is_started
        client.write_file(str(tmp_path / "started.txt"), "yes\n")
        assert client.is_started
    finally:
        client.close()
