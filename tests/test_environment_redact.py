# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A build log's secrets are taken out before it is stored (PLAN_ENV.md, D-15, E1-16).

Two kinds: the build's secret values, in the forms a command prints them in,
and credential-shaped strings nobody listed. Over a whole text, and over output
that arrives in reads, where a secret cut in two by a read must still never be
released. The service storing a chunk redacts it again, so a chunk redacted
here must come back unchanged.
"""

from __future__ import annotations

import base64
import itertools
import json
from urllib.parse import quote

import pytest

from code_sandboxes.environments.redact import (
    MAX_LOG_CHUNK_BYTES,
    REDACTED,
    BuildLogChunker,
    LogChunk,
    redact,
    secret_forms,
)

#: A secret that looks like nothing: only its value can find it.
SECRET = "correct-horse-battery-staple-7391"
#: A multi-line secret, such as a key file, with a line too short to look for alone.
MULTILINE = "-----first line of a key file-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nshort\n"

# Credentials with a shape, assembled so no scanner mistakes them for real ones.
GITHUB = "gh" + "p_" + "Zq3" * 12
AWS = "AKIA" + "IOSFODNN7EXAMPLE"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZGEifQ.c2lnbmF0dXJl"
OPENAI = "sk-" + "proj" + "x1" * 12
KEY_BODY = ("b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ", "AAAAMwAAAAtzc2gtZWQyNTUxOQAAACD")
KEY_BLOCK = (
    "-----BEGIN OPENSSH PRIVATE KEY-----\n"
    + "\n".join(KEY_BODY)
    + "\n-----END OPENSSH PRIVATE KEY-----"
)

SHAPES = [
    (f"export GITHUB_TOKEN={GITHUB}", f"export GITHUB_TOKEN={REDACTED}"),
    (
        f"remote: https://{GITHUB}@github.com/org/repo",
        f"remote: https://{REDACTED}@github.com/org/repo",
    ),
    (f"AWS_ACCESS_KEY_ID={AWS}", f"AWS_ACCESS_KEY_ID={REDACTED}"),
    (
        "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        f"aws_secret_access_key = {REDACTED}",
    ),
    (f"Authorization: Bearer {JWT}", f"Authorization: Bearer {REDACTED}"),
    (
        "curl -H 'Authorization: Basic dXNlcjpwYXNz' https://x.example",
        f"curl -H 'Authorization: Basic {REDACTED}' https://x.example",
    ),
    (f"token {JWT} expired", f"token {REDACTED} expired"),
    (
        "Looking in indexes: https://build:s3cr3t-pass@pypi.example/simple",
        f"Looking in indexes: https://build:{REDACTED}@pypi.example/simple",
    ),
    (
        '{"password": "hunter2 with spaces", "user": "ada"}',
        f'{{"password": "{REDACTED}", "user": "ada"}}',
    ),
    (
        "docker login -u AWS -p abcdef123456 registry.example",
        f"docker login -u AWS -p {REDACTED} registry.example",
    ),
    ("uv publish --token pypi-AgEIcHlwaS5vcmc", f"uv publish --token {REDACTED}"),
    (
        "GET https://bucket.example/o?X-Amz-Signature=abc123&X-Amz-Date=2026",
        f"GET https://bucket.example/o?X-Amz-Signature={REDACTED}&X-Amz-Date=2026",
    ),
    (f"OPENAI_API_KEY is {OPENAI}", f"OPENAI_API_KEY is {REDACTED}"),
    (f"client = OpenAI(api_key='{OPENAI}')", f"client = OpenAI(api_key='{REDACTED}')"),
    (
        f"$ cat id\n{KEY_BLOCK}\nnext\n",
        "$ cat id\n-----BEGIN OPENSSH PRIVATE KEY-----"
        f"{REDACTED}-----END OPENSSH PRIVATE KEY-----\nnext\n",
    ),
]

#: Build output a redactor must leave alone.
ORDINARY = [
    "Collecting tokenizers==0.19.1",
    "Successfully installed geopandas-1.1.1 rasterio-1.4.3 shapely-2.0.6",
    "max_tokens: 1024",
    "task-runner-abcdefghijklmnopqrstuvwxyz finished",
    "PWD=/home/datalayer/content",
    "sha256:4f2a8b0c9d1e7f3a5b6c8d0e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e8f0a",
    "docker login --password-stdin registry.example",
    "Downloading https://files.pythonhosted.org/packages/ab/cd/numpy-2.2.5.tar.gz",
    "Bearer tokens are sent by the client",
    "authorization failed for this index",
    "Using index https://pypi.org/simple",
]


def reads(text: str, size: int) -> list[str]:
    return [text[position : position + size] for position in range(0, len(text), size)]


def chunked(text: str, secrets=(), *, size: int = 1024, **options) -> list[LogChunk]:
    """The chunks a chunker answers for `text` read `size` characters at a time."""
    chunker = BuildLogChunker(secrets, **options)
    chunks: list[LogChunk] = []
    for piece in reads(text, size):
        chunks += chunker.feed(piece)
    return chunks + chunker.close()


def joined(chunks: list[LogChunk]) -> str:
    return "".join(chunk.text for chunk in chunks)


# -- a whole text --------------------------------------------------------------------


def test_a_secret_value_is_redacted_in_every_form_a_command_prints_it_in() -> None:
    data = SECRET.encode()
    printed = [
        SECRET,
        base64.b64encode(data).decode(),
        base64.b64encode(data + b"\n").decode(),
        base64.b64encode(data + b"\n").decode().rstrip("="),
        base64.urlsafe_b64encode(data + b"\n").decode(),
        quote(f"{SECRET}/?", safe="").replace("%2F%3F", ""),
    ]
    for form in printed:
        assert redact(f"printed {form} done\n", [SECRET]) == f"printed {REDACTED} done\n"


def test_a_multi_line_secret_is_redacted_escaped_and_line_by_line() -> None:
    document = json.dumps({"key": MULTILINE})
    assert redact(document, [MULTILINE]) == f'{{"key": "{REDACTED}"}}'
    assert redact("MIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n", [MULTILINE]) == f"{REDACTED}\n"
    assert redact("short\n", [MULTILINE]) == "short\n"


@pytest.mark.parametrize(("line", "expected"), SHAPES)
def test_a_credential_shape_is_redacted_without_being_told(line: str, expected: str) -> None:
    assert redact(line) == expected


@pytest.mark.parametrize("line", ORDINARY)
def test_ordinary_build_output_is_left_alone(line: str) -> None:
    assert redact(line) == line


def test_a_private_key_block_keeps_nothing_of_its_body_however_it_is_printed() -> None:
    word_split = " ".join(KEY_BLOCK.split("\n"))
    escaped = json.dumps({"private_key": KEY_BLOCK})
    unterminated = KEY_BLOCK.rsplit("\n", 1)[0] + "\nDone\n"
    for printed in (KEY_BLOCK, word_split, escaped, unterminated):
        redacted = redact(printed)
        assert all(line not in redacted for line in KEY_BODY), printed


def test_redacting_redacted_text_changes_nothing() -> None:
    """What the service storing a chunk relies on to store what the worker sent."""
    fragments = [line for line, _expected in SHAPES] + ORDINARY + [f"{SECRET} ", "=", "\n"]
    for first, second in itertools.product(fragments, repeat=2):
        once = redact(first + second)
        assert redact(once) == once, first + second


def test_secret_forms_are_longest_first_and_skip_what_is_not_a_value() -> None:
    forms = secret_forms(["", SECRET, None])  # type: ignore[list-item]
    assert forms == sorted(forms, key=lambda form: (-len(form), form))
    assert SECRET in forms and "" not in forms
    assert base64.b64encode(f"{SECRET}\n".encode()).decode() in forms
    assert secret_forms([]) == []


# -- output that arrives in reads ------------------------------------------------------


def test_a_secret_cut_by_a_read_is_redacted_wherever_the_read_ends() -> None:
    text = f"ordinary output\nnow {SECRET} printed\nand GITHUB_TOKEN={GITHUB} too\nlast"
    expected = redact(text, [SECRET])
    for cut in range(1, len(text)):
        chunker = BuildLogChunker([SECRET], max_bytes=256)
        chunks = chunker.feed(text[:cut]) + chunker.flush()
        chunks += chunker.feed(text[cut:]) + chunker.flush() + chunker.close()
        assert all(SECRET not in chunk.text and GITHUB not in chunk.text for chunk in chunks)
        assert joined(chunks) == expected, cut


def test_a_private_key_read_a_character_at_a_time_is_not_released_in_parts() -> None:
    text = f"$ cat id\n{KEY_BLOCK}\nbuilt\n"
    for size in (1, 7, 64):
        chunks = chunked(text, size=size, max_bytes=256)
        assert joined(chunks) == redact(text), size
        assert all(line not in chunk.text for line in KEY_BODY for chunk in chunks)


def test_chunks_hold_at_most_their_bytes_end_after_a_line_and_are_numbered_in_order() -> None:
    lines = "".join(
        f"line {n}: héllo wörld ✓ {SECRET if n % 7 == 0 else ''}\n" for n in range(400)
    )
    chunks = chunked(lines, [SECRET], size=37, max_bytes=1024, first_sequence=5)
    assert len(chunks) > 3
    assert [chunk.sequence for chunk in chunks] == list(range(5, 5 + len(chunks)))
    assert all(chunk.size <= 1024 and chunk.text.endswith("\n") for chunk in chunks)
    assert joined(chunks) == lines.replace(SECRET, REDACTED)
    assert all(redact(chunk.text) == chunk.text for chunk in chunks)


def test_the_default_chunk_is_64_kib() -> None:
    output = "".join(f"Downloading package-{n}-1.0.0.whl ({n} kB)\n" for n in range(6000))
    chunks = chunked(output, size=4093)
    assert MAX_LOG_CHUNK_BYTES == 64 * 1024
    assert len(chunks) > 3 and all(chunk.size <= MAX_LOG_CHUNK_BYTES for chunk in chunks)
    assert max(chunk.size for chunk in chunks) > MAX_LOG_CHUNK_BYTES - 1024
    assert joined(chunks) == output


def test_a_line_longer_than_the_chunker_holds_is_released_without_its_secret() -> None:
    line = ("x" * 3000 + " " + SECRET + " ") * 10
    chunker = BuildLogChunker([SECRET], max_bytes=1024, max_held=2048)
    chunks: list[LogChunk] = []
    for piece in reads(line, 500):
        chunks += chunker.feed(piece)
    assert chunks, "a line with no end is not held whole"
    chunks += chunker.close()
    assert all(SECRET not in chunk.text for chunk in chunks)
    assert joined(chunks) == line.replace(SECRET, REDACTED)


def test_flush_answers_the_lines_ready_and_holds_the_line_not_yet_ended() -> None:
    chunker = BuildLogChunker([SECRET])
    assert chunker.feed("step 1\nstep 2 prints ") == []
    assert chunker.flush() == [LogChunk(0, "step 1\n")]
    assert chunker.flush() == []
    assert chunker.feed(SECRET[:10]) == []
    assert chunker.flush() == []
    chunker.feed(SECRET[10:] + " then ends\n")
    assert chunker.flush() == [LogChunk(1, f"step 2 prints {REDACTED} then ends\n")]
    assert chunker.next_sequence == 2
    assert chunker.close() == []
    assert chunker.close() == []


def test_a_chunker_refuses_what_it_cannot_number_or_hold() -> None:
    with pytest.raises(ValueError, match="counted from 0"):
        BuildLogChunker(first_sequence=-1)
    with pytest.raises(TypeError):
        BuildLogChunker(first_sequence=True)
    with pytest.raises(ValueError, match="at least"):
        BuildLogChunker(max_bytes=100)
    chunker = BuildLogChunker()
    chunker.close()
    with pytest.raises(ValueError, match="closed"):
        chunker.feed("more\n")


def test_output_that_is_not_valid_unicode_is_stored_replaced() -> None:
    undecodable = b"half \xff a character\n".decode("utf-8", "surrogateescape")
    (chunk,) = chunked(undecodable)
    assert chunk.text == "half ? a character\n" and chunk.size == len(chunk.text)
