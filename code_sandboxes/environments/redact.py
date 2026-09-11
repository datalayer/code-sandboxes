# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""A build's log with its secrets taken out, before any of it is stored (PLAN_ENV.md, D-15).

A build prints what its commands print, and a command can print a secret: a
step that echoes a variable, an index URL with its password, a token in a
traceback. The log is stored and read by everyone who may read the build, so
what reaches storage is redacted first, for two kinds of secret:

- **The build's secret values**, which only the process running the build
  holds. Each is replaced wherever it appears, and so are the forms a command
  most often prints one in: base64 (with and without the newline ``echo``
  adds), URL-quoted, JSON-escaped, and each line of a multi-line value.
- **Credential-shaped strings**, which need no list: tokens known by their
  prefix, JWTs, private key blocks, the password in a URL, an
  ``Authorization`` header or a bearer token, a credential passed as a flag,
  and the value assigned to a credential-named key such as ``GITHUB_TOKEN``.

:func:`redact` does both over a whole text. :class:`BuildLogChunker` does both
over output that arrives piece by piece, which is where a naive redactor leaks:
a value split across two reads matches in neither. The chunker holds back the
line not yet ended, and as much as a multi-line value could still be
completing, until the rest arrives. What it releases it cuts into chunks of at
most 64 KiB, each numbered with the sequence it is stored under.

Redacting the credential shapes is a fixed point: redacting redacted text
changes nothing. The service that stores a chunk therefore redacts it again,
without the secret values it never holds, and stores exactly what a worker that
already redacted it sent.

Nothing here names a provider, and nothing beyond the standard library is
imported.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import quote

__all__ = [
    "MAX_HELD_CHARACTERS",
    "MAX_LOG_CHUNK_BYTES",
    "REDACTED",
    "BuildLogChunker",
    "LogChunk",
    "redact",
    "secret_forms",
]

#: What stands where a secret was.
REDACTED = "[REDACTED]"

#: D-15: a log is stored in chunks of at most 64 KiB, measured as UTF-8.
MAX_LOG_CHUNK_BYTES = 64 * 1024

#: How much text a chunker holds while it waits for a line to end. A longer
#: line is released anyway, cut where no secret value could still be completing.
MAX_HELD_CHARACTERS = 256 * 1024

#: The lines of a multi-line secret shorter than this are not looked for on
#: their own: a short line of a key file would take every such line of the log
#: with it.
MIN_SECRET_LINE = 8

#: Room a chunk leaves for the markers redacting it again may add, when the cut
#: took away the context that kept a shape from matching.
_HEADROOM_BYTES = 64

#: Redacting a text until it stops changing takes two passes in practice.
_MAX_PASSES = 8

_LINE_END = re.compile(r"[\r\n]")

#: Tokens known by their prefix, not inside a longer word.
_PREFIXED_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"  # AWS access key ids
    r"|gh[pousr]_[A-Za-z0-9]{36,}"  # GitHub
    r"|github_pat_[A-Za-z0-9_]{22,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"  # GitLab
    r"|sk-[A-Za-z0-9_-]{20,}"  # OpenAI, Anthropic
    r"|[rs]k_(?:live|test)_[A-Za-z0-9]{16,}"  # Stripe
    r"|xox[abposr]-[A-Za-z0-9-]{10,}"  # Slack
    r"|AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"  # Google API keys
    r"|hf_[A-Za-z0-9]{30,}"  # Hugging Face
    r"|npm_[A-Za-z0-9]{36,}"  # npm
    r"|pypi-[A-Za-z0-9_-]{50,}"  # PyPI
    r"|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"  # JWT
    r")"
)

#: A private key block: its body is redacted, its first and last lines kept.
#: The body is the base64 and header lines after the first line, whether the
#: block is printed as lines, as one JSON-escaped string, or on one line; a
#: block whose last line never comes ends at the first line that is not body.
_PEM_SEPARATOR = r"(?:\r?\n|\\r\\n|\\n|[ \t])"
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]{0,40}PRIVATE KEY(?: BLOCK)?)-----"
    r"(?P<secret>(?:" + _PEM_SEPARATOR + r"(?:[A-Za-z0-9+/=]{1,128}"
    r"|[A-Za-z][A-Za-z0-9-]{0,40}:[^\r\n\\]{0,256})?){0,4096})"
    r"(?:-----END (?P=label)-----)?"
)

#: The password of a URL's user.
_URL_PASSWORD = re.compile(
    r"(?i)(?<![a-z0-9+.-])[a-z][a-z0-9+.-]{0,31}://[^\s/@:]{0,256}:(?P<secret>[^\s/@]+)@"
)

#: What an `Authorization` header carries, whichever scheme.
_AUTHORIZATION = re.compile(
    r"(?i)(?<![a-z0-9_-])(?:proxy-)?authorization[\"']?[ \t]*[:=][ \t]*[\"']?"
    r"(?:(?:bearer|basic|token|digest|negotiate|aws4-hmac-sha256)[ \t]+)?"
    r"(?P<secret>[^\s\"']+)"
)

#: A bearer token anywhere else.
_BEARER = re.compile(r"(?i)(?<![a-z0-9_-])bearer[ \t]+(?P<secret>[A-Za-z0-9._~+/-]{8,}=*)")

#: The value of a quoted or unquoted assignment, the quotes kept.
_VALUE = (
    r"(?:\"(?P<secret_double>[^\"\r\n]+)\"|'(?P<secret_single>[^'\r\n]+)'"
    r"|(?P<secret>[^\s\"',;&]+))"
)

#: A credential passed to a command as a flag.
_FLAG = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])--?(?:password|passwd|token|secret|api-key|apikey|access-key"
    r"|secret-key|client-secret|auth-token)(?:=|[ \t]+)" + _VALUE
)

#: `docker login -p`, `helm registry login -p` and their like.
_LOGIN_PASSWORD = re.compile(r"(?i)\blogin\b[^\r\n]{0,512}?[ \t]-p[ \t]+" + _VALUE)

#: A word that makes a key a credential's, as a whole segment of its name:
#: `GITHUB_TOKEN` and `api_key` are, `tokenizers` and `max_tokens` are not.
#: `pwd` is not one: `PWD` is the shell's working directory.
_CREDENTIAL_WORD = (
    r"(?:token|secret|password|passwd|passphrase|apikey|api[_-]key|accesskey|access[_-]key"
    r"|secretkey|secret[_-]key|privatekey|private[_-]key|accountkey|account[_-]key"
    r"|credentials?|auth|signature|sig)"
)

#: The value assigned to a credential-named key: `KEY=value`, `key: value`,
#: `"key": "value"`, `?key=value&`.
_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9]{1,64}[_.-]){0,8}"
    + _CREDENTIAL_WORD
    + r"(?:[_.-][A-Za-z0-9]{1,64}){0,8}(?![A-Za-z0-9])[\"']?[ \t]*(?::|=(?!=))[ \t]*"
    + _VALUE
)

_SHAPES: tuple[re.Pattern[str], ...] = (
    _PRIVATE_KEY,
    _PREFIXED_TOKEN,
    _URL_PASSWORD,
    _AUTHORIZATION,
    _BEARER,
    _FLAG,
    _LOGIN_PASSWORD,
    _ASSIGNMENT,
)


def secret_forms(secrets: Iterable[str]) -> list[str]:
    """Each secret value, and the forms a command most often prints one in, longest first.

    The value itself; its base64, standard and URL-safe, padded or not, of the
    value and of the value followed by the newline ``echo`` adds; the value
    URL-quoted; the value JSON-escaped; and each line of a multi-line value.
    """
    forms: set[str] = set()
    for value in secrets:
        if not isinstance(value, str) or not value:
            continue
        data = value.encode("utf-8", "replace")
        for raw in (data, data + b"\n"):
            standard = base64.b64encode(raw).decode("ascii")
            urlsafe = base64.urlsafe_b64encode(raw).decode("ascii")
            forms.update((standard, standard.rstrip("="), urlsafe, urlsafe.rstrip("=")))
        forms.update(
            (
                value,
                quote(value, safe=""),
                json.dumps(value)[1:-1],
                json.dumps(value, ensure_ascii=False)[1:-1],
            )
        )
        lines = value.splitlines()
        if len(lines) > 1:
            forms.update(line.strip() for line in lines if len(line.strip()) >= MIN_SECRET_LINE)
    forms.discard("")
    return sorted(forms, key=lambda form: (-len(form), form))


def _values_pattern(forms: Sequence[str]) -> re.Pattern[str] | None:
    if not forms:
        return None
    return re.compile("|".join(re.escape(form) for form in forms))


def _spans(
    text: str, patterns: Sequence[re.Pattern[str]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Where the patterns match: each whole match, and the part of it that is redacted."""
    matches: list[tuple[int, int]] = []
    secrets: list[tuple[int, int]] = []
    for pattern in patterns:
        groups = [name for name in pattern.groupindex if name.startswith("secret")]
        for match in pattern.finditer(text):
            matches.append(match.span())
            span = match.span()
            for name in groups:
                if match.start(name) != -1:
                    span = match.span(name)
                    break
            if span[0] < span[1]:
                secrets.append(span)
    return matches, secrets


def _replaced(text: str, spans: list[tuple[int, int]]) -> str:
    """The text with each run of overlapping or touching spans replaced by one marker."""
    if not spans:
        return text
    parts: list[str] = []
    position = 0
    run_start, run_end = -1, -1
    for start, end in sorted(spans):
        if start <= run_end:
            run_end = max(run_end, end)
            continue
        if run_end >= 0:
            parts += [text[position:run_start], REDACTED]
            position = run_end
        run_start, run_end = start, end
    parts += [text[position:run_start], REDACTED, text[run_end:]]
    return "".join(parts)


def _shapes_redacted(text: str) -> str:
    """The text with every credential shape redacted, as a fixed point."""
    for _ in range(_MAX_PASSES):
        redacted = _replaced(text, _spans(text, _SHAPES)[1])
        if redacted == text:
            break
        text = redacted
    return text


def _redacted(text: str, values: re.Pattern[str] | None) -> str:
    if values is not None:
        text = _replaced(text, _spans(text, (values,))[1])
    return _shapes_redacted(text)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """The text with the secret values, their common forms, and every credential shape redacted.

    Without secrets, what the service storing a chunk can still do: the
    credential shapes, as a fixed point, so an already redacted chunk comes
    back unchanged.
    """
    return _redacted(text, _values_pattern(secret_forms(secrets)))


def _last_line_end(text: str, limit: int) -> int:
    """Just after the last line break before `limit`, or 0 when there is none."""
    if limit <= 0:
        return 0
    return max(text.rfind("\n", 0, limit), text.rfind("\r", 0, limit)) + 1


def _split(text: str, budget: int) -> int:
    """Where a chunk of at most `budget` bytes ends.

    After the last line that ends in it, else after its last space, else at
    the last whole character.
    """
    data = text.encode("utf-8")
    if len(data) <= budget:
        return len(text)
    head = data[:budget].decode("utf-8", errors="ignore")
    for separators in ("\n\r", " \t"):
        end = max(head.rfind(separator) for separator in separators) + 1
        if end > 0:
            return end
    return max(1, len(head))


@dataclass(frozen=True)
class LogChunk:
    """One chunk of a build's log: the sequence it is stored under, and its redacted text."""

    sequence: int
    text: str

    @property
    def size(self) -> int:
        """In bytes, as UTF-8: what the 64 KiB bound is measured in."""
        return len(self.text.encode("utf-8"))


class BuildLogChunker:
    """A build's output in, redacted chunks of at most 64 KiB out, numbered in order.

    ``feed`` takes output as it arrives and answers the chunks that are full.
    ``flush`` answers what is ready however short: what a worker sends on a
    timer, so a follower sees the build progress. ``close`` releases everything
    still held and answers the last chunks.

    Held back until more arrives, so that no secret is released in two halves
    neither of which matches: the line not yet ended, and the last characters a
    multi-line secret value could still be completing. A line longer than
    ``max_held`` characters is released anyway, keeping back as many characters
    as the longest value has.
    """

    def __init__(
        self,
        secrets: Iterable[str] = (),
        *,
        first_sequence: int = 0,
        max_bytes: int = MAX_LOG_CHUNK_BYTES,
        max_held: int = MAX_HELD_CHARACTERS,
    ) -> None:
        if isinstance(first_sequence, bool) or not isinstance(first_sequence, int):
            raise TypeError("a log chunk's sequence is a whole number")
        if first_sequence < 0:
            raise ValueError("a log chunk's sequence is counted from 0")
        if max_bytes < 4 * _HEADROOM_BYTES:
            raise ValueError(f"a log chunk holds at least {4 * _HEADROOM_BYTES} bytes")
        forms = secret_forms(secrets)
        self._values = _values_pattern(forms)
        self._patterns = _SHAPES if self._values is None else (self._values, *_SHAPES)
        self._hold = max((len(form) for form in forms), default=1) - 1
        self._multiline_hold = (
            max((len(form) for form in forms if _LINE_END.search(form)), default=1) - 1
        )
        self._max_bytes = max_bytes
        self._max_held = max(max_held, self._hold + 1)
        self._next_sequence = first_sequence
        self._held = ""
        self._ready = ""
        self._closed = False

    @property
    def next_sequence(self) -> int:
        """The sequence the next chunk is numbered with."""
        return self._next_sequence

    def feed(self, text: str) -> list[LogChunk]:
        """Take output as it arrives; answer the chunks that are full."""
        if self._closed:
            raise ValueError("this build log is closed")
        if not text:
            return []
        # Output decoded with surrogate escapes cannot be stored as UTF-8.
        text = text.encode("utf-8", "replace").decode("utf-8")
        self._held += text
        if _LINE_END.search(text) or len(self._held) >= self._max_held:
            self._release(final=False)
        return self._chunks(full_only=True)

    def flush(self) -> list[LogChunk]:
        """Answer every chunk that can be released now, however short."""
        return self._chunks(full_only=False)

    def close(self) -> list[LogChunk]:
        """Release everything still held, and answer the last chunks."""
        if not self._closed:
            self._release(final=True)
            self._closed = True
        return self._chunks(full_only=False)

    def _release(self, *, final: bool) -> None:
        """Redact what can no longer be part of a secret still arriving, and make it ready."""
        held = self._held
        if not held:
            return
        cut = len(held)
        if not final:
            cut = _last_line_end(held, len(held) - self._multiline_hold)
            forced = cut == 0 and len(held) >= self._max_held
            if forced:
                cut = len(held) - self._hold
            # Never inside a match, nor before one that reaches the end of what
            # is held: its end may still be arriving, as a key's body does
            # after its first line.
            for start, end in sorted(self._spans_of(held), reverse=True):
                if start < cut and (cut < end or end == len(held)):
                    cut = start
            if cut <= 0:
                if len(held) < self._max_held:
                    return
                cut = max(1, len(held) - self._hold)
        self._ready += _redacted(held[:cut], self._values)
        self._held = held[cut:]

    def _spans_of(self, text: str) -> list[tuple[int, int]]:
        return _spans(text, self._patterns)[0]

    def _chunks(self, *, full_only: bool) -> list[LogChunk]:
        chunks: list[LogChunk] = []
        while self._ready:
            if full_only and len(self._ready.encode("utf-8")) < self._max_bytes:
                break
            budget = self._max_bytes - _HEADROOM_BYTES
            while True:
                end = _split(self._ready, budget)
                # Cutting can make a shape whole again at the chunk's start.
                text = _shapes_redacted(self._ready[:end])
                if len(text.encode("utf-8")) <= self._max_bytes or end <= 1:
                    break
                budget //= 2
            self._ready = self._ready[end:]
            chunks.append(LogChunk(self._next_sequence, text))
            self._next_sequence += 1
        return chunks
