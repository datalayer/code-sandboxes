# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""JSON canonicalization (RFC 8785), for digests that mean the content.

An Environment's spec digest is a cache key: two specs that say the same
thing must hash the same, whatever order their keys were written in and
however they were spaced. RFC 8785 fixes both — keys sorted by UTF-16 code
units, no whitespace, strings escaped the way ECMAScript escapes them, and
numbers written the way ECMAScript's ``Number.prototype.toString`` writes
them — so a digest computed here matches one computed by any conforming
implementation, in Python or in the browser.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

__all__ = ["canonical_digest", "canonical_json"]

#: The largest integer an IEEE-754 double holds exactly; RFC 8785 numbers are doubles.
_MAX_EXACT_INTEGER = 2**53

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def canonical_json(value: Any) -> bytes:
    """The RFC 8785 serialization of a JSON value, as UTF-8 bytes."""
    return _serialize(value).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """``sha256:<hex>`` of the canonical serialization."""
    return "sha256:" + hashlib.sha256(canonical_json(value)).hexdigest()


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if abs(value) > _MAX_EXACT_INTEGER:
            raise ValueError(f"{value} is beyond what a JSON number holds exactly")
        return str(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise TypeError(f"a JSON object's keys are strings, not {type(key).__name__}")
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(f"{_string(key)}:{_serialize(item)}" for key, item in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(item) for item in value) + "]"
    raise TypeError(f"{type(value).__name__} has no JSON form")


def _string(text: str) -> str:
    out = ['"']
    for char in text:
        escaped = _ESCAPES.get(char)
        if escaped is not None:
            out.append(escaped)
        elif ord(char) < 0x20 or 0xD800 <= ord(char) <= 0xDFFF:
            # Control characters, and surrogates that pair with nothing.
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    out.append('"')
    return "".join(out)


def _number(value: float) -> str:
    """A double as ECMAScript's ``Number.prototype.toString`` writes it.

    Python's ``repr`` gives the shortest digits that round-trip, as
    ECMAScript does; what differs is where the decimal point goes and when an
    exponent is used, which is the placement below.
    """
    if not math.isfinite(value):
        raise ValueError("NaN and Infinity have no JSON form")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    mantissa, _, exponent_text = repr(abs(value)).partition("e")
    exponent = int(exponent_text) if exponent_text else 0
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    significant = digits.rstrip("0")
    trailing = len(digits) - len(significant)
    k = len(significant)
    # The value is 0.<significant> x 10^n.
    n = k + exponent - len(fraction) + trailing
    if k <= n <= 21:
        body = significant + "0" * (n - k)
    elif 0 < n <= 21:
        body = significant[:n] + "." + significant[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + significant
    else:
        power = n - 1
        suffix = f"e{'+' if power >= 0 else '-'}{abs(power)}"
        body = (significant if k == 1 else significant[0] + "." + significant[1:]) + suffix
    return sign + body
