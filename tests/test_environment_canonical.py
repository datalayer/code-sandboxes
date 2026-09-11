# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""RFC 8785, checked: a spec digest must match one computed anywhere else.

The number vectors are the RFC's Appendix B table, as ECMAScript's
`Number.prototype.toString` writes them; each was confirmed against Node.
Non-ASCII characters are written as escapes, so what is tested is visible.
"""

from __future__ import annotations

import math

import pytest

from code_sandboxes.environments.canonical import canonical_digest, canonical_json

NUMBERS = [
    (0.0, "0"),
    (-0.0, "0"),
    (5e-324, "5e-324"),
    (-5e-324, "-5e-324"),
    (1.7976931348623157e308, "1.7976931348623157e+308"),
    (-1.7976931348623157e308, "-1.7976931348623157e+308"),
    (9007199254740992.0, "9007199254740992"),
    (-9007199254740992.0, "-9007199254740992"),
    (295147905179352825856.0, "295147905179352830000"),
    (9.999999999999997e22, "9.999999999999997e+22"),
    (1e23, "1e+23"),
    (1.0000000000000001e23, "1.0000000000000001e+23"),
    (999999999999999700000.0, "999999999999999700000"),
    (999999999999999900000.0, "999999999999999900000"),
    (1e21, "1e+21"),
    (9.999999999999997e-7, "9.999999999999997e-7"),
    (0.000001, "0.000001"),
    (333333333.3333332, "333333333.3333332"),
    (333333333.33333325, "333333333.33333325"),
    (333333333.3333333, "333333333.3333333"),
    (333333333.3333334, "333333333.3333334"),
    (333333333.33333343, "333333333.33333343"),
    (-0.0000033333333333333333, "-0.0000033333333333333333"),
    (1424953923781206.2, "1424953923781206.2"),
    (0.5, "0.5"),
    (100.0, "100"),
    (0.002, "0.002"),
    (1e-7, "1e-7"),
    (123.456, "123.456"),
]


@pytest.mark.parametrize(("value", "expected"), NUMBERS)
def test_a_number_is_written_the_way_ecmascript_writes_it(value: float, expected: str) -> None:
    assert canonical_json(value) == expected.encode()


def test_the_rfc_example_serializes_exactly() -> None:
    """RFC 8785 section 3.2.2's input, and its canonical form byte for byte."""
    document = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": '€$\x0f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }
    expected = (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    )
    assert canonical_json(document) == expected.encode("utf-8")


def test_keys_are_sorted_by_utf16_code_units_not_by_code_points() -> None:
    """An astral character is a surrogate pair, which sorts before U+FB33."""
    document = {
        "€": 1,
        "\r": 2,
        "דּ": 3,
        "1": 4,
        "\U0001f600": 5,
        "": 6,
        "ö": 7,
    }
    serialized = canonical_json(document).decode("utf-8")
    order = ["\\r", "1", "", "ö", "€", "\U0001f600", "דּ"]
    positions = [serialized.index(f'"{key}"') for key in order]
    assert positions == sorted(positions)


def test_strings_escape_what_ecmascript_escapes_and_nothing_else() -> None:
    assert canonical_json("\b\f\n\r\t\x01") == b'"\\b\\f\\n\\r\\t\\u0001"'
    assert canonical_json("é€\U0001f600") == '"é€\U0001f600"'.encode()
    assert canonical_json("\ud800") == b'"\\ud800"'


def test_there_is_no_whitespace_anywhere() -> None:
    assert canonical_json({"b": [1, {"c": None}], "a": "x y"}) == b'{"a":"x y","b":[1,{"c":null}]}'


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_numbers_json_cannot_hold_are_refused(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json(value)


def test_an_integer_a_double_cannot_hold_exactly_is_refused() -> None:
    assert canonical_json(2**53) == b"9007199254740992"
    with pytest.raises(ValueError):
        canonical_json(2**53 + 1)


def test_keys_must_be_strings() -> None:
    with pytest.raises(TypeError):
        canonical_json({1: "one"})


def test_the_digest_does_not_depend_on_key_order() -> None:
    first = canonical_digest({"a": 1, "b": {"c": [1, 2], "d": True}})
    second = canonical_digest({"b": {"d": True, "c": [1, 2]}, "a": 1})
    assert first == second
    assert first.startswith("sha256:") and len(first) == len("sha256:") + 64
    assert canonical_digest({"a": 2, "b": {"c": [1, 2], "d": True}}) != first
