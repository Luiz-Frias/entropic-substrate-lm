"""Boundary predicates for kernel.codec.strict_json: duplicate-key and
non-finite rejection in strict_loads, exact_keys' pure set arithmetic, and
the require_mapping / require_lowercase_sha256 type-and-shape predicates.
`fail` is the only error channel every arm exercises, and the propagation
arms pin that whatever `fail` raises reaches the caller unchanged, for
both a ValueError-subclass and a non-ValueError sentinel."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from typing import NoReturn

import pytest

from entropic_substrate.kernel.codec.strict_json import (
    exact_keys,
    require_lowercase_sha256,
    require_mapping,
    strict_loads,
)

pytestmark = pytest.mark.unit


class _MarkerValueError(ValueError):
    """A ValueError-subclass sentinel raised by a test-owned `fail`."""


class _MarkerRuntimeError(RuntimeError):
    """A non-ValueError sentinel raised by a test-owned `fail`."""


def _raiser(
    exception_type: type[Exception],
) -> tuple[list[str], list[BaseException], Callable[[str], NoReturn]]:
    """Return (messages, raised, fail) where fail records the message it
    received, records the exact instance it raises, and raises it."""

    messages: list[str] = []
    raised: list[BaseException] = []

    def fail(message: str) -> NoReturn:
        messages.append(message)
        instance = exception_type(message)
        raised.append(instance)
        raise instance

    return messages, raised, fail


# --------------------------------------------------------------------------
# strict_loads: duplicate keys
# --------------------------------------------------------------------------


def test_strict_loads_rejects_a_top_level_duplicate_key() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"a": 1, "a": 2}', fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""
    assert "duplicate" in messages[0]
    assert repr("a") in messages[0]


def test_strict_loads_rejects_a_nested_duplicate_key() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"outer": {"b": 1, "b": 2}}', fail=fail)
    assert len(messages) == 1
    assert "duplicate" in messages[0]
    assert repr("b") in messages[0]


def test_strict_loads_accepts_distinct_keys_in_pair_order() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    result = strict_loads('{"a": 1, "b": 2}', fail=fail)
    assert result == {"a": 1, "b": 2}
    assert type(result) is dict
    assert list(result) == ["a", "b"]


# --------------------------------------------------------------------------
# strict_loads: non-finite constants
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "literal,constant",
    [("NaN", "NaN"), ("Infinity", "Infinity"), ("-Infinity", "-Infinity")],
)
def test_strict_loads_rejects_each_nonfinite_constant(literal: str, constant: str) -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads(f'{{"value": {literal}}}', fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""
    assert "non-finite" in messages[0]
    assert repr(constant) in messages[0]


def test_strict_loads_accepts_ordinary_finite_numbers() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    result = strict_loads('{"value": 1.5}', fail=fail)
    assert result == {"value": 1.5}


def test_strict_loads_accepts_ints_and_floats_with_their_own_types() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    result = strict_loads('{"a": 1, "b": 1.5}', fail=fail)
    assert type(result["a"]) is int
    assert type(result["b"]) is float


# --------------------------------------------------------------------------
# strict_loads: overflow-to-infinity floats (parse_float hook)
# --------------------------------------------------------------------------


def test_strict_loads_rejects_a_top_level_overflow_to_infinity_float() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads("[23456789012E666]", fail=fail)  # pragma: allowlist secret
    assert len(messages) == 1
    assert messages[0] != ""
    assert "23456789012E666" in messages[0]  # pragma: allowlist secret


def test_strict_loads_rejects_a_top_level_overflow_to_negative_infinity_float() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads("[-1E999]", fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""
    assert "-1E999" in messages[0]


def test_strict_loads_rejects_a_nested_overflow_to_infinity_float_in_an_object_value() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"value": 23456789012E666}', fail=fail)  # pragma: allowlist secret
    assert len(messages) == 1
    assert messages[0] != ""
    assert "23456789012E666" in messages[0]  # pragma: allowlist secret


def test_strict_loads_overflow_float_propagates_value_error_subclass_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        strict_loads("[23456789012E666]", fail=fail)  # pragma: allowlist secret
    assert excinfo.value is raised[0]
    assert len(messages) == 1


def test_strict_loads_overflow_float_propagates_non_value_error_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerRuntimeError)
    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads("[23456789012E666]", fail=fail)  # pragma: allowlist secret
    assert excinfo.value is raised[0]
    assert len(messages) == 1


# --------------------------------------------------------------------------
# strict_loads: the parse_float seam (FOR-1632)
# --------------------------------------------------------------------------

# 17 significant digits (19 fractional digits): the pinned billed amount.
_BILLED = "0.0017170833889394999"
# 23 significant digits: beyond what float64 round-trips, so a
# Decimal(repr(float(token))) reconstruction cannot equal it.
_WIDE = "0.12345678901234567890123"


class _InfFloat(float):
    """A float subclass carrying infinity, to prove subclasses cannot bypass the check."""


class _InfDecimal(Decimal):
    """A Decimal subclass carrying non-finite values, so subclasses cannot bypass the check."""


class _Sentinel:
    """Neither float nor Decimal: the door returns it unchanged."""


def test_parse_float_decimal_decodes_the_billed_amount_exactly() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    result = strict_loads(f'{{"a": {_BILLED}}}', fail=fail, parse_float=Decimal)
    assert isinstance(result, dict)
    value = result["a"]
    assert type(value) is Decimal
    assert value == Decimal(_BILLED)
    assert str(value) == _BILLED


def test_parse_float_decimal_keeps_digits_beyond_float64_precision() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    result = strict_loads(f"[{_WIDE}]", fail=fail, parse_float=Decimal)
    assert isinstance(result, list)
    value = result[0]
    assert type(value) is Decimal
    assert value == Decimal(_WIDE)
    assert str(value) == _WIDE
    # Positive control: float reconstruction would lose digits, so this pin has teeth.
    assert Decimal(repr(float(_WIDE))) != Decimal(_WIDE)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_parse_float_never_sees_the_non_finite_constants(constant: str) -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    seen: list[str] = []

    def record(token: str) -> object:
        seen.append(token)
        return Decimal(token)

    with pytest.raises(_MarkerValueError):
        strict_loads(f'{{"a": {constant}}}', fail=fail, parse_float=record)
    assert seen == []
    assert messages == [f"non-finite constant: {constant!r}"]


def test_parse_float_none_still_refuses_the_overflow_token() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads("[1e999]", fail=fail, parse_float=None)
    assert messages == ["non-finite float literal: '1e999'"]


def test_parse_float_decimal_admits_the_overflow_token_as_a_finite_decimal() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    result = strict_loads("[1e999]", fail=fail, parse_float=Decimal)
    assert isinstance(result, list)
    value = result[0]
    assert type(value) is Decimal
    assert value == Decimal("1E+999")
    assert value.is_finite()
    assert messages == []


def test_parse_float_is_called_once_per_float_token_with_the_raw_token() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    seen: list[str] = []

    def record(token: str) -> object:
        seen.append(token)
        return Decimal(token)

    strict_loads('{"a": [1.50, 2, {"b": -3.25e2}], "c": 4.0}', fail=fail, parse_float=record)
    assert seen == ["1.50", "-3.25e2", "4.0"]


@pytest.mark.parametrize(
    "refused",
    [
        pytest.param(float("inf"), id="float-inf"),
        pytest.param(float("-inf"), id="float-negative-inf"),
        pytest.param(float("nan"), id="float-nan"),
        pytest.param(_InfFloat("inf"), id="float-subclass-inf"),
        pytest.param(Decimal("Infinity"), id="decimal-inf"),
        pytest.param(Decimal("-Infinity"), id="decimal-negative-inf"),
        pytest.param(Decimal("NaN"), id="decimal-nan"),
        pytest.param(_InfDecimal("Infinity"), id="decimal-subclass-inf"),
        pytest.param(_InfDecimal("-Infinity"), id="decimal-subclass-negative-inf"),
        pytest.param(_InfDecimal("NaN"), id="decimal-subclass-nan"),
    ],
)
def test_parse_float_result_that_is_non_finite_is_refused_through_fail(
    refused: float | Decimal,
) -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        strict_loads('{"a": 1.5}', fail=fail, parse_float=lambda token: refused)
    assert excinfo.value is raised[0]
    assert messages == ["non-finite float literal: '1.5'"]


def test_parse_float_decimal_result_is_judged_by_is_finite_not_by_float_conversion() -> None:
    # float(Decimal("1E+999")) is inf, yet the Decimal is finite and must pass.
    messages, _, fail = _raiser(_MarkerValueError)
    result = strict_loads("[1.5]", fail=fail, parse_float=lambda token: Decimal("1E+999"))
    assert result == [Decimal("1E+999")]
    assert messages == []


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param(1.25, id="finite-float"),
        pytest.param(Decimal("1.25"), id="finite-decimal"),
        pytest.param(_InfFloat(2.5), id="finite-float-subclass"),
        pytest.param("1.25", id="str"),
        pytest.param(7, id="int"),
        pytest.param(_Sentinel(), id="other-object"),
    ],
)
def test_parse_float_result_that_is_finite_or_not_numeric_is_returned_unchanged(
    returned: object,
) -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    result = strict_loads("[1.25]", fail=fail, parse_float=lambda token: returned)
    assert isinstance(result, list)
    assert result[0] is returned
    assert messages == []


def test_parse_float_exception_propagates_unchanged_without_calling_fail() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    boom = _MarkerRuntimeError("boom")

    def explode(token: str) -> object:
        raise boom

    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads("[1.5]", fail=fail, parse_float=explode)
    assert excinfo.value is boom
    assert messages == []


def test_parse_float_keeps_duplicate_key_refusal_through_fail() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"a": 1.5, "a": 2.5}', fail=fail, parse_float=Decimal)
    assert messages == ["duplicate key: 'a'"]


# --------------------------------------------------------------------------
# strict_loads: fail propagates unchanged, both exception families,
# duplicate-key and non-finite paths
# --------------------------------------------------------------------------


def test_strict_loads_duplicate_key_propagates_value_error_subclass_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        strict_loads('{"a": 1, "a": 2}', fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_strict_loads_duplicate_key_propagates_non_value_error_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerRuntimeError)
    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads('{"a": 1, "a": 2}', fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_strict_loads_nonfinite_propagates_value_error_subclass_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        strict_loads('{"value": NaN}', fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_strict_loads_nonfinite_propagates_non_value_error_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerRuntimeError)
    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads('{"value": NaN}', fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


# --------------------------------------------------------------------------
# exact_keys: pure, never raises, five arms
# --------------------------------------------------------------------------


def test_exact_keys_empty_actual_and_empty_required() -> None:
    missing, unknown = exact_keys([], frozenset(), frozenset())
    assert missing == frozenset()
    assert unknown == frozenset()
    assert type(missing) is frozenset
    assert type(unknown) is frozenset


def test_exact_keys_exact_match() -> None:
    missing, unknown = exact_keys(["a", "b"], frozenset({"a", "b"}))
    assert missing == frozenset()
    assert unknown == frozenset()


def test_exact_keys_missing_only() -> None:
    missing, unknown = exact_keys(["a"], frozenset({"a", "b"}))
    assert missing == frozenset({"b"})
    assert unknown == frozenset()


def test_exact_keys_unknown_only() -> None:
    missing, unknown = exact_keys(["a", "c"], frozenset({"a"}))
    assert missing == frozenset()
    assert unknown == frozenset({"c"})


def test_exact_keys_both_missing_and_unknown_with_optional_excluded() -> None:
    missing, unknown = exact_keys(["a", "c", "opt"], frozenset({"a", "b"}), frozenset({"opt"}))
    assert missing == frozenset({"b"})
    assert unknown == frozenset({"c"})


def test_exact_keys_consumes_a_one_shot_iterator_only_once() -> None:
    missing, unknown = exact_keys(iter(["a", "b"]), frozenset({"a", "b"}))
    assert missing == frozenset()
    assert unknown == frozenset()


# --------------------------------------------------------------------------
# require_mapping
# --------------------------------------------------------------------------


def test_require_mapping_accepts_a_dict_with_string_keys_and_returns_it_unchanged() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    value: dict[str, object] = {"a": 1}
    result = require_mapping(value, fail=fail)
    assert result is value


def test_require_mapping_rejects_a_non_dict() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_mapping(["not", "a", "mapping"], fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_mapping_rejects_a_non_str_key() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_mapping({1: "a"}, fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_mapping_propagates_a_value_error_subclass_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        require_mapping("not a mapping", fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_mapping_propagates_a_non_value_error_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerRuntimeError)
    with pytest.raises(_MarkerRuntimeError) as excinfo:
        require_mapping("not a mapping", fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


# --------------------------------------------------------------------------
# require_lowercase_sha256
# --------------------------------------------------------------------------


def test_require_lowercase_sha256_accepts_a_valid_digest_and_returns_it_unchanged() -> None:
    _, _, fail = _raiser(_MarkerValueError)
    digest = "a" * 64
    result = require_lowercase_sha256(digest, fail=fail)
    assert result is digest


def test_require_lowercase_sha256_rejects_uppercase_hex() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("A" * 64, fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_lowercase_sha256_rejects_a_63_character_digest() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("a" * 63, fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_rejects_a_65_character_digest() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("a" * 65, fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_rejects_a_non_str_digest() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256(("a" * 64).encode(), fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_propagates_a_value_error_subclass_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        require_lowercase_sha256("A" * 64, fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_lowercase_sha256_propagates_a_non_value_error_unchanged() -> None:
    messages, raised, fail = _raiser(_MarkerRuntimeError)
    with pytest.raises(_MarkerRuntimeError) as excinfo:
        require_lowercase_sha256("A" * 64, fail=fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


# --------------------------------------------------------------------------
# strict_loads: malformed JSON is a decoder-level error, not a `fail` call
# --------------------------------------------------------------------------


def test_strict_loads_malformed_json_raises_decode_error_and_never_calls_fail() -> None:
    messages, _, fail = _raiser(_MarkerValueError)
    with pytest.raises(json.JSONDecodeError):
        strict_loads("{", fail=fail)
    assert messages == []
