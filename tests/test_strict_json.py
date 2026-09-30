"""Boundary predicates for kernel.codec.strict_json: duplicate-key and
non-finite rejection in strict_loads, exact_keys' pure set arithmetic, and
the require_mapping / require_lowercase_sha256 type-and-shape predicates.
`fail` is the only error channel every arm exercises."""

from __future__ import annotations

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


def _raiser(exception_type: type[Exception]) -> tuple[list[str], object]:
    """Return (messages, fail) where fail records the message it received
    and raises a fresh instance of exception_type."""

    messages: list[str] = []

    def fail(message: str) -> None:
        messages.append(message)
        raise exception_type(message)

    return messages, fail


# --------------------------------------------------------------------------
# strict_loads: duplicate keys
# --------------------------------------------------------------------------


def test_strict_loads_rejects_a_top_level_duplicate_key() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"a": 1, "a": 2}', fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""
    assert "duplicate" in messages[0]
    assert repr("a") in messages[0]


def test_strict_loads_rejects_a_nested_duplicate_key() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads('{"outer": {"b": 1, "b": 2}}', fail=fail)
    assert len(messages) == 1
    assert "duplicate" in messages[0]
    assert repr("b") in messages[0]


def test_strict_loads_accepts_distinct_keys_in_pair_order() -> None:
    _, fail = _raiser(_MarkerValueError)
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
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        strict_loads(f'{{"value": {literal}}}', fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""
    assert "non-finite" in messages[0]
    assert repr(constant) in messages[0]


def test_strict_loads_accepts_ordinary_finite_numbers() -> None:
    _, fail = _raiser(_MarkerValueError)
    result = strict_loads('{"value": 1.5}', fail=fail)
    assert result == {"value": 1.5}


# --------------------------------------------------------------------------
# strict_loads: fail propagates unchanged (both exception families)
# --------------------------------------------------------------------------


def test_strict_loads_duplicate_key_propagates_value_error_subclass_unchanged() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError) as excinfo:
        strict_loads('{"a": 1, "a": 2}', fail=fail)
    assert excinfo.value.args[0] == messages[0]


def test_strict_loads_duplicate_key_propagates_non_value_error_unchanged() -> None:
    messages: list[str] = []
    raised: list[BaseException] = []

    def recording_fail(message: str) -> None:
        instance = _MarkerRuntimeError(message)
        raised.append(instance)
        messages.append(message)
        raise instance

    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads('{"a": 1, "a": 2}', fail=recording_fail)
    assert excinfo.value is raised[0]
    assert len(messages) == 1
    assert messages[0] != ""


def test_strict_loads_nonfinite_propagates_non_value_error_unchanged() -> None:
    raised: list[BaseException] = []
    messages: list[str] = []

    def recording_fail(message: str) -> None:
        instance = _MarkerRuntimeError(message)
        raised.append(instance)
        messages.append(message)
        raise instance

    with pytest.raises(_MarkerRuntimeError) as excinfo:
        strict_loads('{"value": NaN}', fail=recording_fail)
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
    _, fail = _raiser(_MarkerValueError)
    value: dict[str, object] = {"a": 1}
    result = require_mapping(value, fail=fail)
    assert result is value


def test_require_mapping_rejects_a_non_dict() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_mapping(["not", "a", "mapping"], fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_mapping_rejects_a_non_str_key() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_mapping({1: "a"}, fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_mapping_propagates_a_non_value_error_unchanged() -> None:
    raised: list[BaseException] = []

    def recording_fail(message: str) -> None:
        instance = _MarkerRuntimeError(message)
        raised.append(instance)
        raise instance

    with pytest.raises(_MarkerRuntimeError) as excinfo:
        require_mapping("not a mapping", fail=recording_fail)
    assert excinfo.value is raised[0]


# --------------------------------------------------------------------------
# require_lowercase_sha256
# --------------------------------------------------------------------------


def test_require_lowercase_sha256_accepts_a_valid_digest_and_returns_it_unchanged() -> None:
    _, fail = _raiser(_MarkerValueError)
    digest = "a" * 64
    result = require_lowercase_sha256(digest, fail=fail)
    assert result is digest


def test_require_lowercase_sha256_rejects_uppercase_hex() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("A" * 64, fail=fail)
    assert len(messages) == 1
    assert messages[0] != ""


def test_require_lowercase_sha256_rejects_a_63_character_digest() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("a" * 63, fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_rejects_a_65_character_digest() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256("a" * 65, fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_rejects_a_non_str_digest() -> None:
    messages, fail = _raiser(_MarkerValueError)
    with pytest.raises(_MarkerValueError):
        require_lowercase_sha256(("a" * 64).encode(), fail=fail)
    assert len(messages) == 1


def test_require_lowercase_sha256_propagates_a_non_value_error_unchanged() -> None:
    raised: list[BaseException] = []

    def recording_fail(message: str) -> None:
        instance = _MarkerRuntimeError(message)
        raised.append(instance)
        raise instance

    with pytest.raises(_MarkerRuntimeError) as excinfo:
        require_lowercase_sha256("A" * 64, fail=recording_fail)
    assert excinfo.value is raised[0]
