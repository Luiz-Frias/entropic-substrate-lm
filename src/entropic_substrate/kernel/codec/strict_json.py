"""Strict canonical-JSON boundary codec: duplicate-key and non-finite
rejection, exact-key set comparison, and the mapping/digest boundary
predicates promoted from ladder/capture/records/d1_telemetry_payloads.py.

`fail` is the error channel for the predicates this module enforces on
already-decoded JSON: duplicate keys, the three non-finite constants
(NaN, Infinity, -Infinity), and any numeric literal that decodes to a
non-finite float or Decimal. `fail` is typed `Callable[[str], NoReturn]`
and must raise; this module never wraps that call, so whatever `fail`
raises propagates unchanged to the caller. Malformed input that the
stdlib decoder itself rejects (invalid JSON syntax, invalid UTF-8 bytes)
raises `json.JSONDecodeError` / `UnicodeDecodeError` directly, without
calling `fail`; each adopter wraps those at its own boundary.

`parse_float` is the one decode seam. Left as None, every float token goes
through `float(token)` and a non-finite value is refused through `fail`.
Given, each float token goes to `parse_float(token)` exactly once and its
result is judged by type: a `float` (subclasses included) by
`math.isfinite`, a `Decimal` (subclasses included) by `Decimal.is_finite`,
and a non-finite result of either kind is refused through `fail` with the
same `non-finite float literal` message; any other result is returned
unchanged. A Decimal decode therefore admits `1e999` as a finite Decimal,
where the default path refuses it. The three constants never reach
`parse_float`: `parse_constant` refuses them first. An exception raised by
`parse_float` propagates unchanged, exactly as `fail`'s does; each adopter
wraps it at its own boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from decimal import Decimal
from json import loads
from typing import NoReturn

__all__ = [
    "exact_keys",
    "require_lowercase_sha256",
    "require_mapping",
    "strict_loads",
]


def strict_loads(
    raw: str | bytes,
    *,
    fail: Callable[[str], NoReturn],
    parse_float: Callable[[str], object] | None = None,
) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        object_ = {}
        for key, value in pairs:
            if key in object_:
                fail(f"duplicate key: {key!r}")
            object_[key] = value
        return object_

    def constant_hook(constant: str) -> object:
        fail(f"non-finite constant: {constant!r}")

    def float_hook(token: str) -> object:
        value: object = float(token) if parse_float is None else parse_float(token)
        if isinstance(value, float) and not math.isfinite(value):
            fail(f"non-finite float literal: {token!r}")
        if isinstance(value, Decimal) and not value.is_finite():
            fail(f"non-finite float literal: {token!r}")
        return value

    return loads(
        raw,
        object_pairs_hook=pairs_hook,
        parse_constant=constant_hook,
        parse_float=float_hook,
    )


def exact_keys(
    actual: Iterable[str],
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> tuple[frozenset[str], frozenset[str]]:
    actual_keys = frozenset(actual)
    return required - actual_keys, actual_keys - required - optional


def require_mapping(value: object, *, fail: Callable[[str], NoReturn]) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        fail("value must be a JSON object with string keys")
    return value


def require_lowercase_sha256(value: object, *, fail: Callable[[str], NoReturn]) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        fail("value must be 64 lowercase hexadecimal characters")
    return value
