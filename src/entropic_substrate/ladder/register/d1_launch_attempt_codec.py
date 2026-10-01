"""Strict canonical JSON codec for the D1 launch-attempt ledger."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import NoReturn

from entropic_substrate.kernel.codec.strict_json import exact_keys, require_mapping, strict_loads
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1_LAUNCH_ATTEMPT_SCHEMA,
    D1_LAUNCH_LEDGER_SCHEMA,
    D1LaunchAttemptEvent,
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)

_LEDGER_KEYS = frozenset({"events", "schema_version"})
_EVENT_KEYS = frozenset({"event_seq", "identity", "observed_at", "reason", "status"})
_IDENTITY_KEYS = frozenset(
    {
        "arm",
        "family_digest",
        "logical_run_id",
        "optional_stopping_digest",
        "registered_design_digest",
        "run_class",
        "schema_version",
        "seed",
    }
)


class D1LaunchAttemptCodecError(ValueError):
    """Persisted attempt-ledger bytes are malformed, permissive, or forked."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1LaunchAttemptCodecError(message)
    raise D1LaunchAttemptCodecError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    return require_mapping(
        value, fail=lambda _reason: _fail(f"{label} must be a JSON object with string keys")
    )


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    missing, unknown = exact_keys(value, keys)
    if missing or unknown:
        _fail(f"{label} keys must be exact")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        _fail(f"{label} must be a nonempty string")
    return value


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a nonnegative integer")
    return value


def _timestamp(value: object) -> datetime:
    text = _text(value, label="observed_at")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        _fail("observed_at must be an ISO-8601 timestamp", error)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.isoformat() != text:
        _fail("observed_at must use canonical datetime.isoformat() UTC spelling")
    return parsed


def _identity(value: object) -> D1LaunchAttemptIdentity:
    mapping = _mapping(value, label="attempt identity")
    _exact(mapping, _IDENTITY_KEYS, label="attempt identity")
    if mapping["schema_version"] != D1_LAUNCH_ATTEMPT_SCHEMA:
        _fail("attempt identity schema_version is unsupported")
    registered = mapping["registered_design_digest"]
    if registered is not None and type(registered) is not str:
        _fail("registered_design_digest must be a string or null")
    try:
        identity = D1LaunchAttemptIdentity(
            run_class=RunClass(_text(mapping["run_class"], label="run_class")),
            family_digest=_text(mapping["family_digest"], label="family_digest"),
            arm=D1Arm(_text(mapping["arm"], label="arm")),
            seed=_integer(mapping["seed"], label="seed"),
            logical_run_id=_text(mapping["logical_run_id"], label="logical_run_id"),
            registered_design_digest=registered,
        )
    except (TypeError, ValueError) as error:
        _fail(f"attempt identity is invalid: {error}", error)
    if mapping["optional_stopping_digest"] != identity.optional_stopping_digest:
        _fail("attempt identity optional_stopping_digest is forged")
    return identity


def _event(value: object) -> D1LaunchAttemptEvent:
    mapping = _mapping(value, label="attempt event")
    _exact(mapping, _EVENT_KEYS, label="attempt event")
    reason = mapping["reason"]
    if reason is not None and type(reason) is not str:
        _fail("attempt event reason must be a string or null")
    try:
        return D1LaunchAttemptEvent(
            event_seq=_integer(mapping["event_seq"], label="event_seq"),
            identity=_identity(mapping["identity"]),
            status=D1LaunchAttemptStatus(_text(mapping["status"], label="status")),
            observed_at=_timestamp(mapping["observed_at"]),
            reason=reason,
        )
    except D1LaunchAttemptCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"attempt event is invalid: {error}", error)


def launch_attempt_ledger_to_mapping(
    ledger: D1LaunchAttemptLedger,
) -> dict[str, object]:
    if type(ledger) is not D1LaunchAttemptLedger:
        raise TypeError("launch-attempt codec requires D1LaunchAttemptLedger")
    return copy.deepcopy(ledger.manifest)


def launch_attempt_ledger_from_mapping(value: object) -> D1LaunchAttemptLedger:
    mapping = _mapping(value, label="launch-attempt ledger object")
    _exact(mapping, _LEDGER_KEYS, label="launch-attempt ledger")
    if mapping["schema_version"] != D1_LAUNCH_LEDGER_SCHEMA:
        _fail("launch-attempt ledger schema_version is unsupported")
    raw_events = mapping["events"]
    if type(raw_events) is not list:
        _fail("launch-attempt ledger events must be a JSON array")
    try:
        return D1LaunchAttemptLedger(tuple(_event(item) for item in raw_events))
    except D1LaunchAttemptCodecError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"launch-attempt ledger history is invalid: {error}", error)


def serialize_launch_attempt_ledger(ledger: D1LaunchAttemptLedger) -> str:
    return json.dumps(
        launch_attempt_ledger_to_mapping(ledger),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_launch_attempt_ledger(payload: str) -> D1LaunchAttemptLedger:
    if type(payload) is not str or not payload:
        _fail("launch-attempt ledger payload must be nonempty text")
    try:
        value: object = strict_loads(payload, fail=_fail)
    except D1LaunchAttemptCodecError:
        raise
    except json.JSONDecodeError as error:
        _fail("launch-attempt ledger payload is invalid JSON", error)
    ledger = launch_attempt_ledger_from_mapping(value)
    if serialize_launch_attempt_ledger(ledger) != payload:
        _fail("launch-attempt ledger JSON is not canonical")
    return ledger
