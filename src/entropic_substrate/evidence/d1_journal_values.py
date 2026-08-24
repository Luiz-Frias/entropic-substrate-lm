"""Strict values and canonical JSON for the D1 authoritative run journal."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, cast

from entropic_substrate.experiments.d1_primitives import RunClass

JOURNAL_EVENT_SCHEMA: Final = "d1-authoritative-event-v1"
JOURNAL_STATE_SCHEMA: Final = "d1-authoritative-journal-v1"

_EVENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "event_kind",
        "event_seq",
        "logical_run_id",
        "execution_digest",
        "aggregation_digest",
        "construction_digest",
        "family_digest",
        "run_class",
        "successful_step",
        "payload",
    }
)
_STATE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "logical_run_id",
        "execution_digest",
        "last_event_sequence",
        "prefix_sha256",
        "successful_step",
    }
)

type FrozenJsonScalar = str | int | float | bool | None
type FrozenJson = FrozenJsonScalar | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]


class D1JournalError(ValueError):
    """A malformed or inconsistent D1 journal value with a stable error code."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 journal {code} at {path}: {detail}")


def _error(code: str, path: str, detail: str) -> D1JournalError:
    return D1JournalError(code=code, path=path, detail=detail)


def _mapping_exact(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, object]:
    if type(value) is not dict:
        raise _error("invalid_type", path, "must be a JSON object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise _error("invalid_field_name", path, "field names must be strings")
    actual = cast(set[str], set(mapping))
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing:
        raise _error("missing_field", path, f"missing required field(s): {', '.join(missing)}")
    if unknown:
        raise _error("unknown_field", path, f"unknown field(s): {', '.join(unknown)}")
    return cast(Mapping[str, object], mapping)


def _nonempty_string(value: object, *, path: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise _error("invalid_string", path, "must be a nonempty, trimmed string")
    return value


def _nonnegative_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise _error("invalid_integer", path, "must be a nonnegative integer, not a boolean")
    return value


def _sha256(value: object, *, path: str) -> str:
    digest = _nonempty_string(value, path=path)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _error("invalid_sha256", path, "must be 64 lowercase hexadecimal characters")
    return digest


def _freeze_json(value: object, *, path: str) -> FrozenJson:
    if value is None or type(value) in (str, bool, int):
        return cast(FrozenJsonScalar, value)
    if type(value) is float:
        if not math.isfinite(value):
            raise _error("nonfinite_json", path, "floating JSON values must be finite")
        return value
    if type(value) in (list, tuple):
        items = cast(Sequence[object], value)
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(items)
        )
    if type(value) is dict or isinstance(value, MappingProxyType):
        mapping = cast(Mapping[object, object], value)
        frozen: dict[str, FrozenJson] = {}
        for key, item in mapping.items():
            if type(key) is not str:
                raise _error("invalid_field_name", path, "JSON object keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    raise _error("invalid_json", path, "must contain only JSON-compatible values")


def _thaw_json(value: FrozenJson) -> object:
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    return value


class JournalEventKind(StrEnum):
    """The complete event vocabulary for one D1 trainer execution history."""

    RUN_START = "run_start"
    BASELINE_CALIBRATED = "baseline_calibrated"
    TRAIN = "train"
    EVAL = "eval"
    CHECKPOINT = "checkpoint"
    RUN_RESUME = "run_resume"
    RUN_END = "run_end"
    RUN_ABORT = "run_abort"


@dataclass(frozen=True, slots=True)
class JournalIdentity:
    """Identity tuple which must remain constant across one logical journal."""

    logical_run_id: str
    execution_digest: str
    aggregation_digest: str
    construction_digest: str
    family_digest: str
    run_class: RunClass

    def __post_init__(self) -> None:
        _nonempty_string(self.logical_run_id, path="logical_run_id")
        for field in (
            "execution_digest",
            "aggregation_digest",
            "construction_digest",
            "family_digest",
        ):
            _sha256(getattr(self, field), path=field)
        if type(self.run_class) is not RunClass:
            raise TypeError("run_class must be a RunClass")


@dataclass(frozen=True, slots=True)
class D1JournalEvent:
    """One immutable event envelope; serialization always includes its schema."""

    event_kind: JournalEventKind
    event_seq: int
    logical_run_id: str
    execution_digest: str
    aggregation_digest: str
    construction_digest: str
    family_digest: str
    run_class: RunClass
    successful_step: int
    payload: Mapping[str, FrozenJson]

    def __post_init__(self) -> None:
        if type(self.event_kind) is not JournalEventKind:
            raise TypeError("event_kind must be a JournalEventKind")
        _nonnegative_int(self.event_seq, path="event_seq")
        _nonnegative_int(self.successful_step, path="successful_step")
        identity = self.identity
        if type(self.payload) is not dict and not isinstance(self.payload, MappingProxyType):
            raise _error("invalid_type", "payload", "must be a JSON object")
        frozen = _freeze_json(self.payload, path="payload")
        if not isinstance(frozen, Mapping):
            raise _error("invalid_type", "payload", "must be a JSON object")
        object.__setattr__(self, "payload", frozen)
        del identity

    @property
    def schema_version(self) -> str:
        return JOURNAL_EVENT_SCHEMA

    @property
    def identity(self) -> JournalIdentity:
        return JournalIdentity(
            logical_run_id=self.logical_run_id,
            execution_digest=self.execution_digest,
            aggregation_digest=self.aggregation_digest,
            construction_digest=self.construction_digest,
            family_digest=self.family_digest,
            run_class=self.run_class,
        )

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_kind": self.event_kind.value,
            "event_seq": self.event_seq,
            "logical_run_id": self.logical_run_id,
            "execution_digest": self.execution_digest,
            "aggregation_digest": self.aggregation_digest,
            "construction_digest": self.construction_digest,
            "family_digest": self.family_digest,
            "run_class": self.run_class.value,
            "successful_step": self.successful_step,
            "payload": _thaw_json(self.payload),
        }


def canonical_event_line(event: D1JournalEvent) -> bytes:
    """Serialize one event to the sole admitted UTF-8 JSONL representation."""
    if type(event) is not D1JournalEvent:
        raise TypeError("event must be a D1JournalEvent")
    return (
        json.dumps(
            event.manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _error("duplicate_field", key, "duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _error("nonfinite_json", "json", f"{value} is not a finite JSON number")


def parse_event_line(line: bytes) -> D1JournalEvent:
    """Parse one canonical JSONL event while rejecting JSON's permissive extensions."""
    if type(line) is not bytes:
        raise TypeError("line must be bytes")
    if not line.endswith(b"\n"):
        raise _error("partial_line", "line", "event line must end with one newline")
    if b"\n" in line[:-1]:
        raise _error("multiple_events", "line", "line must contain exactly one event")
    try:
        text = line[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error("invalid_encoding", "line", "event must be valid UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except D1JournalError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise _error("invalid_json", "line", "event must be valid JSON") from exc
    data = _mapping_exact(raw, path="event", fields=_EVENT_FIELDS)
    if data["schema_version"] != JOURNAL_EVENT_SCHEMA:
        raise _error("unsupported_schema", "schema_version", "unsupported event schema_version")
    try:
        kind = JournalEventKind(data["event_kind"])
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "event_kind", "unsupported event kind") from exc
    try:
        run_class = RunClass(data["run_class"])
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "run_class", "unsupported run class") from exc
    event = D1JournalEvent(
        event_kind=kind,
        event_seq=_nonnegative_int(data["event_seq"], path="event_seq"),
        logical_run_id=_nonempty_string(data["logical_run_id"], path="logical_run_id"),
        execution_digest=_sha256(data["execution_digest"], path="execution_digest"),
        aggregation_digest=_sha256(data["aggregation_digest"], path="aggregation_digest"),
        construction_digest=_sha256(data["construction_digest"], path="construction_digest"),
        family_digest=_sha256(data["family_digest"], path="family_digest"),
        run_class=run_class,
        successful_step=_nonnegative_int(data["successful_step"], path="successful_step"),
        payload=cast(Mapping[str, FrozenJson], data["payload"]),
    )
    if canonical_event_line(event) != line:
        raise _error("noncanonical_json", "line", "event is not canonical JSON")
    return event


@dataclass(frozen=True, slots=True)
class JournalCheckpointState:
    """The immutable journal prefix identity persisted inside a checkpoint."""

    logical_run_id: str
    execution_digest: str
    last_event_sequence: int
    prefix_sha256: str
    successful_step: int

    def __post_init__(self) -> None:
        _nonempty_string(self.logical_run_id, path="logical_run_id")
        _sha256(self.execution_digest, path="execution_digest")
        _nonnegative_int(self.last_event_sequence, path="last_event_sequence")
        _sha256(self.prefix_sha256, path="prefix_sha256")
        _nonnegative_int(self.successful_step, path="successful_step")

    @property
    def schema_version(self) -> str:
        return JOURNAL_STATE_SCHEMA

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "logical_run_id": self.logical_run_id,
            "execution_digest": self.execution_digest,
            "last_event_sequence": self.last_event_sequence,
            "prefix_sha256": self.prefix_sha256,
            "successful_step": self.successful_step,
        }

    @classmethod
    def from_mapping(cls, value: object) -> JournalCheckpointState:
        data = _mapping_exact(value, path="journal_state", fields=_STATE_FIELDS)
        if data["schema_version"] != JOURNAL_STATE_SCHEMA:
            raise _error("unsupported_schema", "schema_version", "unsupported journal schema")
        return cls(
            logical_run_id=_nonempty_string(data["logical_run_id"], path="logical_run_id"),
            execution_digest=_sha256(data["execution_digest"], path="execution_digest"),
            last_event_sequence=_nonnegative_int(
                data["last_event_sequence"], path="last_event_sequence"
            ),
            prefix_sha256=_sha256(data["prefix_sha256"], path="prefix_sha256"),
            successful_step=_nonnegative_int(data["successful_step"], path="successful_step"),
        )
