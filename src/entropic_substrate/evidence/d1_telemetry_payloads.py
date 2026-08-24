"""Strict D1 identity, provenance, and baseline event payload admission."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Final, cast

from entropic_substrate.evidence.d1_journal_values import D1JournalEvent
from entropic_substrate.evidence.d1_telemetry_values import gap
from entropic_substrate.evidence.resource_ledger_codec import (
    ResourceLedgerCodecGap,
    resource_ledger_state_from_mapping,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.evidence.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    run_start_from_mapping,
)
from entropic_substrate.experiments.d1_plan import D1_RUN_PLAN_SCHEMA
from entropic_substrate.experiments.d1_primitives import canonical_digest

__all__ = [
    "exact_fields",
    "parse_baseline_event",
    "parse_run_start_event",
    "parse_run_start_manifest",
    "parse_wall_origin",
    "positive_int",
    "require_mapping",
    "require_sha256",
]

_START_KEYS: Final = {"plan", "plan_digest", "run_start_provenance", "trainer_wall_origin"}
_PLAN_KEYS: Final = {
    "arm",
    "code",
    "construction_digest",
    "container",
    "data",
    "evidence",
    "family_digest",
    "logical_run_id",
    "pairing_id",
    "preregistration",
    "profile_name",
    "run_class",
    "schema_version",
    "seed",
}


def require_mapping(value: object, *, path: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise gap("invalid_mapping", path, "must be a JSON object with string keys")
    return cast(dict[str, object], value)


def exact_fields(value: dict[str, object], keys: set[str] | frozenset[str], *, path: str) -> None:
    if set(value) != set(keys):
        raise gap("field_set", path, "missing or extra registered fields")


def require_sha256(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise gap("invalid_sha256", path, "must be 64 lowercase hexadecimal characters")
    return value


def positive_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 1:
        raise gap("invalid_count", path, "must be a positive integer")
    return value


def _canonical_z(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise gap("invalid_timestamp", path, "must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise gap("invalid_timestamp", path, "must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or parsed.isoformat() != value:
        raise gap("invalid_timestamp", path, "must be canonical UTC ISO-8601")
    return value.removesuffix("+00:00") + "Z"


def parse_run_start_manifest(value: object, *, path: str) -> RunStartProvenance:
    manifest = require_mapping(value, path=path)
    wire = dict(manifest)
    wire["event"] = "resource.segment_start"
    for field in ("pod_created_at", "trainer_started_at"):
        wire[field] = _canonical_z(wire.get(field), path=f"{path}.{field}")
    quoted = require_mapping(wire.get("quoted_rate"), path=f"{path}.quoted_rate")
    quoted_wire = dict(quoted)
    quoted_wire["observed_at"] = _canonical_z(
        quoted_wire.get("observed_at"), path=f"{path}.quoted_rate.observed_at"
    )
    wire["quoted_rate"] = quoted_wire
    try:
        start = run_start_from_mapping(wire)
    except ResourceProvenanceCodecGap as error:
        raise gap("invalid_provenance", path, str(error)) from error
    if start.manifest != manifest:
        raise gap("noncanonical_provenance", path, "does not equal the typed manifest")
    return start


def parse_wall_origin(value: object, *, path: str, start: RunStartProvenance) -> None:
    origin = require_mapping(value, path=path)
    exact_fields(origin, {"monotonic_ns", "started_at"}, path=path)
    monotonic = origin["monotonic_ns"]
    if type(monotonic) is not int or monotonic < 0:
        raise gap("invalid_clock", f"{path}.monotonic_ns", "must be nonnegative")
    timestamp = _canonical_z(origin["started_at"], path=f"{path}.started_at")
    expected = start.trainer_started_at.isoformat().replace("+00:00", "Z")
    if timestamp != expected:
        raise gap("clock_mismatch", path, "does not match run-start trainer timestamp")


def parse_run_start_event(
    event: D1JournalEvent,
) -> tuple[dict[str, object], str, RunStartProvenance]:
    payload = require_mapping(event.manifest["payload"], path="run_start.payload")
    exact_fields(payload, _START_KEYS, path="run_start.payload")
    plan = require_mapping(payload["plan"], path="run_start.plan")
    exact_fields(plan, _PLAN_KEYS, path="run_start.plan")
    if plan["schema_version"] != D1_RUN_PLAN_SCHEMA:
        raise gap("plan_schema", "run_start.plan.schema_version", "unsupported plan schema")
    plan_digest = require_sha256(payload["plan_digest"], path="run_start.plan_digest")
    if canonical_digest(plan) != plan_digest:
        raise gap("plan_digest", "run_start.plan_digest", "does not bind the embedded plan")
    identity = event.identity
    for field, expected in (
        ("logical_run_id", identity.logical_run_id),
        ("construction_digest", identity.construction_digest),
        ("family_digest", identity.family_digest),
        ("run_class", identity.run_class.value),
    ):
        if plan[field] != expected:
            raise gap("identity_mismatch", f"run_start.plan.{field}", "differs from journal")
    start = parse_run_start_manifest(
        payload["run_start_provenance"], path="run_start.run_start_provenance"
    )
    if start.logical_run_id != identity.logical_run_id:
        raise gap("identity_mismatch", "run_start_provenance.logical_run_id", "differs")
    if start.execution_segment_index != 0 or start.segment_start_mode is not SegmentStartMode.FRESH:
        raise gap("segment_chain", "run_start_provenance", "first segment must be fresh zero")
    parse_wall_origin(
        payload["trainer_wall_origin"], path="run_start.trainer_wall_origin", start=start
    )
    return plan, plan_digest, start


def parse_baseline_event(event: D1JournalEvent) -> object:
    from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation

    payload = require_mapping(event.manifest["payload"], path="baseline.payload")
    exact_fields(payload, {"attestation", "resource_ledger"}, path="baseline.payload")
    raw = require_mapping(payload["attestation"], path="baseline.attestation")
    try:
        attestation = D1BaselineCalibrationAttestation(**raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise gap("invalid_baseline", "baseline.attestation", str(error)) from error
    if attestation.manifest != raw:
        raise gap("invalid_baseline", "baseline.attestation", "is not an exact manifest")
    ledger_raw = require_mapping(payload["resource_ledger"], path="baseline.resource_ledger")
    try:
        ledger = resource_ledger_state_from_mapping(
            {"event": "resource.ledger_state", **ledger_raw}
        )
    except ResourceLedgerCodecGap as error:
        raise gap("invalid_baseline", "baseline.resource_ledger", str(error)) from error
    if ledger.manifest != ledger_raw:
        raise gap("invalid_baseline", "baseline.resource_ledger", "is not canonical")
    if ledger.logical_run_id != event.logical_run_id or ledger.execution_segment_index != 0:
        raise gap("identity_mismatch", "baseline.resource_ledger", "belongs to another segment")
    if ledger.baseline_calibration_positions != attestation.source_span_positions:
        raise gap("baseline_work", "baseline.resource_ledger", "does not match attestation")
    return attestation
