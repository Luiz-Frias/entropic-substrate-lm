"""Canonical declaration parser and authority-rehydrator for D1 capture aggregates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

from entropic_substrate.kernel.identity.d1_primitives import require_sha256
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import (
    D1CapturePartitionProof,
    D1HeterogeneousCaptureAggregate,
    D1TerminalDefectInput,
)
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import (
    D1AttemptEvidence,
    D1CaptureInput,
)
from entropic_substrate.ladder.capture.custody.d1_capture_declaration_codec import (
    attempt_evidence_from_mapping,
    capture_input_from_mapping,
    launch_roster_from_mapping,
    terminal_defect_input_from_mapping,
    validate_capture_lane_path_partition,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1ExpectedRun,
    D1LaunchRoster,
)

__all__ = [
    "D1CaptureAggregateDeclaration",
    "D1CaptureAggregateHydrationError",
    "hydrate_d1_capture_aggregate",
    "parse_d1_capture_aggregate_declaration",
    "serialize_d1_capture_aggregate",
]

_SCHEMA = "d1-heterogeneous-capture-aggregate-v1"
_ROOT_KEYS = frozenset(
    {
        "attempt_lanes",
        "authority",
        "complete_runs",
        "dvc_handoff",
        "pair_exclusions",
        "partition_proof",
        "roster",
        "schema_version",
        "terminal_defects",
    }
)
_COMPLETE_KEYS = frozenset(
    {"artifacts", "checkpoint_receipts", "data_order_segments", "identity", "terminal_step"}
)
_TERMINAL_KEYS = frozenset({"artifacts", "data_order_segments", "defect", "identity"})
_AUTHORITY_KEYS = frozenset({"complete_reader", "defect_reader", "journal", "mlflow"})
_DVC_KEYS = frozenset(
    {"analysis_ready", "artifact_paths", "pending_paths", "pointer_paths", "schema_version"}
)
_PARTITION_KEYS = frozenset(
    {"complete_run_ids", "expected_logical_run_ids", "terminal_defect_run_ids"}
)
_EXCLUSION_KEYS = frozenset({"evidence_paths", "member_run_ids", "pairing_id", "reason"})


class D1CaptureAggregateHydrationError(ValueError):
    """A persisted aggregate declaration cannot re-earn capture authority."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CaptureAggregateHydrationError(message)
    raise D1CaptureAggregateHydrationError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _array(value: object, *, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be a JSON array")
    return cast(list[object], value)


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be a nonempty trimmed string")
    return value


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _expected(
    value: object,
    *,
    by_id: dict[str, D1ExpectedRun],
    label: str,
) -> D1ExpectedRun:
    identity = _mapping(value, label=f"{label}.identity")
    logical_run_id = _text(identity.get("logical_run_id"), label=f"{label}.identity.logical_run_id")
    expected = by_id.get(logical_run_id)
    if expected is None or _canonical(identity) != _canonical(expected.manifest):
        _fail(f"{label} identity differs from decoded roster")
    return expected


def _validate_declared_sections(root: dict[str, object]) -> None:
    _exact(_mapping(root["authority"], label="authority"), _AUTHORITY_KEYS, label="authority")
    _exact(
        _mapping(root["dvc_handoff"], label="dvc_handoff"),
        _DVC_KEYS,
        label="dvc_handoff",
    )
    _exact(
        _mapping(root["partition_proof"], label="partition_proof"),
        _PARTITION_KEYS,
        label="partition_proof",
    )
    for index, value in enumerate(_array(root["pair_exclusions"], label="pair_exclusions")):
        _exact(
            _mapping(value, label=f"pair_exclusions[{index}]"),
            _EXCLUSION_KEYS,
            label=f"pair_exclusions[{index}]",
        )


@dataclass(frozen=True, slots=True)
class D1CaptureAggregateDeclaration:
    """Canonical aggregate claim containing declarations but no validation token."""

    canonical_json: str
    roster: D1LaunchRoster
    complete_inputs: tuple[D1CaptureInput, ...]
    terminal_defect_inputs: tuple[D1TerminalDefectInput, ...]
    attempt_evidence: tuple[D1AttemptEvidence, ...]

    @property
    def manifest(self) -> dict[str, object]:
        value = json.loads(self.canonical_json)
        if type(value) is not dict:  # pragma: no cover - parser pins object root
            raise AssertionError("aggregate declaration lost its object root")
        return cast(dict[str, object], value)

    @property
    def manifest_digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate aggregate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite aggregate JSON constant {value!r}")


def parse_d1_capture_aggregate_declaration(payload: str) -> D1CaptureAggregateDeclaration:
    if type(payload) is not str or not payload:
        _fail("aggregate payload must be nonempty text")
    try:
        value: object = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except D1CaptureAggregateHydrationError:
        raise
    except (json.JSONDecodeError, UnicodeError) as error:
        _fail("aggregate payload is invalid JSON", error)
    root = _mapping(value, label="aggregate")
    _exact(root, _ROOT_KEYS, label="aggregate")
    if root["schema_version"] != _SCHEMA:
        _fail("aggregate schema_version is unsupported")
    if _canonical(root) != payload:
        _fail("aggregate JSON is not canonical")
    _validate_declared_sections(root)
    try:
        roster = launch_roster_from_mapping(root["roster"])
        expected_by_id = {item.logical_run_id: item for item in roster.expected_confirmatory}
        complete: list[D1CaptureInput] = []
        for index, value in enumerate(_array(root["complete_runs"], label="complete_runs")):
            row = _mapping(value, label=f"complete_runs[{index}]")
            _exact(row, _COMPLETE_KEYS, label=f"complete_runs[{index}]")
            expected = _expected(
                row["identity"], by_id=expected_by_id, label=f"complete_runs[{index}]"
            )
            complete.append(capture_input_from_mapping(row["artifacts"], expected=expected))
        defects: list[D1TerminalDefectInput] = []
        for index, value in enumerate(_array(root["terminal_defects"], label="terminal_defects")):
            row = _mapping(value, label=f"terminal_defects[{index}]")
            _exact(row, _TERMINAL_KEYS, label=f"terminal_defects[{index}]")
            expected = _expected(
                row["identity"], by_id=expected_by_id, label=f"terminal_defects[{index}]"
            )
            defects.append(terminal_defect_input_from_mapping(row["artifacts"], expected=expected))
        attempts_by_id = {item.logical_run_id: item for item in roster.attempts}
        attempts: list[D1AttemptEvidence] = []
        for index, value in enumerate(_array(root["attempt_lanes"], label="attempt_lanes")):
            row = _mapping(value, label=f"attempt_lanes[{index}]")
            embedded = _mapping(row.get("attempt"), label=f"attempt_lanes[{index}].attempt")
            attempt_id = _text(
                embedded.get("logical_run_id"),
                label=f"attempt_lanes[{index}].attempt.logical_run_id",
            )
            declared = attempts_by_id.get(attempt_id)
            if declared is None:
                _fail(f"attempt_lanes[{index}] is absent from decoded roster")
            attempts.append(attempt_evidence_from_mapping(row, declared=declared))
        proof = D1CapturePartitionProof(
            roster.expected_run_ids,
            frozenset(item.logical_run_id for item in complete),
            frozenset(item.logical_run_id for item in defects),
        )
        del proof
        if {item.attempt.logical_run_id for item in attempts} != set(attempts_by_id):
            _fail("attempt evidence set differs from decoded roster")
        validate_capture_lane_path_partition(tuple(complete), tuple(defects), tuple(attempts))
    except D1CaptureAggregateHydrationError:
        raise
    except Exception as error:
        _fail(f"aggregate declarations are invalid: {error}", error)
    return D1CaptureAggregateDeclaration(
        payload,
        roster,
        tuple(complete),
        tuple(defects),
        tuple(attempts),
    )


def serialize_d1_capture_aggregate(aggregate: D1HeterogeneousCaptureAggregate) -> str:
    if type(aggregate) is not D1HeterogeneousCaptureAggregate:
        raise TypeError("aggregate codec requires D1HeterogeneousCaptureAggregate")
    return _canonical(aggregate.manifest)


def _expected_digest(value: str) -> str:
    try:
        require_sha256(value, label="expected aggregate manifest digest")
    except (TypeError, ValueError) as error:
        _fail("expected aggregate manifest digest must be lowercase SHA-256", error)
    if value != value.lower():
        _fail("expected aggregate manifest digest must be lowercase SHA-256")
    return value


def hydrate_d1_capture_aggregate(
    payload: str,
    *,
    expected_manifest_digest: str,
    capture_root: Path,
) -> D1HeterogeneousCaptureAggregate:
    """Rebuild authority from raw artifacts and require exact manifest equality."""
    if not isinstance(cast(object, capture_root), Path):
        raise TypeError("aggregate hydration capture_root must be Path")
    expected = _expected_digest(expected_manifest_digest)
    declaration = parse_d1_capture_aggregate_declaration(payload)
    if declaration.manifest_digest != expected:
        _fail("aggregate declaration differs from expected manifest digest")
    try:
        rebuilt = build_d1_capture_aggregate(
            capture_root,
            roster=declaration.roster,
            complete_inputs=declaration.complete_inputs,
            terminal_defect_inputs=declaration.terminal_defect_inputs,
            attempt_evidence=declaration.attempt_evidence,
        )
    except Exception as error:
        _fail(f"aggregate rebuild failed from artifact or reader evidence: {error}", error)
    if serialize_d1_capture_aggregate(rebuilt) != declaration.canonical_json:
        _fail("rebuilt manifest differs from persisted aggregate declaration")
    if rebuilt.manifest_digest != expected:
        _fail("rebuilt manifest digest differs from expected manifest digest")
    return rebuilt
