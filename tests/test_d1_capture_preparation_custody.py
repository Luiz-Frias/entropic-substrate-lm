"""Canonical cross-process custody for D1 COMMIT_REQUIRED capture preparation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from entropic_substrate.evidence.d1_capture_preparation_custody import (
    D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA,
    D1CapturePreparationCustodyError,
    capture_preparation_custody_from_preparation,
    hydrate_d1_capture_preparation_custody,
    parse_d1_capture_preparation_custody,
    serialize_d1_capture_preparation_custody,
)
from entropic_substrate.evidence.d1_capture_service import D1CapturePreparationStatus

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_capture_service import _service

pytestmark = pytest.mark.unit


def _case(tmp_path: Path):
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, _, repo_root = _service(tmp_path, source_root)
    preparation = service.prepare_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    return preparation, repo_root / "results" / "captured"


def _mapping(payload: str) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(payload))


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def test_preparation_round_trip_rehydrates_authority_from_raw_capture_files(
    tmp_path: Path,
) -> None:
    preparation, capture_root = _case(tmp_path)
    record = capture_preparation_custody_from_preparation(preparation)
    payload = serialize_d1_capture_preparation_custody(record)

    restored = parse_d1_capture_preparation_custody(payload)
    hydrated = hydrate_d1_capture_preparation_custody(restored, capture_root=capture_root)

    assert restored == record
    assert restored.schema_version == D1_CAPTURE_PREPARATION_CUSTODY_SCHEMA
    assert restored.status is D1CapturePreparationStatus.COMMIT_REQUIRED
    assert restored.record_sha256 == record.record_sha256
    assert hydrated is not preparation
    assert hydrated.aggregate.manifest == preparation.aggregate.manifest
    assert hydrated.pointer == preparation.pointer
    assert hydrated.pointer_bytes == preparation.pointer_bytes
    assert payload == serialize_d1_capture_preparation_custody(restored)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("status"), "keys"),
        (lambda value: value.update({"extra": True}), "keys"),
        (lambda value: value.update({"schema_version": "unknown"}), "schema"),
        (lambda value: value.update({"status": "captured"}), "status"),
        (lambda value: value["aggregate"].update({"sha256": "0" * 64}), "aggregate"),
        (lambda value: value["pointer"].update({"sha256": "0" * 64}), "pointer"),
        (lambda value: value["pointer"].update({"bytes_base64": "AA=="}), "pointer"),
    ],
)
def test_parser_refuses_missing_unknown_or_substituted_authority(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    preparation, _ = _case(tmp_path)
    value = _mapping(
        serialize_d1_capture_preparation_custody(
            capture_preparation_custody_from_preparation(preparation)
        )
    )
    mutation(value)

    with pytest.raises(D1CapturePreparationCustodyError, match=match):
        parse_d1_capture_preparation_custody(_canonical(value))


def test_parser_refuses_duplicate_nonfinite_and_noncanonical_json(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path)
    payload = serialize_d1_capture_preparation_custody(
        capture_preparation_custody_from_preparation(preparation)
    )
    duplicate = '{"schema_version":"duplicate",' + payload[1:]

    for invalid in (duplicate, '{"value":NaN}', " " + payload, payload + "\n"):
        with pytest.raises(D1CapturePreparationCustodyError):
            parse_d1_capture_preparation_custody(invalid)


def test_hydration_rejects_artifact_drift_after_preparation(tmp_path: Path) -> None:
    preparation, capture_root = _case(tmp_path)
    record = parse_d1_capture_preparation_custody(
        serialize_d1_capture_preparation_custody(
            capture_preparation_custody_from_preparation(preparation)
        )
    )
    artifact = preparation.aggregate.complete_runs[0].capture_input.journal
    (capture_root / artifact.normalized_path).write_text("drift\n", encoding="utf-8")

    with pytest.raises(D1CapturePreparationCustodyError, match="hydrate"):
        hydrate_d1_capture_preparation_custody(record, capture_root=capture_root)


def test_record_digest_changes_with_any_pointer_or_aggregate_authority(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path)
    record = capture_preparation_custody_from_preparation(preparation)
    value = copy.deepcopy(record.manifest)
    value["pointer"]["object_id"] = "0" * 32 + ".dir"  # type: ignore[index]

    assert record.record_sha256 != hashlib.sha256(_canonical(value).encode()).hexdigest()
