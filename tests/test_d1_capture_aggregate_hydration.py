"""Persisted heterogeneous aggregates re-earn authority from raw capture files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_aggregate_codec import (
    D1CaptureAggregateHydrationError,
    hydrate_d1_capture_aggregate,
    parse_d1_capture_aggregate_declaration,
    serialize_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_capture_service import _service

pytestmark = pytest.mark.unit


def _case(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    return aggregate, root


def _canonical(value: dict[str, object]) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _hydrate(mapping: dict[str, object], root: Path, *, digest: str | None = None):
    payload = _canonical(mapping)
    return hydrate_d1_capture_aggregate(
        payload,
        expected_manifest_digest=canonical_digest(mapping) if digest is None else digest,
        capture_root=root,
    )


def test_real_heterogeneous_capture_round_trip_rebuilds_full_authority(tmp_path: Path) -> None:
    original, root = _case(tmp_path)
    payload = serialize_d1_capture_aggregate(original)
    declaration = parse_d1_capture_aggregate_declaration(payload)
    hydrated = hydrate_d1_capture_aggregate(
        payload,
        expected_manifest_digest=original.manifest_digest,
        capture_root=root,
    )

    assert declaration.roster == original.roster
    assert len(declaration.complete_inputs) == 3
    assert len(declaration.terminal_defect_inputs) == 1
    assert len(declaration.attempt_evidence) == 1
    assert declaration.manifest_digest == original.manifest_digest
    assert hydrated.manifest == original.manifest
    assert hydrated.manifest_digest == original.manifest_digest
    assert hydrated is not original
    assert payload == _canonical(original.manifest)


def test_dvc_bound_service_receipt_rehydrates_from_exported_capture_root(tmp_path: Path) -> None:
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, _, repo_root = _service(tmp_path, source_root)
    receipt = service.capture_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    payload = serialize_d1_capture_aggregate(receipt.aggregate)

    hydrated = hydrate_d1_capture_aggregate(
        payload,
        expected_manifest_digest=receipt.aggregate.manifest_digest,
        capture_root=repo_root / "results" / "captured",
    )

    assert hydrated.manifest == receipt.aggregate.manifest
    assert hydrated.dvc_handoff.analysis_ready is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"invented": True}),
        lambda value: value.pop("partition_proof"),
        lambda value: value["complete_runs"][0].update({"invented": True}),
        lambda value: value["terminal_defects"][0].pop("artifacts"),
        lambda value: value["attempt_lanes"][0]["attempt"].update({"invented": True}),
    ],
)
def test_parser_rejects_unknown_missing_or_nested_lane_drift(
    tmp_path: Path, mutation: object
) -> None:
    aggregate, _ = _case(tmp_path)
    mapping = aggregate.manifest
    mutation(mapping)  # type: ignore[operator]

    with pytest.raises(D1CaptureAggregateHydrationError):
        parse_d1_capture_aggregate_declaration(_canonical(mapping))


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"d1-heterogeneous-capture-aggregate-v1","schema_version":"x"}',
        '{"schema_version":NaN}',
        " {}",
        "{}\n",
        "[]",
    ],
)
def test_text_parser_rejects_duplicate_nonfinite_noncanonical_or_nonobject_json(
    payload: str,
) -> None:
    with pytest.raises(D1CaptureAggregateHydrationError):
        parse_d1_capture_aggregate_declaration(payload)


def test_expected_digest_and_artifact_bytes_are_independent_guards(tmp_path: Path) -> None:
    aggregate, root = _case(tmp_path)
    payload = serialize_d1_capture_aggregate(aggregate)
    with pytest.raises(D1CaptureAggregateHydrationError, match="expected manifest digest"):
        hydrate_d1_capture_aggregate(
            payload,
            expected_manifest_digest="0" * 64,
            capture_root=root,
        )

    artifact = aggregate.complete_runs[0].capture_input.journal
    (root / artifact.normalized_path).write_text("drift\n", encoding="utf-8")
    with pytest.raises(D1CaptureAggregateHydrationError, match="artifact"):
        hydrate_d1_capture_aggregate(
            payload,
            expected_manifest_digest=aggregate.manifest_digest,
            capture_root=root,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["partition_proof"]["complete_run_ids"].pop(),
        lambda value: value["pair_exclusions"].clear(),
        lambda value: value["complete_runs"][0]["data_order_segments"].clear(),
        lambda value: value["complete_runs"][0]["checkpoint_receipts"].clear(),
        lambda value: value["authority"].update({"mlflow": "authoritative"}),
        lambda value: value["dvc_handoff"]["artifact_paths"].pop(),
    ],
)
def test_rebuilt_full_manifest_rejects_partition_exclusion_reader_or_handoff_drift(
    tmp_path: Path, mutate: object
) -> None:
    aggregate, root = _case(tmp_path)
    mapping = aggregate.manifest
    mutate(mapping)  # type: ignore[operator]

    with pytest.raises(D1CaptureAggregateHydrationError, match="rebuilt manifest"):
        _hydrate(mapping, root)


def test_lane_identity_path_escape_and_symlink_are_refused(tmp_path: Path) -> None:
    aggregate, root = _case(tmp_path)
    mapping = aggregate.manifest
    mapping["complete_runs"][0]["artifacts"]["journal"]["path"] = "../escape.jsonl"  # type: ignore[index]
    with pytest.raises(D1CaptureAggregateHydrationError):
        _hydrate(mapping, root)

    aggregate, root = _case(tmp_path / "symlink")
    artifact = aggregate.complete_runs[0].capture_input.journal
    path = root / artifact.normalized_path
    target = root / "elsewhere.jsonl"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(D1CaptureAggregateHydrationError, match="artifact"):
        hydrate_d1_capture_aggregate(
            serialize_d1_capture_aggregate(aggregate),
            expected_manifest_digest=aggregate.manifest_digest,
            capture_root=root,
        )
