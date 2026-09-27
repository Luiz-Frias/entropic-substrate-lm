"""Exact bounded encoding for the D1 operator inspection result."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.ladder.analyze.d1_inspect_codec import (
    D1_INSPECT_MAX_RESULT_BYTES,
    D1_INSPECT_RESULT_SCHEMA,
    D1InspectCodecError,
    d1_inspect_result_to_mapping,
    serialize_d1_inspect_result,
)
from entropic_substrate.ladder.analyze.d1_inspect_projection import compose_d1_inspect_view
from entropic_substrate.ladder.analyze.d1_inspect_values import D1InspectJournal
from entropic_substrate.ladder.capture.reader.d1_journal_snapshot import (
    read_d1_local_journal_snapshot,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_to_mapping,
)

from .d1_telemetry_fixtures import valid_events, write_events

pytestmark = pytest.mark.unit

_VECTOR_FIELDS = {
    "predicted_covariance_eigenvalues",
    "predicted_mean",
    "predicted_variance",
    "residual_mean_vector",
    "state_covariance_eigenvalues",
    "state_mean",
    "state_variance",
    "target_covariance_eigenvalues",
    "target_mean",
    "target_variance",
}


def _view(
    tmp_path: Path,
    *,
    events: list[D1JournalEvent] | None = None,
    allow_incomplete: bool = False,
):
    path = tmp_path / "run.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_events(path, valid_events() if events is None else events)
    snapshot = read_d1_local_journal_snapshot(
        path,
        max_bytes=path.stat().st_size,
        allow_incomplete=allow_incomplete,
    )
    return compose_d1_inspect_view(snapshot, evaluation_limit=2), snapshot


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def test_complete_result_mapping_is_exact_bounded_and_reuses_nested_codecs(
    tmp_path: Path,
) -> None:
    view, snapshot = _view(tmp_path)

    mapping = d1_inspect_result_to_mapping(view)
    payload = serialize_d1_inspect_result(view)

    assert set(mapping) == {
        "evaluations",
        "journal",
        "plan",
        "progress",
        "run",
        "schema_version",
        "state",
    }
    assert mapping["schema_version"] == D1_INSPECT_RESULT_SCHEMA
    assert mapping["state"]["status"] == "complete"  # type: ignore[index]
    assert mapping["progress"]["latest_segment"] == run_start_to_mapping(  # type: ignore[index]
        snapshot.run.segments[-1]
    )
    evaluations = mapping["evaluations"]
    assert evaluations["limit"] == 2  # type: ignore[index]
    assert evaluations["returned_count"] == 2  # type: ignore[index]
    assert evaluations["omitted_count"] == 1  # type: ignore[index]
    latest = evaluations["items"][-1]  # type: ignore[index]
    assert latest["resource_work"] == resource_snapshot_to_mapping(
        snapshot.run.evals[-1].resource_work
    )
    assert _VECTOR_FIELDS.isdisjoint(_keys(mapping))
    assert len(payload) <= D1_INSPECT_MAX_RESULT_BYTES
    assert json.loads(payload) == mapping
    assert (
        payload
        == json.dumps(
            mapping,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )


def test_terminal_state_variants_have_exact_nonnullable_shapes(tmp_path: Path) -> None:
    partial, _ = _view(
        tmp_path / "partial",
        events=valid_events()[:-1],
        allow_incomplete=True,
    )
    assert d1_inspect_result_to_mapping(partial)["state"] == {"status": "partial"}

    events = valid_events()
    events[-2] = replace(
        events[-2],
        event_kind=JournalEventKind.TRAINER_ABORTED,
        payload={"error": "secret-bearing provider stderr must not escape"},
    )
    trainer, _ = _view(tmp_path / "trainer", events=events, allow_incomplete=True)
    trainer_state = d1_inspect_result_to_mapping(trainer)["state"]
    assert set(trainer_state) == {  # type: ignore[arg-type]
        "numerical_defect",
        "run_finalization",
        "status",
        "trainer_abort",
    }
    assert "secret-bearing provider stderr" not in json.dumps(trainer_state)

    events = valid_events()
    events[-1] = replace(
        events[-1],
        event_kind=JournalEventKind.RUN_ABORT,
        payload={"error": "operator abort"},
    )
    aborted, _ = _view(tmp_path / "aborted", events=events, allow_incomplete=True)
    assert set(d1_inspect_result_to_mapping(aborted)["state"]) == {  # type: ignore[arg-type]
        "run_abort",
        "status",
    }


def test_encoder_refuses_non_nominal_and_over_budget_results(tmp_path: Path) -> None:
    view, _ = _view(tmp_path)

    with pytest.raises(TypeError, match="D1InspectView"):
        d1_inspect_result_to_mapping({})  # type: ignore[arg-type]
    oversized = replace(
        view,
        journal=D1InspectJournal(
            "/" + "x" * D1_INSPECT_MAX_RESULT_BYTES,
            view.journal.relative_path,
            view.journal.byte_count,
            view.journal.sha256,
        ),
    )
    with pytest.raises(D1InspectCodecError, match="byte limit"):
        serialize_d1_inspect_result(oversized)
