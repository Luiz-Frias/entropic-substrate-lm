"""Terminal numerical defects are valid evidence through the public reader."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteClassification,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.d1_telemetry_values import D1TelemetryGap
from entropic_substrate.evidence.resource_provenance import execution_segment_id
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass

from .d1_capture_fixtures import FAMILY_DIGEST, PAIRING_ID, _events_for
from .d1_telemetry_fixtures import write_events

pytestmark = pytest.mark.unit


def _journal(tmp_path: Path, *, mutation: str = "defect") -> tuple[Path, str]:
    logical_run_id = "d1-sg-s7"
    construction = "2" * 64
    execution = "6" * 64
    aggregation = "7" * 64
    events, plan_digest = _events_for(
        D1Arm.SG,
        logical_run_id=logical_run_id,
        construction_digest=construction,
        execution_digest=execution,
        aggregation_digest=aggregation,
        checkpoint_sha256="f" * 64,
    )
    if mutation != "complete":
        record = D1TerminalNumericalDefectRecord(
            metric_name="nats_per_byte",
            classification=D1NonfiniteClassification.NAN,
            source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
            successful_step=2,
            max_steps=2,
            eval_index=2,
            arm=D1Arm.SG,
            seed=7,
            pairing_id=PAIRING_ID,
            logical_run_id="wrong-run" if mutation == "identity" else logical_run_id,
            run_class=RunClass.CONFIRMATORY,
            execution_segment_index=0,
            execution_segment_id=execution_segment_id(
                "wrong-run" if mutation == "identity" else logical_run_id, 0
            ),
            plan_digest=plan_digest,
            family_digest=FAMILY_DIGEST,
            construction_digest=construction,
            execution_digest=execution,
            aggregation_digest=aggregation,
            checkpoint=D1DefectCheckpointReference.receipt("d" * 64),
        )
        payload: dict[str, object] = (
            {"error": "generic failure"}
            if mutation == "generic_abort"
            else {"error": "nonfinite terminal metric", "numerical_defect": record.manifest}
        )
        events[-2] = replace(
            events[-2], event_kind=JournalEventKind.TRAINER_ABORTED, payload=payload
        )
    path = tmp_path / f"{mutation}.jsonl"
    write_events(path, events)
    return path, plan_digest


def test_public_reader_returns_canonical_terminal_defect_and_uses_strict_run_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, plan_digest = _journal(tmp_path)
    real_reader = telemetry_api.read_d1_run_evidence
    calls: list[Path] = []

    def spy(path: str | Path, **expected: object):
        calls.append(Path(path))
        return real_reader(path, **expected)  # type: ignore[arg-type]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", spy)
    record = telemetry_api.read_d1_terminal_defect(
        path,
        expected_logical_run_id="d1-sg-s7",
        expected_family_digest=FAMILY_DIGEST,
        expected_plan_digest=plan_digest,
    )

    assert calls == [path]
    assert record.metric_name == "nats_per_byte"
    assert record.classification is D1NonfiniteClassification.NAN
    assert record.successful_step == record.max_steps == 2
    assert record.checkpoint.sha256 == "d" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("complete", "trainer_aborted"),
        ("generic_abort", "numerical_defect"),
        ("identity", "identity"),
    ],
)
def test_reader_refuses_complete_generic_abort_or_mismatched_defect_identity(
    tmp_path: Path, mutation: str, message: str
) -> None:
    path, _ = _journal(tmp_path, mutation=mutation)

    with pytest.raises(D1TelemetryGap, match=message):
        telemetry_api.read_d1_terminal_defect(path)
