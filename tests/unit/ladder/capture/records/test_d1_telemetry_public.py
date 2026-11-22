"""Public D1 exports coexist with unchanged legacy telemetry readers."""

from __future__ import annotations

from entropic_substrate.ladder.capture.reader import telemetry
from entropic_substrate.ladder.capture.records import d1_telemetry


def test_public_d1_exports_are_the_evidence_facade_objects() -> None:
    assert telemetry.D1RunEvidence is d1_telemetry.D1RunEvidence
    assert telemetry.D1TrainerAbortEvidence is d1_telemetry.D1TrainerAbortEvidence
    assert telemetry.read_d1_run_evidence is d1_telemetry.read_d1_run_evidence
    assert telemetry.read_d1_run_evidence_bytes is d1_telemetry.read_d1_run_evidence_bytes
    assert telemetry.read_d1_eval_series is d1_telemetry.read_d1_eval_series
    assert telemetry.final_d1_eval_metric is d1_telemetry.final_d1_eval_metric


def test_legacy_reader_exports_remain_available() -> None:
    assert callable(telemetry.read_eval_series)
    assert callable(telemetry.final_eval_metric)
    assert callable(telemetry.read_train_series)
