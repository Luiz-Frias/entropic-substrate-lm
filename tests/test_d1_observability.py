"""D1 journal-first observability and strict MLflow projection contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    EvidenceGap,
    execution_segment_id,
)
from entropic_substrate.kernel.identity.d1_primitives import RunClass
from entropic_substrate.ladder.capture.adapters.d1_observability import (
    D1JournalObserver,
    D1MlflowObserver,
    ObservedD1Journal,
)
from entropic_substrate.ladder.capture.records.d1_journal import (
    JournalTailQuarantineReceipt,
    read_journal,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_work import ResourceWorkSnapshot
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_to_mapping,
)

pytestmark = pytest.mark.unit


def _event(
    kind: JournalEventKind,
    sequence: int,
    step: int,
    *,
    payload: dict[str, object] | None = None,
) -> D1JournalEvent:
    return D1JournalEvent(
        event_kind=kind,
        event_seq=sequence,
        logical_run_id="d1-pilot-both-seed-7",
        execution_digest="1" * 64,
        aggregation_digest="2" * 64,
        construction_digest="3" * 64,
        family_digest="4" * 64,
        run_class=RunClass.PILOT,
        successful_step=step,
        payload=payload or {"kind": kind.value},
    )


def _resource(*, step: int = 1, eval_index: int = 3) -> ResourceWorkSnapshot:
    logical_run_id = "d1-pilot-both-seed-7"

    def gap(field: str) -> EvidenceGap:
        return EvidenceGap(field=field, reason="meter unavailable")

    return ResourceWorkSnapshot(
        logical_run_id=logical_run_id,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(logical_run_id, 0),
        training_step=step,
        eval_index=eval_index,
        observed_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
        trainer_wall_seconds=12.5,
        committed_quality_source_bytes=1024,
        committed_quality_model_tokens=1024,
        online_train_token_positions=512,
        target_train_token_positions=640,
        baseline_calibration_positions=128,
        eval_positions=256,
        total_model_token_positions=1536,
        gpu_train_device_seconds=gap("gpu_train_device_seconds"),
        gpu_eval_device_seconds=gap("gpu_eval_device_seconds"),
        peak_gpu_allocated_bytes=gap("peak_gpu_allocated_bytes"),
        peak_gpu_reserved_bytes=gap("peak_gpu_reserved_bytes"),
        cumulative_train_flops=gap("cumulative_train_flops"),
        cumulative_eval_flops=gap("cumulative_eval_flops"),
        flop_model_identity=gap("flop_model_identity"),
        energy_joules=gap("energy_joules"),
        energy_sample_count=gap("energy_sample_count"),
        rate_estimated_cost=gap("rate_estimated_cost"),
    )


def _eval_event() -> D1JournalEvent:
    work = _resource()
    return _event(
        JournalEventKind.EVAL,
        3,
        1,
        payload={
            "eval_index": work.eval_index,
            "quality": {
                "nats_per_byte": 1.2345678901234567,
                "effective_rank": 7.75,
                "state_rms_l2_norm": 1.25,
                "target_effective_rank": 1.95,
                "target_numerical_rank": 2,
                "target_rms_l2_norm": 0.61,
                "predicted_effective_rank": 1.9,
                "predicted_numerical_rank": 2,
                "predicted_rms_l2_norm": 0.55,
                "residual_mean_l2_norm": 0.02,
                "target_predicted_exact_match_count": 0,
                "target_predicted_exact_match_fraction": 0.0,
                "state_variance": [0.2, 0.3],
                "state_covariance_eigenvalues": [0.1, 0.4],
                "state_sample_covariance": [[1.0, 0.0], [0.0, 1.0]],
            },
            "resource_work": resource_snapshot_to_mapping(work),
        },
    )


def _resume_segment_provenance(checkpoint_sha256: str) -> dict[str, FrozenJson]:
    return cast(
        dict[str, FrozenJson],
        {
            "run_start_provenance": {
                "execution_segment_id": "d1-pilot-both-seed-7:segment:1",
                "execution_segment_index": 1,
                "logical_run_id": "d1-pilot-both-seed-7",
                "resume_checkpoint_sha256": checkpoint_sha256,
                "segment_start_mode": "resume",
            },
            "trainer_wall_origin": {
                "monotonic_ns": 20_000,
                "started_at": "2026-08-24T12:30:00+00:00",
            },
        },
    )


class _OrderingObserver:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def observe(self, event: D1JournalEvent) -> None:
        self.calls.append(f"observe:{event.event_kind.value}")


class _RecordingMlflowClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str, float, int]] = []

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = 0,
    ) -> None:
        if self.fail:
            raise RuntimeError("deliberate MLflow write defect")
        self.calls.append((run_id, key, value, step))


def _started_journal(
    path: Path,
    observer: D1JournalObserver,
) -> ObservedD1Journal:
    journal = ObservedD1Journal.fresh(
        path,
        _event(JournalEventKind.RUN_START, 0, 0),
        observer=observer,
    )
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.TRAIN, 2, 1))
    return journal


def test_observer_runs_only_after_authoritative_journal_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import entropic_substrate.ladder.capture.records.d1_journal as journal_module

    calls: list[str] = []
    monkeypatch.setattr(journal_module.os, "fsync", lambda _fd: calls.append("fsync"))
    observer = _OrderingObserver(calls)
    journal = ObservedD1Journal.fresh(
        tmp_path / "run.jsonl",
        _event(JournalEventKind.RUN_START, 0, 0),
        observer=observer,
    )
    assert calls == ["fsync", "observe:run_start"]

    calls.clear()
    event = _event(JournalEventKind.BASELINE_CALIBRATED, 1, 0)
    journal.append(event)

    assert calls == ["fsync", "observe:baseline_calibrated"]
    assert read_journal(tmp_path / "run.jsonl")[-1] == event
    journal.close()


def test_mlflow_observer_projects_only_registered_scalars_at_journal_step(
    tmp_path: Path,
) -> None:
    client = _RecordingMlflowClient()
    observer = D1MlflowObserver(client=client, run_id="mlflow-run-123")
    journal = _started_journal(tmp_path / "run.jsonl", observer)
    event = _eval_event()

    journal.append(event)

    metrics = {key: value for _run_id, key, value, _step in client.calls}
    assert metrics["d1/quality/measured/nats_per_byte"] == 1.2345678901234567
    assert metrics["d1/quality/derived_from_measured/effective_rank"] == 7.75
    assert metrics["d1/quality/derived_from_measured/predicted_rms_l2_norm"] == 0.55
    assert metrics["d1/quality/measured/target_predicted_exact_match_count"] == 0.0
    assert metrics["d1/resource/measured/trainer_wall_seconds"] == 12.5
    assert metrics["d1/cursor/measured/eval_index"] == 3.0
    assert all(run_id == "mlflow-run-123" for run_id, *_rest in client.calls)
    assert all(step == event.successful_step for *_rest, step in client.calls)
    assert not any("state_variance" in key or "covariance" in key for key in metrics)
    assert not any("gpu_" in key or "flops" in key or "cost" in key for key in metrics)
    assert read_journal(tmp_path / "run.jsonl")[-1] == event
    journal.close()


def test_mlflow_observer_failure_propagates_after_event_is_durable(tmp_path: Path) -> None:
    client = _RecordingMlflowClient(fail=True)
    journal = _started_journal(
        tmp_path / "run.jsonl",
        D1MlflowObserver(client=client, run_id="mlflow-run-123"),
    )
    event = _eval_event()

    with pytest.raises(RuntimeError, match="deliberate MLflow write defect"):
        journal.append(event)

    assert read_journal(tmp_path / "run.jsonl")[-1] == event
    assert journal.state.last_event_sequence == event.event_seq
    journal.close()


def test_checkpoint_and_resume_observe_the_exact_journal_created_tail(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    calls: list[str] = []
    observer = _OrderingObserver(calls)
    journal = _started_journal(path, observer)
    journal.append(_eval_event())
    checkpoint_state = journal.state
    checkpoint_sha256 = "a" * 64

    journal.append_checkpoint(checkpoint_sha256)
    checkpoint_event = read_journal(path)[-1]
    assert checkpoint_event.event_kind is JournalEventKind.CHECKPOINT
    assert calls[-1] == "observe:checkpoint"
    journal.append(_event(JournalEventKind.TRAIN, 5, 2))
    journal.close()

    calls.clear()
    segment_provenance = _resume_segment_provenance(checkpoint_sha256)
    quarantine_path = tmp_path / "run.tail.jsonl"
    resumed = ObservedD1Journal.resume(
        path,
        checkpoint_state,
        checkpoint_sha256=checkpoint_sha256,
        segment_provenance=segment_provenance,
        quarantine_path=quarantine_path,
        observer=observer,
    )
    resume_event = read_journal(path)[-1]
    assert resume_event.event_kind is JournalEventKind.RUN_RESUME
    assert (
        resume_event.payload["run_start_provenance"] == segment_provenance["run_start_provenance"]
    )
    assert resume_event.payload["trainer_wall_origin"] == segment_provenance["trainer_wall_origin"]
    assert calls == ["observe:run_resume"]
    receipt = resumed.tail_quarantine
    assert type(receipt) is JournalTailQuarantineReceipt
    assert receipt.quarantine_path == quarantine_path
    assert resume_event.payload["tail_quarantine"] == receipt.manifest
    resumed.close()


def test_observed_journal_rejects_invalid_observer_and_mlflow_run_identity(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="observer"):
        ObservedD1Journal.fresh(
            tmp_path / "run.jsonl",
            _event(JournalEventKind.RUN_START, 0, 0),
            observer=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="run_id"):
        D1MlflowObserver(client=_RecordingMlflowClient(), run_id=" ")
