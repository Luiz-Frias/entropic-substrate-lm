"""Small canonical D1 journals for authoritative read-side tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.d1_training_step import D1TrainStepResult
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
    canonical_event_line,
)
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_provenance import (
    RESOURCE_TELEMETRY_SCHEMA_VERSION,
    BillingRoundingPolicy,
    BillingUnit,
    EvidenceGap,
    Money,
    PurchaseOption,
    QuotedRate,
    RateScope,
    RoundingMode,
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.experiments.d1_execution import PAIRWISE_COSINE_SEMANTIC
from entropic_substrate.experiments.d1_primitives import EvidenceClass, RunClass, canonical_digest

LOGICAL_RUN_ID = "d1-confirmatory-both-seed-7"
IDENTITY = {
    "logical_run_id": LOGICAL_RUN_ID,
    "execution_digest": "1" * 64,
    "aggregation_digest": "2" * 64,
    "construction_digest": "3" * 64,
    "family_digest": "4" * 64,
    "run_class": RunClass.CONFIRMATORY,
}
STARTED_AT = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


def plan_manifest() -> dict[str, object]:
    return {
        "arm": "both",
        "code": None,
        "construction_digest": IDENTITY["construction_digest"],
        "container": None,
        "data": None,
        "evidence": {"telemetry_schema_version": 1},
        "family_digest": IDENTITY["family_digest"],
        "logical_run_id": LOGICAL_RUN_ID,
        "pairing_id": "seed-7",
        "preregistration": None,
        "profile_name": "d1-both",
        "run_class": RunClass.CONFIRMATORY.value,
        "schema_version": "d1-run-plan-v1",
        "seed": 7,
    }


def start_provenance(segment: int = 0, checkpoint_sha256: str | None = None) -> RunStartProvenance:
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical_run_id=LOGICAL_RUN_ID,
        execution_segment_index=segment,
        execution_segment_id=execution_segment_id(LOGICAL_RUN_ID, segment),
        segment_start_mode=SegmentStartMode.FRESH if segment == 0 else SegmentStartMode.RESUME,
        resume_checkpoint_sha256=checkpoint_sha256,
        accelerator_model="A100-SXM4-80GB",
        accelerator_count=1,
        provider="runpod",
        provider_sku="NVIDIA A100-SXM4-80GB",
        region="US",
        market="secure-cloud",
        purchase_option=PurchaseOption.ON_DEMAND,
        image_digest="sha256:" + "5" * 64,
        cuda_version="12.6",
        driver_version="560.35",
        pytorch_version="2.13.0",
        compiler_enabled=False,
        compiler_version="disabled",
        quoted_rate=QuotedRate(
            amount=Money(Decimal("1.59"), "USD"),
            billing_unit=BillingUnit.HOUR,
            rounding_policy=BillingRoundingPolicy(1, RoundingMode.CEILING, 1),
            rate_scope=RateScope.PER_ACCELERATOR,
            source="runpod-api",
            quote_id=f"quote-{segment}",
            observed_at=STARTED_AT + timedelta(hours=segment, seconds=-1),
            evidence_class=EvidenceClass.MEASURED,
        ),
        pod_created_at=STARTED_AT + timedelta(hours=segment, seconds=-1),
        trainer_started_at=STARTED_AT + timedelta(hours=segment),
    )


def _origin(segment: int) -> dict[str, object]:
    start = STARTED_AT + timedelta(hours=segment)
    return {"monotonic_ns": segment * 1_000_000, "started_at": start.isoformat()}


def _gap(field: str) -> EvidenceGap:
    return EvidenceGap(field=field, reason="fixture meter unavailable")


def resource(step: int, eval_index: int, *, segment: int = 0) -> ResourceWorkSnapshot:
    train_positions = step * 4
    eval_positions = (eval_index + 1) * 4
    baseline_positions = 8
    return ResourceWorkSnapshot(
        logical_run_id=LOGICAL_RUN_ID,
        execution_segment_index=segment,
        execution_segment_id=execution_segment_id(LOGICAL_RUN_ID, segment),
        training_step=step,
        eval_index=eval_index,
        observed_at=STARTED_AT + timedelta(hours=segment, minutes=eval_index + 1),
        trainer_wall_seconds=float(eval_index + 1),
        committed_quality_source_bytes=train_positions,
        committed_quality_model_tokens=train_positions,
        online_train_token_positions=train_positions,
        target_train_token_positions=train_positions,
        baseline_calibration_positions=baseline_positions,
        eval_positions=eval_positions,
        total_model_token_positions=train_positions * 2 + baseline_positions + eval_positions,
        gpu_train_device_seconds=_gap("gpu_train_device_seconds"),
        gpu_eval_device_seconds=_gap("gpu_eval_device_seconds"),
        peak_gpu_allocated_bytes=_gap("peak_gpu_allocated_bytes"),
        peak_gpu_reserved_bytes=_gap("peak_gpu_reserved_bytes"),
        cumulative_train_flops=_gap("cumulative_train_flops"),
        cumulative_eval_flops=_gap("cumulative_eval_flops"),
        flop_model_identity=_gap("flop_model_identity"),
        energy_joules=_gap("energy_joules"),
        energy_sample_count=_gap("energy_sample_count"),
        rate_estimated_cost=_gap("rate_estimated_cost"),
    )


def quality(step: int) -> dict[str, object]:
    language = 0.8 - step * 0.01
    future = 0.9 - step * 0.01
    zero = future + 0.1
    mean = future + 0.05
    return {
        "batch_count": 1,
        "bits_per_byte": language / math.log(2.0),
        "coverage_50": 0.5,
        "coverage_80": 0.8,
        "coverage_95": 0.95,
        "effective_rank": 2.0,
        "future_nll": future,
        "language_loss": language,
        "mean_baseline_improvement": mean - future,
        "mean_displacement_nll": mean,
        "mean_pairwise_cosine_distance": 0.8,
        "nats_per_byte": language,
        "numerical_rank": 2,
        "online_positions": 4,
        "pairwise_cosine_semantic": PAIRWISE_COSINE_SEMANTIC,
        "perplexity": math.exp(language),
        "raw_covariance_penalty": 0.02,
        "raw_variance_penalty": 0.01,
        "residual_elements": 8,
        "residual_mean": 0.0,
        "residual_rms": 1.0,
        "residual_variance": 1.0,
        "state_covariance_eigenvalues": [0.5, 1.5],
        "state_variance": [0.75, 1.25],
        "weighted_covariance_penalty": 0.002,
        "weighted_variance_penalty": 0.001,
        "zero_baseline_improvement": zero - future,
        "zero_displacement_nll": zero,
    }


def _train(step: int) -> dict[str, object]:
    return D1TrainStepResult(
        successful_step=step,
        learning_rate=0.001,
        objective=1.0,
        language_loss=0.8,
        future_nll=0.9,
        variance_penalty=0.0,
        covariance_penalty=0.0,
        raw_variance_penalty=0.0,
        raw_covariance_penalty=0.0,
        zero_displacement_nll=1.0,
        mean_displacement_nll=0.95,
        residual_mean=0.0,
        residual_rms=1.0,
        residual_variance=1.0,
        coverage_50=0.5,
        coverage_80=0.8,
        coverage_95=0.95,
        effective_rank=2.0,
        numerical_rank=2,
        gradient_norm=1.0,
    ).manifest


def event(
    kind: JournalEventKind, sequence: int, step: int, payload: dict[str, object]
) -> D1JournalEvent:
    return D1JournalEvent(
        event_kind=kind,
        event_seq=sequence,
        successful_step=step,
        payload=payload,
        **IDENTITY,  # type: ignore[arg-type]
    )


def valid_events(*, resumed: bool = False) -> list[D1JournalEvent]:
    plan = plan_manifest()
    start = start_provenance()
    baseline = D1BaselineCalibrationAttestation(
        protocol_digest="6" * 64,
        spec_digest="7" * 64,
        split_digest="8" * 64,
        source_order_digest="9" * 64,
        mean_parameter_digest="a" * 64,
        codec_digest="b" * 64,
        initialization_seed=7,
        num_batches=1,
        online_positions=4,
        source_span_positions=8,
    )
    ledger = ResourceLedgerState(
        logical_run_id=LOGICAL_RUN_ID,
        execution_segment_index=0,
        next_eval_index=0,
        last_training_step=None,
        trainer_wall_seconds=0.0,
        committed_quality_source_bytes=0,
        committed_quality_model_tokens=0,
        online_train_token_positions=0,
        target_train_token_positions=0,
        baseline_calibration_positions=8,
        eval_positions=0,
        last_optional_readings=(),
        last_flop_model_identity=None,
        last_energy_sample_count=None,
    )
    items = [
        event(
            JournalEventKind.RUN_START,
            0,
            0,
            {
                "plan": plan,
                "plan_digest": canonical_digest(plan),
                "run_start_provenance": start.manifest,
                "trainer_wall_origin": _origin(0),
            },
        ),
        event(
            JournalEventKind.BASELINE_CALIBRATED,
            1,
            0,
            {"attestation": baseline.manifest, "resource_ledger": ledger.manifest},
        ),
        event(
            JournalEventKind.EVAL,
            2,
            0,
            {
                "eval_index": 0,
                "quality": quality(0),
                "resource_work": resource_snapshot_to_mapping(resource(0, 0)),
            },
        ),
        event(
            JournalEventKind.TRAIN,
            3,
            1,
            {
                "data_order": {"batches_attested": 1, "sha256": "c" * 64},
                "scheduled_log_boundary": False,
                "train": _train(1),
            },
        ),
        event(
            JournalEventKind.EVAL,
            4,
            1,
            {
                "eval_index": 1,
                "quality": quality(1),
                "resource_work": resource_snapshot_to_mapping(resource(1, 1)),
            },
        ),
        event(JournalEventKind.CHECKPOINT, 5, 1, {"checkpoint_sha256": "d" * 64}),
    ]
    if resumed:
        resume = start_provenance(1, "d" * 64)
        items.append(
            event(
                JournalEventKind.RUN_RESUME,
                6,
                1,
                {
                    "checkpoint_sha256": "d" * 64,
                    "run_start_provenance": resume.manifest,
                    "tail_quarantine": {
                        "kind": "no_tail",
                        "schema_version": "d1-tail-quarantine/v1",
                    },
                    "trainer_wall_origin": _origin(1),
                },
            )
        )
    segment = 1 if resumed else 0
    sequence = 7 if resumed else 6
    items.extend(
        [
            event(
                JournalEventKind.TRAIN,
                sequence,
                2,
                {
                    "data_order": {
                        "batches_attested": 1 if resumed else 2,
                        "sha256": "e" * 64,
                    },
                    "scheduled_log_boundary": False,
                    "train": _train(2),
                },
            ),
            event(
                JournalEventKind.EVAL,
                sequence + 1,
                2,
                {
                    "eval_index": 2,
                    "quality": quality(2),
                    "resource_work": resource_snapshot_to_mapping(resource(2, 2, segment=segment)),
                },
            ),
            event(
                JournalEventKind.CHECKPOINT,
                sequence + 2,
                2,
                {"checkpoint_sha256": "f" * 64},
            ),
            event(JournalEventKind.RUN_END, sequence + 3, 2, {"outcome": "success"}),
        ]
    )
    return items


def replace_payload(item: D1JournalEvent, payload: dict[str, object]) -> D1JournalEvent:
    return replace(item, payload=payload)


def resequence(events: list[D1JournalEvent]) -> list[D1JournalEvent]:
    return [replace(item, event_seq=index) for index, item in enumerate(events)]


def write_events(path: Path, events: list[D1JournalEvent]) -> None:
    path.write_bytes(b"".join(canonical_event_line(item) for item in events))
