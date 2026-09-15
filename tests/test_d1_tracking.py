"""D1 run identity and journal-to-MLflow projection contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.d1_tracking import D1MetricExtractor, d1_tracking_descriptor
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
    canonical_event_line,
)
from entropic_substrate.evidence.d1_telemetry_values import D1_REPORT_ONLY_METRICS
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
    unavailable_resource_activation_registration,
)
from entropic_substrate.evidence.resource_work import (
    CostReading,
    ReadingMethod,
    ResourceWorkSnapshot,
    ScalarReading,
)
from entropic_substrate.evidence.resource_work_codec import resource_snapshot_to_mapping
from entropic_substrate.experiments.d1_config import load_d1_profile_catalog
from entropic_substrate.experiments.d1_plan import create_d1_excluded_pilot_plan
from entropic_substrate.experiments.d1_primitives import EvidenceClass
from entropic_substrate.experiments.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    EvidenceEnvelope,
)
from entropic_substrate.sinks import CompositeSink, MlflowSink, build_sink_for_run
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _plan() -> Any:
    profile = next(p for p in load_d1_profile_catalog() if p.name == "d1-both")
    return create_d1_excluded_pilot_plan(
        profile,
        logical_run_id="d1-pilot-both-seed-7",
        seed=7,
        pairing_id="d1-stabilization-pair-7",
        evidence=EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset(
                {EvidenceClass.MEASURED, EvidenceClass.DERIVED_FROM_MEASURED}
            ),
        ),
        data=DataIdentity(
            dataset_name="enwik8",
            dataset_digest="1" * 64,
            split_digest="2" * 64,
            sampler_construction_digest="3" * 64,
        ),
        code=CodeIdentity(commit="a" * 40, repository_digest="b" * 64),
        container=ContainerIdentity(image_digest="sha256:" + "c" * 64),
    )


def _start(logical_run_id: str = "d1-pilot-both-seed-7") -> RunStartProvenance:
    observed = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    return RunStartProvenance(
        schema_version=RESOURCE_TELEMETRY_SCHEMA_VERSION,
        logical_run_id=logical_run_id,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(logical_run_id, 0),
        segment_start_mode=SegmentStartMode.FRESH,
        resume_checkpoint_sha256=None,
        accelerator_model="NVIDIA A100 PCIe 80GB",
        accelerator_count=1,
        provider="runpod",
        provider_sku="a100-pcie-80gb",
        region="US-TX-3",
        market="secure-cloud",
        purchase_option=PurchaseOption.ON_DEMAND,
        image_digest="sha256:" + "c" * 64,
        cuda_version="12.6",
        driver_version="560.35.03",
        pytorch_version="2.13.0",
        compiler_enabled=True,
        compiler_version="triton-3.4.0",
        resource_activation=unavailable_resource_activation_registration(),
        quoted_rate=QuotedRate(
            amount=Money(amount=Decimal("1.19"), currency="USD"),
            billing_unit=BillingUnit.HOUR,
            rounding_policy=BillingRoundingPolicy(
                increment=1, mode=RoundingMode.CEILING, minimum_units=1
            ),
            rate_scope=RateScope.PER_ACCELERATOR,
            source="RunPod pricing API",
            quote_id="runpod-a100-pcie-us-tx-3-20260824",
            observed_at=observed,
            evidence_class=EvidenceClass.MEASURED,
        ),
        pod_created_at=observed,
        trainer_started_at=observed,
    )


def _resource(logical_run_id: str, *, eval_index: int = 3) -> ResourceWorkSnapshot:
    def gap(field: str) -> EvidenceGap:
        return EvidenceGap(field=field, reason="meter unavailable")

    return ResourceWorkSnapshot(
        logical_run_id=logical_run_id,
        execution_segment_index=0,
        execution_segment_id=execution_segment_id(logical_run_id, 0),
        training_step=250,
        eval_index=eval_index,
        observed_at=datetime(2026, 8, 24, 12, 5, tzinfo=UTC),
        trainer_wall_seconds=123.45678901234567,
        committed_quality_source_bytes=32_768,
        committed_quality_model_tokens=32_768,
        online_train_token_positions=32_768,
        target_train_token_positions=32_768,
        baseline_calibration_positions=1_024,
        eval_positions=2_048,
        total_model_token_positions=68_608,
        gpu_train_device_seconds=ScalarReading(
            value=120.125,
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        gpu_eval_device_seconds=gap("gpu_eval_device_seconds"),
        peak_gpu_allocated_bytes=ScalarReading(
            value=4_294_967_296.0,
            evidence_class=EvidenceClass.MEASURED,
            method=ReadingMethod.GPU_DEVICE_COUNTER,
        ),
        peak_gpu_reserved_bytes=gap("peak_gpu_reserved_bytes"),
        cumulative_train_flops=ScalarReading(
            value=1.25e15,
            evidence_class=EvidenceClass.MODEL_ESTIMATE,
            method=ReadingMethod.FLOP_MODEL,
        ),
        cumulative_eval_flops=gap("cumulative_eval_flops"),
        flop_model_identity="f" * 64,
        energy_joules=gap("energy_joules"),
        energy_sample_count=gap("energy_sample_count"),
        rate_estimated_cost=CostReading(
            amount=Money(amount=Decimal("0.042"), currency="USD"),
            evidence_class=EvidenceClass.RATE_ESTIMATE,
            method=ReadingMethod.RATE_TIMES_SCOPE,
        ),
    )


def _event(resource: ResourceWorkSnapshot | None = None) -> D1JournalEvent:
    plan = _plan()
    work = resource or _resource(plan.logical_run_id)
    return D1JournalEvent(
        event_kind=JournalEventKind.EVAL,
        event_seq=9,
        logical_run_id=plan.logical_run_id,
        execution_digest=plan.run.execution_digest,
        aggregation_digest=plan.run.aggregation_digest,
        construction_digest=plan.profile.draft.construction_digest,
        family_digest=plan.profile.family.family_digest,
        run_class=plan.run.run_class,
        successful_step=250,
        payload={
            "eval_index": work.eval_index,
            "quality": {
                "nats_per_byte": 1.2345678901234567,
                "bits_per_byte": 1.780853665290424,
                "future_nll": 0.625,
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


class _RecordingClient(FakeMlflowClient):
    def __init__(self) -> None:
        super().__init__()
        self.metric_steps: list[tuple[str, float, int]] = []

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self.metric_steps.append((key, value, step))
        super().log_metric(run_id, key, value, step)


def test_d1_descriptor_binds_plan_profile_and_execution_start_identity() -> None:
    plan = _plan()
    descriptor = d1_tracking_descriptor(plan, plan.profile, _start())

    assert descriptor.run_name == plan.logical_run_id
    assert descriptor.tiny is False
    assert descriptor.params["d1/plan_digest"] == plan.plan_digest
    assert descriptor.params["d1/execution_digest"] == plan.run.execution_digest
    assert descriptor.params["d1/dataset_digest"] == "1" * 64
    assert descriptor.params["d1/code_commit"] == "a" * 40
    assert descriptor.params["d1/container_image_digest"] == "sha256:" + "c" * 64
    assert descriptor.tags["d1/family_digest"] == plan.profile.family.family_digest
    assert descriptor.tags["resource/start_identity_digest"] == _start().identity_digest
    assert descriptor.params["resource/activation_mode"] == "gaps-only"
    assert descriptor.params["resource/activation_factory_digest"] == (
        _start().resource_activation.factory_registration_digest
    )
    assert descriptor.params["resource/meter_provider"] == "unavailable"
    assert descriptor.params["resource/meter_device_scope"] == "unavailable"
    assert descriptor.params["resource/flop_model_identity"] == "unavailable"
    assert descriptor.params["resource/power_integration_method"] == "unavailable"


def test_d1_descriptor_refuses_profile_logical_run_and_container_drift() -> None:
    plan = _plan()
    other_profile = next(p for p in load_d1_profile_catalog() if p.name == "d1-sg")
    with pytest.raises(ValueError, match="profile"):
        d1_tracking_descriptor(plan, other_profile, _start())
    with pytest.raises(ValueError, match="logical_run_id"):
        d1_tracking_descriptor(plan, plan.profile, _start("different-run"))
    with pytest.raises(ValueError, match="image_digest"):
        d1_tracking_descriptor(
            plan,
            plan.profile,
            replace(_start(), image_digest="sha256:" + "d" * 64),
        )


def test_d1_extractor_projects_only_evidence_qualified_scalars_at_successful_step(
    tmp_path: Path,
) -> None:
    plan = _plan()
    descriptor = d1_tracking_descriptor(plan, plan.profile, _start())
    client = _RecordingClient()
    sink = build_sink_for_run(
        descriptor,
        results_path=tmp_path / "journal.jsonl",
        client=client,
        metric_extractor=D1MetricExtractor(),
    )
    assert type(sink) is CompositeSink
    mirror = sink.best_effort[0]
    assert type(mirror) is MlflowSink
    event = _event()
    before = canonical_event_line(event)

    mirror.log(event)

    metrics = client._runs[mirror.run_id].data.metrics
    assert metrics["d1/quality/measured/nats_per_byte"] == 1.2345678901234567
    assert metrics["d1/quality/derived_from_measured/bits_per_byte"] == 1.780853665290424
    assert metrics["d1/quality/derived_from_measured/target_rms_l2_norm"] == 0.61
    assert metrics["d1/quality/measured/target_predicted_exact_match_count"] == 0.0
    expected_report_only = {
        f"d1/quality/{'measured' if name.endswith('_count') else 'derived_from_measured'}/{name}"
        for name in D1_REPORT_ONLY_METRICS
    }
    assert {key for key in metrics if key.rsplit("/", 1)[-1] in D1_REPORT_ONLY_METRICS} == (
        expected_report_only
    )
    assert metrics["d1/resource/measured/trainer_wall_seconds"] == 123.45678901234567
    assert metrics["d1/resource/model_estimate/cumulative_train_flops"] == 1.25e15
    assert metrics["d1/resource/rate_estimate/rate_estimated_cost_usd"] == 0.042
    assert metrics["d1/cursor/measured/eval_index"] == 3.0
    assert all(step == event.successful_step for _, _, step in client.metric_steps)
    assert not any("covariance" in key or "state_variance" in key for key in metrics)
    assert not any("gpu_eval_device_seconds" in key for key in metrics)
    assert not any("cumulative_eval_flops" in key for key in metrics)
    assert canonical_event_line(event) == before


def test_d1_extractor_refuses_cursor_mismatch_and_mlflow_counts_it_best_effort() -> None:
    event = _event(_resource("d1-pilot-both-seed-7", eval_index=4))
    payload = event.manifest["payload"]
    assert isinstance(payload, dict)
    payload["eval_index"] = 3
    malformed = replace(event, payload=payload)
    with pytest.raises(ValueError, match="eval_index"):
        D1MetricExtractor().extract(malformed)

    client = FakeMlflowClient()
    mirror = MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="d1",
        client=client,
        metric_extractor=D1MetricExtractor(),
    )
    mirror.log(malformed)
    assert mirror.failed_writes == 1
