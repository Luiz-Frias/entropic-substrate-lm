"""D1 adapters into the non-authoritative operational tracking plane."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Final, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    EvidenceGap,
    RunStartProvenance,
)
from entropic_substrate.kernel.identity.d1_primitives import EvidenceClass
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap, parse_d1_cosine
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    ExtractedMetrics,
    TrackingRunDescriptor,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1_REPORT_ONLY_METRICS
from entropic_substrate.ladder.capture.resource.resource_work import CostReading, ScalarReading
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_from_mapping,
)
from entropic_substrate.ladder.register.d1_config import D1LaunchProfile, D1LaunchScale
from entropic_substrate.ladder.register.d1_plan import D1RunPlan
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
)

__all__ = ["D1MetricExtractor", "d1_tracking_descriptor"]

_QUALITY_EVIDENCE: Final[Mapping[str, EvidenceClass]] = {
    "language_loss": EvidenceClass.MEASURED,
    "nats_per_byte": EvidenceClass.MEASURED,
    "future_nll": EvidenceClass.MEASURED,
    "zero_displacement_nll": EvidenceClass.MEASURED,
    "mean_displacement_nll": EvidenceClass.MEASURED,
    "batch_count": EvidenceClass.MEASURED,
    "online_positions": EvidenceClass.MEASURED,
    "residual_elements": EvidenceClass.MEASURED,
    "bits_per_byte": EvidenceClass.DERIVED_FROM_MEASURED,
    "perplexity": EvidenceClass.DERIVED_FROM_MEASURED,
    "zero_baseline_improvement": EvidenceClass.DERIVED_FROM_MEASURED,
    "mean_baseline_improvement": EvidenceClass.DERIVED_FROM_MEASURED,
    "residual_mean": EvidenceClass.DERIVED_FROM_MEASURED,
    "residual_rms": EvidenceClass.DERIVED_FROM_MEASURED,
    "residual_variance": EvidenceClass.DERIVED_FROM_MEASURED,
    "coverage_50": EvidenceClass.DERIVED_FROM_MEASURED,
    "coverage_80": EvidenceClass.DERIVED_FROM_MEASURED,
    "coverage_95": EvidenceClass.DERIVED_FROM_MEASURED,
    "effective_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "numerical_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "mean_pairwise_cosine_distance": EvidenceClass.DERIVED_FROM_MEASURED,
    "raw_variance_penalty": EvidenceClass.DERIVED_FROM_MEASURED,
    "weighted_variance_penalty": EvidenceClass.DERIVED_FROM_MEASURED,
    "raw_covariance_penalty": EvidenceClass.DERIVED_FROM_MEASURED,
    "weighted_covariance_penalty": EvidenceClass.DERIVED_FROM_MEASURED,
    "state_rms_l2_norm": EvidenceClass.DERIVED_FROM_MEASURED,
    "target_effective_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "target_numerical_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "target_rms_l2_norm": EvidenceClass.DERIVED_FROM_MEASURED,
    "predicted_effective_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "predicted_numerical_rank": EvidenceClass.DERIVED_FROM_MEASURED,
    "predicted_rms_l2_norm": EvidenceClass.DERIVED_FROM_MEASURED,
    "residual_mean_l2_norm": EvidenceClass.DERIVED_FROM_MEASURED,
    "target_predicted_exact_match_count": EvidenceClass.MEASURED,
    "target_predicted_exact_match_fraction": EvidenceClass.DERIVED_FROM_MEASURED,
}
_EXPECTED_REPORT_ONLY_EVIDENCE = dict.fromkeys(
    D1_REPORT_ONLY_METRICS, EvidenceClass.DERIVED_FROM_MEASURED
)
_EXPECTED_REPORT_ONLY_EVIDENCE["target_predicted_exact_match_count"] = EvidenceClass.MEASURED
if {name: _QUALITY_EVIDENCE.get(name) for name in D1_REPORT_ONLY_METRICS} != (
    _EXPECTED_REPORT_ONLY_EVIDENCE
):  # pragma: no cover - import invariant
    raise RuntimeError("D1 tracking report-only evidence classes differ from the contract")
_REQUIRED_RESOURCE_COUNTERS: Final = (
    "trainer_wall_seconds",
    "committed_quality_source_bytes",
    "committed_quality_model_tokens",
    "online_train_token_positions",
    "target_train_token_positions",
    "baseline_calibration_positions",
    "eval_positions",
    "total_model_token_positions",
)
_OPTIONAL_RESOURCE_READINGS: Final = (
    "gpu_train_device_seconds",
    "gpu_eval_device_seconds",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "cumulative_train_flops",
    "cumulative_eval_flops",
    "energy_sample_count",
    "energy_joules",
    "rate_estimated_cost",
)


def _require_bound_provenance(plan: D1RunPlan, start: RunStartProvenance) -> None:
    execution = plan.run.execution
    if plan.run.data is None:
        raise ValueError("D1 tracking requires bound dataset identity")
    if execution.code is None:
        raise ValueError("D1 tracking requires bound code identity")
    if execution.container is None:
        raise ValueError("D1 tracking requires bound container identity")
    if start.logical_run_id != plan.logical_run_id:
        raise ValueError("D1 tracking logical_run_id differs from the run plan")
    if start.image_digest != execution.container.image_digest:
        raise ValueError("D1 tracking start image_digest differs from the run plan")


def d1_tracking_descriptor(
    plan: D1RunPlan,
    profile: D1LaunchProfile,
    start: RunStartProvenance,
) -> TrackingRunDescriptor:
    """Bind the selected D1 plan, profile, and physical start into MLflow metadata."""
    if type(profile) is not D1LaunchProfile:
        raise TypeError("D1 tracking profile must be a D1LaunchProfile")
    if type(start) is not RunStartProvenance:
        raise TypeError("D1 tracking start must be RunStartProvenance")
    if plan.profile != profile:
        raise ValueError("D1 tracking profile differs from the run plan")
    _require_bound_provenance(plan, start)
    run = plan.run
    execution = run.execution
    data = cast(DataIdentity, run.data)
    code = cast(CodeIdentity, execution.code)
    container = cast(ContainerIdentity, execution.container)
    config_key = run.aggregation_digest[:12]
    activation = start.resource_activation
    power = activation.power_sampling
    identities = {
        "d1/construction_digest": profile.draft.construction_digest,
        "d1/family_digest": profile.family.family_digest,
        "d1/aggregation_digest": run.aggregation_digest,
        "d1/execution_digest": run.execution_digest,
        "d1/plan_digest": plan.plan_digest,
        "d1/dataset_digest": data.dataset_digest,
        "d1/code_commit": code.commit,
        "d1/container_image_digest": container.image_digest,
        "resource/start_identity_digest": start.identity_digest,
        "resource/activation_identity_digest": activation.identity_digest,
    }
    params = {
        "exp": profile.name,
        "seed": plan.seed,
        "config_hash": config_key,
        "d1/profile_name": profile.name,
        "d1/arm": profile.arm.value,
        "d1/run_class": run.run_class.value,
        "d1/logical_run_id": plan.logical_run_id,
        "d1/pairing_id": execution.pairing_id,
        "d1/dataset_name": data.dataset_name,
        "d1/dataset_split_digest": data.split_digest,
        "d1/dataset_sampler_digest": data.sampler_construction_digest,
        "d1/repository_digest": code.repository_digest,
        "resource/execution_segment_id": start.execution_segment_id,
        "resource/accelerator_model": start.accelerator_model,
        "resource/accelerator_count": start.accelerator_count,
        "resource/provider": start.provider,
        "resource/provider_sku": start.provider_sku,
        "resource/region": start.region,
        "resource/market": start.market,
        "resource/purchase_option": start.purchase_option.value,
        "resource/quoted_rate_amount": start.quoted_rate.amount.canonical_amount,
        "resource/quoted_rate_currency": start.quoted_rate.amount.currency,
        "resource/activation_mode": activation.mode.value,
        "resource/activation_factory_digest": activation.factory_registration_digest,
        "resource/meter_provider": activation.meter_provider,
        "resource/meter_device_scope": activation.device_scope,
        "resource/flop_model_identity": activation.flop_model_identity or "unavailable",
        "resource/power_cadence_seconds": (
            power.cadence_seconds if power is not None else "unavailable"
        ),
        "resource/power_max_gap_seconds": (
            power.max_gap_seconds if power is not None else "unavailable"
        ),
        "resource/power_integration_method": (
            power.integration_method.value if power is not None else "unavailable"
        ),
        "resource/power_provider": power.provider if power is not None else "unavailable",
        "resource/power_device_scope": (power.device_scope if power is not None else "unavailable"),
        **identities,
    }
    tags = {
        "exp": profile.name,
        "config_hash": config_key,
        "d1/arm": profile.arm.value,
        "d1/run_class": run.run_class.value,
        "d1/logical_run_id": plan.logical_run_id,
        **identities,
    }
    return TrackingRunDescriptor(
        run_name=plan.logical_run_id,
        tiny=profile.scale in (D1LaunchScale.TINY, D1LaunchScale.GPU_TINY),
        params=params,
        tags=tags,
        config_hash=config_key,
    )


def _quality_metrics(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("D1 EVAL quality must be a JSON object")
    metrics: dict[str, float] = {}
    for field, evidence in _QUALITY_EVIDENCE.items():
        if field not in value:
            continue
        item = value[field]
        if field == "mean_pairwise_cosine_distance":
            cosine = parse_d1_cosine(item)
            if type(cosine) is D1CosineGap:
                if cosine.total_positions != value.get("online_positions"):
                    raise ValueError("D1 cosine gap positions differ from quality population")
                prefix = f"d1/diagnostic_gap/{cosine.reason}"
                metrics[f"{prefix}/affected_positions"] = float(cosine.affected_positions)
                metrics[f"{prefix}/total_positions"] = float(cosine.total_positions)
                continue
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise ValueError(f"D1 quality scalar {field!r} must be finite")
        metrics[f"d1/quality/{evidence.value}/{field}"] = float(item)
    return metrics


class D1MetricExtractor:
    """Project one canonical D1 EVAL event without manufacturing missing evidence."""

    def extract(self, event: D1JournalEvent) -> ExtractedMetrics | None:
        if type(event) is not D1JournalEvent:
            raise TypeError("D1 metric extraction requires D1JournalEvent")
        if event.event_kind is not JournalEventKind.EVAL:
            return None
        payload = cast(dict[str, object], event.manifest["payload"])
        if set(payload) != {"eval_index", "quality", "resource_work"}:
            raise ValueError("D1 EVAL payload keys must be eval_index, quality, resource_work")
        eval_index = payload["eval_index"]
        if type(eval_index) is not int or eval_index < 0:
            raise ValueError("D1 EVAL eval_index must be a nonnegative integer")
        resource = resource_snapshot_from_mapping(payload["resource_work"])
        if resource.eval_index != eval_index:
            raise ValueError("D1 EVAL eval_index differs from resource_work")
        if resource.training_step != event.successful_step:
            raise ValueError("D1 EVAL successful_step differs from resource_work training_step")
        if resource.logical_run_id != event.logical_run_id:
            raise ValueError("D1 EVAL logical_run_id differs from resource_work")
        metrics = _quality_metrics(payload["quality"])
        metrics["d1/cursor/measured/eval_index"] = float(eval_index)
        for field in _REQUIRED_RESOURCE_COUNTERS:
            metrics[f"d1/resource/measured/{field}"] = float(getattr(resource, field))
        for field in _OPTIONAL_RESOURCE_READINGS:
            reading = getattr(resource, field)
            if type(reading) is EvidenceGap:
                continue
            if type(reading) is int:
                metrics[f"d1/resource/{EvidenceClass.MEASURED.value}/{field}"] = float(reading)
                continue
            if type(reading) is ScalarReading:
                metrics[f"d1/resource/{reading.evidence_class.value}/{field}"] = reading.value
            elif type(reading) is CostReading:
                currency = reading.amount.currency.lower()
                metrics[f"d1/resource/{reading.evidence_class.value}/{field}_{currency}"] = float(
                    reading.amount.amount
                )
            else:  # pragma: no cover - the strict resource codec refuses this
                raise AssertionError("validated resource_work contains an unknown reading")
        return ExtractedMetrics(step=event.successful_step, metrics=metrics)
