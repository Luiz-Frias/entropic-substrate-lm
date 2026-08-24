"""Per-eval cumulative resource work evidence, distinct from quality exposure."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from entropic_substrate.evidence.resource_provenance import EvidenceGap, Money, execution_segment_id
from entropic_substrate.experiments.d1_primitives import EvidenceClass, canonical_digest

__all__ = [
    "CostReading",
    "CountOrGap",
    "IdentityOrGap",
    "ReadingMethod",
    "ResourceWorkSnapshot",
    "ScalarReading",
    "validate_resource_work_series",
]


class ReadingMethod(StrEnum):
    GPU_DEVICE_COUNTER = "gpu-device-counter"
    FLOP_PROFILER = "flop-profiler"
    FLOP_MODEL = "flop-model"
    ENERGY_INTEGRATED = "energy-integrated"
    ENERGY_COUNTER = "energy-counter"
    RATE_TIMES_SCOPE = "rate-times-scope"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        msg = f"{label} must be a UTC timestamp"
        raise ValueError(msg)


def _nonempty(value: str, *, label: str) -> None:
    if type(value) is not str or not value.strip():
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)


def _count(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        msg = f"{label} must be a nonnegative integer"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScalarReading:
    value: float
    evidence_class: EvidenceClass
    method: ReadingMethod

    def __post_init__(self) -> None:
        if type(self.value) is not float or not math.isfinite(self.value) or self.value < 0.0:
            msg = "scalar reading value must be a finite nonnegative float"
            raise ValueError(msg)
        if type(self.evidence_class) is not EvidenceClass:
            msg = "scalar reading evidence_class must be EvidenceClass"
            raise TypeError(msg)
        if type(self.method) is not ReadingMethod:
            msg = "scalar reading method must be ReadingMethod"
            raise TypeError(msg)


@dataclass(frozen=True, slots=True)
class CostReading:
    amount: Money
    evidence_class: EvidenceClass
    method: ReadingMethod

    def __post_init__(self) -> None:
        if type(self.amount) is not Money:
            msg = "cost reading amount must be Money"
            raise TypeError(msg)
        if self.evidence_class is not EvidenceClass.RATE_ESTIMATE:
            msg = "cost reading evidence_class must be EvidenceClass.RATE_ESTIMATE"
            raise ValueError(msg)
        if self.method is not ReadingMethod.RATE_TIMES_SCOPE:
            msg = "cost reading method must be ReadingMethod.RATE_TIMES_SCOPE"
            raise ValueError(msg)


ScalarOrGap = ScalarReading | EvidenceGap
CostOrGap = CostReading | EvidenceGap
IdentityOrGap = str | EvidenceGap
CountOrGap = int | EvidenceGap


def _scalar_or_gap(
    value: ScalarOrGap,
    *,
    field: str,
    allowed: tuple[tuple[EvidenceClass, ReadingMethod], ...],
) -> None:
    if type(value) is EvidenceGap:
        if value.field != field:
            msg = f"evidence gap must name exactly {field!r}"
            raise ValueError(msg)
        return
    if type(value) is not ScalarReading:
        msg = f"{field} must be ScalarReading or EvidenceGap"
        raise TypeError(msg)
    if (value.evidence_class, value.method) not in allowed:
        msg = f"{field} has an invalid evidence_class/method pairing"
        raise ValueError(msg)


def _cost_or_gap(value: CostOrGap, *, field: str) -> None:
    if type(value) is EvidenceGap:
        if value.field != field:
            msg = f"evidence gap must name exactly {field!r}"
            raise ValueError(msg)
        return
    if type(value) is not CostReading:
        msg = f"{field} must be CostReading or EvidenceGap"
        raise TypeError(msg)


def _identity_or_gap(value: IdentityOrGap, *, field: str) -> None:
    if type(value) is EvidenceGap:
        if value.field != field:
            raise ValueError(f"evidence gap must name exactly {field!r}")
        return
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 identity or EvidenceGap")


def _count_or_gap(value: CountOrGap, *, field: str) -> None:
    if type(value) is EvidenceGap:
        if value.field != field:
            raise ValueError(f"evidence gap must name exactly {field!r}")
        return
    if type(value) is not int:
        raise TypeError(f"{field} must be a nonnegative integer or EvidenceGap")
    _count(value, label=field)


@dataclass(frozen=True, slots=True)
class ResourceWorkSnapshot:
    """Immutable cumulative work evidence at one evaluation cursor."""

    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    training_step: int
    eval_index: int
    observed_at: datetime
    trainer_wall_seconds: float
    committed_quality_source_bytes: int
    committed_quality_model_tokens: int
    online_train_token_positions: int
    target_train_token_positions: int
    baseline_calibration_positions: int
    eval_positions: int
    total_model_token_positions: int
    gpu_train_device_seconds: ScalarOrGap
    gpu_eval_device_seconds: ScalarOrGap
    peak_gpu_allocated_bytes: ScalarOrGap
    peak_gpu_reserved_bytes: ScalarOrGap
    cumulative_train_flops: ScalarOrGap
    cumulative_eval_flops: ScalarOrGap
    flop_model_identity: IdentityOrGap
    energy_joules: ScalarOrGap
    energy_sample_count: CountOrGap
    rate_estimated_cost: CostOrGap
    schema_version: str = "resource-work/v2"

    def __post_init__(self) -> None:
        _nonempty(self.logical_run_id, label="logical_run_id")
        if self.schema_version != "resource-work/v2":
            msg = "unsupported resource work schema_version"
            raise ValueError(msg)
        _count(self.execution_segment_index, label="execution_segment_index")
        if self.execution_segment_id != execution_segment_id(
            self.logical_run_id, self.execution_segment_index
        ):
            msg = "execution_segment_id must match logical_run_id and execution_segment_index"
            raise ValueError(msg)
        _count(self.training_step, label="training_step")
        _count(self.eval_index, label="eval_index")
        _utc(self.observed_at, label="observed_at")
        if (
            type(self.trainer_wall_seconds) is not float
            or not math.isfinite(self.trainer_wall_seconds)
            or self.trainer_wall_seconds < 0.0
        ):
            msg = "trainer_wall_seconds must be a finite nonnegative float"
            raise ValueError(msg)
        for label, value in (
            ("committed_quality_source_bytes", self.committed_quality_source_bytes),
            ("committed_quality_model_tokens", self.committed_quality_model_tokens),
            ("online_train_token_positions", self.online_train_token_positions),
            ("target_train_token_positions", self.target_train_token_positions),
            ("baseline_calibration_positions", self.baseline_calibration_positions),
            ("eval_positions", self.eval_positions),
            ("total_model_token_positions", self.total_model_token_positions),
        ):
            _count(value, label=label)
        if self.total_model_token_positions != sum(
            (
                self.online_train_token_positions,
                self.target_train_token_positions,
                self.baseline_calibration_positions,
                self.eval_positions,
            )
        ):
            msg = "total_model_token_positions must equal the exact work-breakdown sum"
            raise ValueError(msg)
        _scalar_or_gap(
            self.gpu_train_device_seconds,
            field="gpu_train_device_seconds",
            allowed=((EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER),),
        )
        _scalar_or_gap(
            self.gpu_eval_device_seconds,
            field="gpu_eval_device_seconds",
            allowed=((EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER),),
        )
        for field in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            _scalar_or_gap(
                getattr(self, field),
                field=field,
                allowed=((EvidenceClass.MEASURED, ReadingMethod.GPU_DEVICE_COUNTER),),
            )
        flop_methods = (
            (EvidenceClass.MEASURED, ReadingMethod.FLOP_PROFILER),
            (EvidenceClass.MODEL_ESTIMATE, ReadingMethod.FLOP_MODEL),
        )
        _scalar_or_gap(
            self.cumulative_train_flops, field="cumulative_train_flops", allowed=flop_methods
        )
        _scalar_or_gap(
            self.cumulative_eval_flops, field="cumulative_eval_flops", allowed=flop_methods
        )
        _identity_or_gap(self.flop_model_identity, field="flop_model_identity")
        model_estimated = any(
            type(value) is ScalarReading and value.evidence_class is EvidenceClass.MODEL_ESTIMATE
            for value in (self.cumulative_train_flops, self.cumulative_eval_flops)
        )
        if model_estimated and type(self.flop_model_identity) is EvidenceGap:
            raise ValueError("flop_model_identity is required for MODEL_ESTIMATE FLOPs")
        _scalar_or_gap(
            self.energy_joules,
            field="energy_joules",
            allowed=(
                (EvidenceClass.DERIVED_FROM_MEASURED, ReadingMethod.ENERGY_INTEGRATED),
                (EvidenceClass.MEASURED, ReadingMethod.ENERGY_COUNTER),
            ),
        )
        _count_or_gap(self.energy_sample_count, field="energy_sample_count")
        if type(self.energy_joules) is ScalarReading:
            if self.energy_joules.method is ReadingMethod.ENERGY_INTEGRATED:
                if type(self.energy_sample_count) is not int or self.energy_sample_count < 2:
                    raise ValueError(
                        "energy_sample_count must be at least two for integrated joules"
                    )
            elif type(self.energy_sample_count) is not EvidenceGap:
                raise ValueError("energy_sample_count applies only to integrated energy")
        _cost_or_gap(self.rate_estimated_cost, field="rate_estimated_cost")

    @property
    def cursor_identity(self) -> str:
        return canonical_digest(self.manifest)

    @property
    def manifest(self) -> dict[str, object]:
        def reading(value: ScalarOrGap | CostOrGap) -> dict[str, object]:
            if type(value) is EvidenceGap:
                return {"field": value.field, "kind": "gap", "reason": value.reason}
            if type(value) is ScalarReading:
                return {
                    "evidence_class": value.evidence_class.value,
                    "kind": "scalar",
                    "method": value.method.value,
                    "value": value.value,
                }
            if type(value) is CostReading:
                return {
                    "amount": value.amount.manifest,
                    "evidence_class": value.evidence_class.value,
                    "kind": "cost",
                    "method": value.method.value,
                }
            raise AssertionError("validated resource reading has an unknown type")

        def identity(value: IdentityOrGap) -> dict[str, object]:
            if type(value) is EvidenceGap:
                return {"field": value.field, "kind": "gap", "reason": value.reason}
            return {"kind": "identity", "sha256": value}

        def count(value: CountOrGap) -> dict[str, object]:
            if type(value) is EvidenceGap:
                return {"field": value.field, "kind": "gap", "reason": value.reason}
            return {"kind": "count", "value": value}

        return {
            "baseline_calibration_positions": self.baseline_calibration_positions,
            "committed_quality_model_tokens": self.committed_quality_model_tokens,
            "committed_quality_source_bytes": self.committed_quality_source_bytes,
            "cumulative_eval_flops": reading(self.cumulative_eval_flops),
            "cumulative_train_flops": reading(self.cumulative_train_flops),
            "energy_joules": reading(self.energy_joules),
            "energy_sample_count": count(self.energy_sample_count),
            "eval_index": self.eval_index,
            "eval_positions": self.eval_positions,
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "gpu_eval_device_seconds": reading(self.gpu_eval_device_seconds),
            "gpu_train_device_seconds": reading(self.gpu_train_device_seconds),
            "flop_model_identity": identity(self.flop_model_identity),
            "logical_run_id": self.logical_run_id,
            "online_train_token_positions": self.online_train_token_positions,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
            "rate_estimated_cost": reading(self.rate_estimated_cost),
            "peak_gpu_allocated_bytes": reading(self.peak_gpu_allocated_bytes),
            "peak_gpu_reserved_bytes": reading(self.peak_gpu_reserved_bytes),
            "required_counter_evidence": EvidenceClass.MEASURED.value,
            "schema_version": self.schema_version,
            "target_train_token_positions": self.target_train_token_positions,
            "total_model_token_positions": self.total_model_token_positions,
            "trainer_wall_seconds": self.trainer_wall_seconds,
            "training_step": self.training_step,
        }


def validate_resource_work_series(records: tuple[ResourceWorkSnapshot, ...]) -> None:
    """Refuse cursor/order/counter regressions while preserving declared meter gaps."""
    if type(records) is not tuple or not records:
        msg = "resource work series must be a non-empty tuple"
        raise ValueError(msg)
    first: ResourceWorkSnapshot | None = None
    previous: ResourceWorkSnapshot | None = None
    last_optional: dict[str, ScalarReading | CostReading] = {}
    required = (
        "trainer_wall_seconds",
        "committed_quality_source_bytes",
        "committed_quality_model_tokens",
        "online_train_token_positions",
        "target_train_token_positions",
        "baseline_calibration_positions",
        "eval_positions",
        "total_model_token_positions",
    )
    optional = (
        "gpu_train_device_seconds",
        "gpu_eval_device_seconds",
        "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes",
        "cumulative_train_flops",
        "cumulative_eval_flops",
        "energy_joules",
        "rate_estimated_cost",
    )
    last_flop_identity: str | None = None
    last_energy_sample_count: int | None = None
    for record in records:
        if type(record) is not ResourceWorkSnapshot:
            msg = "resource work series must contain ResourceWorkSnapshot values"
            raise TypeError(msg)
        if first is None:
            first = record
            if record.eval_index != 0 or record.execution_segment_index != 0:
                msg = "complete resource work series must start at eval and segment zero"
                raise ValueError(msg)
        elif record.logical_run_id != first.logical_run_id:
            msg = "resource work series logical_run_id changed"
            raise ValueError(msg)
        if previous is not None:
            if record.eval_index <= previous.eval_index:
                msg = "eval_index must be strictly increasing"
                raise ValueError(msg)
            if record.training_step < previous.training_step:
                msg = "training_step must be nondecreasing"
                raise ValueError(msg)
            if record.observed_at <= previous.observed_at:
                msg = "observed_at must be strictly increasing"
                raise ValueError(msg)
            if record.execution_segment_index < previous.execution_segment_index:
                msg = "execution segment index must be nondecreasing"
                raise ValueError(msg)
            if record.execution_segment_index > previous.execution_segment_index + 1:
                msg = "execution segment indexes must be consecutive when changing"
                raise ValueError(msg)
            for field in required:
                if getattr(record, field) < getattr(previous, field):
                    msg = f"required cumulative counter {field} is not monotonic"
                    raise ValueError(msg)
        for field in optional:
            value = getattr(record, field)
            if type(value) is EvidenceGap:
                continue
            prior = last_optional.get(field)
            if prior is not None:
                if (
                    value.evidence_class is not prior.evidence_class
                    or value.method is not prior.method
                ):
                    msg = f"{field} evidence_class/method changed across the series"
                    raise ValueError(msg)
                if type(value) is CostReading and type(prior) is CostReading:
                    if value.amount.currency != prior.amount.currency:
                        msg = f"{field} currency changed across the series"
                        raise ValueError(msg)
                    if value.amount.amount < prior.amount.amount:
                        msg = f"{field} is not monotonic across gaps"
                        raise ValueError(msg)
                elif type(value) is ScalarReading and type(prior) is ScalarReading:
                    if value.value < prior.value:
                        msg = f"{field} is not monotonic across gaps"
                        raise ValueError(msg)
                else:
                    msg = f"{field} reading type changed across the series"
                    raise ValueError(msg)
            last_optional[field] = value
        identity = record.flop_model_identity
        if type(identity) is str:
            if last_flop_identity is not None and identity != last_flop_identity:
                raise ValueError("flop_model_identity changed across the series")
            last_flop_identity = identity
        sample_count = record.energy_sample_count
        if type(sample_count) is int:
            if last_energy_sample_count is not None and sample_count < last_energy_sample_count:
                raise ValueError("energy_sample_count is not monotonic across gaps")
            last_energy_sample_count = sample_count
        previous = record
