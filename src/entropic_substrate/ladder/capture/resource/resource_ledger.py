"""Mutable per-run accumulator for committed resource-work snapshot evidence."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast, runtime_checkable

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    EvidenceGap,
    RunStartProvenance,
    execution_segment_id,
)
from entropic_substrate.ladder.capture.resource.resource_work import (
    CostReading,
    ResourceWorkSnapshot,
    ScalarOrGap,
    ScalarReading,
)

__all__ = [
    "ResourceLedger",
    "ResourceLedgerState",
    "ResourceMeterSnapshot",
    "ResourceMeterStrategy",
    "UnavailableMeterStrategy",
]

_OPTIONAL_FIELDS = (
    "gpu_train_device_seconds",
    "gpu_eval_device_seconds",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "cumulative_train_flops",
    "cumulative_eval_flops",
    "energy_joules",
    "rate_estimated_cost",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _count(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        msg = f"{label} must be a nonnegative integer"
        raise ValueError(msg)


def _monotonic_ns(value: int, *, label: str) -> None:
    if type(value) is not int or value < 0:
        msg = f"{label} must be a nonnegative monotonic-clock nanosecond integer"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ResourceMeterSnapshot:
    """One typed meter read; every unavailable channel is an exact field-bound gap."""

    gpu_train_device_seconds: ScalarOrGap
    gpu_eval_device_seconds: ScalarOrGap
    peak_gpu_allocated_bytes: ScalarOrGap
    peak_gpu_reserved_bytes: ScalarOrGap
    cumulative_train_flops: ScalarOrGap
    cumulative_eval_flops: ScalarOrGap
    flop_model_identity: str | EvidenceGap
    energy_joules: ScalarOrGap
    energy_sample_count: int | EvidenceGap
    rate_estimated_cost: CostReading | EvidenceGap


@runtime_checkable
class ResourceMeterStrategy(Protocol):
    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot: ...


class UnavailableMeterStrategy:
    """Explicit default: no optional meter is inferred from wall time or rate."""

    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot:
        if (
            type(trainer_wall_seconds) is not float
            or not math.isfinite(trainer_wall_seconds)
            or trainer_wall_seconds < 0.0
        ):
            raise ValueError("trainer_wall_seconds must be finite and nonnegative")

        def gap(field: str) -> EvidenceGap:
            return EvidenceGap(field=field, reason="meter strategy unavailable")

        return ResourceMeterSnapshot(
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


@dataclass(frozen=True, slots=True)
class ResourceLedgerState:
    logical_run_id: str
    execution_segment_index: int
    next_eval_index: int
    last_training_step: int | None
    trainer_wall_seconds: float
    committed_quality_source_bytes: int
    committed_quality_model_tokens: int
    online_train_token_positions: int
    target_train_token_positions: int
    baseline_calibration_positions: int
    eval_positions: int
    last_optional_readings: tuple[tuple[str, ScalarReading | CostReading], ...]
    last_flop_model_identity: str | None
    last_energy_sample_count: int | None
    schema_version: str = "resource-ledger/v2"

    def __post_init__(self) -> None:
        if type(self.logical_run_id) is not str or not self.logical_run_id.strip():
            raise ValueError("resource ledger state logical_run_id must be nonempty")
        if self.schema_version != "resource-ledger/v2":
            raise ValueError("unsupported resource ledger state schema")
        for label, value in (
            ("execution_segment_index", self.execution_segment_index),
            ("next_eval_index", self.next_eval_index),
            ("committed_quality_source_bytes", self.committed_quality_source_bytes),
            ("committed_quality_model_tokens", self.committed_quality_model_tokens),
            ("online_train_token_positions", self.online_train_token_positions),
            ("target_train_token_positions", self.target_train_token_positions),
            ("baseline_calibration_positions", self.baseline_calibration_positions),
            ("eval_positions", self.eval_positions),
        ):
            _count(value, label=label)
        if self.last_training_step is not None:
            _count(self.last_training_step, label="last_training_step")
        if (
            type(self.trainer_wall_seconds) is not float
            or not math.isfinite(self.trainer_wall_seconds)
            or self.trainer_wall_seconds < 0.0
        ):
            raise ValueError("resource ledger state trainer_wall_seconds must be finite")
        if type(self.last_optional_readings) is not tuple:
            raise TypeError("last_optional_readings must be an immutable tuple")
        fields = [field for field, _ in self.last_optional_readings]
        if len(fields) != len(set(fields)) or any(
            field not in _OPTIONAL_FIELDS for field in fields
        ):
            raise ValueError("last_optional_readings fields must be unique registered meters")
        if any(
            type(value) not in (ScalarReading, CostReading)
            for _, value in self.last_optional_readings
        ):
            raise TypeError("last_optional_readings values must be typed readings")
        if self.last_flop_model_identity is not None and (
            type(self.last_flop_model_identity) is not str
            or _SHA256.fullmatch(self.last_flop_model_identity) is None
        ):
            raise ValueError("last_flop_model_identity must be a lowercase SHA-256 or None")
        if self.last_energy_sample_count is not None:
            _count(self.last_energy_sample_count, label="last_energy_sample_count")

    @property
    def manifest(self) -> dict[str, object]:
        readings: dict[str, object] = {}
        for field, value in self.last_optional_readings:
            if type(value) is ScalarReading:
                readings[field] = {
                    "evidence_class": value.evidence_class.value,
                    "kind": "scalar",
                    "method": value.method.value,
                    "value": value.value,
                }
            elif type(value) is CostReading:
                readings[field] = {
                    "amount": value.amount.manifest,
                    "evidence_class": value.evidence_class.value,
                    "kind": "cost",
                    "method": value.method.value,
                }
            else:  # pragma: no cover - rejected by __post_init__
                raise AssertionError("resource ledger state contains an unknown reading")
        total = (
            self.online_train_token_positions
            + self.target_train_token_positions
            + self.baseline_calibration_positions
            + self.eval_positions
        )
        return {
            "baseline_calibration_positions": self.baseline_calibration_positions,
            "committed_quality_model_tokens": self.committed_quality_model_tokens,
            "committed_quality_source_bytes": self.committed_quality_source_bytes,
            "eval_positions": self.eval_positions,
            "execution_segment_id": execution_segment_id(
                self.logical_run_id, self.execution_segment_index
            ),
            "execution_segment_index": self.execution_segment_index,
            "last_optional_readings": readings,
            "last_flop_model_identity": self.last_flop_model_identity,
            "last_energy_sample_count": self.last_energy_sample_count,
            "last_training_step": self.last_training_step,
            "logical_run_id": self.logical_run_id,
            "next_eval_index": self.next_eval_index,
            "online_train_token_positions": self.online_train_token_positions,
            "schema_version": self.schema_version,
            "target_train_token_positions": self.target_train_token_positions,
            "total_model_token_positions": total,
            "trainer_wall_seconds": self.trainer_wall_seconds,
        }


class ResourceLedger:
    """Accumulate work continuously while advancing quality only on a successful commit."""

    def __init__(
        self,
        start: RunStartProvenance,
        *,
        monotonic_origin_ns: int,
        base_trainer_wall_seconds: float = 0.0,
        meter_strategy: ResourceMeterStrategy | None = None,
    ) -> None:
        if type(start) is not RunStartProvenance:
            msg = "resource ledger requires RunStartProvenance"
            raise TypeError(msg)
        _monotonic_ns(monotonic_origin_ns, label="monotonic_origin_ns")
        if (
            type(base_trainer_wall_seconds) is not float
            or not math.isfinite(base_trainer_wall_seconds)
            or base_trainer_wall_seconds < 0.0
        ):
            msg = "base_trainer_wall_seconds must be a finite nonnegative float"
            raise ValueError(msg)
        if meter_strategy is not None and not isinstance(
            cast(object, meter_strategy), ResourceMeterStrategy
        ):
            msg = "meter_strategy must implement ResourceMeterStrategy"
            raise TypeError(msg)
        self.start = start
        self._origin_ns = monotonic_origin_ns
        self._last_monotonic_ns = monotonic_origin_ns
        self._base_wall_seconds = base_trainer_wall_seconds
        self._meter = meter_strategy or UnavailableMeterStrategy()
        self._next_eval_index = 0
        self._last_training_step: int | None = None
        self._quality_bytes = self._quality_tokens = 0
        self._online = self._target = self._baseline = self._eval = 0
        self._last_optional: dict[str, ScalarReading | CostReading] = {}
        self._last_flop_identity: str | None = None
        self._last_energy_count: int | None = None

    @classmethod
    def restore(
        cls,
        start: RunStartProvenance,
        state: ResourceLedgerState,
        *,
        monotonic_origin_ns: int,
        meter_strategy: ResourceMeterStrategy | None = None,
    ) -> ResourceLedger:
        if type(state) is not ResourceLedgerState:
            msg = "resource ledger restore requires ResourceLedgerState"
            raise TypeError(msg)
        if start.logical_run_id != state.logical_run_id:
            msg = "restore start logical_run_id does not match ledger state"
            raise ValueError(msg)
        if start.execution_segment_index != state.execution_segment_index + 1:
            msg = "restore start must be the exact next execution segment"
            raise ValueError(msg)
        ledger = cls(
            start,
            monotonic_origin_ns=monotonic_origin_ns,
            base_trainer_wall_seconds=state.trainer_wall_seconds,
            meter_strategy=meter_strategy,
        )
        ledger._next_eval_index = state.next_eval_index
        ledger._last_training_step = state.last_training_step
        ledger._quality_bytes = state.committed_quality_source_bytes
        ledger._quality_tokens = state.committed_quality_model_tokens
        ledger._online = state.online_train_token_positions
        ledger._target = state.target_train_token_positions
        ledger._baseline = state.baseline_calibration_positions
        ledger._eval = state.eval_positions
        ledger._last_optional = dict(state.last_optional_readings)
        ledger._last_flop_identity = state.last_flop_model_identity
        ledger._last_energy_count = state.last_energy_sample_count
        return ledger

    def record_baseline(self, *, target_positions: int) -> None:
        _count(target_positions, label="target_positions")
        self._baseline += target_positions

    def record_train_attempt(
        self,
        *,
        online_positions: int,
        target_positions: int,
        quality_source_byte_exposures: int,
        committed: bool,
    ) -> None:
        _count(online_positions, label="online_positions")
        _count(target_positions, label="target_positions")
        _count(quality_source_byte_exposures, label="quality_source_byte_exposures")
        if type(committed) is not bool:
            msg = "committed must be bool"
            raise TypeError(msg)
        self._online += online_positions
        self._target += target_positions
        if committed:
            self._quality_bytes += quality_source_byte_exposures
            self._quality_tokens += online_positions

    def record_discarded_work(
        self,
        *,
        online_train_positions: int,
        target_train_positions: int,
        eval_positions: int,
    ) -> None:
        """Carry quarantined post-checkpoint work without advancing quality evidence."""
        _count(online_train_positions, label="online_train_positions")
        _count(target_train_positions, label="target_train_positions")
        _count(eval_positions, label="eval_positions")
        self._online += online_train_positions
        self._target += target_train_positions
        self._eval += eval_positions

    def _wall_seconds(self, monotonic_ns: int) -> float:
        _monotonic_ns(monotonic_ns, label="monotonic_ns")
        if monotonic_ns < self._last_monotonic_ns:
            msg = "monotonic_ns must not move backward"
            raise ValueError(msg)
        return self._base_wall_seconds + (monotonic_ns - self._origin_ns) / 1_000_000_000

    def _check_optional_continuity(
        self, meters: ResourceMeterSnapshot
    ) -> dict[str, ScalarReading | CostReading]:
        candidate = dict(self._last_optional)
        for field in _OPTIONAL_FIELDS:
            value = getattr(meters, field)
            if type(value) is EvidenceGap:
                continue
            prior = candidate.get(field)
            if prior is not None:
                if (
                    value.evidence_class is not prior.evidence_class
                    or value.method is not prior.method
                ):
                    msg = f"{field} evidence_class/method changed"
                    raise ValueError(msg)
                if type(value) is ScalarReading and type(prior) is ScalarReading:
                    if value.value < prior.value:
                        msg = f"{field} regressed"
                        raise ValueError(msg)
                elif type(value) is CostReading and type(prior) is CostReading:
                    if value.amount.currency != prior.amount.currency:
                        msg = f"{field} currency changed"
                        raise ValueError(msg)
                    if value.amount.amount < prior.amount.amount:
                        msg = f"{field} regressed"
                        raise ValueError(msg)
                else:
                    msg = f"{field} reading type changed"
                    raise ValueError(msg)
            candidate[field] = value
        return candidate

    def _check_meter_metadata(self, meters: ResourceMeterSnapshot) -> tuple[str | None, int | None]:
        identity = self._last_flop_identity
        observed_identity = meters.flop_model_identity
        if type(observed_identity) is str:
            if identity is not None and observed_identity != identity:
                raise ValueError("flop_model_identity changed")
            identity = observed_identity
        count = self._last_energy_count
        observed_count = meters.energy_sample_count
        if type(observed_count) is int:
            if count is not None and observed_count < count:
                raise ValueError("energy_sample_count regressed")
            count = observed_count
        return identity, count

    def record_eval(
        self,
        *,
        online_positions: int,
        target_positions: int,
        training_step: int,
        observed_at: datetime,
        monotonic_ns: int,
    ) -> ResourceWorkSnapshot:
        _count(online_positions, label="online_positions")
        _count(target_positions, label="target_positions")
        _count(training_step, label="training_step")
        if self._last_training_step is not None and training_step < self._last_training_step:
            msg = "training_step must be nondecreasing"
            raise ValueError(msg)
        wall = self._wall_seconds(monotonic_ns)
        meters = self._meter.snapshot(trainer_wall_seconds=wall)
        if type(meters) is not ResourceMeterSnapshot:
            msg = "meter strategy must return ResourceMeterSnapshot"
            raise TypeError(msg)
        next_optional = self._check_optional_continuity(meters)
        next_identity, next_energy_count = self._check_meter_metadata(meters)
        next_eval = self._eval + online_positions + target_positions
        snapshot = ResourceWorkSnapshot(
            logical_run_id=self.start.logical_run_id,
            execution_segment_index=self.start.execution_segment_index,
            execution_segment_id=self.start.execution_segment_id,
            training_step=training_step,
            eval_index=self._next_eval_index,
            observed_at=observed_at,
            trainer_wall_seconds=wall,
            committed_quality_source_bytes=self._quality_bytes,
            committed_quality_model_tokens=self._quality_tokens,
            online_train_token_positions=self._online,
            target_train_token_positions=self._target,
            baseline_calibration_positions=self._baseline,
            eval_positions=next_eval,
            total_model_token_positions=self._online + self._target + self._baseline + next_eval,
            gpu_train_device_seconds=meters.gpu_train_device_seconds,
            gpu_eval_device_seconds=meters.gpu_eval_device_seconds,
            peak_gpu_allocated_bytes=meters.peak_gpu_allocated_bytes,
            peak_gpu_reserved_bytes=meters.peak_gpu_reserved_bytes,
            cumulative_train_flops=meters.cumulative_train_flops,
            cumulative_eval_flops=meters.cumulative_eval_flops,
            flop_model_identity=meters.flop_model_identity,
            energy_joules=meters.energy_joules,
            energy_sample_count=meters.energy_sample_count,
            rate_estimated_cost=meters.rate_estimated_cost,
        )
        self._eval = next_eval
        self._next_eval_index += 1
        self._last_training_step = training_step
        self._last_monotonic_ns = monotonic_ns
        self._last_optional = next_optional
        self._last_flop_identity = next_identity
        self._last_energy_count = next_energy_count
        return snapshot

    def record_meter_observation(self, *, monotonic_ns: int) -> ResourceMeterSnapshot:
        """Persist current optional meters without manufacturing an evaluation cursor."""
        wall = self._wall_seconds(monotonic_ns)
        meters = self._meter.snapshot(trainer_wall_seconds=wall)
        if type(meters) is not ResourceMeterSnapshot:
            raise TypeError("meter strategy must return ResourceMeterSnapshot")
        next_optional = self._check_optional_continuity(meters)
        next_identity, next_energy_count = self._check_meter_metadata(meters)
        self._last_monotonic_ns = monotonic_ns
        self._last_optional = next_optional
        self._last_flop_identity = next_identity
        self._last_energy_count = next_energy_count
        return meters

    def current_state(self) -> ResourceLedgerState:
        """Return an immutable checkpoint payload with all cumulative counters and meter continuity."""
        wall = self._base_wall_seconds + (self._last_monotonic_ns - self._origin_ns) / 1_000_000_000
        return ResourceLedgerState(
            logical_run_id=self.start.logical_run_id,
            execution_segment_index=self.start.execution_segment_index,
            next_eval_index=self._next_eval_index,
            last_training_step=self._last_training_step,
            trainer_wall_seconds=wall,
            committed_quality_source_bytes=self._quality_bytes,
            committed_quality_model_tokens=self._quality_tokens,
            online_train_token_positions=self._online,
            target_train_token_positions=self._target,
            baseline_calibration_positions=self._baseline,
            eval_positions=self._eval,
            last_optional_readings=tuple(sorted(self._last_optional.items())),
            last_flop_model_identity=self._last_flop_identity,
            last_energy_sample_count=self._last_energy_count,
        )
