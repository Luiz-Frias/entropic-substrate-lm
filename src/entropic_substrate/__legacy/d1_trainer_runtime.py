"""Fresh D1 composition root: clocks, device, corpus, model, and samplers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, cast, runtime_checkable

import numpy as np
import torch

from entropic_substrate.d1_optimizer import D1OptimizerBundle, build_d1_optimizer
from entropic_substrate.data.batching import Corpus, SequenceSampler
from entropic_substrate.data.build import build_corpus
from entropic_substrate.evidence.resource_ledger import (
    ResourceLedger,
    ResourceMeterStrategy,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = [
    "D1FreshComponents",
    "D1ModelRuntime",
    "D1TrainerClock",
    "D1TrainerWallOrigin",
    "SystemD1TrainerClock",
    "build_d1_fresh_components",
    "create_d1_resource_ledger",
    "materialize_d1_model_runtime",
    "resolve_d1_device",
]


def _require_utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")


@dataclass(frozen=True, slots=True)
class D1TrainerWallOrigin:
    """Composition-root clock reading taken before corpus or model construction."""

    started_at: datetime
    monotonic_ns: int

    def __post_init__(self) -> None:
        _require_utc(self.started_at, label="D1 trainer wall origin started_at")
        if type(self.monotonic_ns) is not int or self.monotonic_ns < 0:
            raise ValueError("D1 trainer wall origin monotonic_ns must be nonnegative")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "started_at": self.started_at.isoformat(),
        }


@runtime_checkable
class D1TrainerClock(Protocol):
    """Two-clock port; elapsed work must never be derived from wall-clock time."""

    def utc_now(self) -> datetime: ...

    def monotonic_ns(self) -> int: ...


class SystemD1TrainerClock:
    def utc_now(self) -> datetime:
        from datetime import UTC

        return datetime.now(UTC)

    def monotonic_ns(self) -> int:
        import time

        return time.monotonic_ns()


def _validate_fresh_identity(
    *, plan: D1RunPlan, start: RunStartProvenance, origin: D1TrainerWallOrigin
) -> None:
    if start.logical_run_id != plan.logical_run_id:
        raise ValueError("D1 run-start logical_run_id does not match the run plan")
    if start.execution_segment_index != 0 or start.segment_start_mode is not SegmentStartMode.FRESH:
        raise ValueError("D1 fresh trainer requires execution segment zero in fresh mode")
    if start.resume_checkpoint_sha256 is not None or plan.train.resume is not None:
        raise ValueError("D1 fresh trainer refuses resume checkpoint state")
    if start.trainer_started_at != origin.started_at:
        raise ValueError("run-start trainer_started_at must equal the supplied wall origin")
    protocol = plan.profile.draft.execution_protocol
    if start.compiler_enabled is not protocol.optimization.compile:
        raise ValueError("run-start compiler state differs from the D1 optimization plan")
    container = plan.run.execution.container
    if container is not None and start.image_digest != container.image_digest:
        raise ValueError("run-start image digest differs from the D1 run plan")


def create_d1_resource_ledger(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    meter_strategy: ResourceMeterStrategy | None = None,
) -> ResourceLedger:
    """Validate the pre-construction time boundary and materialize its work ledger."""
    if not isinstance(cast(object, plan), D1RunPlan):
        raise TypeError("D1 fresh trainer requires a typed D1RunPlan")
    if type(start) is not RunStartProvenance:
        raise TypeError("D1 fresh trainer requires RunStartProvenance")
    if type(origin) is not D1TrainerWallOrigin:
        raise TypeError("D1 fresh trainer requires D1TrainerWallOrigin")
    _validate_fresh_identity(plan=plan, start=start, origin=origin)
    return ResourceLedger(
        start,
        monotonic_origin_ns=origin.monotonic_ns,
        meter_strategy=meter_strategy,
    )


def resolve_d1_device(spec: str) -> torch.device:
    """Resolve the plan's device without silently degrading an explicit request."""
    if type(spec) is not str or not spec.strip():
        raise ValueError("D1 device specification must be a nonempty string")
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(spec)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid D1 device specification {spec!r}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"D1 plan requested unavailable CUDA device {spec!r}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("D1 plan requested unavailable MPS device")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"D1 plan requested unsupported device type {device.type!r}")
    return device


def _seed_from_plan(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True, slots=True)
class D1ModelRuntime:
    """Segment-independent model, optimizer, corpus, and sampler runtime."""

    device: torch.device
    corpus: Corpus
    system: D1PredictiveSystem
    optimizer: D1OptimizerBundle
    train_sampler: SequenceSampler
    eval_sampler: SequenceSampler


def materialize_d1_model_runtime(plan: D1RunPlan) -> D1ModelRuntime:
    """Construct the one model runtime used by fresh and resumed D1 segments."""
    if not isinstance(cast(object, plan), D1RunPlan):
        raise TypeError("D1 model materialization requires a typed D1RunPlan")
    train = plan.train
    data = plan.profile.data
    protocol = plan.profile.draft.execution_protocol
    if train.seed != plan.seed:
        raise ValueError("D1 run-plan seed did not reach the training adapter")
    device = resolve_d1_device(train.device)
    _seed_from_plan(plan.seed)
    corpus = build_corpus(data, "byte")
    if corpus.meta.vocab_size != plan.profile.draft.codec.cardinality:
        raise ValueError("D1 corpus alphabet differs from the registered observation codec")
    if not math.isclose(corpus.meta.bytes_per_token, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("D1 model runtime requires exactly one source byte per model token")
    system = D1PredictiveSystem.from_family(
        plan.profile.family,
        plan.profile.arm,
        initializer=SeededD1BranchInitializer(seed=plan.seed),
    ).to(device)
    optimizer = build_d1_optimizer(system, protocol.optimization)
    batching = protocol.batching
    return D1ModelRuntime(
        device=device,
        corpus=corpus,
        system=system,
        optimizer=optimizer,
        train_sampler=SequenceSampler(
            corpus.train,
            batch_size=batching.batch_size,
            seq_len=batching.sequence_length,
            train=True,
            seed=plan.seed,
            device=device,
        ),
        eval_sampler=SequenceSampler(
            corpus.val,
            batch_size=batching.batch_size,
            seq_len=batching.sequence_length,
            train=False,
            seed=plan.seed,
            device=device,
        ),
    )


@dataclass(frozen=True, slots=True)
class D1FreshComponents:
    """Materialized segment adapters sharing one validated D1 plan."""

    plan: D1RunPlan
    device: torch.device
    corpus: Corpus
    system: D1PredictiveSystem
    optimizer: D1OptimizerBundle
    train_sampler: SequenceSampler
    eval_sampler: SequenceSampler
    resource_ledger: ResourceLedger


def build_d1_fresh_components(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    resource_ledger: ResourceLedger,
) -> D1FreshComponents:
    """Build every heavyweight component after admission of the supplied wall origin."""
    if not isinstance(cast(object, plan), D1RunPlan):
        raise TypeError("D1 fresh construction requires a typed D1RunPlan")
    if type(resource_ledger) is not ResourceLedger:
        raise TypeError("D1 fresh construction requires a ResourceLedger")
    _validate_fresh_identity(plan=plan, start=start, origin=origin)
    if resource_ledger.start != start:
        raise ValueError("D1 resource ledger does not belong to the admitted run start")
    runtime = materialize_d1_model_runtime(plan)
    return D1FreshComponents(
        plan=plan,
        device=runtime.device,
        corpus=runtime.corpus,
        system=runtime.system,
        optimizer=runtime.optimizer,
        train_sampler=runtime.train_sampler,
        eval_sampler=runtime.eval_sampler,
        resource_ledger=resource_ledger,
    )
