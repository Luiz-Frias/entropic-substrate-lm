"""Journal-first admission and exact runtime restoration for resumed D1 segments."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.d1_observability import (
    D1JournalObserver,
    ObservedD1Journal,
)
from entropic_substrate.d1_optimizer import D1OptimizerBundle, build_d1_optimizer
from entropic_substrate.d1_resource_activation import (
    D1ResourceActivation,
    D1ResourceActivationFactory,
    UnavailableResourceActivationFactory,
)
from entropic_substrate.d1_trainer_runtime import (
    D1FreshComponents,
    D1TrainerWallOrigin,
    resolve_d1_device,
)
from entropic_substrate.data.batching import Corpus, SequenceSampler
from entropic_substrate.data.build import build_corpus
from entropic_substrate.evidence.d1_journal_values import (
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_ledger import (
    ResourceLedger,
    ResourceLedgerState,
)
from entropic_substrate.evidence.resource_ledger_codec import (
    resource_ledger_state_from_mapping,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.predictive_state.checkpoint_envelope import validate_d1_checkpoint
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    load_checkpoint_candidate,
)
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = [
    "D1RestoredResume",
    "D1ResumeAdmission",
    "D1ResumeRunLocations",
    "open_d1_resume",
    "restore_d1_resume_runtime",
]


@dataclass(frozen=True, slots=True)
class D1ResumeRunLocations:
    journal_path: Path
    checkpoint_path: Path
    quarantine_path: Path | None = None

    def __post_init__(self) -> None:
        for label, path in (("journal", self.journal_path), ("checkpoint", self.checkpoint_path)):
            if not isinstance(cast(object, path), Path):
                raise TypeError(f"D1 resume {label} path must be a Path")
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"D1 resume {label} path must be a regular existing file")
        if self.journal_path == self.checkpoint_path:
            raise ValueError("D1 resume journal and checkpoint paths must differ")
        quarantine = self.quarantine_path
        if quarantine is not None:
            if not isinstance(cast(object, quarantine), Path):
                raise TypeError("D1 resume quarantine_path must be a Path")
            if not quarantine.parent.is_dir() or quarantine.parent.is_symlink():
                raise ValueError("D1 resume quarantine parent must be a real directory")
            if quarantine in {self.journal_path, self.checkpoint_path}:
                raise ValueError("D1 resume quarantine path must be distinct")


@dataclass(frozen=True, slots=True)
class D1ResumeAdmission:
    plan: D1RunPlan
    start: RunStartProvenance
    origin: D1TrainerWallOrigin
    locations: D1ResumeRunLocations
    checkpoint: dict[str, object]
    checkpoint_receipt: D1CheckpointArtifactReceipt
    checkpoint_journal: JournalCheckpointState
    checkpoint_ledger: ResourceLedgerState
    successful_step: int
    journal: ObservedD1Journal
    resource_ledger: ResourceLedger
    resource_activation: D1ResourceActivation


@dataclass(frozen=True, slots=True)
class D1RestoredResume:
    components: D1FreshComponents
    baseline: D1BaselineCalibrationAttestation
    successful_step: int
    best_nats_per_byte: float


@dataclass(frozen=True, slots=True)
class _UnrestoredRuntime:
    device: torch.device
    corpus: Corpus
    system: D1PredictiveSystem
    optimizer: D1OptimizerBundle
    train_sampler: SequenceSampler
    eval_sampler: SequenceSampler


def _validate_start(
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
) -> None:
    if not isinstance(cast(object, plan), D1RunPlan):
        raise TypeError("D1 resume requires a typed D1RunPlan")
    if type(start) is not RunStartProvenance or type(origin) is not D1TrainerWallOrigin:
        raise TypeError("D1 resume requires typed start provenance and wall origin")
    if start.logical_run_id != plan.logical_run_id:
        raise ValueError("D1 resume logical_run_id does not match the run plan")
    if start.execution_segment_index < 1 or start.segment_start_mode is not SegmentStartMode.RESUME:
        raise ValueError("D1 resume requires a nonzero resume execution segment")
    if start.resume_checkpoint_sha256 is None:
        raise ValueError("D1 resume start is missing checkpoint SHA-256")
    if start.trainer_started_at != origin.started_at:
        raise ValueError("D1 resume trainer_started_at differs from its wall origin")
    protocol = plan.profile.draft.execution_protocol
    if start.compiler_enabled is not protocol.optimization.compile:
        raise ValueError("D1 resume compiler state differs from the execution plan")
    container = plan.run.execution.container
    if container is not None and start.image_digest != container.image_digest:
        raise ValueError("D1 resume image digest differs from the run plan")


def _candidate_evidence(
    checkpoint: dict[str, object],
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
) -> tuple[JournalCheckpointState, ResourceLedgerState, int]:
    try:
        journal = JournalCheckpointState.from_mapping(checkpoint["journal"])
        ledger = resource_ledger_state_from_mapping(checkpoint["resource_ledger"])
        step = checkpoint["successful_step"]
    except KeyError as error:
        raise ValueError(f"D1 resume checkpoint is missing {error.args[0]!r}") from error
    if type(step) is not int or step < 0:
        raise ValueError("D1 resume successful_step must be a nonnegative integer")
    if journal.successful_step != step or ledger.last_training_step != step:
        raise ValueError("D1 resume checkpoint cursors disagree")
    if (
        journal.logical_run_id != plan.logical_run_id
        or ledger.logical_run_id != plan.logical_run_id
    ):
        raise ValueError("D1 resume checkpoint logical_run_id differs from the plan")
    if journal.execution_digest != plan.run.execution_digest:
        raise ValueError("D1 resume checkpoint execution identity differs from the plan")
    if start.execution_segment_index != ledger.execution_segment_index + 1:
        raise ValueError("D1 resume start must be the exact next execution segment")
    if step >= plan.profile.draft.execution_protocol.optimization.max_steps:
        raise ValueError("D1 resume optimization budget is exhausted")
    lifecycle = checkpoint.get("lifecycle")
    if (
        type(lifecycle) is not dict
        or lifecycle.get("schema_version") != "d1-trainer-lifecycle-v1"
        or lifecycle.get("state") != "running"
        or lifecycle.get("completion_reason") is not None
        or lifecycle.get("abort_reason") is not None
    ):
        raise ValueError("D1 resume checkpoint lifecycle is not running")
    return journal, ledger, step


def open_d1_resume(
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    origin: D1TrainerWallOrigin,
    locations: D1ResumeRunLocations,
    observer: D1JournalObserver,
    resource_activation_factory: D1ResourceActivationFactory | None = None,
) -> D1ResumeAdmission:
    """Verify the artifact and durably record this segment before model construction."""
    if type(locations) is not D1ResumeRunLocations:
        raise TypeError("D1 resume requires D1ResumeRunLocations")
    _validate_start(plan, start, origin)
    checkpoint, receipt = load_checkpoint_candidate(
        locations.checkpoint_path,
        expected_sha256=start.resume_checkpoint_sha256,
    )
    checkpoint_journal, checkpoint_ledger, step = _candidate_evidence(
        checkpoint,
        plan=plan,
        start=start,
    )
    activation_factory = resource_activation_factory or UnavailableResourceActivationFactory()
    if not isinstance(cast(object, activation_factory), D1ResourceActivationFactory):
        raise TypeError("D1 resource activation factory is invalid")
    activation = activation_factory.resume(start, checkpoint_ledger)
    ledger = ResourceLedger.restore(
        start,
        checkpoint_ledger,
        monotonic_origin_ns=origin.monotonic_ns,
        meter_strategy=activation.meter,
    )
    provenance = cast(
        Mapping[str, FrozenJson],
        {
            "run_start_provenance": run_start_to_mapping(start),
            "trainer_wall_origin": origin.manifest,
        },
    )
    journal = ObservedD1Journal.resume(
        locations.journal_path,
        checkpoint_journal,
        checkpoint_sha256=receipt.sha256,
        segment_provenance=provenance,
        quarantine_path=locations.quarantine_path,
        observer=observer,
    )
    admission = D1ResumeAdmission(
        plan,
        start,
        origin,
        locations,
        checkpoint,
        receipt,
        checkpoint_journal,
        checkpoint_ledger,
        step,
        journal,
        ledger,
        activation,
    )
    _record_quarantined_work(admission, ledger)
    return admission


def _materialize(plan: D1RunPlan) -> _UnrestoredRuntime:
    train, data = plan.train, plan.profile.data
    protocol = plan.profile.draft.execution_protocol
    random.seed(plan.seed)
    np.random.seed(plan.seed % (2**32))
    torch.manual_seed(plan.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(plan.seed)
    device = resolve_d1_device(train.device)
    corpus = build_corpus(data, "byte")
    if corpus.meta.vocab_size != plan.profile.draft.codec.cardinality:
        raise ValueError("D1 resume corpus alphabet differs from the observation codec")
    if not math.isclose(corpus.meta.bytes_per_token, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("D1 resume requires exactly one source byte per model token")
    system = D1PredictiveSystem.from_family(
        plan.profile.family,
        plan.profile.arm,
        initializer=SeededD1BranchInitializer(seed=plan.seed),
    ).to(device)
    optimizer = build_d1_optimizer(system, protocol.optimization)
    batching = protocol.batching
    return _UnrestoredRuntime(
        device,
        corpus,
        system,
        optimizer,
        SequenceSampler(
            corpus.train,
            batching.batch_size,
            batching.sequence_length,
            train=True,
            seed=plan.seed,
            device=device,
        ),
        SequenceSampler(
            corpus.val,
            batching.batch_size,
            batching.sequence_length,
            train=False,
            seed=plan.seed,
            device=device,
        ),
    )


def _baseline(value: object) -> D1BaselineCalibrationAttestation:
    return D1BaselineCalibrationAttestation(
        **cast(dict[str, object], value)  # type: ignore[arg-type]
    )


def _restore_rng(value: object) -> None:
    rng = cast(dict[str, object], value)
    random.setstate(cast(tuple[object, ...], rng["python"]))
    numpy = cast(dict[str, object], rng["numpy"])
    keys = cast(torch.Tensor, numpy["keys"])
    np.random.set_state(
        (
            cast(str, numpy["bit_generator"]),
            keys.cpu().numpy(),
            cast(int, numpy["position"]),
            cast(int, numpy["has_gauss"]),
            cast(float, numpy["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cast(torch.Tensor, rng["torch_cpu"]))
    cuda = cast(tuple[torch.Tensor, ...] | None, rng["cuda"])
    if torch.cuda.is_available() != (cuda is not None):
        raise RuntimeError("D1 exact resume CUDA RNG availability differs from checkpoint")
    if cuda is not None:
        if len(cuda) != torch.cuda.device_count():
            raise RuntimeError("D1 exact resume CUDA RNG device count differs from checkpoint")
        torch.cuda.set_rng_state_all(list(cuda))


def _record_quarantined_work(admission: D1ResumeAdmission, ledger: ResourceLedger) -> None:
    quarantine = admission.journal.tail_quarantine
    if quarantine is None:
        return
    train_count = sum(
        event.event_kind is JournalEventKind.TRAIN for event in quarantine.tail_events
    )
    eval_count = sum(event.event_kind is JournalEventKind.EVAL for event in quarantine.tail_events)
    protocol = admission.plan.profile.draft.execution_protocol
    batching = protocol.batching
    online = batching.batch_size * batching.sequence_length
    target = batching.batch_size * (batching.sequence_length + batching.horizon)
    eval_positions = protocol.evaluation.num_batches * (online + target)
    ledger.record_discarded_work(
        online_train_positions=train_count * online,
        target_train_positions=train_count * target,
        eval_positions=eval_count * eval_positions,
    )


def restore_d1_resume_runtime(
    admission: D1ResumeAdmission,
) -> D1RestoredResume:
    """Materialize only after segment evidence, then fully validate and restore state."""
    if type(admission) is not D1ResumeAdmission:
        raise TypeError("D1 resume restoration requires D1ResumeAdmission")
    runtime = _materialize(admission.plan)
    state = validate_d1_checkpoint(
        admission.checkpoint,
        system=runtime.system,
        optimizer=runtime.optimizer.optimizer,
        sampler=runtime.train_sampler,
        run=admission.plan.run,
        protocol=admission.plan.profile.draft.execution_protocol,
        for_resume=True,
    )
    runtime.system.load_state_dict(cast(dict[str, object], state["model"]), strict=True)
    runtime.optimizer.optimizer.load_state_dict(cast(dict[str, object], state["optimizer"]))
    runtime.train_sampler.load_state_dict(cast(dict[str, torch.Tensor | int], state["sampler"]))
    _restore_rng(state["rng"])
    best = state["best_nats_per_byte"]
    if isinstance(best, bool) or not isinstance(best, (int, float)):
        raise ValueError("D1 resumable checkpoint requires best_nats_per_byte")
    components = D1FreshComponents(
        admission.plan,
        runtime.device,
        runtime.corpus,
        runtime.system,
        runtime.optimizer,
        runtime.train_sampler,
        runtime.eval_sampler,
        admission.resource_ledger,
    )
    return D1RestoredResume(
        components,
        _baseline(state["baseline_calibration"]),
        admission.successful_step,
        float(best),
    )
