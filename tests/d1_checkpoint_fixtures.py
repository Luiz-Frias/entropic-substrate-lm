"""Deterministic typed values shared by D1 checkpoint contract tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from d1_runtime_fixtures import fit_runtime_baseline, runtime_family
from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.kernel.identity.d1_primitives import (
    D1Arm,
    EvidenceClass,
    RunClass,
    canonical_digest,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalCheckpointState
from entropic_substrate.ladder.capture.resource.resource_ledger import ResourceLedgerState
from entropic_substrate.ladder.capture.resource.resource_work import ReadingMethod, ScalarReading
from entropic_substrate.ladder.register.d1_execution import D1ExecutionProtocol
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)
from entropic_substrate.ladder.register.d1_runs import D1RunSpec
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem
from entropic_substrate.ladder.run.trainer.checkpoint_values import build_d1_checkpoint
from entropic_substrate.ladder.run.trainer.d1_calibration import D1BaselineCalibrationAttestation


@dataclass(frozen=True, slots=True)
class CheckpointCase:
    system: D1PredictiveSystem
    optimizer: torch.optim.AdamW
    sampler: SequenceSampler
    run: D1RunSpec
    protocol: D1ExecutionProtocol
    baseline: D1BaselineCalibrationAttestation
    ledger: ResourceLedgerState
    journal: JournalCheckpointState


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_case() -> CheckpointCase:
    family = runtime_family()
    system = D1PredictiveSystem.from_family(
        family, D1Arm.BOTH, initializer=SeededD1BranchInitializer(seed=7)
    )
    fit_runtime_baseline(system)
    protocol = family.draft_for(D1Arm.BOTH).execution_protocol
    optimizer = torch.optim.AdamW(system.optimizer_parameters(), lr=6e-4)
    sampler = SequenceSampler(
        TokenSplit(np.arange(4096, dtype=np.int64) % 11),
        protocol.batching.batch_size,
        protocol.batching.sequence_length,
        seed=7,
    )
    execution = ExecutionIdentity(
        run_class=RunClass.CONFIRMATORY,
        seed=7,
        pairing_id="d1-pair-7",
        code=CodeIdentity(commit="a" * 40, repository_digest="b" * 64),
        container=ContainerIdentity(image_digest="sha256:" + "c" * 64),
    )
    data = DataIdentity(
        dataset_name="enwik8",
        dataset_digest="d" * 64,
        split_digest="e" * 64,
        sampler_construction_digest="f" * 64,
    )
    run = family.validate_confirmatory_arm(
        arm=D1Arm.BOTH,
        execution=execution,
        data=data,
        evidence=EvidenceEnvelope(
            schema_version="d1-evidence-v1",
            required_classes=frozenset({EvidenceClass.MEASURED}),
        ),
        preregistration=PreregistrationIdentity(
            path="docs/d1-preregistration.md",
            commit="1" * 40,
            family_digest=family.family_digest,
        ),
    )
    batches = protocol.baseline.num_batches
    online_positions = batches * protocol.batching.batch_size * protocol.batching.sequence_length
    span_positions = (
        batches
        * protocol.batching.batch_size
        * (protocol.batching.sequence_length + protocol.batching.horizon)
    )
    baseline = D1BaselineCalibrationAttestation(
        protocol_digest=protocol.protocol_digest,
        spec_digest=canonical_digest(protocol.baseline.manifest),
        split_digest=data.split_digest,
        source_order_digest="2" * 64,
        mean_parameter_digest=_tensor_digest(system.mean_displacement_baseline.mean),
        codec_digest=run.draft.codec.construction_digest,
        initialization_seed=run.execution.seed,
        num_batches=batches,
        online_positions=online_positions,
        source_span_positions=span_positions,
    )
    logical_run_id = "d1-confirmatory-both-seed-7"
    ledger = ResourceLedgerState(
        logical_run_id=logical_run_id,
        execution_segment_index=0,
        next_eval_index=1,
        last_training_step=0,
        trainer_wall_seconds=0.0,
        committed_quality_source_bytes=0,
        committed_quality_model_tokens=0,
        online_train_token_positions=0,
        target_train_token_positions=0,
        baseline_calibration_positions=span_positions,
        eval_positions=0,
        last_optional_readings=(
            (
                "gpu_train_device_seconds",
                ScalarReading(
                    value=0.0,
                    evidence_class=EvidenceClass.MEASURED,
                    method=ReadingMethod.GPU_DEVICE_COUNTER,
                ),
            ),
        ),
        last_flop_model_identity=None,
        last_energy_sample_count=None,
    )
    journal = JournalCheckpointState(
        logical_run_id=logical_run_id,
        execution_digest=run.execution_digest,
        last_event_sequence=2,
        prefix_sha256="3" * 64,
        successful_step=0,
    )
    return CheckpointCase(system, optimizer, sampler, run, protocol, baseline, ledger, journal)


def build_checkpoint(case: CheckpointCase, *, run: D1RunSpec | None = None) -> dict[str, object]:
    return build_d1_checkpoint(
        system=case.system,
        optimizer=case.optimizer,
        sampler=case.sampler,
        run=case.run if run is None else run,
        protocol=case.protocol,
        successful_step=0,
        baseline_calibration=case.baseline,
        resource_ledger=case.ledger,
        journal_state=case.journal,
        lifecycle={
            "schema_version": "d1-trainer-lifecycle-v1",
            "state": "running",
            "completion_reason": None,
            "abort_reason": None,
        },
        best_nats_per_byte=1.25,
    )


def checkpoint_state() -> tuple[dict[str, object], CheckpointCase]:
    case = checkpoint_case()
    return build_checkpoint(case), case


def set_path(state: dict[str, object], path: tuple[str, ...], value: object) -> None:
    target = state
    for key in path[:-1]:
        target = cast(dict[str, object], target[key])
    target[path[-1]] = value
