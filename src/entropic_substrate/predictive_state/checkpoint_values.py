"""Construction of the pure, versioned D1 exact-resume checkpoint value."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Final, cast

import numpy as np
import torch
from torch import nn

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.evidence.d1_journal_values import JournalCheckpointState
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_ledger_codec import resource_ledger_state_to_mapping
from entropic_substrate.experiments.d1_execution import D1ExecutionProtocol
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.experiments.d1_runs import D1RunSpec
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = [
    "D1_CHECKPOINT_KIND",
    "D1_CHECKPOINT_SCHEMA_VERSION",
    "D1CheckpointError",
    "build_d1_checkpoint",
]

D1_CHECKPOINT_KIND: Final = "entropic-substrate-d1-checkpoint"
D1_CHECKPOINT_SCHEMA_VERSION: Final = 1
TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    (
        "kind schema_version rung successful_step identity model optimizer sampler rng "
        "baseline_calibration resource_ledger journal best_nats_per_byte lifecycle"
    ).split()
)


class D1CheckpointError(RuntimeError):
    """A snapshot cannot be trusted without changing its experimental meaning."""


def require_mapping(
    value: object, label: str, keys: frozenset[str] | set[str] | None = None
) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise D1CheckpointError(f"{label} must be a mapping with string keys")
    if keys is not None and set(value) != keys:
        missing, unknown = sorted(keys - set(value)), sorted(set(value) - keys)
        raise D1CheckpointError(f"{label} keys must be exact; missing={missing}, unknown={unknown}")
    return value


def require_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise D1CheckpointError(f"{label} must be a nonnegative integer")
    return value


def require_finite(
    value: object, label: str, *, optional: bool = False, nonnegative: bool = True
) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D1CheckpointError(f"{label} must be a finite scalar")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "nonnegative " if nonnegative else ""
        raise D1CheckpointError(f"{label} must be a finite {qualifier}scalar")


def _ownership_digest(system: D1PredictiveSystem, optimizer: torch.optim.Optimizer) -> str:
    system.validate_optimizer_ownership(optimizer)
    names = {id(parameter): name for name, parameter in system.named_parameters()}
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        owned: list[str] = []
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter) or id(parameter) not in names:
                raise D1CheckpointError("optimizer owns a parameter outside the D1 runtime")
            owned.append(names[id(parameter)])
        groups.append(owned)
    return canonical_digest({"parameter_groups": groups})


def expected_d1_checkpoint_identity(
    system: object,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    run: object,
    protocol: D1ExecutionProtocol,
) -> dict[str, object]:
    """Bind runtime state to the exact registered run and optimizer ownership."""
    if not isinstance(system, D1PredictiveSystem) or not isinstance(run, D1RunSpec):
        raise D1CheckpointError("D1 checkpoint requires validated runtime and run identities")
    if type(protocol) is not D1ExecutionProtocol or protocol != run.draft.execution_protocol:
        raise D1CheckpointError("checkpoint execution protocol does not match its D1 run")
    if system.draft != run.draft or system.construction_identity != run.construction_digest:
        raise D1CheckpointError("checkpoint runtime and run construction identities differ")
    family = system.family_identity
    if family is None or (run.family is not None and family != run.family_digest):
        raise D1CheckpointError("checkpoint runtime and run family identities differ")
    if system.initialization_attestation.seed != run.execution.seed:
        raise D1CheckpointError("checkpoint initialization seed differs from execution seed")
    if run.evidence.checkpoint_schema_version != D1_CHECKPOINT_SCHEMA_VERSION:
        raise D1CheckpointError("D1 evidence declares an unsupported checkpoint schema")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise D1CheckpointError("D1 checkpoint optimizer must be AdamW")
    if type(sampler) is not SequenceSampler or not sampler.train:
        raise D1CheckpointError("D1 checkpoint requires its training SequenceSampler")
    if (
        sampler.batch_size != protocol.batching.batch_size
        or sampler.seq_len != protocol.batching.sequence_length
    ):
        raise D1CheckpointError("checkpoint sampler construction differs from execution protocol")
    attestation = system.initialization_attestation
    return {
        "arm": run.draft.arm.value,
        "construction_digest": run.construction_digest,
        "family_digest": family,
        "aggregation_digest": run.aggregation_digest,
        "execution_digest": run.execution_digest,
        "run_manifest": run.execution_manifest,
        "execution_protocol_digest": protocol.protocol_digest,
        "optimizer_protocol_digest": canonical_digest(protocol.optimization.manifest),
        "optimizer_ownership_digest": _ownership_digest(system, optimizer),
        "initialization": {
            "identity": attestation.identity.manifest,
            "seed": attestation.seed,
            "online_parameter_digest": attestation.online_parameter_digest,
            "target_parameter_digest": attestation.target_parameter_digest,
        },
    }


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_checkpoint_alignment(
    *,
    system: object,
    run: object,
    protocol: D1ExecutionProtocol,
    successful_step: int,
    baseline: object,
    ledger: object,
    journal: object,
    serialized_mean: torch.Tensor | None = None,
) -> None:
    """Require all typed evidence cursors to name one eval-boundary history."""
    if not isinstance(system, D1PredictiveSystem) or not isinstance(run, D1RunSpec):
        raise D1CheckpointError("checkpoint alignment requires typed runtime and run values")
    if not isinstance(baseline, D1BaselineCalibrationAttestation):
        raise D1CheckpointError("baseline_calibration must be a typed attestation")
    if not isinstance(ledger, ResourceLedgerState):
        raise D1CheckpointError("resource_ledger must be a typed ledger state")
    if not isinstance(journal, JournalCheckpointState):
        raise D1CheckpointError("journal must be a typed checkpoint state")
    require_nonnegative_int(successful_step, "successful_step")
    if successful_step > protocol.optimization.max_steps:
        raise D1CheckpointError("successful_step exceeds the registered optimization budget")
    terminal = successful_step == protocol.optimization.max_steps
    if successful_step % protocol.evaluation.eval_interval and not terminal:
        raise D1CheckpointError("checkpoint successful_step must be an evaluation boundary")
    batches, batching = protocol.baseline.num_batches, protocol.batching
    expected = {
        "protocol_digest": protocol.protocol_digest,
        "spec_digest": canonical_digest(protocol.baseline.manifest),
        "num_batches": batches,
        "online_positions": batches * batching.batch_size * batching.sequence_length,
        "source_span_positions": (
            batches * batching.batch_size * (batching.sequence_length + batching.horizon)
        ),
        "successful_step_at_fit": 0,
        "codec_digest": run.draft.codec.construction_digest,
        "initialization_seed": run.execution.seed,
    }
    for field, value in expected.items():
        if getattr(baseline, field) != value:
            raise D1CheckpointError(f"baseline_calibration {field} does not match the D1 run")
    if run.data is not None and baseline.split_digest != run.data.split_digest:
        raise D1CheckpointError("baseline_calibration split_digest does not match the D1 run")
    mean = system.mean_displacement_baseline.mean if serialized_mean is None else serialized_mean
    if type(mean) is not torch.Tensor:
        raise D1CheckpointError("checkpoint baseline mean must be a tensor")
    if baseline.mean_parameter_digest != _tensor_digest(mean):
        raise D1CheckpointError("baseline_calibration mean_parameter_digest does not match model")
    if ledger.last_training_step != successful_step:
        raise D1CheckpointError("resource_ledger last_training_step must equal successful_step")
    if ledger.baseline_calibration_positions != baseline.source_span_positions:
        raise D1CheckpointError("resource_ledger baseline calibration work does not match")
    if journal.successful_step != successful_step:
        raise D1CheckpointError("journal successful_step must equal checkpoint successful_step")
    if journal.execution_digest != run.execution_digest:
        raise D1CheckpointError("journal execution_digest does not match the D1 run")
    if ledger.logical_run_id != journal.logical_run_id:
        raise D1CheckpointError("resource_ledger and journal logical_run_id values differ")


def _rng_state() -> dict[str, object]:
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    cuda = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "cuda": cuda,
    }


def build_d1_checkpoint(
    *,
    system: D1PredictiveSystem,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    run: D1RunSpec,
    protocol: D1ExecutionProtocol,
    successful_step: int,
    baseline_calibration: D1BaselineCalibrationAttestation,
    resource_ledger: ResourceLedgerState,
    journal_state: JournalCheckpointState,
    lifecycle: dict[str, object],
    best_nats_per_byte: float | None,
) -> dict[str, object]:
    """Capture one torch-serializable mapping without performing I/O."""
    identity = expected_d1_checkpoint_identity(system, optimizer, sampler, run, protocol)
    require_checkpoint_alignment(
        system=system,
        run=run,
        protocol=protocol,
        successful_step=successful_step,
        baseline=baseline_calibration,
        ledger=resource_ledger,
        journal=journal_state,
    )
    return {
        "kind": D1_CHECKPOINT_KIND,
        "schema_version": D1_CHECKPOINT_SCHEMA_VERSION,
        "rung": "d1",
        "successful_step": successful_step,
        "identity": identity,
        "model": dict(system.state_dict()),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": _rng_state(),
        "baseline_calibration": baseline_calibration.manifest,
        "resource_ledger": resource_ledger_state_to_mapping(resource_ledger),
        "journal": journal_state.manifest,
        "best_nats_per_byte": best_nats_per_byte,
        "lifecycle": dict(lifecycle),
    }
