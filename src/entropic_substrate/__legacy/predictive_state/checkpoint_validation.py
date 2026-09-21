"""Strict D1 checkpoint admission and continuation-policy validation."""

from __future__ import annotations

from typing import cast

import torch

from entropic_substrate.d1_calibration import D1BaselineCalibrationAttestation
from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    JournalCheckpointState,
)
from entropic_substrate.evidence.resource_ledger_codec import (
    ResourceLedgerCodecGap,
    resource_ledger_state_from_mapping,
)
from entropic_substrate.experiments.d1_execution import D1ExecutionProtocol
from entropic_substrate.experiments.d1_primitives import TargetUpdateStrategy
from entropic_substrate.experiments.d1_runs import D1RunSpec
from entropic_substrate.predictive_state.checkpoint_values import (
    D1_CHECKPOINT_KIND,
    D1_CHECKPOINT_SCHEMA_VERSION,
    TOP_LEVEL_KEYS,
    D1CheckpointError,
    expected_d1_checkpoint_identity,
    require_checkpoint_alignment,
    require_finite,
    require_mapping,
    require_nonnegative_int,
)
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

__all__ = ["validate_d1_checkpoint"]


def _validate_model(
    value: object,
    expected: dict[str, object],
    system: D1PredictiveSystem,
    run: D1RunSpec,
    step: int,
) -> None:
    model = require_mapping(value, "model")
    required: frozenset[str] = frozenset(
        (
            "_extra_state mean_displacement_baseline._mean "
            "mean_displacement_baseline._fitted mean_displacement_baseline._extra_state"
        ).split()
    )
    if not required <= set(model):
        raise D1CheckpointError(
            f"model is missing exact-resume state {sorted(required - set(model))}"
        )
    extra = require_mapping(
        model["_extra_state"],
        "model._extra_state",
        {"version", "construction_identity", "family_identity", "initialization", "target_policy"},
    )
    runtime_extra = system.get_extra_state()
    for key in ("version", "construction_identity", "family_identity", "initialization"):
        if extra[key] != runtime_extra[key]:
            raise D1CheckpointError(f"model {key} does not match checkpoint identity")
    if extra["initialization"] != expected["initialization"]:
        raise D1CheckpointError("model initialization does not match execution identity")
    fitted = model["mean_displacement_baseline._fitted"]
    if type(fitted) is not torch.Tensor or fitted.numel() != 1 or not bool(fitted.item()):
        raise D1CheckpointError("model baseline must be fitted before a D1 checkpoint")
    baseline_extra = require_mapping(
        model["mean_displacement_baseline._extra_state"],
        "model baseline provenance",
        {"codec_identity"},
    )
    codec = run.draft.codec
    expected_codec = {
        "name": codec.name,
        "construction_digest": codec.construction_digest,
        "cardinality": codec.cardinality,
    }
    if baseline_extra["codec_identity"] != expected_codec:
        raise D1CheckpointError("model baseline codec provenance does not match the D1 run")
    policy = require_mapping(extra["target_policy"], "model target_policy")
    expected_policy = system.target_policy_state()
    if run.draft.target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA:
        if set(policy) != set(expected_policy) or any(
            policy[key] != expected_policy[key] for key in ("kind", "version", "schedule")
        ):
            raise D1CheckpointError("model target policy differs from the registered EMA policy")
        if policy["update_cursor"] != step:
            raise D1CheckpointError("EMA update_cursor must equal successful_step")
    elif policy != expected_policy:
        raise D1CheckpointError("model target policy is not the registered online control")


def _validate_rng(value: object) -> None:
    rng = require_mapping(value, "rng", {"python", "numpy", "torch_cpu", "cuda"})
    python = rng["python"]
    if type(python) is not tuple or len(python) != 3:
        raise D1CheckpointError("rng.python is malformed")
    numpy = require_mapping(
        rng["numpy"],
        "rng.numpy",
        {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
    )
    keys = numpy["keys"]
    if (
        numpy["bit_generator"] != "MT19937"
        or type(keys) is not torch.Tensor
        or keys.ndim != 1
        or keys.dtype != torch.uint32
    ):
        raise D1CheckpointError("rng.numpy state is malformed")
    position = require_nonnegative_int(numpy["position"], "rng.numpy.position")
    has_gauss = require_nonnegative_int(numpy["has_gauss"], "rng.numpy.has_gauss")
    if position > keys.numel() or has_gauss not in {0, 1}:
        raise D1CheckpointError("rng.numpy position or Gaussian-cache flag is invalid")
    require_finite(numpy["cached_gaussian"], "rng.numpy.cached_gaussian", nonnegative=False)
    cpu = rng["torch_cpu"]
    if type(cpu) is not torch.Tensor or cpu.ndim != 1 or cpu.dtype != torch.uint8:
        raise D1CheckpointError("rng.torch_cpu must be a rank-1 byte tensor")
    cuda = rng["cuda"]
    if cuda is not None and (
        type(cuda) is not tuple
        or any(
            type(item) is not torch.Tensor or item.ndim != 1 or item.dtype != torch.uint8
            for item in cuda
        )
    ):
        raise D1CheckpointError("rng.cuda must be null or a tuple of rank-1 byte tensors")


_BASELINE_FIELDS = frozenset(
    (
        "protocol_digest spec_digest split_digest source_order_digest mean_parameter_digest "
        "codec_digest initialization_seed num_batches online_positions source_span_positions "
        "successful_step_at_fit schema_version"
    ).split()
)


def _baseline_from_mapping(value: object) -> D1BaselineCalibrationAttestation:
    data = require_mapping(value, "baseline_calibration", _BASELINE_FIELDS)
    try:
        return D1BaselineCalibrationAttestation(
            protocol_digest=cast(str, data["protocol_digest"]),
            spec_digest=cast(str, data["spec_digest"]),
            split_digest=cast(str, data["split_digest"]),
            source_order_digest=cast(str, data["source_order_digest"]),
            mean_parameter_digest=cast(str, data["mean_parameter_digest"]),
            codec_digest=cast(str, data["codec_digest"]),
            initialization_seed=cast(int, data["initialization_seed"]),
            num_batches=cast(int, data["num_batches"]),
            online_positions=cast(int, data["online_positions"]),
            source_span_positions=cast(int, data["source_span_positions"]),
            successful_step_at_fit=cast(int, data["successful_step_at_fit"]),
            schema_version=cast(str, data["schema_version"]),
        )
    except (TypeError, ValueError) as exc:
        raise D1CheckpointError(f"baseline_calibration is invalid: {exc}") from exc


def _validate_typed_evidence(
    state: dict[str, object],
    system: D1PredictiveSystem,
    run: D1RunSpec,
    protocol: D1ExecutionProtocol,
    step: int,
) -> None:
    baseline = _baseline_from_mapping(state["baseline_calibration"])
    model = require_mapping(state["model"], "model")
    serialized_mean = model.get("mean_displacement_baseline._mean")
    if type(serialized_mean) is not torch.Tensor:
        raise D1CheckpointError("model baseline mean must be a tensor")
    try:
        ledger = resource_ledger_state_from_mapping(state["resource_ledger"])
    except ResourceLedgerCodecGap as exc:
        raise D1CheckpointError(f"resource ledger state is invalid: {exc}") from exc
    try:
        journal = JournalCheckpointState.from_mapping(state["journal"])
    except (D1JournalError, TypeError, ValueError) as exc:
        raise D1CheckpointError(f"journal checkpoint state is invalid: {exc}") from exc
    require_checkpoint_alignment(
        system=system,
        run=run,
        protocol=protocol,
        successful_step=step,
        baseline=baseline,
        ledger=ledger,
        journal=journal,
        serialized_mean=serialized_mean,
    )


def _validate_lifecycle(value: object, *, for_resume: bool) -> None:
    lifecycle = require_mapping(
        value,
        "lifecycle",
        {"schema_version", "state", "completion_reason", "abort_reason"},
    )
    status = lifecycle["state"]
    if lifecycle["schema_version"] != "d1-trainer-lifecycle-v1" or status not in {
        "running",
        "completed",
        "aborted",
    }:
        raise D1CheckpointError("trainer lifecycle state or schema is unsupported")
    completion, abort = lifecycle["completion_reason"], lifecycle["abort_reason"]
    valid = (
        (status == "running" and completion is None and abort is None)
        or (
            status == "completed"
            and type(completion) is str
            and bool(completion.strip())
            and abort is None
        )
        or (
            status == "aborted"
            and completion is None
            and type(abort) is str
            and bool(abort.strip())
        )
    )
    if not valid:
        raise D1CheckpointError("trainer lifecycle completion/abort reasons are inconsistent")
    if status != "running" and for_resume:
        raise D1CheckpointError(f"trainer lifecycle {status!r} snapshot is not resumable")


def validate_d1_checkpoint(
    state: object,
    *,
    system: D1PredictiveSystem,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    run: D1RunSpec,
    protocol: D1ExecutionProtocol,
    for_resume: bool = False,
) -> dict[str, object]:
    """Validate exact identity and continuation state without loading or writing."""
    payload = require_mapping(state, "D1 checkpoint", TOP_LEVEL_KEYS)
    if payload["kind"] != D1_CHECKPOINT_KIND:
        raise D1CheckpointError("D1 checkpoint kind is unsupported")
    if payload["schema_version"] != D1_CHECKPOINT_SCHEMA_VERSION:
        raise D1CheckpointError("D1 checkpoint schema is unsupported; legacy guessing is forbidden")
    if payload["rung"] != "d1":
        raise D1CheckpointError("D1 checkpoint rung is not d1")
    step = require_nonnegative_int(payload["successful_step"], "successful_step")
    if step > protocol.optimization.max_steps:
        raise D1CheckpointError("successful_step exceeds the registered optimization budget")
    expected = expected_d1_checkpoint_identity(system, optimizer, sampler, run, protocol)
    identity = require_mapping(payload["identity"], "identity", set(expected))
    for key, value in expected.items():
        if identity[key] != value:
            raise D1CheckpointError(f"checkpoint {key} identity does not match the D1 run")
    _validate_typed_evidence(payload, system, run, protocol, step)
    _validate_model(payload["model"], expected, system, run, step)
    optimizer_state = require_mapping(payload["optimizer"], "optimizer", {"state", "param_groups"})
    if (
        type(optimizer_state["state"]) is not dict
        or type(optimizer_state["param_groups"]) is not list
        or not optimizer_state["param_groups"]
        or (step > 0 and not optimizer_state["state"])
    ):
        raise D1CheckpointError("optimizer state and param_groups must be complete")
    sampler_state = require_mapping(payload["sampler"], "sampler", {"generator", "eval_cursor"})
    generator = sampler_state["generator"]
    if type(generator) is not torch.Tensor or generator.ndim != 1 or generator.dtype != torch.uint8:
        raise D1CheckpointError("sampler.generator must be a rank-1 byte tensor")
    require_nonnegative_int(sampler_state["eval_cursor"], "sampler.eval_cursor")
    _validate_rng(payload["rng"])
    _validate_lifecycle(payload["lifecycle"], for_resume=for_resume)
    require_finite(payload["best_nats_per_byte"], "best_nats_per_byte", optional=True)
    return payload
