"""One optimizer/evaluation path shared by tiny and admitted scientific runs."""

from __future__ import annotations

import math
from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch

from entropic_substrate.core.corpus.batching import SequenceSampler
from entropic_substrate.core.corpus.ladder_batching import paired_predictive_batch
from entropic_substrate.core.corpus.ladder_source_order import SourceOrderAudit
from entropic_substrate.core.latent.target_updates import OptimizationStepCommitted
from entropic_substrate.historical.d_ladder.d_ladder_resources import LadderResources
from entropic_substrate.ladder.capture.resource.resource_meter_models import ResourcePhase
from entropic_substrate.ladder.register.d1_execution import D1AmpPolicy, D1ExecutionProtocol
from entropic_substrate.ladder.run.construction.ladder_runtime import (
    LadderPredictiveResult,
    LadderPredictiveSystem,
)
from entropic_substrate.ladder.run.trainer.d1_training_step import d1_learning_rate


def ladder_optimizer(
    system: LadderPredictiveSystem, protocol: D1ExecutionProtocol
) -> torch.optim.AdamW:
    """Honor the existing matrix-only AdamW decay contract."""
    config = protocol.optimization
    parameters = system.optimizer_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": config.weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
    )
    system.validate_optimizer_ownership(optimizer)
    return optimizer


def gradient_measurements(system: LadderPredictiveSystem) -> dict[str, object]:
    measured: dict[str, object] = {
        name: {"status": reason, "norm": None}
        for name, reason in system.gradient_absence_reasons().items()
    }
    for name, parameters in system.optimizer_roles().items():
        if any(parameter.grad is None for parameter in parameters):
            raise ValueError(f"missing required gradient: {name}")
        gradients = [
            parameter.grad.detach().float()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise ValueError(f"nonfinite gradient: {name}")
        norm = math.sqrt(sum(float(gradient.double().square().sum()) for gradient in gradients))
        measured[name] = {
            "status": "measured_zero" if norm == 0 else "measured_nonzero",
            "norm": norm,
        }
    return measured


def result_metrics(result: LadderPredictiveResult) -> dict[str, object]:
    states = result.online_states.values.detach().float().flatten(0, 1)
    metrics: dict[str, object] = {
        "language_nats": float(result.objective.language.detach()),
        "total_loss": float(result.total_loss.detach()),
        "online_numerical_rank": int(torch.linalg.matrix_rank(states - states.mean(0))),
        "online_mean_variance": float(states.var(0, unbiased=False).mean()),
        "online_zero_norm_positions": int((states.norm(dim=-1) == 0).sum()),
        "future_nll_by_horizon": {
            str(h): float(loss.nll.detach()) for h, loss in result.losses_by_horizon.items()
        },
        "absence_reasons": dict(result.absence_reasons),
    }
    for name, value in result.metrics.items():
        detached = value.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError(f"nonfinite mechanism diagnostic: {name}")
        metrics[name] = detached.tolist()
    for name, attribute in (
        ("future_nll", "nll"),
        ("future_zero_nll", "zero_displacement_nll"),
        ("future_mean_nll", "mean_displacement_nll"),
    ):
        by_horizon = {
            str(h): float(getattr(loss, attribute).detach())
            for h, loss in result.losses_by_horizon.items()
        }
        metrics[name + "_by_horizon"] = by_horizon
        if by_horizon:
            metrics[name] = sum(by_horizon.values()) / len(by_horizon)
    return metrics


def resource_phase(
    resources: LadderResources | None, phase: ResourcePhase
) -> AbstractContextManager[None]:
    """Measure ``phase`` when a meter is present; a plain null context otherwise.

    ``resources is None`` is a real state (smoke runs carry no meter) that
    ``run_ladder`` distinguishes from scientific mode, so it is never replaced
    by a stand-in meter.
    """
    return nullcontext() if resources is None else resources.measure(phase)


def train_ladder_step(
    system: LadderPredictiveSystem,
    optimizer: torch.optim.Optimizer,
    sampler: SequenceSampler,
    protocol: D1ExecutionProtocol,
    step: int,
    resources: LadderResources | None,
    source_order: SourceOrderAudit | None = None,
) -> tuple[dict[str, object], int, int]:
    """Finite zeros commit normally; defects preserve EMA and ownership ordering."""
    system.validate_optimizer_ownership(optimizer)
    system.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = d1_learning_rate(protocol.optimization, successful_step=step)
    phase = resource_phase(resources, ResourcePhase.TRAIN)
    with phase:
        batch = paired_predictive_batch(
            sampler, max(system.spec.horizons), reservation=system.spec.sampling_horizon_reservation
        )
        if source_order is not None:
            source_order.record(tuple(batch.source_starts.tolist()))
        with torch.autocast(
            device_type=batch.online_observations.device.type,
            dtype=torch.bfloat16,
            enabled=protocol.optimization.amp is D1AmpPolicy.BF16,
        ):
            result = system(batch)
        online = int(result.metrics["online_source_positions"].item())
        target = int(result.metrics["target_encoder_positions"].item())
        committed = False
        try:
            if not bool(torch.isfinite(result.total_loss)):
                raise ValueError("nonfinite training objective")
            result.total_loss.backward()
            gradients = gradient_measurements(system)
            aggregate = torch.nn.utils.clip_grad_norm_(
                system.optimizer_parameters(),
                protocol.optimization.grad_clip_norm,
                error_if_nonfinite=True,
            )
            component_norms: list[float] = []
            for value in gradients.values():
                norm = cast(dict[str, object], value)["norm"]
                if norm is not None:
                    component_norms.append(cast(float, norm))
            if not math.isclose(
                float(aggregate), math.hypot(*component_norms), rel_tol=1e-5, abs_tol=1e-12
            ):
                raise ValueError("component gradients do not reconstruct the aggregate norm")
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in system.optimizer_parameters()):
                raise ValueError("nonfinite parameter after optimizer commit")
            system.after_successful_optimizer_step(OptimizationStepCommitted(step))
            committed = True
        finally:
            if resources is not None:
                resources.train(online=online, target=target, committed=committed)
    return gradients, online, target


def evaluate_ladder(
    system: LadderPredictiveSystem,
    sampler: SequenceSampler,
    protocol: D1ExecutionProtocol,
    resources: LadderResources | None,
    step: int,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Fixed-grid evaluation; retain all batch diagnostics and measured state rank."""
    system.eval()
    sampler.reset_eval()
    records: list[dict[str, object]] = []
    states: list[torch.Tensor] = []
    online = target = 0
    phase = resource_phase(resources, ResourcePhase.EVAL)
    with phase:
        for _ in range(protocol.evaluation.num_batches):
            batch = paired_predictive_batch(
                sampler,
                max(system.spec.horizons),
                reservation=system.spec.sampling_horizon_reservation,
            )
            # G requires autograd for its Jacobian action even for evaluation.
            with torch.autocast(
                device_type=batch.online_observations.device.type,
                dtype=torch.bfloat16,
                enabled=protocol.optimization.amp is D1AmpPolicy.BF16,
            ):
                result = system(batch)
            records.append(result_metrics(result))
            states.append(result.online_states.values.detach().float().flatten(0, 1))
            online += int(result.metrics["online_source_positions"].item())
            target += int(result.metrics["target_encoder_positions"].item())
    combined = torch.cat(states)
    metrics: dict[str, object] = {"absence_reasons": records[0]["absence_reasons"]}
    for name in ("language_nats", "total_loss", "future_nll", "future_zero_nll", "future_mean_nll"):
        if name not in records[0]:
            continue
        metrics[name] = sum(cast(float, record[name]) for record in records) / len(records)
    for name in ("future_nll", "future_zero_nll", "future_mean_nll"):
        field = name + "_by_horizon"
        keys = cast(dict[str, float], records[0][field])
        metrics[field] = {
            key: sum(cast(dict[str, float], record[field])[key] for record in records)
            / len(records)
            for key in keys
        }
    metrics.update(
        online_numerical_rank=int(torch.linalg.matrix_rank(combined - combined.mean(0))),
        online_mean_variance=float(combined.var(0, unbiased=False).mean()),
        online_zero_norm_positions=int((combined.norm(dim=-1) == 0).sum()),
        batches=records,
    )
    snapshot = (
        None if resources is None else resources.evaluate(online=online, target=target, step=step)
    )
    return metrics, snapshot
