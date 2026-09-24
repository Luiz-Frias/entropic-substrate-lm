"""D1 AdamW composition root with exact ownership attestation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.register.d1_execution import D1OptimizationSpec
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem

__all__ = ["D1OptimizerBundle", "build_d1_optimizer"]


@dataclass(frozen=True, slots=True)
class D1OptimizerBundle:
    """Materialized optimizer plus stable spec and ownership identities."""

    optimizer: torch.optim.AdamW
    spec_digest: str
    ownership_manifest: dict[str, object]

    @property
    def ownership_digest(self) -> str:
        return canonical_digest(self.ownership_manifest)

    def set_learning_rate(self, value: float) -> None:
        """Apply the registered schedule output to every fixed ownership group."""
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0.0:
            msg = "D1 optimizer learning rate must be a finite positive scalar"
            raise ValueError(msg)
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)


def _parameter_manifest(
    named: tuple[tuple[str, nn.Parameter], ...],
) -> list[dict[str, object]]:
    return [
        {
            "dtype": str(parameter.dtype),
            "name": name,
            "numel": parameter.numel(),
            "shape": list(parameter.shape),
        }
        for name, parameter in named
    ]


def build_d1_optimizer(
    system: D1PredictiveSystem,
    spec: D1OptimizationSpec,
) -> D1OptimizerBundle:
    """Build the only D1 optimizer and validate its exact runtime ownership."""
    if type(system) is not D1PredictiveSystem:
        msg = "D1 optimizer factory requires a D1PredictiveSystem"
        raise TypeError(msg)
    if type(spec) is not D1OptimizationSpec:
        msg = "D1 optimizer factory requires a D1OptimizationSpec"
        raise TypeError(msg)
    if spec != system.draft.execution_protocol.optimization:
        msg = "D1 optimizer spec does not match the runtime construction"
        raise ValueError(msg)
    trainable = tuple(
        (name, parameter)
        for name, parameter in system.named_parameters()
        if parameter.requires_grad
    )
    if not trainable:
        msg = "D1 runtime exposes no trainable optimizer parameters"
        raise ValueError(msg)
    decay = tuple(item for item in trainable if item[1].ndim >= 2)
    no_decay = tuple(item for item in trainable if item[1].ndim < 2)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in decay],
                "weight_decay": float(spec.weight_decay),
            },
            {
                "params": [parameter for _, parameter in no_decay],
                "weight_decay": 0.0,
            },
        ],
        lr=float(spec.learning_rate),
        betas=(float(spec.betas[0]), float(spec.betas[1])),
        eps=float(spec.epsilon),
    )
    system.validate_optimizer_ownership(optimizer)
    ownership_manifest: dict[str, object] = {
        "decay_parameter_manifest": _parameter_manifest(decay),
        "decay_parameters": [name for name, _ in decay],
        "no_decay_parameter_manifest": _parameter_manifest(no_decay),
        "no_decay_parameters": [name for name, _ in no_decay],
        "schema_version": "d1-optimizer-ownership-v1",
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "weight_decay_policy": spec.weight_decay_policy.value,
    }
    return D1OptimizerBundle(
        optimizer=optimizer,
        spec_digest=canonical_digest(spec.manifest),
        ownership_manifest=ownership_manifest,
    )
