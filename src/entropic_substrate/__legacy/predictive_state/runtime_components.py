"""Narrow internal components shared by the D1 runtime composition root."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from entropic_substrate.observation import CodecIdentity, LearnedObservationAdapter
from entropic_substrate.predictive_state.future_law import FutureLawLoss
from entropic_substrate.predictive_state.objective import PredictiveObjectiveTerms
from entropic_substrate.predictive_state.stabilization import CollapseResult
from entropic_substrate.predictive_state.state import CausalStateEncoder, LatentTrajectory
from entropic_substrate.predictive_state.target_values import StabilizedTargets


class D1StateBranch(nn.Module):
    """One independently-owned observation-adapter plus causal-encoder branch."""

    def __init__(self, *, adapter: LearnedObservationAdapter, encoder: CausalStateEncoder) -> None:
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder

    def forward(self, observation_ids: torch.Tensor, codec: CodecIdentity) -> LatentTrajectory:
        return self.encoder(self.adapter(observation_ids, codec))

    def parameter_signature(self) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
        """Stable structural signature used to attest the matched pair."""
        return tuple(
            (name, tuple(parameter.shape), parameter.dtype)
            for name, parameter in self.named_parameters()
        )


@dataclass(frozen=True, slots=True)
class D1PredictiveResult:
    """Explicit D1 objective terms and all reportable runtime diagnostics."""

    language_logits: torch.Tensor
    online_states: LatentTrajectory
    target: StabilizedTargets
    future_law_loss: FutureLawLoss
    collapse: CollapseResult
    objective: PredictiveObjectiveTerms

    def __post_init__(self) -> None:
        if self.language_logits.ndim != 3:
            msg = "D1 language logits must be rank-3"
            raise ValueError(msg)
        if self.language_logits.shape[:2] != self.online_states.values.shape[:2]:
            msg = "D1 language logits must align with online latent states"
            raise ValueError(msg)
