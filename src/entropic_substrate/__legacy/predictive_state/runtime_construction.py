"""Translate validated D1 construction values into runtime strategy objects."""

from __future__ import annotations

from entropic_substrate.config import ModelConfig
from entropic_substrate.experiments.d1_family import D1DraftSpec
from entropic_substrate.experiments.d1_primitives import (
    CollapseStrategy,
    EMAKind,
    TargetUpdateStrategy,
)
from entropic_substrate.predictive_state.ema_schedules import CosineEmaDecaySchedule
from entropic_substrate.predictive_state.stabilization import (
    MomentCollapseRegularizer,
    NoCollapseRegularizer,
)
from entropic_substrate.predictive_state.target_updates import (
    OnlineTargetUpdate,
    StopGradientEmaUpdate,
)

__all__ = [
    "collapse_regularizer_from_draft",
    "encoder_config_from_draft",
    "target_policy_from_draft",
]


def encoder_config_from_draft(draft: D1DraftSpec) -> ModelConfig:
    """Map every registered encoder field into the existing block factory."""
    spec = draft.encoder_pair.online
    return ModelConfig(
        d_model=spec.latent_width,
        n_layers=spec.n_layers,
        n_heads=spec.n_heads,
        mlp_ratio=spec.mlp_ratio,
        block_size=spec.block_size,
        dropout=spec.dropout,
        bias=spec.bias,
        tokenizer="byte",
        vocab_size=draft.codec.cardinality,
        tie_embeddings=False,
        block="baseline",
        norm="layernorm",
    )


def target_policy_from_draft(draft: D1DraftSpec) -> OnlineTargetUpdate | StopGradientEmaUpdate:
    """Construct the one target-update strategy declared by ``draft``."""
    if draft.target_update is TargetUpdateStrategy.ONLINE:
        return OnlineTargetUpdate()
    if draft.ema is None:
        msg = "stop-gradient D1 draft requires EMA parameters"
        raise ValueError(msg)
    if draft.ema.kind is EMAKind.FIXED:
        return StopGradientEmaUpdate(tau=float(draft.ema.decay))
    if draft.ema.final_decay is None or draft.ema.schedule_steps is None:
        msg = "cosine stop-gradient D1 draft requires complete EMA parameters"
        raise ValueError(msg)
    return StopGradientEmaUpdate(
        schedule=CosineEmaDecaySchedule(
            start_decay=float(draft.ema.decay),
            final_decay=float(draft.ema.final_decay),
            total_successful_commits=draft.ema.schedule_steps,
        )
    )


def collapse_regularizer_from_draft(
    draft: D1DraftSpec,
) -> NoCollapseRegularizer | MomentCollapseRegularizer:
    """Construct the explicit collapse-off control or registered moment strategy."""
    if draft.collapse_strategy is CollapseStrategy.NONE:
        return NoCollapseRegularizer()
    if draft.collapse is None:
        msg = "moment-collapse D1 draft requires collapse parameters"
        raise ValueError(msg)
    return MomentCollapseRegularizer(
        gamma=float(draft.collapse.std_floor),
        variance_weight=float(draft.collapse.variance_weight),
        covariance_weight=float(draft.collapse.covariance_weight),
        variance_eps=float(draft.collapse.variance_epsilon),
    )
