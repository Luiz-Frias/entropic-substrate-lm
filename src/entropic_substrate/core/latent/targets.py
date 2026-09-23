"""Compatibility facade for D1 target values, schedules, and update policies."""

from entropic_substrate.core.latent.ema_schedules import (
    CosineEmaDecaySchedule,
    EmaDecaySchedule,
    FixedEmaDecaySchedule,
)
from entropic_substrate.core.latent.target_updates import (
    OnlineTargetUpdate,
    OptimizationStepCommitted,
    StopGradientEmaUpdate,
    TargetUpdatePolicy,
)
from entropic_substrate.core.latent.target_values import (
    FutureTarget,
    StabilizedTargets,
    one_step_displacement,
    one_step_target_from_context,
)

__all__ = [
    "CosineEmaDecaySchedule",
    "EmaDecaySchedule",
    "FixedEmaDecaySchedule",
    "FutureTarget",
    "OnlineTargetUpdate",
    "OptimizationStepCommitted",
    "StabilizedTargets",
    "StopGradientEmaUpdate",
    "TargetUpdatePolicy",
    "one_step_displacement",
    "one_step_target_from_context",
]
