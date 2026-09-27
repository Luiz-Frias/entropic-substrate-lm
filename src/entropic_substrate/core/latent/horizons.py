"""D3 fixed-horizon targets and equal-weight Gaussian future-law composite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from entropic_substrate.core.latent.future_law import (
    DiagonalGaussianFutureLaw,
    FutureLaw,
    FutureLawLoss,
    MeanDisplacementBaseline,
)
from entropic_substrate.core.latent.state import LatentTrajectory
from entropic_substrate.core.latent.target_values import FutureTarget, StabilizedTargets
from entropic_substrate.core.observation.contracts import ObservationAlignment

REGISTERED_HORIZONS = (1, 4, 16, 64)


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if (
        type(horizons) is not tuple
        or not horizons
        or any(type(h) is not int or h not in REGISTERED_HORIZONS for h in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        msg = "horizons must be unique registered source-byte horizons from (1, 4, 16, 64)"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AlignedFutureTarget:
    """Future displacement with the source intervals needed to audit its horizon.

    Stabilization remains owned by the target-update strategy. Wrap its output
    with the same alignment before scoring to retain detach/EMA semantics.
    """

    target: FutureTarget | StabilizedTargets
    alignment: ObservationAlignment

    def __post_init__(self) -> None:
        if type(self.target) not in (FutureTarget, StabilizedTargets):
            msg = "aligned target requires FutureTarget or StabilizedTargets"
            raise TypeError(msg)
        if type(self.alignment) is not ObservationAlignment:
            msg = "aligned target requires ObservationAlignment"
            raise TypeError(msg)
        _validate_horizons((self.alignment.horizon,))
        if self.target.values.shape[1] != self.alignment.online_observations.length:
            msg = "target time must match its aligned online source span"
            raise ValueError(msg)


def horizon_displacement(
    target_states: LatentTrajectory, alignment: ObservationAlignment
) -> AlignedFutureTarget:
    """Construct ``xi[t+H] - xi[t]`` from exactly the declared ``T+H`` context.

    The context belongs to the target encoder only. The returned target has
    exactly T positions; it cannot be used to extend the online observation span.
    No detach occurs here: the caller's target-update strategy owns that choice.
    """
    if type(target_states) is not LatentTrajectory:
        msg = "horizon displacement requires LatentTrajectory"
        raise TypeError(msg)
    if type(alignment) is not ObservationAlignment:
        msg = "horizon displacement requires ObservationAlignment"
        raise TypeError(msg)
    _validate_horizons((alignment.horizon,))
    if target_states.time != alignment.target_encoder_context.length:
        msg = "target context length must match the declared horizon alignment exactly"
        raise ValueError(msg)
    with torch.autocast(device_type=target_states.values.device.type, enabled=False):
        values = target_states.values.float()
        displacement = values[:, alignment.horizon :] - values[:, : -alignment.horizon]
    target = FutureTarget(displacement, target_states.codec_identity)
    return AlignedFutureTarget(target, alignment)


@dataclass(frozen=True, slots=True)
class HorizonLawLoss:
    """Equal-weight per-dimension NLL plus each horizon's full grounding readout."""

    nll: torch.Tensor
    per_horizon: Mapping[int, FutureLawLoss]


class HorizonFutureLaw(nn.Module):
    """One existing nonlinear fixed-unit-variance Gaussian head per horizon.

    A single-horizon roster is the fixed-horizon arm. The four-head roster is
    the preregistered equal-weight composite, with no learned horizon weights.
    Each head is exposed in ``heads`` so parameter inventories and optimizer
    ownership include every trainable predictor.
    """

    def __init__(self, *, latent_width: int, hidden_width: int, horizons: tuple[int, ...]) -> None:
        super().__init__()
        _validate_horizons(horizons)
        self.horizons = horizons
        self.heads = nn.ModuleDict(
            {
                str(h): DiagonalGaussianFutureLaw(
                    latent_width=latent_width, hidden_width=hidden_width
                )
                for h in horizons
            }
        )

    def forward(self, states: LatentTrajectory) -> dict[int, FutureLaw]:
        """Predict all laws from the same causal online states."""
        return {
            h: cast(DiagonalGaussianFutureLaw, self.heads[str(h)])(states) for h in self.horizons
        }

    def loss(
        self,
        states: LatentTrajectory,
        targets: Mapping[int, AlignedFutureTarget],
        *,
        mean_baselines: Mapping[int, MeanDisplacementBaseline],
    ) -> HorizonLawLoss:
        """Score aligned targets using separately training-fitted mean baselines."""
        if set(targets) != set(self.horizons) or set(mean_baselines) != set(self.horizons):
            msg = "target and mean-baseline roster must exactly match the horizon roster"
            raise ValueError(msg)
        reference_span = targets[self.horizons[0]].alignment.online_observations
        losses: dict[int, FutureLawLoss] = {}
        for horizon in self.horizons:
            aligned = targets[horizon]
            if aligned.alignment.horizon != horizon:
                msg = "target alignment horizon must match its registered head"
                raise ValueError(msg)
            if aligned.alignment.online_observations != reference_span:
                msg = "all horizon targets must share one online source span"
                raise ValueError(msg)
            head = cast(DiagonalGaussianFutureLaw, self.heads[str(horizon)])
            losses[horizon] = head.loss(
                states, aligned.target, mean_baseline=mean_baselines[horizon]
            )
        nll = torch.stack([losses[h].nll for h in self.horizons]).mean()
        return HorizonLawLoss(nll, losses)
