"""Single-site inverse-Fisher dynamics, separate from sentence representation."""

from typing import Literal, cast

import torch

from entropic_substrate.__legacy.predictive_state.dynamics import (
    DEFAULT_SOLVER_CONFIG,
    SolverConfig,
    SolverDiagnostics,
    euclidean_residual,
    inverse_fisher_residual,
)
from entropic_substrate.__legacy.predictive_state.metric import numerical_pair
from entropic_substrate.core.sentence.head import SentenceLawHead
from entropic_substrate.core.sentence.numerics import (
    InterfaceMode,
    checked_coordinates,
    coordinate_map,
)

type ConsumerMode = Literal["isotropic", "fisher"]


class SentenceConsumer:
    """A per-forward strategy, not a second owner of the law head's parameters.

    The spine passes the final block's original proposal. In spherical mode
    its chord is split into radial and tangent components; only the latter is
    transformed. Composing the feature map with unit direction restricts its
    pullback to the tangent plane. No target sentence enters this operation.
    """

    def __init__(
        self,
        head: SentenceLawHead,
        carrier: InterfaceMode,
        mode: ConsumerMode,
        config: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> None:
        if not isinstance(cast(object, head), SentenceLawHead) or type(config) is not SolverConfig:
            raise TypeError("sentence consumer requires a law head and solver config")
        if carrier not in ("sphere", "magnitude"):
            raise ValueError("unknown sentence carrier")
        if mode not in ("isotropic", "fisher"):
            raise ValueError("unknown sentence consumer")
        self.head, self.carrier, self.mode, self.config = head, carrier, mode, config
        self.diagnostics: SolverDiagnostics | None = None

    def __call__(self, current: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        self.diagnostics = None
        current, proposed = numerical_pair(current, proposed)
        with torch.autocast(device_type=current.device.type, enabled=False):
            delta = proposed - current
            radial = torch.zeros_like(delta[..., :1])
            if self.carrier == "sphere":
                for value in (current, proposed):
                    coordinates = checked_coordinates(value)
                    if bool((coordinates.log_radius.abs() > 1e-5).any()):
                        raise ValueError("spherical consumer requires unit states and proposals")
                radial = (current * delta).sum(dim=-1, keepdim=True)
                delta = delta - radial * current

            if self.mode == "isotropic":
                solved = euclidean_residual(lambda x: x, current, delta, self.config)
                self.diagnostics = solved.diagnostics
                # Preserve the exact control, including its rounding convention.
                return proposed

            # Validate the effective law outside vmap; retain no law/solve graph
            # on this observer. Only detached numerical diagnostics survive.
            self.head(current)
            parameters = self.head.metric_parameters()

            def features(state: torch.Tensor, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
                if self.carrier == "sphere":
                    state = coordinate_map(state).direction
                return self.head.features_with_parameters(state, values)

            solved = inverse_fisher_residual(
                lambda x: features(x, parameters),
                current,
                delta,
                self.config,
                parameterized_mean_map=features,
                mean_parameters=parameters,
            )
            result = current + solved.update
            if self.carrier == "sphere":
                result = checked_coordinates(result + radial * current).direction
            self.diagnostics = solved.diagnostics
            return result
