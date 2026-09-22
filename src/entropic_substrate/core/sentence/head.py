"""Trainable future-sentence law and its exact fixed-dispersion Fisher map."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.core.latent.metric import finite_tensor
from entropic_substrate.core.sentence.law import DirectionalMagnitudeLaw, LawSpec
from entropic_substrate.core.sentence.numerics import (
    UndefinedDirection,
    checked_coordinates,
    coordinate_map,
)


class SentenceLawHead(nn.Module):
    """Read raw carrier values without overwriting them or creating dormant capacity.

    Direction is a unit mean parameter, not the returned model state. Fixed law
    dispersion makes its pullback exactly J(features)^T J(features), where the
    direction and optional log-radius have their law-specific Fisher weights.
    """

    def __init__(self, input_width: int, spec: LawSpec) -> None:
        super().__init__()
        if type(input_width) is not int or input_width < 2 or type(spec) is not LawSpec:
            raise ValueError("law head requires an integer width >= 2 and a LawSpec")
        self.input_width, self.spec = input_width, spec
        self.direction = nn.Linear(input_width, spec.width)
        self.radius = nn.Linear(input_width, 1) if spec.mode == "magnitude" else None

    def metric_parameters(self) -> tuple[torch.Tensor, ...]:
        """Stable functional order, including biases for mixed solve derivatives."""
        result = (self.direction.weight, self.direction.bias)
        if self.radius is not None:
            result += (self.radius.weight, self.radius.bias)
        return result

    def parameters_with_values(
        self, states: torch.Tensor, parameters: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Transform-safe parameters; data-dependent validation belongs in forward.

        This accepts explicit parameter values so higher-order differentiation
        cannot accidentally close over detached coefficients at zero residuals.
        """
        if states.ndim < 1 or states.shape[-1] != self.input_width:
            raise ValueError("law head state width mismatch")
        if not states.is_floating_point():
            raise TypeError("law head states must be floating")
        if len(parameters) != (4 if self.radius is not None else 2):
            raise ValueError("law head parameter roster mismatch")
        with torch.autocast(device_type=states.device.type, enabled=False):
            dtype = torch.float64 if states.dtype == torch.float64 else torch.float32
            values = states.to(dtype)
            raw = F.linear(values, parameters[0].to(dtype), parameters[1].to(dtype))
            direction = coordinate_map(raw).direction
            radius = None
            if self.radius is not None:
                radius = F.linear(values, parameters[2].to(dtype), parameters[3].to(dtype)).squeeze(
                    -1
                )
            return direction, radius

    def features_with_parameters(
        self, states: torch.Tensor, parameters: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Exact effective-law Fisher features, not a substitute latent carrier."""
        direction, radius = self.parameters_with_values(states, parameters)
        with torch.autocast(device_type=states.device.type, enabled=False):
            weighted = math.sqrt(self.spec.fisher_factor) * direction
            if radius is None:
                return weighted
            return torch.cat((weighted, (radius / self.spec.log_radius_std).unsqueeze(-1)), dim=-1)

    def forward(self, states: torch.Tensor) -> DirectionalMagnitudeLaw:
        finite_tensor(states, "law_head_state")
        direction, radius = self.parameters_with_values(states, self.metric_parameters())
        coordinates = checked_coordinates(direction, allow_zero=True)
        if bool(coordinates.zero_mask.any()):
            raise UndefinedDirection(int(coordinates.zero_mask.sum()))
        return DirectionalMagnitudeLaw(self.spec, direction, radius)
