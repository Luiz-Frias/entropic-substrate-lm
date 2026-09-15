"""Power Spherical direction and optional Gaussian log-magnitude future law.

The raw-vector density is with respect to Cartesian volume, including -d*rho.
The spherical density uses sphere surface measure and marginalizes magnitude.
These are conditional laws, never alphabet-sized replacement state vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from entropic_substrate.predictive_state.metric import FisherNumericalError, finite_tensor
from entropic_substrate.sentence_prediction.numerics import (
    InterfaceMode,
    checked_coordinates,
    coordinate_map,
)
from entropic_substrate.sentence_prediction.records import content_identity


@dataclass(frozen=True, slots=True)
class LawSpec:
    width: int
    mode: InterfaceMode
    mean_resultant: float = 0.9
    log_radius_std: float = 0.1

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 2:
            raise ValueError("directional law width must be an integer >= 2")
        if self.mode not in ("sphere", "magnitude"):
            raise ValueError("unknown directional law interface")
        for value in (self.mean_resultant, self.log_radius_std):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("law constants must be finite numbers")
        if not 0 < self.mean_resultant < 1 or self.log_radius_std <= 0:
            raise ValueError("invalid law concentration or log-radius scale")
        try:
            if (self.width - 1) / 2 + self.concentration <= 1:
                raise ValueError("nonregular directional law has no finite Fisher moment")
            if not all(
                math.isfinite(v)
                for v in (self.concentration, self.fisher_factor, self.log_normalizer)
            ):
                raise ValueError("law coefficients must remain finite")
        except OverflowError as error:
            raise ValueError("law coefficients must remain finite") from error

    @property
    def concentration(self) -> float:
        return self.mean_resultant * (self.width - 1) / (1 - self.mean_resultant)

    @property
    def fisher_factor(self) -> float:
        """Tangent mean-direction Fisher factor kappa²/(d-3+2*kappa).

        Derived using T=2*Beta((d-1)/2+kappa,(d-1)/2)-1 and checked
        against score second moments. A bare mean Jacobian omits this factor.
        """
        return self.concentration / (2 + (self.width - 3) / self.concentration)

    @property
    def identity(self) -> str:
        return content_identity(
            (
                "sentence-law-v1",
                str(self.width),
                self.mode,
                str(self.mean_resultant),
                str(self.log_radius_std) if self.mode == "magnitude" else "radius-marginalized",
            )
        )

    @property
    def log_normalizer(self) -> float:
        beta = (self.width - 1) / 2
        alpha = beta + self.concentration
        return (
            (alpha + beta) * math.log(2)
            + beta * math.log(math.pi)
            + math.lgamma(alpha)
            - math.lgamma(alpha + beta)
        )


@dataclass(frozen=True)
class DirectionalMagnitudeLaw:
    """Validated parameters; scalar magnitude is absent in spherical mode."""

    spec: LawSpec
    direction: torch.Tensor
    log_radius_mean: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if type(self.spec) is not LawSpec:
            raise TypeError("future law requires a LawSpec")
        coordinates = checked_coordinates(self.direction)
        if self.direction.shape[-1] != self.spec.width:
            raise ValueError("law parameter width mismatch")
        if self.direction.dtype not in (torch.float32, torch.float64):
            raise ValueError("law parameters require FP32 or FP64 oracle precision")
        if bool((coordinates.log_radius.abs() > 1e-5).any()):
            raise ValueError("law mean direction must be unit norm")
        if self.spec.mode == "sphere":
            if self.log_radius_mean is not None:
                raise ValueError("spherical law magnitude is inactive and must be absent")
        else:
            radius = self.log_radius_mean
            if not isinstance(radius, torch.Tensor) or radius.shape != self.direction.shape[:-1]:
                raise ValueError("magnitude law requires aligned scalar log-radius means")
            if radius.dtype != self.direction.dtype or radius.device != self.direction.device:
                raise ValueError("law parameter dtype/device mismatch")
            finite_tensor(radius, "log_radius_mean")

    @classmethod
    def from_vector(cls, spec: LawSpec, vector: torch.Tensor) -> DirectionalMagnitudeLaw:
        """Encoder proposal parameters derived from its vector, without replacing it."""
        if type(spec) is not LawSpec:
            raise TypeError("future law requires a LawSpec")
        coordinates = checked_coordinates(vector)
        radius = coordinates.log_radius if spec.mode == "magnitude" else None
        return cls(spec, coordinates.direction, radius)

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> torch.Tensor:
        """Reparameterize direction and active radius; checkpoint the Torch RNG.

        Concentration is fixed, so Beta noise needs no concentration derivative.
        A projected Gaussian gives a uniform tangent direction without a pole-
        singular Householder chart. Gradients through projection are retained.
        """
        if type(sample_shape) is not tuple or any(
            type(n) is not int or n < 1 for n in sample_shape
        ):
            raise ValueError("sample shape must contain positive integer dimensions")
        with torch.autocast(device_type=self.direction.device.type, enabled=False):
            shape = (*sample_shape, *self.direction.shape[:-1])
            beta = (self.spec.width - 1) / 2
            marginal = torch.distributions.Beta(
                self.direction.new_tensor(beta + self.spec.concentration),
                self.direction.new_tensor(beta),
            )
            t = 2 * marginal.sample(shape) - 1
            normal = torch.randn(
                (*shape, self.spec.width), dtype=self.direction.dtype, device=self.direction.device
            )
            tangent = normal - (normal * self.direction).sum(dim=-1, keepdim=True) * self.direction
            tangent_coordinates = checked_coordinates(tangent, allow_zero=True)
            if bool(tangent_coordinates.zero_mask.any()):
                raise FisherNumericalError("zero_tangent_noise", "directional sampling")
            tangent = tangent_coordinates.direction
            value = (
                t.unsqueeze(-1) * self.direction + (1 - t.square()).sqrt().unsqueeze(-1) * tangent
            )
            if self.log_radius_mean is not None:
                noise = torch.randn(shape, dtype=self.direction.dtype, device=self.direction.device)
                rho = self.log_radius_mean + self.spec.log_radius_std * noise
                value = rho.exp().unsqueeze(-1) * value
            finite_tensor(value, "latent_sample")
            return value

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score surface/Cartesian density, not decoder or language likelihood.

        Use log chord length for log(1+mu·u), avoiding cancellation near the
        antipode. Exact antipodes have genuine zero density (-infinity), not
        an epsilon replacement. Other nonfinite arithmetic is a defect.
        """
        coordinates = checked_coordinates(value)
        if value.shape[-1] != self.spec.width or value.device != self.direction.device:
            raise ValueError("law observation width/device mismatch")
        if coordinates.direction.dtype != self.direction.dtype:
            raise ValueError("law observation precision mismatch")
        try:
            torch.broadcast_shapes(value.shape[:-1], self.direction.shape[:-1])
        except RuntimeError as error:
            raise ValueError("law observation batch shape mismatch") from error
        if self.spec.mode == "sphere" and bool((coordinates.log_radius.abs() > 1e-5).any()):
            raise ValueError("spherical law observations must be unit vectors")
        with torch.autocast(device_type=value.device.type, enabled=False):
            chord = coordinate_map(self.direction + coordinates.direction)
            result = (
                self.spec.concentration * (2 * chord.log_radius - math.log(2))
                - self.spec.log_normalizer
            )
            if self.log_radius_mean is not None:
                rho = coordinates.log_radius
                standardized = (rho - self.log_radius_mean) / self.spec.log_radius_std
                result = (
                    result
                    - 0.5 * standardized.square()
                    - math.log(self.spec.log_radius_std)
                    - 0.5 * math.log(2 * math.pi)
                    - self.spec.width * rho
                )
            admissible = torch.isfinite(result) | (chord.zero_mask & torch.isneginf(result))
            if not bool(admissible.all()):
                raise FisherNumericalError("nonfinite_log_density", "directional/magnitude law")
            return result
