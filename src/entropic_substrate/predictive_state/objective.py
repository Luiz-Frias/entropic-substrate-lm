"""D1 objective value contract without the legacy auxiliary-loss channel."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["PredictiveObjectiveTerms"]


@dataclass(frozen=True, slots=True)
class PredictiveObjectiveTerms:
    """Keep language, future, variance, and covariance terms independently visible."""

    language: torch.Tensor
    future: torch.Tensor
    variance: torch.Tensor
    covariance: torch.Tensor

    def __post_init__(self) -> None:
        terms = (self.language, self.future, self.variance, self.covariance)
        for name, term in zip(("language", "future", "variance", "covariance"), terms, strict=True):
            if type(term) is not torch.Tensor or term.ndim != 0 or not term.is_floating_point():
                msg = f"{name} objective term must be a floating scalar tensor"
                raise ValueError(msg)
            if term.device != self.language.device:
                msg = "all predictive objective terms must share one device"
                raise ValueError(msg)

    @property
    def total(self) -> torch.Tensor:
        """FP32 sum of the separately reported terms; no ``aux_loss`` alias exists."""
        with torch.autocast(device_type=self.language.device.type, enabled=False):
            return (
                self.language.float()
                + self.future.float()
                + self.variance.float()
                + self.covariance.float()
            )
