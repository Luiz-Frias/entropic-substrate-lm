"""Fail-loud numerical boundaries for D1 losses and collapse evidence."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.latent.future_law import FutureLaw, FutureLawLoss
from entropic_substrate.core.latent.objective import PredictiveObjectiveTerms
from entropic_substrate.core.latent.stabilization import (
    CollapseDiagnostics,
    CollapseResult,
)
from predictive_state_helpers import codec_identity

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_future_law_refuses_nonfinite_means(nonfinite: float) -> None:
    values = torch.zeros(1, 2, 3, dtype=torch.float32)
    values[0, 0, 0] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        FutureLaw(mean=values, codec_identity=codec_identity())


@pytest.mark.parametrize("field", ["nll", "residual_rms", "coverage_80"])
def test_future_law_loss_refuses_a_nonfinite_scalar(field: str) -> None:
    values: dict[str, torch.Tensor] = {
        "nll": torch.tensor(1.0),
        "zero_displacement_nll": torch.tensor(1.0),
        "mean_displacement_nll": torch.tensor(1.0),
        "residual_mean": torch.tensor(0.0),
        "residual_rms": torch.tensor(1.0),
        "residual_variance": torch.tensor(1.0),
        "coverage_50": torch.tensor(0.5),
        "coverage_80": torch.tensor(0.8),
        "coverage_95": torch.tensor(0.95),
    }
    values[field] = torch.tensor(float("nan"))

    with pytest.raises(ValueError, match="finite"):
        FutureLawLoss(**values, mean=torch.zeros(1, 2, 3))


@pytest.mark.parametrize("field", ["language", "future", "variance", "covariance"])
def test_predictive_objective_refuses_nonfinite_terms(field: str) -> None:
    values = {name: torch.tensor(1.0) for name in ("language", "future", "variance", "covariance")}
    values[field] = torch.tensor(float("inf"))

    with pytest.raises(ValueError, match="finite"):
        PredictiveObjectiveTerms(**values)


def _diagnostics(*, corrupt: str | None = None) -> CollapseDiagnostics:
    variance = torch.ones(3, dtype=torch.float32)
    covariance = torch.eye(3, dtype=torch.float32)
    eigenvalues = torch.ones(3, dtype=torch.float32)
    if corrupt == "variance":
        variance[0] = float("nan")
    elif corrupt == "covariance":
        covariance[0, 0] = float("inf")
    elif corrupt == "eigenvalues":
        eigenvalues[0] = float("-inf")
    return CollapseDiagnostics(
        variance=variance,
        std=torch.ones(3, dtype=torch.float32),
        covariance=covariance,
        covariance_eigenvalues=eigenvalues,
        effective_rank=3.0,
        numerical_rank=3,
    )


@pytest.mark.parametrize("field", ["variance", "covariance", "eigenvalues"])
def test_collapse_diagnostics_refuse_nonfinite_tensors(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        _diagnostics(corrupt=field)


def test_collapse_result_refuses_nonfinite_penalty() -> None:
    with pytest.raises(ValueError, match="finite"):
        CollapseResult(
            raw_variance_penalty=torch.tensor(0.0),
            weighted_variance_penalty=torch.tensor(float("nan")),
            raw_covariance_penalty=torch.tensor(0.0),
            weighted_covariance_penalty=torch.tensor(0.0),
            total=torch.tensor(0.0),
            diagnostics=_diagnostics(),
        )
