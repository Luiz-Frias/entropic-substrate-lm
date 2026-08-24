"""D1 objective-term visibility contract."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.predictive_state import PredictiveObjectiveTerms

pytestmark = pytest.mark.unit


def test_predictive_objective_terms_keep_language_future_and_moments_separate() -> None:
    terms = PredictiveObjectiveTerms(
        language=torch.tensor(1.0),
        future=torch.tensor(2.0),
        variance=torch.tensor(3.0),
        covariance=torch.tensor(4.0),
    )
    assert terms.total.item() == 10.0
    assert not hasattr(terms, "aux_loss")
    assert terms.language.dtype == torch.float32 and terms.future.dtype == torch.float32
