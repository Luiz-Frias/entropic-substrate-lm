"""Focused structural contracts for reusable D1 runtime components."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.observation import LearnedObservationAdapter
from entropic_substrate.predictive_state import CausalStateEncoder
from entropic_substrate.predictive_state.runtime_components import D1StateBranch
from predictive_state_helpers import codec_identity, model_config

pytestmark = pytest.mark.unit


def test_runtime_state_branch_exposes_the_adapter_encoder_structural_signature() -> None:
    branch = D1StateBranch(
        adapter=LearnedObservationAdapter(vocab_size=17, latent_width=8),
        encoder=CausalStateEncoder(model_config()),
    )

    assert branch.parameter_signature() == tuple(
        (name, tuple(parameter.shape), parameter.dtype)
        for name, parameter in branch.named_parameters()
    )
    assert branch(torch.tensor([[1, 2]], dtype=torch.int64), codec_identity()).values.shape == (
        1,
        2,
        8,
    )
