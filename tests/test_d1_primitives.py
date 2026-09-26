"""Feature-owned tests for D1 causal encoder and boundary contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.kernel.identity.d1_primitives import (
    CausalEncoderSpec,
    D1BoundaryStateSpec,
    EncoderInitializationIdentity,
    FutureActivation,
    MatchedEncoderPairSpec,
)
from entropic_substrate.ladder.register.d1_stabilization import FutureLawSpec

pytestmark = pytest.mark.unit


def _initialization() -> EncoderInitializationIdentity:
    return EncoderInitializationIdentity(
        name="d1-causal-random-v1",
        construction_digest="5" * 64,
    )


def _encoder(*, latent_width: int = 64, n_layers: int = 2) -> CausalEncoderSpec:
    return CausalEncoderSpec(
        latent_width=latent_width,
        n_layers=n_layers,
        n_heads=4,
        mlp_ratio=4.0,
        block_size=256,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=_initialization(),
    )


def test_encoder_construction_binds_factory_relevant_architecture() -> None:
    value = _encoder()

    assert value.manifest == {
        "bias": False,
        "block_size": 256,
        "dropout": 0.0,
        "initialization": _initialization().manifest,
        "latent_width": 64,
        "mlp_ratio": 4.0,
        "n_heads": 4,
        "n_layers": 2,
        "rms_epsilon": 1e-5,
    }
    with pytest.raises(ValueError, match="divisible"):
        replace(value, n_heads=3)
    with pytest.raises(ValueError, match="even"):
        replace(value, latent_width=60)
    with pytest.raises(ValueError, match="exactly match"):
        MatchedEncoderPairSpec(online=value, target=_encoder(n_layers=3))


def test_d1_family_binds_boundary_and_future_law_choices() -> None:
    law = FutureLawSpec()
    boundary = D1BoundaryStateSpec()

    assert law.predictor_hidden_width == 64
    assert law.activation is FutureActivation.SILU
    assert boundary == D1BoundaryStateSpec()
    assert boundary.online_target_observation_adapters_tied is False
    assert boundary.language_readout_tied_to_observation_adapter is False
    assert "cardinality" not in boundary.manifest
    with pytest.raises(ValueError, match="untied"):
        D1BoundaryStateSpec(online_target_observation_adapters_tied=True)
    with pytest.raises(ValueError, match="SiLU"):
        FutureLawSpec(activation=FutureActivation.GELU)
