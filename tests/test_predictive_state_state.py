"""Causal latent-state and encoder-pair contracts for D1."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.latent.future_law import DiagonalGaussianFutureLaw
from entropic_substrate.core.latent.state import (
    AffineFreeRMSNorm,
    CausalStateEncoder,
    LatentTrajectory,
    TargetEncoder,
    build_encoder_pair,
)
from entropic_substrate.core.observation.contracts import EncodedObservations
from entropic_substrate.core.observation.modules import LanguageReadout
from predictive_state_helpers import encoded, model_config

pytestmark = pytest.mark.unit


def test_affine_free_rms_norm_has_no_trainable_affine_and_reduces_in_fp32() -> None:
    norm = AffineFreeRMSNorm(width=3, eps=1e-6)
    values = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = norm(values)
    expected = values.float() * values.float().square().mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    assert list(norm.parameters()) == []
    assert actual.dtype == values.dtype
    torch.testing.assert_close(actual.float(), expected, atol=2e-2, rtol=2e-2)


def test_latent_trajectory_validates_width_and_carries_codec_provenance() -> None:
    observations = encoded(width=8)
    trajectory = LatentTrajectory(
        values=observations.values, codec_identity=observations.codec_identity
    )
    assert trajectory.width == 8
    assert trajectory.codec_identity == observations.codec_identity
    with pytest.raises(ValueError, match="rank-3"):
        LatentTrajectory(values=observations.values[0], codec_identity=observations.codec_identity)


def test_causal_state_encoder_is_strictly_causal_under_suffix_mutation() -> None:
    encoder = CausalStateEncoder(model_config()).eval()
    observations = encoded(time=7)
    altered_values = observations.values.clone()
    altered_values[:, 4:] = altered_values[:, 4:] + 100.0
    altered = EncodedObservations(values=altered_values, codec_identity=observations.codec_identity)
    before, after = encoder(observations).values, encoder(altered).values
    torch.testing.assert_close(before[:, :4], after[:, :4], atol=0.0, rtol=0.0)
    assert before.shape == (2, 7, 8)


def test_language_readout_and_future_predictor_cannot_alter_causal_state() -> None:
    encoder = CausalStateEncoder(model_config()).eval()
    observations = encoded()
    readout = LanguageReadout(latent_width=8, vocab_size=17)
    predictor = DiagonalGaussianFutureLaw(latent_width=8, hidden_width=11)
    baseline = encoder(observations).values
    with torch.no_grad():
        for parameter in [*readout.parameters(), *predictor.parameters()]:
            parameter.normal_(mean=3.0, std=2.0)
    _ = readout(baseline)
    _ = predictor(encoder(observations))
    torch.testing.assert_close(encoder(observations).values, baseline, atol=0.0, rtol=0.0)


def test_encoder_rejects_encoded_observation_width_mismatch() -> None:
    with pytest.raises(ValueError, match="latent width"):
        CausalStateEncoder(model_config())(encoded(width=7))


def test_encoder_pair_starts_bit_identical_without_shared_parameter_storage() -> None:
    pair = build_encoder_pair(model_config())
    assert isinstance(pair.target, TargetEncoder)
    online, target = dict(pair.online.named_parameters()), dict(pair.target.named_parameters())
    assert online.keys() == target.keys()
    for name, online_parameter in online.items():
        assert torch.equal(online_parameter, target[name]), name
        assert online_parameter.data_ptr() != target[name].data_ptr(), name
