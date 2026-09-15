"""Reconstruction and prior-conditioned training have distinct evidence meanings."""

import pytest
import torch
from tests.test_sentence_model import history, tiny_model

from entropic_substrate.sentence_prediction.codec import teacher_forcing_prefix
from entropic_substrate.sentence_prediction.objective import codec_warmup_loss, prediction_loss

pytestmark = pytest.mark.unit


def test_teacher_forcing_never_places_terminal_controls_inside_prefixes() -> None:
    targets = torch.tensor([[65, 256, 259, 259], [66, 67, 46, 257]])
    prefix = teacher_forcing_prefix(targets, 16)
    assert prefix.tolist() == [[258, 65, 259, 259], [258, 66, 67, 46]]
    with pytest.raises(ValueError, match="boundary"):
        teacher_forcing_prefix(torch.tensor([[65, 66]]), 16)


def test_warmup_uses_oracle_reconstruction_without_spine_or_future_law() -> None:
    model = tiny_model()
    result = codec_warmup_loss(model, torch.tensor([[65, 46, 256], [66, 257, 259]]))
    assert result.meaning == "oracle_reconstruction_training"
    assert result.future_nll_per_dimension is None
    assert result.readout is None
    assert result.source_bytes == 3 and result.boundary_symbols == 2
    assert result.encoder_bytes == 3
    result.total.backward()
    assert all(p.grad is not None for p in model.codec.parameters())
    assert all(p.grad is None for p in model.spine.parameters())
    assert all(p.grad is None for p in model.head.parameters())


@pytest.mark.parametrize("weight", [0.0, 0.3])
@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_prediction_uses_prior_samples_and_optional_detached_absolute_targets(
    weight: float, interface: str
) -> None:
    model = tiny_model(interface)
    model.codec.freeze_encoder()
    target = torch.tensor([[67, 46, 257]])
    result = prediction_loss(model, history(), target, future_weight=weight)
    assert result.meaning == "prior_sample_decoder_training"
    assert result.readout is not None
    assert result.source_bytes == 2 and result.boundary_symbols == 1
    assert result.encoder_bytes == 4 + (2 if weight else 0)
    if weight:
        assert result.future_nll_per_dimension is not None
        torch.testing.assert_close(
            result.total, result.decoder_ce + weight * result.future_nll_per_dimension
        )
    else:
        assert result.future_nll_per_dimension is None
        torch.testing.assert_close(result.total, result.decoder_ce, atol=0, rtol=0)
    result.total.backward()
    assert all(p.grad is None for p in model.codec.encoder.parameters())
    assert all(p.grad is not None for p in model.head.parameters())


def test_changing_target_changes_score_but_cannot_change_states_or_prior_sample() -> None:
    model = tiny_model()
    model.codec.freeze_encoder()
    rng = torch.get_rng_state()
    first = prediction_loss(model, history(), torch.tensor([[67, 46, 257]]), future_weight=0.3)
    torch.set_rng_state(rng)
    second = prediction_loss(model, history(), torch.tensor([[99, 33, 257]]), future_weight=0.3)
    torch.testing.assert_close(
        first.readout.prediction.states, second.readout.prediction.states, atol=0, rtol=0
    )
    torch.testing.assert_close(first.readout.sample, second.readout.sample, atol=0, rtol=0)
    assert not torch.allclose(first.total, second.total)


def test_disabled_future_objective_does_not_visit_the_target_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model()
    model.codec.freeze_encoder()

    def forbidden(*args: object) -> None:
        raise AssertionError("disabled target encoder was used")

    monkeypatch.setattr(model.codec, "posterior", forbidden)
    prediction_loss(model, history(), torch.tensor([[67, 257]]), future_weight=0.0)


def test_training_phase_cannot_silently_unfreeze_or_hide_frozen_codec() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="frozen"):
        prediction_loss(model, history(), torch.tensor([[67, 257]]), future_weight=0.0)
    model.codec.freeze_encoder()
    with pytest.raises(ValueError, match="warm-up"):
        codec_warmup_loss(model, torch.tensor([[67, 257]]))


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True])
def test_invalid_auxiliary_weights_are_refused(weight: float) -> None:
    model = tiny_model()
    model.codec.freeze_encoder()
    with pytest.raises(ValueError, match="weight"):
        prediction_loss(model, history(), torch.tensor([[67, 257]]), future_weight=weight)
