"""Importance weighting distinguishes predictive probability from reconstruction."""

import math

import pytest
import torch
from tests.test_sentence_model import history, tiny_model

from entropic_substrate.sentence_prediction.scoring import (
    importance_evaluate,
    importance_statistics,
)

pytestmark = pytest.mark.unit


def test_logspace_statistics_retain_tiny_weights_and_report_likelihood_uncertainty() -> None:
    values = torch.tensor([[-10000.0], [-10001.0], [-10002.0], [-10003.0]], dtype=torch.float64)
    result = importance_statistics(values)[0]
    expected = float(torch.logsumexp(values[:, 0], 0) - math.log(4))
    assert result.log_estimate == expected and result.gap is None
    normalized = values[:, 0].softmax(0)
    assert result.effective_sample_size == pytest.approx(float(1 / normalized.square().sum()))
    assert result.likelihood_relative_mcse > 0


def test_zero_weight_mass_is_an_explicit_gap_not_a_fabricated_score() -> None:
    result = importance_statistics(torch.full((4, 1), -torch.inf))[0]
    assert result.gap == "zero_importance_weight_mass"
    assert result.log_estimate is None and result.effective_sample_size is None
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            importance_statistics(torch.full((4, 1), value))


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_uniform_decoder_and_equal_prior_proposal_match_exact_text_probability(
    interface: str,
) -> None:
    model = tiny_model(interface).eval()
    model.codec.freeze_encoder()
    target = torch.tensor([[67, 46, 257]])
    with torch.no_grad():
        proposal = model.codec.posterior(target)
        model.codec.decoder.readout.weight.zero_()
        model.head.direction.weight.zero_()
        model.head.direction.bias.copy_(proposal.direction[0])
        if model.head.radius is not None:
            model.head.radius.weight.zero_()
            model.head.radius.bias.copy_(proposal.log_radius_mean)
    result = importance_evaluate(
        model, history(), target, sample_counts=(4, 8), sample_chunk_size=2
    )
    assert result.meaning == "importance_weighted_log_marginal_estimate"
    assert result.log_weights.shape == (8, 1)
    assert not result.log_weights.requires_grad
    assert result.source_bytes == (2,) and result.boundary_symbols == (1,)
    for count, rows in result.by_sample_count.items():
        assert rows[0].log_estimate == pytest.approx(-3 * math.log(258), abs=1e-5)
        assert rows[0].effective_sample_size == pytest.approx(count, abs=1e-5)


def test_target_proposal_cannot_change_prior_states_and_model_is_never_updated() -> None:
    model = tiny_model().eval()
    model.codec.freeze_encoder()
    before = [p.detach().clone() for p in model.parameters()]
    rng = torch.get_rng_state()
    first = importance_evaluate(
        model, history(), torch.tensor([[67, 257]]), sample_counts=(4,), sample_chunk_size=1
    )
    torch.set_rng_state(rng)
    second = importance_evaluate(
        model, history(), torch.tensor([[99, 257]]), sample_counts=(4,), sample_chunk_size=1
    )
    torch.testing.assert_close(first.prior_states, second.prior_states, atol=0, rtol=0)
    assert not torch.allclose(first.log_weights, second.log_weights)
    for parameter, expected in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        assert parameter.grad is None


def test_importance_correction_is_not_omitted_for_a_nonmatching_proposal() -> None:
    model = tiny_model().eval()
    model.codec.freeze_encoder()
    result = importance_evaluate(
        model, history(), torch.tensor([[67, 257]]), sample_counts=(4,), sample_chunk_size=2
    )
    torch.testing.assert_close(
        result.log_weights,
        result.decoder_log_prob + result.prior_log_prob - result.proposal_log_prob,
    )
    assert not torch.allclose(result.log_weights, result.decoder_log_prob)


def test_chunking_changes_memory_batching_not_the_scored_samples() -> None:
    model = tiny_model().eval()
    model.codec.freeze_encoder()
    target = torch.tensor([[67, 46, 257]])
    rng = torch.get_rng_state()
    first = importance_evaluate(model, history(), target, sample_counts=(4, 8), sample_chunk_size=1)
    torch.set_rng_state(rng)
    second = importance_evaluate(
        model, history(), target, sample_counts=(4, 8), sample_chunk_size=4
    )
    torch.testing.assert_close(first.log_weights, second.log_weights, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("counts", [(), (1,), (4, 2), (True,), (4, 4)])
def test_invalid_sampling_envelopes_are_refused(counts: tuple[int, ...]) -> None:
    model = tiny_model().eval()
    model.codec.freeze_encoder()
    with pytest.raises(ValueError, match="sample"):
        importance_evaluate(
            model, history(), torch.tensor([[67, 257]]), sample_counts=counts, sample_chunk_size=1
        )


def test_scoring_requires_a_frozen_eval_construction() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="evaluation"):
        importance_evaluate(
            model, history(), torch.tensor([[67, 257]]), sample_counts=(4,), sample_chunk_size=1
        )


def test_prior_and_proposal_batches_cannot_broadcast_into_different_examples() -> None:
    model = tiny_model().eval()
    model.codec.freeze_encoder()
    with pytest.raises(ValueError, match="batch"):
        importance_evaluate(
            model,
            history().repeat(2, 1, 1),
            torch.tensor([[67, 257]]),
            sample_counts=(4,),
            sample_chunk_size=1,
        )
