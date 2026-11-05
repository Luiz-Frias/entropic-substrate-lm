"""Importance weighting distinguishes predictive probability from reconstruction."""

import math

import pytest
import torch
from tests.test_sentence_model import history, tiny_model

from entropic_substrate.core.sentence.codec import teacher_forcing_prefix
from entropic_substrate.core.sentence.scoring import (
    NULL_LATENT_RADIUS,
    LatentUsageOperand,
    UnsupportedLatentOperand,
    derived_stream_keys,
    evaluate_proposals,
    importance_evaluate,
    importance_statistics,
    latent_usage_estimator,
    log_weight_sd,
    mixture_log_density,
    posterior_mean_vector,
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


# ---------------------------------------------------------------------------------------------
# FOR-1704 (L4d): the prior and mixture proposals, diagnostics and latent usage beside q
# ---------------------------------------------------------------------------------------------
POSTERIOR_MEAN = latent_usage_estimator(LatentUsageOperand.posterior_mean)
TWO_TARGETS = torch.tensor([[67, 46, 257], [99, 100, 257]])


def _frozen(interface: str = "magnitude") -> "torch.nn.Module":
    model = tiny_model(interface).eval()
    model.codec.freeze_encoder()
    return model


def _proposals(model, target=None, *, counts=(2, 4, 8), chunk=3, estimator=POSTERIOR_MEAN):  # type: ignore[no-untyped-def]
    target = TWO_TARGETS if target is None else target
    return evaluate_proposals(
        model,
        history().repeat(target.shape[0], 1, 1),
        target,
        sample_counts=counts,
        sample_chunk_size=chunk,
        latent_usage=estimator,
        prior_stream=11,
        mixture_stream=22,
    )


def _train_decoder_on_posterior_mean(model) -> None:  # type: ignore[no-untyped-def]
    """Teach the decoder to read the q posterior mean, so the latent carries information."""
    prefix = teacher_forcing_prefix(TWO_TARGETS, 16)
    optimizer = torch.optim.AdamW(model.codec.decoder.parameters(), lr=1e-2)
    with torch.enable_grad():
        for _ in range(150):
            latent = posterior_mean_vector(model.codec.posterior(TWO_TARGETS)).detach()
            logits = model.codec.decoder(latent, prefix)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), TWO_TARGETS.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_prior_and_mixture_proposals_sit_beside_the_q_rows(interface: str) -> None:
    model = _frozen(interface)
    torch.manual_seed(5)
    result = _proposals(model)
    torch.manual_seed(5)
    q_only = importance_evaluate(
        model, history().repeat(2, 1, 1), TWO_TARGETS, sample_counts=(2, 4, 8), sample_chunk_size=3
    )
    # the q rows are the v1 rows, byte for byte, drawn on the caller's stream
    for name in ("log_weights", "decoder_log_prob", "prior_log_prob", "proposal_log_prob"):
        assert torch.equal(getattr(result.q, name), getattr(q_only, name)), name
    assert result.q.by_sample_count == q_only.by_sample_count
    # the prior and the mixture carry a full envelope of their own draws and their own rows
    for population, name in ((result.prior, "prior"), (result.mixture, "mixture")):
        assert population.proposal == name
        assert population.log_weights.shape == (8, 2)
        assert set(population.by_sample_count) == {2, 4, 8}
        assert all(len(rows) == 2 for rows in population.by_sample_count.values())
    assert not torch.equal(result.prior.log_weights, result.q.log_weights)
    assert not torch.equal(result.mixture.decoder_log_prob, result.prior.decoder_log_prob)
    # exact prior importance weights collapse to the decoder term (proposal == prior)
    torch.testing.assert_close(
        result.prior.log_weights, result.prior.decoder_log_prob, atol=1e-9, rtol=0
    )
    # the exact mixture density, at EVERY draw, from both component columns
    mixture = result.mixture
    assert mixture.q_log_prob is not None and mixture.component is not None
    expected = torch.logsumexp(torch.stack([mixture.q_log_prob, mixture.prior_log_prob]), 0) - (
        math.log(2)
    )
    assert torch.equal(mixture.proposal_log_prob, expected)
    torch.testing.assert_close(
        mixture.log_weights,
        mixture.decoder_log_prob + mixture.prior_log_prob - mixture.proposal_log_prob,
    )
    producing = torch.where(mixture.component, mixture.prior_log_prob, mixture.q_log_prob)
    assert not torch.equal(mixture.proposal_log_prob, producing)  # never the producing density
    assert 0 < int(mixture.component.sum()) < mixture.component.numel()
    assert result.latent_operand is LatentUsageOperand.posterior_mean


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_the_mixture_allocation_is_half_and_half_not_just_recorded_as_such(
    interface: str,
) -> None:
    """S10 (L3 review): the record says ``bernoulli_half`` and the weights use the 1/2-1/2 density.

    Both would be false if the component mask were drawn at another rate, and the reader cannot
    tell (the mask is recorded, not re-derived). Two pins: the prior-component fraction, and
    ``E_mix[p / q_mix] = 1``, which holds only when the draws really come from the equal mixture
    (p / q_mix <= 2, so the average converges fast). MEASURED 2026-11-04 at the tip: fraction
    0.5002 and log-mean ratios 0.0049 / -0.0042; with the rate flipped to 0.9, 0.8998 and
    0.5855 / 0.5896.
    """
    model = _frozen(interface)
    result = _proposals(model, counts=(4000,), chunk=500)
    mixture = result.mixture
    assert mixture.component is not None
    fraction = float(mixture.component.double().mean())
    assert fraction == pytest.approx(0.5, abs=0.03)  # sd 0.0056 over 8000 draws
    ratio = torch.exp(mixture.prior_log_prob - mixture.proposal_log_prob).mean(dim=0)
    assert bool((torch.log(ratio).abs() < 0.1).all())


def test_the_extra_proposals_never_move_the_callers_random_stream() -> None:
    model = _frozen()
    torch.manual_seed(9)
    _proposals(model)
    after = torch.get_rng_state()
    torch.manual_seed(9)
    importance_evaluate(
        model, history().repeat(2, 1, 1), TWO_TARGETS, sample_counts=(2, 4, 8), sample_chunk_size=3
    )
    assert torch.equal(after, torch.get_rng_state())


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_all_three_proposals_agree_when_the_prior_equals_q(interface: str) -> None:
    model = _frozen(interface)
    target = torch.tensor([[67, 46, 257]])
    with torch.no_grad():
        proposal = model.codec.posterior(target)
        model.codec.decoder.readout.weight.zero_()
        model.head.direction.weight.zero_()
        model.head.direction.bias.copy_(proposal.direction[0])
        if model.head.radius is not None:
            model.head.radius.weight.zero_()
            model.head.radius.bias.copy_(proposal.log_radius_mean)
    result = _proposals(model, target, counts=(4, 8))
    exact = -3 * math.log(258)
    for population_rows in (
        result.q.by_sample_count,
        result.prior.by_sample_count,
        result.mixture.by_sample_count,
    ):
        for count, rows in population_rows.items():
            assert rows[0].log_estimate == pytest.approx(exact, abs=1e-5)
            assert rows[0].effective_sample_size == pytest.approx(count, abs=1e-5)
    torch.testing.assert_close(result.prior_q_cosine[0], 1.0, atol=1e-6, rtol=0)


def test_the_mixture_density_is_a_logsumexp_not_a_log_of_a_sum() -> None:
    far = mixture_log_density(torch.tensor([0.0, -5000.0]), torch.tensor([-2000.0, 0.0]))
    assert torch.equal(far, torch.tensor([-math.log(2), -math.log(2)], dtype=torch.float64))
    assert torch.isfinite(far).all()  # a log of a sum of exponentials underflows to -inf here
    # both components contribute at the same draw
    tied = mixture_log_density(torch.tensor([-3.0]), torch.tensor([-3.0]))
    assert tied.item() == pytest.approx(-3.0)
    # an exact antipode (one component -inf) is admissible; the other component carries it
    antipode = mixture_log_density(torch.tensor([-math.inf]), torch.tensor([-7.0]))
    assert antipode.item() == pytest.approx(-7.0 - math.log(2))
    assert mixture_log_density(torch.tensor([1.0]), torch.tensor([2.0])).dtype == torch.float64


def test_the_log_weight_sd_is_unbiased_and_absent_for_a_zero_density_row() -> None:
    weights = torch.tensor([[1.0, -math.inf], [3.0, 0.0], [5.0, 1.0]], dtype=torch.float64)
    sd = log_weight_sd(weights, 3)
    assert sd[0] == pytest.approx(2.0) and sd[1] is None  # ddof = 1: sqrt(8 / 2)
    assert log_weight_sd(weights, 2)[0] == pytest.approx(math.sqrt(2.0))


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_the_cosine_is_taken_on_the_law_and_proposal_directions_not_raw_states(
    interface: str,
) -> None:
    model = _frozen(interface)
    target = TWO_TARGETS
    result = _proposals(model, target)
    law = model.predict(history().repeat(2, 1, 1)).law
    posterior = model.codec.posterior(target)
    expected = (law.direction.double() * posterior.direction.double()).sum(-1)
    assert result.prior_q_cosine == pytest.approx(tuple(expected.tolist()), abs=1e-6)
    raw = torch.nn.functional.cosine_similarity(
        model.predict(history().repeat(2, 1, 1)).states[:, -1].double(),
        model.codec.encoder(target).double(),
        dim=-1,
    )
    assert not torch.allclose(
        torch.tensor(result.prior_q_cosine, dtype=torch.float64), raw, atol=1e-3
    )
    assert all(abs(c) <= 1 + 1e-5 for c in result.prior_q_cosine)


def test_log_radius_means_are_absent_in_a_sphere_law_and_present_in_a_magnitude_one() -> None:
    sphere = _proposals(_frozen("sphere"))
    assert sphere.prior_log_radius_mean is None and sphere.q_log_radius_mean is None
    model = _frozen("magnitude")
    magnitude = _proposals(model)
    law = model.predict(history().repeat(2, 1, 1)).law
    posterior = model.codec.posterior(TWO_TARGETS)
    assert law.log_radius_mean is not None and posterior.log_radius_mean is not None
    assert magnitude.prior_log_radius_mean == pytest.approx(tuple(law.log_radius_mean.tolist()))
    assert magnitude.q_log_radius_mean == pytest.approx(tuple(posterior.log_radius_mean.tolist()))


def test_latent_usage_is_positive_when_q_helps() -> None:
    model = _frozen("sphere")
    _train_decoder_on_posterior_mean(model)
    result = _proposals(model)
    gain = sum(
        q - n for q, n in zip(result.q_latent_log_prob, result.null_latent_log_prob, strict=True)
    )
    assert gain > 1.0  # nats over the two rows: the null latent forfeits the sentence's content


def test_latent_usage_is_negative_and_preserved_when_q_hurts() -> None:
    """An operand that scores the OPPOSITE of the posterior mean hurts; the sign must survive."""

    class Opposite:
        operand = LatentUsageOperand.posterior_mean

        def q_decoder_log_prob(self, score, posterior):  # type: ignore[no-untyped-def]
            return score(-posterior_mean_vector(posterior))

    model = _frozen("sphere")
    _train_decoder_on_posterior_mean(model)
    result = _proposals(model, estimator=Opposite())
    gain = sum(
        q - n for q, n in zip(result.q_latent_log_prob, result.null_latent_log_prob, strict=True)
    )
    assert gain < -1.0, "a clamp at zero would hide exactly this"


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_latent_usage_is_exactly_zero_for_a_decoder_that_ignores_the_latent(
    interface: str,
) -> None:
    model = _frozen(interface)
    with torch.no_grad():
        model.codec.decoder.condition.weight.zero_()
    result = _proposals(model)
    assert result.q_latent_log_prob == result.null_latent_log_prob  # bitwise, not a tolerance


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
def test_latent_usage_scores_the_posterior_mean_vector_and_the_null_at_the_named_radius(
    interface: str,
) -> None:
    model = _frozen(interface)
    _train_decoder_on_posterior_mean(model)
    result = _proposals(model)
    posterior = model.codec.posterior(TWO_TARGETS)
    prefix = teacher_forcing_prefix(TWO_TARGETS, 16)

    @torch.no_grad()
    def score(latent: torch.Tensor) -> tuple[float, ...]:
        logits = model.codec.decoder(latent, prefix)
        lp = logits.double().log_softmax(-1).gather(-1, TWO_TARGETS[..., None]).squeeze(-1)
        return tuple(lp.sum(-1).tolist())

    mean = posterior.direction
    if posterior.log_radius_mean is not None:
        mean = mean * posterior.log_radius_mean.exp().unsqueeze(-1)
    assert result.q_latent_log_prob == pytest.approx(score(mean), abs=1e-6)
    assert result.null_latent_log_prob == pytest.approx(
        score(posterior.direction * NULL_LATENT_RADIUS), abs=1e-6
    )
    assert NULL_LATENT_RADIUS == 1e-4
    # a draw is a different latent: scoring one is a different (non-deterministic) number
    draw = posterior.rsample()
    assert score(draw) != pytest.approx(result.q_latent_log_prob, abs=1e-6)


def test_the_operand_is_a_closed_selection_and_draw_average_is_refused_by_name() -> None:
    assert [m.value for m in LatentUsageOperand] == ["posterior_mean", "draw_average"]
    assert POSTERIOR_MEAN.operand is LatentUsageOperand.posterior_mean
    with pytest.raises(UnsupportedLatentOperand, match="draw_average"):
        latent_usage_estimator(LatentUsageOperand.draw_average)
    with pytest.raises(TypeError):
        latent_usage_estimator("posterior_mean")  # type: ignore[arg-type]
    assert issubclass(UnsupportedLatentOperand, ValueError)


def test_derived_stream_keys_are_pure_distinct_and_refuse_a_collision() -> None:
    q_key = (42 * 0x9E3779B1 + 3) % 2**32
    first = derived_stream_keys(seed=42, step=3, batch=0, q_key=q_key)
    assert first == derived_stream_keys(seed=42, step=3, batch=0, q_key=q_key)
    seen = {q_key}
    for batch in range(20):
        for key in derived_stream_keys(seed=42, step=3, batch=batch, q_key=q_key):
            assert 0 <= key < 2**32 and key not in seen
            seen.add(key)
    assert len(seen) == 1 + 2 * 20
    assert derived_stream_keys(seed=43, step=3, batch=0, q_key=q_key) != first
    with pytest.raises(ValueError, match="collide"):
        derived_stream_keys(seed=42, step=3, batch=0, q_key=first[0])


@pytest.mark.parametrize(("device", "forked"), [("cpu", []), ("cuda:0", [0]), ("cuda:1", [1])])
def test_the_derived_stream_forks_the_generators_of_the_device_the_draws_live_on(
    monkeypatch: pytest.MonkeyPatch, device: str, forked: list[int]
) -> None:
    """On a GPU run the q draws consume the CUDA generator; a fork of only the CPU one would let
    the extra proposals advance it. Recorded here without a GPU by spying on ``fork_rng``."""
    import contextlib

    from entropic_substrate.core.sentence import scoring

    seen: list[list[int]] = []

    @contextlib.contextmanager
    def spy(devices: list[int]):  # type: ignore[no-untyped-def]
        seen.append(list(devices))
        yield

    monkeypatch.setattr(torch.random, "fork_rng", spy)
    monkeypatch.setattr(torch, "manual_seed", lambda key: None)
    with scoring._stream(7, torch.device(device)):  # pyright: ignore[reportPrivateUsage]
        pass
    assert seen == [forked]
