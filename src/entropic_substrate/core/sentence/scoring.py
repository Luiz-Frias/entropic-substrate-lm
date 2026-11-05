"""Importance-weighted sentence likelihood estimates with retained sample evidence."""

import hashlib
import math
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, assert_never

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.codec import teacher_forcing_prefix
from entropic_substrate.core.sentence.law import DirectionalMagnitudeLaw
from entropic_substrate.core.sentence.model import SentencePredictor


@dataclass(frozen=True)
class ImportanceRow:
    log_estimate: float | None
    effective_sample_size: float | None
    likelihood_relative_mcse: float | None
    gap: Literal["zero_importance_weight_mass"] | None = None


def importance_statistics(log_weights: torch.Tensor) -> tuple[ImportanceRow, ...]:
    """Summarize independent proposal draws, not independent model replicates.

    MCSE is the estimated relative standard error of the likelihood estimator,
    not a confidence interval for model quality or a guarantee on log likelihood.
    All-zero importance mass is an explicit diagnostic gap, not an invented ESS.
    """
    if (
        log_weights.ndim != 2
        or log_weights.shape[0] < 2
        or log_weights.shape[1] < 1
        or not log_weights.is_floating_point()
    ):
        raise ValueError("importance samples require a floating [K>=2, B>=1] tensor")
    values = log_weights.detach().double().cpu()
    if bool((torch.isnan(values) | torch.isposinf(values)).any()):
        raise ValueError("nonfinite importance arithmetic")
    rows: list[ImportanceRow] = []
    count = values.shape[0]
    for column in values.unbind(dim=1):
        if bool(torch.isneginf(column).all()):
            rows.append(ImportanceRow(None, None, None, "zero_importance_weight_mass"))
            continue
        scaled = (column - column.max()).exp()
        normalized = scaled / scaled.sum()
        rows.append(
            ImportanceRow(
                float(torch.logsumexp(column, 0) - math.log(count)),
                float(1 / normalized.square().sum()),
                float(scaled.std(unbiased=True) / (math.sqrt(count) * scaled.mean())),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ImportanceEvaluation:
    """A finite-sample estimate; not a certified pointwise lower bound.

    Its expectation is the importance-weighted variational lower bound under
    the usual integrability conditions. A better observed estimate alone does
    not certify better true text likelihood. Boundary symbols are scored and
    reported separately from canonical source-byte counts.
    """

    meaning: Literal["importance_weighted_log_marginal_estimate"]
    log_weights: torch.Tensor
    decoder_log_prob: torch.Tensor
    prior_log_prob: torch.Tensor
    proposal_log_prob: torch.Tensor
    prior_states: torch.Tensor
    source_bytes: tuple[int, ...]
    boundary_symbols: tuple[int, ...]
    by_sample_count: dict[int, tuple[ImportanceRow, ...]]


def _validate_envelope(
    model: SentencePredictor, sample_counts: tuple[int, ...], sample_chunk_size: int
) -> int:
    if model.training or any(p.requires_grad for p in model.codec.encoder.parameters()):
        raise ValueError("importance evaluation requires frozen-encoder evaluation mode")
    if (
        type(sample_counts) is not tuple
        or not sample_counts
        or any(type(k) is not int or k < 2 for k in sample_counts)
        or tuple(sorted(set(sample_counts))) != sample_counts
    ):
        raise ValueError("sample counts must be increasing distinct integers >= 2")
    count = sample_counts[-1]
    if type(sample_chunk_size) is not int or not 1 <= sample_chunk_size <= count:
        raise ValueError("sample chunk size must be positive and within the envelope")
    return count


@dataclass(frozen=True)
class _Target:
    """The shared per-batch inputs every proposal scores against (no RNG is consumed here)."""

    model: SentencePredictor
    prefix: torch.Tensor
    labels: torch.Tensor
    padding: torch.Tensor
    complete: torch.Tensor
    prediction_law: DirectionalMagnitudeLaw
    prior_states: torch.Tensor
    posterior: DirectionalMagnitudeLaw
    chunk: int


def _target(
    model: SentencePredictor, history: torch.Tensor, complete: torch.Tensor, chunk: int
) -> _Target:
    prefix = teacher_forcing_prefix(complete, model.codec.spec.max_sentence_bytes)
    prediction = model.predict(history)
    if prediction.states.shape[0] != complete.shape[0]:
        raise ValueError("prior and target proposal batch sizes differ")
    padding = complete == SentenceByteCodec.pad
    return _Target(
        model,
        prefix,
        complete.masked_fill(padding, 0),
        padding,
        complete,
        prediction.law,
        prediction.states.detach().cpu(),
        model.codec.posterior(complete),
        chunk,
    )


def _decoder_log_prob(target: _Target, samples: torch.Tensor) -> torch.Tensor:
    """Per-draw, per-row decoder log-probability of the complete target, FP64 on the CPU."""
    count, (batch, length) = samples.shape[0], target.complete.shape
    scores: list[torch.Tensor] = []
    for start in range(0, count, target.chunk):
        selected = samples[start : start + target.chunk]
        size = selected.shape[0]
        expanded = target.prefix.unsqueeze(0).expand(size, -1, -1).reshape(size * batch, length)
        logits = target.model.codec.decoder(selected.reshape(size * batch, -1), expanded)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probs = logits.double().log_softmax(-1).reshape(size, batch, length, -1)
            scored = log_probs.gather(
                -1, target.labels[None, ..., None].expand(size, -1, -1, -1)
            ).squeeze(-1)
            scores.append(scored.masked_fill(target.padding.unsqueeze(0), 0).sum(-1).cpu())
    return torch.cat(scores)


@torch.no_grad()
def importance_evaluate(
    model: SentencePredictor,
    history: torch.Tensor,
    complete: torch.Tensor,
    *,
    sample_counts: tuple[int, ...],
    sample_chunk_size: int,
) -> ImportanceEvaluation:
    """Estimate integral p(bytes|c,history) p(c|history) dc using frozen q(c|bytes).

    The target-conditioned proposal never enters the spine. The effective law
    already marginalizes unused magnitude for spherical sentence interfaces.
    Chunk only decoder execution; sample the declared proposal population once
    so changing the memory envelope cannot select a different sample set.
    """
    count = _validate_envelope(model, sample_counts, sample_chunk_size)
    target = _target(model, history, complete, sample_chunk_size)
    return _q_population(target, sample_counts, count)


def _q_population(
    target: _Target, sample_counts: tuple[int, ...], count: int
) -> ImportanceEvaluation:
    """The q proposal, drawn on the CALLER's RNG stream exactly as the v1 evaluation did."""
    proposal = target.posterior
    samples = proposal.rsample((count,))
    prior_log = target.prediction_law.log_prob(samples).double().cpu()
    proposal_log = proposal.log_prob(samples).double().cpu()
    if not bool(torch.isfinite(proposal_log).all()):
        raise ValueError("nonfinite density at a sample drawn from the proposal")
    decoder_log = _decoder_log_prob(target, samples)
    log_weights = decoder_log + prior_log - proposal_log
    return ImportanceEvaluation(
        "importance_weighted_log_marginal_estimate",
        log_weights,
        decoder_log,
        prior_log,
        proposal_log,
        target.prior_states,
        tuple(int(n) for n in (target.complete < 256).sum(-1).tolist()),
        tuple(1 for _ in range(target.complete.shape[0])),
        {k: importance_statistics(log_weights[:k]) for k in sample_counts},
    )


# --------------------------------------------------------------------------------------------
# FOR-1704 (L4d): the prior and the q-plus-prior mixture proposals, beside q
# --------------------------------------------------------------------------------------------

Proposal = Literal["q", "prior", "mixture"]
PROPOSALS: tuple[Proposal, ...] = ("q", "prior", "mixture")
MIXTURE_ALLOCATION = "bernoulli_half"
# Radius of the latent-blind comparison point: the q posterior direction scaled to this length.
# A named constant, not a parameter: the null side of latent usage is shared by every operand.
NULL_LATENT_RADIUS = 1e-4
# One cosine of two unit directions may exceed 1 by the product of the law's unit-norm tolerance
# (core.sentence.law admits |log radius| <= 1e-5 per direction): exp(2e-5).
COSINE_SLACK = math.exp(2e-5)
_STREAM_LABEL = "h-evaluation-stream-v1"


def derived_stream_keys(*, seed: int, step: int, batch: int, q_key: int) -> tuple[int, int]:
    """RNG keys for the prior and mixture draws of ONE evaluation batch.

    Pure in ``(seed, step, batch)`` (and the proposal), within the 32 low bits the CPU Mersenne
    Twister consumes, and refused when they equal the q key of the evaluation or each other, so
    the extra proposals can never replay q's noise. They are consumed inside nested
    ``fork_rng`` blocks, so q's own stream (seeded once, then advanced across the batches) is
    never moved by them.
    """
    keys = tuple(
        int.from_bytes(
            hashlib.sha256(f"{_STREAM_LABEL}:{seed}:{step}:{batch}:{name}".encode()).digest()[:4],
            "big",
        )
        for name in ("prior", "mixture")
    )
    if len({keys[0], keys[1], q_key}) != 3:
        raise ValueError("derived evaluation streams collide with the q stream or each other")
    return keys[0], keys[1]


@contextmanager
def _stream(key: int, device: torch.device) -> Generator[None]:
    """Seed the global generators with ``key`` and restore them (CPU and CUDA) on exit.

    ``device`` is where the draws live (the law's direction): the CUDA generator is forked only
    when they are on CUDA. Never pass a tensor the caller moved to the CPU for storage.
    """
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(key)
        yield


class LatentUsageOperand(StrEnum):
    """The closed set of q-side operands latent usage can be scored with (DH-52)."""

    posterior_mean = "posterior_mean"
    draw_average = "draw_average"


class UnsupportedLatentOperand(ValueError):
    """The selected operand has no implementation yet (``draw_average`` is FOR-1721)."""


DecoderScorer = Callable[[torch.Tensor], torch.Tensor]


class LatentUsageEstimator(Protocol):
    """The q-side operand of latent usage, at a level every operand satisfies.

    ``q_decoder_log_prob(score, posterior)`` returns the per-row decoder log-probability of the
    target on the q side, ``[batch]`` FP64. ``score`` maps a ``[batch, width]`` latent to its
    per-row log-probability under the one shared call shape, so the null side (scored by the
    caller with the same ``score``) cannot differ from the q side in anything but the latent.
    A later operand (a draw average) implements the same method without reshaping the record.
    """

    @property
    def operand(self) -> LatentUsageOperand: ...

    def q_decoder_log_prob(
        self, score: DecoderScorer, posterior: DirectionalMagnitudeLaw
    ) -> torch.Tensor: ...


def posterior_mean_vector(law: DirectionalMagnitudeLaw) -> torch.Tensor:
    """The law's location vector: ``direction * exp(log_radius_mean)``; the unit direction (sphere).

    Deterministic and RNG-free. It is the vector the law is parameterized by, NOT ``E_q[z]``
    (the expectation of a unit-direction draw shortens by the mean resultant length). A
    different estimand is a new operand, never a redefinition of this one.
    """
    if law.log_radius_mean is None:
        return law.direction
    return law.direction * law.log_radius_mean.exp().unsqueeze(-1)


@dataclass(frozen=True)
class PosteriorMeanUsage:
    @property
    def operand(self) -> LatentUsageOperand:
        return LatentUsageOperand.posterior_mean

    def q_decoder_log_prob(
        self, score: DecoderScorer, posterior: DirectionalMagnitudeLaw
    ) -> torch.Tensor:
        return score(posterior_mean_vector(posterior))


def latent_usage_estimator(operand: LatentUsageOperand) -> LatentUsageEstimator:
    """The one factory from the closed selection to an implementation."""
    if type(operand) is not LatentUsageOperand:
        raise TypeError("the latent usage operand must be a LatentUsageOperand member")
    if operand is LatentUsageOperand.posterior_mean:
        return PosteriorMeanUsage()
    if operand is LatentUsageOperand.draw_average:
        raise UnsupportedLatentOperand(
            "latent usage operand 'draw_average' is not built (FOR-1721)"
        )
    assert_never(operand)


@dataclass(frozen=True)
class ProposalPopulation:
    """The full-envelope draws of one non-q proposal and their importance rows.

    ``q_log_prob`` and ``component`` (True = drawn from the prior) are set for the mixture only;
    its ``proposal_log_prob`` is ``logsumexp(log q, log p) - log 2`` at EVERY draw.
    """

    proposal: Proposal
    log_weights: torch.Tensor
    decoder_log_prob: torch.Tensor
    prior_log_prob: torch.Tensor
    proposal_log_prob: torch.Tensor
    by_sample_count: dict[int, tuple[ImportanceRow, ...]]
    q_log_prob: torch.Tensor | None = None
    component: torch.Tensor | None = None


@dataclass(frozen=True)
class ProposalEvaluation:
    """q (v1 shape) beside the prior and mixture populations and the per-batch diagnostics."""

    q: ImportanceEvaluation
    prior: ProposalPopulation
    mixture: ProposalPopulation
    prior_log_radius_mean: tuple[float, ...] | None
    q_log_radius_mean: tuple[float, ...] | None
    prior_q_cosine: tuple[float, ...]
    log_weight_sd: dict[Proposal, dict[int, tuple[float | None, ...]]]
    latent_operand: LatentUsageOperand
    q_latent_log_prob: tuple[float, ...]
    null_latent_log_prob: tuple[float, ...]


def mixture_log_density(q_log_prob: torch.Tensor, prior_log_prob: torch.Tensor) -> torch.Tensor:
    """``logsumexp(log q, log p) - log 2`` in FP64, both components at every draw.

    Never the producing component's density alone and never a log of a sum of exponentials:
    measured prior-minus-q gaps of thousands of nats underflow any exp-sum. A component that is
    exactly ``-inf`` (an antipodal draw) is admissible; both being ``-inf`` is not reachable
    from a draw of either component, and the caller refuses it.
    """
    with torch.autocast(device_type="cpu", enabled=False):
        both = torch.stack([q_log_prob.double(), prior_log_prob.double()])
        return torch.logsumexp(both, dim=0) - math.log(2)


def log_weight_sd(log_weights: torch.Tensor, k: int) -> tuple[float | None, ...]:
    """Unbiased (ddof = 1) sd of each row's first ``k`` log-weights, FP64.

    A row whose k-prefix holds a non-finite weight (an exact zero density) has no sd: ``None``,
    never a NaN, and the extractor counts it out of the mean.
    """
    values = log_weights[:k].double()
    return tuple(
        float(column.std(unbiased=True)) if bool(torch.isfinite(column).all()) else None
        for column in values.unbind(dim=1)
    )


def _row_floats(values: torch.Tensor) -> tuple[float, ...]:
    if not bool(torch.isfinite(values).all()):
        raise ValueError("nonfinite per-row evaluation diagnostic")
    return tuple(float(v) for v in values.double().cpu().tolist())


@torch.no_grad()
def evaluate_proposals(
    model: SentencePredictor,
    history: torch.Tensor,
    complete: torch.Tensor,
    *,
    sample_counts: tuple[int, ...],
    sample_chunk_size: int,
    latent_usage: LatentUsageEstimator,
    prior_stream: int,
    mixture_stream: int,
) -> ProposalEvaluation:
    """q (on the caller's stream), the prior and the mixture (on their own), and the diagnostics.

    q is drawn first, on the caller's global stream, by the same code that drew it before, so
    its rows are unchanged. The prior and mixture draws run inside nested ``fork_rng`` blocks
    seeded from ``prior_stream`` / ``mixture_stream``: they neither advance nor reorder the
    caller's stream, which matters because the evaluation loops over many batches on it.

    The mixture draws a full envelope from q and a full envelope from the prior on its own
    stream and keeps, per draw, the prior's with probability 1/2 (``bernoulli_half``): every
    k-prefix is then a valid mixture sample. Every draw is weighted by the MIXTURE density.
    """
    count = _validate_envelope(model, sample_counts, sample_chunk_size)
    target = _target(model, history, complete, sample_chunk_size)
    q = _q_population(target, sample_counts, count)
    law, posterior = target.prediction_law, target.posterior

    with _stream(prior_stream, law.direction.device):
        prior_samples = law.rsample((count,))
    prior_log = law.log_prob(prior_samples).double().cpu()
    if not bool(torch.isfinite(prior_log).all()):
        raise ValueError("nonfinite density at a sample drawn from the prior")
    prior_decoder = _decoder_log_prob(target, prior_samples)
    prior_weights = prior_decoder + prior_log - prior_log
    prior = ProposalPopulation(
        "prior",
        prior_weights,
        prior_decoder,
        prior_log,
        prior_log,
        {k: importance_statistics(prior_weights[:k]) for k in sample_counts},
    )

    with _stream(mixture_stream, law.direction.device):
        from_q = posterior.rsample((count,))
        from_prior = law.rsample((count,))
        batch = complete.shape[0]
        component = torch.rand((count, batch), device=from_q.device) < 0.5
    mixed = torch.where(component.unsqueeze(-1), from_prior, from_q)
    mix_q = posterior.log_prob(mixed).double().cpu()
    mix_prior = law.log_prob(mixed).double().cpu()
    chosen = component.cpu()
    producing = torch.where(chosen, mix_prior, mix_q)
    density = mixture_log_density(mix_q, mix_prior)
    if not bool(torch.isfinite(producing).all()) or not bool(torch.isfinite(density).all()):
        raise ValueError("nonfinite density at a sample drawn from the mixture")
    mix_decoder = _decoder_log_prob(target, mixed)
    mix_weights = mix_decoder + mix_prior - density
    mixture = ProposalPopulation(
        "mixture",
        mix_weights,
        mix_decoder,
        mix_prior,
        density,
        {k: importance_statistics(mix_weights[:k]) for k in sample_counts},
        q_log_prob=mix_q,
        component=chosen,
    )

    with torch.autocast(device_type=law.direction.device.type, enabled=False):
        left, right = law.direction.double(), posterior.direction.double()
        cosine = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1))

    def scorer(latents: torch.Tensor) -> torch.Tensor:
        return _decoder_log_prob(target, latents.unsqueeze(0))[0]

    q_latent = latent_usage.q_decoder_log_prob(scorer, posterior)
    null_latent = scorer(posterior.direction * NULL_LATENT_RADIUS)
    return ProposalEvaluation(
        q,
        prior,
        mixture,
        None if law.log_radius_mean is None else _row_floats(law.log_radius_mean),
        None if posterior.log_radius_mean is None else _row_floats(posterior.log_radius_mean),
        _row_floats(cosine),
        {
            "q": {k: log_weight_sd(q.log_weights, k) for k in sample_counts},
            "prior": {k: log_weight_sd(prior.log_weights, k) for k in sample_counts},
            "mixture": {k: log_weight_sd(mixture.log_weights, k) for k in sample_counts},
        },
        latent_usage.operand,
        _row_floats(q_latent),
        _row_floats(null_latent),
    )
