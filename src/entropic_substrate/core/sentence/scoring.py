"""Importance-weighted sentence likelihood estimates with retained sample evidence."""

import math
from dataclasses import dataclass
from typing import Literal

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.codec import teacher_forcing_prefix
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
    prefix = teacher_forcing_prefix(complete, model.codec.spec.max_sentence_bytes)
    prediction = model.predict(history)
    if prediction.states.shape[0] != complete.shape[0]:
        raise ValueError("prior and target proposal batch sizes differ")
    proposal = model.codec.posterior(complete)
    samples = proposal.rsample((count,))
    prior_log = prediction.law.log_prob(samples).double().cpu()
    proposal_log = proposal.log_prob(samples).double().cpu()
    if not bool(torch.isfinite(proposal_log).all()):
        raise ValueError("nonfinite density at a sample drawn from the proposal")
    padding = complete == SentenceByteCodec.pad
    labels = complete.masked_fill(padding, 0)
    batch, length = complete.shape
    decoder_scores: list[torch.Tensor] = []
    for start in range(0, count, sample_chunk_size):
        selected = samples[start : start + sample_chunk_size]
        size = selected.shape[0]
        expanded = prefix.unsqueeze(0).expand(size, -1, -1).reshape(size * batch, length)
        logits = model.codec.decoder(selected.reshape(size * batch, -1), expanded)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            log_probs = logits.double().log_softmax(-1).reshape(size, batch, length, -1)
            scored = log_probs.gather(-1, labels[None, ..., None].expand(size, -1, -1, -1)).squeeze(
                -1
            )
            decoder_scores.append(scored.masked_fill(padding.unsqueeze(0), 0).sum(-1).cpu())
    decoder_log = torch.cat(decoder_scores)
    log_weights = decoder_log + prior_log - proposal_log
    return ImportanceEvaluation(
        "importance_weighted_log_marginal_estimate",
        log_weights,
        decoder_log,
        prior_log,
        proposal_log,
        prediction.states.detach().cpu(),
        tuple(int(n) for n in (complete < 256).sum(-1).tolist()),
        tuple(1 for _ in range(batch)),
        {k: importance_statistics(log_weights[:k]) for k in sample_counts},
    )
