"""Warm-up reconstruction and future-sentence training, never marginal likelihood."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from entropic_substrate.predictive_state.metric import finite_tensor
from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.codec import teacher_forcing_prefix
from entropic_substrate.sentence_prediction.model import SentencePredictor, SentenceReadout


@dataclass(frozen=True)
class SentenceLoss:
    """Training quantities with boundary-inclusive CE and explicit exposure counts.

    source_bytes counts decoder-scored canonical bytes (not control symbols).
    History/encoder counts name repeated exposure, not unique data acquired.
    Future density is averaged per latent dimension, with its declared measure.
    Neither training meaning is a marginalized generator likelihood estimate.
    """

    meaning: Literal["oracle_reconstruction_training", "prior_sample_decoder_training"]
    total: torch.Tensor
    decoder_ce: torch.Tensor
    future_nll_per_dimension: torch.Tensor | None
    readout: SentenceReadout | None
    source_bytes: int
    boundary_symbols: int
    history_bytes: int
    encoder_bytes: int


def _decoder_loss(logits: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        loss = F.cross_entropy(
            logits.float().flatten(0, 1), complete.flatten(), ignore_index=SentenceByteCodec.pad
        )
    finite_tensor(loss, "sentence_decoder_objective")
    return loss


def codec_warmup_loss(model: SentencePredictor, complete: torch.Tensor) -> SentenceLoss:
    """Reconstruct an original sentence via its learned stochastic encoding."""
    if any(not p.requires_grad for p in model.codec.parameters()):
        raise ValueError("codec warm-up requires a trainable encoder and decoder")
    prefix = teacher_forcing_prefix(complete, model.codec.spec.max_sentence_bytes)
    logits = model.codec.reconstruct(complete, prefix)
    loss = _decoder_loss(logits, complete)
    source_bytes = int((complete < 256).sum())
    return SentenceLoss(
        "oracle_reconstruction_training",
        loss,
        loss,
        None,
        None,
        source_bytes,
        complete.shape[0],
        0,
        source_bytes,
    )


def prediction_loss(
    model: SentencePredictor,
    history: torch.Tensor,
    complete: torch.Tensor,
    *,
    future_weight: float,
) -> SentenceLoss:
    """Prior-sampled decoder CE, optionally supervised by absolute future states.

    The target encoder is frozen and its samples detached. Disabling its loss
    skips that target exposure entirely but leaves the head trainable through
    the sampled prior and observable decoder objective.
    """
    if (
        type(future_weight) not in (int, float)
        or not math.isfinite(future_weight)
        or future_weight < 0
    ):
        raise ValueError("future weight must be a finite nonnegative number")
    if any(p.requires_grad for p in model.codec.encoder.parameters()):
        raise ValueError("initial prediction construction requires a frozen encoder")
    prefix = teacher_forcing_prefix(complete, model.codec.spec.max_sentence_bytes)
    readout = model(history, prefix)
    decoder_ce = _decoder_loss(readout.logits, complete)
    future = None
    total = decoder_ce
    source_bytes = int((complete < 256).sum())
    history_bytes = int((history < 256).sum())
    encoder_bytes = history_bytes
    if future_weight > 0:
        with torch.no_grad():
            target = model.codec.posterior(complete).rsample()
        future = -readout.prediction.law.log_prob(target).mean() / model.codec.spec.width
        finite_tensor(future, "sentence_future_objective")
        total = decoder_ce + future_weight * future
        encoder_bytes += source_bytes
    finite_tensor(total, "sentence_objective")
    return SentenceLoss(
        "prior_sample_decoder_training",
        total,
        decoder_ce,
        future,
        readout,
        source_bytes,
        complete.shape[0],
        history_bytes,
        encoder_bytes,
    )
