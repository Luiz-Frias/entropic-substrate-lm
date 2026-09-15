"""One-byte rollout transitions with operative feedback and shadow predictions."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.model import SentencePredictor
from entropic_substrate.sentence_prediction.numerics import apply_interface

type RolloutStatus = Literal[
    "active", "document_end", "empty_sentence", "invalid_utf8", "length_limit", "context_limit"
]


@dataclass(frozen=True)
class SentenceEmission:
    source: bytes
    terminal: int
    shadow: torch.Tensor
    operative: torch.Tensor


@dataclass(frozen=True)
class SentenceRollout:
    """Explicit feedback state; checkpoint it alongside model and Torch RNG.

    Histories and emitted source bytes are canonical. Pending observations use
    the bound byte codec. Failed outcomes retain the sample, observations, and
    sampled-token log scores; no malformed emission becomes operative context.
    """

    codec_identity: str
    history: tuple[bytes, ...]
    context: torch.Tensor
    pending: tuple[int, ...] = ()
    sample: torch.Tensor | None = None
    emissions: tuple[SentenceEmission, ...] = ()
    observations: tuple[int, ...] = ()
    log_scores: tuple[float, ...] = ()
    status: RolloutStatus = "active"


@torch.no_grad()
def begin_rollout(
    model: SentencePredictor, prompt: tuple[bytes, ...], codec: SentenceByteCodec
) -> SentenceRollout:
    """A tuple declares completed prompt spans; no partial-sentence inference."""
    if model.training:
        raise ValueError("rollout requires evaluation mode")
    if type(prompt) is not tuple:
        raise TypeError("completed prompt spans must be an immutable tuple")
    if len(prompt) > model.spine.spec.context_sentences:
        raise ValueError("prompt exceeds the sentence context limit")
    for source in prompt:
        if type(source) is not bytes or not 1 <= len(source) <= model.codec.spec.max_sentence_bytes:
            raise ValueError("prompt spans must be nonempty bounded bytes")
        source.decode("utf-8", errors="strict")
    width = max((len(source) + 1 for source in prompt), default=1)
    observations = torch.full(
        (1, len(prompt), width), codec.pad, dtype=torch.int64, device=model.spine.bos.device
    )
    for index, source in enumerate(prompt):
        values = [*(codec.permutation[b] for b in source), codec.sentence_end]
        observations[0, index, : len(values)] = torch.tensor(values, device=observations.device)
    context = model.encode_history(observations).detach()
    return SentenceRollout(codec.identity, prompt, context)


@torch.no_grad()
def advance_rollout(
    model: SentencePredictor, state: SentenceRollout, codec: SentenceByteCodec
) -> SentenceRollout:
    """Sample exactly one byte/control, never resample a pending sentence latent."""
    if model.training:
        raise ValueError("rollout requires evaluation mode")
    if state.status != "active":
        raise ValueError("cannot advance a terminal rollout")
    if state.codec_identity != codec.identity:
        raise ValueError("rollout codec identity mismatch")
    if state.context.shape != (1, len(state.history), model.codec.spec.width):
        raise ValueError("rollout context and source history are misaligned")
    if any(type(value) is not int or not 0 <= value < 256 for value in state.pending):
        raise ValueError("pending prefix must contain only encoded bytes")
    if len(state.pending) > model.codec.spec.max_sentence_bytes:
        raise ValueError("active rollout cannot contain an oversize pending prefix")
    sample = state.sample
    if sample is None:
        if state.pending:
            raise ValueError("pending bytes require their original sampled latent")
        sample = model.predict_latents(state.context).law.rsample().detach()
    prefix = torch.tensor([[codec.bos, *state.pending]], device=state.context.device)
    logits = model.codec.decoder(sample, prefix)[0, -1]
    # Stable log scores, no probability floor or top-k/top-p truncation.
    log_probabilities = logits.log_softmax(dim=-1)
    observation = int(torch.multinomial(log_probabilities.exp(), 1).item())
    scores = (*state.log_scores, float(log_probabilities[observation]))
    updated = replace(
        state, sample=sample, observations=(*state.observations, observation), log_scores=scores
    )
    if observation < 256:
        pending = (*state.pending, observation)
        status: RolloutStatus = (
            "length_limit" if len(pending) > model.codec.spec.max_sentence_bytes else "active"
        )
        return replace(updated, pending=pending, status=status)
    if observation not in (codec.sentence_end, codec.document_end):
        raise ValueError("decoder sampled an input-only control")
    if not state.pending:
        return replace(updated, status="empty_sentence")
    source = bytes(codec.canonical_byte(value) for value in state.pending)
    try:
        source.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return replace(updated, status="invalid_utf8")
    complete = torch.tensor([[*state.pending, observation]], device=state.context.device)
    operative = apply_interface(model.codec.encoder(complete), model.codec.law_spec.mode).detach()
    emission = SentenceEmission(source, observation, sample, operative)
    history = (*state.history, source)
    status = "document_end" if observation == codec.document_end else "active"
    if status == "active" and len(history) > model.spine.spec.context_sentences:
        status = "context_limit"
    return replace(
        updated,
        history=history,
        context=torch.cat((state.context, operative.unsqueeze(1)), dim=1),
        pending=(),
        sample=None,
        emissions=(*state.emissions, emission),
        status=status,
    )
