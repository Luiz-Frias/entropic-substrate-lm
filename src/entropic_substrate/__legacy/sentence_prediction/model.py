"""Codec-separated next-sentence prediction with no target-state forward input."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from entropic_substrate.predictive_state.dynamics import (
    DEFAULT_SOLVER_CONFIG,
    SolverConfig,
    SolverDiagnostics,
)
from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.codec import CodecSpec, NeuralSentenceCodec
from entropic_substrate.sentence_prediction.consumer import ConsumerMode, SentenceConsumer
from entropic_substrate.sentence_prediction.head import SentenceLawHead
from entropic_substrate.sentence_prediction.law import DirectionalMagnitudeLaw, LawSpec
from entropic_substrate.sentence_prediction.numerics import apply_interface, checked_coordinates
from entropic_substrate.sentence_prediction.spine import SentenceSpine, SpineSpec


@dataclass(frozen=True)
class SentencePrediction:
    """History states and next-sentence law; not a language-likelihood estimate."""

    sentences: torch.Tensor
    states: torch.Tensor
    law: DirectionalMagnitudeLaw
    diagnostics: SolverDiagnostics | None


@dataclass(frozen=True)
class SentenceReadout:
    prediction: SentencePrediction
    sample: torch.Tensor
    logits: torch.Tensor


class SentencePredictor(nn.Module):
    """Encoder → sentence-clock Transformer → latent law → local byte decoder.

    The completed history is encoded deterministically. Only the next latent
    is sampled; a rollout must re-encode emitted bytes for its operative next
    history, retaining that sample separately as a shadow prediction.
    """

    def __init__(
        self,
        codec: CodecSpec,
        spine: SpineSpec,
        law: LawSpec,
        *,
        consumer: ConsumerMode,
        solver: SolverConfig = DEFAULT_SOLVER_CONFIG,
    ) -> None:
        super().__init__()
        if type(spine) is not SpineSpec or type(solver) is not SolverConfig:
            raise TypeError("predictor requires explicit spine and solver specifications")
        if consumer not in ("isotropic", "fisher"):
            raise ValueError("unknown sentence consumer")
        self.codec = NeuralSentenceCodec(codec, law)
        self.spine = SentenceSpine(codec.width, spine)
        self.head = SentenceLawHead(spine.width, law)
        self.consumer_mode: ConsumerMode = consumer
        self.solver = solver

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Accept completed, nonterminal prompt sentences, including an empty prompt.

        Document-end is not a sentence-end alias: continuation after it would
        cross an observation boundary. Truncation/context-window policy belongs
        to the source sampler or rollout, never this numerical path.
        """
        if not isinstance(cast(object, history), torch.Tensor):
            raise TypeError("sentence history must be a tensor")
        if history.ndim != 3 or history.shape[0] < 1 or history.shape[2] < 1:
            raise ValueError("sentence history must have shape [batch, sentences, bytes]")
        if history.dtype != torch.int64:
            raise TypeError("sentence history must be int64")
        if history.shape[1] > self.spine.spec.context_sentences:
            raise ValueError("sentence history exceeds the declared context")
        if bool((history == SentenceByteCodec.document_end).any()):
            raise ValueError("cannot predict beyond a document-end observation")
        if history.shape[1] == 0:
            return self.spine.bos.new_empty((history.shape[0], 0, self.codec.spec.width))
        encoded = self.codec.encoder(history.flatten(0, 1))
        encoded = apply_interface(encoded, self.codec.law_spec.mode)
        return encoded.reshape(history.shape[0], history.shape[1], self.codec.spec.width)

    def predict(self, history: torch.Tensor) -> SentencePrediction:
        """Predict from observations alone; neither targets nor local prefixes enter."""
        sentences = self.encode_history(history)
        return self.predict_latents(sentences)

    def predict_latents(self, sentences: torch.Tensor) -> SentencePrediction:
        """Use already encoded operative history, preserving checkpointed feedback."""
        if self.codec.law_spec.mode == "sphere" and sentences.numel():
            coordinates = checked_coordinates(sentences)
            if bool((coordinates.log_radius.abs() > 1e-5).any()):
                raise ValueError("spherical sentence interface requires unit operative states")
        consumer = SentenceConsumer(
            self.head, self.spine.spec.carrier, self.consumer_mode, self.solver
        )
        states = self.spine(sentences, consumer=consumer)
        return SentencePrediction(sentences, states, self.head(states[:, -1]), consumer.diagnostics)

    def forward(self, history: torch.Tensor, prefix: torch.Tensor) -> SentenceReadout:
        """Sample at temperature one with no truncation; decode only local prefix."""
        prediction = self.predict(history)
        sample = prediction.law.rsample()
        logits = self.codec.decoder(sample, prefix)
        return SentenceReadout(prediction, sample, logits)
