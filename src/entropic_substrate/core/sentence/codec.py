"""Learned sentence encoding and local causal byte realization, without a spine."""

from dataclasses import dataclass
from typing import Self, cast

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.__legacy.predictive_state.metric import finite_tensor
from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.law import DirectionalMagnitudeLaw, LawSpec


@dataclass(frozen=True, slots=True)
class CodecSpec:
    width: int = 384
    encoder_layers: int = 4
    decoder_width: int = 384
    decoder_layers: int = 4
    head_width: int = 64
    max_sentence_bytes: int = 4096

    def __post_init__(self) -> None:
        values = (
            self.width,
            self.encoder_layers,
            self.decoder_width,
            self.decoder_layers,
            self.head_width,
            self.max_sentence_bytes,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("codec dimensions and limits must be positive integers")
        if self.width < 2 or self.head_width < 2:
            raise ValueError("codec latent and head widths must be >= 2")
        if self.width % self.head_width or self.decoder_width % self.head_width:
            raise ValueError("codec head width must divide encoder and decoder widths")


def _tokens(tokens: torch.Tensor, limit: int, *, complete: bool) -> torch.Tensor:
    """Validate complete-sentence or local-prefix framing and return padding mask."""
    if not isinstance(cast(object, tokens), torch.Tensor):
        raise TypeError("codec token batch must be a tensor")
    if tokens.ndim != 2 or tokens.shape[0] < 1 or not 1 <= tokens.shape[1] <= limit + 1:
        raise ValueError("codec token batch has invalid shape or length")
    if tokens.dtype != torch.int64:
        raise TypeError("codec tokens must be int64")
    if bool(((tokens < 0) | (tokens >= SentenceByteCodec.input_size)).any()):
        raise ValueError("codec token outside the declared alphabet")
    padding = tokens == SentenceByteCodec.pad
    lengths = (~padding).sum(dim=1)
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    if not torch.equal(padding, positions >= lengths.unsqueeze(1)):
        raise ValueError("codec padding must be a contiguous suffix")
    if complete:
        if bool((lengths < 2).any()):
            raise ValueError("complete sentence requires bytes and a boundary")
        terminal = tokens.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        if bool(
            (
                (terminal != SentenceByteCodec.sentence_end)
                & (terminal != SentenceByteCodec.document_end)
            ).any()
        ):
            raise ValueError("complete sentence is missing its terminal boundary")
        body = positions < (lengths - 1).unsqueeze(1)
        if bool((body & (tokens >= 256)).any()):
            raise ValueError("control symbol inside sentence bytes")
    else:
        if bool((lengths < 1).any()) or bool((tokens[:, 0] != SentenceByteCodec.bos).any()):
            raise ValueError("local decoder prefix must start with BOS")
        body = (positions > 0) & ~padding
        if bool((body & (tokens >= 256)).any()):
            raise ValueError("local decoder prefix contains a terminal or input control")
    return padding


def teacher_forcing_prefix(complete: torch.Tensor, limit: int) -> torch.Tensor:
    """Shift validated labels, masking boundaries out of shorter padded prefixes."""
    padding = _tokens(complete, limit, complete=True)
    start = torch.full_like(complete[:, :1], SentenceByteCodec.bos)
    prefix = torch.cat((start, complete[:, :-1]), dim=1)
    return prefix.masked_fill(padding, SentenceByteCodec.pad)


class _FloatNorm(nn.LayerNorm):
    def __init__(self, width: int) -> None:
        super().__init__(width)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=input.device.type, enabled=False):
            dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
            return F.layer_norm(
                input.to(dtype),
                self.normalized_shape,
                self.weight.to(dtype),
                self.bias.to(dtype),
                self.eps,
            )


class _TokenBlock(nn.Module):
    """Explicit pre-norm views; no fused path can bypass their FP32 islands."""

    def __init__(self, width: int, head_width: int) -> None:
        super().__init__()
        self.norm1, self.norm2 = _FloatNorm(width), _FloatNorm(width)
        self.self_attn = nn.MultiheadAttention(
            width, width // head_width, dropout=0.0, batch_first=True
        )
        self.linear1, self.linear2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        src_key_padding_mask: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        view = self.norm1(hidden)
        update = self.self_attn(
            view,
            view,
            view,
            key_padding_mask=src_key_padding_mask,
            attn_mask=src_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        hidden = hidden + update
        return hidden + self.linear2(F.gelu(self.linear1(self.norm2(hidden))))


def _blocks(width: int, layers: int, head_width: int) -> nn.ModuleList:
    # Independent initializations, with no final normalization of the carrier.
    return nn.ModuleList(_TokenBlock(width, head_width) for _ in range(layers))


class SentenceEncoder(nn.Module):
    """Bidirectional within a completed span; output is one unconstrained vector."""

    def __init__(self, spec: CodecSpec) -> None:
        super().__init__()
        self.spec = spec
        self.tokens = nn.Embedding(
            SentenceByteCodec.input_size, spec.width, padding_idx=SentenceByteCodec.pad
        )
        self.positions = nn.Embedding(spec.max_sentence_bytes + 1, spec.width)
        self.blocks = _blocks(spec.width, spec.encoder_layers, spec.head_width)
        self.pool_query = nn.Parameter(torch.randn(spec.width) / spec.width**0.5)
        self.projection = nn.Linear(spec.width, spec.width, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        padding = _tokens(tokens, self.spec.max_sentence_bytes, complete=True)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        for block in self.blocks:
            hidden = block(hidden, src_key_padding_mask=padding)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
            hidden = hidden.to(dtype)
            scores = (hidden * self.pool_query).sum(dim=-1) / self.spec.width**0.5
            weights = scores.masked_fill(padding, -torch.inf).softmax(dim=-1)
            pooled = (weights.unsqueeze(-1) * hidden).sum(dim=1)
            result = self.projection(pooled)
        finite_tensor(result, "sentence_encoding")
        return result


class LocalByteDecoder(nn.Module):
    """No raw-history bypass: condition on a latent and this sentence's prefix."""

    def __init__(self, spec: CodecSpec) -> None:
        super().__init__()
        self.spec = spec
        self.tokens = nn.Embedding(
            SentenceByteCodec.input_size, spec.decoder_width, padding_idx=SentenceByteCodec.pad
        )
        self.positions = nn.Embedding(spec.max_sentence_bytes + 1, spec.decoder_width)
        self.condition = nn.Linear(spec.width, spec.decoder_width, bias=False)
        self.blocks = _blocks(spec.decoder_width, spec.decoder_layers, spec.head_width)
        self.readout = nn.Linear(spec.decoder_width, SentenceByteCodec.output_size, bias=False)

    def forward(self, latent: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        padding = _tokens(prefix, self.spec.max_sentence_bytes, complete=False)
        if latent.shape != (prefix.shape[0], self.spec.width) or latent.device != prefix.device:
            raise ValueError("decoder latent shape/device must match its local prefix")
        if not latent.is_floating_point():
            raise TypeError("decoder latent must be floating")
        finite_tensor(latent, "decoder_condition")
        positions = torch.arange(prefix.shape[1], device=prefix.device)
        hidden = (
            self.tokens(prefix) + self.positions(positions) + self.condition(latent).unsqueeze(1)
        )
        causal = torch.ones(
            prefix.shape[1], prefix.shape[1], dtype=torch.bool, device=prefix.device
        ).triu(1)
        for block in self.blocks:
            hidden = block(hidden, src_mask=causal, src_key_padding_mask=padding, is_causal=True)
        logits = self.readout(hidden).float()
        finite_tensor(logits, "decoder_logits")
        return logits


class NeuralSentenceCodec(nn.Module):
    """A reconstruction path, explicitly not a next-sentence generation model."""

    def __init__(self, spec: CodecSpec, law_spec: LawSpec) -> None:
        super().__init__()
        if type(spec) is not CodecSpec or type(law_spec) is not LawSpec:
            raise TypeError("codec requires explicit codec and law specs")
        if spec.width != law_spec.width:
            raise ValueError("codec and law width must match")
        self.spec, self.law_spec = spec, law_spec
        self.encoder = SentenceEncoder(spec)
        self.decoder = LocalByteDecoder(spec)
        self._encoder_frozen = False

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def posterior(self, complete: torch.Tensor) -> DirectionalMagnitudeLaw:
        return DirectionalMagnitudeLaw.from_vector(self.law_spec, self.encoder(complete))

    def reconstruct(self, complete: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        """Training/oracle reconstruction only; never label as generator likelihood."""
        return self.decoder(self.posterior(complete).rsample(), prefix)

    def freeze_encoder(self) -> None:
        """Freeze only the sentence encoder; the independent decoder remains trainable."""
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self._encoder_frozen = True

    def get_extra_state(self) -> dict[str, bool]:
        """Freezing is runtime state, not recoverable from parameter values alone."""
        return {"encoder_frozen": self._encoder_frozen}

    def set_extra_state(self, state: object) -> None:
        if type(state) is not dict or set(state) != {"encoder_frozen"}:
            raise ValueError("codec freeze state fields are invalid")
        frozen = state["encoder_frozen"]
        if type(frozen) is not bool:
            raise ValueError("codec encoder_frozen must be boolean")
        self._encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)
