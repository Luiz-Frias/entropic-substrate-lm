"""Causal sentence-clock Transformer with independent latent carrier policy."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.predictive_state.metric import finite_tensor
from entropic_substrate.sentence_prediction.numerics import InterfaceMode, checked_coordinates

type SpineConsumer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True, slots=True)
class SpineSpec:
    width: int = 768
    layers: int = 12
    head_width: int = 64
    context_sentences: int = 64
    carrier: InterfaceMode = "sphere"
    alpha_init: float = 0.05

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1
            for n in (self.width, self.layers, self.head_width, self.context_sentences)
        ):
            raise ValueError("spine dimensions and context must be positive integers")
        if self.head_width < 2 or self.width % self.head_width:
            raise ValueError("spine head width must divide its width and be >= 2")
        if self.carrier not in ("sphere", "magnitude"):
            raise ValueError("unknown spine carrier")
        if (
            type(self.alpha_init) not in (int, float)
            or not math.isfinite(self.alpha_init)
            or self.alpha_init <= 0
        ):
            raise ValueError("spine alpha must be finite and positive")


class CarrierBlock(nn.Module):
    """Shared nGPT-style proposal; only carrier retraction is varied.

    Zero sublayer/query views remain zero, with a diagnostic count, rather than
    being called unit points. Normalized-dot-product attention is defined on
    those zero views; a spherical carrier itself requires a nonzero direction.
    """

    def __init__(self, spec: SpineSpec) -> None:
        super().__init__()
        self.spec = spec
        self.qkv = nn.Linear(spec.width, 3 * spec.width, bias=False)
        self.attn_out = nn.Linear(spec.width, spec.width, bias=False)
        self.up = nn.Linear(spec.width, 4 * spec.width, bias=False)
        self.down = nn.Linear(4 * spec.width, spec.width, bias=False)
        scale = spec.width**-0.5
        self.attn_alpha = nn.Parameter(torch.full((spec.width,), scale))
        self.mlp_alpha = nn.Parameter(torch.full((spec.width,), scale))
        self.qk_scale = nn.Parameter(
            torch.full((spec.width // spec.head_width,), spec.head_width**0.5)
        )
        self.zero_views = 0

    def _view(self, values: torch.Tensor) -> torch.Tensor:
        coordinates = checked_coordinates(values, allow_zero=True)
        self.zero_views += int(coordinates.zero_mask.sum())
        return coordinates.direction

    def _residual(
        self, current: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        a, b = self._view(current), self._view(proposal)
        step = (alpha * (self.spec.alpha_init / self.spec.width**-0.5)).abs() * (b - a)
        if self.spec.carrier == "sphere":
            return checked_coordinates(a + step).direction
        return current + step

    def forward(self, current: torch.Tensor) -> torch.Tensor:
        self.zero_views = 0
        batch, time, _ = current.shape
        heads = self.spec.width // self.spec.head_width
        qkv = self.qkv(self._view(current)).reshape(batch, time, 3, heads, self.spec.head_width)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        with torch.autocast(device_type=current.device.type, enabled=False):
            q, k = self._view(q), self._view(k)
            q = q * self.qk_scale.abs().view(1, -1, 1, 1)
            attended = F.scaled_dot_product_attention(
                q, k, v.to(q.dtype), is_causal=True, dropout_p=0.0, scale=1.0
            )
        attended = attended.transpose(1, 2).reshape(batch, time, self.spec.width)
        current = self._residual(current, self.attn_out(attended), self.attn_alpha)
        return self._residual(
            current, self.down(F.gelu(self.up(self._view(current)))), self.mlp_alpha
        )

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Same parameter normalization in both carrier arms, never state clipping."""
        for linear in (self.qkv, self.up):
            linear.weight.copy_(checked_coordinates(linear.weight, allow_zero=True).direction)
        for linear in (self.attn_out, self.down):
            linear.weight.copy_(checked_coordinates(linear.weight.T, allow_zero=True).direction.T)


class SentenceSpine(nn.Module):
    """One state per completed sentence plus an explicit zero-byte BOS state."""

    def __init__(self, input_width: int, spec: SpineSpec) -> None:
        super().__init__()
        if type(input_width) is not int or input_width < 2 or type(spec) is not SpineSpec:
            raise ValueError("spine requires an input width and explicit spec")
        self.input_width, self.spec = input_width, spec
        self.projection = nn.Linear(input_width, spec.width, bias=False)
        self.bos = nn.Parameter(torch.randn(spec.width) / spec.width**0.5)
        self.positions = nn.Embedding(spec.context_sentences + 1, spec.width)
        self.blocks = nn.ModuleList(CarrierBlock(spec) for _ in range(spec.layers))

    def initial_states(self, sentences: torch.Tensor) -> torch.Tensor:
        if not isinstance(cast(object, sentences), torch.Tensor):
            raise TypeError("spine sentence states must be a tensor")
        if sentences.ndim != 3 or sentences.shape[0] < 1 or sentences.shape[-1] != self.input_width:
            raise ValueError("spine sentence batch has incompatible shape")
        if sentences.shape[1] > self.spec.context_sentences:
            raise ValueError("spine sentence context exceeds its registered limit")
        if not sentences.is_floating_point():
            raise TypeError("spine sentence states must be floating")
        finite_tensor(sentences, "spine_inputs")
        embedded = self.projection(sentences)
        start = self.bos.view(1, 1, -1).expand(sentences.shape[0], 1, -1)
        states = torch.cat((start, embedded), dim=1)
        states = states + self.positions(torch.arange(states.shape[1], device=states.device))
        finite_tensor(states, "spine_initial_state")
        return checked_coordinates(states).direction if self.spec.carrier == "sphere" else states

    def forward(
        self, sentences: torch.Tensor, *, consumer: SpineConsumer | None = None
    ) -> torch.Tensor:
        states = self.initial_states(sentences)
        for index, block in enumerate(self.blocks):
            proposed = block(states)
            if consumer is not None and index == len(self.blocks) - 1:
                proposed = consumer(states, proposed)
                if (
                    proposed.shape != states.shape
                    or proposed.dtype != states.dtype
                    or proposed.device != states.device
                ):
                    raise ValueError("spine consumer must preserve shape, dtype, and device")
                if self.spec.carrier == "sphere":
                    coordinates = checked_coordinates(proposed, allow_zero=True)
                    if bool(coordinates.zero_mask.any()) or bool(
                        (coordinates.log_radius.abs() > 1e-5).any()
                    ):
                        raise ValueError("spine consumer violated the unit carrier contract")
            finite_tensor(proposed, "spine_state")
            states = proposed
        # No final RMS/UnitNorm overwrite in the magnitude-preserving arm.
        return states

    @torch.no_grad()
    def normalize_weights(self) -> None:
        self.projection.weight.copy_(
            checked_coordinates(self.projection.weight, allow_zero=True).direction
        )
        self.positions.weight.copy_(
            checked_coordinates(self.positions.weight, allow_zero=True).direction
        )
        self.bos.copy_(checked_coordinates(self.bos, allow_zero=True).direction)
        for block in self.blocks:
            cast(CarrierBlock, block).normalize_weights()
