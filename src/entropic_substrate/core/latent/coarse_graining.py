"""D4 causal segment summaries with explicit boundary and work measurements.

All modes share a macro recurrence and readout. ``identity`` means singleton
segments (the identity partition), not bypassing that matched module. Partial
segments never flush at end of input: future input cannot alter a prefix.
The tracer evaluates macro candidates densely even when few are committed;
its boundary counts therefore are not a claim of sparse execution savings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

_MODES = frozenset(
    {
        "identity",
        "fixed",
        "learned",
        "always",
        "never",
        "random",
        "fixed_duration",
        "whitespace",
        "punctuation",
    }
)
_HEURISTICS = frozenset({"whitespace", "punctuation"})
_WHITESPACE = (9, 10, 11, 12, 13, 32)
_PUNCTUATION = tuple(b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@dataclass(frozen=True, slots=True)
class CoarseGrainingDiagnostics:
    """Hard event counts and differentiable router probabilities.

    ``durations`` is zero away from closing positions. ``pending_lengths``
    counts the unfinished trailing segment separately. Work counters count
    batch elements, not Python calls, tokens saved, or FLOPs.
    """

    boundaries: torch.Tensor
    probabilities: torch.Tensor
    durations: torch.Tensor
    segment_counts: torch.Tensor
    pending_lengths: torch.Tensor
    boundary_rate: torch.Tensor
    boundary_entropy: torch.Tensor
    macro_updates: int
    macro_candidate_evaluations: int


@dataclass(frozen=True, slots=True)
class CoarseGrainingResult:
    """Fused causal trajectory and a separately weighted router objective."""

    states: torch.Tensor
    diagnostics: CoarseGrainingDiagnostics
    rate_penalty: torch.Tensor


def _linear32(layer: nn.Linear, values: torch.Tensor) -> torch.Tensor:
    optional_bias = cast("torch.Tensor | None", layer.bias)
    bias = None if optional_bias is None else optional_bias.float()
    return F.linear(values, layer.weight.float(), bias)


class CausalCoarseGrainer(nn.Module):
    """Compose current microstates with completed-segment macro memory.

    Learned probabilities use only ``(h_t, h_{t-1})`` plus a causal EMA.
    A straight-through boundary carries task gradients through commit and
    accumulator-reset decisions. The rate penalty targets ``1 / window``
    using hard rates forward and probability derivatives backward; it is
    returned unweighted for an explicit training-objective owner.

    ``byte_ids`` is required only by declared heuristic controls. It is
    forbidden for learned routing to quarantine the byte alphabet boundary.
    """

    def __init__(
        self,
        *,
        width: int,
        mode: str = "fixed",
        window: int = 8,
        smoothing: float = 0.5,
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        if mode not in _MODES:
            raise ValueError(f"unsupported coarse-graining mode: {mode}")
        if not math.isfinite(smoothing) or not 0 <= smoothing < 1:
            raise ValueError("smoothing must be finite and in [0, 1)")
        if type(random_seed) is not int or not 0 <= random_seed < 2**63:
            raise ValueError("random_seed must be an integer in [0, 2**63)")
        self.width, self.mode, self.window = width, mode, window
        self.smoothing, self.random_seed = smoothing, random_seed
        self.router = nn.Linear(2 * width, 1)
        self.macro = nn.Linear(2 * width, width)
        self.readout = nn.Linear(width, width, bias=False)
        # Avoid an arbitrary initial router-rate offset; preserve input sensitivity.
        nn.init.constant_(self.router.bias, math.log(1 / max(window - 1, 1)))

    def _validate(self, values: torch.Tensor, byte_ids: torch.Tensor | None) -> None:
        if not values.is_floating_point():
            raise TypeError("coarse-graining states must have floating dtype")
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] != self.width:
            raise ValueError("coarse-graining states require (positive batch, time, width)")
        if not torch.isfinite(values).all():
            raise ValueError("coarse-graining states must be finite")
        if self.mode not in _HEURISTICS:
            if byte_ids is not None:
                raise ValueError("byte_ids are admitted only by heuristic controls")
            return
        if byte_ids is None:
            raise ValueError("heuristic controls require byte_ids")
        if (
            byte_ids.dtype != torch.long
            or byte_ids.shape != values.shape[:2]
            or byte_ids.device != values.device
        ):
            raise ValueError("byte_ids must be aligned int64 byte observations")
        if torch.any((byte_ids < 0) | (byte_ids > 255)):
            raise ValueError("byte_ids must lie in [0, 255]")

    def _control(self, values: torch.Tensor, byte_ids: torch.Tensor | None) -> torch.Tensor:
        batch, time, _ = values.shape
        if self.mode in {"identity", "always"}:
            return values.new_ones((batch, time))
        if self.mode == "never":
            return values.new_zeros((batch, time))
        if self.mode in {"fixed", "fixed_duration"}:
            return (
                ((torch.arange(time, device=values.device) + 1) % self.window == 0)
                .expand(batch, time)
                .float()
            )
        if self.mode == "random":
            generator = torch.Generator(device="cpu").manual_seed(self.random_seed)
            # Time-major draws preserve prefixes when the sequence length changes.
            return (torch.rand(time, batch, generator=generator).T < 1 / self.window).to(
                device=values.device, dtype=torch.float32
            )
        if byte_ids is None:
            raise ValueError("heuristic controls require byte_ids")
        symbols = _WHITESPACE if self.mode == "whitespace" else _PUNCTUATION
        return torch.isin(byte_ids, byte_ids.new_tensor(symbols)).float()

    def forward(
        self, values: torch.Tensor, *, byte_ids: torch.Tensor | None = None
    ) -> CoarseGrainingResult:
        """Return same-shape causal states; accept an empty time axis explicitly."""
        self._validate(values, byte_ids)
        with torch.autocast(device_type=values.device.type, enabled=False):
            return self._forward32(values, byte_ids)

    def _forward32(
        self, values: torch.Tensor, byte_ids: torch.Tensor | None
    ) -> CoarseGrainingResult:
        states = values.float()
        batch, time, width = states.shape
        controls = None if self.mode == "learned" else self._control(states, byte_ids)
        macro = states.new_zeros((batch, width))
        segment_sum, previous = torch.zeros_like(macro), torch.zeros_like(macro)
        count = states.new_zeros((batch, 1))
        duration = torch.zeros(batch, dtype=torch.long, device=states.device)
        probability = states.new_full((batch, 1), 1 / self.window)
        outputs: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        durations: list[torch.Tensor] = []
        for position in range(time):
            current = states[:, position]
            if controls is None:
                raw = _linear32(self.router, torch.cat((current, previous), dim=-1)).sigmoid()
                probability = self.smoothing * probability + (1 - self.smoothing) * raw
                hard = (probability >= 0.5).float()
                gate = hard + probability - probability.detach()
            else:
                gate = controls[:, position, None]
                probability = gate
            segment_sum, count = segment_sum + current, count + 1
            duration = duration + 1
            summary = segment_sum / count.clamp_min(1)
            candidate = _linear32(self.macro, torch.cat((summary, macro), dim=-1)).tanh()
            macro = gate * candidate + (1 - gate) * macro
            outputs.append(current + _linear32(self.readout, macro))
            boundary = gate.detach().squeeze(-1) > 0.5
            durations.append(torch.where(boundary, duration, 0))
            duration = torch.where(boundary, 0, duration)
            segment_sum, count = (1 - gate) * segment_sum, (1 - gate) * count
            gates.append(gate.squeeze(-1))
            probabilities.append(probability.squeeze(-1))
            previous = current
        gates_tensor = torch.stack(gates, dim=1) if time else states.new_empty((batch, 0))
        probs = torch.stack(probabilities, dim=1) if time else states.new_empty((batch, 0))
        boundaries = gates_tensor.detach() > 0.5
        lengths = (
            torch.stack(durations, dim=1)
            if time
            else torch.empty((batch, 0), device=states.device, dtype=torch.long)
        )
        rate = gates_tensor.mean(dim=1) if time else states.new_zeros(batch)
        observed_probs = probs.detach()
        entropy = (
            (torch.special.entr(observed_probs) + torch.special.entr(1 - observed_probs)).mean(
                dim=1
            )
            if time
            else states.new_zeros(batch)
        )
        penalty = (
            (rate - 1 / self.window).square().mean()
            if time and self.mode == "learned"
            else states.sum() * 0
        )
        diagnostics = CoarseGrainingDiagnostics(
            boundaries=boundaries,
            probabilities=probs,
            durations=lengths,
            segment_counts=boundaries.sum(dim=1),
            pending_lengths=duration,
            boundary_rate=rate,
            boundary_entropy=entropy,
            macro_updates=int(boundaries.sum().item()),
            macro_candidate_evaluations=batch * time,
        )
        result = torch.stack(outputs, dim=1) if time else states
        if not torch.isfinite(result).all() or not torch.isfinite(penalty):
            raise ValueError("coarse-graining output or objective is nonfinite")
        return CoarseGrainingResult(
            states=result.to(values.dtype), diagnostics=diagnostics, rate_penalty=penalty
        )
