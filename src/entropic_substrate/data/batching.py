"""Contiguous-sequence batch sampling over a flat token/byte stream."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "DATA_ORDER_DIGEST_BATCHES",
    "Corpus",
    "CorpusMeta",
    "SequenceSampler",
    "TokenSplit",
]

#: How many train batches :meth:`SequenceSampler.data_order_digest` attests.
#: Enough to evidence order identity between paired runs (the E1
#: preregistration's pairing claim — docs/e1-preregistration.md Appendix A)
#: without hashing an entire run; the window is fixed so the digest is stable
#: from batch 32 onward.
DATA_ORDER_DIGEST_BATCHES = 32


@dataclass(slots=True)
class CorpusMeta:
    """Everything the model / metric layer needs to know about the data."""

    tokenizer: str
    vocab_size: int
    #: utf-8 bytes represented by one token *measured on the eval split*.
    #: Exactly 1.0 for byte models; ~4.x and never assumed for GPT-2 BPE.
    bytes_per_token: float
    train_tokens: int
    val_tokens: int
    val_bytes: int
    source: str


@dataclass(slots=True)
class TokenSplit:
    """A flat 1-D token stream for one split."""

    tokens: np.ndarray

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


@dataclass(slots=True)
class Corpus:
    """Train/val token streams plus metadata."""

    train: TokenSplit
    val: TokenSplit
    meta: CorpusMeta


class SequenceSampler:
    """Sample ``(batch, seq_len)`` next-token-prediction pairs.

    ``train=True`` draws uniform random offsets from a seeded generator
    (reproducible across resumes because the generator state is saved with the
    checkpoint); ``train=False`` walks a fixed, evenly spaced grid so eval
    numbers are comparable step to step and across experiments.
    """

    def __init__(
        self,
        split: TokenSplit,
        batch_size: int,
        seq_len: int,
        *,
        train: bool = True,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        n = len(split)
        if n < seq_len + 2:
            msg = f"split has {n} tokens, need at least {seq_len + 2}"
            raise ValueError(msg)
        self.split = split
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train = train
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._max_start = n - seq_len - 1
        self._eval_cursor = 0
        self._order_hasher = hashlib.sha256()
        self._order_batches = 0

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        """Serialisable sampler state (for exact resume).

        The data-order attestation is DELIBERATELY not serialised: a resumed
        run has not observed batches 1..N from a fresh start, so it must not
        attest to them — the trainer only emits the digest for fresh runs.
        """
        return {
            "generator": self.generator.get_state(),
            "eval_cursor": self._eval_cursor,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int]) -> None:
        """Restore sampler state produced by :meth:`state_dict`."""
        gen = state.get("generator")
        if isinstance(gen, torch.Tensor):
            self.generator.set_state(gen)
        cursor = state.get("eval_cursor", 0)
        self._eval_cursor = int(cursor) if isinstance(cursor, int) else 0

    def _starts(self) -> np.ndarray:
        if self.train:
            idx = torch.randint(
                0,
                self._max_start + 1,
                (self.batch_size,),
                generator=self.generator,
            )
            starts = idx.numpy()
            if self._order_batches < DATA_ORDER_DIGEST_BATCHES:
                # '<i8' pins width AND byte order: the digest is pairing
                # evidence that travels via captured JSONLs, so it must be
                # comparable across machines, not just across processes.
                self._order_hasher.update(starts.astype("<i8").tobytes())
                self._order_batches += 1
            return starts
        stride = max(self._max_start // max(self.batch_size, 1), 1)
        base = (self._eval_cursor * self.batch_size * stride) % (self._max_start + 1)
        self._eval_cursor += 1
        offsets = (base + np.arange(self.batch_size) * stride) % (self._max_start + 1)
        return offsets

    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one ``(inputs, targets)`` pair of shape ``(B, T)`` int64."""
        starts = self._starts()
        tokens = self.split.tokens
        xs = np.stack([tokens[s : s + self.seq_len] for s in starts])
        ys = np.stack([tokens[s + 1 : s + 1 + self.seq_len] for s in starts])
        x = torch.from_numpy(xs.astype(np.int64))
        y = torch.from_numpy(ys.astype(np.int64))
        if self.device.type == "cuda":
            # pin_memory is a CUDA page-locked host-buffer optimization and is
            # only meaningful on that path. MEASURED 2026-08-20 (torch 2.13,
            # mps): the previous `!= "cpu"` guard sent mps through
            # pin_memory().to("mps"), which raises "Attempted to set the
            # storage of a tensor on device cpu to a storage on different
            # device mps:0" — first batch, every time.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        elif self.device.type != "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y

    def data_order_digest(self) -> tuple[str, int] | None:
        """Attestation of this sampler's train-batch source offsets.

        Returns ``(sha256_hex, batches_attested)`` once at least one train
        batch has been drawn; the window caps at
        :data:`DATA_ORDER_DIGEST_BATCHES`, after which the digest is stable.
        ``None`` for eval samplers — the eval grid is deterministic by
        construction, and attesting it would let an eval digest masquerade as
        train-pairing evidence — and before any batch is drawn.
        """
        if not self.train or self._order_batches == 0:
            return None
        return self._order_hasher.hexdigest(), self._order_batches

    def reset_eval(self) -> None:
        """Rewind the deterministic eval grid."""
        self._eval_cursor = 0

    def iter_batches(self, n: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``n`` batches."""
        for _ in range(n):
            yield self.batch()
