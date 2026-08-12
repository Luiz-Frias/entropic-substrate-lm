"""Byte-level and GPT-2-BPE language models built from a :class:`ModelConfig`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.config import ModelConfig
from entropic_substrate.geometry.kappa import clamp_kappa
from entropic_substrate.init.fisher import (
    build_fisher_init,
    load_init,
    random_sphere_init,
)
from entropic_substrate.layers.kappa_norm import GatedKappaNorm, KappaNormStats, UnitNorm
from entropic_substrate.models.blocks import CausalSelfAttention, build_block, build_norm

__all__ = [
    "LMOutput",
    "LanguageModel",
    "build_model",
    "count_parameters",
    "vocab_size_for",
]


def vocab_size_for(cfg: ModelConfig) -> int:
    """Resolve the vocabulary size implied by ``cfg`` (explicit value wins)."""
    if cfg.vocab_size is not None:
        return cfg.vocab_size
    if cfg.tokenizer == "byte":
        return 256
    if cfg.tokenizer == "gpt2":
        import tiktoken

        return int(tiktoken.get_encoding("gpt2").n_vocab)
    msg = f"unknown tokenizer {cfg.tokenizer!r}"
    raise ValueError(msg)


@dataclass(slots=True)
class LMOutput:
    """Forward result: logits, task loss, MoE aux loss and router telemetry."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    kappa_stats: list[KappaNormStats] = field(default_factory=list)


class LanguageModel(nn.Module):
    """Decoder-only LM. ``ByteLM`` and ``BPELM`` are the same class, different config.

    ``cfg.tokenizer='byte'`` gives a 256-way head over raw bytes;
    ``cfg.tokenizer='gpt2'`` gives the 50257-way tiktoken head (tied embeddings
    by default so the parameter budget stays comparable).
    """

    # Registered as None on the non-hypersphere path.
    sz: nn.Parameter | None

    def __init__(self, cfg: ModelConfig, fisher_source: bytes | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size_for(cfg)
        self.hypersphere = cfg.block == "ngpt"

        self.wte = nn.Embedding(self.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([build_block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f: nn.Module = UnitNorm() if self.hypersphere else build_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, self.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        if self.hypersphere:
            # nGPT logit scaling `sz`: q,k,h are unit-norm, so logits need a
            # learnable per-class temperature to recover useful dynamic range.
            self.sz_scale = cfg.d_model**-0.5
            self.sz = nn.Parameter(torch.full((self.vocab_size,), self.sz_scale))
        else:
            self.register_parameter("sz", None)

        self.apply(self._init_weights)
        self._init_embeddings(fisher_source)
        if self.hypersphere:
            self.normalize_weights()

    # ---------------------------------------------------------------- init
    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.d_model**-0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            # torch stubs type Linear.bias as Parameter, but bias=False makes it None.
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _init_embeddings(self, fisher_source: bytes | None) -> None:
        mode = self.cfg.embedding_init
        if mode == "default":
            return
        if mode == "random_sphere":
            weights = random_sphere_init(self.vocab_size, self.cfg.d_model)
        elif mode == "fisher_sqrt":
            weights = self._fisher_weights(fisher_source)
        else:
            msg = f"unknown embedding_init {mode!r}"
            raise ValueError(msg)
        with torch.no_grad():
            self.wte.weight.copy_(weights.to(self.wte.weight.dtype))

    def _fisher_weights(self, fisher_source: bytes | None) -> torch.Tensor:
        if self.cfg.tokenizer != "byte":
            msg = (
                "fisher_sqrt init is defined over the 256-symbol byte alphabet; "
                f"tokenizer={self.cfg.tokenizer!r} is not supported"
            )
            raise ValueError(msg)
        path = self.cfg.fisher_init_path
        if path is not None and Path(path).exists():
            weights = load_init(path)
            expected = (256, self.cfg.d_model)
            if tuple(weights.shape) != expected:
                msg = f"{path} has shape {tuple(weights.shape)}, expected {expected}"
                raise ValueError(msg)
            return weights
        if fisher_source is None:
            msg = (
                "fisher_sqrt init needs either an existing fisher_init_path "
                "(run `entropic-substrate init-fisher`) or a corpus sample "
                "passed as fisher_source"
            )
            raise ValueError(msg)
        return build_fisher_init(fisher_source, self.cfg.d_model)

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Unit-norm every embedding-dimension weight row (nGPT constraint)."""
        if not self.hypersphere:
            return
        for block in self.blocks:
            # torch stubs give ModuleList items as Tensor | Module; both blocks
            # define normalize_weights.
            block.normalize_weights()  # pyright: ignore[reportCallIssue]
        for weight in {
            id(self.wte.weight): self.wte.weight,
            id(self.lm_head.weight): self.lm_head.weight,
        }.values():
            weight.copy_(weight / weight.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))

    def kappa_modules(self) -> list[GatedKappaNorm]:
        """All gated-curvature norms in the model (empty unless E3)."""
        return [m for m in self.modules() if isinstance(m, GatedKappaNorm)]

    def kappa_attention_modules(self) -> list[CausalSelfAttention]:
        """Attention modules carrying a learnable per-head curvature (E3)."""
        return [
            m for m in self.modules() if isinstance(m, CausalSelfAttention) and m.kappa is not None
        ]

    def collect_aux_loss(self) -> torch.Tensor:
        """Sum of the load-balancing losses recorded by the last forward pass."""
        mods = self.kappa_modules()
        if not mods:
            return torch.zeros((), device=self.wte.weight.device)
        return torch.stack([m.aux_loss for m in mods]).sum()

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Total parameter count, optionally excluding the token embedding."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.wte.weight.numel()
        return total

    # ------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> LMOutput:
        """``idx``/``targets``: ``(B, T)`` int64 token ids."""
        _, t = idx.shape
        if t > self.cfg.block_size:
            msg = f"sequence length {t} exceeds block_size {self.cfg.block_size}"
            raise ValueError(msg)

        h = self.drop(self.wte(idx))
        if self.hypersphere:
            h = self.norm_f(h)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)

        logits = self.lm_head(h)
        if self.hypersphere and self.sz is not None:
            logits = logits * (self.sz * (1.0 / self.sz_scale))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                targets.reshape(-1),
            )

        mods = self.kappa_modules()
        aux = self.collect_aux_loss() if mods else None
        stats = [m.stats for m in mods]
        # E3: per-head attention curvatures ride the same telemetry channel;
        # one stats row per layer, gate fields empty (heads are the mixture).
        for attn in self.kappa_attention_modules():
            head_kappa = attn.kappa
            if head_kappa is None:
                continue
            stats.append(
                KappaNormStats(
                    kappa=[round(float(v), 6) for v in clamp_kappa(head_kappa.detach()).cpu()],
                    gate_fraction=[],
                    router_prob=[],
                    aux_loss=0.0,
                    entropy=0.0,
                )
            )
        return LMOutput(
            logits=logits,
            loss=loss,
            aux_loss=aux,
            kappa_stats=stats,
        )

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling (used for qualitative checks)."""
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.block_size :]
            logits = self(window).logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.shape[-1]))[0][:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


def build_model(cfg: ModelConfig, fisher_source: bytes | None = None) -> LanguageModel:
    """Construct a :class:`LanguageModel` from config (ByteLM or BPELM)."""
    return LanguageModel(cfg, fisher_source=fisher_source)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
