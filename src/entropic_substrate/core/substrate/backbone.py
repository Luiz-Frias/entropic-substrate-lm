"""Byte-level and GPT-2-BPE language models built from a :class:`ModelConfig`."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.core.geometry.kappa import clamp_kappa
from entropic_substrate.core.geometry.kappa_norm import GatedKappaNorm, KappaNormStats, UnitNorm
from entropic_substrate.core.substrate.blocks import CausalSelfAttention, build_block, build_norm
from entropic_substrate.core.substrate.field_encoder import FieldEncoder, FieldEncoding
from entropic_substrate.core.substrate.fisher import (
    build_fisher_init,
    load_init,
    random_sphere_init,
)
from entropic_substrate.kernel.runtime.config import ModelConfig

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
    # E1.5c: the field encoder's own ``(B, T, 256)`` next-byte logits, from
    # the SAME pass that produced the injected ``z`` — so ``Trainer.
    # evaluate``'s calibration diagnostics read them here instead of running
    # ``field_encoder.field_logits(x)`` as a second forward over the same
    # input (CR PR #79). ``None`` under ``state_source="table"``, where no
    # field encoder exists at all — absent channel is absent, the same
    # convention ``aux_loss``/``kappa_stats`` already use for features most
    # arms do not carry. AMBIENT dtype, not fp32: see
    # :class:`~entropic_substrate.models.field_encoder.FieldEncoding` for
    # why the cast belongs to the consumer.
    field_logits: torch.Tensor | None = None


class LanguageModel(nn.Module):
    """Decoder-only LM. ``ByteLM`` and ``BPELM`` are the same class, different config.

    ``cfg.tokenizer='byte'`` gives a 256-way head over raw bytes;
    ``cfg.tokenizer='gpt2'`` gives the 50257-way tiktoken head (tied embeddings
    by default so the parameter budget stays comparable).
    """

    # Registered as None on the non-hypersphere path.
    sz: nn.Parameter | None
    # Registered as None unless cfg.state_source == "field" (E1.5c).
    field_encoder: FieldEncoder | None

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
        self._apply_embedding_freeze()

        # E1.5c: built AFTER the backbone's own init/normalize/freeze
        # sequence above, so FieldEncoder's internal `self.apply(...)` (its
        # own field_d_model-scaled init) is never re-walked by the
        # backbone's own `self.apply(self._init_weights)` call, which
        # already ran. wte/lm_head stay in place as the output classifier
        # (still Fisher-seeded above when embedding_init="fisher_sqrt") —
        # state_source only swaps which tensor becomes `h` at the start of
        # forward(), never the classifier. embedding_freeze="input"/"both"
        # is refused above for this same reason (self.wte.weight stops
        # being the input representation); gram_lambda gets the symmetric
        # guard immediately below, for the same root cause — L2 finding,
        # 2026-08-19.
        if cfg.state_source == "field":
            if cfg.gram_lambda != 0.0:
                # train.py's Gram penalty (_gram_penalty_terms) is defined
                # over normalized_gram(self.wte.weight) — the input table
                # E1.5b was built against. Under state_source="field" that
                # tensor is not what generates the model's input
                # (field_encoder.wte is, and forward() never reads
                # self.wte(idx) on this path), so the penalty would
                # silently co-optimize a tensor disconnected from the
                # measured "preserve input geometry" claim. Measured: with
                # gram_lambda=0.5, gram_penalty moved self.wte.weight across
                # steps (0 -> 5.72039 -> 1.58559 -> 1.14893 -> 1.09639)
                # while field_encoder was never referenced by the penalty.
                # No registered arm combines these two axes.
                msg = (
                    "gram_lambda != 0.0 requires state_source='table': the "
                    "Gram penalty is defined over self.wte.weight, which is "
                    "not the input representation under state_source='field' "
                    "(field_encoder.wte is) — the penalty would silently "
                    "co-optimize a tensor the forward pass never reads"
                )
                raise ValueError(msg)
            if cfg.field_mode == "frozen" and cfg.field_loss_lambda != 0.0:
                # Frozen leaves get no gradient, so λ·L_field burns one
                # encoder forward per training step for a term that cannot
                # train anything — contaminating the compute-accounting
                # lane (FLOPs/step) the E1.5c contract reports. Frozen-field
                # CE telemetry is the EVAL-time job (FOR-925), not a
                # train-loss rider. CR PR #74 + P6 fix-agent converged on
                # this gap independently.
                msg = (
                    "field_mode='frozen' requires field_loss_lambda=0.0: frozen "
                    "field weights receive no gradient, so the calibration term "
                    "cannot train anything and only adds a wasted encoder forward "
                    "per step — frozen-field CE belongs to eval telemetry"
                )
                raise ValueError(msg)
            self.field_encoder = FieldEncoder(cfg)
        elif cfg.state_source == "table":
            if (
                cfg.field_loss_lambda != 0.0
                or cfg.field_ckpt_path is not None
                or cfg.field_mode != "joint"
                or cfg.field_state != "sqrt_prob"
            ):
                # No field encoder exists on this path — a non-default
                # calibration knob (or, P7/FOR-924: a non-default
                # field_state — F/P/H selects a CONSTRUCTION of a field
                # encoder's output, meaningless without one) would be
                # silently ignored, which reads as "configured" while
                # measuring nothing (loud-refusal posture; same class as
                # the gram_lambda × field guard).
                msg = (
                    "field_loss_lambda/field_ckpt_path/field_mode/field_state require "
                    "state_source='field'; under state_source='table' no field "
                    "encoder exists and these knobs would be silently ignored"
                )
                raise ValueError(msg)
            self.field_encoder = None
        else:
            msg = f"unknown state_source {cfg.state_source!r}"
            raise ValueError(msg)

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
            if not self.cfg.tie_embeddings:
                # Untied: the output table gets no init pass of its own
                # otherwise (only `_init_weights`'s generic normal draw), so
                # e1-5a-output's "frozen at its Fisher point" reading would
                # be false — this seeds it from the SAME table the tied E1
                # arm would have given it.
                self.lm_head.weight.copy_(weights.to(self.lm_head.weight.dtype))

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

    def _apply_embedding_freeze(self) -> None:
        """E1.5a: freeze the named half of the embedding at its init value.

        Runs at model-build time, AFTER init + the first ``normalize_weights``
        pass (so a frozen table is captured at its true, already-normalised
        init point) and BEFORE ``Trainer._build_optimizer`` ever runs
        (train.py:190) — the optimiser's existing ``if not
        param.requires_grad: continue`` filter therefore excludes frozen
        params by construction, with no freeze-awareness of its own needed.

        "input"/"output" require ``tie_embeddings=False``: a tied table is
        one ``nn.Parameter`` shared by both roles (``__init__`` sets
        ``self.lm_head.weight = self.wte.weight``), so it cannot be frozen on
        only ONE of the two. "both" has no such contradiction and is allowed
        tied or untied.
        """
        mode = self.cfg.embedding_freeze
        if mode == "none":
            return
        if mode not in ("input", "output", "both"):
            msg = f"unknown embedding_freeze {mode!r}"
            raise ValueError(msg)
        if mode in ("input", "output") and self.cfg.tie_embeddings:
            msg = (
                f"embedding_freeze={mode!r} requires tie_embeddings=False: a tied "
                "table cannot be frozen on only one of its two roles (wte.weight "
                "and lm_head.weight are the SAME nn.Parameter when tied)"
            )
            raise ValueError(msg)
        if mode in ("input", "both") and self.cfg.state_source == "field":
            # L2 finding, 2026-08-19: under state_source="field" wte.weight
            # is not the input representation (field_encoder.wte is —
            # field_encoder.py, LanguageModel.forward's field_encoder
            # branch); freezing it is a silent no-op on the axis this mode
            # promises to control (measured: field_encoder.wte stayed
            # requires_grad=True and trained fully unfrozen while
            # embedding_freeze="input" froze self.wte.weight). "output" is
            # exempt: lm_head.weight remains the real output classifier
            # under state_source="field" regardless (see the field_encoder
            # branch below), so freezing it keeps its name's promise.
            msg = (
                f"embedding_freeze={mode!r} requires state_source='table': "
                "under state_source='field' self.wte.weight is not the input "
                "representation (field_encoder.wte is), so freezing it is a "
                "silent no-op on the axis this mode promises to control — "
                "no registered arm combines these two axes"
            )
            raise ValueError(msg)
        if mode in ("input", "both"):
            self.wte.weight.requires_grad_(False)
        if mode in ("output", "both"):
            self.lm_head.weight.requires_grad_(False)

    # ------------------------------------------------------------- helpers
    def normalize_weights(self) -> None:
        """Unit-norm every embedding-dimension weight row (nGPT constraint)."""
        with torch.no_grad():
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
                if not weight.requires_grad:
                    # E1.5a bypass guard: a frozen table is already unit-norm
                    # from init to fp32 precision (recomputed row norms
                    # measure ~1.19e-07 off 1.0, 2026-08-18), but `copy_` is
                    # in-place and does not consult requires_grad — dividing
                    # by that not-bit-exactly-1.0 norm would silently change
                    # the "frozen" bits on every single step otherwise.
                    continue
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

    def field_encoder_parameter_count(self) -> int | None:
        """Parameter count of the field-estimator submodule (E1.5c), or
        ``None`` when ``state_source != "field"`` — absent channel is
        absent, the same convention E1.5b's ``gram_frob_sq``/``gram_penalty``
        already use for a feature most arms don't carry."""
        if self.field_encoder is None:
            return None
        return sum(p.numel() for p in self.field_encoder.parameters())

    def parameter_attestation(self) -> dict[str, int | str]:
        """Unblinded freeze attestation for the double-blind review packet.

        The packet scrubs every freeze-axis config field, so a reviewer cannot
        otherwise verify that one arm froze a subset, another a DIFFERENT
        subset, and the control nothing (review finding MAJOR-1,
        docs/reviews/e1-5a-review-cloud.md). These three fields distinguish
        the arms without naming the axis.

        ``frozen_mask_digest`` is a pure function of WHICH parameters are
        frozen — names and shapes, never values — so it is constant across
        seeds within an arm and differs between arms freezing different
        subsets. An unfrozen model hashes the EMPTY string rather than
        omitting the key: every run logs a digest, control arms included, so
        the key's presence never distinguishes an arm.

        The mask enumerates ``named_parameters(remove_duplicate=False)`` while
        the counts use the deduplicated ``parameters()``. MEASURED: a tied
        table is ONE ``nn.Parameter`` under two names, so under dedup
        ``embedding_freeze="both"`` (tied) and ``="input"`` (untied) both
        reduce to the mask {wte.weight} with equal frozen element counts —
        the digest would not separate the two E1.5a arms at all. Without
        dedup the tied arm's mask carries both role names, which is also the
        honest reading: freezing both roles of one table is not freezing one
        role of two. The counts stay deduplicated so
        ``trainable + frozen == num_parameters()``.
        """
        frozen_names = sorted(
            f"{name}:{tuple(p.shape)}"
            for name, p in self.named_parameters(remove_duplicate=False)
            if not p.requires_grad
        )
        digest = hashlib.sha256("\n".join(frozen_names).encode("utf-8")).hexdigest()
        return {
            "trainable_param_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_param_count": sum(p.numel() for p in self.parameters() if not p.requires_grad),
            "frozen_mask_digest": digest,
        }

    # ------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> LMOutput:
        """``idx``/``targets``: ``(B, T)`` int64 token ids."""
        _, t = idx.shape
        if t > self.cfg.block_size:
            msg = f"sequence length {t} exceeds block_size {self.cfg.block_size}"
            raise ValueError(msg)

        field_logits: torch.Tensor | None = None
        if self.field_encoder is not None:
            # E1.5c: z_t = Q . sqrt(p_phi(. | x_<=t)) replaces the static
            # table lookup — z_t ALONE (the registered v1 combination;
            # docs/e1_5-instruction.md's z_t (+) E[b_t] variant is a declared
            # follow-up, not built here). wte/lm_head remain the output
            # classifier untouched by this branch.
            #
            # Called through `__call__` (not `.forward(...)`/`._encode(...)`)
            # so the module's forward hooks still fire on the injection path
            # — tests/test_field_encoder.py's anti-leakage hook test reads
            # exactly that. The FP32 island around the softmax -> sqrt ->
            # lift step lives INSIDE `_encode` and is untouched by this
            # branch.
            encoded: FieldEncoding = self.field_encoder(idx)
            field_logits = encoded.logits
            h = self.drop(encoded.z)
        else:
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
            field_logits=field_logits,
        )

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling (used for qualitative checks)."""
        with torch.no_grad():
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
