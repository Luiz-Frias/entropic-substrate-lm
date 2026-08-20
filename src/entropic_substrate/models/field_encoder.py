r"""E1.5c field encoder: the online contextual Fisher state ``p_phi(.|x_<=t)``.

``docs/e1_5-instruction.md`` names the pivot this rung tests: E1 warm-starts a
*static* per-symbol embedding table from a corpus-wide Fisher-Rao snapshot;
E1.5c replaces that fixed point with a *live* one — at every position, a
causal density estimator produces the next-byte distribution conditioned on
everything seen so far, and that distribution (not a lookup) is the state
entering the backbone:

.. math::
    z_t = Q \cdot \sqrt{p_\phi(\cdot \mid x_{\le t})}

``Q`` is E1's own fixed semi-orthogonal lift (``init/fisher.py``,
``seed=0``) — the SAME map that lifted the static Fisher table, so ``z_t``
and E1's table origin share one geometry and remain angle-comparable.

:class:`FieldEncoder` is a small, self-contained causal transformer over the
256-symbol byte alphabet — its own embedding, its own stack of baseline
blocks (reusing :func:`entropic_substrate.models.blocks.build_block`'s
already-causal ``CausalSelfAttention``), its own tied 256-way head — entirely
separate from the backbone's ``wte``/``lm_head``, which stay in place as the
OUTPUT classifier only (still Fisher-seeded when ``embedding_init ==
"fisher_sqrt"``; see ``backbone.py``). Injection is ``z_t`` ALONE — the
registered v1 combination (``docs/e1_5-instruction.md``'s ``z_t \oplus
E[b_t]`` variant is a declared follow-up, not built here).

Byte-only: ``p_phi`` is defined over the 256-symbol byte alphabet (the same
domain E1's ``fisher_sqrt`` init uses); a BPE-tokenized run has no 256-way
target for it to predict, so ``tokenizer != "byte"`` is refused loudly at
construction rather than silently building a nonsensical head.

Non-negotiable numerics (CLAUDE.md): the ``softmax -> sqrt -> lift`` step
runs in FP32 inside an explicit ``torch.autocast(enabled=False)`` block
regardless of the caller's ambient autocast state, mirroring
``geometry/gram.py`` and ``geometry/kappa.py``.

Cross-axis compatibility (L2 finding, 2026-08-19): E1.5a's
``embedding_freeze`` and E1.5b's ``gram_lambda`` both predate this axis and
are defined over the backbone's ``self.wte.weight`` being the INPUT
representation. Under ``state_source="field"`` that stops being true —
``self.wte(idx)`` is never called on this path, ``FieldEncoder.wte`` is —
so ``backbone.py``'s ``LanguageModel.__init__`` refuses
``embedding_freeze in ("input", "both")`` and any nonzero ``gram_lambda``
together with ``state_source="field"`` rather than silently measuring a
tensor the forward pass never reads. ``embedding_freeze="output"`` stays
legal: ``lm_head.weight`` remains the real output classifier regardless of
``state_source``. No registered arm (``configs/e1-5*.yaml``) combines these
axes — each varies exactly one from baseline.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.checkpoint import normalize_checkpoint
from entropic_substrate.config import ModelConfig
from entropic_substrate.init.fisher import semi_orthogonal_lift
from entropic_substrate.models.blocks import build_block, build_norm

__all__ = ["FieldEncoder"]

# clamp_min floor for the post-softmax probabilities, before the sqrt: a
# softmax output is strictly positive by construction, so this floor only
# guards against a symbol whose logit underflows to a probability the fp32
# sqrt would otherwise round to exactly 0.0 (which the E1.5c doc's `z_t`
# formula spells out explicitly as `.clamp_min(1e-12)`).
_PROB_EPS: float = 1e-12


class FieldEncoder(nn.Module):
    """Causal ``p_phi(.|x_<=t)`` estimator: ``forward(idx) -> z_t``.

    ``idx`` is ``(B, T)`` byte ids; the return is ``(B, T, cfg.d_model)`` —
    the BACKBONE's embedding width, not the encoder's own (smaller)
    ``field_d_model``, since ``z_t`` is what gets injected in place of the
    backbone's embedding lookup (``backbone.py``, ``state_source ==
    "field"``).
    """

    # Registered via register_buffer below; annotated so static analysis
    # resolves attribute access as Tensor rather than nn.Module's generic
    # `Tensor | Module` __getattr__ overload.
    lift: torch.Tensor

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.tokenizer != "byte":
            msg = (
                "FieldEncoder (state_source='field') is defined over the "
                f"256-symbol byte alphabet; tokenizer={cfg.tokenizer!r} is not supported"
            )
            raise ValueError(msg)
        if cfg.field_estimator != "transformer_v1":
            msg = f"unknown field_estimator {cfg.field_estimator!r}"
            raise ValueError(msg)
        if cfg.field_mode not in ("joint", "frozen"):
            msg = f"unknown field_mode {cfg.field_mode!r}"
            raise ValueError(msg)
        if cfg.field_mode == "frozen" and cfg.field_ckpt_path is None:
            msg = (
                "field_mode='frozen' requires field_ckpt_path — a frozen field "
                "cannot start from a random draw, only from a checkpoint of a "
                "previously trained p_phi (docs/e1_5-instruction.md Repair 1's "
                "pretrained/frozen regime)"
            )
            raise ValueError(msg)
        if cfg.field_mode == "joint" and cfg.field_ckpt_path is not None:
            # Fail loud rather than silently ignore the path (CLAUDE.md Errors
            # principle): a set field_ckpt_path with field_mode="joint" is an
            # UNREGISTERED combination today (no arm loads a checkpoint and
            # then keeps training it jointly), not a permanently nonsensical
            # one — pretrain-then-finetune is a plausible later arm — so this
            # refuses the specific unregistered pairing rather than claiming
            # the combination can never mean anything.
            msg = (
                "field_ckpt_path is set but field_mode='joint' — loading a "
                "checkpoint into a field that also trains jointly would "
                "silently discard the checkpoint's frozen-p_phi purpose and "
                "no registered arm combines these; set field_mode='frozen' "
                "or clear field_ckpt_path"
            )
            raise ValueError(msg)

        self.field_d_model = cfg.field_d_model
        # Internal ModelConfig replica (docs/e1_5-instruction.md v1: "a
        # 2-layer, d=128 causal transformer p_phi"): the geometry-defining
        # fields are pinned to the baseline shape regardless of what the
        # OUTER experiment sets (an e3-style anchor_mode/attn_kernel="kappa"
        # config must not silently leak into this always-baseline estimator),
        # while generic knobs (dropout, bias, block_size) pass through from
        # the outer config since they are not part of this arm's declared
        # axis.
        inner_cfg = dataclasses.replace(
            cfg,
            d_model=cfg.field_d_model,
            n_layers=cfg.field_n_layers,
            n_heads=cfg.field_n_heads,
            mlp_ratio=4.0,
            block="baseline",
            norm="layernorm",
            attn_kernel="cosine",
            anchor_mode="none",
            tie_embeddings=True,
            embedding_init="default",
        )
        self.wte = nn.Embedding(256, cfg.field_d_model)
        self.blocks = nn.ModuleList([build_block(inner_cfg) for _ in range(cfg.field_n_layers)])
        self.norm_f = build_norm(inner_cfg)
        self.head = nn.Linear(cfg.field_d_model, 256, bias=False)
        self.head.weight = self.wte.weight  # tied, per the plan's "own tied 256-way head"

        # E1's fixed lift: (256, cfg.d_model) semi-orthogonal, seed=0 — a
        # buffer (never trained, always checkpointed), cast to fp32
        # explicitly since semi_orthogonal_lift returns fp64 (the precision
        # its own QR factorisation runs in).
        self.register_buffer(
            "lift", semi_orthogonal_lift(256, cfg.d_model, seed=0).float(), persistent=True
        )

        # Own init pass, AFTER wte/blocks/head/lift are all built (mirrors
        # LanguageModel's own `self.apply(self._init_weights)` timing) and
        # scaled to THIS module's own width (field_d_model), not the
        # backbone's — applying the backbone's std here would under-scale a
        # narrower estimator (e.g. d_model=768 vs field_d_model=128 is a
        # 2.4x smaller std than the estimator's own width calls for).
        self.apply(self._init_weights)

        # E1.5c Repair 1, frozen regime: overwrite the fresh init above with
        # a previously trained p_phi and freeze it — mirrors E1.5a's
        # model-build-time freeze in backbone.py (`_apply_embedding_freeze`),
        # timed the same way (AFTER init, so the frozen snapshot is the
        # checkpoint's true weights, never the local random draw). Unlike
        # that freeze, this module has no `normalize_weights`-style in-place
        # renormalisation pass of its own to bypass-guard against: nothing in
        # LanguageModel.normalize_weights walks `self.field_encoder` (it only
        # iterates the BACKBONE's own blocks/wte/lm_head), so `requires_grad_
        # (False)` alone is sufficient here — there is no copy_ trap to close.
        if cfg.field_mode == "frozen":
            ckpt_path = cfg.field_ckpt_path
            if ckpt_path is None:
                # Unreachable given the raise above — restated as a real
                # control-flow check, not `assert`, since CI's bandit gate
                # (B101) flags asserts as stripped-by-`python -O` and this
                # repo's own `_require_loss` docstring names exactly that
                # hazard: a narrowing check must fail loud even under -O,
                # not silently vanish into a `Path(None)` TypeError instead.
                msg = "field_mode='frozen' with field_ckpt_path=None reached model build"
                raise RuntimeError(msg)
            self._load_frozen_weights(ckpt_path)
            self.requires_grad_(False)

    def _load_frozen_weights(self, path: str) -> None:
        """Load this module's own weights from a checkpoint's ``model`` key.

        Wire format: ``{"format_version": ..., "model": <FieldEncoder.state_
        dict()>}`` — a checkpoint scoped to THIS module alone (not a full
        ``Trainer.state_dict()``), produced by whatever separate run trained
        p_phi as its own small causal LM (docs/e1_5-instruction.md's
        pretrained/frozen regime). Nothing in this repo produces that
        checkpoint yet (the standalone-p_phi training pipeline is a later
        rung); this defines the format the loader expects. Routed through
        ``normalize_checkpoint`` for the same reason ``Trainer.load_checkpoint``
        is: a stray legacy/malformed payload fails loud here rather than
        surfacing as an opaque shape error three lines later. ``for_resume=
        False`` — this is a fresh module taking a frozen INITIAL value, not
        this Trainer resuming its own prior optimiser/step state.
        """
        # weights_only=True matches Trainer.load_checkpoint: self-produced
        # checkpoints hold only tensors and primitives (bandit B614).
        raw = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = normalize_checkpoint(raw, source=str(path), for_resume=False)
        self.load_state_dict(state["model"])

    def _init_weights(self, module: nn.Module) -> None:
        std = self.field_d_model**-0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _encode(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared body for :meth:`forward` / :meth:`field_logits`.

        Returns ``(logits, z)``: ``logits`` is the ambient-dtype ``(B, T,
        256)`` pre-softmax output of ``self.head``; ``z`` is the ``(B, T,
        cfg.d_model)`` unit-norm Fisher state, ambient dtype. Both callers
        below invoke this INDEPENDENTLY — the E1.5c Repair 1 calibration seam
        (``train.py``'s ``Trainer._field_loss_terms``) calls
        ``field_logits(idx)`` as a second, separate forward pass through this
        (small) encoder rather than one shared with whatever call
        ``LanguageModel.forward`` already made through ``forward(idx)`` for
        injection. That duplicates this module's own compute when both
        injection AND calibration are active in the same train step, but
        NOT the backbone's — ``field_d_model``/``field_n_layers`` (v1: 128,
        2) keep this submodule negligible next to the ~85M backbone it feeds
        (``run_start``'s ``field_encoder_params``), and threading a shared
        result through ``LanguageModel.forward``/``LMOutput`` instead would
        touch ``backbone.py``, out of this phase's scope (FOR-923).
        """
        h = self.wte(idx)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        logits = self.head(h)
        # FP32 island: softmax -> clamp -> sqrt -> lift, regardless of the
        # caller's ambient autocast (CLAUDE.md non-negotiable numerics).
        with torch.autocast(device_type=idx.device.type, enabled=False):
            probs = F.softmax(logits.float(), dim=-1).clamp_min(_PROB_EPS)
            z = probs.sqrt() @ self.lift
        return logits, z.to(h.dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` byte ids -> ``(B, T, cfg.d_model)`` unit-norm ``z_t`` rows.

        Only ``idx`` is consumed — no ``targets`` parameter exists on this
        signature, so the estimator structurally cannot see the label it
        will be scored against (docs/e1_5-instruction.md's anti-leakage pin).
        """
        _logits, z = self._encode(idx)
        return z

    def field_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` byte ids -> ``(B, T, 256)`` FP32 next-byte logits.

        The E1.5c Repair 1 calibration target: ``train.py``'s seam scores
        these against the SAME targets the backbone's own CE uses (``L_field
        = CE(field_logits, y)``) — never fed back in, only scored, matching
        this method's signature: no ``targets`` parameter, same anti-leakage
        pin ``forward`` carries. ``.float()`` here only guarantees the
        RETURNED dtype; it does not by itself guarantee FP32 *compute* for
        ``self.head``'s matmul under an ambient bf16 autocast (PyTorch's
        autocast policy may still run that matmul in bf16 before this casts
        the result). The calibration seam that calls this (``Trainer.
        _field_loss_terms``) supplies the actual FP32-compute guarantee by
        wrapping the call in its own explicit ``autocast(enabled=False)``
        island, mirroring ``geometry/gram.py`` (CLAUDE.md non-negotiable
        numerics) — this method's own `.float()` is a type contract for
        callers that do NOT provide that island themselves, not a substitute
        for one.
        """
        logits, _z = self._encode(idx)
        return logits.float()
