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

Four-arm table, P7 (``docs/e1_5-instruction.md`` Repair 2, FOR-924):
``cfg.field_state`` selects among three CONSTRUCTIONS of ``z_t`` from the
SAME ``p_phi`` — F (``"sqrt_prob"``, the formula above, unit-norm rows), P
(``"prob"``, ``p_phi @ lift`` with no sqrt — the control for the sqrt map
itself, rows NOT unit-norm in general), and H (``"hidden"``, ``W @ h_phi,t``
through a TRAINABLE ``state_lift`` — contextual preprocessing without
probability geometry, the only leg with a trainable lift). T, the E1
reference, is ``state_source="table"`` and carries no ``field_state`` at
all. See ``_encode`` for the per-leg construction and its FP32-island
scoping (F/P share it; H does not — it is a plain learned layer, not
geometric math).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.checkpoint import normalize_checkpoint
from entropic_substrate.config import ModelConfig
from entropic_substrate.init.fisher import semi_orthogonal_lift
from entropic_substrate.models.blocks import build_block, build_norm

__all__ = ["FieldCalibrationAccumulator", "FieldEncoder", "FieldEncoding"]

# clamp_min floor for the post-softmax probabilities, before the sqrt: a
# softmax output is strictly positive by construction, so this floor only
# guards against a symbol whose logit underflows to a probability the fp32
# sqrt would otherwise round to exactly 0.0 (which the E1.5c doc's `z_t`
# formula spells out explicitly as `.clamp_min(1e-12)`).
_PROB_EPS: float = 1e-12

# P8 (FOR-925): expected-calibration-error bin count. A STANDARD choice (Guo
# et al. 2017, "On Calibration of Modern Neural Networks") — named so a
# future change is a one-line, reviewable diff rather than a buried literal.
_ECE_BINS: int = 15


class FieldEncoding(NamedTuple):
    """One field-encoder pass: BOTH of the things ``_encode`` produces.

    ``FieldEncoder.forward`` returns this rather than ``z`` alone so a
    caller that needs the next-byte logits too (``LanguageModel.forward``,
    which forwards them on ``LMOutput.field_logits`` for
    ``Trainer.evaluate``'s calibration diagnostics) gets them from the pass
    it already paid for, instead of running the encoder a second time
    through :meth:`FieldEncoder.field_logits`. Returning a NamedTuple
    rather than a bare 2-tuple keeps the two channels named at every call
    site — and keeps ``forward`` the module's single entry point, so the
    injection path still goes through ``nn.Module.__call__`` (hooks fire;
    ``tests/test_field_encoder.py``'s anti-leakage hook test depends on
    exactly that).

    ``logits``: ``(B, T, 256)`` pre-softmax, AMBIENT dtype — deliberately
    not ``.float()``-cast here the way :meth:`FieldEncoder.field_logits`
    casts its own return. A train-time forward never reads this channel, so
    casting at construction would spend a ``(B, T, 256)`` fp32 copy per
    training step for nothing; the fp32 cast belongs to the consumer that
    actually reads it (``Trainer.evaluate``), and
    ``FieldCalibrationAccumulator.update`` refuses non-fp32 input loudly if
    a future consumer forgets.

    ``z``: ``(B, T, cfg.d_model)`` state handed to the backbone, ambient
    dtype — see :meth:`FieldEncoder._encode` for its per-``field_state``
    construction.
    """

    logits: torch.Tensor
    z: torch.Tensor


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
    # P7 (FOR-924): registered as a real nn.Linear only when
    # cfg.field_state == "hidden" — the ONE arm with a trainable lift; None
    # on the F/P legs, which share the fixed `lift` buffer above instead.
    state_lift: nn.Linear | None

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
        if cfg.field_state not in ("sqrt_prob", "prob", "hidden"):
            msg = f"unknown field_state {cfg.field_state!r}"
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
        self.field_state = cfg.field_state
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
        # its own QR factorisation runs in). Shared by the F and P legs
        # (field_state in {"sqrt_prob", "prob"}) — always registered
        # regardless of field_state, so a run that starts "hidden" and is
        # later reconfigured to "sqrt_prob"/"prob" still finds a valid lift
        # in its own checkpoint rather than a missing buffer.
        self.register_buffer(
            "lift", semi_orthogonal_lift(256, cfg.d_model, seed=0).float(), persistent=True
        )
        # P7 (FOR-924): H's own trainable lift — Linear(field_d_model,
        # d_model, bias=False). The ONLY field_state with a trainable lift;
        # F/P read the fixed `lift` buffer above instead. Built here (before
        # `self.apply(self._init_weights)` below) so it receives the SAME
        # field_d_model-scaled init pass every other Linear in this module
        # gets, not a separate uninitialised draw.
        self.state_lift = (
            nn.Linear(cfg.field_d_model, cfg.d_model, bias=False)
            if self.field_state == "hidden"
            else None
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
            self._freeze_field_body()

    # The learned content of p_phi: everything sized by `field_d_model`, and
    # nothing sized by the BACKBONE's `d_model`. See _load_frozen_weights.
    _BODY_PREFIXES = ("wte.", "blocks.", "norm_f.", "head.")

    def _freeze_field_body(self) -> None:
        """Freeze p_phi's learned body — and ONLY that.

        Not ``self.requires_grad_(False)``: ``state_lift`` is H's own
        trainable adapter (``docs/e1_5-instruction.md`` Repair 2 defines H as
        "the only arm with a trainable lift"), created fresh by this arm and
        never part of the pretrained field. Freezing it too would leave a
        frozen random projection over a frozen hidden state — a different
        experiment than the registered H arm. ``lift`` is a buffer, so it
        carries no ``requires_grad`` either way.
        """
        for name, param in self.named_parameters():
            if name.startswith(self._BODY_PREFIXES):
                param.requires_grad_(False)

    def _load_frozen_weights(self, path: str) -> None:
        """Restore p_phi's LEARNED BODY from a checkpoint's ``model`` key.

        Body = everything sized by ``field_d_model`` (``wte``, ``blocks``,
        ``norm_f``, the tied ``head``). Deliberately NOT the whole module:

        * ``lift`` is the fixed ``semi_orthogonal_lift(256, d_model, seed=0)``
          buffer, sized by the BACKBONE's width — the matched-total lane runs
          d_model=756 against a checkpoint written at 768. It is
          deterministic, so this arm's own construction is the source of
          truth and the checkpoint's copy is ignored.
        * ``state_lift`` is H's own trainable adapter, which an F-shaped
          checkpoint never contained.

        MEASURED 2026-08-21 against the real pretrained artifact: a
        whole-module ``load_state_dict`` raised ``size mismatch for lift`` on
        both matched-total twins and ``Missing key(s): "state_lift.weight"``
        on H — i.e. it could not deliver the registered contract that F, H
        and the matched-total twins all share ONE p_phi
        (``docs/e1-5c-preregistration.md``). Because the body depends only on
        ``field_d_model``, that shared field transfers to every arm at every
        backbone width. Guard teeth in
        ``tests/test_frozen_field_sharing.py``.

        Wire format: ``{"format_version": ..., "model": <FieldEncoder.state_
        dict()>}`` — a checkpoint scoped to THIS module alone (not a full
        ``Trainer.state_dict()``), produced by a separate run that trained
        p_phi as its own small causal LM (docs/e1_5-instruction.md's
        pretrained/frozen regime). RETRACTION (2026-08-20): this docstring
        said "Nothing in this repo produces that checkpoint yet (the
        standalone-p_phi training pipeline is a later rung)" — true when
        written, false as of ``entropic_substrate.field_pretrain``, which is
        that producer and writes exactly this shape (its own roundtrip test
        loads it back through THIS method). The format is still DEFINED
        here, by the loader. Routed through
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
        model_state = state["model"]
        if not isinstance(model_state, dict):
            msg = f"{path}: checkpoint 'model' is not a state dict"
            raise RuntimeError(msg)
        body = {k: v for k, v in model_state.items() if k.startswith(self._BODY_PREFIXES)}
        if not body:
            msg = (
                f"{path}: checkpoint carries no p_phi body "
                f"({'/'.join(self._BODY_PREFIXES)}) — a frozen field must load a "
                "PRETRAINED estimator, and silently keeping this module's fresh "
                "random init would report an untrained field as a frozen one"
            )
            raise RuntimeError(msg)
        # strict=False tolerates exactly the two arm-local tensors above being
        # absent from the payload; it does NOT weaken shape checking (a size
        # mismatch still raises), and a missing BODY key is re-raised below —
        # so a genuinely wrong field still fails loud.
        incompatible = self.load_state_dict(body, strict=False)
        body_missing = [k for k in incompatible.missing_keys if k.startswith(self._BODY_PREFIXES)]
        if body_missing:
            msg = f"{path}: checkpoint is missing p_phi body keys {sorted(body_missing)}"
            raise RuntimeError(msg)

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
        cfg.d_model)`` state handed to the backbone, ambient dtype — its
        CONSTRUCTION depends on ``self.field_state`` (P7, FOR-924, the
        four-arm table's F/P/H legs):

        - ``"sqrt_prob"`` (F, the full thesis) — ``sqrt(p_phi) @ lift``,
          unit-norm rows (``lift`` semi-orthogonal).
        - ``"prob"`` (P, the sqrt-map control) — ``p_phi @ lift``, no sqrt;
          rows are NOT unit-norm in general (Cauchy-Schwarz:
          ``sum(p_i^2) <= 1`` with equality only at a one-hot distribution).
        - ``"hidden"`` (H, contextual preprocessing without probability
          geometry) — ``state_lift(h_phi)``, the encoder's own pre-head
          hidden state through a TRAINABLE lift; unconstrained, no relation
          to ``p_phi`` or the fixed ``lift`` buffer at all.

        The two callers below no longer both pay for their own pass.
        RETRACTION (2026-08-20, CR PR #79): this docstring previously said
        both callers "invoke this INDEPENDENTLY", accepting a duplicated
        encoder forward on BOTH the train and the eval path because
        threading a shared result through ``LanguageModel.forward``/
        ``LMOutput`` was out of that phase's scope (FOR-923). That is now
        done for the EVAL path — :meth:`forward` returns
        :class:`FieldEncoding` (logits AND z), ``LanguageModel.forward``
        carries the logits out on ``LMOutput.field_logits``, and
        ``Trainer.evaluate`` reads them there instead of calling
        :meth:`field_logits` a second time.

        The TRAIN path still runs a genuinely separate pass on purpose:
        ``Trainer._field_loss_terms`` needs the calibration logits computed
        INSIDE its own ``autocast(enabled=False)`` island (they are on the
        gradient path — CLAUDE.md non-negotiable numerics), which the
        injection forward, running in the ambient training dtype, cannot
        supply. Its cost stays bounded for the reason the original note
        gave: ``field_d_model``/``field_n_layers`` (v1: 128, 2) keep this
        submodule negligible next to the ~85M backbone it feeds
        (``run_start``'s ``field_encoder_params``).
        """
        h = self.wte(idx)
        for block in self.blocks:
            h = block(h)
        h = self.norm_f(h)
        logits = self.head(h)
        if self.field_state == "hidden":
            # H: a plain trainable Linear over the encoder's own hidden
            # state, run in the caller's ambient dtype like every other
            # learned layer in this module — no probability geometry, no
            # unit-norm claim, so no FP32 island (CLAUDE.md's non-negotiable
            # numerics govern GEOMETRIC math; this is neither).
            if self.state_lift is None:
                # Unreachable given __init__'s field_state -> state_lift
                # wiring — restated as a real control-flow check rather than
                # `assert` (bandit B101: stripped under `python -O`, per
                # this repo's own `_require_loss` docstring in train.py).
                msg = "field_state='hidden' reached _encode with state_lift=None"
                raise RuntimeError(msg)
            z = self.state_lift(h)
        else:
            # F/P share this island: softmax -> clamp -> (sqrt) -> lift, in
            # FP32 regardless of the caller's ambient autocast (CLAUDE.md
            # non-negotiable numerics) — the map from p_phi to a point
            # against the fixed semi-orthogonal `lift` is exactly the
            # "geometric math" that rule covers, for both legs.
            with torch.autocast(device_type=idx.device.type, enabled=False):
                probs = F.softmax(logits.float(), dim=-1).clamp_min(_PROB_EPS)
                state = probs.sqrt() if self.field_state == "sqrt_prob" else probs
                z = (state @ self.lift).to(h.dtype)
        return logits, z

    def forward(self, idx: torch.Tensor) -> FieldEncoding:
        """``(B, T)`` byte ids -> :class:`FieldEncoding` ``(logits, z)``.

        ``z`` is the ``(B, T, cfg.d_model)`` state; its rows are unit-norm
        ONLY for ``field_state="sqrt_prob"`` (F) — the "prob" (P) and
        "hidden" (H) arms are NOT unit-norm in general, see :meth:`_encode`
        for the per-arm construction. ``logits`` is the ``(B, T, 256)``
        pre-softmax output of the SAME pass, returned rather than discarded
        so the eval-time calibration consumer does not have to run this
        encoder again (see :class:`FieldEncoding`).

        Only ``idx`` is consumed — no ``targets`` parameter exists on this
        signature, so the estimator structurally cannot see the label it
        will be scored against (docs/e1_5-instruction.md's anti-leakage pin;
        adding the logits to the RETURN cannot weaken that, since a return
        value carries no information the pass did not already have).
        """
        logits, z = self._encode(idx)
        return FieldEncoding(logits=logits, z=z)

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


class FieldCalibrationAccumulator:
    """Streaming next-byte calibration diagnostics for ``p_phi`` over an eval pass.

    Predictive calibration (docs/e1_5-instruction.md "Calibration, geometry,
    and context-sensitivity diagnostics", FOR-925): field nats/byte, Brier
    score, expected calibration error (ECE), mean predictive entropy, and
    top-1 next-byte accuracy — "logged for p_phi THROUGHOUT", i.e. whenever a
    field encoder exists, independent of whether the calibration training
    seam (``field_loss_lambda``, ``train.py``'s ``_field_loss_terms``) is
    active. ``Trainer.evaluate`` calls :meth:`update` once per eval batch and
    :meth:`finalize` once at the end.

    Accumulates SCALAR reductions per :meth:`update` call — never the raw
    ``(N, num_classes)`` probability tensor across calls — so a full eval
    pass (many batches) costs O(batches) transient memory, not
    O(eval_set_size x num_classes): at production eval sizes (e.g. 20 batches
    x 16 x 1024 x 256 fp32) concatenating logits across the whole pass would
    be ~336MB kept live for the duration of the eval; this keeps only O(1)
    running totals plus one O(n_bins) histogram between calls.

    Trusts its input is already FP32 (mirrors :meth:`FieldEncoder.field_logits`'s
    own documented contract: its own ``.float()`` is a type contract, not a
    compute guarantee) — a caller must produce ``logits`` inside its own
    ``torch.autocast(enabled=False)`` island, exactly as ``Trainer.
    _field_loss_terms`` already does for the calibration-loss seam.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._ce_sum: float = 0.0
        self._brier_sum: float = 0.0
        self._entropy_sum: float = 0.0
        self._correct_sum: float = 0.0
        # float64 running histograms: O(n_bins) state, independent of how
        # many `update` calls (eval batches) contribute to it.
        self._bin_count = torch.zeros(_ECE_BINS, dtype=torch.float64)
        self._bin_correct = torch.zeros(_ECE_BINS, dtype=torch.float64)
        self._bin_conf = torch.zeros(_ECE_BINS, dtype=torch.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Fold one batch's worth of tokens into the running totals.

        ``logits``: ``(N, num_classes)`` FP32 pre-softmax; ``targets``:
        ``(N,)`` int64 class ids. A zero-sized batch (``N == 0``) is a
        deliberate no-op, not an error — distinct from calling
        :meth:`finalize` having NEVER updated at all (see its docstring).
        """
        if logits.dtype != torch.float32:
            msg = f"FieldCalibrationAccumulator.update expects fp32 logits, got {logits.dtype}"
            raise ValueError(msg)
        n = logits.shape[0]
        if n == 0:
            return
        probs = F.softmax(logits, dim=-1)
        ce_per_token = F.cross_entropy(logits, targets, reduction="none")
        onehot = F.one_hot(targets, num_classes=logits.shape[-1]).to(probs.dtype)
        brier_per_token = (probs - onehot).pow(2).sum(dim=-1)
        clamped = probs.clamp_min(_PROB_EPS)
        entropy_per_token = -(clamped * clamped.log()).sum(dim=-1)
        confidence, preds = probs.max(dim=-1)
        correct = (preds == targets).to(probs.dtype)

        self._count += n
        self._ce_sum += float(ce_per_token.sum())
        self._brier_sum += float(brier_per_token.sum())
        self._entropy_sum += float(entropy_per_token.sum())
        self._correct_sum += float(correct.sum())

        bin_idx = (confidence * _ECE_BINS).floor().clamp(0, _ECE_BINS - 1).long().cpu()
        ones = torch.ones_like(bin_idx, dtype=torch.float64)
        self._bin_count.index_add_(0, bin_idx, ones)
        self._bin_correct.index_add_(0, bin_idx, correct.double().cpu())
        self._bin_conf.index_add_(0, bin_idx, confidence.double().cpu())

    def finalize(self) -> dict[str, float]:
        """Mean CE / Brier / entropy, top-1 accuracy, and ECE over every
        token seen across all :meth:`update` calls so far.

        Fails loud on an accumulator that never received a nonempty batch —
        the same fail-loud posture ``Trainer.evaluate``'s own ``np.mean``
        over an eval-batch loss list would need if that list ever came back
        empty (division by a zero count must never silently read as NaN).
        """
        if self._count == 0:
            msg = "FieldCalibrationAccumulator.finalize() called with no updates"
            raise RuntimeError(msg)
        ece = 0.0
        for b in range(_ECE_BINS):
            count_b = float(self._bin_count[b])
            if count_b == 0.0:
                continue
            acc_b = float(self._bin_correct[b]) / count_b
            conf_b = float(self._bin_conf[b]) / count_b
            ece += (count_b / self._count) * abs(acc_b - conf_b)
        return {
            "ce": self._ce_sum / self._count,
            "brier": self._brier_sum / self._count,
            "entropy": self._entropy_sum / self._count,
            "top1": self._correct_sum / self._count,
            "ece": ece,
        }
