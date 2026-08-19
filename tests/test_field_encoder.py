"""E1.5c: the online contextual Fisher state ``z_t = Q . sqrt(p_phi(.|x_<=t))``.

``models/field_encoder.py`` provides :class:`FieldEncoder`, the causal
density estimator ``p_phi``; ``backbone.py`` injects its output ``z_t`` in
place of the static table lookup at ``LanguageModel.forward`` when
``cfg.state_source == "field"`` — z_t ALONE (docs/e1_5-instruction.md's
``z_t (+) E[b_t]`` combination is a declared follow-up, not this rung).

Three independent hazards this file pins with RED-first proofs (mirrors
``test_freeze.py``'s and ``test_gram_penalty.py``'s convention; all three
were verified RED by hand during implementation — command + observed output
recorded in the phase report, not shipped here):

1. **causality leak** — if the field estimator's attention were not
   strictly causal (e.g. ``is_causal=False`` on the underlying
   ``scaled_dot_product_attention`` call), a perturbation at the LAST
   position would move every earlier position's ``z_t`` too.
   ``test_field_encoder_is_strictly_causal`` is RED against exactly that:
   monkeypatching ``CausalSelfAttention.forward`` to call SDPA with
   ``is_causal=False`` turns this test red immediately, since the
   perturbed run's earlier positions then differ from the base run's.
2. **trivial-pass hazard** — a causality test alone cannot distinguish "the
   estimator is causal" from "the estimator ignores its input entirely" (a
   constant function is trivially causal). ``test_field_encoder_next_byte_alignment``
   adds the positive control: perturbing position ``t`` itself MUST move
   ``z_t``.
3. **injection-seam absent** — a `FieldEncoder` that is built but never
   wired into ``LanguageModel.forward`` (the backbone silently keeps using
   ``self.wte(idx)``) is invisible to every OTHER test here: causality and
   unit-norm call the encoder directly, the run_start param count reads off
   module existence rather than use, and even the tiny-train smoke still
   converges via the untouched ``wte`` path.
   ``test_field_encoder_gradients_reach_its_own_parameters_through_backbone_ce``
   is the one guard-tooth that closes this gap — with the injection branch
   in ``backbone.py`` deliberately reverted to the unconditional
   ``h = self.drop(self.wte(idx))`` path, this test goes RED (every field
   encoder parameter's ``.grad`` comes back ``None`` after backward, since
   no seam calls it during the forward pass at all).
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from entropic_substrate.config import Config
from entropic_substrate.init.fisher import build_fisher_init, semi_orthogonal_lift
from entropic_substrate.models.backbone import build_model
from entropic_substrate.models.field_encoder import FieldEncoder
from entropic_substrate.train import Trainer
from helpers import tiny_config, tiny_model_config

pytestmark = pytest.mark.unit


def _field_model_config(**overrides: object) -> object:
    defaults: dict[str, object] = {
        "state_source": "field",
        "field_d_model": 16,
        "field_n_layers": 2,
        "field_n_heads": 2,
    }
    defaults.update(overrides)
    return tiny_model_config(**defaults)


def _field_config(tmp_path: Path, **overrides: object) -> Config:
    defaults: dict[str, object] = {
        "state_source": "field",
        "field_d_model": 16,
        "field_n_layers": 2,
        "field_n_heads": 2,
    }
    defaults.update(overrides)
    return tiny_config(tmp_path, **defaults)


# --------------------------------------------------------- construction / guards
def test_field_encoder_rejects_bpe_tokenizer() -> None:
    cfg = _field_model_config(tokenizer="gpt2", vocab_size=300)
    with pytest.raises(ValueError, match="byte alphabet"):
        FieldEncoder(cfg)


def test_field_encoder_rejects_unknown_field_estimator() -> None:
    cfg = _field_model_config(field_estimator="quantum")
    with pytest.raises(ValueError, match="unknown field_estimator"):
        FieldEncoder(cfg)


def test_build_model_rejects_field_state_source_with_bpe_tokenizer() -> None:
    """The byte-only refusal must be reachable from `build_model` (the real
    entry point trainers use), not only from constructing FieldEncoder
    directly."""
    cfg = tiny_model_config(state_source="field", tokenizer="gpt2", vocab_size=300)
    with pytest.raises(ValueError, match="byte alphabet"):
        build_model(cfg)


def test_build_model_rejects_unknown_state_source() -> None:
    cfg = tiny_model_config(state_source="quantum")
    with pytest.raises(ValueError, match="unknown state_source"):
        build_model(cfg)


def test_build_model_with_table_state_source_has_no_field_encoder() -> None:
    model = build_model(tiny_model_config(state_source="table"))
    assert model.field_encoder is None
    assert model.field_encoder_parameter_count() is None


# ---------------------------------------------- cross-axis guards (L2 finding)
# state_source="field" swaps which tensor generates the model's INPUT
# (field_encoder.wte replaces self.wte(idx) in forward — see
# LanguageModel.forward). E1.5a's embedding_freeze and E1.5b's gram_lambda
# both predate this axis and are defined in terms of self.wte.weight being
# that input generator. Verified 2026-08-19 (L2 review, this branch): with
# gram_lambda=0.5 + state_source="field", gram_penalty co-optimizes
# self.wte.weight across steps (0 -> 5.72039 -> 1.58559 -> 1.14893 ->
# 1.09639) while field_encoder.wte -- the tensor forward() actually reads --
# is never referenced by the penalty; symmetrically,
# embedding_freeze="input"+state_source="field" freezes self.wte.weight
# (requires_grad=False, bit-identical after 3 steps) while
# field_encoder.wte trains fully unfrozen (requires_grad=True), making the
# freeze a no-op on the axis its name promises to control. Neither
# registered arm combines these axes (each e1-5*.yaml varies exactly one),
# so the fix is to refuse the combination loudly rather than invent
# untested redirect semantics onto field_encoder.wte.
def test_build_model_rejects_field_state_source_with_gram_lambda() -> None:
    cfg = tiny_model_config(state_source="field", gram_lambda=0.5)
    with pytest.raises(ValueError, match="gram_lambda"):
        build_model(cfg)


@pytest.mark.parametrize("mode", ["input", "both"])
def test_build_model_rejects_field_state_source_with_input_freeze(mode: str) -> None:
    cfg = tiny_model_config(state_source="field", embedding_freeze=mode, tie_embeddings=False)
    with pytest.raises(ValueError, match="embedding_freeze"):
        build_model(cfg)


def test_build_model_permits_field_state_source_with_output_freeze() -> None:
    """embedding_freeze='output' still freezes the REAL output classifier
    (lm_head.weight) under state_source='field' -- wte/lm_head stay in
    place as the output classifier regardless of state_source (backbone.py)
    -- so this combination is legitimate and must NOT be refused."""
    model = build_model(
        tiny_model_config(state_source="field", embedding_freeze="output", tie_embeddings=False)
    )
    assert not model.lm_head.weight.requires_grad
    assert model.field_encoder is not None


def test_build_model_permits_field_state_source_with_axis_defaults() -> None:
    """The guard must not over-fire on state_source='field' alone (the
    registered e1-5c-field arm's own defaults: gram_lambda=0.0,
    embedding_freeze='none')."""
    model = build_model(tiny_model_config(state_source="field"))
    assert model.field_encoder is not None


# ------------------------------------------------------------------- geometry
def test_field_encoder_output_matches_backbone_width() -> None:
    cfg = _field_model_config(d_model=32)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    z = encoder(idx)
    assert z.shape == (2, 10, 32)


def test_field_encoder_z_rows_are_unit_norm() -> None:
    """``semi_orthogonal_lift``'s own contract (init/fisher.py) is an EXACT
    isometry only when ``d_out >= d_in`` (here: ``d_model >= 256``) — below
    that (the tiny smoke configs' ``d_model=32``) it is a JL-style
    projection and rows are not unit-norm. This test exercises the real
    regime (``d_model=300``) the isometry claim actually covers; field_*
    stays tiny since only the lift's output width matters here."""
    cfg = _field_model_config(d_model=300)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    z = encoder(idx)
    norms = z.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


def test_field_encoder_lift_buffer_matches_e1s_fixed_lift_and_has_no_grad() -> None:
    cfg = _field_model_config(d_model=32)
    encoder = FieldEncoder(cfg)
    expected = semi_orthogonal_lift(256, cfg.d_model, seed=0).float()
    torch.testing.assert_close(encoder.lift, expected, atol=0, rtol=0)
    assert not encoder.lift.requires_grad


# ----------------------------------------------------------------- causality
def test_field_encoder_is_strictly_causal() -> None:
    """Perturbing the LAST position's input byte must not move any earlier
    position's z_t. RED-first proof recorded in the phase report:
    monkeypatching the underlying attention to `is_causal=False` fails this
    test immediately."""
    cfg = _field_model_config()
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (2, 10))
    base = encoder(idx)

    bumped = idx.clone()
    bumped[:, -1] = (bumped[:, -1] + 137) % 256
    perturbed = encoder(bumped)
    torch.testing.assert_close(perturbed[:, :-1], base[:, :-1], atol=1e-6, rtol=0)


def test_field_encoder_next_byte_alignment() -> None:
    """x_t affects z_t (positive control — rules out a trivially-causal
    constant encoder); x_{t+1} (a future byte) does not affect z_t (the
    causality contract restated at a single position, independent of the
    whole-sequence proof above)."""
    cfg = _field_model_config(field_n_layers=1)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (1, 6))
    base = encoder(idx)

    changed_t = idx.clone()
    changed_t[:, 3] = (changed_t[:, 3] + 91) % 256
    z_changed_t = encoder(changed_t)
    # Finite BEFORE nonzero: `nan != nan`/`allclose(nan, nan)==False` means a
    # NaN-poisoned z_changed_t would otherwise slip past `not allclose(...)`
    # and read as "position t changed" instead of "the encoder broke"
    # (mirrors test_gram_penalty.py's identical guard-ordering rationale).
    assert torch.isfinite(z_changed_t[:, 3]).all()
    assert not torch.allclose(z_changed_t[:, 3], base[:, 3])

    changed_future = idx.clone()
    changed_future[:, 4] = (changed_future[:, 4] + 91) % 256
    z_changed_future = encoder(changed_future)
    torch.testing.assert_close(z_changed_future[:, 3], base[:, 3], atol=1e-6, rtol=0)


def test_field_encoder_forward_has_no_targets_parameter() -> None:
    """Structural pin (docs/e1_5-instruction.md's anti-leakage requirement):
    p_phi consumes x_<=t only. Pinning the signature itself catches a future
    refactor that threads `targets` through, at review/import time rather
    than only behaviorally."""
    sig = inspect.signature(FieldEncoder.forward)
    assert list(sig.parameters) == ["self", "idx"]


def test_backbone_logits_are_unaffected_by_the_targets_argument(tmp_path: Path) -> None:
    """Integration-level version of the structural pin: the field encoder
    sits upstream of the loss, so swapping `targets` for an unrelated
    tensor must not move a single logit — if the encoder (or anything else
    in the forward path) ever read `targets`, this would catch it even
    without inspecting the signature."""
    trainer = Trainer(_field_config(tmp_path))
    x, y = trainer.train_sampler.batch()
    trainer.model.eval()
    with torch.no_grad():
        logits_with_targets = trainer.model(x, y).logits
        logits_without_targets = trainer.model(x, None).logits
        garbage_targets = (y + 73) % 256
        logits_with_garbage = trainer.model(x, garbage_targets).logits
    torch.testing.assert_close(logits_with_targets, logits_without_targets, atol=0, rtol=0)
    torch.testing.assert_close(logits_with_targets, logits_with_garbage, atol=0, rtol=0)


# -------------------------------------------------------------- differentiability
def test_field_encoder_gradients_reach_its_own_parameters_through_backbone_ce(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_field_config(tmp_path))
    encoder = trainer.raw_model.field_encoder
    assert encoder is not None
    x, y = trainer.train_sampler.batch()
    out = trainer.model(x, y)
    assert out.loss is not None
    out.loss.backward()

    trainable = [p for p in encoder.parameters() if p.requires_grad]
    assert trainable, "field encoder has no trainable parameters"
    grads = [p.grad for p in trainable]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.any(g != 0) for g in grads if g is not None)
    # lift is a buffer (requires_grad=False), never touched by backward.
    assert encoder.lift.grad is None


def test_field_encoder_finite_under_bf16_autocast() -> None:
    cfg = _field_model_config(field_n_layers=1)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = encoder(idx)
    assert torch.isfinite(z.float()).all()


def test_field_encoder_fp32_island_holds_under_bf16_autocast() -> None:
    """``isfinite`` alone does not pin the FP32 island (CLAUDE.md
    non-negotiable numerics): under CPU bf16 autocast, PyTorch's own
    autocast policy already keeps ``softmax`` in fp32 and a plain bf16
    ``sqrt``/matmul does not overflow to inf/nan, so
    ``test_field_encoder_finite_under_bf16_autocast`` stays GREEN even with
    the explicit ``torch.autocast(..., enabled=False)`` island
    (field_encoder.py, forward) deleted (verified 2026-08-19 by hand:
    removing the island left that test passing).

    This test instead re-runs ``test_field_encoder_z_rows_are_unit_norm``'s
    own regime (``d_model=300``, where ``semi_orthogonal_lift`` is an EXACT
    isometry and rows are unit-norm to fp32 precision, atol=1e-4) UNDER an
    ambient bf16 autocast context. With the island intact, the sqrt/matmul
    step still runs in fp32 regardless of the ambient context, so the norm
    deviation stays ~1e-7 (well inside atol=1e-4). With the island removed,
    the ambient autocast downcasts the matmul against ``self.lift`` to
    bf16, and the measured deviation jumps to ~6e-4 -- five orders of
    magnitude past atol=1e-4 -- so a silently-dropped island goes RED here."""
    cfg = _field_model_config(d_model=300)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = encoder(idx)
    norms = z.float().norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


# ------------------------------------------------------------------- injection
def test_fisher_still_seeds_the_output_head_when_state_source_is_field(
    sample_bytes: bytes,
) -> None:
    """state_source stays a ONE-variable axis against e1-fisher: the
    backbone's own wte/lm_head (the OUTPUT classifier here) is still
    embedding_init='fisher_sqrt'-seeded exactly as it would be under
    state_source='table' — the field encoder swaps only the INPUT path."""
    cfg = tiny_model_config(
        state_source="field",
        field_d_model=16,
        field_n_layers=1,
        field_n_heads=2,
        embedding_init="fisher_sqrt",
        tie_embeddings=True,
    )
    model = build_model(cfg, fisher_source=sample_bytes)
    expected = build_fisher_init(sample_bytes, cfg.d_model)
    torch.testing.assert_close(model.lm_head.weight, expected, atol=1e-5, rtol=1e-5)


def test_run_start_reports_field_encoder_params_when_state_source_is_field(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_field_config(tmp_path))
    encoder = trainer.raw_model.field_encoder
    assert encoder is not None
    expected = sum(p.numel() for p in encoder.parameters())
    assert expected > 0

    trainer.fit(verbose=False)
    records = list(trainer.iter_history())
    start = next(r for r in records if r["event"] == "run_start")
    assert start["field_encoder_params"] == expected
    assert json.dumps(start)  # round-trips


def test_run_start_omits_field_encoder_params_when_state_source_is_table(
    tmp_path: Path,
) -> None:
    trainer = Trainer(tiny_config(tmp_path, state_source="table"))
    trainer.fit(verbose=False)
    records = list(trainer.iter_history())
    start = next(r for r in records if r["event"] == "run_start")
    assert "field_encoder_params" not in start


# ---------------------------------------------------------------------- smoke
def test_five_step_tiny_train_with_field_encoder_decreases_loss(tmp_path: Path) -> None:
    trainer = Trainer(_field_config(tmp_path))
    history = trainer.fit(verbose=False)
    assert len(history) == 5
    losses = [h["loss"] for h in history]
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0], losses
