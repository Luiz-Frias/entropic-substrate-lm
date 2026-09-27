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
import itertools
import json
import math
from pathlib import Path

import pytest
import torch

from entropic_substrate.core.substrate.backbone import build_model
from entropic_substrate.core.substrate.field_encoder import (
    FieldCalibrationAccumulator,
    FieldEncoder,
    FieldEncoding,
)
from entropic_substrate.core.substrate.fisher import build_fisher_init, semi_orthogonal_lift
from entropic_substrate.historical.e_series.train import Trainer
from entropic_substrate.kernel.runtime.config import Config
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


@pytest.mark.parametrize("field_state", ["prob", "hidden"])
def test_build_model_rejects_field_state_axis_with_table_source(field_state: str) -> None:
    """P7 (FOR-924): field_state selects among F/P/H CONSTRUCTIONS of a
    field encoder's output — meaningless without one. Same loud-refusal
    posture as field_loss_lambda/field_ckpt_path/field_mode above, for the
    same root cause: a non-default field_state under state_source='table'
    would be silently ignored (no field encoder exists to read it).
    Parametrized over BOTH non-default legs (P and H) — the guard's
    condition is a flat `!= "sqrt_prob"` check with no leg-specific branch,
    so one arm passing does not establish the other does."""
    cfg = tiny_model_config(state_source="table", field_state=field_state)
    with pytest.raises(ValueError, match="field_state"):
        build_model(cfg)


# ------------------------------------------------------------------- geometry
def test_field_encoder_output_matches_backbone_width() -> None:
    cfg = _field_model_config(d_model=32)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    z = encoder(idx).z
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
    z = encoder(idx).z
    norms = z.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


def test_field_encoder_lift_buffer_matches_e1s_fixed_lift_and_has_no_grad() -> None:
    cfg = _field_model_config(d_model=32)
    encoder = FieldEncoder(cfg)
    expected = semi_orthogonal_lift(256, cfg.d_model, seed=0).float()
    torch.testing.assert_close(encoder.lift, expected, atol=0, rtol=0)
    assert not encoder.lift.requires_grad


# ---------------------------------------------------------- P7: H/P arms
# docs/e1_5-instruction.md Repair 2 (the four-arm table): F/P/H share ONE
# p_phi (architecture, size, context, data) and differ only in how p_phi's
# output becomes the state z_t handed to the backbone. field_state selects
# among the three constructions this rung registers (F is the pre-existing
# default "sqrt_prob"; T, the static-table arm, has no field_state at all).
_ARM_FIELD_STATES: tuple[str, ...] = ("sqrt_prob", "prob", "hidden")


def test_field_encoder_rejects_unknown_field_state() -> None:
    cfg = _field_model_config(field_state="quantum")
    with pytest.raises(ValueError, match="unknown field_state"):
        FieldEncoder(cfg)


def test_field_encoder_prob_rows_are_generally_not_unit_norm() -> None:
    """P (field_state='prob'): z = p . Q, no sqrt. Cauchy-Schwarz pins
    Sum(p_i^2) <= (Sum p_i)^2 = 1 with equality only at a one-hot
    distribution, so ||z|| = sqrt(Sum p_i^2) < 1 for any non-degenerate
    p_phi (certainly the near-uniform output of a freshly-initialised
    estimator) — the documented deliberate contrast with F's unit-norm rows,
    P being the control for the sqrt map itself."""
    cfg = _field_model_config(field_state="prob", d_model=300)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    z = encoder(idx).z
    norms = z.norm(dim=-1)
    assert torch.all(norms < 0.999), norms
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


def test_field_encoder_hidden_state_uses_a_trainable_linear_lift() -> None:
    """H (field_state='hidden'): z = W . h_phi,t via a trainable
    Linear(field_d_model, d_model, bias=False) — the ONLY arm whose lift
    trains. F/P share the fixed semi_orthogonal_lift buffer and carry no
    such parameter."""
    cfg = _field_model_config(field_state="hidden", d_model=48)
    encoder = FieldEncoder(cfg)
    assert encoder.state_lift is not None
    assert isinstance(encoder.state_lift, torch.nn.Linear)
    assert encoder.state_lift.weight.requires_grad
    assert tuple(encoder.state_lift.weight.shape) == (48, cfg.field_d_model)
    assert encoder.state_lift.bias is None


@pytest.mark.parametrize("field_state", ["sqrt_prob", "prob"])
def test_field_encoder_non_hidden_states_carry_no_trainable_lift(field_state: str) -> None:
    cfg = _field_model_config(field_state=field_state)
    encoder = FieldEncoder(cfg)
    assert encoder.state_lift is None


def test_hidden_state_adds_exactly_the_trainable_lift_over_prob() -> None:
    """Direct measurement of the "only trainable lift, param delta" claim
    (FOR-924), independent of test_config.py's whole-model band pin: H's
    FieldEncoder must carry EXACTLY field_d_model*d_model more parameters
    than an otherwise-identical P encoder — no more, no less (both share
    the same fixed 256x256... buffer-excluded p_phi backbone)."""
    prob_cfg = _field_model_config(field_state="prob", d_model=64)
    hidden_cfg = _field_model_config(field_state="hidden", d_model=64)
    prob_params = sum(p.numel() for p in FieldEncoder(prob_cfg).parameters())
    hidden_params = sum(p.numel() for p in FieldEncoder(hidden_cfg).parameters())
    assert hidden_params - prob_params == hidden_cfg.field_d_model * hidden_cfg.d_model


# ----------------------------------------------------------------- causality
@pytest.mark.parametrize("field_state", _ARM_FIELD_STATES)
def test_field_encoder_is_strictly_causal(field_state: str) -> None:
    """Perturbing the LAST position's input byte must not move any earlier
    position's z_t. RED-first proof recorded in the phase report:
    monkeypatching the underlying attention to `is_causal=False` fails this
    test immediately. Parametrized over all three registered field_state
    arms (FOR-924): F/P/H share the same causal p_phi backbone this proof
    already covers, but the STATE CONSTRUCTION differs downstream of it, so
    the guarantee must be re-proven for each rather than assumed to transfer."""
    cfg = _field_model_config(field_state=field_state)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (2, 10))
    base = encoder(idx).z

    bumped = idx.clone()
    bumped[:, -1] = (bumped[:, -1] + 137) % 256
    perturbed = encoder(bumped).z
    torch.testing.assert_close(perturbed[:, :-1], base[:, :-1], atol=1e-6, rtol=0)


@pytest.mark.parametrize("field_state", _ARM_FIELD_STATES)
def test_field_encoder_next_byte_alignment(field_state: str) -> None:
    """x_t affects z_t (positive control — rules out a trivially-causal
    constant encoder); x_{t+1} (a future byte) does not affect z_t (the
    causality contract restated at a single position, independent of the
    whole-sequence proof above). Parametrized over all three arms — see
    test_field_encoder_is_strictly_causal."""
    cfg = _field_model_config(field_state=field_state, field_n_layers=1)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (1, 6))
    base = encoder(idx).z

    changed_t = idx.clone()
    changed_t[:, 3] = (changed_t[:, 3] + 91) % 256
    z_changed_t = encoder(changed_t).z
    # Finite BEFORE nonzero: `nan != nan`/`allclose(nan, nan)==False` means a
    # NaN-poisoned z_changed_t would otherwise slip past `not allclose(...)`
    # and read as "position t changed" instead of "the encoder broke"
    # (mirrors test_gram_penalty.py's identical guard-ordering rationale).
    assert torch.isfinite(z_changed_t[:, 3]).all()
    assert not torch.allclose(z_changed_t[:, 3], base[:, 3])

    changed_future = idx.clone()
    changed_future[:, 4] = (changed_future[:, 4] + 91) % 256
    z_changed_future = encoder(changed_future).z
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
@pytest.mark.parametrize("field_state", _ARM_FIELD_STATES)
def test_field_encoder_gradients_reach_its_own_parameters_through_backbone_ce(
    tmp_path: Path, field_state: str
) -> None:
    trainer = Trainer(_field_config(tmp_path, field_state=field_state))
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
    if field_state == "hidden":
        # H's own trainable lift must be among the params gradient reached —
        # this is what "the only trainable lift" (FOR-924) actually promises.
        assert encoder.state_lift is not None
        assert encoder.state_lift.weight.grad is not None
        assert torch.isfinite(encoder.state_lift.weight.grad).all()
        assert torch.any(encoder.state_lift.weight.grad != 0)


@pytest.mark.parametrize("field_state", _ARM_FIELD_STATES)
def test_field_encoder_finite_under_bf16_autocast(field_state: str) -> None:
    cfg = _field_model_config(field_state=field_state, field_n_layers=1)
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = encoder(idx).z
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
        z = encoder(idx).z
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


@pytest.mark.parametrize("field_state", _ARM_FIELD_STATES)
def test_run_start_records_field_state(tmp_path: Path, field_state: str) -> None:
    """FOR-924: "run_start records field_state" — satisfied structurally by
    ``run_start_record["config"] = self.cfg.to_dict()`` (train.py) once
    ``field_state`` is a real ``ModelConfig`` field (``dataclasses.asdict``
    walks it automatically); this pins that channel rather than trusting it
    stays wired as the record's shape evolves."""
    trainer = Trainer(_field_config(tmp_path, field_state=field_state))
    trainer.fit(verbose=False)
    records = list(trainer.iter_history())
    start = next(r for r in records if r["event"] == "run_start")
    assert start["config"]["model"]["field_state"] == field_state


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


# ============================================================== P8 (FOR-925)
# Calibration telemetry, behavioral anti-leakage, geometry correspondence and
# context-sensitivity diagnostics for p_phi (docs/e1_5-instruction.md
# "Calibration, geometry, and context-sensitivity diagnostics").


# --------------------------------------------------- calibration accumulator
def test_field_calibration_accumulator_matches_hand_computed_toy_values() -> None:
    """4-sample, 2-class toy input, entirely hand-derived (not reusing the
    implementation under test): two samples land in one confidence bin
    (0.5, one correct/one wrong), two in another (0.9, both correct) — the
    minimal case that exercises binning as well as the four scalar
    reductions. ln(9) makes softmax([ln9, 0]) exactly [0.9, 0.1]."""
    ln9 = 2.1972245773362196
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0], [ln9, 0.0], [ln9, 0.0]])
    targets = torch.tensor([0, 1, 0, 0])

    acc = FieldCalibrationAccumulator()
    acc.update(logits, targets)
    stats = acc.finalize()

    assert stats["ce"] == pytest.approx(0.3992538481088858, rel=1e-6)
    assert stats["brier"] == pytest.approx(0.26, rel=1e-6)
    assert stats["entropy"] == pytest.approx(0.5091150769756968, rel=1e-6)
    assert stats["top1"] == pytest.approx(0.75, rel=1e-9)
    assert stats["ece"] == pytest.approx(0.05, rel=1e-6)


def test_field_calibration_accumulator_rejects_non_fp32_logits() -> None:
    acc = FieldCalibrationAccumulator()
    with pytest.raises(ValueError, match="fp32"):
        acc.update(torch.zeros(2, 4, dtype=torch.float64), torch.zeros(2, dtype=torch.long))


def test_field_calibration_accumulator_finalize_without_update_fails_loud() -> None:
    """Empty-input pin: never silently return a NaN-poisoned summary (mirrors
    ``Trainer.evaluate``'s np.mean-on-empty hazard the module docstring warns
    against, at the accumulator level)."""
    acc = FieldCalibrationAccumulator()
    with pytest.raises(RuntimeError, match="no updates"):
        acc.finalize()


def test_field_calibration_accumulator_ignores_a_zero_sized_update() -> None:
    """An empty batch (N=0) must be a no-op, not a div-by-zero inside update
    itself — the RuntimeError above is reserved for NEVER having updated at
    all, not for one incidental zero-sized call among real ones."""
    acc = FieldCalibrationAccumulator()
    acc.update(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
    acc.update(torch.tensor([[0.0, 0.0]]), torch.tensor([0]))
    stats = acc.finalize()
    assert stats["top1"] == pytest.approx(1.0)


def test_field_calibration_accumulator_streaming_matches_a_single_shot_call() -> None:
    """The eval loop calls ``update`` once per batch; the reported summary
    must not depend on how the eval set was chunked. Same data, fed as four
    ragged batches vs one single call, must finalize to matching stats."""
    torch.manual_seed(3)
    n_classes = 256
    all_logits = torch.randn(60, n_classes)
    all_targets = torch.randint(0, n_classes, (60,))

    streamed = FieldCalibrationAccumulator()
    for lo, hi in ((0, 10), (10, 25), (25, 45), (45, 60)):
        streamed.update(all_logits[lo:hi], all_targets[lo:hi])
    streamed_stats = streamed.finalize()

    single_shot = FieldCalibrationAccumulator()
    single_shot.update(all_logits, all_targets)
    single_shot_stats = single_shot.finalize()

    for key in streamed_stats:
        assert streamed_stats[key] == pytest.approx(single_shot_stats[key], rel=1e-6), key


# ------------------------------------------------------ behavioral anti-leakage
def test_field_encoder_prefix_identical_sequences_share_prefix_states_and_diverge_after() -> None:
    """Registered behavioral complement (docs/e1_5-instruction.md's
    anti-leakage section) to the structural no-targets signature pin: two
    FULL sequences identical through position t, divergent after, must
    produce bit-tolerant-EQUAL z_{<=t} AND states that differ once the
    inputs do — the positive control ruling out a degenerate constant
    encoder trivially passing the equality half."""
    cfg = _field_model_config(d_model=48)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    t = 4
    torch.manual_seed(5)
    prefix = torch.randint(0, 256, (1, t + 1))
    tail_a = torch.randint(0, 256, (1, 5))
    tail_b = tail_a.clone()
    tail_b[:, 0] = (tail_a[:, 0] + 113) % 256  # force genuine divergence at t+1

    seq_a = torch.cat([prefix, tail_a], dim=1)
    seq_b = torch.cat([prefix, tail_b], dim=1)

    z_a = encoder(seq_a).z
    z_b = encoder(seq_b).z

    torch.testing.assert_close(z_a[:, : t + 1], z_b[:, : t + 1], atol=1e-6, rtol=0)
    assert torch.isfinite(z_a[:, t + 1]).all()
    assert torch.isfinite(z_b[:, t + 1]).all()
    assert not torch.allclose(z_a[:, t + 1], z_b[:, t + 1])


def test_targets_never_alter_the_field_encoders_own_output(tmp_path: Path) -> None:
    """Behavioral complement to test_field_encoder_forward_has_no_targets_
    parameter: a forward hook on the field encoder itself, inside a REAL
    ``LanguageModel.forward`` call, proves the encoder's output is
    bit-identical whether ``targets`` is ``None``, the real batch targets,
    or an unrelated garbage tensor — a check the pure signature pin and even
    the logits-level integration test cannot make, since both stop at the
    encoder's caller rather than its own output.

    The hook FIRING at all is itself load-bearing (asserted below): the
    backbone must reach the encoder through ``nn.Module.__call__``, not by
    calling ``forward``/``_encode`` directly, or this whole check would
    silently observe nothing. Both channels of :class:`FieldEncoding` are
    compared, since the calibration logits now leave the encoder too
    (``LMOutput.field_logits``) and are just as target-blind as ``z``."""
    trainer = Trainer(_field_config(tmp_path))
    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    x, y = trainer.train_sampler.batch()
    trainer.model.eval()

    captured: list[FieldEncoding] = []

    def _capture(_module: object, _inputs: object, output: FieldEncoding) -> None:
        captured.append(
            FieldEncoding(logits=output.logits.detach().clone(), z=output.z.detach().clone())
        )

    handle = field_encoder.register_forward_hook(_capture)
    with torch.no_grad():
        trainer.model(x, None)
        trainer.model(x, y)
        garbage_targets = (y + 173) % 256
        trainer.model(x, garbage_targets)
    handle.remove()

    assert len(captured) == 3, "forward hook never fired — the backbone bypassed __call__"
    for other in captured[1:]:
        torch.testing.assert_close(captured[0].z, other.z, atol=0, rtol=0)
        torch.testing.assert_close(captured[0].logits, other.logits, atol=0, rtol=0)


# --------------------------------------------------------- geometry correspondence
def test_geometry_correspondence_hellinger_identity_under_the_semiorthogonal_lift() -> None:
    """theta(z_t, z_s) = arccos(z_t . z_s) against the Hellinger distance
    between the underlying predictions: H^2(p, q) = 1 - BC(p, q), with the
    Bhattacharyya coefficient BC(p, q) = sum(sqrt(p_i * q_i)). Because
    z = sqrt(p) @ Q and Q is EXACTLY semi-orthogonal at d_model=300 (>=256 —
    semi_orthogonal_lift's own isometry regime, same as
    test_field_encoder_z_rows_are_unit_norm), z_t . z_s reproduces BC(p_t,
    p_s) numerically — this proves the correspondence the docs section
    claims, rather than assuming it holds by construction.

    Seeded (see below): the theta-level assertion's tolerance is DERIVED
    from this fixture's measured Bhattacharyya coefficient, so the fixture
    has to be the same one on every run."""
    # Seeds BOTH the encoder's own init draw and `idx` below — everything
    # `bhattacharyya` depends on. Without it this test's measured value
    # (and therefore the arccos tolerance derived from it) is a per-run
    # lottery; 11 is the same arbitrary seed the mixture-path test below
    # uses, chosen before measuring rather than picked for a flattering
    # number.
    torch.manual_seed(11)
    cfg = _field_model_config(d_model=300)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (2, 6))
    logits = encoder.field_logits(idx)
    probs = torch.softmax(logits, dim=-1)
    z = encoder(idx).z

    p_t, p_s = probs[0, 0], probs[0, 3]
    z_t, z_s = z[0, 0], z[0, 3]

    # H^2(p, q) = 1 - BC(p, q) (docstring above) — bhattacharyya below IS
    # BC(p_t, p_s); no separate hellinger_sq local is kept, since
    # `1.0 - (1.0 - bhattacharyya)` collapses back to `bhattacharyya`
    # exactly and computing it just to re-derive that identity was the
    # redundant second assertion removed below (CodeRabbit finding, P8
    # review round).
    bhattacharyya = torch.sum(torch.sqrt(p_t * p_s))
    cos_theta = torch.dot(z_t, z_s)

    torch.testing.assert_close(cos_theta, bhattacharyya, atol=1e-4, rtol=1e-4)

    # RETRACTION of the note that stood here: it deferred this theta-level
    # check as unmeasurable because the test carried no `torch.manual_seed`.
    # It is seeded now, so the measurement exists and the check lands.
    #
    # MEASURED at seed 11 (2026-08-20, fp32, CPU/arm64):
    #   cos_theta     = 0.7361146211624146
    #   bhattacharyya = 0.7361146211624146   (bit-identical at this fixture)
    #   theta         = 0.743484357 rad, for both -> residual exactly 0.0
    # Same probe over seeds 0/7/23 to size the fp32 noise this identity
    # actually carries (the lift's own QR is only semi-orthogonal to fp32,
    # and the two sides sum different numbers of terms in different
    # orders): worst cos-space residual 1.192e-07 (seed 0), worst
    # theta-space residual 1.797e-07.
    #
    # DERIVATION. d/dx arccos(x) = -1/sqrt(1 - x^2), so a cos-space
    # residual r shows up in theta as ~A*r with A = 1/sqrt(1 - cos^2). At
    # the measured cos_theta, A = 1/sqrt(1 - 0.73611462^2) = 1.4774. That
    # fixes the tolerance from BOTH sides:
    #
    #   floor   — it must clear the amplified fp32 noise:
    #             1.4774 * 1.192e-7 = 1.8e-7 (matching the measured 1.797e-7).
    #             A loose naive-summation bound over the 300-term dot,
    #             300 * 6e-8 * A ~= 2.7e-5, is the pessimistic cross-BLAS
    #             version of the same number.
    #   ceiling — it must stay BELOW what the cos assertion above already
    #             implies, or this line could never fail on its own (house
    #             rule 5). That assertion tolerates
    #             atol + rtol*|bc| = 1e-4 + 1e-4*0.73611 = 1.7361e-4 in cos
    #             space, i.e. 1.4774 * 1.7361e-4 = 2.565e-4 in theta.
    #
    # atol=5e-5 (rtol=0) sits between them: ~280x over the measured noise,
    # ~2x over the pessimistic cross-BLAS bound, and ~5x under the ceiling
    # — so it is a genuinely independent constraint, not a restatement.
    # GUARD-TOOTH (2026-08-20): with `bhattacharyya + 1.2e-4` substituted
    # into both assertions, the cos-space assert_close still PASSED (1.2e-4
    # < 1.7361e-4) while this one went RED at |dtheta| = 1.7732e-4 > 5e-5 —
    # the residual band where only this assertion bites is real and
    # reachable, which is exactly what the removed `1 - (1 - x)` assertion
    # could never claim.
    #
    # The fp32-near-1 trap 85ee3fd fixed for the context-sensitivity test
    # (A -> infinity as cos -> 1) does not bite here, and the seed is what
    # makes that a fact rather than a hope: this fixture sits at
    # cos_theta = 0.736, where A is 1.48. No `.clamp(-1, 1)` before arccos
    # for the same reason — at 0.736 the domain edge is 0.26 away, ~4.4e6
    # fp32 ULPs, so a clamp here could only ever hide a fixture that had
    # silently become a different test.
    theta_cos = torch.arccos(cos_theta)
    theta_bc = torch.arccos(bhattacharyya)
    torch.testing.assert_close(theta_cos, theta_bc, atol=5e-5, rtol=0)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (p.log() - q.log()))


def _jensen_shannon(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def test_geometry_correspondence_theta_and_js_move_together_along_a_mixture_path() -> None:
    """JS divergence is NOT a closed function of theta in general (only
    Hellinger has the exact arccos identity proven above) — a bare
    rank-correlation-1 assertion over arbitrary distribution pairs would be
    FALSE. This instead proves the weaker, honest claim
    docs/e1_5-instruction.md actually makes ("nearby states have similar
    continuation distributions") along a deterministic mixture path
    p_alpha = (1-alpha)*p0 + alpha*p1: both theta(z_0, z_alpha) and
    JS(p_0, p_alpha) must increase monotonically as alpha grows."""
    torch.manual_seed(11)
    d_model = 300
    lift = semi_orthogonal_lift(256, d_model, seed=0).float()
    p0 = torch.softmax(torch.randn(256), dim=-1)
    p1 = torch.softmax(torch.randn(256), dim=-1)
    z0 = torch.sqrt(p0.clamp_min(1e-12)) @ lift

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    thetas: list[float] = []
    jss: list[float] = []
    for alpha in alphas:
        p_alpha = (1.0 - alpha) * p0 + alpha * p1
        z_alpha = torch.sqrt(p_alpha.clamp_min(1e-12)) @ lift
        cos_theta = torch.clamp(torch.dot(z0, z_alpha), -1.0, 1.0)
        thetas.append(float(torch.arccos(cos_theta)))
        jss.append(float(_jensen_shannon(p0, p_alpha)))

    assert all(b > a for a, b in itertools.pairwise(thetas)), thetas
    assert all(b > a for a, b in itertools.pairwise(jss)), jss


# ----------------------------------------------------------- context sensitivity
def test_context_sensitivity_field_gives_nonzero_spread_table_gives_exactly_zero() -> None:
    """The clearest demonstration the state is no longer "byte encoding"
    (docs/e1_5-instruction.md): for the SAME current byte in different
    contexts, a genuine contextual field separates the resulting states
    (nonzero pairwise angle spread). A static table lookup is the
    DISCRIMINATING CONTROL — it must give EXACTLY zero spread, since it is
    indexed by the byte id alone and never reads what precedes it."""
    byte_value = 42
    contexts = [
        torch.tensor([[10, 20, byte_value]]),
        torch.tensor([[200, 5, byte_value]]),
        torch.tensor([[byte_value, byte_value, byte_value]]),
    ]

    field_encoder = FieldEncoder(_field_model_config(d_model=300))
    field_encoder.eval()
    with torch.no_grad():
        field_states = [field_encoder(ctx).z[0, -1] for ctx in contexts]
    # Direct tensor-equality check FIRST: this is the load-bearing
    # assertion. An arccos-based spread check alone cannot catch a
    # context-blind (one-hot-per-byte lookup) encoder, because even a
    # BIT-IDENTICAL pair of states returns torch.dot(state, state) ==
    # 1 - 2**-23 (one fp32 ULP below 1.0, from non-associative float32
    # summation across d_model=300 terms) rather than exactly 1.0, and
    # arccos amplifies that residual to sqrt(2 * 2**-23) == 4.8828125e-04
    # rad — ABOVE a naive 1e-4 threshold even for a genuinely
    # context-blind encoder. Measured directly against a one-hot@lift
    # regression that made `_encode` byte-lookup-only: this test PASSED
    # under that regression before this fix (PR #79 review finding).
    for i in range(len(field_states)):
        for j in range(i + 1, len(field_states)):
            assert not torch.allclose(field_states[i], field_states[j], atol=0.0, rtol=0.0), (
                i,
                j,
                "field states must differ across contexts sharing the same trailing byte",
            )
    field_angles = [
        torch.arccos(torch.clamp(torch.dot(field_states[i], field_states[j]), -1.0, 1.0))
        for i in range(len(field_states))
        for j in range(i + 1, len(field_states))
    ]
    # Secondary spread check, threshold set ~10x above the measured fp32
    # arccos-near-1 noise floor (4.8828125e-04 rad, see comment above) so
    # it stays a meaningful spread assertion rather than an instrument
    # that a bit-identical pair can pass by fp32 accident.
    assert any(float(angle) > 5e-3 for angle in field_angles), field_angles

    table_model = build_model(tiny_model_config(state_source="table", d_model=300))
    table_model.eval()
    with torch.no_grad():
        table_states = [table_model.wte(ctx)[0, -1] for ctx in contexts]
    for i in range(len(table_states)):
        for j in range(i + 1, len(table_states)):
            torch.testing.assert_close(table_states[i], table_states[j], atol=0, rtol=0)
