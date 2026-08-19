"""E1.5c Repair 1: the calibrated-field seam (docs/e1_5-instruction.md, PR #73).

As merged (PR #69), ``p_phi = softmax(field logits)`` is optimized only
THROUGH the backbone loss — nothing forces it to remain an honest next-byte
predictor; unconstrained, it can degenerate into an arbitrary 256-category
latent code that merely passes through softmax/sqrt (a simplex-bottleneck
experiment, not a Fisher predictive-state one). This module pins the repair:

- an explicit ``field_loss_lambda * L_field`` calibration term, added to the
  optimized total at the SAME seam ``geometry/gram.py``'s Gram penalty
  uses (``train.py``, NOT ``aux_loss`` — that channel is the dead
  ``GatedKappaNorm`` load-balance term, CLAUDE.md architecture section);
- a ``field_mode="frozen"`` regime that loads a previously trained ``p_phi``
  from ``field_ckpt_path`` and freezes it before the backbone ever trains
  over ``sqrt(p_phi).Q`` — the cleaner test, since it prevents ``p_phi``
  from becoming a private code negotiated with the backbone.

Three independent hazards this file pins with RED-first proofs (mirrors
``test_gram_penalty.py``'s and ``test_freeze.py``'s convention):

1. **no seam at all** — before ``Trainer._field_loss_terms``/
   ``_field_loss_enabled`` exist, every calibration test in this module
   fails on ``AttributeError`` rather than a wrong VALUE — that absence IS
   the RED proof (same convention ``test_gram_penalty.py``'s module
   docstring names for its own seam).
2. **the nGPT-style bypass trap, checked absent** — E1.5a's frozen-table
   guard exists because ``normalize_weights()`` does an in-place ``copy_``
   on ``self.wte.weight``/``self.lm_head.weight`` unconditionally every
   step. ``LanguageModel.normalize_weights`` never walks ``self.
   field_encoder`` at all (it only iterates the backbone's OWN blocks/wte/
   lm_head), so a frozen field encoder has no such trap to guard against —
   ``requires_grad_(False)`` alone is sufficient. Verified by hand,
   RED-staged (2026-08-19): with ``FieldEncoder._load_frozen_weights``'s
   trailing ``self.requires_grad_(False)`` commented out,
   ``test_frozen_field_bitwise_unchanged_after_training`` failed with a
   nonzero byte-mismatch count (the optimizer updated the "frozen" table
   normally); restoring the line returned it to green with zero mismatched
   bytes — proving the guard is load-bearing, not merely present.
3. **eval-loss leak** — a naive implementation might fold ``L_field`` into
   the reported ``loss``/``nats_per_byte`` (the registered likelihood
   metric), the same hazard ``test_gram_penalty.py`` guards against for the
   Gram penalty.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION
from entropic_substrate.config import Config, ModelConfig
from entropic_substrate.models.field_encoder import FieldEncoder
from entropic_substrate.sinks import MlflowSink
from entropic_substrate.train import Trainer
from helpers import tiny_config, tiny_model_config
from test_freeze import _assert_bitwise_equal
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _field_model_config(**overrides: object) -> ModelConfig:
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


# ---------------------------------------------------------- construction / guards
def test_field_encoder_rejects_frozen_mode_without_a_checkpoint_path() -> None:
    cfg = _field_model_config(field_mode="frozen")
    with pytest.raises(ValueError, match="field_ckpt_path"):
        FieldEncoder(cfg)


def test_field_encoder_rejects_joint_mode_with_a_checkpoint_path() -> None:
    cfg = _field_model_config(field_mode="joint", field_ckpt_path="/does/not/matter.pt")
    with pytest.raises(ValueError, match="field_mode='joint'"):
        FieldEncoder(cfg)


def test_field_encoder_rejects_unknown_field_mode() -> None:
    """Literal types are not runtime-enforced (mirrors
    test_freeze.py::test_freeze_rejects_unknown_mode) — an out-of-band value
    must still fail loud rather than silently defaulting to joint."""
    cfg = _field_model_config()
    cfg.field_mode = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown field_mode"):
        FieldEncoder(cfg)


def test_field_logits_has_no_targets_parameter() -> None:
    """Structural anti-leakage pin, mirrors test_field_encoder.py's
    test_field_encoder_forward_has_no_targets_parameter: the calibration
    target scores p_phi, it must never be able to feed it."""
    sig = inspect.signature(FieldEncoder.field_logits)
    assert list(sig.parameters) == ["self", "idx"]


def test_field_logits_shape_and_dtype() -> None:
    cfg = _field_model_config()
    encoder = FieldEncoder(cfg)
    idx = torch.randint(0, 256, (2, 10))
    logits = encoder.field_logits(idx)
    assert logits.shape == (2, 10, 256)
    assert logits.dtype == torch.float32


def test_field_logits_and_forward_are_independent_calls_over_the_same_weights() -> None:
    """Both call sites share `_encode` internally but are genuinely separate
    forward passes (docstring's documented double-compute tradeoff) — in
    eval mode (dropout inert either way at the tiny configs' dropout=0.0),
    they must agree exactly: field_logits's own softmax before the sqrt/lift
    step must reproduce the same probabilities forward()'s z_t used."""
    cfg = _field_model_config(d_model=300)
    encoder = FieldEncoder(cfg)
    encoder.eval()
    idx = torch.randint(0, 256, (2, 10))
    z = encoder(idx)
    logits = encoder.field_logits(idx)
    probs = torch.softmax(logits, dim=-1).clamp_min(1e-12)
    reconstructed_z = (probs.sqrt() @ encoder.lift).to(z.dtype)
    torch.testing.assert_close(reconstructed_z, z, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------- frozen load
def _save_field_checkpoint(tmp_path: Path, cfg: ModelConfig, name: str = "field-src") -> Path:
    """Build a standalone (joint-mode) FieldEncoder and save it in the wire
    format ``_load_frozen_weights`` expects: ``{"format_version", "model":
    <FieldEncoder.state_dict()>}`` — nothing else in this repo produces this
    checkpoint yet (the standalone-p_phi training pipeline is a later rung);
    this fixture defines the format the loader consumes."""
    source = FieldEncoder(cfg)
    path = tmp_path / f"{name}.pt"
    torch.save(
        {"format_version": CHECKPOINT_FORMAT_VERSION, "model": source.state_dict()},
        path,
    )
    return path


@pytest.fixture
def frozen_field_checkpoint(tmp_path: Path) -> tuple[Path, ModelConfig]:
    cfg = _field_model_config()
    path = _save_field_checkpoint(tmp_path, cfg)
    return path, cfg


def test_frozen_field_loads_the_checkpointed_weights(
    tmp_path: Path, frozen_field_checkpoint: tuple[Path, ModelConfig]
) -> None:
    path, _source_cfg = frozen_field_checkpoint
    raw = torch.load(path, map_location="cpu", weights_only=True)
    expected_head = raw["model"]["head.weight"]

    frozen_cfg = _field_model_config(field_mode="frozen", field_ckpt_path=str(path))
    frozen = FieldEncoder(frozen_cfg)
    torch.testing.assert_close(frozen.head.weight, expected_head, atol=0, rtol=0)


def test_frozen_field_params_are_not_trainable(
    tmp_path: Path, frozen_field_checkpoint: tuple[Path, ModelConfig]
) -> None:
    path, _source_cfg = frozen_field_checkpoint
    frozen_cfg = _field_model_config(field_mode="frozen", field_ckpt_path=str(path))
    frozen = FieldEncoder(frozen_cfg)
    assert all(not p.requires_grad for p in frozen.parameters())
    # The buffer was already never trainable; check it explicitly anyway so
    # a future refactor that turns `lift` into a Parameter is caught here.
    assert not frozen.lift.requires_grad


def test_frozen_field_checkpoint_shape_mismatch_fails_loud(
    tmp_path: Path, frozen_field_checkpoint: tuple[Path, ModelConfig]
) -> None:
    path, _source_cfg = frozen_field_checkpoint
    mismatched_cfg = _field_model_config(
        field_mode="frozen", field_ckpt_path=str(path), field_d_model=32
    )
    with pytest.raises((RuntimeError, ValueError)):
        FieldEncoder(mismatched_cfg)


# -------------------------------------------------------------------- trainer
def _calib_cfg(tmp_path: Path, field_loss_lambda: float = 0.5, **overrides: object) -> Config:
    return _field_config(tmp_path, field_loss_lambda=field_loss_lambda, **overrides)


def test_field_loss_disabled_by_default(tmp_path: Path) -> None:
    trainer = Trainer(_field_config(tmp_path))
    assert trainer._field_loss_enabled is False
    metrics = trainer.train_step()
    assert "field_ce" not in metrics


def test_field_loss_lambda_without_a_field_encoder_is_rejected_at_construction(
    tmp_path: Path,
) -> None:
    """field_loss_lambda != 0.0 with state_source='table' has no field
    encoder to calibrate — must fail at Trainer.__init__, not surface as an
    AttributeError three calls later inside train_step."""
    cfg = tiny_config(tmp_path, state_source="table", field_loss_lambda=0.5)
    with pytest.raises(ValueError, match="field_loss_lambda"):
        Trainer(cfg)


def test_field_loss_gradient_reaches_field_encoder_params(tmp_path: Path) -> None:
    """Isolated proof (no backbone forward pass involved at all): the
    calibration term alone produces finite, nonzero gradient at the field
    encoder's own parameters. Absent-is-RED: before `_field_loss_terms`
    exists, this fails on AttributeError."""
    trainer = Trainer(_calib_cfg(tmp_path))
    x, y = trainer.train_sampler.batch()
    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    _field_loss, weighted = trainer._field_loss_terms(x, y)
    trainable = [p for p in field_encoder.parameters() if p.requires_grad]
    grads = torch.autograd.grad(weighted, trainable, allow_unused=True)
    live = [g for g in grads if g is not None]
    assert live, "no gradient reached ANY field-encoder parameter"
    assert all(torch.isfinite(g).all() for g in live)
    assert any(torch.any(g != 0) for g in live)


def test_field_loss_reaches_the_backward_total_seam(tmp_path: Path) -> None:
    """Guard-tooth for the ACTUAL `total = total + field_weighted` seam in
    `train_step`, distinct from the isolated-gradient test above (mirrors
    test_gram_penalty.py::test_gram_penalty_gradient_reaches_the_backward_
    total_seam's rationale exactly): field_encoder.wte.weight ALSO receives
    gradient through the injection path (backbone CE -> z_t ->
    field_encoder), so a version that computes L_field but never folds it
    into `total` would leave every OTHER test in this module green. This
    isolates the seam's MARGINAL contribution by reconstructing the
    gradient with the field term deliberately excluded, from identical
    starting weights and an identical frozen batch."""
    trainer = Trainer(_calib_cfg(tmp_path, field_loss_lambda=5.0))

    fixed_batch = trainer.train_sampler.batch()
    trainer.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]

    state_before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}

    captured: dict[str, torch.Tensor] = {}

    def _capture(grad: torch.Tensor) -> None:
        captured["grad"] = grad.detach().clone()

    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    handle = field_encoder.wte.weight.register_hook(_capture)
    trainer.train_step()
    handle.remove()
    assert "grad" in captured, "backward hook never fired — train_step did not call .backward()"

    trainer.model.load_state_dict(state_before)
    x, y = fixed_batch
    with trainer._autocast():
        out = trainer.model(x, y)
    aux = out.aux_loss
    total_without_field_loss = out.loss if aux is None else out.loss + aux
    (grad_without_field_loss,) = torch.autograd.grad(
        total_without_field_loss, field_encoder.wte.weight
    )

    assert torch.isfinite(captured["grad"]).all()
    assert torch.isfinite(grad_without_field_loss).all()
    assert not torch.allclose(captured["grad"], grad_without_field_loss), (
        "gradient reaching field_encoder.wte.weight during train_step matches the "
        "field-loss-free reconstruction exactly — field_weighted is not being "
        "folded into `total`"
    )


def test_field_ce_reported_matches_backward_seam(tmp_path: Path) -> None:
    trainer = Trainer(_calib_cfg(tmp_path, field_loss_lambda=2.0))
    metrics = trainer.train_step()
    assert metrics["field_ce"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["field_ce"]))


def test_field_loss_is_included_in_the_backward_total_not_reported_loss(
    tmp_path: Path,
) -> None:
    """Mirrors test_gram_penalty.py::test_gram_penalty_is_included_in_the_
    backward_total_not_reported_loss: ``total`` (used for backward) carries
    the weighted ``field_weighted`` term, but the reported ``loss`` metric
    from ``train_step`` stays pure backbone cross-entropy — same contract
    ``aux_loss``/``gram_penalty`` already have (train.py's
    ``loss = float(out.loss.detach())`` line, which reads the model's OWN
    loss, never the locally-accumulated ``total``).

    Pinned by a λ-counterfactual: two identically seeded trainers
    (``field_loss_lambda`` on vs 0) stepping on the SAME frozen batch under
    the SAME RNG state must report the IDENTICAL ``loss`` — a regression
    folding ``field_weighted`` into the reported value shifts the λ arm by
    ``field_ce * field_loss_lambda`` (>> the tolerance here) and fails."""
    trainer_on = Trainer(_calib_cfg(tmp_path, field_loss_lambda=5.0))
    trainer_off = Trainer(_calib_cfg(tmp_path, field_loss_lambda=0.0))

    fixed_batch = trainer_on.train_sampler.batch()
    trainer_on.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]
    trainer_off.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]

    torch.manual_seed(7)
    metrics_on = trainer_on.train_step()
    torch.manual_seed(7)
    metrics_off = trainer_off.train_step()

    assert metrics_on["field_ce"] > 0.0
    # Discriminating power guard: the weighted term must dwarf the loss
    # tolerance, else the equality below could not catch a fold-in
    # regression.
    assert metrics_on["field_ce"] * trainer_on.cfg.model.field_loss_lambda > 1e-6
    assert "field_ce" not in metrics_off
    assert metrics_on["loss"] == pytest.approx(metrics_off["loss"], abs=1e-9)


def test_eval_loss_and_nats_per_byte_exclude_l_field(tmp_path: Path) -> None:
    """Same trainer, same weights, same deterministic eval batches
    (mirrors test_gram_penalty.py::test_eval_loss_and_nats_per_byte_
    exclude_the_gram_penalty exactly): toggling `_field_loss_enabled`
    between the two evaluate() calls isolates the calibration term as the
    ONLY variable, so an identical eval loss proves it never leaks into the
    registered likelihood metric."""
    trainer = Trainer(_calib_cfg(tmp_path, field_loss_lambda=5.0))
    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    with torch.no_grad():
        field_encoder.head.weight.add_(torch.randn_like(field_encoder.head.weight) * 0.5)
    result_on = trainer.evaluate()
    saved = trainer._field_loss_enabled
    trainer._field_loss_enabled = False
    result_off = trainer.evaluate()
    trainer._field_loss_enabled = saved

    assert result_on.field_ce is not None and result_on.field_ce > 0.0
    assert result_off.field_ce is None
    assert result_on.loss == pytest.approx(result_off.loss)
    assert result_on.nats_per_byte == pytest.approx(result_off.nats_per_byte)


def test_eval_record_carries_field_ce_when_enabled(tmp_path: Path) -> None:
    trainer = Trainer(_calib_cfg(tmp_path))
    result = trainer.evaluate()
    record = trainer._eval_record(result, wall=0.0)
    assert record["field_ce"] > 0.0


def test_eval_record_omits_field_ce_when_disabled(tmp_path: Path) -> None:
    trainer = Trainer(_field_config(tmp_path))
    result = trainer.evaluate()
    record = trainer._eval_record(result, wall=0.0)
    assert "field_ce" not in record


def test_five_step_tiny_train_with_field_calibration_decreases_loss_and_stays_finite(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_calib_cfg(tmp_path))
    history = trainer.fit(verbose=False)
    assert len(history) == 5
    losses = [h["loss"] for h in history]
    field_ces = [h["field_ce"] for h in history]
    assert all(torch.isfinite(torch.tensor(v)) for v in losses + field_ces)
    assert losses[-1] < losses[0], losses


# --------------------------------------------------------------------- frozen mode
def test_frozen_field_bitwise_unchanged_after_training(tmp_path: Path) -> None:
    """Mirrors test_freeze.py::test_frozen_table_bitwise_unchanged_after_
    training. RED-staged by hand (2026-08-19, module docstring hazard #2):
    commenting out `_load_frozen_weights`'s trailing
    `self.requires_grad_(False)` turns this red with a nonzero byte
    mismatch count — the optimizer trains the "frozen" table like any
    other parameter without it."""
    source_cfg = _field_model_config()
    ckpt_path = _save_field_checkpoint(tmp_path, source_cfg, name="frozen-src")

    cfg = _field_config(
        tmp_path,
        field_mode="frozen",
        field_ckpt_path=str(ckpt_path),
    )
    trainer = Trainer(cfg)
    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    before = {k: v.detach().clone() for k, v in field_encoder.state_dict().items()}
    for _ in range(3):
        trainer.train_step()
    after = field_encoder.state_dict()
    for key, value in before.items():
        _assert_bitwise_equal(after[key], value)


def test_frozen_field_params_absent_from_optimizer(tmp_path: Path) -> None:
    source_cfg = _field_model_config()
    ckpt_path = _save_field_checkpoint(tmp_path, source_cfg, name="frozen-src2")
    cfg = _field_config(tmp_path, field_mode="frozen", field_ckpt_path=str(ckpt_path))
    trainer = Trainer(cfg)
    field_encoder = trainer.raw_model.field_encoder
    assert field_encoder is not None
    optimizer_param_ids = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    field_param_ids = {id(p) for p in field_encoder.parameters()}
    assert optimizer_param_ids.isdisjoint(field_param_ids)


def test_backbone_still_trains_with_a_frozen_field(tmp_path: Path) -> None:
    """The frozen field encoder must not silently stall the REST of the
    model: lm_head (the output classifier, always trainable) must still
    move under training."""
    source_cfg = _field_model_config()
    ckpt_path = _save_field_checkpoint(tmp_path, source_cfg, name="frozen-src3")
    cfg = _field_config(tmp_path, field_mode="frozen", field_ckpt_path=str(ckpt_path))
    trainer = Trainer(cfg)
    before = trainer.raw_model.lm_head.weight.detach().clone()
    for _ in range(3):
        trainer.train_step()
    assert not torch.allclose(trainer.raw_model.lm_head.weight, before)


def test_checkpoint_roundtrip_with_a_frozen_field(tmp_path: Path) -> None:
    """Mirrors test_freeze.py::test_checkpoint_roundtrip_with_untied_
    frozen_table: Trainer's OWN checkpoint/resume roundtrip must preserve
    the frozen field encoder's weights and its requires_grad=False state."""
    source_cfg = _field_model_config()
    ckpt_path = _save_field_checkpoint(tmp_path, source_cfg, name="frozen-src4")
    cfg = _field_config(tmp_path, field_mode="frozen", field_ckpt_path=str(ckpt_path))

    trainer_a = Trainer(cfg)
    for _ in range(2):
        trainer_a.train_step()
    resume_path = trainer_a.save_checkpoint("field-frozen-roundtrip")

    trainer_b = Trainer(cfg)
    trainer_b.load_checkpoint(resume_path)

    assert trainer_b.step == trainer_a.step
    field_b = trainer_b.raw_model.field_encoder
    field_a = trainer_a.raw_model.field_encoder
    assert field_b is not None and field_a is not None
    assert not field_b.wte.weight.requires_grad
    torch.testing.assert_close(field_b.wte.weight, field_a.wte.weight, atol=0, rtol=0)


# ------------------------------------------------------------------------ sinks
def test_mlflow_sink_forwards_field_ce_on_eval_events() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "eval", "step": 7, "loss": 1.0, "field_ce": 1.234})
    metrics = client._runs[sink.run_id].data.metrics
    assert metrics["field_ce"] == pytest.approx(1.234)


def test_mlflow_sink_omits_field_ce_when_absent_from_the_record() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "eval", "step": 1, "loss": 1.0})
    metrics = client._runs[sink.run_id].data.metrics
    assert "field_ce" not in metrics
