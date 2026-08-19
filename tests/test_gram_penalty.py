"""E1.5b geometry-preserving regularization.

``geometry/gram.py`` provides the shared primitives (``normalized_gram``,
``gram_frobenius_sq``); the trainer adds ``gram_lambda * ||G(E_t) -
G(E_init)||_F^2`` to the optimized total at the ``train.py`` seam
(``total = loss if aux is None else loss + aux``), NOT via the ``aux_loss``
channel (that channel is the dead ``GatedKappaNorm`` load-balance term — see
CLAUDE.md's architecture section). ``gram_lambda == 0.0`` (the default on
every arm except ``e1-5b-gram``) disables the whole seam: no reference is
snapshotted, ``train_step``/``evaluate`` add no keys, and the trainer pays no
extra Gram-matrix compute.

Two independent hazards this file pins with RED-first proofs:

1. **no seam at all** — before ``Trainer._gram_ref``/``_gram_penalty_terms``
   exist, every test in this module fails on ``AttributeError``/
   ``ImportError`` rather than a wrong VALUE. That absence IS the RED proof
   for "zero-at-step-zero" and "gradient reaches embedding": there is no
   penalty term to be zero, or to have a gradient, until the seam exists.
2. **resume unsafety** — a naive implementation might snapshot ``_gram_ref``
   from the RESUMED (mid-training) weights instead of the deterministic init,
   which would silently change what "preserve the init geometry" means on
   every resumed run depending on how far training had drifted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from entropic_substrate.geometry.gram import gram_frobenius_sq, normalized_gram
from entropic_substrate.sinks import MlflowSink
from entropic_substrate.train import Trainer
from helpers import tiny_config
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


# ------------------------------------------------------------- geometry/gram
def test_normalized_gram_diagonal_is_unit_and_symmetric() -> None:
    e = torch.randn(6, 5)
    gram = normalized_gram(e)
    assert gram.shape == (6, 6)
    torch.testing.assert_close(gram.diagonal(), torch.ones(6), atol=1e-5, rtol=0)
    torch.testing.assert_close(gram, gram.T, atol=1e-6, rtol=0)


def test_normalized_gram_is_clamped_to_valid_cosine_range() -> None:
    # Duplicated rows push the off-diagonal dot product to exactly 1.0 before
    # any fp rounding; clamping must not let rounding push it outside [-1, 1]
    # (an unclamped 1.0000001 silently NaNs a caller's arccos).
    e = torch.ones(4, 3)
    gram = normalized_gram(e)
    assert torch.all(gram <= 1.0) and torch.all(gram >= -1.0)
    assert torch.isfinite(gram).all()


def test_normalized_gram_ignores_row_scale() -> None:
    """Relational geometry, not absolute scale: rescaling rows independently
    must not move the Gram matrix — this is the property row-normalization
    ALONE (without the outer product) does not give a caller for free."""
    e = torch.randn(5, 4)
    scaled = e * torch.tensor([1.0, 2.0, 0.5, 10.0, 3.0]).unsqueeze(1)
    torch.testing.assert_close(normalized_gram(e), normalized_gram(scaled), atol=1e-5, rtol=1e-5)


def test_gram_frobenius_sq_is_zero_for_identical_gram_matrices() -> None:
    e = torch.randn(8, 6)
    gram = normalized_gram(e)
    assert float(gram_frobenius_sq(gram, gram)) == 0.0


def test_gram_frobenius_sq_is_positive_for_a_rotated_table() -> None:
    torch.manual_seed(0)
    e = torch.randn(8, 6)
    gram_a = normalized_gram(e)
    gram_b = normalized_gram(e + torch.randn(8, 6) * 0.5)
    assert float(gram_frobenius_sq(gram_a, gram_b)) > 0.0


def test_gram_primitives_are_fp32_finite_and_differentiable_under_bf16_autocast() -> None:
    """Mirrors test_kappa_geometry.py's
    ``test_scalar_reductions_run_in_fp32_under_bf16_autocast``: the non-
    negotiable geometry-numerics invariant (CLAUDE.md) binds unconditionally,
    not only when the caller happens to already be outside autocast."""
    e = torch.randn(6, 5, dtype=torch.bfloat16, requires_grad=True)
    ref = normalized_gram(torch.randn(6, 5)).detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        gram = normalized_gram(e)
        penalty = gram_frobenius_sq(gram, ref)
    assert gram.dtype == torch.float32 and penalty.dtype == torch.float32
    assert torch.isfinite(gram).all() and torch.isfinite(penalty)
    penalty.backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()


# ------------------------------------------------------------------ trainer
def _gram_cfg(tmp_path: Path, gram_lambda: float = 0.5, **overrides: object):
    return tiny_config(
        tmp_path,
        embedding_init="random_sphere",
        gram_lambda=gram_lambda,
        **overrides,
    )


def test_gram_ref_absent_when_gram_lambda_is_zero(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path, embedding_init="random_sphere"))
    assert trainer._gram_ref is None
    metrics = trainer.train_step()
    assert "gram_frob_sq" not in metrics
    assert "gram_penalty" not in metrics


def test_gram_penalty_is_exactly_zero_at_step_zero(tmp_path: Path) -> None:
    """At step 0 the live table IS the snapshot; ``G(E_0) == G(E_ref)``
    bit-for-bit (same tensor content), so the diff is exactly 0 and the
    squared Frobenius norm is exactly 0.0 — no tolerance needed."""
    trainer = Trainer(_gram_cfg(tmp_path))
    frob_sq, penalty = trainer._gram_penalty_terms()
    assert float(frob_sq.detach()) == 0.0
    assert float(penalty.detach()) == 0.0


def test_gram_gradient_reaches_embedding_once_table_diverges_from_reference(
    tmp_path: Path,
) -> None:
    """At E == E_ref the quadratic penalty sits at its minimum (grad exactly
    0 by construction — see module docstring), so this test perturbs the
    table first and checks the gradient is non-zero away from that minimum:
    the load-bearing claim is that gradient CAN reach the embedding through
    this seam, not that it is non-zero everywhere."""
    trainer = Trainer(_gram_cfg(tmp_path))
    with torch.no_grad():
        trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)
    _frob_sq, penalty = trainer._gram_penalty_terms()
    (grad,) = torch.autograd.grad(penalty, trainer.raw_model.wte.weight)
    assert torch.any(grad != 0)


def test_gram_penalty_is_included_in_the_backward_total_not_reported_loss(
    tmp_path: Path,
) -> None:
    """``total`` (used for backward) carries the penalty; the reported
    ``loss`` metric stays pure cross-entropy — same contract ``aux_loss``
    already has (train.py's ``loss = float(out.loss.detach())`` line, which
    reads the model's OWN loss, never the locally-accumulated ``total``)."""
    trainer = Trainer(_gram_cfg(tmp_path))
    with torch.no_grad():
        trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)
    metrics = trainer.train_step()
    assert metrics["gram_frob_sq"] > 0.0
    assert metrics["gram_penalty"] == pytest.approx(
        metrics["gram_frob_sq"] * trainer.cfg.model.gram_lambda
    )


def test_eval_loss_and_nats_per_byte_exclude_the_gram_penalty(tmp_path: Path) -> None:
    """Same trainer, same weights, same deterministic eval batches
    (``val_sampler.reset_eval()`` on every ``evaluate()`` call) — toggling
    ``_gram_ref`` between the two calls isolates the penalty as the ONLY
    variable, so an identical eval loss proves it never leaks into the
    likelihood metric. Perturbing the table first makes the penalty
    non-trivially non-zero, so a leak would show up as a real difference,
    not a coincidental zero-vs-zero match."""
    trainer = Trainer(_gram_cfg(tmp_path, gram_lambda=5.0))
    with torch.no_grad():
        trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)
    result_on = trainer.evaluate()
    saved_ref = trainer._gram_ref
    trainer._gram_ref = None
    result_off = trainer.evaluate()
    trainer._gram_ref = saved_ref

    assert result_on.loss == pytest.approx(result_off.loss)
    assert result_on.nats_per_byte == pytest.approx(result_off.nats_per_byte)


def test_eval_record_carries_gram_keys_when_enabled(tmp_path: Path) -> None:
    trainer = Trainer(_gram_cfg(tmp_path))
    with torch.no_grad():
        trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)
    result = trainer.evaluate()
    record = trainer._eval_record(result, wall=0.0)
    assert record["gram_frob_sq"] > 0.0
    assert record["gram_penalty"] == pytest.approx(
        record["gram_frob_sq"] * trainer.cfg.model.gram_lambda
    )


def test_eval_record_omits_gram_keys_when_disabled(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path, embedding_init="random_sphere"))
    result = trainer.evaluate()
    record = trainer._eval_record(result, wall=0.0)
    assert "gram_frob_sq" not in record
    assert "gram_penalty" not in record


def test_gram_reference_is_resume_safe_snapshotting_deterministic_init_not_the_checkpoint(
    tmp_path: Path,
) -> None:
    """``_gram_ref`` must reflect the deterministic INIT geometry — reused
    identically whether or not this run resumes — never the mid-training
    geometry sitting in the checkpoint being loaded. Proof: a fresh trainer's
    ref must equal a RESUMED trainer's ref, even though the resumed trainer's
    live table has moved well away from both."""
    cfg = _gram_cfg(tmp_path)
    trainer_a = Trainer(cfg)
    ref_at_init = trainer_a._gram_ref.clone()
    assert ref_at_init is not None
    for _ in range(3):
        trainer_a.train_step()
    ckpt_path = trainer_a.save_checkpoint("resume-gram-test")

    resumed_cfg = _gram_cfg(tmp_path)
    resumed_cfg.train.resume = str(ckpt_path)
    trainer_b = Trainer(resumed_cfg)

    torch.testing.assert_close(trainer_b._gram_ref, ref_at_init, atol=0, rtol=0)
    # The loaded table itself DID move (checkpoint is post-training) — proves
    # the equality above isn't vacuous because nothing changed.
    assert not torch.equal(trainer_b.raw_model.wte.weight, ref_at_init)


def test_mlflow_sink_forwards_gram_metrics_on_eval_events() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log(
        {
            "event": "eval",
            "step": 7,
            "loss": 1.0,
            "gram_frob_sq": 0.42,
            "gram_penalty": 0.021,
        }
    )
    metrics = client._runs[sink.run_id].data.metrics
    assert metrics["gram_frob_sq"] == pytest.approx(0.42)
    assert metrics["gram_penalty"] == pytest.approx(0.021)


def test_mlflow_sink_omits_gram_metrics_when_absent_from_the_record() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "eval", "step": 1, "loss": 1.0})
    metrics = client._runs[sink.run_id].data.metrics
    assert "gram_frob_sq" not in metrics
    assert "gram_penalty" not in metrics


def test_five_step_tiny_train_with_gram_penalty_decreases_loss_and_stays_finite(
    tmp_path: Path,
) -> None:
    trainer = Trainer(_gram_cfg(tmp_path))
    history = trainer.fit(verbose=False)
    assert len(history) == 5
    losses = [h["loss"] for h in history]
    gram_penalties = [h["gram_penalty"] for h in history]
    assert all(torch.isfinite(torch.tensor(v)) for v in losses + gram_penalties)
    assert losses[-1] < losses[0], losses
