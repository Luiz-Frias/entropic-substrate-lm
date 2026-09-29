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

from entropic_substrate.core.geometry.gram import (
    _clamp_cosine,
    gram_frobenius_sq,
    normalized_gram,
)
from entropic_substrate.historical.e_series.train import Trainer
from entropic_substrate.kernel.runtime.config import Config
from entropic_substrate.ladder.capture.reader.sinks import MlflowSink
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


def test_normalized_gram_suppresses_a_near_zero_row_instead_of_amplifying_noise() -> None:
    """Pins the _NORM_EPS floor's documented behavior (geometry/gram.py):
    a row whose norm is far below the floor gets its cosine values
    suppressed toward zero (correctly signaling "no real direction here"),
    not blown up into a spurious unit-norm direction manufactured from fp32
    rounding noise. Also checks the degenerate row's gradient stays finite,
    since a divide this close to its floor is exactly where a gradient
    could blow up if the floor were chosen wrong."""
    torch.manual_seed(0)
    e = torch.randn(4, 5, requires_grad=True)
    with torch.no_grad():
        e[0] = torch.randn(5) * 1e-20  # degenerate: norm far below _NORM_EPS
    gram = normalized_gram(e)
    assert torch.isfinite(gram).all()
    # The degenerate row's cosine with every OTHER row (including itself)
    # is suppressed near zero, not a fabricated unit-magnitude value.
    assert torch.all(gram[0].abs() < 1e-3)
    assert torch.all(gram[:, 0].abs() < 1e-3)
    # The three normal rows are unaffected: still unit self-cosine.
    torch.testing.assert_close(gram.diagonal()[1:], torch.ones(3), atol=1e-4, rtol=0)
    gram.sum().backward()
    assert e.grad is not None and torch.isfinite(e.grad).all()


def test_clamp_cosine_bounds_out_of_range_cosines_deterministically() -> None:
    """The deterministic complement to the backend-dependent seed search
    below: feeds the extracted clamp values just outside ``[-1, 1]``
    directly, so this arm cannot vary with BLAS summation order. The seed
    search is RETAINED (conductor ruling, 2026-08-19): it is the only test
    that MEASURES the overshoot hazard actually occurring on the running
    backend; this test pins that the clamp clamps."""
    overshoot = torch.tensor([1.0 + 1e-6, -1.0 - 1e-6, 0.5])
    clamped = _clamp_cosine(overshoot)
    assert clamped[0].item() == 1.0
    assert clamped[1].item() == -1.0
    assert clamped[2].item() == 0.5
    assert torch.arccos(clamped).isfinite().all()


def test_normalized_gram_clamp_is_load_bearing_against_fp32_overshoot() -> None:
    """The ``ones(4, 3)`` case above never actually overshoots 1.0 in fp32
    (its raw, pre-clamp diagonal lands at 0.9999999403953552 on this
    machine's CPU BLAS backend — UNDER 1.0), so it cannot tell a clamped
    output from an unclamped one and removing ``.clamp(-1.0, 1.0)`` leaves
    it green.

    fp32 dot-product overshoot is a real but BLAS-backend/summation-order
    -dependent effect (MEASURED here, darwin/Accelerate,
    ``torch.manual_seed(0); torch.randn(16, 64)``: raw pre-clamp max
    1.0000003576278687), not something a single pinned seed can promise on
    every platform (a different BLAS on CI's linux runners may sum in a
    different order). So this test SEARCHES a small, deterministic seed
    range for a fixture that reproduces the overshoot on whatever backend is
    running it, recomputing ``normalized_gram``'s pre-clamp arithmetic by
    hand each time, then checks the real function's output has been clamped
    away from the overshoot it found.
    """
    raw_max = None
    seed = None
    for candidate_seed in range(64):
        torch.manual_seed(candidate_seed)
        e = torch.randn(16, 64)
        unit = e / e.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        raw = unit @ unit.T
        if raw.max().item() > 1.0:
            raw_max = raw.max().item()
            seed = candidate_seed
            break
    assert seed is not None, (
        "no seed in range(64) reproduced an fp32 dot-product overshoot on this "
        "BLAS backend — widen the search range or pick a different shape"
    )

    torch.manual_seed(seed)
    e = torch.randn(16, 64)
    gram = normalized_gram(e)
    assert gram.max().item() <= 1.0, f"raw overshoot was {raw_max} at seed {seed}"
    assert torch.all(gram <= 1.0) and torch.all(gram >= -1.0)


def test_normalized_gram_ignores_row_scale() -> None:
    """Relational geometry, not absolute scale: rescaling rows independently
    must not move the Gram matrix — this is the property row-normalization
    ALONE (without the outer product) does not give a caller for free."""
    e = torch.randn(5, 4)
    scaled = e * torch.tensor([1.0, 2.0, 0.5, 10.0, 3.0]).unsqueeze(1)
    torch.testing.assert_close(normalized_gram(e), normalized_gram(scaled), atol=1e-5, rtol=1e-5)


def test_normalized_gram_is_invariant_to_a_genuine_orthogonal_rotation() -> None:
    """The module docstring claims invariance to ``any orthogonal transform
    of the table`` — a GENUINE rotation (``e @ Q`` for orthogonal ``Q``,
    which moves every row's coordinates but preserves all pairwise angles),
    not the additive perturbation the neighboring ``gram_frobenius_sq``
    test below exercises (that one moves the angles on purpose, hence its
    positive-distance assertion). This is the direct proof of the claim."""
    torch.manual_seed(0)
    e = torch.randn(8, 6)
    q, _ = torch.linalg.qr(torch.randn(6, 6))  # random orthogonal matrix
    torch.testing.assert_close(normalized_gram(e), normalized_gram(e @ q), atol=1e-4, rtol=1e-4)


def test_gram_frobenius_sq_is_zero_for_identical_gram_matrices() -> None:
    e = torch.randn(8, 6)
    gram = normalized_gram(e)
    assert float(gram_frobenius_sq(gram, gram)) == 0.0


def test_gram_frobenius_sq_is_positive_for_a_perturbed_table() -> None:
    """Additive per-row noise (unlike the rotation above) DOES move the
    pairwise angles, so the Gram matrix — and this penalty term — must
    register it as nonzero."""
    torch.manual_seed(0)
    e = torch.randn(8, 6)
    gram_a = normalized_gram(e)
    gram_b = normalized_gram(e + torch.randn(8, 6) * 0.5)
    assert float(gram_frobenius_sq(gram_a, gram_b)) > 0.0


def test_gram_frobenius_sq_rejects_mismatched_shapes() -> None:
    """Both callers in this repo (``Trainer._gram_penalty_terms`` and
    E1.5-D's planned ``cross_arm_gram_agreement``) always diff two Gram
    matrices, each ``(n_symbols, n_symbols)`` for the SAME table's own
    symbol count — a shape mismatch can only mean two tables of different
    vocabularies were compared (e.g. byte vs bpe), which is a caller bug,
    not a case ``@`` broadcasting should silently paper over. Fail fast
    (CLAUDE.md Errors principle) rather than let a mismatched pair
    silently broadcast into a nonsense value."""
    gram_a = normalized_gram(torch.randn(8, 6))
    gram_b = normalized_gram(torch.randn(5, 6))
    with pytest.raises(ValueError, match="shape"):
        gram_frobenius_sq(gram_a, gram_b)


def test_normalized_gram_rejects_non_2d_input() -> None:
    """``embedding`` is documented as ``(n_symbols, d)``; a 1-D or 3-D input
    reaching ``e.norm(p=2, dim=1, ...)`` would either error confusingly
    (1-D: no dim 1) or silently normalize per-slice instead of per-row
    (3-D), so fail fast on the actual contract instead (CLAUDE.md Errors
    principle) rather than let either happen."""
    with pytest.raises(ValueError, match="2-D"):
        normalized_gram(torch.randn(6))
    with pytest.raises(ValueError, match="2-D"):
        normalized_gram(torch.randn(2, 6, 5))


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
def _gram_cfg(tmp_path: Path, gram_lambda: float = 0.5, **overrides: object) -> Config:
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
    # Finite BEFORE nonzero: `nan != 0` is True, so a NaN-poisoned gradient
    # would otherwise slip past `torch.any(grad != 0)` and read as "reached
    # the embedding" — MEASURED: torch.any(torch.tensor([float("nan")]) !=
    # 0) is True.
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0)


def test_gram_penalty_is_included_in_the_backward_total_not_reported_loss(
    tmp_path: Path,
) -> None:
    """``total`` (used for backward) carries the penalty; the reported
    ``loss`` metric stays pure cross-entropy — same contract ``aux_loss``
    already has (train.py's ``loss = float(out.loss.detach())`` line, which
    reads the model's OWN loss, never the locally-accumulated ``total``).

    The reported-loss arm is pinned by a λ-counterfactual: two identically
    seeded trainers (``gram_lambda`` on vs 0) stepping on the SAME frozen
    batch under the SAME RNG state must report the IDENTICAL ``loss`` —
    a regression folding the penalty into the reported value shifts the
    λ arm by ``gram_penalty`` (>> the tolerance here) and fails."""
    trainer_on = Trainer(_gram_cfg(tmp_path))
    trainer_off = Trainer(_gram_cfg(tmp_path, gram_lambda=0.0))
    for trainer in (trainer_on, trainer_off):
        torch.manual_seed(1234)
        with torch.no_grad():
            trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)

    fixed_batch = trainer_on.train_sampler.batch()
    trainer_on.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]
    trainer_off.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]

    torch.manual_seed(7)
    metrics_on = trainer_on.train_step()
    torch.manual_seed(7)
    metrics_off = trainer_off.train_step()

    assert metrics_on["gram_frob_sq"] > 0.0
    assert metrics_on["gram_penalty"] == pytest.approx(
        metrics_on["gram_frob_sq"] * trainer_on.cfg.model.gram_lambda
    )
    # Discriminating power guard: the penalty must dwarf the loss tolerance,
    # else the equality below could not catch a fold-in regression.
    assert metrics_on["gram_penalty"] > 1e-6
    assert metrics_on["loss"] == pytest.approx(metrics_off["loss"], abs=1e-9)


def test_gram_penalty_gradient_reaches_the_backward_total_seam(tmp_path: Path) -> None:
    """Guard-tooth for the ACTUAL ``total = total + gram_penalty`` seam in
    ``train_step`` — not just the reported ``gram_frob_sq``/``gram_penalty``
    metrics. Those metrics are computed independently of whether they are
    folded into ``total`` (``_gram_penalty_terms()`` is called either way),
    so a regression that computes the penalty but never adds it to ``total``
    leaves every OTHER test in this module green — that gap is exactly what
    this test closes.

    Captures the REAL gradient reaching ``wte.weight`` during a live
    ``train_step()`` call via a backward hook, then compares it against a
    gradient reconstructed with the gram term deliberately excluded from the
    total — same starting weights (full state_dict snapshot/restore, not
    just the embedding table) and the same input batch (the sampler is
    frozen to return one fixed draw). If the two gradients match, the seam
    is a no-op.
    """
    trainer = Trainer(_gram_cfg(tmp_path, gram_lambda=5.0))
    with torch.no_grad():
        trainer.raw_model.wte.weight.add_(torch.randn_like(trainer.raw_model.wte.weight) * 0.1)

    # Freeze the batch train_step will draw so the real run and the manual
    # "no-penalty" reconstruction below see identical inputs.
    fixed_batch = trainer.train_sampler.batch()
    trainer.train_sampler.batch = lambda: fixed_batch  # type: ignore[method-assign]

    state_before = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}

    captured: dict[str, torch.Tensor] = {}

    def _capture(grad: torch.Tensor) -> None:
        captured["grad"] = grad.detach().clone()

    handle = trainer.raw_model.wte.weight.register_hook(_capture)
    trainer.train_step()
    handle.remove()
    assert "grad" in captured, "backward hook never fired — train_step did not call .backward()"

    # Reconstruct the gradient WITHOUT the gram term, from the identical
    # pre-step weights (every param, not just wte) and the identical batch.
    trainer.model.load_state_dict(state_before)
    x, y = fixed_batch
    with trainer._autocast():
        out = trainer.model(x, y)
    aux = out.aux_loss
    total_without_penalty = out.loss if aux is None else out.loss + aux
    (grad_without_penalty,) = torch.autograd.grad(
        total_without_penalty, trainer.raw_model.wte.weight
    )

    # Finite BEFORE the difference check: `torch.allclose` with the default
    # equal_nan=False already returns False when either side has a NaN, so a
    # NaN-poisoned `captured["grad"]` would otherwise slip past `not
    # torch.allclose(...)` and read as "the seam changed the gradient"
    # instead of "the seam broke the gradient."
    assert torch.isfinite(captured["grad"]).all()
    assert torch.isfinite(grad_without_penalty).all()
    assert not torch.allclose(captured["grad"], grad_without_penalty), (
        "gradient reaching wte.weight during train_step matches the gram-free "
        "reconstruction exactly — gram_penalty is not being folded into `total`"
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
    # Assert on the SOURCE before touching it: `.clone()` on a None would
    # die as AttributeError, and a clone can never be None afterwards.
    assert trainer_a._gram_ref is not None
    ref_at_init = trainer_a._gram_ref.clone()
    for _ in range(3):
        trainer_a.train_step()
    ckpt_path = trainer_a.save_checkpoint("resume-gram-test")

    resumed_cfg = _gram_cfg(tmp_path)
    resumed_cfg.train.resume = str(ckpt_path)
    trainer_b = Trainer(resumed_cfg)

    torch.testing.assert_close(trainer_b._gram_ref, ref_at_init, atol=0, rtol=0)
    # The loaded table's OWN geometry DID move (checkpoint is post-training)
    # — proves the equality above isn't vacuous because nothing changed.
    # Compared via normalized_gram, not raw weights: `ref_at_init` is
    # (n_symbols, n_symbols) while `wte.weight` is (n_symbols, d_model), so a
    # raw-weight `torch.equal` against `ref_at_init` compares mismatched
    # shapes whenever d_model != n_symbols (true here: tiny_config's
    # d_model=32 vs the 256-byte vocab) — MEASURED: torch.equal on
    # mismatched shapes returns False unconditionally rather than raising,
    # which made the original form of this assertion pass regardless of
    # whether resume actually worked.
    assert not torch.allclose(normalized_gram(trainer_b.raw_model.wte.weight), ref_at_init)


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
    sink.flush()
    metrics = client._runs[sink.run_id].data.metrics
    assert metrics["gram_frob_sq"] == pytest.approx(0.42)
    assert metrics["gram_penalty"] == pytest.approx(0.021)


def test_mlflow_sink_omits_gram_metrics_when_absent_from_the_record() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "eval", "step": 1, "loss": 1.0})
    sink.flush()
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
