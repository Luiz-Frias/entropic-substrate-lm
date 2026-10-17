"""E1.5-D embedding-geometry diagnostics: angular drift, Gram retention,
cross-arm agreement, Procrustes alignment, functional divergence, and the
checkpoint-loading/init-reconstruction retrieval side.

Five guard-teeth this file pins with RED-first proofs (hand-verified during
implementation — command + observed output recorded in the phase report, not
shipped here, matching ``test_field_encoder.py``'s / ``test_gram_penalty.py``'s
convention):

1. **planted-rotation Procrustes recovery** — a table rotated by a KNOWN
   orthogonal ``Q`` must recover ``R ≈ Q`` and a near-zero residual; a
   ``procrustes_align`` that silently returned the identity (e.g. an SVD call
   dropped) would still pass a trivial "runs without raising" test but fail
   this one loudly.
2. **random-arm-seed-0 pin** — ``backbone.py``'s ``_init_embeddings`` calls
   ``random_sphere_init`` with NO seed argument, so every random-arm run
   shares ONE init table regardless of its own ``train.seed`` (CLAUDE.md
   finding #2). ``test_reconstruct_init_table_random_sphere_shares_seed_0_
   across_train_seeds`` is RED against a naive implementation that instead
   forwarded ``cfg.train.seed`` into the reconstruction.
3. **malformed-checkpoint-loud** — a checkpoint capture/load path that
   swallowed a corrupt ``.pt`` (or that re-raised the raw pickle/zip error
   unnamed) would pass silently or unhelpfully; both
   :func:`~entropic_substrate.analysis.embedding_geometry.load_checkpoint_model`
   and the ``embedding-geometry`` CLI subcommand are pinned to raise/report a
   NAMED :class:`~entropic_substrate.checkpoint.CheckpointFormatError`.
4. **field-state-source refused, not silently mismeasured** (L2 finding,
   2026-08-19) — a ``state_source="field"`` (E1.5c) checkpoint is
   table-shaped (same ``d_model``/vocab as any table-arm fisher checkpoint),
   so it passes ``_check_paired_shape`` silently; without a guard the CLI
   would read ``ckpt.model.wte.weight`` — the output classifier under
   ``state_source="field"``, not the input representation
   (``FieldEncoder.wte`` is) — and emit a plausible-looking but wrong
   diagnostic with zero error or warning.
   ``test_cli_embedding_geometry_field_state_source_is_a_loud_refusal`` is
   RED against the pre-fix CLI (verified: without the guard it exits 0 and
   prints a full JSON payload instead of refusing).
5. **init-table cache-key discrimination** (FOR-920) — the memoized
   ``reconstruct_init_table`` wrapper in ``cli.py`` keys its cache on a
   tuple derived from ``cfg``; a key that dropped a component (e.g. keyed
   on ``embedding_init`` alone) would silently SHARE a cache entry between
   two configs that could reconstruct to DIFFERENT tables.
   ``test_cached_reconstruct_init_table_key_discriminates_on_d_model`` is
   RED against a ``_init_table_cache_key`` narrowed to ``embedding_init``
   only (hand-verified during implementation — command + observed output
   recorded in the phase report).
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from entropic_substrate.composition.entrypoints.cli import (
    _cached_reconstruct_init_table,
    _init_table_cache_key,
    main,
)
from entropic_substrate.core.substrate.fisher import (
    build_fisher_init,
    random_sphere_init,
    save_init,
)
from entropic_substrate.historical.e_series.train import Trainer
from entropic_substrate.kernel.runtime.checkpoint import CheckpointFormatError
from entropic_substrate.kernel.runtime.config import Config
from entropic_substrate.ladder.analyze.embedding_geometry import (
    CheckpointModel,
    _pearson,
    angular_drift,
    cross_arm_gram_agreement,
    functional_divergence,
    gram_retention,
    load_checkpoint_model,
    procrustes_align,
    reconstruct_init_table,
)
from helpers import tiny_config

pytestmark = pytest.mark.unit


def _seeded_tiny_config(tmp_path: Path, *, seed: int, **model_overrides: object) -> Config:
    """``tiny_config`` with an explicit ``train.seed`` override.

    ``tiny_config`` has no seed parameter of its own (its default,
    ``TrainConfig``'s ``42``, is what every OTHER tiny-config test in this
    repo relies on) — E1.5-D's pairing tests need distinct, EXPLICIT seeds
    on demand, hence this small wrapper rather than widening ``tiny_config``
    itself (out of this phase's touch-scope).
    """
    cfg = tiny_config(tmp_path, **model_overrides)
    return dataclasses.replace(cfg, train=dataclasses.replace(cfg.train, seed=seed))


def _random_orthogonal(d: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(d, d, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


# --------------------------------------------------------------- no-matplotlib
def test_analysis_module_imports_no_matplotlib() -> None:
    """Pinned in a FRESH subprocess: this same test file's CLI end-to-end
    test legitimately imports matplotlib (via ``charts.py``, for the
    embedding-geometry PNG-writing path) elsewhere in this session, so an
    in-process ``sys.modules`` check here would be a false pass once that
    other test has already run."""
    script = (
        "import sys; import entropic_substrate.ladder.analyze.embedding_geometry; "
        "assert 'matplotlib' not in sys.modules, sorted("
        "m for m in sys.modules if 'matplotlib' in m)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------- load_checkpoint_model
def test_load_checkpoint_model_needs_no_fisher_table_on_disk(tmp_path: Path) -> None:
    """``embedding_init="fisher_sqrt"`` is forced to ``"default"`` for the
    pre-load build — an analysis machine holding neither the fisher table
    NOR the corpus must still load this checkpoint."""
    cfg = tiny_config(tmp_path, embedding_init="fisher_sqrt")
    trainer = Trainer(cfg)  # builds via a real corpus, gets a real fisher table
    ckpt_path = trainer.save_checkpoint("final")
    assert cfg.model.fisher_init_path is None  # nothing on disk to find

    loaded = load_checkpoint_model(ckpt_path)
    assert isinstance(loaded, CheckpointModel)
    assert loaded.step == 0  # trainer never called fit(); step is still 0
    torch.testing.assert_close(loaded.model.wte.weight, trainer.raw_model.wte.weight)
    assert not loaded.model.training  # eval mode


def test_load_checkpoint_model_malformed_bytes_raise_named_error(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a torch checkpoint at all")
    with pytest.raises(CheckpointFormatError):
        load_checkpoint_model(bad)


# ------------------------------------------------------------ reconstruct_init_table
def test_reconstruct_init_table_random_sphere_shares_seed_0_across_train_seeds(
    tmp_path: Path,
) -> None:
    """THE finding #2 pin: random-arm reconstruction must NOT vary with
    ``cfg.train.seed`` — every random-arm run drew from the SAME table."""
    cfg_a = _seeded_tiny_config(tmp_path, seed=6, embedding_init="random_sphere")
    cfg_b = _seeded_tiny_config(tmp_path, seed=7, embedding_init="random_sphere")
    table_a = reconstruct_init_table(cfg_a)
    table_b = reconstruct_init_table(cfg_b)
    torch.testing.assert_close(table_a, table_b)
    expected = random_sphere_init(256, cfg_a.model.d_model)
    torch.testing.assert_close(table_a, expected)


def test_reconstruct_init_table_fisher_sqrt_prefers_the_saved_table(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, embedding_init="fisher_sqrt")
    saved = torch.randn(256, cfg.model.d_model)
    saved = saved / saved.norm(dim=1, keepdim=True)
    path = save_init(saved, tmp_path / "fisher.pt")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, fisher_init_path=str(path)))
    reconstructed = reconstruct_init_table(cfg)
    torch.testing.assert_close(reconstructed, saved)


def test_reconstruct_init_table_fisher_sqrt_rebuilds_from_corpus_when_no_path(
    tmp_path: Path,
) -> None:
    cfg = tiny_config(tmp_path, embedding_init="fisher_sqrt")
    assert cfg.model.fisher_init_path is None
    reconstructed = reconstruct_init_table(cfg)
    from entropic_substrate.ladder.capture.custody.build import corpus_byte_stream

    stream = b"".join(corpus_byte_stream(cfg.data, 20_000_000))
    expected = build_fisher_init(stream, cfg.model.d_model)
    torch.testing.assert_close(reconstructed, expected)


def test_reconstruct_init_table_default_embedding_init_refuses(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)  # embedding_init="default"
    with pytest.raises(ValueError, match="no init table exists"):
        reconstruct_init_table(cfg)


# --------------------------------------------------------------------- angular_drift
def test_angular_drift_is_zero_when_nothing_moved() -> None:
    table = torch.randn(5, 4)
    table = table / table.norm(dim=1, keepdim=True)
    drift = angular_drift(table, table.clone())
    torch.testing.assert_close(drift, torch.zeros(5, dtype=torch.float64), atol=1e-6, rtol=0)


def test_angular_drift_recovers_a_planted_angle() -> None:
    """One row, moved by an EXACT planted angle theta: build an orthonormal
    pair (u, v) and set final = cos(theta) u + sin(theta) v — a global
    rotation's per-row angle is not uniform, so this is the clean recipe."""
    u = torch.tensor([[1.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0]])
    theta = 0.37
    final = math.cos(theta) * u + math.sin(theta) * v
    drift = angular_drift(u, final)
    assert drift.item() == pytest.approx(theta, abs=1e-6)


def test_angular_drift_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        angular_drift(torch.randn(3, 4), torch.randn(3, 5))


# --------------------------------------------------------------------- gram_retention
def test_gram_retention_identical_tables_have_spearman_one() -> None:
    """Rank correlation is invariant to the specific monotonic transform on
    each side (2*arccos vs 1-cos of THE SAME underlying cosine values), so
    identical init/final tables must produce an EXACT spearman_r == 1.0 —
    unlike pearson_r, which is transform-sensitive and only approximately 1."""
    table = torch.randn(8, 5)
    table = table / table.norm(dim=1, keepdim=True)
    retention = gram_retention(table, table.clone())
    assert retention.spearman_r == pytest.approx(1.0, abs=1e-9)
    assert retention.pearson_r > 0.99
    assert retention.n_pairs == 8 * 7 // 2


def test_gram_retention_independent_tables_have_near_zero_correlation() -> None:
    """The other arm of the identical-tables pin: two INDEPENDENTLY drawn
    tables (nothing preserved) must score near ZERO on both estimators — a
    stub that always returned ``1.0`` (or any implementation that measured
    the same underlying cosine values on both sides by mistake) would pass
    the identical-tables test above but fail this one loudly."""
    torch.manual_seed(0)
    n, d = 64, 32
    init = torch.randn(n, d)
    init = init / init.norm(dim=1, keepdim=True)
    final = torch.randn(n, d)
    final = final / final.norm(dim=1, keepdim=True)
    retention = gram_retention(init, final)
    assert abs(retention.pearson_r) < 0.3
    assert abs(retention.spearman_r) < 0.3


def test_pearson_does_not_attenuate_a_true_correlation_at_small_scale() -> None:
    """CodeRabbit finding (FOR-909 P4 follow-up): a naive ``clamp_min`` on
    the PRODUCT of the two centered norms silently attenuates a genuine
    strong correlation whenever both spreads are merely small (not
    degenerate) — two perfectly-correlated vectors at norm ~2e-7 each give
    a norm-product of ~1e-14, clamped up to the 1e-12 floor, reporting
    r ~= 0.1 instead of the true r == 1.0. Each norm must be tested
    SEPARATELY against the degeneracy floor instead."""
    x = torch.tensor([0.0, 1e-7, 2e-7, 3e-7], dtype=torch.float64)
    y = 2.0 * x  # perfectly correlated by construction
    assert _pearson(x, y) == pytest.approx(1.0, abs=1e-6)


def test_pearson_is_zero_for_a_degenerate_constant_side() -> None:
    x = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    y = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert _pearson(x, y) == 0.0


def test_gram_retention_needs_at_least_two_symbols() -> None:
    with pytest.raises(ValueError, match="at least 2 symbols"):
        gram_retention(torch.randn(1, 4), torch.randn(1, 4))


def test_gram_retention_refuses_oversized_vocab_before_materializing_gram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CodeRabbit finding (FOR-909 P4 follow-up): neither this function nor
    the CLI restricts callers to the byte alphabet, so a GPT-2-vocab table
    (~50,257 symbols) would silently materialize two ~10GB fp32 Gram
    matrices — an OOM kill, not a refusal. Refuses BEFORE calling
    normalized_gram: patches it to raise if reached, proving the size check
    runs first rather than merely coincidentally succeeding on small input."""
    import entropic_substrate.ladder.analyze.embedding_geometry as eg_module

    def _must_not_be_called(_: torch.Tensor) -> torch.Tensor:
        raise AssertionError("normalized_gram must not run past the size guard")

    monkeypatch.setattr(eg_module, "normalized_gram", _must_not_be_called)
    n = eg_module._MAX_GRAM_SYMBOLS + 1
    with pytest.raises(ValueError, match="would materialize two"):
        gram_retention(torch.randn(n, 4), torch.randn(n, 4))


# ------------------------------------------------------------ cross_arm_gram_agreement
def test_cross_arm_gram_agreement_is_zero_for_identical_tables() -> None:
    table = torch.randn(6, 4)
    assert cross_arm_gram_agreement(table, table.clone()) == pytest.approx(0.0, abs=1e-8)


def test_cross_arm_gram_agreement_is_rotation_invariant() -> None:
    """E' = E @ Q for an orthogonal Q leaves the normalized Gram matrix
    unchanged — two arms whose absolute ORIENTATION differs but relational
    geometry agrees must score identically to two bitwise-equal tables."""
    torch.manual_seed(0)
    table = torch.randn(10, 6, dtype=torch.float64)
    q = _random_orthogonal(6, seed=1)
    rotated = table @ q
    agreement = cross_arm_gram_agreement(table.float(), rotated.float())
    assert agreement == pytest.approx(0.0, abs=1e-4)


def test_cross_arm_gram_agreement_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        cross_arm_gram_agreement(torch.randn(3, 4), torch.randn(4, 4))


# ------------------------------------------------------------------- procrustes_align
def test_procrustes_align_recovers_a_planted_rotation() -> None:
    torch.manual_seed(0)
    table_a = torch.randn(12, 5, dtype=torch.float64)
    q = _random_orthogonal(5, seed=2)
    table_b = table_a @ q

    result = procrustes_align(table_a.float(), table_b.float())
    torch.testing.assert_close(result.rotation, q, atol=1e-3, rtol=0)
    assert result.residual == pytest.approx(0.0, abs=1e-3)


def test_procrustes_align_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        procrustes_align(torch.randn(3, 4), torch.randn(3, 5))


# ------------------------------------------------------------- functional_divergence
def test_functional_divergence_is_zero_against_itself(tmp_path: Path) -> None:
    cfg = _seeded_tiny_config(tmp_path, seed=6, embedding_init="random_sphere")
    trainer = Trainer(cfg)
    ckpt = load_checkpoint_model(trainer.save_checkpoint("final"))
    result = functional_divergence(ckpt, ckpt, num_batches=2)
    assert result.kl_a_to_b == pytest.approx(0.0, abs=1e-5)
    assert result.kl_b_to_a == pytest.approx(0.0, abs=1e-5)
    assert result.js == pytest.approx(0.0, abs=1e-5)
    assert result.n_positions > 0


def test_functional_divergence_is_bounded_by_ln2_and_deterministic(tmp_path: Path) -> None:
    """JS is a bounded property of JS divergence itself, not a fact about
    these particular models — it must hold for two GENUINELY different
    models (distinct ``train.seed`` — ``seed_everything`` reseeds torch's
    global RNG before every non-embedding weight is drawn, ``train.py``'s
    own ``Trainer.__init__``, so same-seed configs of otherwise-identical
    shape build BIT-IDENTICAL backbones, not merely similar ones: an
    earlier version of this test used ``d_model=32`` to vary cfg_b, but
    ``tiny_config``'s own default is ALSO ``d_model=32`` — a no-op override
    that silently made "maximally different" mean "the same model twice",
    js == 0.0 exactly, and the js > 0 guard-tooth below RED against it)."""
    cfg_a = _seeded_tiny_config(tmp_path / "a", seed=6, embedding_init="random_sphere")
    cfg_b = _seeded_tiny_config(tmp_path / "b", seed=11, embedding_init="random_sphere")
    ckpt_a = load_checkpoint_model(Trainer(cfg_a).save_checkpoint("final"))
    ckpt_b = load_checkpoint_model(Trainer(cfg_b).save_checkpoint("final"))

    first = functional_divergence(ckpt_a, ckpt_b, num_batches=3)
    second = functional_divergence(ckpt_a, ckpt_b, num_batches=3)
    assert first.js <= math.log(2.0) + 1e-6
    # The other arm of the zero-against-itself pin (test above): two
    # actually-different models must actually disagree, or an
    # always-returns-zero implementation would satisfy both tests.
    assert first.js > 0.0
    assert first.kl_a_to_b > 0.0
    assert first.kl_b_to_a > 0.0
    assert first == second  # deterministic eval-split iteration (train=False stride walk)


def test_functional_divergence_refuses_mismatched_data_config(tmp_path: Path) -> None:
    cfg_a = _seeded_tiny_config(tmp_path / "a", seed=6, embedding_init="random_sphere")
    cfg_b = tiny_config(tmp_path / "b", embedding_init="random_sphere")
    cfg_b = dataclasses.replace(
        cfg_b,
        data=dataclasses.replace(cfg_b.data, synthetic_bytes=cfg_b.data.synthetic_bytes + 1),
        train=dataclasses.replace(cfg_b.train, seed=6),
    )
    ckpt_a = load_checkpoint_model(Trainer(cfg_a).save_checkpoint("final"))
    ckpt_b = load_checkpoint_model(Trainer(cfg_b).save_checkpoint("final"))
    with pytest.raises(ValueError, match="different DataConfig"):
        functional_divergence(ckpt_a, ckpt_b)


def test_functional_divergence_refuses_mismatched_tokenizer(tmp_path: Path) -> None:
    cfg_a = _seeded_tiny_config(tmp_path / "a", seed=6, embedding_init="random_sphere")
    cfg_b = _seeded_tiny_config(
        tmp_path / "b", seed=6, embedding_init="default", tokenizer="gpt2", tie_embeddings=True
    )
    ckpt_a = load_checkpoint_model(Trainer(cfg_a).save_checkpoint("final"))
    ckpt_b = load_checkpoint_model(Trainer(cfg_b).save_checkpoint("final"))
    with pytest.raises(ValueError, match="different tokenizers"):
        functional_divergence(ckpt_a, ckpt_b)


# ------------------------------------------------------------------------- CLI
def _train_and_capture_final(tmp_path: Path, *, seed: int, **model_overrides: object) -> Path:
    cfg = _seeded_tiny_config(tmp_path, seed=seed, **model_overrides)
    trainer = Trainer(cfg)
    trainer.fit(verbose=False)
    return trainer.save_checkpoint("final")


def test_cli_embedding_geometry_end_to_end_on_two_tiny_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6, embedding_init="random_sphere")
    ckpt_b = _train_and_capture_final(
        tmp_path / "b", seed=6, embedding_init="random_sphere", block="ngpt", norm="ngpt_norm"
    )

    code = main(
        [
            "embedding-geometry",
            "--ckpt-a",
            str(ckpt_a),
            "--ckpt-b",
            str(ckpt_b),
            "--json",
            "--no-functional",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["paired_seeds"] == [6]
    assert payload["unpaired_seeds"] == []
    (entry,) = payload["results"]
    assert entry["seed"] == 6
    assert "angular_drift_mean" in entry["arm_a"]
    assert "gram_retention_spearman" in entry["arm_b"]
    assert "cross_arm_gram_agreement" in entry
    assert "procrustes_residual" in entry
    assert "functional_divergence" not in entry  # --no-functional


def test_cli_embedding_geometry_with_functional_divergence(tmp_path: Path) -> None:
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6, embedding_init="random_sphere")
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=6, embedding_init="random_sphere")
    out_dir = tmp_path / "charts"

    code = main(
        [
            "embedding-geometry",
            "--ckpt-a",
            str(ckpt_a),
            "--ckpt-b",
            str(ckpt_b),
            "--out-dir",
            str(out_dir),
            "--num-eval-batches",
            "2",
        ]
    )
    assert code == 0
    assert (out_dir / "fig_e15d_angular_drift.png").exists()
    assert (out_dir / "fig_e15d_gram_retention.png").exists()


def test_cli_embedding_geometry_reports_unpaired_seeds_never_drops_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_a_seed6 = _train_and_capture_final(tmp_path / "a6", seed=6, embedding_init="random_sphere")
    ckpt_a_seed9 = _train_and_capture_final(tmp_path / "a9", seed=9, embedding_init="random_sphere")
    ckpt_b_seed6 = _train_and_capture_final(tmp_path / "b6", seed=6, embedding_init="random_sphere")

    code = main(
        [
            "embedding-geometry",
            "--ckpt-a",
            str(ckpt_a_seed6),
            "--ckpt-a",
            str(ckpt_a_seed9),
            "--ckpt-b",
            str(ckpt_b_seed6),
            "--json",
            "--no-functional",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["paired_seeds"] == [6]
    assert payload["unpaired_seeds"] == [9]
    assert len(payload["results"]) == 1


def _assert_error_line(capsys: pytest.CaptureFixture[str], needle: str) -> None:
    """``needle`` appears on a formatted ERROR line of stderr (FOR-1649).

    Anchored on the level column of the production human format and guarded
    against loguru's "Logging error" block: a bare ``needle in err`` also
    matched the one-line ``Record was: {...}`` dump a closed sink writes.
    """
    err = capsys.readouterr().err
    assert "Logging error" not in err, err
    assert any("| ERROR    | " in ln and needle in ln for ln in err.splitlines()), err


def test_cli_embedding_geometry_requires_both_arms(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["embedding-geometry", "--ckpt-a", "somewhere.pt"])
    assert code == 2
    # Exit 2 alone does not distinguish THIS refusal from any other
    # controlled refusal in this command — pin the message so a
    # wrong-reason exit 2 fails here too.
    assert "--ckpt-a and --ckpt-b" in capsys.readouterr().err


def test_cli_embedding_geometry_zero_pairs_is_a_controlled_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6, embedding_init="random_sphere")
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=9, embedding_init="random_sphere")
    code = main(["embedding-geometry", "--ckpt-a", str(ckpt_a), "--ckpt-b", str(ckpt_b), "--json"])
    assert code == 2
    _assert_error_line(capsys, "no seed-paired checkpoints")


def test_cli_embedding_geometry_default_embedding_init_is_a_loud_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The landmine: ``tiny_config`` defaults to ``embedding_init="default"``,
    which ``reconstruct_init_table`` correctly refuses — this must surface as
    a controlled exit 2, not an unhandled traceback."""
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6)  # embedding_init="default"
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=6)
    code = main(["embedding-geometry", "--ckpt-a", str(ckpt_a), "--ckpt-b", str(ckpt_b), "--json"])
    assert code == 2
    _assert_error_line(capsys, "no init table exists")


def test_cli_embedding_geometry_field_state_source_is_a_loud_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """L2 finding, 2026-08-19 (this branch): under ``state_source="field"``
    (E1.5c) ``ckpt.model.wte.weight`` is only the output classifier, not the
    input representation the field encoder (``FieldEncoder.wte``) actually
    generates — a table-shaped checkpoint captured under
    ``configs/e1-5c-field.yaml`` passes ``_check_paired_shape`` silently
    (same ``d_model``, same vocab), so without this guard the CLI would emit
    a plausible-looking but wrong ``angular_drift``/``gram_retention``/
    ``cross_arm_gram_agreement``/``procrustes_align`` reading with zero
    error or warning. Mirrors ``backbone.py``'s own ``gram_lambda``/
    ``embedding_freeze`` guards for this exact axis (~109-131, ~223-237)."""
    ckpt_a = _train_and_capture_final(
        tmp_path / "a",
        seed=6,
        embedding_init="fisher_sqrt",
        state_source="field",
        field_d_model=16,
        field_n_layers=1,
        field_n_heads=2,
    )
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=6, embedding_init="fisher_sqrt")
    code = main(["embedding-geometry", "--ckpt-a", str(ckpt_a), "--ckpt-b", str(ckpt_b), "--json"])
    assert code == 2
    _assert_error_line(capsys, "state_source")


def test_cli_embedding_geometry_duplicate_seed_on_one_side_is_a_loud_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt_a1 = _train_and_capture_final(tmp_path / "a1", seed=6, embedding_init="random_sphere")
    ckpt_a2 = _train_and_capture_final(tmp_path / "a2", seed=6, embedding_init="random_sphere")
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=6, embedding_init="random_sphere")
    code = main(
        [
            "embedding-geometry",
            "--ckpt-a",
            str(ckpt_a1),
            "--ckpt-a",
            str(ckpt_a2),
            "--ckpt-b",
            str(ckpt_b),
            "--json",
        ]
    )
    assert code == 2
    _assert_error_line(capsys, "ambiguous pairing")


def test_cli_embedding_geometry_malformed_checkpoint_is_a_loud_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a torch checkpoint at all")
    good = _train_and_capture_final(tmp_path / "b", seed=6, embedding_init="random_sphere")
    code = main(["embedding-geometry", "--ckpt-a", str(bad), "--ckpt-b", str(good), "--json"])
    assert code == 2
    # Raised as CheckpointFormatError (not ValueError), so it is caught by
    # main()'s OUTER handler, not cmd_embedding_geometry's own ValueError
    # catch — the message still names the offending path.
    _assert_error_line(capsys, "garbage.pt")


# ------------------------------------------------ init-table cache (FOR-920)
#
# cmd_embedding_geometry reconstructs each checkpoint's init embedding table
# via reconstruct_init_table, and a multi-seed sweep pairs N seeds of the
# SAME arm against each other — every one of those seeds' configs shares an
# init-table identity (embedding_init/d_model/tokenizer/vocab_size/
# fisher_init_path/data) that does NOT vary with train.seed (that's exactly
# reconstruct_init_table's own random-arm seed-0-sharing guarantee, guard-
# tooth #2 above). Reconstructing per-seed therefore rebuilt the IDENTICAL
# table redundantly: measured 2026-08-19, 9 fisher-arm rebuilds x ~7.5 min
# each (the 20MB corpus-stream fallback) = ~68 min duplicated work in one
# invocation.
#
# These tests exercise the memoizing wrapper directly (`_cached_
# reconstruct_init_table` / `_init_table_cache_key` in cli.py) against bare
# Config objects with a monkeypatched stand-in for the expensive
# `reconstruct_init_table` call — the property under test is config-key
# identity, not the reconstruction math itself (already covered by the
# `reconstruct_init_table` tests above), so a real Trainer/corpus round trip
# would only add cost without adding coverage.
def _monkeypatch_reconstruct(monkeypatch: pytest.MonkeyPatch, calls: list[Config]) -> None:
    """Replace the real ``reconstruct_init_table`` with a counting stub that
    returns a config-shaped-but-otherwise-arbitrary tensor, so these tests
    never pay for a real corpus stream or fisher build."""

    def _fake(cfg: Config) -> torch.Tensor:
        calls.append(cfg)
        return torch.full((4, cfg.model.d_model), float(len(calls)))

    monkeypatch.setattr(
        "entropic_substrate.ladder.analyze.embedding_geometry.reconstruct_init_table", _fake
    )


def test_cached_reconstruct_init_table_reconstructs_identical_configs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two DISTINCT ``Config`` objects (different ``train.seed``, otherwise
    identical — exactly what two seeds of the SAME arm look like in a
    multi-seed sweep) must trigger exactly ONE underlying reconstruction,
    and both callers must see BITWISE-IDENTICAL tables."""
    calls: list[Config] = []
    _monkeypatch_reconstruct(monkeypatch, calls)

    cfg_seed6 = _seeded_tiny_config(tmp_path, seed=6, embedding_init="fisher_sqrt")
    cfg_seed9 = _seeded_tiny_config(tmp_path, seed=9, embedding_init="fisher_sqrt")
    assert cfg_seed6 != cfg_seed9  # sanity: genuinely two distinct Config objects

    cache: dict[tuple[object, ...], torch.Tensor] = {}
    table_a = _cached_reconstruct_init_table(cache, cfg_seed6)
    table_b = _cached_reconstruct_init_table(cache, cfg_seed9)

    assert len(calls) == 1  # ONE underlying reconstruction, not two
    assert torch.equal(table_a, table_b)  # bitwise-identical, not merely close


def test_cached_reconstruct_init_table_key_discriminates_on_embedding_init(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configs differing ONLY on ``embedding_init`` must NOT share a cache
    entry — they can reconstruct to entirely different tables."""
    calls: list[Config] = []
    _monkeypatch_reconstruct(monkeypatch, calls)

    cfg_fisher = _seeded_tiny_config(tmp_path, seed=6, embedding_init="fisher_sqrt")
    cfg_random = _seeded_tiny_config(tmp_path, seed=6, embedding_init="random_sphere")

    cache: dict[tuple[object, ...], torch.Tensor] = {}
    _cached_reconstruct_init_table(cache, cfg_fisher)
    _cached_reconstruct_init_table(cache, cfg_random)

    assert len(calls) == 2  # separate reconstructions, never shared


def test_cached_reconstruct_init_table_key_discriminates_on_d_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configs differing ONLY on ``d_model`` must NOT share a cache entry —
    the reconstructed table's shape (and, for fisher_sqrt, its values)
    depends on ``d_model``."""
    calls: list[Config] = []
    _monkeypatch_reconstruct(monkeypatch, calls)

    cfg_small = _seeded_tiny_config(tmp_path, seed=6, embedding_init="random_sphere", d_model=16)
    cfg_large = _seeded_tiny_config(tmp_path, seed=6, embedding_init="random_sphere", d_model=32)

    cache: dict[tuple[object, ...], torch.Tensor] = {}
    _cached_reconstruct_init_table(cache, cfg_small)
    _cached_reconstruct_init_table(cache, cfg_large)

    assert len(calls) == 2  # separate reconstructions, never shared


def test_cached_reconstruct_init_table_protects_against_caller_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller that mutates ITS OWN returned table in place must never
    corrupt the cache entry (or a later caller's view of it)."""
    calls: list[Config] = []
    _monkeypatch_reconstruct(monkeypatch, calls)

    cfg = _seeded_tiny_config(tmp_path, seed=6, embedding_init="fisher_sqrt")
    cache: dict[tuple[object, ...], torch.Tensor] = {}

    first = _cached_reconstruct_init_table(cache, cfg)
    expected = first.clone()
    first += 999.0  # mutate the CALLER's own returned tensor in place

    second = _cached_reconstruct_init_table(cache, cfg)
    assert len(calls) == 1  # still only one underlying reconstruction
    assert torch.equal(second, expected)  # unaffected by `first`'s mutation
    assert not torch.equal(second, first)  # the mutation really did happen


def test_init_table_cache_key_differs_on_fisher_init_path(tmp_path: Path) -> None:
    """A saved ``fisher_init_path`` selects an entirely different
    reconstruction branch (load-from-disk vs. corpus-stream rebuild) than
    ``None`` — the two must never share a cache entry."""
    cfg_no_path = _seeded_tiny_config(tmp_path, seed=6, embedding_init="fisher_sqrt")
    cfg_with_path = dataclasses.replace(
        cfg_no_path,
        model=dataclasses.replace(cfg_no_path.model, fisher_init_path=str(tmp_path / "f.pt")),
    )
    assert _init_table_cache_key(cfg_no_path) != _init_table_cache_key(cfg_with_path)


def test_init_table_cache_key_ignores_train_seed(tmp_path: Path) -> None:
    """The key must be blind to ``train.seed`` — that is exactly what makes
    same-arm-different-seed configs collapse to one cache entry."""
    cfg_seed6 = _seeded_tiny_config(tmp_path, seed=6, embedding_init="fisher_sqrt")
    cfg_seed9 = _seeded_tiny_config(tmp_path, seed=9, embedding_init="fisher_sqrt")
    assert _init_table_cache_key(cfg_seed6) == _init_table_cache_key(cfg_seed9)


def test_cli_embedding_geometry_reconstructs_shared_arm_configs_once_across_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end wiring proof: TWO seeds of the SAME two arms (fisher_sqrt
    with no saved path -> corpus fallback; random_sphere -> seed-0-shared by
    construction — the two cases FOR-920 names by name) through the REAL
    CLI must reconstruct each arm's table exactly ONCE, not once per seed."""
    ckpt_a6 = _train_and_capture_final(tmp_path / "a6", seed=6, embedding_init="fisher_sqrt")
    ckpt_a9 = _train_and_capture_final(tmp_path / "a9", seed=9, embedding_init="fisher_sqrt")
    ckpt_b6 = _train_and_capture_final(tmp_path / "b6", seed=6, embedding_init="random_sphere")
    ckpt_b9 = _train_and_capture_final(tmp_path / "b9", seed=9, embedding_init="random_sphere")

    calls: list[Config] = []
    real_reconstruct = reconstruct_init_table

    def _counting(cfg: Config) -> torch.Tensor:
        calls.append(cfg)
        return real_reconstruct(cfg)

    monkeypatch.setattr(
        "entropic_substrate.ladder.analyze.embedding_geometry.reconstruct_init_table", _counting
    )

    code = main(
        [
            "embedding-geometry",
            "--ckpt-a",
            str(ckpt_a6),
            "--ckpt-a",
            str(ckpt_a9),
            "--ckpt-b",
            str(ckpt_b6),
            "--ckpt-b",
            str(ckpt_b9),
            "--json",
            "--no-functional",
        ]
    )
    assert code == 0
    # MEASURED 2026-08-19 (this test, temporarily reverting the two wired
    # call sites in cmd_embedding_geometry back to bare
    # eg.reconstruct_init_table(...) calls): 4 — one call per seed per side,
    # with no memoization. With _cached_reconstruct_init_table wired in, the
    # same scenario measures exactly one reconstruction per DISTINCT
    # arm-config identity.
    # (angular_drift compares init against each seed's own FINAL trained
    # table, which legitimately differs by seed — so drift readings are
    # NOT expected to match across seeds even with a shared init table;
    # bitwise-identity of the init table itself is pinned directly by
    # test_cached_reconstruct_init_table_reconstructs_identical_configs_once
    # above, this test's job is only to prove the real CLI path is wired to
    # the memoizing wrapper at all.)
    assert len(calls) == 2

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["paired_seeds"] == [6, 9]
    assert {entry["seed"] for entry in payload["results"]} == {6, 9}


# --------------------------------------------------------------- fig_e15d_* charts
def test_fig_e15d_angular_drift_renders_with_redundant_non_color_channels() -> None:
    from entropic_substrate.ladder.analyze.charts import fig_e15d_angular_drift

    series = [
        # Deliberately UNSORTED (descending): fig_e15d_angular_drift's own
        # docstring promises it sorts ascending so the DISTRIBUTION shape,
        # not symbol identity, is compared — sorted-ascending input (the
        # prior version of this test) cannot distinguish "sorts correctly"
        # from "torch.sort silently dropped, input happened to be sorted
        # already".
        ("arm-a", torch.linspace(0.0, 0.5, 20, dtype=torch.float64).flip(0)),
        ("arm-b", torch.linspace(0.0, 0.9, 20, dtype=torch.float64).flip(0)),
    ]
    fig = fig_e15d_angular_drift(series)
    (ax,) = fig.get_axes()
    lines = [ln for ln in ax.get_lines() if ln.get_label() in {"arm-a", "arm-b"}]
    assert len(lines) == 2
    for line in lines:
        ys = list(line.get_ydata())
        assert ys == sorted(ys)
        # x is symbol RANK normalized to [0, 1] (least- to most-drifted),
        # not a raw index.
        xs = list(line.get_xdata())
        assert xs[0] == pytest.approx(0.0)
        assert xs[-1] == pytest.approx(1.0)
    assert len({ln.get_color() for ln in lines}) == 2
    # Colorblind rule: color alone is never the only channel telling two
    # series apart — linestyle AND marker must both actually differ, not
    # merely both technically be set (see test_analysis_charts.py's same
    # convention for fig_e3_kappa_trajectories).
    assert len({ln.get_linestyle() for ln in lines}) >= 2
    assert len({ln.get_marker() for ln in lines}) >= 2


def test_fig_e15d_angular_drift_refuses_empty_series() -> None:
    from entropic_substrate.ladder.analyze.charts import fig_e15d_angular_drift

    with pytest.raises(ValueError, match="no drift series given"):
        fig_e15d_angular_drift([])


def test_fig_e15d_gram_retention_renders_with_redundant_non_color_channels() -> None:
    from entropic_substrate.ladder.analyze.charts import fig_e15d_gram_retention
    from entropic_substrate.ladder.analyze.embedding_geometry import GramRetention

    points = [
        ("arm-a", GramRetention(pearson_r=0.9, spearman_r=0.95, n_pairs=10)),
        ("arm-b", GramRetention(pearson_r=0.2, spearman_r=0.3, n_pairs=10)),
    ]
    fig = fig_e15d_gram_retention(points)
    (ax,) = fig.get_axes()
    lines = [ln for ln in ax.get_lines() if ln.get_label() not in (None, "")]
    markers = [ln for ln in lines if not str(ln.get_label()).startswith("_")]
    reference = [ln for ln in lines if str(ln.get_label()).startswith("_")]
    assert len(markers) == 2
    assert len({ln.get_color() for ln in markers}) == 2
    assert len({ln.get_marker() for ln in markers}) >= 2
    # The y=x reference line IS drawn (not merely implied by axis symmetry)
    # — CLAUDE.md's own rule that a threshold "computed but hidden" is
    # forbidden — and both axes are the documented fixed [-1.05, 1.05] range.
    assert len(reference) == 1
    assert ax.get_xlim() == pytest.approx((-1.05, 1.05))
    assert ax.get_ylim() == pytest.approx((-1.05, 1.05))


def test_fig_e15d_gram_retention_refuses_empty_points() -> None:
    from entropic_substrate.ladder.analyze.charts import fig_e15d_gram_retention

    with pytest.raises(ValueError, match="no retention points given"):
        fig_e15d_gram_retention([])
