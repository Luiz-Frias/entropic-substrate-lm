"""E1.5-D embedding-geometry diagnostics: angular drift, Gram retention,
cross-arm agreement, Procrustes alignment, functional divergence, and the
checkpoint-loading/init-reconstruction retrieval side.

Three guard-teeth this file pins with RED-first proofs (hand-verified during
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

from entropic_substrate.analysis.embedding_geometry import (
    CheckpointModel,
    angular_drift,
    cross_arm_gram_agreement,
    functional_divergence,
    gram_retention,
    load_checkpoint_model,
    procrustes_align,
    reconstruct_init_table,
)
from entropic_substrate.checkpoint import CheckpointFormatError
from entropic_substrate.cli import main
from entropic_substrate.config import Config
from entropic_substrate.init.fisher import build_fisher_init, random_sphere_init, save_init
from entropic_substrate.train import Trainer
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
        "import sys; import entropic_substrate.analysis.embedding_geometry; "
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
    from entropic_substrate.data.build import corpus_byte_stream

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


def test_gram_retention_needs_at_least_two_symbols() -> None:
    with pytest.raises(ValueError, match="at least 2 symbols"):
        gram_retention(torch.randn(1, 4), torch.randn(1, 4))


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
    these particular (independently-initialized, hence maximally-different)
    models — it must hold even for two models that agree nowhere."""
    cfg_a = _seeded_tiny_config(tmp_path / "a", seed=6, embedding_init="random_sphere")
    cfg_b = _seeded_tiny_config(tmp_path / "b", seed=6, embedding_init="random_sphere", d_model=32)
    ckpt_a = load_checkpoint_model(Trainer(cfg_a).save_checkpoint("final"))
    ckpt_b = load_checkpoint_model(Trainer(cfg_b).save_checkpoint("final"))

    first = functional_divergence(ckpt_a, ckpt_b, num_batches=3)
    second = functional_divergence(ckpt_a, ckpt_b, num_batches=3)
    assert first.js <= math.log(2.0) + 1e-6
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


def test_cli_embedding_geometry_requires_both_arms(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["embedding-geometry", "--ckpt-a", "somewhere.pt"])
    assert code == 2


def test_cli_embedding_geometry_zero_pairs_is_a_controlled_refusal(tmp_path: Path) -> None:
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6, embedding_init="random_sphere")
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=9, embedding_init="random_sphere")
    code = main(["embedding-geometry", "--ckpt-a", str(ckpt_a), "--ckpt-b", str(ckpt_b), "--json"])
    assert code == 2


def test_cli_embedding_geometry_default_embedding_init_is_a_loud_refusal(tmp_path: Path) -> None:
    """The landmine: ``tiny_config`` defaults to ``embedding_init="default"``,
    which ``reconstruct_init_table`` correctly refuses — this must surface as
    a controlled exit 2, not an unhandled traceback."""
    ckpt_a = _train_and_capture_final(tmp_path / "a", seed=6)  # embedding_init="default"
    ckpt_b = _train_and_capture_final(tmp_path / "b", seed=6)
    code = main(["embedding-geometry", "--ckpt-a", str(ckpt_a), "--ckpt-b", str(ckpt_b), "--json"])
    assert code == 2


def test_cli_embedding_geometry_duplicate_seed_on_one_side_is_a_loud_refusal(
    tmp_path: Path,
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


def test_cli_embedding_geometry_malformed_checkpoint_is_a_loud_refusal(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.pt"
    bad.write_bytes(b"not a torch checkpoint at all")
    good = _train_and_capture_final(tmp_path / "b", seed=6, embedding_init="random_sphere")
    code = main(["embedding-geometry", "--ckpt-a", str(bad), "--ckpt-b", str(good), "--json"])
    assert code == 2


# --------------------------------------------------------------- fig_e15d_* charts
def test_fig_e15d_angular_drift_renders_with_redundant_non_color_channels() -> None:
    from entropic_substrate.analysis.charts import fig_e15d_angular_drift

    series = [
        ("arm-a", torch.linspace(0.0, 0.5, 20, dtype=torch.float64)),
        ("arm-b", torch.linspace(0.0, 0.9, 20, dtype=torch.float64)),
    ]
    fig = fig_e15d_angular_drift(series)
    (ax,) = fig.get_axes()
    lines = [ln for ln in ax.get_lines() if ln.get_label() in {"arm-a", "arm-b"}]
    assert len(lines) == 2
    assert len({ln.get_color() for ln in lines}) == 2
    # Colorblind rule: color alone is never the only channel telling two
    # series apart — linestyle AND marker must both actually differ, not
    # merely both technically be set (see test_analysis_charts.py's same
    # convention for fig_e3_kappa_trajectories).
    assert len({ln.get_linestyle() for ln in lines}) >= 2
    assert len({ln.get_marker() for ln in lines}) >= 2


def test_fig_e15d_angular_drift_refuses_empty_series() -> None:
    from entropic_substrate.analysis.charts import fig_e15d_angular_drift

    with pytest.raises(ValueError, match="no drift series given"):
        fig_e15d_angular_drift([])


def test_fig_e15d_gram_retention_renders_with_redundant_non_color_channels() -> None:
    from entropic_substrate.analysis.charts import fig_e15d_gram_retention
    from entropic_substrate.analysis.embedding_geometry import GramRetention

    points = [
        ("arm-a", GramRetention(pearson_r=0.9, spearman_r=0.95, n_pairs=10)),
        ("arm-b", GramRetention(pearson_r=0.2, spearman_r=0.3, n_pairs=10)),
    ]
    fig = fig_e15d_gram_retention(points)
    (ax,) = fig.get_axes()
    lines = [ln for ln in ax.get_lines() if ln.get_label() not in (None, "")]
    markers = [ln for ln in lines if not str(ln.get_label()).startswith("_")]
    assert len(markers) == 2
    assert len({ln.get_color() for ln in markers}) == 2
    assert len({ln.get_marker() for ln in markers}) >= 2


def test_fig_e15d_gram_retention_refuses_empty_points() -> None:
    from entropic_substrate.analysis.charts import fig_e15d_gram_retention

    with pytest.raises(ValueError, match="no retention points given"):
        fig_e15d_gram_retention([])
