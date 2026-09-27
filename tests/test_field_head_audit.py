"""Exploratory E1.5c head audit: rank/conditioning + reconstruction of the frozen head.

The verdict's causal ladder (docs/verdicts/e1-5c-frozen-verdict.md §3) measures
P−H at -0.0508 nats/byte. Before that gap is attributed to *compression* — the
128-dim field hidden state squeezing through the 256-way head — the smallest
honest diagnostic is: (a) the spectrum/conditioning of the frozen head, and
(b) whether ``h`` is recoverable in practice from the probabilities the P arm
actually consumed. ``entropic_substrate.analysis.field_head_audit`` is that
diagnostic; this file pins its contract RED-first.

Three hazards these tests are specifically toothed against:

1. **wrong-matrix hazard** — softmax is shift-invariant (``softmax(l) ==
   softmax(l + c*1)``), so injectivity of ``h -> softmax(W @ h)`` is governed
   by the ROW-CENTERED head ``(I - 11^T/256) @ W``, not raw ``W``. An audit
   that SVDs raw ``W`` alone can report full rank while the softmax map has a
   one-dimensional kernel. ``test_head_spectrum_detects_ones_direction_rank_drop``
   constructs exactly that W (all-ones vector in the column span): raw rank d,
   centered rank d-1 — a raw-only implementation goes RED there.
2. **injective-but-lossy hazard** — the pipeline computes softmax in fp32 and
   applies ``clamp_min(1e-12)`` (``field_encoder._encode``), so an
   analytically injective head still destroys tail-logit information on peaked
   distributions. ``test_reconstruction_degrades_under_clamp_saturation`` pins
   that the report flags clamped positions AND that reconstruction error blows
   up relative to the benign regime — a rank-only audit cannot pass it.
3. **wrong-tensor hazard** — H's ``state_lift`` and the head both consume the
   POST-``norm_f`` hidden state (``field_encoder._encode``). A hook that
   captures pre-norm ``h`` yields states the head never saw.
   ``test_collect_field_states_hidden_is_the_heads_own_input`` pins
   ``hidden @ W^T == logits`` against the encoder's own forward output.

Exploratory posture: this is an E1.5-D-style endpoint diagnostic (no training,
CPU, checkpoints only) — it feeds judgment, not a registered gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from entropic_substrate.core.substrate.field_encoder import _PROB_EPS, FieldEncoder
from entropic_substrate.core.substrate.fisher import semi_orthogonal_lift
from entropic_substrate.kernel.runtime.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointFormatError,
)
from entropic_substrate.ladder.analyze.field_head_audit import (
    PROB_FLOOR,
    FieldHeadAudit,
    HeadSpectrum,
    ReconstructionReport,
    audit_field_head,
    center_logit_columns,
    collect_field_states,
    head_spectrum,
    load_frozen_head,
    realized_spectrum,
    reconstruct_hidden,
    reconstruction_report,
)
from helpers import assert_bitwise_equal, tiny_model_config

pytestmark = pytest.mark.unit

_N_BYTES = 256


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _random_head(d: int, *, seed: int = 0) -> torch.Tensor:
    """A generic full-column-rank (256, d) head, fp32 like a real checkpoint."""
    return torch.randn(_N_BYTES, d, generator=_gen(seed), dtype=torch.float32)


def _orthonormal_head(d: int) -> torch.Tensor:
    """(256, d) with exactly orthonormal columns — every singular value 1.0."""
    return semi_orthogonal_lift(_N_BYTES, d, seed=0).float()


def _pipeline_probs(hidden: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Probabilities exactly as ``field_encoder._encode`` produces them for the
    P arm: fp32 softmax over ``h @ W^T``, then ``clamp_min`` at the pipeline
    floor. The audit must be exercised against THIS construction, not an
    idealized fp64 softmax — hazard 2 in the module docstring lives here."""
    logits = hidden.float() @ w.float().T
    return torch.softmax(logits, dim=-1).clamp_min(_PROB_EPS)


def _field_encoder(**overrides: object) -> FieldEncoder:
    defaults: dict[str, object] = {
        "state_source": "field",
        "field_d_model": 16,
        "field_n_layers": 2,
        "field_n_heads": 2,
    }
    defaults.update(overrides)
    return FieldEncoder(tiny_model_config(**defaults))


# --------------------------------------------------------------- prob floor pin
def test_prob_floor_matches_the_pipelines_own_clamp() -> None:
    """The audit's clamp-detection floor must be the SAME constant the encoder
    clamps with — if ``field_encoder._PROB_EPS`` moves and this module's copy
    does not, clamp detection silently measures a floor the pipeline no longer
    uses."""
    assert PROB_FLOOR == _PROB_EPS


# ----------------------------------------------------------- center_logit_columns
def test_center_logit_columns_zeroes_column_means() -> None:
    w = _random_head(16)
    centered = center_logit_columns(w)
    assert centered.shape == w.shape
    assert centered.dtype == torch.float64
    assert torch.allclose(centered.mean(dim=0), torch.zeros(16, dtype=torch.float64), atol=1e-12)


def test_center_logit_columns_is_invariant_to_softmax_shift_directions() -> None:
    """Adding ``1 @ v^T`` to W shifts every logit vector by the scalar
    ``v . h`` — a softmax no-op. The centered head must be blind to it."""
    w = _random_head(16, seed=1)
    v = torch.randn(16, generator=_gen(2), dtype=torch.float32)
    shifted = w + torch.ones(_N_BYTES, 1) @ v.unsqueeze(0)
    assert torch.allclose(center_logit_columns(shifted), center_logit_columns(w), atol=1e-4)


# ----------------------------------------------------------------- head_spectrum
def test_head_spectrum_reports_fp64_descending_spectra() -> None:
    spec = head_spectrum(_random_head(16))
    assert isinstance(spec, HeadSpectrum)
    for values in (spec.singular_values, spec.centered_singular_values):
        assert values.dtype == torch.float64
        assert values.shape == (16,)
        assert bool(torch.all(values[:-1] >= values[1:])), "spectrum must be descending"


def test_head_spectrum_on_orthonormal_columns() -> None:
    """Orthonormal columns: every raw singular value is 1, so raw rank, the
    condition number's numerator/denominator, and the entropy effective rank
    are all known in closed form — the calibration case for the estimator."""
    spec = head_spectrum(_orthonormal_head(32))
    assert spec.rank == 32
    assert torch.allclose(spec.singular_values, torch.ones(32, dtype=torch.float64), atol=1e-10)
    # seed=0 lift: the all-ones direction is not in the column span (generic
    # for a Haar draw; deterministic for this fixed seed), so centering costs
    # no rank and the map into the softmax quotient stays injective.
    assert spec.centered_rank == 32
    assert spec.sigma_min_centered > 0.0
    assert spec.condition_number_centered >= 1.0


def test_head_spectrum_effective_rank_equals_count_for_flat_spectrum() -> None:
    """Entropy effective rank (exp of the spectral entropy) is exactly the
    column count when all singular values are equal — pinning the definition,
    not just monotonicity."""
    spec = head_spectrum(_orthonormal_head(32))
    assert spec.effective_rank_centered == pytest.approx(32.0, abs=0.5)


def test_head_spectrum_detects_ones_direction_rank_drop() -> None:
    """GUARD TOOTH (hazard 1): W with ``1_256`` in its column span is full
    rank raw but rank-deficient in the softmax quotient — two hidden states
    differing along that column's preimage produce IDENTICAL probabilities.
    A raw-SVD-only implementation reports 16/16 here and goes RED."""
    w = _random_head(16, seed=3)
    w[:, 0] = 1.0
    spec = head_spectrum(w)
    assert spec.rank == 16
    assert spec.centered_rank == 15
    assert spec.sigma_min_centered == pytest.approx(0.0, abs=1e-6)


def test_head_spectrum_on_rank_deficient_input() -> None:
    u = torch.randn(_N_BYTES, 1, generator=_gen(4))
    v = torch.randn(1, 16, generator=_gen(5))
    spec = head_spectrum(u @ v)
    assert spec.rank == 1
    assert spec.centered_rank <= 1


# ------------------------------------------------------------ reconstruction
def test_reconstruct_hidden_roundtrips_in_the_benign_regime() -> None:
    """Well-conditioned head, O(1) logits: fp32 softmax + clamp loses nothing,
    so least-squares through the centered head must recover h to fp32-noise
    levels. This is the 'injective in practice' half of the audit."""
    w = _orthonormal_head(16)
    hidden = torch.randn(64, 16, generator=_gen(6))
    probs = _pipeline_probs(hidden, w)
    h_hat = reconstruct_hidden(probs, w)
    assert h_hat.shape == (64, 16)
    assert h_hat.dtype == torch.float64
    rel = (h_hat - hidden.double()).norm(dim=1) / hidden.double().norm(dim=1)
    assert float(rel.median()) < 1e-3


def test_reconstruction_report_counts_clamped_positions_exactly() -> None:
    """Clamp accounting is a count, not a vibe: rows whose minimum sits at the
    pipeline floor are clamped, others are not — 2 of 4 here."""
    w = _orthonormal_head(8)
    probs = torch.full((4, _N_BYTES), 1.0 / _N_BYTES)
    probs[0, 0] = PROB_FLOOR
    probs[2, 5] = PROB_FLOOR
    hidden = torch.zeros(4, 8)
    report = reconstruction_report(hidden, probs, w)
    assert isinstance(report, ReconstructionReport)
    assert report.n_positions == 4
    assert report.clamped_fraction == pytest.approx(0.5)


def test_reconstruction_degrades_under_clamp_saturation() -> None:
    """GUARD TOOTH (hazard 2): scale h until tail probabilities underflow the
    1e-12 clamp — the clamp overwrites their logits, and reconstruction error
    must blow up RELATIVE to the benign regime while the report flags the
    saturation. Contrast-pinned (ratio, not absolute threshold) so the tooth
    measures the mechanism rather than one lucky draw."""
    w = _orthonormal_head(16)
    hidden_small = torch.randn(64, 16, generator=_gen(7))
    hidden_big = 3000.0 * torch.randn(64, 16, generator=_gen(7))
    report_small = reconstruction_report(hidden_small, _pipeline_probs(hidden_small, w), w)
    report_big = reconstruction_report(hidden_big, _pipeline_probs(hidden_big, w), w)
    assert report_small.clamped_fraction == 0.0
    assert report_big.clamped_fraction > 0.5
    assert report_small.median_rel_error < 1e-3
    assert report_big.median_rel_error > 100.0 * report_small.median_rel_error


def test_reconstruction_report_refuses_mismatched_positions() -> None:
    w = _orthonormal_head(8)
    with pytest.raises(ValueError, match="position"):
        reconstruction_report(torch.zeros(3, 8), torch.full((4, _N_BYTES), 1.0 / _N_BYTES), w)


def test_reconstruction_report_refuses_zero_positions() -> None:
    """Measurement path fails LOUD (CLAUDE.md failure semantics): an empty
    audit must never report a silent NaN/0.0 as if something was measured."""
    w = _orthonormal_head(8)
    with pytest.raises(ValueError, match=r"zero|no positions|empty"):
        reconstruction_report(torch.zeros(0, 8), torch.zeros(0, _N_BYTES), w)


# ---------------------------------------------------------- realized_spectrum
def test_realized_spectrum_recovers_planted_rank() -> None:
    """If the data only excites k directions of the 128-dim state, the '128
    dimensions' framing overstates what the head had to preserve — the audit
    must see the planted k."""
    basis = torch.randn(3, 16, generator=_gen(8))
    coeffs = torch.randn(200, 3, generator=_gen(9))
    spec = realized_spectrum(coeffs @ basis)
    assert spec.rank == 3
    assert spec.singular_values.dtype == torch.float64
    assert spec.effective_rank <= 3.0 + 1e-6


def test_realized_spectrum_refuses_fewer_than_two_positions() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        realized_spectrum(torch.zeros(1, 16))


# ----------------------------------------------------------- load_frozen_head
def _write_ckpt(path: Path, model_state: dict[str, torch.Tensor]) -> Path:
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "model": model_state}, path)
    return path


def test_load_frozen_head_roundtrips_tied_payload(tmp_path: Path) -> None:
    w = _random_head(16, seed=10)
    path = _write_ckpt(tmp_path / "field.pt", {"wte.weight": w, "head.weight": w})
    assert_bitwise_equal(load_frozen_head(path), w)


def test_load_frozen_head_accepts_wte_only_payload(tmp_path: Path) -> None:
    """Tied parameters are one storage; a payload carrying only ``wte.weight``
    still names the head unambiguously."""
    w = _random_head(16, seed=11)
    path = _write_ckpt(tmp_path / "field.pt", {"wte.weight": w})
    assert_bitwise_equal(load_frozen_head(path), w)


def test_load_frozen_head_refuses_untied_payload(tmp_path: Path) -> None:
    """``head.weight != wte.weight`` is not the registered tied head — loading
    either tensor silently would audit a head the runs never used."""
    path = _write_ckpt(
        tmp_path / "field.pt",
        {"wte.weight": _random_head(16, seed=12), "head.weight": _random_head(16, seed=13)},
    )
    with pytest.raises(RuntimeError, match="tied"):
        load_frozen_head(path)


def test_load_frozen_head_refuses_headless_payload(tmp_path: Path) -> None:
    path = _write_ckpt(tmp_path / "field.pt", {"norm_f.weight": torch.ones(16)})
    with pytest.raises(RuntimeError, match=str(path)):
        load_frozen_head(path)


def test_load_frozen_head_routes_through_normalize_checkpoint(tmp_path: Path) -> None:
    """A non-mapping payload must be refused by the repo's validated reader
    (``CheckpointFormatError``), not surface as an AttributeError downstream."""
    path = tmp_path / "junk.pt"
    torch.save([1, 2, 3], path)
    with pytest.raises(CheckpointFormatError):
        load_frozen_head(path)


# -------------------------------------------------------- collect_field_states
def test_collect_field_states_shapes_and_floor() -> None:
    encoder = _field_encoder()
    batches = [
        torch.randint(0, _N_BYTES, (2, 8), generator=_gen(14)),
        torch.randint(0, _N_BYTES, (3, 8), generator=_gen(15)),
    ]
    states = collect_field_states(encoder, batches)
    assert states.hidden.shape == (40, 16)
    assert states.probs.shape == (40, _N_BYTES)
    assert not states.hidden.requires_grad
    assert float(states.probs.min()) >= PROB_FLOOR
    sums = states.probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_collect_field_states_hidden_is_the_heads_own_input() -> None:
    """GUARD TOOTH (hazard 3): the captured hidden state must be the POST-
    ``norm_f`` tensor the head consumes — ``hidden @ W^T`` must reproduce the
    encoder's own forward logits. A pre-norm capture goes RED here."""
    encoder = _field_encoder()
    idx = torch.randint(0, _N_BYTES, (2, 8), generator=_gen(16))
    states = collect_field_states(encoder, [idx])
    with torch.no_grad():
        logits = encoder(idx).logits.reshape(-1, _N_BYTES)
    recomputed = states.hidden @ encoder.head.weight.T
    assert torch.allclose(recomputed, logits, atol=1e-5)


def test_collect_field_states_refuses_empty_batches() -> None:
    with pytest.raises(ValueError, match=r"no batches|empty"):
        collect_field_states(_field_encoder(), [])


# ------------------------------------------------------------ audit_field_head
def test_audit_field_head_composes_all_three_diagnostics() -> None:
    encoder = _field_encoder()
    batches = [torch.randint(0, _N_BYTES, (2, 8), generator=_gen(17)) for _ in range(3)]
    audit = audit_field_head(encoder, batches)
    assert isinstance(audit, FieldHeadAudit)
    assert isinstance(audit.spectrum, HeadSpectrum)
    assert isinstance(audit.reconstruction, ReconstructionReport)
    assert audit.n_positions == 48
    assert audit.reconstruction.n_positions == 48
    assert audit.spectrum.singular_values.shape == (16,)
    assert audit.realized.singular_values.dtype == torch.float64
