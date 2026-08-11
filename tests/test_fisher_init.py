r"""Tests for the :math:`\sqrt{p}` Fisher-Rao byte initialisation."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from entropic_substrate.init.fisher import (
    build_fisher_init,
    count_transitions,
    fisher_sqrt_simplex_embedding,
    load_init,
    random_sphere_init,
    save_init,
    semi_orthogonal_lift,
)

pytestmark = pytest.mark.unit


def test_count_transitions_matches_bruteforce() -> None:
    blob = bytes([1, 2, 3, 1, 2, 3, 7, 7, 7, 0])
    counts = count_transitions(blob)
    expected = torch.zeros(256, 256, dtype=torch.float64)
    for a, b in pairwise(blob):
        expected[a, b] += 1
    assert torch.equal(counts, expected)
    assert counts.sum().item() == len(blob) - 1


def test_count_transitions_stitches_chunk_boundaries() -> None:
    whole = bytes([5, 9, 5, 9, 200, 3])
    chunked = count_transitions([bytes([5, 9]), bytes([5, 9]), bytes([200, 3])])
    assert torch.equal(chunked, count_transitions(whole))


def test_count_transitions_respects_max_bytes() -> None:
    blob = bytes(range(256)) * 4
    counts = count_transitions(blob, max_bytes=100)
    assert counts.sum().item() == 99


def test_sqrt_embedding_is_unit_norm_and_positive(sample_bytes: bytes) -> None:
    rows = fisher_sqrt_simplex_embedding(count_transitions(sample_bytes))
    assert rows.shape == (256, 256)
    torch.testing.assert_close(
        rows.norm(dim=1), torch.ones(256, dtype=torch.float64), atol=1e-12, rtol=0
    )
    # Strictly inside the positive orthant thanks to Laplace smoothing.
    assert bool((rows > 0).all())


def test_sqrt_embedding_rejects_zero_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        fisher_sqrt_simplex_embedding(torch.zeros(256, 256), smoothing=0.0)


def test_semi_orthogonal_lift_is_isometric() -> None:
    lift = semi_orthogonal_lift(256, 768, seed=3)
    assert lift.shape == (256, 768)
    gram = lift @ lift.T
    torch.testing.assert_close(gram, torch.eye(256, dtype=gram.dtype), atol=1e-10, rtol=0)

    x = torch.randn(16, 256, dtype=torch.float64)
    y = torch.randn(16, 256, dtype=torch.float64)
    torch.testing.assert_close((x @ lift) @ (y @ lift).T, x @ y.T, atol=1e-10, rtol=0)


def test_semi_orthogonal_lift_is_seed_deterministic() -> None:
    assert torch.equal(semi_orthogonal_lift(8, 32, seed=1), semi_orthogonal_lift(8, 32, seed=1))
    assert not torch.equal(semi_orthogonal_lift(8, 32, seed=1), semi_orthogonal_lift(8, 32, seed=2))


def test_semi_orthogonal_projection_when_shrinking() -> None:
    """d_out < d_in (CPU-tiny only) degrades to an orthogonal projection."""
    proj = semi_orthogonal_lift(256, 64, seed=0)
    assert proj.shape == (256, 64)
    torch.testing.assert_close(proj.T @ proj, torch.eye(64, dtype=proj.dtype), atol=1e-10, rtol=0)


def test_semi_orthogonal_lift_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        semi_orthogonal_lift(0, 32)


def test_build_fisher_init_shape_norm_and_geometry(sample_bytes: bytes) -> None:
    d_model = 768
    table = build_fisher_init(sample_bytes, d_model, seed=0)
    assert table.shape == (256, d_model)
    assert table.dtype == torch.float32
    torch.testing.assert_close(table.norm(dim=1), torch.ones(256), atol=1e-5, rtol=0)

    # The lift preserves Fisher-Rao angles: pairwise inner products in R^d must
    # equal the Bhattacharyya coefficients computed in the 256-dim simplex chart.
    sqrt_p = fisher_sqrt_simplex_embedding(count_transitions(sample_bytes))
    expected = (sqrt_p @ sqrt_p.T).float()
    torch.testing.assert_close(table @ table.T, expected, atol=1e-5, rtol=0)


def test_build_fisher_init_is_deterministic(sample_bytes: bytes) -> None:
    a = build_fisher_init(sample_bytes, 384, seed=11)
    b = build_fisher_init(sample_bytes, 384, seed=11)
    assert torch.equal(a, b)


def test_build_fisher_init_works_for_tiny_dims(sample_bytes: bytes) -> None:
    """The CPU-tiny configs use d_model < 256; the table must still be valid."""
    table = build_fisher_init(sample_bytes, 64)
    assert table.shape == (256, 64)
    torch.testing.assert_close(table.norm(dim=1), torch.ones(256), atol=1e-5, rtol=0)


def test_random_sphere_init_is_unit_norm() -> None:
    table = random_sphere_init(256, 96, seed=5)
    assert table.shape == (256, 96)
    torch.testing.assert_close(table.norm(dim=1), torch.ones(256), atol=1e-6, rtol=0)
    # Unlike the Fisher init it is *not* confined to the positive orthant.
    assert bool((table < 0).any())


def test_save_and_load_roundtrip(tmp_path, sample_bytes: bytes) -> None:
    table = build_fisher_init(sample_bytes, 320)
    path = save_init(table, tmp_path / "nested" / "fisher.pt")
    assert path.exists()
    assert torch.equal(load_init(path), table)
