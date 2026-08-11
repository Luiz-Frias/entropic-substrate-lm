r"""Validate our :math:`\kappa`-stereographic primitives against ``geoopt``.

``geoopt.Stereographic`` is the *oracle*: it is a mature, independently tested
implementation of the same gyrovector formulas. Our version exists to be
branch-free / compile-friendly and to hold a learnable curvature, so it must
agree numerically wherever geoopt is well defined.
"""

from __future__ import annotations

import geoopt
import pytest
import torch

from entropic_substrate.geometry.kappa import (
    KAPPA_MAX,
    KAPPA_MIN,
    TAYLOR_EPS,
    artan_k,
    clamp_kappa,
    dist0,
    expmap0,
    logmap0,
    mobius_add,
    projx,
    tan_k,
)

pytestmark = pytest.mark.unit

# geoopt names the curvature `k` with the same sign convention as our `kappa`.
CURVATURES = [-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0]
ATOL = 1e-5


def _points(kappa: float, n: int = 64, d: int = 16) -> torch.Tensor:
    """Random points comfortably inside the chart for this curvature."""
    torch.manual_seed(0)
    x = torch.randn(n, d, dtype=torch.float64)
    x = x / x.norm(dim=-1, keepdim=True)
    radius = 0.35 if kappa >= 0 else min(0.35, 0.7 / abs(kappa) ** 0.5)
    scale = torch.linspace(0.01, radius, n, dtype=torch.float64).unsqueeze(-1)
    return x * scale


@pytest.mark.parametrize("kappa", CURVATURES)
def test_tan_k_matches_geoopt(kappa: float) -> None:
    from geoopt.manifolds.stereographic.math import tan_k as oracle

    k = torch.tensor(kappa, dtype=torch.float64)
    x = torch.linspace(-0.6, 0.6, 101, dtype=torch.float64)
    torch.testing.assert_close(tan_k(x, k), oracle(x, k), atol=ATOL, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_artan_k_matches_geoopt(kappa: float) -> None:
    from geoopt.manifolds.stereographic.math import artan_k as oracle

    k = torch.tensor(kappa, dtype=torch.float64)
    x = torch.linspace(-0.6, 0.6, 101, dtype=torch.float64)
    torch.testing.assert_close(artan_k(x, k), oracle(x, k), atol=ATOL, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_expmap0_logmap0_dist0_match_geoopt(kappa: float) -> None:
    manifold = geoopt.Stereographic(k=kappa)
    k = torch.tensor(kappa, dtype=torch.float64)
    u = _points(kappa)

    torch.testing.assert_close(expmap0(u, k), manifold.expmap0(u), atol=ATOL, rtol=0)
    y = manifold.expmap0(u)
    torch.testing.assert_close(logmap0(y, k), manifold.logmap0(y), atol=ATOL, rtol=0)
    torch.testing.assert_close(dist0(y, k), manifold.dist0(y), atol=ATOL, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_projx_matches_geoopt(kappa: float) -> None:
    manifold = geoopt.Stereographic(k=kappa)
    k = torch.tensor(kappa, dtype=torch.float64)
    # Deliberately include points outside the hyperbolic ball.
    x = _points(kappa) * 12.0
    torch.testing.assert_close(projx(x, k), manifold.projx(x), atol=1e-6, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_mobius_add_matches_geoopt(kappa: float) -> None:
    manifold = geoopt.Stereographic(k=kappa)
    k = torch.tensor(kappa, dtype=torch.float64)
    x, y = _points(kappa), _points(kappa).flip(0)
    torch.testing.assert_close(mobius_add(x, y, k), manifold.mobius_add(x, y), atol=ATOL, rtol=0)


def test_zero_curvature_is_the_euclidean_identity() -> None:
    """The kappa -> 0 Taylor path must reduce exp/log to the identity."""
    k = torch.tensor(0.0, dtype=torch.float64)
    u = _points(0.0)
    torch.testing.assert_close(expmap0(u, k), u, atol=1e-12, rtol=0)
    torch.testing.assert_close(logmap0(u, k), u, atol=1e-12, rtol=0)
    torch.testing.assert_close(projx(u * 100, k), u * 100, atol=0, rtol=0)


@pytest.mark.parametrize("kappa", [-1e-6, -1e-5, 0.0, 1e-5, 1e-6])
def test_taylor_guard_region_is_finite_and_continuous(kappa: float) -> None:
    """Near kappa = 0 the closed forms blow up; the Taylor branch must not."""
    assert abs(kappa) < TAYLOR_EPS
    k = torch.tensor(kappa, dtype=torch.float64, requires_grad=True)
    x = _points(0.0)
    y = expmap0(x, k)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert k.grad is not None
    assert torch.isfinite(k.grad).all()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_taylor_branch_agrees_with_closed_form_at_the_seam(sign: float) -> None:
    """At the switchover curvature both branches must give the same answer."""
    from entropic_substrate.geometry.kappa import _tan_k_taylor

    k = torch.tensor(sign * TAYLOR_EPS * 1.001, dtype=torch.float64)
    x = torch.linspace(-1.5, 1.5, 201, dtype=torch.float64)
    torch.testing.assert_close(tan_k(x, k), _tan_k_taylor(x, k), atol=1e-12, rtol=0)


def test_expmap_is_continuous_across_the_seam() -> None:
    """No visible jump when the learnable curvature crosses the guard band."""
    x = _points(0.0)
    inside = expmap0(x, torch.tensor(TAYLOR_EPS * 0.99, dtype=torch.float64))
    outside = expmap0(x, torch.tensor(TAYLOR_EPS * 1.01, dtype=torch.float64))
    torch.testing.assert_close(inside, outside, atol=1e-5, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_expmap0_logmap0_are_mutual_inverses(kappa: float) -> None:
    k = torch.tensor(kappa, dtype=torch.float64)
    u = _points(kappa)
    torch.testing.assert_close(logmap0(expmap0(u, k), k), u, atol=1e-9, rtol=0)


def test_kappa_is_clamped_to_the_configured_range() -> None:
    raw = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])
    clamped = clamp_kappa(raw)
    assert float(clamped.min()) == KAPPA_MIN
    assert float(clamped.max()) == KAPPA_MAX
    # Out-of-range curvature must not produce non-finite geometry.
    y = expmap0(torch.randn(8, 16) * 0.1, torch.tensor(1e6))
    assert torch.isfinite(y).all()


def test_gradients_flow_to_kappa_through_expmap() -> None:
    k = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    u = _points(-0.7)
    dist0(expmap0(u, k), k).sum().backward()
    assert k.grad is not None
    assert torch.isfinite(k.grad).all()
    assert float(k.grad.abs()) > 0.0


def test_projx_keeps_points_inside_the_hyperbolic_ball() -> None:
    k = torch.tensor(-4.0)
    x = torch.randn(128, 16) * 50.0
    inside = projx(x, k)
    assert bool((inside.norm(dim=-1) <= 1.0 / abs(float(k)) ** 0.5).all())


def test_float32_path_is_finite_for_extreme_inputs() -> None:
    """FP32 is the runtime dtype for all geometry; it must never emit NaN."""
    for kappa in (-5.0, -1.0, 0.0, 1.0, 5.0):
        k = torch.tensor(kappa, dtype=torch.float32)
        x = torch.randn(256, 64, dtype=torch.float32) * 1e3
        y = expmap0(logmap0(projx(x, k), k), k)
        assert torch.isfinite(y).all(), kappa
