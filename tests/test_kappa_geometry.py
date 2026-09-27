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

from entropic_substrate.core.geometry.kappa import (
    KAPPA_MAX,
    KAPPA_MIN,
    TAYLOR_EPS,
    artan_k,
    clamp_kappa,
    dist0,
    dist_k,
    dist_k_from_cos,
    dist_k_from_inner,
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
    from entropic_substrate.core.geometry.kappa import _tan_k_taylor

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


# --------------------------------------------------------------------------
# dist_k — pairwise geodesic distance (the E3 attention kernel's primitive)
# --------------------------------------------------------------------------


def _point_pairs(kappa: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Two distinct clouds comfortably inside the chart."""
    x = _points(kappa)
    torch.manual_seed(1)
    y = torch.randn_like(x)
    y = y / y.norm(dim=-1, keepdim=True)
    radius = 0.35 if kappa >= 0 else min(0.35, 0.7 / abs(kappa) ** 0.5)
    scale = torch.linspace(0.02, radius, y.shape[0], dtype=torch.float64).unsqueeze(-1)
    return x, y * scale


@pytest.mark.parametrize("kappa", CURVATURES)
def test_dist_k_matches_geoopt(kappa: float) -> None:
    manifold = geoopt.Stereographic(k=kappa)
    k = torch.tensor(kappa, dtype=torch.float64)
    x, y = _point_pairs(kappa)
    torch.testing.assert_close(dist_k(x, y, k), manifold.dist(x, y), atol=ATOL, rtol=0)


def test_dist_k_zero_curvature_is_twice_euclidean() -> None:
    k = torch.tensor(0.0, dtype=torch.float64)
    x, y = _point_pairs(0.0)
    torch.testing.assert_close(dist_k(x, y, k), 2.0 * (x - y).norm(dim=-1), atol=ATOL, rtol=0)


@pytest.mark.parametrize("kappa", CURVATURES)
def test_dist_k_identical_points_and_symmetry(kappa: float) -> None:
    k = torch.tensor(kappa, dtype=torch.float64)
    x, y = _point_pairs(kappa)
    assert float(dist_k(x, x, k).abs().max()) < 1e-6
    torch.testing.assert_close(dist_k(x, y, k), dist_k(y, x, k), atol=ATOL, rtol=0)


def test_dist_k_taylor_seam_is_continuous() -> None:
    """No jump when the learnable curvature crosses the guard band — the
    Taylor branch just inside must agree with the closed form just outside.
    (Inside the band the distance genuinely varies with kappa — that
    first-order dependence is the point — so the comparison is across the
    branch switch at matched kappa, not against kappa = 0.)"""
    x, y = _point_pairs(0.0)
    for sign in (-1.0, 1.0):
        inside = dist_k(x, y, torch.tensor(sign * TAYLOR_EPS * 0.99, dtype=torch.float64))
        outside = dist_k(x, y, torch.tensor(sign * TAYLOR_EPS * 1.01, dtype=torch.float64))
        torch.testing.assert_close(inside, outside, atol=1e-5, rtol=0)


def test_dist_k_gradient_reaches_kappa_at_exactly_zero() -> None:
    """The invariant the annihilated norm layer could not satisfy: between two
    DISTINCT points, curvature has first-order effect even at the Taylor seam."""
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    y = torch.randn(8, 16)
    x = 0.3 * x / x.norm(dim=-1, keepdim=True)
    y = 0.2 * y / y.norm(dim=-1, keepdim=True)
    kappa = torch.zeros((), requires_grad=True)
    dist_k(x, y, kappa).sum().backward()
    assert kappa.grad is not None
    assert float(kappa.grad.abs()) > 1e-4


@pytest.mark.parametrize("kappa", CURVATURES)
@pytest.mark.parametrize("radius", [0.2, 0.35])
def test_dist_k_from_cos_matches_dist_k_on_common_radius(kappa: float, radius: float) -> None:
    if kappa < 0:
        radius = min(radius, 0.7 / abs(kappa) ** 0.5)
    k = torch.tensor(kappa, dtype=torch.float64)
    torch.manual_seed(2)
    x = torch.randn(32, 16, dtype=torch.float64)
    y = torch.randn(32, 16, dtype=torch.float64)
    x = radius * x / x.norm(dim=-1, keepdim=True)
    y = radius * y / y.norm(dim=-1, keepdim=True)
    cos = (x * y).sum(-1) / radius**2
    r = torch.tensor(radius, dtype=torch.float64)
    torch.testing.assert_close(dist_k_from_cos(cos, k, r), dist_k(x, y, k), atol=ATOL, rtol=0)


def test_dist_k_from_cos_boundary_cosines() -> None:
    r = torch.tensor(0.35, dtype=torch.float64)
    for kappa in (-2.0, 0.0, 2.0):
        k = torch.tensor(kappa, dtype=torch.float64)
        same = dist_k_from_cos(torch.tensor(1.0, dtype=torch.float64), k, r)
        anti = dist_k_from_cos(torch.tensor(-1.0, dtype=torch.float64), k, r)
        assert float(same.abs()) < 1e-6
        assert torch.isfinite(anti) and float(anti) > 0.0


def test_scalar_reductions_run_in_fp32_under_bf16_autocast() -> None:
    """Both scalar reductions are called from inside an attention block that may
    be under autocast. The gyro rational functions are not conditioned for
    bf16 operands, so they must promote to fp32 and keep gradients finite."""
    for kappa in (-2.0, 0.0, 2.0):
        dot = torch.tensor([0.05, -0.02], dtype=torch.bfloat16, requires_grad=True)
        x_sq = torch.tensor([0.09, 0.09], dtype=torch.bfloat16)
        y_sq = torch.tensor([0.16, 0.04], dtype=torch.bfloat16)
        cos = torch.tensor([0.9, -0.9], dtype=torch.bfloat16, requires_grad=True)
        radius = torch.tensor(0.35, dtype=torch.bfloat16)
        k = torch.tensor(kappa, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            inner = dist_k_from_inner(dot, x_sq, y_sq, k)
            from_cos = dist_k_from_cos(cos, k, radius)
        assert inner.dtype == torch.float32 and from_cos.dtype == torch.float32
        assert torch.isfinite(inner).all() and torch.isfinite(from_cos).all()
        (inner.sum() + from_cos.sum()).backward()
        for leaf in (dot, cos, k):
            assert leaf.grad is not None and torch.isfinite(leaf.grad).all(), kappa


def test_scalar_reductions_preserve_float64() -> None:
    """The geoopt oracle comparisons run in float64 — promotion must floor at
    fp32, never cap there."""
    k = torch.tensor(-1.0, dtype=torch.float64)
    r = torch.tensor(0.35, dtype=torch.float64)
    cos = torch.tensor([0.5], dtype=torch.float64)
    assert dist_k_from_cos(cos, k, r).dtype == torch.float64
    sq = torch.tensor([0.1225], dtype=torch.float64)
    assert dist_k_from_inner(cos * sq, sq, sq, k).dtype == torch.float64


def test_dist_k_from_inner_matches_dist_k_on_mixed_norms() -> None:
    """General scalar reduction (anchored kernel): points of DIFFERENT norms."""
    torch.manual_seed(3)
    for kappa in CURVATURES:
        x, y = _point_pairs(kappa)
        k = torch.tensor(kappa, dtype=torch.float64)
        dot = (x * y).sum(-1)
        x2 = x.pow(2).sum(-1)
        y2 = y.pow(2).sum(-1)
        torch.testing.assert_close(
            dist_k_from_inner(dot, x2, y2, k), dist_k(x, y, k), atol=ATOL, rtol=0
        )
