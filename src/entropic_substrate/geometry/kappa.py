r""":math:`\kappa`-stereographic primitives, implemented directly.

The :math:`\kappa`-stereographic model is a single chart that interpolates
smoothly between hyperbolic space (:math:`\kappa < 0`, Poincare ball of radius
:math:`1/\sqrt{-\kappa}`), Euclidean space (:math:`\kappa = 0`) and the sphere
(:math:`\kappa > 0`, stereographic projection).

Why re-implement instead of calling ``geoopt``:

* every op here is written with ``torch.where`` blends and *no* data-dependent
  Python control flow, so the whole module is ``torch.compile``/CUDA-graph
  friendly and can be called with a per-component learnable ``kappa`` that
  freely changes sign during training;
* all branches are fed *safe* inputs (clamped radicands / arguments) so that
  the unused branch of a ``torch.where`` cannot poison the backward pass with
  ``NaN``;
* a signed Taylor guard is used for :math:`|\kappa| < \varepsilon` where the
  closed forms are numerically unstable.

Formulas follow the vendor pattern of ``geoopt`` (Apache-2.0),
``geoopt/manifolds/stereographic/math.py``, itself derived from:

    [1] Ganea, Becigneul, Hofmann. "Hyperbolic neural networks." NeurIPS 2018.
    [2] Bachmann, Becigneul, Ganea. "Constant Curvature Graph Convolutional
        Networks." arXiv:1911.05076.
    [3] Skopek, Ganea, Becigneul. "Mixed-curvature Variational Autoencoders."
        arXiv:1911.08411.

``geoopt.Stereographic`` is used as a *test oracle* only (see
``tests/test_kappa_geometry.py``); it is not a runtime dependency of the model.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "KAPPA_MAX",
    "KAPPA_MIN",
    "TAYLOR_EPS",
    "artan_k",
    "clamp_kappa",
    "dist0",
    "dist_k",
    "dist_k_from_cos",
    "expmap0",
    "logmap0",
    "mobius_add",
    "projx",
    "tan_k",
]

# Curvature is clamped to this interval; outside it the stereographic chart
# becomes numerically hostile for the norms we actually use (|x| ~ 1).
KAPPA_MIN: float = -5.0
KAPPA_MAX: float = 5.0

# |kappa| below this switches to the signed Taylor expansion.
TAYLOR_EPS: float = 1e-4

# Generic small numbers.
_NORM_EPS: float = 1e-15
# Keep tan() away from its pole at pi/2 (spherical injectivity radius).
_TAN_LIMIT: float = math.pi / 2 - 1e-4
# Keep artanh() inside (-1, 1).
_ARTANH_LIMIT: float = 1.0 - 1e-6
# Keep tanh() out of the saturated / non-finite regime.
_TANH_LIMIT: float = 15.0
# Boundary margin used by ``projx`` for kappa < 0.
_BALL_EPS_FP32: float = 4e-3
_BALL_EPS_FP64: float = 1e-5


def clamp_kappa(kappa: torch.Tensor) -> torch.Tensor:
    """Clamp curvature into ``[KAPPA_MIN, KAPPA_MAX]`` (gradient preserving)."""
    return kappa.clamp(KAPPA_MIN, KAPPA_MAX)


def _safe_abs_sqrt(kappa: torch.Tensor) -> torch.Tensor:
    """``sqrt(|kappa|)`` with a floor so the derivative stays finite at 0."""
    return kappa.abs().clamp_min(TAYLOR_EPS).sqrt()


def _tan_k_taylor(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    r"""Order-11 signed Taylor series of :math:`\tan_\kappa(x)` around ``kappa=0``.

    Valid for *both* signs of :math:`\kappa`:
    :math:`\tan_\kappa(x) = x + \tfrac{\kappa}{3}x^3 + \tfrac{2\kappa^2}{15}x^5
    + \tfrac{17\kappa^3}{315}x^7 + \tfrac{62\kappa^4}{2835}x^9 + O(\kappa^5)`.
    """
    x2 = x * x
    k = kappa
    return x * (
        1.0
        + (1.0 / 3.0) * k * x2
        + (2.0 / 15.0) * k**2 * x2**2
        + (17.0 / 315.0) * k**3 * x2**3
        + (62.0 / 2835.0) * k**4 * x2**4
    )


def _artan_k_taylor(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    r"""Order-9 signed Taylor series of :math:`\tan^{-1}_\kappa(x)`.

    :math:`\tan^{-1}_\kappa(x) = x - \tfrac{\kappa}{3}x^3 + \tfrac{\kappa^2}{5}x^5
    - \tfrac{\kappa^3}{7}x^7 + \tfrac{\kappa^4}{9}x^9 + O(\kappa^5)`.
    """
    x2 = x * x
    k = kappa
    return x * (
        1.0
        - (1.0 / 3.0) * k * x2
        + (1.0 / 5.0) * k**2 * x2**2
        - (1.0 / 7.0) * k**3 * x2**3
        + (1.0 / 9.0) * k**4 * x2**4
    )


def tan_k(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    r"""Curvature-aware tangent.

    .. math::
        \tan_\kappa(x) = \begin{cases}
            \kappa^{-1/2}\tan(\sqrt{\kappa}\,x) & \kappa > 0\\
            x                                   & \kappa = 0\\
            |\kappa|^{-1/2}\tanh(\sqrt{|\kappa|}\,x) & \kappa < 0
        \end{cases}

    ``x`` and ``kappa`` are broadcast against each other.
    """
    kappa = clamp_kappa(kappa)
    sqrt_abs_k = _safe_abs_sqrt(kappa)
    scaled = x * sqrt_abs_k
    inv = sqrt_abs_k.reciprocal()

    spherical = inv * scaled.clamp(-_TAN_LIMIT, _TAN_LIMIT).tan()
    hyperbolic = inv * scaled.clamp(-_TANH_LIMIT, _TANH_LIMIT).tanh()
    euclidean = _tan_k_taylor(x, kappa)

    nonzero = torch.where(kappa > 0, spherical, hyperbolic)
    return torch.where(kappa.abs() < TAYLOR_EPS, euclidean, nonzero)


def artan_k(x: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    r"""Curvature-aware arctangent, the inverse of :func:`tan_k`.

    .. math::
        \tan^{-1}_\kappa(x) = \begin{cases}
            \kappa^{-1/2}\arctan(\sqrt{\kappa}\,x) & \kappa > 0\\
            x                                      & \kappa = 0\\
            |\kappa|^{-1/2}\operatorname{artanh}(\sqrt{|\kappa|}\,x) & \kappa < 0
        \end{cases}
    """
    kappa = clamp_kappa(kappa)
    sqrt_abs_k = _safe_abs_sqrt(kappa)
    scaled = x * sqrt_abs_k
    inv = sqrt_abs_k.reciprocal()

    spherical = inv * scaled.atan()
    clamped = scaled.clamp(-_ARTANH_LIMIT, _ARTANH_LIMIT)
    hyperbolic = inv * 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))
    euclidean = _artan_k_taylor(x, kappa)

    nonzero = torch.where(kappa > 0, spherical, hyperbolic)
    return torch.where(kappa.abs() < TAYLOR_EPS, euclidean, nonzero)


def projx(
    x: torch.Tensor, kappa: torch.Tensor, dim: int = -1, eps: float | None = None
) -> torch.Tensor:
    r"""Project ``x`` inside the chart.

    For :math:`\kappa < 0` the chart is the open ball of radius
    :math:`1/\sqrt{-\kappa}`; points are radially shrunk to
    :math:`(1-\varepsilon)/\sqrt{-\kappa}`. For :math:`\kappa \ge 0` the chart is
    all of :math:`\mathbb{R}^n` and this is (numerically) the identity.
    """
    kappa = clamp_kappa(kappa)
    if eps is None:
        eps = _BALL_EPS_FP64 if x.dtype == torch.float64 else _BALL_EPS_FP32
    sqrt_abs_k = _safe_abs_sqrt(kappa)
    maxnorm = (1.0 - eps) / sqrt_abs_k
    maxnorm = torch.where(kappa < 0, maxnorm, torch.full_like(maxnorm, 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(_NORM_EPS)
    projected = x / norm * maxnorm
    return torch.where(norm > maxnorm, projected, x)


def expmap0(u: torch.Tensor, kappa: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Exponential map at the origin: :math:`T_0\mathcal{M} \to \mathcal{M}`.

    .. math:: \exp^\kappa_0(u) = \tan_\kappa(\|u\|)\frac{u}{\|u\|}
    """
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(_NORM_EPS)
    return tan_k(u_norm, kappa) * (u / u_norm)


def logmap0(y: torch.Tensor, kappa: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""Logarithmic map at the origin: :math:`\mathcal{M} \to T_0\mathcal{M}`.

    .. math:: \log^\kappa_0(y) = \tan^{-1}_\kappa(\|y\|)\frac{y}{\|y\|}
    """
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(_NORM_EPS)
    return (y / y_norm) * artan_k(y_norm, kappa)


def dist0(
    x: torch.Tensor, kappa: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    r"""Geodesic distance to the origin: :math:`2\tan^{-1}_\kappa(\|x\|)`."""
    return 2.0 * artan_k(x.norm(dim=dim, p=2, keepdim=keepdim), kappa)


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    r"""Gyro-addition :math:`x \oplus_\kappa y` in the stereographic model.

    .. math::
        x \oplus_\kappa y = \frac{(1 - 2\kappa\langle x,y\rangle - \kappa\|y\|^2)x
        + (1 + \kappa\|x\|^2)y}{1 - 2\kappa\langle x,y\rangle
        + \kappa^2\|x\|^2\|y\|^2}
    """
    kappa = clamp_kappa(kappa)
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * kappa * xy - kappa * y2) * x + (1 + kappa * x2) * y
    denom = 1 - 2 * kappa * xy + kappa**2 * x2 * y2
    return num / denom.clamp_min(_NORM_EPS)


def dist_k(
    x: torch.Tensor, y: torch.Tensor, kappa: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    r"""Pairwise geodesic distance :math:`d_\kappa(x, y) = 2\tan^{-1}_\kappa(\|(-x)\oplus_\kappa y\|)`.

    Reduces to :math:`2\|x - y\|` at :math:`\kappa = 0` (the stereographic
    conformal factor is 2 at the origin — the same convention geoopt uses).
    """
    kappa = clamp_kappa(kappa)
    return dist0(mobius_add(-x, y, kappa, dim=dim), kappa, dim=dim, keepdim=keepdim)


def dist_k_from_cos(cos: torch.Tensor, kappa: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    r"""``dist_k`` between two points of common norm ``radius`` and cosine ``cos``.

    Scalar reduction for the E3 attention kernel: when :math:`\|x\| = \|y\| = r`,
    :math:`\|(-x)\oplus_\kappa y\|` is a rational function of :math:`(c, r^2)`
    alone, so pairwise distances need only the ``(T, T)`` cosine matrix — never
    a ``(T, T, d)`` Möbius broadcast. With :math:`a = 1 + 2\kappa r^2 c -
    \kappa r^2`, :math:`b = 1 + \kappa r^2`:

    .. math::
        \|(-x)\oplus_\kappa y\| = \frac{r\sqrt{a^2 + b^2 - 2abc}}
        {1 + 2\kappa r^2 c + \kappa^2 r^4}
    """
    kappa = clamp_kappa(kappa)
    c = cos.clamp(-1.0, 1.0)
    r2 = radius * radius
    kr2 = kappa * r2
    a = 1 + 2 * kr2 * c - kr2
    b = 1 + kr2
    # a² + b² − 2abc is exactly 0 at c = 1 (x = y); the floor keeps sqrt's
    # gradient finite on the attention diagonal without moving off-diagonals.
    gyro_norm = radius * (a * a + b * b - 2 * a * b * c).clamp_min(_NORM_EPS).sqrt()
    denom = (1 + 2 * kr2 * c + kr2 * kr2).clamp_min(_NORM_EPS)
    return 2.0 * artan_k(gyro_norm / denom, kappa)
