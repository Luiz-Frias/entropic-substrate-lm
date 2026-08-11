"""Constant-curvature (kappa-stereographic) geometry primitives."""

from __future__ import annotations

from entropic_substrate.geometry.kappa import (
    KAPPA_MAX,
    KAPPA_MIN,
    artan_k,
    clamp_kappa,
    dist0,
    expmap0,
    logmap0,
    mobius_add,
    projx,
    tan_k,
)

__all__ = [
    "KAPPA_MAX",
    "KAPPA_MIN",
    "artan_k",
    "clamp_kappa",
    "dist0",
    "expmap0",
    "logmap0",
    "mobius_add",
    "projx",
    "tan_k",
]
