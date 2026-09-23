"""RED-first guard-teeth for SI-1 ("all geometric math runs in FP32 inside
``torch.amp.autocast(enabled=False)`` blocks regardless of surrounding bf16
autocast", CLAUDE.md Non-negotiable numerics; p6-core brief SI-1).

``entropic_substrate.core.geometry.kappa`` carries exactly TWO independent
FP32 islands, ``dist_k_from_cos`` and ``dist_k_from_inner``, and they NEST:
``dist_k_from_cos``'s island body calls ``dist_k_from_inner``, which opens
its own island. A spy exercised only through ``dist_k_from_cos`` is
RED-blind to half the mutation space -- removing ``dist_k_from_inner``'s own
wrapper would still read "autocast disabled" because the OUTER wrapper
(``dist_k_from_cos``'s) is still in effect. Two independent spies are
installed here, one per island, each substituting a module-global callable
the island calls from INSIDE its own ``with torch.autocast(..., enabled=
False)`` block (the ``tests/test_kappa_norm.py::
test_geometry_runs_in_fp32_even_under_autocast`` spy pattern). The
``dist_k_from_inner`` arm calls that function DIRECTLY (never through
``dist_k_from_cos``) so its own wrapper is the outermost one on that call
path -- otherwise removing it would be invisible to the same reason.

The load-bearing arm is the autocast-STATE spy, never a dtype assertion:
``_geometry_dtype()`` floors float16/bfloat16 operands to float32
unconditionally, with or without autocast disabled, so an
``inner.dtype == torch.float32`` assertion (the shape
``tests/test_kappa_geometry.py::
test_scalar_reductions_run_in_fp32_under_bf16_autocast`` uses) survives
wrapper removal and is measured VACUOUS here (FOR-1517). This file never
reports a dtype arm as SI-1 teeth.
"""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.geometry import kappa as kappa_module

pytestmark = pytest.mark.unit


def test_dist_k_from_cos_island_disables_autocast_before_calling_inner() -> None:
    """Mutation: remove the ``with torch.autocast(..., enabled=False)``
    wrapper inside ``dist_k_from_cos`` -> this test goes RED (the spy then
    observes autocast enabled)."""
    seen: list[bool] = []
    original = kappa_module.dist_k_from_inner

    def spy(dot: torch.Tensor, x_sqnorm: torch.Tensor, y_sqnorm: torch.Tensor, kap: torch.Tensor):
        seen.append(torch.is_autocast_enabled("cpu"))
        return original(dot, x_sqnorm, y_sqnorm, kap)

    kappa_module.dist_k_from_inner = spy  # type: ignore[attr-defined]
    try:
        cos = torch.tensor([0.3, -0.2], dtype=torch.bfloat16)
        kap = torch.tensor([0.5, -1.5], dtype=torch.bfloat16)
        radius = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")  # control: surrounding autocast IS on
            kappa_module.dist_k_from_cos(cos, kap, radius)
    finally:
        kappa_module.dist_k_from_inner = original  # type: ignore[attr-defined]

    assert seen == [False], (
        "dist_k_from_cos's FP32 island must disable autocast before calling "
        f"dist_k_from_inner; observed autocast-enabled state(s): {seen}"
    )


def test_dist_k_from_inner_island_disables_autocast_called_directly() -> None:
    """Mutation: remove the ``with torch.autocast(..., enabled=False)``
    wrapper inside ``dist_k_from_inner`` -> this test goes RED. Calls
    ``dist_k_from_inner`` DIRECTLY (not through ``dist_k_from_cos``) so this
    island's own wrapper -- not the caller's -- is what the spy observes.

    ``clamp_kappa`` is called TWICE inside the island's body: once directly,
    once again from inside ``artan_k`` (which itself calls ``clamp_kappa``)
    -- both calls happen while still inside the same ``with`` block, so the
    control asserts every observed state is False, not that there is
    exactly one."""
    seen: list[bool] = []
    original = kappa_module.clamp_kappa

    def spy(kap: torch.Tensor) -> torch.Tensor:
        seen.append(torch.is_autocast_enabled("cpu"))
        return original(kap)

    kappa_module.clamp_kappa = spy  # type: ignore[attr-defined]
    try:
        dot = torch.tensor([0.1, -0.05], dtype=torch.bfloat16)
        x_sq = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        y_sq = torch.tensor([1.0, 1.0], dtype=torch.bfloat16)
        kap = torch.tensor([0.5, -1.5], dtype=torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu")  # control: surrounding autocast IS on
            kappa_module.dist_k_from_inner(dot, x_sq, y_sq, kap)
    finally:
        kappa_module.clamp_kappa = original  # type: ignore[attr-defined]

    assert seen, "clamp_kappa spy was never called -- the island body did not run"
    assert all(state is False for state in seen), (
        "dist_k_from_inner's own FP32 island must disable autocast; observed "
        f"autocast-enabled state(s): {seen}"
    )
