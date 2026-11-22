"""Tests for :class:`GatedKappaNorm` - the "MoE over geometries" layer."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch

from entropic_substrate.core.geometry.kappa import KAPPA_MAX, KAPPA_MIN
from entropic_substrate.core.substrate.blocks import GatedKappaNorm, UnitNorm

pytestmark = pytest.mark.unit


@pytest.fixture
def layer() -> GatedKappaNorm:
    torch.manual_seed(0)
    return GatedKappaNorm(32, n_components=3, top_k=1)


def test_unit_norm_retraction() -> None:
    x = torch.randn(4, 6, 32) * 7.0
    y = UnitNorm()(x)
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(4, 6), atol=1e-6, rtol=0)


def test_default_kappa_init_spans_the_three_geometries() -> None:
    layer = GatedKappaNorm(16, n_components=3)
    assert layer.stats.kappa == [-1.0, 0.0, 1.0]


@pytest.mark.parametrize("top_k", [1, 2])
def test_forward_shape_and_unit_output(top_k: int) -> None:
    layer = GatedKappaNorm(32, n_components=3, top_k=top_k)
    x = torch.randn(2, 5, 32)
    y = layer(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(2, 5), atol=1e-5, rtol=0)


def test_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError, match="top_k must be 1 or 2"):
        GatedKappaNorm(8, top_k=3)
    with pytest.raises(ValueError, match="exceeds n_components"):
        GatedKappaNorm(8, n_components=1, top_k=2)
    with pytest.raises(ValueError, match="gate_pool"):
        GatedKappaNorm(8, gate_pool="global")
    with pytest.raises(ValueError, match="kappa_init"):
        GatedKappaNorm(8, n_components=3, kappa_init=[0.0])


def test_geometry_runs_in_fp32_even_under_autocast(layer: GatedKappaNorm) -> None:
    """All manifold math must be FP32 inside an autocast-disabled region."""
    seen: list[torch.dtype] = []
    autocast_enabled: list[bool] = []

    original = layer.gate.forward

    def spy(inp: torch.Tensor) -> torch.Tensor:
        seen.append(inp.dtype)
        autocast_enabled.append(torch.is_autocast_enabled("cpu"))
        return original(inp)

    layer.gate.forward = spy  # type: ignore[method-assign]
    x = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        y = layer(x)

    assert seen == [torch.float32]
    assert autocast_enabled == [False]
    # Output is cast back to the surrounding dtype so the block stays bf16.
    assert y.dtype == torch.bfloat16


def test_aux_loss_is_finite_positive_and_differentiable(layer: GatedKappaNorm) -> None:
    x = torch.randn(3, 7, 32, requires_grad=True)
    layer(x)
    aux = layer.aux_loss
    assert torch.isfinite(aux)
    assert float(aux.detach()) > 0.0
    assert aux.requires_grad
    aux.backward()
    assert layer.gate.weight.grad is not None
    assert torch.isfinite(layer.gate.weight.grad).all()


def test_balanced_router_hits_the_switch_lower_bound() -> None:
    """For K experts, `K * sum_i f_i P_i` is 1.0 when routing is uniform."""
    layer = GatedKappaNorm(16, n_components=4, top_k=1, aux_loss_weight=1.0)
    with torch.no_grad():
        layer.gate.weight.zero_()  # uniform probabilities, deterministic argmax
    layer(torch.randn(8, 8, 16))
    # Uniform probs => P_i = 1/K; argmax ties resolve to expert 0 => f = [K,0,0,0]
    # after the K/top_k scaling, so the loss saturates at its maximum of 1.0.
    assert float(layer.aux_loss.detach()) == pytest.approx(1.0, abs=1e-5)


def test_kappa_norm_is_radially_annihilated() -> None:
    """Design record 2026-08-11 (docs/e-3-instruction.md): origin log/exp maps
    are radial and the output retraction erases radius, so this layer IS
    ``UnitNorm`` to float precision and kappa is unlearnable here — curvature
    is relational and moved to the attention kernel (E3). This pin documents
    the theorem; if it ever fails, the layer regained a non-radial path and
    the design record must be revisited before trusting either result."""
    torch.manual_seed(0)
    x = torch.randn(4, 6, 16)
    unit = UnitNorm()(x)
    for n_components, top_k, init in [
        (1, 1, [0.0]),
        (3, 1, None),
        (3, 2, [-1.0, 0.0, 1.0]),
    ]:
        layer = GatedKappaNorm(16, n_components=n_components, top_k=top_k, kappa_init=init)
        out = layer(x)
        torch.testing.assert_close(out, unit, atol=1e-6, rtol=0)
        out.square().sum().backward()
        assert layer.kappa.grad is not None
        assert torch.isfinite(layer.kappa.grad).all()
        # Exactly zero for top-1; float residue (~1e-7) at most for top-2.
        assert float(layer.kappa.grad.abs().sum()) < 1e-5


def test_stats_are_populated_and_normalised(layer: GatedKappaNorm) -> None:
    layer(torch.randn(2, 9, 32))
    stats = layer.stats
    assert len(stats.kappa) == 3
    assert len(stats.gate_fraction) == 3
    assert len(stats.router_prob) == 3
    assert sum(stats.gate_fraction) == pytest.approx(1.0, abs=1e-5)
    assert sum(stats.router_prob) == pytest.approx(1.0, abs=1e-5)
    assert 0.0 <= stats.entropy <= math.log(3.0) + 1e-5


def test_top2_routing_dispatches_two_experts_per_token() -> None:
    layer = GatedKappaNorm(16, n_components=3, top_k=2)
    layer(torch.randn(4, 4, 16))
    # Each token occupies 2 of 3 slots => fractions sum to 2.
    assert sum(layer.stats.gate_fraction) == pytest.approx(2.0, abs=1e-5)


def test_curvature_stays_clamped_after_a_wild_update(layer: GatedKappaNorm) -> None:
    with torch.no_grad():
        layer.kappa.copy_(torch.tensor([-500.0, 0.0, 500.0]))
    y = layer(torch.randn(2, 3, 32))
    assert torch.isfinite(y).all()
    used = layer.current_kappa()
    assert float(used.min()) == KAPPA_MIN
    assert float(used.max()) == KAPPA_MAX


def test_output_is_finite_for_pathological_inputs(layer: GatedKappaNorm) -> None:
    for scale in (1e-8, 1.0, 1e6):
        y = layer(torch.randn(2, 4, 32) * scale)
        assert torch.isfinite(y).all(), scale


def test_sequence_pooled_gate_routes_all_tokens_together() -> None:
    layer = GatedKappaNorm(16, n_components=3, top_k=1, gate_pool="sequence")
    layer(torch.randn(1, 12, 16))
    # A single pooled routing decision for the whole sequence.
    assert max(layer.stats.gate_fraction) == pytest.approx(1.0, abs=1e-5)


def test_norm_family_lives_beside_build_norm_with_no_forwarding_module() -> None:
    """FOR-1597: the geometry package's ``kappa_norm`` is deleted, never a re-export shim."""
    geometry = importlib.import_module("entropic_substrate.core.geometry")
    assert importlib.util.find_spec(f"{geometry.__name__}.kappa_norm") is None


def test_state_dict_keys_do_not_depend_on_the_defining_module() -> None:
    """Checkpoints address parameters by attribute path, not by import path."""
    assert list(GatedKappaNorm(8).state_dict()) == ["kappa", "gate.weight"]
    assert list(UnitNorm().state_dict()) == []
