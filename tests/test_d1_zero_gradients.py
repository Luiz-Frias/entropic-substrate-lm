"""A connected finite stationary state is evidence, not a broken backward graph."""

from dataclasses import replace

import pytest
import torch

from d1_runtime_fixtures import fit_runtime_baseline, runtime_batch, runtime_system
from entropic_substrate.d1_optimizer import build_d1_optimizer
from entropic_substrate.d1_training_step import (
    D1ComponentGradientEvidence,
    D1GradientEvidenceKind,
    apply_d1_training_step,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def test_new_gradient_semantics_have_a_distinct_construction_epoch() -> None:
    from entropic_substrate.experiments.d1_primitives import CONSTRUCTION_SCHEMA

    assert CONSTRUCTION_SCHEMA == "d1-construction-v4"


def test_finite_collapsed_target_is_retained_after_a_successful_step() -> None:
    system = runtime_system(D1Arm.NONE)
    with torch.no_grad():
        for component in (system.target_branch, system.future_law):
            for parameter in component.parameters():
                parameter.zero_()
    fit_runtime_baseline(system)
    spec = system.draft.execution_protocol.optimization
    bundle = build_d1_optimizer(system, spec)
    result = apply_d1_training_step(
        system=system,
        optimizer_bundle=bundle,
        batch=runtime_batch(),
        spec=spec,
        successful_step=0,
    )
    assert result.successful_step == 1
    assert result.future_nll == result.zero_displacement_nll
    assert result.effective_rank > 0.0
    for component in (
        result.gradient_reachability.target_branch,
        result.gradient_reachability.future_law,
    ):
        assert component.kind.value == "measured_zero"
        assert component.pre_clip_l2_norm == 0.0
        assert component.manifest == {"kind": "measured_zero", "pre_clip_l2_norm": 0.0}
    assert bundle.optimizer.state


def test_zero_evidence_cannot_launder_nonzero_absent_or_nonfinite_norms() -> None:
    zero = D1ComponentGradientEvidence(D1GradientEvidenceKind("measured_zero"), 0.0)
    for norm in (None, 1.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(zero, pre_clip_l2_norm=norm)
    with pytest.raises(ValueError):
        D1ComponentGradientEvidence(D1GradientEvidenceKind.MEASURED_NONZERO, 0.0)


def test_zero_connected_gradients_do_not_skip_ema_or_adamw() -> None:
    system = runtime_system(D1Arm.BOTH)
    fit_runtime_baseline(system)
    spec = system.draft.execution_protocol.optimization
    bundle = build_d1_optimizer(system, spec)
    handles = [p.register_hook(torch.zeros_like) for p in system.optimizer_parameters()]
    before = next(system.online_branch.parameters()).detach().clone()
    try:
        result = apply_d1_training_step(
            system=system,
            optimizer_bundle=bundle,
            batch=runtime_batch(),
            spec=spec,
            successful_step=0,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert result.gradient_norm == 0.0
    assert result.gradient_reachability.online_branch.kind.value == "measured_zero"
    assert result.gradient_reachability.target_branch.kind is (
        D1GradientEvidenceKind.STRUCTURALLY_ABSENT_STOP_GRADIENT
    )
    assert system.target_policy_state()["update_cursor"] == 1
    assert not torch.equal(before, next(system.online_branch.parameters()))
