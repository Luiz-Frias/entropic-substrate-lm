"""Registered D1 AdamW materialization and optimizer ownership evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from d1_runtime_fixtures import runtime_system
from entropic_substrate.kernel.identity.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.ladder.run.trainer.d1_optimizer import build_d1_optimizer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("arm", list(D1Arm))
def test_optimizer_materializes_the_registered_protocol_and_exact_ownership(arm: D1Arm) -> None:
    system = runtime_system(arm)
    spec = system.draft.execution_protocol.optimization

    bundle = build_d1_optimizer(system, spec)

    system.validate_optimizer_ownership(bundle.optimizer)
    assert bundle.spec_digest == canonical_digest(spec.manifest)
    assert bundle.ownership_digest == canonical_digest(bundle.ownership_manifest)
    assert bundle.ownership_manifest["weight_decay_policy"] == "matrix_only"
    assert bundle.ownership_manifest["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in system.optimizer_parameters()
    )
    assert bundle.optimizer.defaults["lr"] == spec.learning_rate
    assert bundle.optimizer.defaults["betas"] == spec.betas
    assert bundle.optimizer.defaults["eps"] == spec.epsilon


def test_matrix_only_weight_decay_groups_are_complete_disjoint_and_exact() -> None:
    system = runtime_system(D1Arm.NONE)
    spec = system.draft.execution_protocol.optimization
    bundle = build_d1_optimizer(system, spec)
    groups = bundle.optimizer.param_groups

    decay = set(groups[0]["params"])
    no_decay = set(groups[1]["params"])
    expected = set(system.optimizer_parameters())
    assert decay.isdisjoint(no_decay)
    assert decay | no_decay == expected
    assert all(parameter.ndim >= 2 for parameter in decay)
    assert all(parameter.ndim < 2 for parameter in no_decay)
    assert groups[0]["weight_decay"] == spec.weight_decay
    assert groups[1]["weight_decay"] == 0.0


def test_stop_gradient_teacher_is_absent_from_optimizer_and_ownership_manifest() -> None:
    system = runtime_system(D1Arm.SG)

    bundle = build_d1_optimizer(system, system.draft.execution_protocol.optimization)

    target_names = {f"target_branch.{name}" for name, _ in system.target_branch.named_parameters()}
    owned_names = set(bundle.ownership_manifest["decay_parameters"]) | set(
        bundle.ownership_manifest["no_decay_parameters"]
    )
    assert target_names.isdisjoint(owned_names)
    system.target_policy.validate_optimizer_excludes_target(bundle.optimizer, system.target_branch)


def test_optimizer_factory_refuses_protocol_drift_from_the_runtime_construction() -> None:
    system = runtime_system(D1Arm.NONE)
    changed = replace(
        system.draft.execution_protocol.optimization,
        learning_rate=3e-4,
    )

    with pytest.raises(ValueError, match="does not match"):
        build_d1_optimizer(system, changed)


def test_learning_rate_update_changes_every_group_without_changing_ownership() -> None:
    system = runtime_system(D1Arm.BOTH)
    bundle = build_d1_optimizer(system, system.draft.execution_protocol.optimization)
    before = bundle.ownership_digest

    bundle.set_learning_rate(1.25e-4)

    assert {group["lr"] for group in bundle.optimizer.param_groups} == {1.25e-4}
    assert bundle.ownership_digest == before


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_learning_rate_update_refuses_invalid_values(value: object) -> None:
    system = runtime_system(D1Arm.AC)
    bundle = build_d1_optimizer(system, system.draft.execution_protocol.optimization)

    with pytest.raises(ValueError, match="learning rate"):
        bundle.set_learning_rate(value)  # type: ignore[arg-type]
