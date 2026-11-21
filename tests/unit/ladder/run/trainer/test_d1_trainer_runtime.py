"""Fresh D1 composition-root and construction-order contracts."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest
import torch

import entropic_substrate.ladder.run.trainer.d1_trainer_runtime as runtime_module
from d1_trainer_fixtures import ORIGIN_NS, STARTED_AT, micro_plan, run_start
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import (
    D1TrainerWallOrigin,
    build_d1_fresh_components,
    create_d1_resource_ledger,
)

pytestmark = pytest.mark.unit


def _origin() -> D1TrainerWallOrigin:
    return D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS)


@pytest.mark.parametrize("arm", list(D1Arm))
def test_plan_builds_every_arm_on_cpu_without_trainer_arm_branches(arm: D1Arm) -> None:
    plan = micro_plan(arm)
    start = run_start(plan)
    ledger = create_d1_resource_ledger(plan=plan, start=start, origin=_origin())

    components = build_d1_fresh_components(
        plan=plan,
        start=start,
        origin=_origin(),
        resource_ledger=ledger,
    )

    assert components.plan is plan
    assert components.device == torch.device("cpu")
    assert components.system.draft == plan.profile.draft
    assert components.system.initialization_attestation.seed == plan.seed
    assert components.train_sampler.train is True
    assert components.eval_sampler.train is False
    assert components.train_sampler.generator.initial_seed() == plan.seed
    assert components.corpus.meta.tokenizer == "byte"
    assert components.corpus.meta.vocab_size == 256


def test_wall_origin_and_fresh_segment_are_admitted_before_construction() -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    invalid = replace(_origin(), started_at=STARTED_AT + timedelta(seconds=1))
    with pytest.raises(ValueError, match="trainer_started_at"):
        create_d1_resource_ledger(plan=plan, start=start, origin=invalid)
    with pytest.raises(ValueError, match="monotonic"):
        D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=-1)

    with pytest.raises(ValueError, match="logical_run_id"):
        create_d1_resource_ledger(
            plan=plan,
            start=replace(start, logical_run_id="wrong", execution_segment_id="wrong:segment:0"),
            origin=_origin(),
        )


def test_fresh_construction_binds_the_shared_model_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = micro_plan(D1Arm.SG)
    start = run_start(plan)
    real_materialize = runtime_module.materialize_d1_model_runtime
    observed: list[object] = []

    def record_materialization(candidate: object):
        observed.append(candidate)
        return real_materialize(candidate)  # type: ignore[arg-type]

    monkeypatch.setattr(runtime_module, "materialize_d1_model_runtime", record_materialization)
    components = build_d1_fresh_components(
        plan=plan,
        start=start,
        origin=_origin(),
        resource_ledger=create_d1_resource_ledger(plan=plan, start=start, origin=_origin()),
    )

    assert observed == [plan]
    assert components.system.draft == plan.profile.draft


def test_online_state_is_blind_to_changed_future_target_after_plan_composition() -> None:
    plan = micro_plan(D1Arm.BOTH)
    start = run_start(plan)
    components = build_d1_fresh_components(
        plan=plan,
        start=start,
        origin=_origin(),
        resource_ledger=create_d1_resource_ledger(plan=plan, start=start, origin=_origin()),
    )
    batch = components.train_sampler.predictive_batch(horizon=1)
    changed_context = batch.target_encoder_context.clone()
    changed_context[:, -1] = (changed_context[:, -1] + 31) % 256
    changed = replace(batch, target_encoder_context=changed_context)

    first = components.system.online_branch(
        batch.online_observations, components.system.draft.codec
    ).values
    second = components.system.online_branch(
        changed.online_observations, components.system.draft.codec
    ).values

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
