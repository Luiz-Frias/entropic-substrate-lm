"""Direct contracts for the shared optimizer and evaluation transaction."""

import numpy as np
import pytest
import torch

from entropic_substrate.composition.wiring.d_ladder_cli import tiny_rung_spec
from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.historical.d_ladder.d_ladder_training import (
    evaluate_ladder,
    gradient_measurements,
    ladder_optimizer,
    train_ladder_step,
)
from entropic_substrate.ladder.run.construction.ladder_runtime import LadderPredictiveSystem

pytestmark = pytest.mark.unit


def system_and_sampler(experiment="d2-l", *, train=True):
    system = LadderPredictiveSystem(tiny_rung_spec(experiment, seed=7))
    sampler = SequenceSampler(
        TokenSplit(np.arange(512, dtype=np.int64) % 256), 2, 8, seed=7, train=train
    )
    return system, sampler, system.spec.base.execution_protocol


def test_measured_zero_nonzero_and_structural_absence_remain_distinct():
    system, _, _ = system_and_sampler("d2-s")
    for parameter in system.optimizer_parameters():
        parameter.grad = torch.zeros_like(parameter)
    measured = gradient_measurements(system)
    assert measured["target_branch"] == {"status": "stop_gradient_target", "norm": None}
    assert all(
        value["status"] == "measured_zero"
        for key, value in measured.items()
        if key != "target_branch"
    )
    system.optimizer_parameters()[0].grad.fill_(1)
    assert any(
        value["status"] == "measured_nonzero" for value in gradient_measurements(system).values()
    )


@pytest.mark.parametrize("missing", [True, False])
def test_missing_and_nonfinite_required_gradients_are_defects(missing):
    system, _, _ = system_and_sampler()
    for parameter in system.optimizer_parameters():
        parameter.grad = torch.zeros_like(parameter)
    parameter = system.optimizer_parameters()[0]
    parameter.grad = None if missing else torch.full_like(parameter, torch.nan)
    with pytest.raises(ValueError, match="missing" if missing else "nonfinite"):
        gradient_measurements(system)


@pytest.mark.parametrize("corrupt_total", [False, True])
def test_zero_transaction_proceeds_but_inconsistent_aggregate_stops(monkeypatch, corrupt_total):
    system, sampler, protocol = system_and_sampler()
    optimizer = ladder_optimizer(system, protocol)
    events = []
    real_step = optimizer.step

    def step(*args, **kwargs):
        events.append("optimizer")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", step)
    monkeypatch.setattr(
        system, "after_successful_optimizer_step", lambda event: events.append("after_step")
    )
    handles = [
        parameter.register_hook(torch.zeros_like) for parameter in system.optimizer_parameters()
    ]
    if corrupt_total:
        clip = torch.nn.utils.clip_grad_norm_
        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **kw: clip(*a, **kw) + 1)
    try:
        if corrupt_total:
            with pytest.raises(ValueError, match="aggregate"):
                train_ladder_step(system, optimizer, sampler, protocol, 1, None)
            assert events == [] and not optimizer.state
        else:
            measured, online, target = train_ladder_step(
                system, optimizer, sampler, protocol, 1, None
            )
            assert events == ["optimizer", "after_step"] and optimizer.state
            assert (online, target) == (16, 0)
            assert measured["online_branch"] == {"status": "measured_zero", "norm": 0.0}
    finally:
        for handle in handles:
            handle.remove()


def test_wrong_optimizer_owner_is_rejected_before_sampling():
    system, sampler, protocol = system_and_sampler()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    with pytest.raises(ValueError, match="own every and only"):
        train_ladder_step(system, optimizer, sampler, protocol, 1, None)
    assert not optimizer.state


def test_collapsed_online_state_retains_zero_rank_and_variance():
    system, sampler, protocol = system_and_sampler(train=False)
    with torch.no_grad():
        for parameter in system.core.online_branch.parameters():
            parameter.zero_()
    metrics, resources = evaluate_ladder(system, sampler, protocol, None, 1)
    assert metrics["online_numerical_rank"] == metrics["online_mean_variance"] == 0
    assert metrics["online_zero_norm_positions"] == 16 * protocol.evaluation.num_batches
    assert resources is None
