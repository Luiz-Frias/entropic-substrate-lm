"""Streaming, deterministic evaluation for the D1 predictive-state system."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from d1_runtime_fixtures import fit_runtime_baseline, runtime_family
from entropic_substrate.d1_evaluation import D1Evaluator
from entropic_substrate.data.batching import SequenceSampler, TokenSplit
from entropic_substrate.experiments.d1_execution import D1AmpPolicy
from entropic_substrate.experiments.d1_family import build_d1_experiment_family
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

pytestmark = pytest.mark.unit


def _system(*, amp: D1AmpPolicy = D1AmpPolicy.OFF) -> D1PredictiveSystem:
    template = runtime_family()
    draft = template.drafts[0]
    protocol = replace(
        draft.execution_protocol,
        optimization=replace(draft.execution_protocol.optimization, amp=amp),
        batching=replace(
            draft.execution_protocol.batching,
            batch_size=2,
            sequence_length=4,
        ),
        evaluation=replace(draft.execution_protocol.evaluation, num_batches=3),
    )
    family = build_d1_experiment_family(
        codec=draft.codec,
        encoder_pair=draft.encoder_pair,
        execution_protocol=protocol,
    )
    system = D1PredictiveSystem.from_family(
        family,
        D1Arm.BOTH,
        initializer=SeededD1BranchInitializer(seed=7),
    )
    fit_runtime_baseline(system)
    return system


def _sampler() -> SequenceSampler:
    split = TokenSplit(np.arange(256, dtype=np.int32) % 11)
    return SequenceSampler(split, batch_size=2, seq_len=4, train=False)


def _materialize(system: D1PredictiveSystem, sampler: SequenceSampler) -> torch.Tensor:
    sampler.reset_eval()
    states: list[torch.Tensor] = []
    was_training = system.training
    system.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            for _ in range(3):
                result = system(sampler.predictive_batch(horizon=1))
                states.append(result.online_states.values.float().reshape(-1, 8))
    finally:
        system.train(was_training)
    return torch.cat(states)


def test_evaluation_is_repeatable_count_weighted_and_restores_training_mode() -> None:
    system = _system()
    sampler = _sampler()
    system.train()
    sampler.predictive_batch(horizon=1)
    observed_grad_modes: list[bool] = []
    handle = system.register_forward_pre_hook(
        lambda _module, _args: observed_grad_modes.append(torch.is_grad_enabled())
    )
    try:
        first = D1Evaluator().evaluate(system=system, sampler=sampler)
        second = D1Evaluator().evaluate(system=system, sampler=sampler)
    finally:
        handle.remove()

    assert first == second
    assert system.training is True
    assert observed_grad_modes == [False] * 6
    assert first.batch_count == 3
    assert first.online_positions == 24
    assert first.residual_elements == 24 * 8
    assert first.nats_per_byte == first.language_loss
    assert first.bits_per_byte == pytest.approx(first.nats_per_byte / math.log(2.0))
    assert first.perplexity == pytest.approx(math.exp(first.language_loss))
    assert first.zero_baseline_improvement == pytest.approx(
        first.zero_displacement_nll - first.future_nll
    )
    assert first.mean_baseline_improvement == pytest.approx(
        first.mean_displacement_nll - first.future_nll
    )


def test_global_covariance_rank_and_pairwise_cosine_match_materialized_oracle() -> None:
    system = _system()
    sampler = _sampler()
    result = D1Evaluator().evaluate(system=system, sampler=sampler)
    states = _materialize(system, sampler)
    covariance = torch.cov(states.T, correction=1)
    normalized = functional.normalize(states, dim=1)
    similarities = normalized @ normalized.T
    distinct = ~torch.eye(states.shape[0], dtype=torch.bool)
    pairwise_distance = 1.0 - similarities[distinct].mean()

    torch.testing.assert_close(torch.tensor(result.state_sample_covariance), covariance)
    torch.testing.assert_close(torch.tensor(result.state_variance), covariance.diagonal())
    assert result.numerical_rank == int(torch.linalg.matrix_rank(covariance).item())
    assert result.mean_pairwise_cosine_distance == pytest.approx(float(pairwise_distance), abs=1e-6)


@pytest.mark.parametrize(
    ("amp", "expected"),
    [(D1AmpPolicy.OFF, False), (D1AmpPolicy.BF16, True)],
)
def test_evaluation_honors_registered_autocast_policy(amp: D1AmpPolicy, expected: bool) -> None:
    system = _system(amp=amp)
    observed: list[bool] = []
    handle = system.register_forward_pre_hook(
        lambda _module, _args: observed.append(torch.is_autocast_enabled("cpu"))
    )
    try:
        D1Evaluator().evaluate(system=system, sampler=_sampler())
    finally:
        handle.remove()
    assert observed == [expected] * 3


def test_result_is_immutable_and_evaluation_fails_loud_on_invalid_evidence() -> None:
    system = _system()
    result = D1Evaluator().evaluate(system=system, sampler=_sampler())
    with pytest.raises(FrozenInstanceError):
        result.batch_count = 4  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite"):
        replace(result, future_nll=float("nan"))

    with pytest.raises(ValueError, match="evaluation sampler"):
        D1Evaluator().evaluate(
            system=system,
            sampler=SequenceSampler(
                TokenSplit(np.arange(256, dtype=np.int32) % 11),
                batch_size=2,
                seq_len=4,
                train=True,
            ),
        )

    def corrupt_state(_module: object, _args: object, output: object) -> object:
        output.online_states.values[0, 0, 0] = float("nan")  # type: ignore[attr-defined]
        return output

    handle = system.register_forward_hook(corrupt_state)
    try:
        with pytest.raises(ValueError, match="finite"):
            D1Evaluator().evaluate(system=system, sampler=_sampler())
    finally:
        handle.remove()
