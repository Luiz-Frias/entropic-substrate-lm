"""Streaming, deterministic evaluation for the D1 predictive-state system."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import torch

import entropic_substrate.ladder.run.trainer.d1_evaluation as evaluation_module
from d1_runtime_fixtures import fit_runtime_baseline, runtime_family
from entropic_substrate.core.corpus.batching import SequenceSampler, TokenSplit
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.register.d1_execution import D1AmpPolicy
from entropic_substrate.ladder.register.d1_family import build_d1_experiment_family
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1ChannelMoments, D1Evaluator

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


def _materialize(
    system: D1PredictiveSystem, sampler: SequenceSampler, channel: str = "state"
) -> torch.Tensor:
    sampler.reset_eval()
    states: list[torch.Tensor] = []
    was_training = system.training
    system.eval()
    try:
        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            for _ in range(3):
                result = system(sampler.predictive_batch(horizon=1))
                values = {
                    "state": result.online_states.values,
                    "target": result.target.values,
                    "predicted": result.future_law_loss.mean,
                }.get(channel)
                if values is None:
                    raise ValueError("unknown test channel")
                states.append(values.detach().float().reshape(-1, 8))
    finally:
        system.train(was_training)
    return torch.cat(states)


def _channel_oracle(values: torch.Tensor) -> D1ChannelMoments:
    covariance = torch.cov(values.T, correction=1)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    mass = eigenvalues.sum()
    effective_rank = 0.0
    if float(mass.item()) > 0.0:
        probabilities = eigenvalues / mass
        positive = probabilities > 0.0
        effective_rank = float(
            (-(probabilities[positive] * probabilities[positive].log()).sum()).exp().item()
        )
    return D1ChannelMoments(
        mean=tuple(float(value) for value in values.mean(dim=0).tolist()),
        variance=tuple(float(value) for value in covariance.diagonal().tolist()),
        covariance_eigenvalues=tuple(float(value) for value in eigenvalues.tolist()),
        effective_rank=effective_rank,
        numerical_rank=int(torch.linalg.matrix_rank(covariance).item()),
        rms_l2_norm=float(values.square().sum(dim=1).mean().sqrt().item()),
    )


def _assert_channel_close(observed: D1ChannelMoments, expected: D1ChannelMoments) -> None:
    for field in ("mean", "variance", "covariance_eigenvalues"):
        torch.testing.assert_close(
            torch.tensor(getattr(observed, field)),
            torch.tensor(getattr(expected, field)),
            rtol=1e-5,
            atol=1e-6,
        )
    assert observed.effective_rank == pytest.approx(expected.effective_rank, rel=1e-5)
    assert observed.rms_l2_norm == pytest.approx(expected.rms_l2_norm, rel=1e-5)


def _bf16_forward_states(system: D1PredictiveSystem, sampler: SequenceSampler) -> torch.Tensor:
    sampler.reset_eval()
    states: list[torch.Tensor] = []
    was_training = system.training
    system.eval()
    try:
        with torch.no_grad():
            for _ in range(3):
                batch = sampler.predictive_batch(horizon=1)
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    result = system(batch)
                states.append(result.online_states.values.detach().float().reshape(-1, 8))
    finally:
        system.train(was_training)
    return torch.cat(states)


def _fp32_state_oracle(
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, int, float]:
    with torch.autocast(device_type="cpu", enabled=False):
        values = states.float()
        covariance = torch.cov(values.T, correction=1)
        covariance = (covariance + covariance.T) * 0.5
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        probabilities = eigenvalues / eigenvalues.sum()
        positive = probabilities > 0.0
        effective_rank = float(
            (-(probabilities[positive] * probabilities[positive].log()).sum()).exp().item()
        )
        numerical_rank = int(torch.linalg.matrix_rank(covariance).item())
        norms = torch.linalg.vector_norm(values, dim=1)
        assert bool((norms > 0.0).all().item())
        similarities = (values / norms.unsqueeze(1)) @ (values / norms.unsqueeze(1)).T
        distinct = ~torch.eye(values.shape[0], dtype=torch.bool)
        pairwise = float((1.0 - similarities[distinct].mean()).item())
        return (
            covariance,
            covariance.diagonal(),
            eigenvalues,
            effective_rank,
            numerical_rank,
            pairwise,
        )


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
    assert "state_sample_covariance" not in first.telemetry_manifest
    assert "gradient_reachability" not in first.telemetry_manifest
    assert first.telemetry_manifest["state_variance"] == list(first.state_variance)
    assert first.telemetry_manifest["pairwise_cosine_semantic"] == (
        "all_distinct_online_state_positions_v1"
    )


def test_global_covariance_rank_and_pairwise_cosine_match_materialized_oracle() -> None:
    system = _system()
    sampler = _sampler()
    result = D1Evaluator().evaluate(system=system, sampler=sampler)
    states = _materialize(system, sampler)
    covariance = torch.cov(states.T, correction=1)
    norms = torch.linalg.vector_norm(states, dim=1)
    assert bool((norms > 0.0).all().item())
    normalized = states / norms.unsqueeze(1)
    similarities = normalized @ normalized.T
    distinct = ~torch.eye(states.shape[0], dtype=torch.bool)
    pairwise_distance = 1.0 - similarities[distinct].mean()

    torch.testing.assert_close(torch.tensor(result.state_sample_covariance), covariance)
    torch.testing.assert_close(torch.tensor(result.state_variance), covariance.diagonal())
    assert result.numerical_rank == int(torch.linalg.matrix_rank(covariance).item())
    assert result.mean_pairwise_cosine_distance == pytest.approx(float(pairwise_distance), abs=1e-6)


def test_direct_state_target_and_prediction_channels_match_materialized_oracles() -> None:
    system = _system()
    sampler = _sampler()
    result = D1Evaluator().evaluate(system=system, sampler=sampler)
    state = _materialize(system, sampler)
    target = _materialize(system, sampler, "target")
    predicted = _materialize(system, sampler, "predicted")

    _assert_channel_close(result.state_moments, _channel_oracle(state))
    _assert_channel_close(result.target_moments, _channel_oracle(target))
    _assert_channel_close(result.predicted_moments, _channel_oracle(predicted))
    residual = target - predicted
    expected_mean = residual.mean(dim=0)
    assert result.residual_mean_vector == pytest.approx(tuple(expected_mean.tolist()))
    assert result.residual_mean_l2_norm == pytest.approx(
        float(torch.linalg.vector_norm(expected_mean).item())
    )
    assert result.target_predicted_exact_match_fraction == pytest.approx(
        float((target == predicted).float().mean().item())
    )
    assert result.target_predicted_exact_match_semantic == "elementwise_fp32_equality_v1"
    assert result.state_mean == result.state_moments.mean
    assert result.state_rms_l2_norm == result.state_moments.rms_l2_norm


def test_exact_target_copy_canary_reports_complete_match_and_zero_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system()
    real_forward = system.forward

    def copied_target(batch: object):
        output = real_forward(batch)  # type: ignore[arg-type]
        copied_law = replace(
            output.future_law_loss,
            mean=output.target.values.detach().float(),
        )
        return replace(output, future_law_loss=copied_law)

    monkeypatch.setattr(system, "forward", copied_target)
    result = D1Evaluator().evaluate(system=system, sampler=_sampler())

    assert result.target_predicted_exact_match_fraction == 1.0
    assert result.residual_mean_vector == (0.0,) * 8
    assert result.residual_mean_l2_norm == 0.0
    assert result.residual_rms == 0.0


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


def test_bf16_evaluation_autocast_is_forward_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = _system(amp=D1AmpPolicy.BF16)
    forward_autocast: list[bool] = []
    reduction_autocast: list[bool] = []
    weighted_autocast: list[bool] = []
    finish_autocast: list[bool] = []
    real_add = evaluation_module._ChannelAccumulator.add  # pyright: ignore[reportPrivateUsage]
    real_weighted = evaluation_module._add_weighted  # pyright: ignore[reportPrivateUsage]
    real_finish = evaluation_module._ChannelAccumulator.finish  # pyright: ignore[reportPrivateUsage]

    def record_add(moment: object, states: torch.Tensor) -> None:
        reduction_autocast.append(torch.is_autocast_enabled("cpu"))
        real_add(moment, states)  # type: ignore[arg-type]

    def record_weighted(*args: object, **kwargs: object) -> None:
        weighted_autocast.append(torch.is_autocast_enabled("cpu"))
        real_weighted(*args, **kwargs)  # type: ignore[arg-type]

    def record_finish(moment: object):
        finish_autocast.append(torch.is_autocast_enabled("cpu"))
        return real_finish(moment)  # type: ignore[arg-type]

    handle = system.register_forward_pre_hook(
        lambda _module, _args: forward_autocast.append(torch.is_autocast_enabled("cpu"))
    )
    monkeypatch.setattr(evaluation_module._ChannelAccumulator, "add", record_add)
    monkeypatch.setattr(evaluation_module, "_add_weighted", record_weighted)
    monkeypatch.setattr(evaluation_module._ChannelAccumulator, "finish", record_finish)
    try:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            D1Evaluator().evaluate(system=system, sampler=_sampler())
    finally:
        handle.remove()

    assert forward_autocast == [True] * 3
    assert reduction_autocast == [False] * 9
    assert weighted_autocast == [False] * 3
    assert finish_autocast == [False] * 3


def test_bf16_diagnostics_match_fp32_materialized_oracle() -> None:
    system = _system(amp=D1AmpPolicy.BF16)
    oracle = _fp32_state_oracle(_bf16_forward_states(system, _sampler()))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = D1Evaluator().evaluate(system=system, sampler=_sampler())
    covariance, variance, eigenvalues, effective_rank, numerical_rank, pairwise = oracle

    torch.testing.assert_close(
        torch.tensor(result.state_sample_covariance), covariance, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(torch.tensor(result.state_variance), variance, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.tensor(result.state_covariance_eigenvalues),
        eigenvalues,
        rtol=1e-5,
        atol=1e-6,
    )
    assert result.effective_rank == pytest.approx(effective_rank, rel=1e-5, abs=1e-6)
    assert result.numerical_rank == numerical_rank
    assert result.mean_pairwise_cosine_distance == pytest.approx(pairwise, rel=1e-5, abs=1e-6)


def test_result_is_immutable_and_evaluation_fails_loud_on_invalid_evidence() -> None:
    system = _system()
    result = D1Evaluator().evaluate(system=system, sampler=_sampler())
    with pytest.raises(FrozenInstanceError):
        result.batch_count = 4  # type: ignore[misc]
    with pytest.raises(TypeError, match="immutable tuples"):
        replace(result.state_moments, mean=list(result.state_mean))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="immutable tuple rows"):
        replace(result, state_sample_covariance=list(result.state_sample_covariance))  # type: ignore[arg-type]
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
