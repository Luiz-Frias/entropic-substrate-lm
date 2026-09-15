"""Production scientific composition contracts for the D ladder."""

from dataclasses import replace

import pytest
import torch

from d1_runtime_fixtures import runtime_batch, runtime_family
from entropic_substrate.experiments.d1 import D1Arm
from entropic_substrate.experiments.ladder import Rung, RungSpec, Treatment
from entropic_substrate.predictive_state.ladder_runtime import LadderPredictiveSystem

pytestmark = pytest.mark.unit


def spec(treatment: Treatment = Treatment.S) -> RungSpec:
    return RungSpec(
        base=runtime_family().draft_for(D1Arm.BOTH), rung=Rung.D2, treatment=treatment, seed=7
    )


def test_l_and_s_pair_fresh_weights_but_own_different_gradient_roles() -> None:
    language = LadderPredictiveSystem(spec(Treatment.L))
    supervised = LadderPredictiveSystem(spec())
    for left, right in zip(
        language.core.online_branch.parameters(),
        supervised.core.online_branch.parameters(),
        strict=True,
    ):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
    assert "future_law" not in language.optimizer_roles()
    assert language.gradient_absence_reasons()["future_law"] == "objective_disabled"
    result = language(runtime_batch())
    assert result.target is None and result.future_law_loss is None
    assert result.objective.future.item() == 0
    result.objective.total.backward()
    assert all(p.grad is None for p in language.core.future_law.parameters())
    assert any(p.grad is not None for p in language.core.online_branch.parameters())


def test_s_requires_fitted_baseline_and_target_is_future_only() -> None:
    system = LadderPredictiveSystem(spec())
    with pytest.raises(ValueError, match="fitted"):
        system(runtime_batch())
    system.fit_mean_displacement_baseline([runtime_batch()])
    before = system(runtime_batch())
    after = system(runtime_batch(future_last_id=8))
    torch.testing.assert_close(before.online_states.values, after.online_states.values)
    assert not torch.equal(before.target.values, after.target.values)
    assert before.target.detached


def test_spec_rejects_wrong_rung_mechanisms_and_resumes() -> None:
    with pytest.raises(ValueError, match="D2"):
        replace(spec(), horizons=(4,))
    with pytest.raises(ValueError, match="seed"):
        replace(spec(), seed=True)
    first = LadderPredictiveSystem(spec(Treatment.L))
    second = LadderPredictiveSystem(spec(Treatment.S))
    with pytest.raises(ValueError, match="construction"):
        second.load_state_dict(first.state_dict())


def test_final_residual_callback_preserves_identity_and_can_change_output() -> None:
    system = LadderPredictiveSystem(spec())
    adapter, encoder = system.core.online_branch.adapter, system.core.online_branch.encoder
    encoded = adapter(runtime_batch().online_observations, system.draft.codec)
    expected = encoder(encoded).values
    calls = []

    def identity(state: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        calls.append(state.shape)
        return residual

    torch.testing.assert_close(encoder(encoded, final_residual=identity).values, expected)
    assert len(calls) == 1
    changed = encoder(encoded, final_residual=lambda _state, residual: residual * 0).values
    assert not torch.equal(changed, expected)


def test_geometry_consumer_changes_states_and_language_gradients_reach_decoder() -> None:
    supervised = LadderPredictiveSystem(spec())
    geometric = LadderPredictiveSystem(spec(Treatment.G))
    for system in (supervised, geometric):
        system.fit_mean_displacement_baseline([runtime_batch()])
    plain, curved = supervised(runtime_batch()), geometric(runtime_batch())
    assert not torch.allclose(plain.online_states.values, curved.online_states.values)
    curved.objective.language.backward()
    assert any(
        p.grad is not None and p.grad.norm() > 0 for p in geometric.core.future_law.parameters()
    )
    assert torch.all(curved.metrics["fisher_residual_norm"] <= curved.metrics["fisher_tolerance"])


def test_geometry_evaluation_and_mixed_precision_share_the_production_consumer() -> None:
    system = LadderPredictiveSystem(spec(Treatment.G))
    system.fit_mean_displacement_baseline([runtime_batch()])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = system(runtime_batch())
    assert torch.isfinite(output.total_loss)
    assert "fisher_iterations" in output.metrics


def test_probe_encoding_is_the_same_future_blind_production_path() -> None:
    system = LadderPredictiveSystem(spec(Treatment.G))
    system.fit_mean_displacement_baseline([runtime_batch()])
    actual = system.encode_observations(runtime_batch().online_observations)
    torch.testing.assert_close(actual.values, system(runtime_batch()).online_states.values)
    with pytest.raises(ValueError, match="observation"):
        system.encode_observations(torch.tensor([[99]], dtype=torch.long))
