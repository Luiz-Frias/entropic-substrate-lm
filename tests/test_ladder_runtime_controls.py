"""D3-D5 production controls and semantic-axis influence."""

from dataclasses import replace

import pytest
import torch

from d1_runtime_fixtures import runtime_batch
from entropic_substrate.data.batching import PredictiveBatch
from entropic_substrate.experiments.ladder import CoarseSpec, Rung
from entropic_substrate.observation import ByteObservationCodec, ObservationAlignment
from entropic_substrate.observation.permuted_byte_codec import PermutedByteObservationCodec
from entropic_substrate.predictive_state.ladder_runtime import LadderPredictiveSystem
from test_ladder_runtime import spec

pytestmark = pytest.mark.unit


def batch_for(horizon: int, *, future: int = 7) -> PredictiveBatch:
    base = runtime_batch()
    context = torch.cat(
        (base.online_observations.clone(), torch.full((2, horizon), future, dtype=torch.int64)),
        dim=1,
    )
    return PredictiveBatch(
        base.online_observations,
        base.language_targets,
        context,
        base.source_starts,
        tuple(
            ObservationAlignment.predictive(start=int(start), sequence_length=3, horizon=horizon)
            for start in base.source_starts
        ),
        horizon,
    )


def test_d3_composite_scores_every_horizon_without_future_leakage() -> None:
    d2 = spec()
    d3 = replace(d2, rung=Rung.D3, horizons=(1, 4, 16, 64), predecessor=d2)
    system = LadderPredictiveSystem(d3)
    batch = batch_for(64)
    system.fit_mean_displacement_baseline([batch])
    output = system(batch)
    assert set(output.losses_by_horizon) == {1, 4, 16, 64}
    torch.testing.assert_close(
        output.objective.future,
        torch.stack([loss.nll for loss in output.losses_by_horizon.values()]).mean(),
    )
    altered = system(batch_for(64, future=8))
    torch.testing.assert_close(output.online_states.values, altered.online_states.values)
    output.total_loss.backward()
    assert all(
        any(p.grad is not None for p in head.parameters()) for head in system.laws.heads.values()
    )


def test_baseline_reports_actual_target_encoder_positions() -> None:
    predecessor = spec()
    system = LadderPredictiveSystem(
        replace(predecessor, rung=Rung.D3, horizons=(1, 4, 16, 64), predecessor=predecessor)
    )
    calls = []
    handle = system.core.target_branch.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].numel())
    )
    try:
        observed = system.fit_mean_displacement_baseline([batch_for(64), batch_for(64)])
    finally:
        handle.remove()
    assert len(calls) == 8
    assert observed == sum(calls)


@pytest.mark.parametrize("mode", ["identity", "fixed", "learned", "never", "whitespace"])
def test_d4_mechanism_enters_production_objective_and_ownership(mode: str) -> None:
    d2 = spec()
    d3 = replace(d2, rung=Rung.D3, predecessor=d2)
    d4 = replace(d3, rung=Rung.D4, predecessor=d3, coarse=CoarseSpec(mode=mode))
    system = LadderPredictiveSystem(d4)
    system.fit_mean_displacement_baseline([runtime_batch()])
    output = system(runtime_batch())
    assert "coarse_boundaries" in output.metrics
    output.total_loss.backward()
    assert ("boundary_router" in system.optimizer_roles()) == (mode == "learned")
    if mode == "learned":
        assert any(p.grad is not None for p in system.coarse.router.parameters())
    optimizer = torch.optim.AdamW(system.optimizer_parameters())
    system.validate_optimizer_ownership(optimizer)
    with pytest.raises(ValueError, match="every and only"):
        system.validate_optimizer_ownership(torch.optim.AdamW([next(system.parameters())]))


def test_d5_exact_coupling_is_separate_from_fresh_paired_training() -> None:
    d2 = replace(spec(), base=replace(spec().base, codec=ByteObservationCodec().identity))
    d3 = replace(d2, rung=Rung.D3, predecessor=d2)
    d4 = replace(d3, rung=Rung.D4, predecessor=d3, coarse=CoarseSpec(mode="identity"))
    d5 = replace(d4, rung=Rung.D5, predecessor=d4)
    identity = LadderPredictiveSystem(d5)
    codec = PermutedByteObservationCodec.from_seed(19)
    permuted = LadderPredictiveSystem(replace(d5, permutation=codec.mapping))
    identity.fit_mean_displacement_baseline([runtime_batch()])
    permuted.fit_mean_displacement_baseline([runtime_batch()])
    assert not torch.equal(
        identity(runtime_batch()).language_logits, permuted(runtime_batch()).language_logits
    )
    # Exact algebraic coupling is a separate intervention, never automatic init.
    with torch.no_grad():
        for branch in (permuted.core.online_branch, permuted.core.target_branch):
            branch.adapter.embedding.weight.copy_(
                codec.permute_boundary_rows(branch.adapter.embedding.weight)
            )
        projection = permuted.core.language_readout.projection
        projection.weight.copy_(codec.permute_boundary_rows(projection.weight))
        if projection.bias is not None:
            projection.bias.copy_(codec.permute_boundary_rows(projection.bias))
    left, right = identity(runtime_batch()), permuted(runtime_batch())
    torch.testing.assert_close(left.language_logits, right.language_logits)
    torch.testing.assert_close(left.online_states.values, right.online_states.values)
    assert right.online_states.codec_identity == codec.identity
