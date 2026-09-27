"""D5 comparisons bind paired source observations and preserve missing evidence."""

from __future__ import annotations

import hashlib
import io
from dataclasses import replace

import pytest
import torch

from d1_runtime_fixtures import runtime_batch, runtime_family
from entropic_substrate.core.observation.byte_codec import ByteObservationCodec
from entropic_substrate.core.observation.permuted_byte_codec import PermutedByteObservationCodec
from entropic_substrate.ladder.analyze.codec_equivariance import (
    CodecComparisonMode,
    CodecSnapshot,
    ScalePolicyObservation,
    compare_codec_pair,
)
from entropic_substrate.ladder.register.d1 import D1Arm
from entropic_substrate.ladder.register.ladder import CoarseSpec, Rung, RungSpec, Treatment
from entropic_substrate.ladder.run.construction.ladder_runtime import LadderPredictiveSystem

pytestmark = pytest.mark.unit


def snapshot(**changes: object) -> CodecSnapshot:
    torch.manual_seed(27)
    result = CodecSnapshot(
        seed=7,
        sample_digest="a" * 64,
        checkpoint_digest="b" * 64,
        codec_identity=ByteObservationCodec().identity,
        context_starts=(0, 0, 0, 10, 10, 10),
        source_positions=(0, 1, 2, 10, 11, 12),
        canonical_targets=(1, 2, 3, 4, 5, 6),
        canonical_support=tuple(range(256)),
        canonical_logits=torch.randn(6, 256),
        states=torch.randn(6, 8),
        scale_policy=None,
    )
    return replace(result, **changes)


def test_rotation_preserves_relations_and_function_not_raw_coordinates() -> None:
    left = snapshot()
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    right = replace(
        left,
        states=left.states.double() @ rotation,
        checkpoint_digest="c" * 64,
        codec_identity=PermutedByteObservationCodec.from_seed(42).identity,
    )
    report = compare_codec_pair(left, right, mode=CodecComparisonMode.COUPLED_CONTROL)
    assert report.mode is CodecComparisonMode.COUPLED_CONTROL
    assert report.evidence_class == "DERIVED_FROM_MEASURED"
    assert report.metrics["js_nats"] == pytest.approx(0, abs=1e-12)
    assert report.metrics["gram_rmse"] == pytest.approx(0, abs=1e-12)
    assert report.metrics["procrustes_rms"] == pytest.approx(0, abs=1e-12)
    assert report.metrics["left_state_rank"] == report.metrics["right_state_rank"]
    assert report.gaps["scale_policy"].reason == "not_chunked"
    assert "boundary_disagreement" not in report.metrics


def test_collapsed_states_keep_rank_and_procrustes_but_expose_undefined_gram() -> None:
    left = snapshot(states=torch.zeros(6, 8))
    report = compare_codec_pair(left, left, mode=CodecComparisonMode.INDEPENDENT_TRAINING)
    assert report.metrics["left_state_rank"] == 0
    assert report.metrics["left_effective_rank"] == 0
    assert report.metrics["procrustes_rms"] == 0
    assert report.gaps["gram"].reason == "undefined_zero_norm"
    assert report.gaps["gram"].affected_positions == 6
    assert "gram_rmse" not in report.metrics


def test_scale_policy_compares_boundaries_and_completed_duration_values() -> None:
    left = snapshot(scale_policy=ScalePolicyObservation((False, False, True) * 2, (0, 0, 3) * 2))
    right = replace(
        left, scale_policy=ScalePolicyObservation((False, True, False) * 2, (0, 2, 0) * 2)
    )
    report = compare_codec_pair(left, right, mode=CodecComparisonMode.INDEPENDENT_TRAINING)
    assert report.metrics["boundary_disagreement"] == pytest.approx(4 / 6)
    assert report.metrics["duration_event_mae"] == pytest.approx(10 / 6)
    assert report.metrics["mean_completed_duration_gap"] == -1


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_digest": "e" * 64},
        {"context_starts": (0, 0, 1, 10, 10, 10)},
        {"source_positions": (0, 1, 3, 10, 11, 12)},
        {"canonical_targets": (0, 2, 3, 4, 5, 6)},
        {"states": torch.ones(6, 9)},
        {"seed": 8},
    ],
)
def test_pair_rejects_mismatched_regime(changes: dict[str, object]) -> None:
    left = snapshot()
    with pytest.raises(ValueError, match="paired"):
        compare_codec_pair(left, replace(left, **changes), mode=CodecComparisonMode.COUPLED_CONTROL)


def test_snapshot_rejects_unsupported_targets_nonfinite_and_oversized_gram() -> None:
    with pytest.raises(ValueError, match="support"):
        snapshot(canonical_support=tuple(reversed(range(256))))
    with pytest.raises(ValueError, match="finite"):
        snapshot(states=torch.full((6, 8), float("nan")))
    with pytest.raises(ValueError, match="256"):
        snapshot(states=torch.zeros(257, 8))
    with pytest.raises(ValueError, match="digest"):
        snapshot(checkpoint_digest="")
    with pytest.raises(ValueError, match="duration"):
        ScalePolicyObservation((False, True), (1, 2))


def test_absent_segment_duration_is_never_a_zero_measurement() -> None:
    left = snapshot(scale_policy=ScalePolicyObservation((False,) * 6, (0,) * 6))
    report = compare_codec_pair(left, left, mode=CodecComparisonMode.INDEPENDENT_TRAINING)
    assert report.metrics["boundary_disagreement"] == 0
    assert report.gaps["mean_completed_duration_gap"].reason == "no_completed_segments"
    assert "mean_completed_duration_gap" not in report.metrics
    with pytest.raises(ValueError, match="paired scale"):
        compare_codec_pair(left, snapshot(), mode=CodecComparisonMode.INDEPENDENT_TRAINING)


def test_actual_d5_pair_trains_and_produces_comparable_checkpoint_bound_metrics() -> None:
    base = replace(runtime_family().draft_for(D1Arm.BOTH), codec=ByteObservationCodec().identity)
    d2 = RungSpec(base=base, rung=Rung.D2, treatment=Treatment.S, seed=7)
    d3 = replace(d2, rung=Rung.D3, predecessor=d2)
    d4 = replace(d3, rung=Rung.D4, predecessor=d3, coarse=CoarseSpec(mode="fixed", window=2))
    d5 = replace(d4, rung=Rung.D5, predecessor=d4)
    codec = PermutedByteObservationCodec.from_seed(42)
    systems = [
        LadderPredictiveSystem(d5),
        LadderPredictiveSystem(replace(d5, permutation=codec.mapping)),
    ]
    batch = runtime_batch()
    captured = []
    for system in systems:
        system.fit_mean_displacement_baseline([batch])
        optimizer = torch.optim.AdamW(system.optimizer_parameters(), lr=1e-3)
        system(batch).total_loss.backward()
        optimizer.step()
        system.eval()
        with torch.no_grad():
            output = system(batch)
        checkpoint = io.BytesIO()
        torch.save(system.state_dict(), checkpoint)
        captured.append(
            snapshot(
                sample_digest=hashlib.sha256(
                    bytes(batch.online_observations.flatten().tolist())
                    + bytes(batch.language_targets.flatten().tolist())
                    + bytes(batch.source_starts.tolist())
                ).hexdigest(),
                checkpoint_digest=hashlib.sha256(checkpoint.getvalue()).hexdigest(),
                codec_identity=system.codec_identity,
                canonical_logits=output.language_logits.reshape(-1, 256),
                states=output.online_states.values.reshape(-1, 8),
                canonical_targets=tuple(batch.language_targets.flatten().tolist()),
                scale_policy=ScalePolicyObservation(
                    tuple(bool(x) for x in output.metrics["coarse_boundaries"].flatten()),
                    tuple(int(x) for x in output.metrics["coarse_durations"].flatten()),
                ),
            )
        )
    report = compare_codec_pair(*captured, mode=CodecComparisonMode.INDEPENDENT_TRAINING)
    assert report.metrics["js_nats"] > 0
    assert report.metrics["boundary_disagreement"] == 0
    assert report.metrics["left_state_rank"] > 0
    assert report.metrics["right_state_rank"] > 0
    assert report.left_checkpoint != report.right_checkpoint
