"""Collapsed online states remain admissible evidence with explicit cosine gaps."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap
from entropic_substrate.ladder.capture.records.d1_telemetry_measurements import parse_quality
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1Evaluator
from entropic_substrate.ladder.run.trainer.d1_evaluation_codec import (
    d1_evaluation_from_mapping,
    d1_evaluation_to_mapping,
    parse_d1_evaluation,
    serialize_d1_evaluation,
)
from test_d1_evaluation import _sampler, _system

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("zero_positions", [1, 8])
def test_zero_states_preserve_moments_and_explicit_gap(zero_positions: int) -> None:
    system = _system()

    def collapse(_module: object, _args: object, output: object) -> object:
        output.online_states.values.reshape(-1, 8)[:zero_positions] = 0.0  # type: ignore[attr-defined]
        return output

    handle = system.register_forward_hook(collapse)
    try:
        result = D1Evaluator().evaluate(system=system, sampler=_sampler())
    finally:
        handle.remove()
    gap = result.mean_pairwise_cosine_distance
    assert gap.reason == "undefined_zero_norm"  # type: ignore[union-attr]
    assert gap.affected_positions == zero_positions * 3  # type: ignore[union-attr]
    assert gap.total_positions == result.online_positions == 24  # type: ignore[union-attr]
    if zero_positions == 8:
        assert result.effective_rank == result.numerical_rank == 0
        assert result.state_variance == (0.0,) * 8
    assert parse_d1_evaluation(serialize_d1_evaluation(result)) == result
    quality = parse_quality(result.telemetry_manifest)
    assert quality.manifest == result.telemetry_manifest
    assert quality.scalar_metric("numerical_rank") == result.numerical_rank
    with pytest.raises(ValueError, match="undefined_zero_norm"):
        quality.scalar_metric("mean_pairwise_cosine_distance")
    wire = d1_evaluation_to_mapping(result)
    assert wire["schema_version"] == "d1-evaluation-wire-v3"
    mutated = dict(result.telemetry_manifest)
    mutated["online_positions"] = 23
    with pytest.raises(ValueError, match="positions"):
        parse_quality(mutated)


def test_historical_float_shape_and_nonfinite_refusal() -> None:
    result = D1Evaluator().evaluate(system=_system(), sampler=_sampler())
    wire = d1_evaluation_to_mapping(result)
    assert wire["schema_version"] == "d1-evaluation-wire-v2"
    assert type(wire["mean_pairwise_cosine_distance"]) is float
    assert d1_evaluation_from_mapping(wire) == result
    assert parse_quality(result.telemetry_manifest).manifest == result.telemetry_manifest
    with pytest.raises(ValueError, match="finite"):
        replace(result, mean_pairwise_cosine_distance=float("nan"))


def test_nonfinite_online_state_is_still_defect() -> None:
    system = _system()

    def corrupt(_module: object, _args: object, output: object) -> object:
        output.online_states.values.fill_(torch.nan)  # type: ignore[attr-defined]
        return output

    handle = system.register_forward_hook(corrupt)
    try:
        with pytest.raises(ValueError, match="finite"):
            D1Evaluator().evaluate(system=system, sampler=_sampler())
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "mutation",
    [
        {"affected_positions": 0},
        {"affected_positions": 25},
        {"total_positions": True},
        {"reason": "missing"},
        {"schema_version": "unknown"},
    ],
)
def test_gap_payload_refuses_unknown_or_invalid_counts(mutation: dict[str, object]) -> None:
    result = D1Evaluator().evaluate(system=_system(), sampler=_sampler())
    quality = dict(result.telemetry_manifest)
    quality["mean_pairwise_cosine_distance"] = {**D1CosineGap(1, 24).manifest, **mutation}
    with pytest.raises(ValueError):
        parse_quality(quality)


def test_tracking_preserves_gap_counts_without_emitting_numeric_cosine() -> None:
    from entropic_substrate.ladder.capture.adapters.d1_tracking import D1MetricExtractor
    from test_d1_tracking import _event

    event = _event()
    payload = dict(event.manifest["payload"])
    payload["quality"] = {
        "numerical_rank": 0,
        "online_positions": 24,
        "mean_pairwise_cosine_distance": D1CosineGap(24, 24).manifest,
    }
    observed = D1MetricExtractor().extract(replace(event, payload=payload))
    assert observed is not None
    assert not any(key.endswith("/mean_pairwise_cosine_distance") for key in observed.metrics)
    assert observed.metrics["d1/diagnostic_gap/undefined_zero_norm/affected_positions"] == 24.0


def test_analysis_retains_gap_population_while_admitting_other_metrics(tmp_path: Path) -> None:
    from entropic_substrate.ladder.analyze.d1_capture import _cell

    from .test_d1_analysis_capture import _analysis_case

    _, _, _, aggregate, runs = _analysis_case(tmp_path)
    expected = aggregate.roster.expected_confirmatory[0]
    run = runs[expected.logical_run_id]
    collapsed = replace(
        run,
        evals=tuple(
            replace(
                item,
                quality=replace(
                    item.quality,
                    mean_pairwise_cosine_distance=D1CosineGap(1, item.quality.online_positions),
                ),
            )
            for item in run.evals
        ),
    )
    cell = _cell(expected, collapsed)
    assert "mean_pairwise_cosine_distance" not in cell.metrics
    assert len(cell.diagnostic_gaps["mean_pairwise_cosine_distance"]) == len(run.evals)
    assert "effective_rank" in cell.metrics and "nats_per_byte" in cell.metrics
    from entropic_substrate.ladder.analyze.d1_clause_policy import (
        D1ConditionSource,
        D1RegisteredCondition,
        D1ThresholdOperator,
        _endpoint,
    )

    condition = D1RegisteredCondition(
        source=D1ConditionSource.ARM_ENDPOINT_MEAN,
        operator=D1ThresholdOperator.AT_LEAST,
        threshold=0.1,
        metric="mean_pairwise_cosine_distance",
        estimator="last_k_mean",
        smoothing_window=1,
        contrast_id=None,
    )
    assert _endpoint(condition, cell.arm, (cell,)) is None


def test_complete_collapsed_journal_is_read_by_authoritative_reader(tmp_path: Path) -> None:
    from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence
    from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
    from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunStatus

    from .d1_telemetry_fixtures import valid_events, write_events

    events = valid_events()
    for index, event in enumerate(events):
        if event.event_kind is not JournalEventKind.EVAL:
            continue
        payload = cast(dict[str, object], event.manifest["payload"])
        quality = cast(dict[str, object], payload["quality"])
        positions = cast(int, quality["online_positions"])
        width = len(cast(list[float], quality["state_mean"]))
        quality.update(
            {
                "state_mean": [0.0] * width,
                "state_variance": [0.0] * width,
                "state_covariance_eigenvalues": [0.0] * width,
                "state_rms_l2_norm": 0.0,
                "numerical_rank": 0,
                "effective_rank": 0.0,
                "mean_pairwise_cosine_distance": D1CosineGap(positions, positions).manifest,
            }
        )
        events[index] = replace(event, payload=payload)
    journal = tmp_path / "collapsed.jsonl"
    write_events(journal, events)
    observed = read_d1_run_evidence(journal)
    assert observed.status is D1RunStatus.COMPLETE
    for item in observed.evals:
        assert item.quality.scalar_metric("effective_rank") == 0.0
        assert type(item.quality.mean_pairwise_cosine_distance) is D1CosineGap
