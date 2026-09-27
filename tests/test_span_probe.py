"""Frozen span probes isolate training labels, coordinates, and model state."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_span_state_collection import model

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.span_probe import fit_span_probe, score_span_probe
from entropic_substrate.ladder.run.span_states import SpanStateSet, collect_span_states

pytestmark = pytest.mark.unit
type Samples = tuple[SentenceExecution, SpanStateSet, SpanStateSet]


def probe_samples(tmp_path: Path) -> Samples:
    run = model(tmp_path)
    train = collect_span_states(
        run, "checkpoint", split="train", count=8, lags=(2, 4, 16, 64), seed=42
    )
    heldout = collect_span_states(
        run, "checkpoint", split="validation", count=8, lags=(2, 4, 16, 64), seed=42
    )
    return run, train, heldout


@pytest.fixture
def samples(tmp_path: Path) -> Samples:
    return probe_samples(tmp_path)


def config() -> ProbeConfig:
    return ProbeConfig(train_positions=8, eval_positions=8, updates=4, batch_size=4)


def test_fit_is_deterministic_detached_and_frozen(samples: Samples) -> None:
    run, train, heldout = samples
    parameters = [p.detach().clone() for p in run.model.parameters()]
    rng = torch.get_rng_state().clone()
    probe = fit_span_probe(train, config(), control="observed")
    repeat = fit_span_probe(train, config(), control="observed")
    assert probe.manifest.probe_id == repeat.manifest.probe_id
    assert probe.manifest.feature_map == "state-plus-log1p-horizon-v1"
    assert probe.manifest.fit_seconds >= 0
    assert all(not t.requires_grad for t in probe.state_dict().values())
    exported = probe.state_dict()
    exported["weight"].fill_(123)
    probe.verify()
    report = score_span_probe(probe, heldout)
    assert report.summary.count == 8
    assert report.uniform_summary.mean_gain == 0
    assert len(report.pairs) == 8 and len(report.uniform) == 8
    for p, before in zip(run.model.parameters(), parameters, strict=True):
        torch.testing.assert_close(p, before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_shuffle_is_horizon_matched_and_recorded(samples: Samples) -> None:
    _, train, heldout = samples
    # Equal horizons can occur on opposite sides of different sentence events.
    positions = tuple(
        replace(p, span_end=p.span_start + 2, target_position=p.span_start + 1 + p.lag)
        for p in train.positions
    )
    train = replace(train, positions=positions, before_horizons=train.after_horizons + 2)
    probe = fit_span_probe(train, config(), control="shuffled")
    horizons = torch.cat((train.before_horizons, train.after_horizons))
    permutation = torch.tensor(probe.manifest.training_permutation)
    torch.testing.assert_close(horizons[permutation], horizons, rtol=0, atol=0)
    assert permutation.sort().values.tolist() == list(range(16))
    assert torch.all(permutation[:8] < 8) and torch.all(permutation[8:] >= 8)
    assert probe.manifest.control == "shuffled"
    assert score_span_probe(probe, heldout).summary.count == 8


@pytest.mark.parametrize("field", ["checkpoint_id", "codec_id", "source_digest"])
def test_scoring_rejects_cross_artifact_states(samples: Samples, field: str) -> None:
    _, train, heldout = samples
    probe = fit_span_probe(train, config(), control="observed")
    with pytest.raises(ValueError, match="binding"):
        score_span_probe(probe, replace(heldout, **{field: "other"}))


def test_fitting_cannot_use_heldout_and_scoring_cannot_use_train(samples: Samples) -> None:
    _, train, heldout = samples
    with pytest.raises(ValueError, match="training split"):
        fit_span_probe(heldout, config(), control="observed")
    probe = fit_span_probe(train, config(), control="observed")
    with pytest.raises(ValueError, match="held-out"):
        score_span_probe(probe, train)
    positions = (
        replace(heldout.positions[0], document_id=train.positions[0].document_id),
        *heldout.positions[1:],
    )
    forged = replace(heldout, positions=positions)
    with pytest.raises(ValueError, match="documents overlap"):
        score_span_probe(probe, forged)


def test_invalid_and_mutated_tensors_fail_loudly(samples: Samples) -> None:
    _, train, heldout = samples
    probe = fit_span_probe(train, config(), control="observed")
    with pytest.raises(ValueError, match="canonical byte"):
        fit_span_probe(replace(train, labels=torch.full((8,), 256)), config(), control="observed")
    with pytest.raises(ValueError, match="horizons"):
        score_span_probe(probe, replace(heldout, after_horizons=heldout.after_horizons + 1))
    with pytest.raises(ValueError, match="detached finite CPU FP32"):
        score_span_probe(
            probe, replace(heldout, before=torch.full_like(heldout.before, float("nan")))
        )
    probe._weight.add_(1)
    with pytest.raises(ValueError, match="identity"):
        score_span_probe(probe, heldout)


def test_zero_states_are_valid_probe_data(samples: Samples) -> None:
    _, train, heldout = samples
    train = replace(
        train, before=torch.zeros_like(train.before), after=torch.zeros_like(train.after)
    )
    heldout = replace(
        heldout, before=torch.zeros_like(heldout.before), after=torch.zeros_like(heldout.after)
    )
    probe = fit_span_probe(train, config(), control="observed")
    assert score_span_probe(probe, heldout).summary.count == 8


def test_known_observable_signal_is_learned_with_fp32_probe_island(samples: Samples) -> None:
    _, train, heldout = samples
    labels = torch.arange(8) % 2
    signal = torch.nn.functional.one_hot(labels, 8).float()
    train = replace(train, before=torch.zeros_like(signal), after=signal, labels=labels)
    heldout = replace(heldout, before=torch.zeros_like(signal), after=signal, labels=labels)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        probe = fit_span_probe(
            train, replace(config(), updates=80, learning_rate=0.05), control="observed"
        )
        report = score_span_probe(probe, heldout)
    assert report.after_accuracy == 1.0
    assert report.summary.mean_gain > 0
    assert report.summary.after_nats_per_byte < report.uniform_summary.after_nats_per_byte
    assert probe.state_dict()["weight"].dtype == torch.float32
