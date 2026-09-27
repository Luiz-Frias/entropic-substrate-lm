"""Observable probes remain detached, reproducible, and source aligned."""

from dataclasses import replace

import pytest
import torch

from entropic_substrate.core.probe.predictive_probe import (
    FrozenPredictiveProbe,
    ProbeConfig,
    StateSequence,
    fit_predictive_probe,
    score_predictive_probe,
)

pytestmark = pytest.mark.unit


def sequence(split: str = "train") -> StateSequence:
    return StateSequence(
        states=torch.zeros(80, 3, requires_grad=True),
        canonical_bytes=torch.arange(80) % 2,
        checkpoint_id="ckpt",
        codec_id="identity",
        split_id=split,
        source_id="enwik8",
        source_offset=0 if split == "train" else 1000,
        split=split,
    )


def tiny() -> ProbeConfig:
    return ProbeConfig(train_positions=8, eval_positions=4, updates=2, batch_size=4)


def test_defaults_and_refusal_to_fit_heldout() -> None:
    config = ProbeConfig()
    assert config.lags == (2, 4, 16, 64)
    assert (config.train_positions, config.eval_positions) == (65536, 8192)
    assert (config.updates, config.batch_size, config.seed) == (200, 256, 42)
    with pytest.raises(ValueError, match="training"):
        fit_predictive_probe(sequence("heldout"), tiny())


def test_fitting_detaches_freezes_and_preserves_rng() -> None:
    train = sequence()
    rng = torch.random.get_rng_state().clone()
    fitted = fit_predictive_probe(train, tiny())
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert train.states.grad is None
    assert all(not tensor.requires_grad for tensor in fitted.state_dict().values())
    again = fit_predictive_probe(train, tiny())
    assert fitted.manifest.probe_id == again.manifest.probe_id
    assert fitted.manifest.training_positions == again.manifest.training_positions


def test_score_pairs_same_target_and_controls_are_captured() -> None:
    fitted = fit_predictive_probe(sequence(), tiny())
    heldout = sequence("heldout")
    report = score_predictive_probe(fitted, heldout)
    assert len(report.primary) == 16
    assert len(report.shuffled) == len(report.uniform) == len(report.primary)
    assert report.uniform_summary.mean_gain == 0.0
    assert report.uniform_summary.after_nats_per_byte == pytest.approx(5.545177444)
    assert report.primary == report.shuffled  # all hidden states are identical
    for pair in report.primary:
        assert pair.after.target_position > pair.after.state_position
        assert pair.before.target_position == pair.after.target_position
        assert pair.before.target_byte == pair.after.target_byte
        assert pair.before.binding.probe_id == fitted.manifest.probe_id
        local_target = pair.before.target_position - heldout.source_offset
        assert pair.before.target_byte == int(heldout.canonical_bytes[local_target])
        assert pair.after.remaining_horizon == pair.before.remaining_horizon - 1
    assert heldout.states.grad is None
    assert report.score_seconds >= 0 and fitted.manifest.fit_seconds >= 0


@pytest.mark.parametrize("field", ["checkpoint_id", "codec_id"])
def test_mismatch_refused(field: str) -> None:
    fitted = fit_predictive_probe(sequence(), tiny())
    with pytest.raises(ValueError, match="identity"):
        score_predictive_probe(fitted, replace(sequence("heldout"), **{field: "other"}))


def test_scoring_train_or_overlapping_source_refused() -> None:
    fitted = fit_predictive_probe(sequence(), tiny())
    with pytest.raises(ValueError, match="held-out"):
        score_predictive_probe(fitted, sequence())
    with pytest.raises(ValueError, match="overlap"):
        score_predictive_probe(fitted, replace(sequence("heldout"), source_offset=0))


def test_bad_shape_nonfinite_insufficient_positions_and_mutated_probe_refused() -> None:
    train = sequence()
    with pytest.raises(ValueError, match="finite"):
        replace(train, states=torch.full((80, 3), float("nan")))
    with pytest.raises(ValueError, match="aligned"):
        replace(train, canonical_bytes=torch.zeros(79, dtype=torch.long))
    with pytest.raises(ValueError, match="positions"):
        fit_predictive_probe(train, replace(tiny(), train_positions=1000))
    fitted = fit_predictive_probe(train, tiny())
    snapshot = fitted.state_dict()
    next(iter(snapshot.values())).fill_(999)
    with pytest.raises(ValueError, match="artifact identity"):
        FrozenPredictiveProbe(fitted.manifest, snapshot["weight"], snapshot["bias"])
    assert fitted.manifest.probe_id == fit_predictive_probe(train, tiny()).manifest.probe_id


def test_labels_never_modify_supplied_states() -> None:
    heldout = sequence("heldout")
    states = heldout.states.clone()
    fitted = fit_predictive_probe(sequence(), tiny())
    score_predictive_probe(fitted, heldout)
    score_predictive_probe(fitted, replace(heldout, canonical_bytes=1 - heldout.canonical_bytes))
    assert torch.equal(states, heldout.states)


def test_probe_capture_roundtrip_and_sample_positions_reproduce() -> None:
    fitted = fit_predictive_probe(sequence(), tiny())
    weights = fitted.state_dict()
    restored = FrozenPredictiveProbe(fitted.manifest, weights["weight"], weights["bias"])
    left = score_predictive_probe(fitted, sequence("heldout"))
    right = score_predictive_probe(restored, sequence("heldout"))
    assert left.primary == right.primary
    assert left.shuffled_source_positions == right.shuffled_source_positions
    assert left.evaluation_content_digest == right.evaluation_content_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"lags": ()},
        {"lags": (1,)},
        {"lags": (2, 2)},
        {"updates": 0},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
    ],
)
def test_probe_config_rejects_invalid_contract(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(tiny(), **changes)


def test_nonzero_states_fit_without_changing_existing_model_gradients() -> None:
    model_states = torch.randn(80, 3, requires_grad=True)
    model_states.square().sum().backward()
    original_gradient = model_states.grad.clone()
    original_states = model_states.detach().clone()
    train = replace(sequence(), states=model_states)
    fit_predictive_probe(train, tiny())
    torch.testing.assert_close(model_states.grad, original_gradient)
    torch.testing.assert_close(model_states.detach(), original_states)


def test_window_resets_exclude_targets_crossing_context_boundaries() -> None:
    config = replace(tiny(), lags=(2, 4), train_positions=12, eval_positions=12)
    train = replace(sequence(), window_starts=(0, 20, 40, 60))
    fitted = fit_predictive_probe(train, config)
    heldout = replace(sequence("heldout"), window_starts=train.window_starts)
    report = score_predictive_probe(fitted, heldout)
    assert fitted.manifest.training_window_starts == train.window_starts
    assert report.evaluation_window_starts == train.window_starts
    for pair in report.primary:
        assert (pair.before.state_position - 1000) // 20 == (
            pair.after.target_position - 1000
        ) // 20


@pytest.mark.parametrize("starts", [(1,), (0, 20, 20), (0, 80), (0, -1)])
def test_invalid_window_layout_refused(starts: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="window"):
        replace(sequence(), window_starts=starts)
