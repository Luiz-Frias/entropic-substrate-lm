"""Optimizer transactions distinguish execution defects from finite zero evidence."""

import pytest
import torch
from tests.test_sentence_model import history, tiny_model

from entropic_substrate.sentence_prediction.objective import codec_warmup_loss, prediction_loss
from entropic_substrate.sentence_prediction.training import (
    SentenceTrainingDefect,
    apply_sentence_step,
    build_sentence_optimizer,
    sentence_roles,
)

pytestmark = pytest.mark.unit


def test_codec_warmup_has_explicit_inactive_roles_and_updates_only_codec() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.01
    )
    spine_before = {name: p.detach().clone() for name, p in model.spine.named_parameters()}
    result = apply_sentence_step(
        model,
        optimizer,
        "codec_warmup",
        lambda: codec_warmup_loss(model, torch.tensor([[65, 46, 256]])).total,
        clip_norm=1.0,
    )
    assert result.gradients["spine"]["status"] == "structurally_absent_codec_warmup"
    assert result.gradients["future_law"]["norm"] is None
    for name, parameter in model.spine.named_parameters():
        torch.testing.assert_close(parameter, spine_before[name], atol=0, rtol=0)
    assert result.aggregate_norm > 0


def test_prediction_transaction_preserves_frozen_encoder_and_normalizes_spine() -> None:
    model = tiny_model()
    model.codec.freeze_encoder()
    before = {name: p.detach().clone() for name, p in model.codec.encoder.named_parameters()}
    optimizer = build_sentence_optimizer(model, "prediction", learning_rate=1e-3, weight_decay=0.01)
    result = apply_sentence_step(
        model,
        optimizer,
        "prediction",
        lambda: (
            prediction_loss(model, history(), torch.tensor([[67, 257]]), future_weight=0.3).total
        ),
        clip_norm=1.0,
    )
    assert result.gradients["encoder"]["status"] == "structurally_absent_frozen_encoder"
    for name, parameter in model.codec.encoder.named_parameters():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    torch.testing.assert_close(model.spine.projection.weight.norm(dim=-1), torch.ones(8))


def test_finite_zero_gradients_do_not_skip_adamw_transaction() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(model, "codec_warmup", learning_rate=0.1, weight_decay=0.1)
    before = model.codec.encoder.projection.weight.detach().clone()
    parameters = tuple(model.codec.parameters())
    result = apply_sentence_step(
        model,
        optimizer,
        "codec_warmup",
        lambda: sum((p.sum() * 0 for p in parameters), parameters[0].new_zeros(())),
        clip_norm=1.0,
    )
    assert result.aggregate_norm == 0.0
    assert result.gradients["encoder"] == {"status": "measured_zero", "norm": 0.0}
    assert result.gradients["decoder"] == {"status": "measured_zero", "norm": 0.0}
    torch.testing.assert_close(model.codec.encoder.projection.weight, before * 0.99)
    assert all(float(optimizer.state[p]["step"]) == 1 for p in parameters)


def test_missing_graph_participation_is_a_defect_before_optimizer_commit() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.01
    )
    before = model.codec.encoder.projection.weight.detach().clone()
    with pytest.raises(SentenceTrainingDefect, match="missing_gradient"):
        apply_sentence_step(
            model,
            optimizer,
            "codec_warmup",
            lambda: model.codec.encoder.projection.weight.sum(),
            clip_norm=1.0,
        )
    assert not optimizer.state
    torch.testing.assert_close(model.codec.encoder.projection.weight, before, atol=0, rtol=0)


def test_nonfinite_gradient_fails_before_optimizer_commit() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.01
    )
    target = torch.tensor([[65, 46, 256]])
    parameter = model.codec.encoder.projection.weight
    with parameter.register_hook(lambda gradient: gradient * float("nan")):
        with pytest.raises(SentenceTrainingDefect, match="nonfinite_gradient"):
            apply_sentence_step(
                model,
                optimizer,
                "codec_warmup",
                lambda: codec_warmup_loss(model, target).total,
                clip_norm=1.0,
            )
    assert not optimizer.state


def test_wrong_optimizer_owner_is_rejected_before_forward() -> None:
    model = tiny_model()
    optimizer = torch.optim.AdamW(tiny_model().parameters())

    def forbidden() -> torch.Tensor:
        raise AssertionError("forward must not run after ownership failure")

    with pytest.raises(SentenceTrainingDefect, match="optimizer_ownership"):
        apply_sentence_step(model, optimizer, "codec_warmup", forbidden, clip_norm=1.0)


def test_inactive_role_cannot_receive_an_unowned_gradient() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.01
    )

    def incorrect_objective() -> torch.Tensor:
        return (
            codec_warmup_loss(model, torch.tensor([[65, 256]])).total
            + model.head.direction.weight.sum()
        )

    with pytest.raises(SentenceTrainingDefect, match="unexpected_gradient"):
        apply_sentence_step(model, optimizer, "codec_warmup", incorrect_objective, clip_norm=1.0)
    assert not optimizer.state


def test_exact_owner_sgd_cannot_impersonate_the_adamw_construction() -> None:
    model = tiny_model()
    optimizer = torch.optim.SGD(model.codec.parameters(), lr=0.1)
    with pytest.raises(SentenceTrainingDefect, match="optimizer_contract"):
        apply_sentence_step(
            model, optimizer, "codec_warmup", lambda: torch.tensor(0.0), clip_norm=1.0
        )


@pytest.mark.parametrize("mutation", ["decay_vectors", "swap_parameters", "maximize"])
def test_optimizer_group_mutations_are_refused(mutation: str) -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.1
    )
    if mutation == "decay_vectors":
        optimizer.param_groups[1]["weight_decay"] = 0.1
    elif mutation == "maximize":
        optimizer.param_groups[0]["maximize"] = True
    else:
        first, second = optimizer.param_groups
        first["params"][0], second["params"][0] = second["params"][0], first["params"][0]
    with pytest.raises(SentenceTrainingDefect, match="optimizer_contract"):
        apply_sentence_step(
            model, optimizer, "codec_warmup", lambda: torch.tensor(0.0), clip_norm=1.0
        )


def test_phase_and_optimizer_hyperparameters_cannot_be_inferred() -> None:
    model = tiny_model()
    with pytest.raises(ValueError, match="phase"):
        sentence_roles(model, "unknown")
    with pytest.raises(ValueError, match="frozen"):
        sentence_roles(model, "prediction")
    for learning_rate, decay in ((0, 0), (float("nan"), 0), (1e-3, -1)):
        with pytest.raises(ValueError):
            build_sentence_optimizer(
                model, "codec_warmup", learning_rate=learning_rate, weight_decay=decay
            )


def test_nonfinite_aggregate_is_structured_even_when_elementwise_gradients_are_finite() -> None:
    model = tiny_model()
    optimizer = build_sentence_optimizer(
        model, "codec_warmup", learning_rate=1e-3, weight_decay=0.0
    )
    parameters = tuple(model.codec.parameters())
    with pytest.raises(SentenceTrainingDefect, match="gradient_clipping"):
        apply_sentence_step(
            model,
            optimizer,
            "codec_warmup",
            lambda: sum((p.sum() * 1e30 for p in parameters), parameters[0].new_zeros(())),
            clip_norm=1.0,
        )
    assert not optimizer.state
