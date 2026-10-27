"""Optimizer transactions distinguish execution defects from finite zero evidence."""

from collections.abc import Callable, Sequence
from typing import cast

import pytest
import torch
from tests.test_sentence_model import history, tiny_model
from torch import nn

from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.objective import codec_warmup_loss, prediction_loss
from entropic_substrate.ladder.run import training
from entropic_substrate.ladder.run.training import (
    SentencePhase,
    SentenceStep,
    SentenceTrainingDefect,
    apply_sentence_step,
    build_sentence_optimizer,
    sentence_roles,
)

pytestmark = pytest.mark.unit


class _Codec(nn.Module):
    def __init__(self, encoder: Sequence[nn.Parameter], decoder: Sequence[nn.Parameter]) -> None:
        super().__init__()
        self.encoder, self.decoder = nn.ParameterList(encoder), nn.ParameterList(decoder)


class _Predictor(nn.Module):
    """Only the role surface `sentence_roles` reads; warm-up never touches spine or head."""

    def __init__(self, encoder: Sequence[nn.Parameter], decoder: Sequence[nn.Parameter]) -> None:
        super().__init__()
        self.codec = _Codec(encoder, decoder)
        self.spine, self.head = nn.Linear(1, 1), nn.Linear(1, 1)


def synthetic_step(
    encoder: Sequence[nn.Parameter],
    decoder: Sequence[nn.Parameter],
    gradients: dict[int, torch.Tensor],
) -> tuple[nn.Module, torch.optim.AdamW, Callable[[], SentenceStep]]:
    """Warm-up step whose objective plants exactly `gradients` on each codec parameter."""
    model = _Predictor(encoder, decoder)
    predictor = cast(SentencePredictor, model)
    optimizer = build_sentence_optimizer(
        predictor, "codec_warmup", learning_rate=1e-3, weight_decay=0.0
    )
    unique = list(model.codec.parameters())

    def objective() -> torch.Tensor:
        return sum(((p * gradients[id(p)]).sum() for p in unique), unique[0].new_zeros(()))

    def step() -> SentenceStep:
        return apply_sentence_step(predictor, optimizer, "codec_warmup", objective, clip_norm=1.0)

    return model, optimizer, step


def planted(shapes: Sequence[int], seed: int) -> tuple[list[nn.Parameter], dict[int, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    parameters = [nn.Parameter(torch.zeros(size)) for size in shapes]
    gradients = {
        id(p): torch.randn(p.shape, generator=generator)
        * 10 ** (torch.rand(p.shape, generator=generator) * 4 - 2)
        for p in parameters
    }
    return parameters, gradients


def test_reconciliation_commits_at_development_scale() -> None:
    # MEASURED 2026-10-27 on this fixture, 15,360,000 parameters over 100 tensors
    # (2 x 3.6M + 48 x 10k per role, gradients spanning four decades): the fp32
    # clip_grad_norm_ total sat 3.253e-4 relative from the float64 role norms, so
    # the old 1e-5 comparison refused a correctly partitioned step.
    shapes = [3_600_000] * 2 + [10_000] * 48
    encoder, encoder_gradients = planted(shapes, seed=0)
    decoder, decoder_gradients = planted(shapes, seed=1)
    _, optimizer, step = synthetic_step(encoder, decoder, encoder_gradients | decoder_gradients)
    result = step()
    assert result.gradients["encoder"]["status"] == "measured_nonzero"
    assert result.gradients["decoder"]["status"] == "measured_nonzero"
    assert all(float(optimizer.state[p]["step"]) == 1 for p in encoder + decoder)


@pytest.mark.parametrize("tied_gradient", ["nonzero", "zero"])
def test_parameter_listed_in_two_roles_is_refused_before_forward(tied_gradient: str) -> None:
    # A tied parameter sits in both roles and the optimizer carries the same
    # duplicate, so the old length-and-set ownership check passed and AdamW would
    # step (and decay) it twice. With a zero gradient no norm comparison can see
    # it, so the identity check must be exact.
    encoder, gradients = planted([64, 32], seed=2)
    decoder, decoder_gradients = planted([16], seed=3)
    if tied_gradient == "zero":
        gradients[id(encoder[0])] = torch.zeros(64)
    with pytest.warns(UserWarning, match="duplicate parameters"):
        _, optimizer, step = synthetic_step(
            encoder, [*decoder, encoder[0]], gradients | decoder_gradients
        )
    with pytest.raises(SentenceTrainingDefect, match="optimizer_ownership"):
        step()
    assert not optimizer.state


@pytest.mark.parametrize("stray_gradient", ["nonzero", "zero"])
def test_role_parameter_outside_the_model_is_a_reconciliation_defect(
    monkeypatch: pytest.MonkeyPatch, stray_gradient: str
) -> None:
    # The roles, the optimizer and `expected` all derive from sentence_roles, and
    # unexpected_gradient scans only model.parameters(); a role parameter the model
    # does not own passes all of them. With a zero gradient no norm comparison can
    # see it, so only an identity reconciliation against the model exposes it.
    encoder, gradients = planted([64, 32], seed=5)
    decoder, decoder_gradients = planted([16], seed=6)
    stray, stray_gradients = planted([8], seed=7)
    if stray_gradient == "zero":
        stray_gradients[id(stray[0])] = torch.zeros(8)
    real_roles = training.sentence_roles

    def roles_with_stray(
        model: SentencePredictor, phase: SentencePhase
    ) -> tuple[dict[str, tuple[nn.Parameter, ...]], dict[str, str]]:
        roles, absent = real_roles(model, phase)
        return roles | {"decoder": (*roles["decoder"], *stray)}, absent

    monkeypatch.setattr(training, "sentence_roles", roles_with_stray)
    model, optimizer, _ = synthetic_step(encoder, decoder, gradients | decoder_gradients)
    planted_gradients = gradients | decoder_gradients | stray_gradients

    def objective() -> torch.Tensor:
        owned = [*model.codec.parameters(), *stray]
        return sum(((p * planted_gradients[id(p)]).sum() for p in owned), stray[0].new_zeros(()))

    with pytest.raises(SentenceTrainingDefect, match="gradient_reconciliation"):
        apply_sentence_step(
            cast(SentencePredictor, model), optimizer, "codec_warmup", objective, clip_norm=1.0
        )
    assert not optimizer.state


def test_all_zero_role_beside_a_single_tensor_role_commits() -> None:
    encoder, gradients = planted([64, 32, 8], seed=4)
    decoder = [nn.Parameter(torch.zeros(16))]
    gradients[id(decoder[0])] = torch.zeros(16)
    _, optimizer, step = synthetic_step(encoder, decoder, gradients)
    result = step()
    assert result.gradients["decoder"] == {"status": "measured_zero", "norm": 0.0}
    assert result.gradients["encoder"]["status"] == "measured_nonzero"
    assert float(optimizer.state[decoder[0]]["step"]) == 1


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
