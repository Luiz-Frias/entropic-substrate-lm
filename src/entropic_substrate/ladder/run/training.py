"""Phase-specific ownership and checked sentence-model optimizer transactions."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypedDict

import torch
from torch import nn

from entropic_substrate.core.sentence.model import SentencePredictor

type SentencePhase = Literal["codec_warmup", "prediction"]


class SentenceTrainingDefect(ValueError):
    """A failed execution invariant, not a scientific collapse classification."""

    def __init__(self, code: str, component: str) -> None:
        self.code, self.component = code, component
        super().__init__(f"{code}: {component}")


class GradientMeasurement(TypedDict):
    status: str
    norm: float | None


@dataclass(frozen=True)
class SentenceStep:
    objective: float
    aggregate_norm: float
    gradients: dict[str, GradientMeasurement]


def sentence_roles(
    model: SentencePredictor, phase: SentencePhase
) -> tuple[dict[str, tuple[nn.Parameter, ...]], dict[str, str]]:
    """Explicit inactive reasons, never infer expected participation from gradients."""
    if phase not in ("codec_warmup", "prediction"):
        raise ValueError("unknown sentence training phase")
    roles = {
        "encoder": tuple(model.codec.encoder.parameters()),
        "decoder": tuple(model.codec.decoder.parameters()),
        "spine": tuple(model.spine.parameters()),
        "future_law": tuple(model.head.parameters()),
    }
    if phase == "prediction":
        if any(p.requires_grad for p in roles.pop("encoder")):
            raise ValueError("prediction requires a frozen encoder")
        absent: dict[str, str] = {"encoder": "structurally_absent_frozen_encoder"}
    else:
        roles.pop("spine")
        roles.pop("future_law")
        absent = dict.fromkeys(("spine", "future_law"), "structurally_absent_codec_warmup")
    if any(not p.requires_grad for parameters in roles.values() for p in parameters):
        raise ValueError("active phase component contains a frozen parameter")
    return roles, absent


def build_sentence_optimizer(
    model: SentencePredictor,
    phase: SentencePhase,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    epsilon: float = 1e-8,
) -> torch.optim.AdamW:
    """Matrix-only AdamW decay; explicit rates, development moments shown in config."""
    if any(
        type(v) not in (int, float) or not math.isfinite(v) for v in (learning_rate, weight_decay)
    ):
        raise ValueError("optimizer rates must be finite numbers")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("invalid learning rate or weight decay")
    roles, _ = sentence_roles(model, phase)
    parameters = [p for group in roles.values() for p in group]
    return torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        weight_decay=weight_decay,
    )


def require_sentence_adamw_contract(optimizer: torch.optim.Optimizer, phase: SentencePhase) -> None:
    """Match the builder's grouping and update policy, not merely parameter IDs."""
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 2:
        raise SentenceTrainingDefect("optimizer_contract", phase)
    for index, group in enumerate(optimizer.param_groups):
        if any((p.ndim >= 2) != (index == 0) for p in group["params"]):
            raise SentenceTrainingDefect("optimizer_contract", phase)
        decay = optimizer.defaults["weight_decay"] if index == 0 else 0.0
        if group["weight_decay"] != decay or group["maximize"] or group["differentiable"]:
            raise SentenceTrainingDefect("optimizer_contract", phase)
        for key in ("lr", "betas", "eps"):
            if group[key] != optimizer.defaults[key]:
                raise SentenceTrainingDefect("optimizer_contract", phase)


def apply_sentence_step(
    model: SentencePredictor,
    optimizer: torch.optim.Optimizer,
    phase: SentencePhase,
    objective: Callable[[], torch.Tensor],
    *,
    clip_norm: float,
) -> SentenceStep:
    """Measure before clipping; successful zero transactions still advance AdamW.

    A failure after optimizer mutation is not rolled back or called committed:
    the runner must retain its defect and recover the last sealed checkpoint.
    Spherical undefined-direction gaps propagate separately from these defects.
    """
    if type(clip_norm) not in (int, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("gradient clip norm must be finite and positive")
    roles, absent = sentence_roles(model, phase)
    expected = [p for parameters in roles.values() for p in parameters]
    expected_ids = {id(p) for p in expected}
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    # Equal lengths and sets still admit a parameter listed twice on both sides (tied
    # weights across roles): AdamW would step and decay it twice, invisibly to any norm
    # when its gradient is zero, so multiplicity is checked exactly here.
    if (
        len(expected_ids) != len(expected)
        or len(owned) != len(expected)
        or {id(p) for p in owned} != expected_ids
    ):
        raise SentenceTrainingDefect("optimizer_ownership", phase)
    require_sentence_adamw_contract(optimizer, phase)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = objective()
    if loss.ndim != 0 or not bool(torch.isfinite(loss)) or not loss.requires_grad:
        raise SentenceTrainingDefect("invalid_objective", phase)
    loss.backward()
    if any(p.grad is not None for p in model.parameters() if id(p) not in expected_ids):
        raise SentenceTrainingDefect("unexpected_gradient", phase)
    gradients: dict[str, GradientMeasurement] = {
        name: {"status": reason, "norm": None} for name, reason in absent.items()
    }
    for name, parameters in roles.items():
        if any(p.grad is None for p in parameters):
            raise SentenceTrainingDefect("missing_gradient", name)
        measured = [p.grad.detach() for p in parameters if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in measured):
            raise SentenceTrainingDefect("nonfinite_gradient", name)
        norm = math.sqrt(sum(float(g.double().square().sum()) for g in measured))
        if not math.isfinite(norm):
            raise SentenceTrainingDefect("nonfinite_gradient_norm", name)
        gradients[name] = {
            "status": "measured_zero" if norm == 0 else "measured_nonzero",
            "norm": norm,
        }
    # gradient_reconciliation means set identity, not a norm: the model's gradient-carrying
    # parameters must be exactly the role parameters. Roles, `expected` and the optimizer
    # all derive from sentence_roles and the scan above covers only model.parameters(), so
    # a role parameter the model does not own passes every other check; a zero-gradient
    # parameter is invisible to any norm comparison.
    # RETRACTED 2026-10-27: this guard compared the fp32 clip_grad_norm_ total against
    # float64 role norms at rel_tol=1e-5, which measured fp32 precision, not the partition.
    # MEASURED gaps 5.3e-7 at 85,216 and 6.85e-6 at 15,183,744 parameters on warm-up step
    # 1, later dev-width warm-up steps crossed 1e-5, and 3.253e-4 at 15,360,000 parameters
    # over 100 tensors refused a correct step; the clip value stays the step's aggregate
    # measurement only.
    if {id(p) for p in model.parameters() if p.grad is not None} != expected_ids:
        raise SentenceTrainingDefect("gradient_reconciliation", phase)
    try:
        aggregate = float(nn.utils.clip_grad_norm_(expected, clip_norm, error_if_nonfinite=True))
    except RuntimeError as error:
        raise SentenceTrainingDefect("gradient_clipping", phase) from error
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in expected):
        raise SentenceTrainingDefect("nonfinite_updated_parameter", phase)
    if phase == "prediction":
        model.spine.normalize_weights()
    return SentenceStep(float(loss.detach()), aggregate, gradients)
