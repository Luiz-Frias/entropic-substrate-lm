"""Checkpoint format versioning and the ``tau`` semantics change.

``tau`` was an effective temperature multiplied straight into the logits; it is
now an unconstrained pre-activation read through ``softplus``. Both eras write
the same tensor under the same name and *both* predate the version stamp, so an
unversioned ``tau`` is genuinely ambiguous — 0.54 is a plausible temperature and
a plausible pre-activation. These tests pin the refusal, not a guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from entropic_substrate.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    TAU_SEMANTICS_ENV,
    CheckpointFormatError,
    inverse_softplus,
    normalize_checkpoint,
)
from entropic_substrate.train import Trainer
from helpers import tiny_config

pytestmark = pytest.mark.unit


def _kappa_trainer(tmp_path: Path) -> Trainer:
    cfg = tiny_config(tmp_path, block="ngpt", norm="ngpt_norm", attn_kernel="kappa")
    cfg.name = "pytest-e3"
    return Trainer(cfg)


def _legacy(state: dict[str, Any], tau: float | None = 1.0) -> dict[str, Any]:
    """Strip the version stamp; optionally rewrite tau to the legacy meaning."""
    out = {k: v for k, v in state.items() if k != "format_version"}
    if tau is not None:
        out["model"] = {
            k: (torch.full_like(v, tau) if k.endswith("tau") else v)
            for k, v in state["model"].items()
        }
    return out


# ------------------------------------------------------------ inverse_softplus
@pytest.mark.parametrize("y", [1e-4, 0.5, 1.0, 1.3132616, 5.0, 50.0])
def test_inverse_softplus_round_trips(y: float) -> None:
    x = inverse_softplus(torch.tensor([y], dtype=torch.float32))
    torch.testing.assert_close(F.softplus(x), torch.tensor([y]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("y", [0.0, -1.0])
def test_inverse_softplus_rejects_non_positive(y: float) -> None:
    """softplus has range (0, inf): a non-positive temperature has no preimage."""
    with pytest.raises(ValueError, match="positive"):
        inverse_softplus(torch.tensor([y]))


# --------------------------------------------------------------- versioning
def test_saved_checkpoints_carry_the_format_version(tmp_path: Path) -> None:
    trainer = _kappa_trainer(tmp_path)
    state = torch.load(trainer.save_checkpoint("v"), map_location="cpu", weights_only=False)
    assert state["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_current_version_passes_through_untouched(tmp_path: Path) -> None:
    trainer = _kappa_trainer(tmp_path)
    state = trainer.state_dict()
    before = {k: v.clone() for k, v in state["model"].items() if k.endswith("tau")}
    out = normalize_checkpoint(state, source="mem")
    for key, value in before.items():
        torch.testing.assert_close(out["model"][key], value, atol=0, rtol=0)


def test_future_version_is_refused(tmp_path: Path) -> None:
    state = _kappa_trainer(tmp_path).state_dict()
    state["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    with pytest.raises(CheckpointFormatError, match="newer"):
        normalize_checkpoint(state, source="mem")


# ------------------------------------------- unversioned WITHOUT tau: fine
def test_unversioned_checkpoint_without_tau_loads(tmp_path: Path) -> None:
    """E0/E1/E2 predate the kappa kernel entirely — nothing is ambiguous."""
    trainer = Trainer(tiny_config(tmp_path, block="ngpt", norm="ngpt_norm"))
    state = _legacy(trainer.state_dict(), tau=None)
    assert not [k for k in state["model"] if k.endswith("tau")]
    out = normalize_checkpoint(state, source="mem")
    assert out["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_unversioned_e2_checkpoint_still_resumes(tmp_path: Path) -> None:
    """Regression: the existing e1/e2 artifacts must stay loadable."""
    cfg = tiny_config(tmp_path, block="ngpt", norm="ngpt_norm")
    trainer = Trainer(cfg)
    trainer.train_step()
    path = trainer.save_checkpoint("legacy")
    state = _legacy(torch.load(path, map_location="cpu", weights_only=False), tau=None)
    torch.save(state, path)

    revived = Trainer(tiny_config(tmp_path, block="ngpt", norm="ngpt_norm"))
    revived.load_checkpoint(path)
    assert revived.step == 1


# ---------------------------------------------- unversioned WITH tau: refuse
def test_unversioned_checkpoint_with_tau_is_refused(tmp_path: Path) -> None:
    state = _legacy(_kappa_trainer(tmp_path).state_dict())
    with pytest.raises(CheckpointFormatError) as excinfo:
        normalize_checkpoint(state, source="models/e3-final.pt")
    message = str(excinfo.value)
    assert TAU_SEMANTICS_ENV in message
    assert "models/e3-final.pt" in message


def test_refusal_reaches_the_trainer_resume_path(tmp_path: Path) -> None:
    trainer = _kappa_trainer(tmp_path)
    path = trainer.save_checkpoint("ambiguous")
    torch.save(_legacy(torch.load(path, map_location="cpu", weights_only=False)), path)
    with pytest.raises(CheckpointFormatError, match=TAU_SEMANTICS_ENV):
        _kappa_trainer(tmp_path).load_checkpoint(path)


# ------------------------------------------------- operator-declared eras
def test_declared_softplus_era_loads_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The common case: an unversioned checkpoint written AFTER the change."""
    monkeypatch.setenv(TAU_SEMANTICS_ENV, "softplus")
    state = _legacy(_kappa_trainer(tmp_path).state_dict(), tau=0.5390461)
    out = normalize_checkpoint(state, source="mem")
    tau = next(v for k, v in out["model"].items() if k.endswith("tau"))
    torch.testing.assert_close(tau, torch.full_like(tau, 0.5390461), atol=1e-7, rtol=0)
    assert out["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert TAU_SEMANTICS_ENV in capsys.readouterr().err  # declared, and said so


def test_declared_temperature_era_is_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conversion preserves BEHAVIOUR: softplus(new tau) == the old temperature."""
    monkeypatch.setenv(TAU_SEMANTICS_ENV, "temperature")
    state = _legacy(_kappa_trainer(tmp_path).state_dict(), tau=1.0)
    out = normalize_checkpoint(state, source="mem")
    taus = [v for k, v in out["model"].items() if k.endswith("tau")]
    assert taus
    for tau in taus:
        torch.testing.assert_close(F.softplus(tau), torch.ones_like(tau), atol=1e-6, rtol=0)


def test_declared_temperature_refuses_a_non_positive_tau(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative legacy temperature inverted the kernel; softplus cannot express it."""
    monkeypatch.setenv(TAU_SEMANTICS_ENV, "temperature")
    state = _legacy(_kappa_trainer(tmp_path).state_dict(), tau=-0.5)
    with pytest.raises(CheckpointFormatError, match="positive"):
        normalize_checkpoint(state, source="mem")


def test_declared_temperature_refuses_optimizer_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AdamW moments for tau are in the pre-conversion parameterisation, so a
    converted resume is not the exact resume the state dict promises."""
    monkeypatch.setenv(TAU_SEMANTICS_ENV, "temperature")
    state = _legacy(_kappa_trainer(tmp_path).state_dict(), tau=1.0)
    with pytest.raises(CheckpointFormatError, match="resume"):
        normalize_checkpoint(state, source="mem", for_resume=True)


def test_unknown_declaration_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAU_SEMANTICS_ENV, "banana")
    state = _legacy(_kappa_trainer(tmp_path).state_dict())
    with pytest.raises(CheckpointFormatError, match="banana"):
        normalize_checkpoint(state, source="mem")
