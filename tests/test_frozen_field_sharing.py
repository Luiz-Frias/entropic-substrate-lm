"""One pretrained p_phi, shared by every frozen E1.5c arm.

``docs/e1-5c-preregistration.md`` registers the frozen wave as F, H and the
matched-total twins ALL loading **the same** ``field_ckpt_path`` — that
shared field is the attribution the four-arm contract rests on (identical
field weights under every construction of ``z_t``).

MEASURED 2026-08-21, against the real pretrained artifact: a whole-module
``load_state_dict`` cannot deliver that contract, because the checkpoint
carries two tensors that are NOT part of p_phi's learned content:

* ``lift`` — the FIXED ``semi_orthogonal_lift(256, d_model, seed=0)`` buffer,
  sized by the BACKBONE's width. The matched-total twins run d_model=756
  against a checkpoint written at 768: ``size mismatch for lift``.
* ``state_lift`` — H's own trainable adapter, which an F-shaped checkpoint
  has never heard of: ``Missing key(s): "state_lift.weight"``.

p_phi's real body (``wte`` / ``blocks`` / ``norm_f`` / the tied ``head``) is
sized by ``field_d_model`` ALONE, so it transfers to every arm at every
backbone width. The loader therefore restores the body and leaves the two
arm-local tensors to their own construction: ``lift`` is deterministic and
rebuilt at the right width, ``state_lift`` keeps its fresh init AND STAYS
TRAINABLE — H is defined as "the only arm with a trainable lift"
(``docs/e1_5-instruction.md`` Repair 2), so freezing it would silently
measure a different arm than the one registered.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION
from entropic_substrate.init.fisher import semi_orthogonal_lift
from entropic_substrate.models.field_encoder import FieldEncoder
from helpers import assert_bitwise_equal, tiny_model_config

pytestmark = pytest.mark.unit


def _field_cfg(**overrides: object):
    """A tiny FIELD-arm ModelConfig (the E1.5c axis plus shrunk widths)."""
    base: dict[str, object] = {
        "state_source": "field",
        "field_d_model": 32,
        "field_n_layers": 1,
        "field_n_heads": 2,
    }
    base.update(overrides)
    return tiny_model_config(**base)


def _write_ckpt(tmp_path: Path, cfg) -> tuple[Path, FieldEncoder]:
    """Persist a trained-shaped p_phi in the wire format the loader reads."""
    torch.manual_seed(11)
    source = FieldEncoder(cfg)
    path = tmp_path / "pphi.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": 20_000,
            "model": source.state_dict(),
        },
        path,
    )
    return path, source


def _frozen(cfg, path: Path) -> FieldEncoder:
    return FieldEncoder(dataclasses.replace(cfg, field_mode="frozen", field_ckpt_path=str(path)))


# --------------------------------------------------- the shared-field contract
def test_frozen_load_restores_the_field_body(tmp_path: Path) -> None:
    path, source = _write_ckpt(tmp_path, _field_cfg())
    loaded = _frozen(_field_cfg(), path)
    assert_bitwise_equal(loaded.wte.weight, source.wte.weight)
    assert_bitwise_equal(loaded.head.weight, source.head.weight)
    for got, want in zip(loaded.blocks.parameters(), source.blocks.parameters(), strict=True):
        assert_bitwise_equal(got, want)


def test_one_checkpoint_loads_into_the_hidden_arm(tmp_path: Path) -> None:
    """H shares F's p_phi even though an F-shaped checkpoint has no
    ``state_lift`` — RED before the loader stopped demanding every key."""
    path, source = _write_ckpt(tmp_path, _field_cfg())
    loaded = _frozen(_field_cfg(field_state="hidden"), path)
    assert loaded.state_lift is not None
    assert_bitwise_equal(loaded.wte.weight, source.wte.weight)


def test_one_checkpoint_loads_at_a_different_backbone_width(tmp_path: Path) -> None:
    """The matched-total lane shrinks the BACKBONE (d_model), which resizes
    the ``lift`` buffer but not p_phi — so the same field must still load.
    RED before the loader stopped taking ``lift`` from the checkpoint."""
    path, source = _write_ckpt(tmp_path, _field_cfg(d_model=64))
    loaded = _frozen(_field_cfg(d_model=32), path)
    assert loaded.lift.shape == (256, 32)
    assert_bitwise_equal(loaded.wte.weight, source.wte.weight)


def test_lift_is_reconstructed_not_restored(tmp_path: Path) -> None:
    """``lift`` is deterministic (seed 0), so the arm's own construction is
    the source of truth — never the checkpoint's copy at another width."""
    path, _ = _write_ckpt(tmp_path, _field_cfg())
    loaded = _frozen(_field_cfg(), path)
    expected = semi_orthogonal_lift(256, 32, seed=0).float()
    torch.testing.assert_close(loaded.lift, expected, atol=0, rtol=0)


# ------------------------------------------------------------ freeze semantics
def test_field_body_is_frozen(tmp_path: Path) -> None:
    path, _ = _write_ckpt(tmp_path, _field_cfg())
    loaded = _frozen(_field_cfg(), path)
    assert not loaded.wte.weight.requires_grad
    assert all(not p.requires_grad for p in loaded.blocks.parameters())
    assert all(not p.requires_grad for p in loaded.norm_f.parameters())


def test_hidden_arms_state_lift_stays_trainable(tmp_path: Path) -> None:
    """The arm-defining guard: H is 'the only arm with a trainable lift'.
    Freezing the whole module would leave a frozen random projection over a
    frozen hidden state — a DIFFERENT experiment than the registered H."""
    path, _ = _write_ckpt(tmp_path, _field_cfg())
    loaded = _frozen(_field_cfg(field_state="hidden"), path)
    assert loaded.state_lift is not None
    assert loaded.state_lift.weight.requires_grad, (
        "H's trainable lift must survive the frozen-p_phi load — freezing it "
        "measures an arm this registration never declared"
    )
    # ...while the body it reads from is still frozen.
    assert not loaded.wte.weight.requires_grad


def test_frozen_body_receives_no_gradient(tmp_path: Path) -> None:
    """requires_grad is a flag; this checks the BEHAVIOUR it promises."""
    path, _ = _write_ckpt(tmp_path, _field_cfg(field_state="hidden"))
    loaded = _frozen(_field_cfg(field_state="hidden"), path)
    idx = torch.randint(0, 256, (2, 8))
    loaded(idx).z.sum().backward()
    assert loaded.wte.weight.grad is None
    assert loaded.state_lift is not None
    assert loaded.state_lift.weight.grad is not None


# ------------------------------------------------------------- guard teeth
def test_a_genuinely_wrong_field_shape_still_fails_loud(tmp_path: Path) -> None:
    """The loosening must not become 'load whatever happens to fit'. A
    checkpoint whose p_phi BODY was trained at another width is a real
    mismatch — it must still raise, not silently keep the fresh init."""
    path, _ = _write_ckpt(tmp_path, _field_cfg(field_d_model=64, field_n_heads=2))
    with pytest.raises(RuntimeError, match=r"size mismatch|shape"):
        _frozen(_field_cfg(field_d_model=32), path)


def test_a_checkpoint_missing_the_field_body_fails_loud(tmp_path: Path) -> None:
    """An empty/foreign payload must refuse rather than train an
    unpretrained 'frozen' field — the frozen arm's whole premise is that
    p_phi was trained elsewhere."""
    path = tmp_path / "empty.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "step": 1, "model": {}}, path)
    with pytest.raises(RuntimeError, match=r"no p_phi body|Missing key"):
        _frozen(_field_cfg(), path)
