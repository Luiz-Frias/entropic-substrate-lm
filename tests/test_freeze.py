"""E1.5a embedding freeze: model-build-time freeze + the nGPT bypass guard.

``embedding_freeze`` (config.py) names which half of the (possibly untied)
embedding stays at its init value while the rest of the backbone trains:
"input" freezes ``wte.weight``, "output" freezes ``lm_head.weight``, "both"
freezes both. A tied table (``tie_embeddings=True``) cannot be frozen on only
ONE of its two roles — ``wte.weight`` and ``lm_head.weight`` are the SAME
``nn.Parameter`` when tied, so "input"/"output" require ``tie_embeddings:
False`` (entailed); "both" freezing a tied table is consistent and allowed.

Freezing happens in :class:`LanguageModel.__init__`, BEFORE
``Trainer._build_optimizer`` (train.py:190) ever runs — the optimizer's
existing ``if not param.requires_grad: continue`` filter therefore excludes
frozen params by construction, with no freeze-awareness of its own needed.

Two independent hazards guard the frozen table's BITS, not just its
``requires_grad`` flag:

1. wiring absent — nothing sets ``requires_grad_(False)`` at all, so the
   optimiser updates the "frozen" table like any other parameter.
2. the nGPT bypass trap — ``normalize_weights()`` (backbone.py) runs
   unconditionally after every optimiser step and does an in-place
   ``weight.copy_(weight / weight.norm(...))`` on ``wte.weight`` /
   ``lm_head.weight``. Even a frozen table starts unit-norm from init to
   fp32 precision (recomputed row norms measure ~1.19e-07 off 1.0), but
   dividing by that not-bit-exactly-``1.0`` norm still changes the bits
   every step — ``requires_grad=False`` does not stop an in-place
   ``copy_``. ``test_frozen_table_bitwise_unchanged_after_training``
   is RED on both hazards independently: on ``baseline`` it is RED until (1)
   is fixed (baseline's ``normalize_weights`` is a no-op, so (2) never bites
   there); on ``ngpt`` it stays RED until (2) is ALSO fixed, even after (1)
   is wired up.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from entropic_substrate.models.backbone import build_model
from entropic_substrate.train import Trainer
from helpers import assert_bitwise_equal, tiny_config, tiny_model_config

pytestmark = pytest.mark.unit


# --------------------------------------------------------- _init_embeddings
@pytest.mark.parametrize("freeze", ["none", "input", "output", "both"])
def test_untied_init_seeds_both_tables(freeze: str) -> None:
    """Untied + non-default init: the output table starts at the SAME point
    the tied E1 arm's shared table would have — not a fresh random draw.

    Parametrized over freeze modes: seeding lives in ``_init_embeddings``
    and conditions on ``tie_embeddings``, never on freeze, and ``__init__``
    applies freeze AFTER init — so no freeze mode may perturb the seeded
    point."""
    model = build_model(
        tiny_model_config(
            embedding_init="random_sphere",
            tie_embeddings=False,
            embedding_freeze=freeze,
        )
    )
    assert model.wte.weight.data_ptr() != model.lm_head.weight.data_ptr()
    assert_bitwise_equal(model.wte.weight, model.lm_head.weight)


def test_untied_default_init_leaves_embeddings_untouched() -> None:
    """embedding_init='default' (blank case): no copy happens to either table."""
    torch.manual_seed(0)
    model = build_model(tiny_model_config(embedding_init="default", tie_embeddings=False))
    # Independently-initialised Linear/Embedding normal draws are not equal.
    assert not torch.allclose(model.wte.weight, model.lm_head.weight)


# ------------------------------------------------------------------ freeze
def test_freeze_none_leaves_everything_trainable() -> None:
    model = build_model(tiny_model_config(embedding_freeze="none"))
    assert model.wte.weight.requires_grad
    assert model.lm_head.weight.requires_grad


@pytest.mark.parametrize("mode", ["input", "output"])
def test_partial_freeze_rejects_tied_embeddings(mode: str) -> None:
    cfg = tiny_model_config(embedding_freeze=mode, tie_embeddings=True)
    with pytest.raises(ValueError, match="tie_embeddings"):
        build_model(cfg)


def test_both_freeze_permits_tied_embeddings() -> None:
    """Unlike input/output, freezing BOTH roles of one shared table is
    consistent — there is no partial-freeze contradiction to reject."""
    model = build_model(tiny_model_config(embedding_freeze="both", tie_embeddings=True))
    assert model.wte.weight is model.lm_head.weight
    assert not model.wte.weight.requires_grad
    assert not model.lm_head.weight.requires_grad


@pytest.mark.parametrize(
    ("mode", "frozen_attr", "trainable_attr"),
    [("input", "wte", "lm_head"), ("output", "lm_head", "wte")],
)
def test_partial_freeze_freezes_only_the_named_half(
    mode: str, frozen_attr: str, trainable_attr: str
) -> None:
    model = build_model(tiny_model_config(embedding_freeze=mode, tie_embeddings=False))
    frozen = getattr(model, frozen_attr).weight
    trainable = getattr(model, trainable_attr).weight
    assert not frozen.requires_grad
    assert trainable.requires_grad


def test_freeze_rejects_unknown_mode() -> None:
    """Literal types are not runtime-enforced; an out-of-band value must
    still fail loud rather than silently freezing nothing."""
    cfg = tiny_model_config(embedding_freeze="both")
    cfg.embedding_freeze = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown embedding_freeze"):
        build_model(cfg)


# ------------------------------------------------------- bitwise unchanged
@pytest.mark.parametrize(
    ("block", "norm"),
    [("baseline", "layernorm"), ("ngpt", "ngpt_norm")],
    ids=["baseline", "ngpt"],
)
def test_frozen_table_bitwise_unchanged_after_training(
    tmp_path: Path, block: str, norm: str
) -> None:
    cfg = tiny_config(
        tmp_path,
        block=block,
        norm=norm,
        embedding_init="random_sphere",
        embedding_freeze="both",
        tie_embeddings=True,
    )
    trainer = Trainer(cfg)
    before = trainer.raw_model.wte.weight.clone()
    for _ in range(3):
        trainer.train_step()
    assert_bitwise_equal(trainer.raw_model.wte.weight, before)


def test_trainable_side_still_learns(tmp_path: Path) -> None:
    cfg = tiny_config(
        tmp_path,
        embedding_init="random_sphere",
        embedding_freeze="input",
        tie_embeddings=False,
    )
    trainer = Trainer(cfg)
    before = trainer.raw_model.lm_head.weight.clone()
    for _ in range(3):
        trainer.train_step()
    assert not torch.allclose(trainer.raw_model.lm_head.weight, before)


def test_frozen_params_absent_from_optimizer(tmp_path: Path) -> None:
    cfg = tiny_config(
        tmp_path,
        embedding_init="random_sphere",
        embedding_freeze="both",
        tie_embeddings=True,
    )
    trainer = Trainer(cfg)
    optimizer_param_ids = {
        id(p) for group in trainer.optimizer.param_groups for p in group["params"]
    }
    assert id(trainer.raw_model.wte.weight) not in optimizer_param_ids


@pytest.mark.parametrize(
    ("mode", "frozen_attr", "trainable_attr"),
    [("input", "wte", "lm_head"), ("output", "lm_head", "wte")],
)
def test_ngpt_partial_freeze_preserves_frozen_table(
    tmp_path: Path, mode: str, frozen_attr: str, trainable_attr: str
) -> None:
    """The guard must skip only the FROZEN half — the trainable half still
    gets renormalised every step, same as an unfrozen nGPT run.

    Untied "input"/"output" freeze puts wte.weight and lm_head.weight in TWO
    distinct entries of the ``normalize_weights`` dedup dict (unlike "both"
    +tied, which collapses to one), so this is the branch that actually
    exercises the guard discriminating a frozen entry from a trainable one
    within the same iteration.
    """
    cfg = tiny_config(
        tmp_path,
        block="ngpt",
        norm="ngpt_norm",
        embedding_init="random_sphere",
        embedding_freeze=mode,
        tie_embeddings=False,
    )
    trainer = Trainer(cfg)
    frozen_before = getattr(trainer.raw_model, frozen_attr).weight.detach().clone()
    trainable_before = getattr(trainer.raw_model, trainable_attr).weight.detach().clone()
    for _ in range(3):
        trainer.train_step()
    assert_bitwise_equal(getattr(trainer.raw_model, frozen_attr).weight, frozen_before)
    weight = getattr(trainer.raw_model, trainable_attr).weight
    # The trainable half must have actually MOVED — the init is already
    # unit-norm, so the row-norm check alone would pass a run where training
    # never touched this table at all.
    assert not torch.equal(weight, trainable_before)
    norms = weight.norm(p=2, dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


# --------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_with_untied_frozen_table(tmp_path: Path) -> None:
    cfg = tiny_config(
        tmp_path,
        embedding_init="random_sphere",
        embedding_freeze="output",
        tie_embeddings=False,
    )
    trainer_a = Trainer(cfg)
    for _ in range(2):
        trainer_a.train_step()
    path = trainer_a.save_checkpoint("freeze-roundtrip")

    trainer_b = Trainer(cfg)
    trainer_b.load_checkpoint(path)

    assert trainer_b.step == trainer_a.step
    assert not trainer_b.raw_model.lm_head.weight.requires_grad
    torch.testing.assert_close(
        trainer_b.raw_model.lm_head.weight,
        trainer_a.raw_model.lm_head.weight,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        trainer_b.raw_model.wte.weight,
        trainer_a.raw_model.wte.weight,
        atol=0,
        rtol=0,
    )


# -------------------------------------------------------------- attestation
# MAJOR-1 (docs/reviews/e1-5a-review-cloud.md): the double-blind packet scrubs
# every freeze-axis config field, so a reviewer cannot verify that the arms
# froze DIFFERENT parameter subsets at all. ``parameter_attestation`` is the
# unblinded channel that discharges it without naming the axis.
def test_attestation_of_an_unfrozen_model_is_the_empty_mask() -> None:
    """Every run logs a digest, control arms included — so the KEY's presence
    never distinguishes an arm. The empty mask hashes the empty string."""
    model = build_model(tiny_model_config(embedding_freeze="none"))
    attestation = model.parameter_attestation()
    assert attestation["frozen_param_count"] == 0
    assert attestation["trainable_param_count"] == model.num_parameters()
    assert attestation["frozen_mask_digest"] == hashlib.sha256(b"").hexdigest()


def test_attestation_distinguishes_the_two_freeze_variants() -> None:
    """The digest, not the counts, is what separates these two arms.

    MEASURED: untied "input" freezes {wte.weight}; tied "both" freezes one
    shared nn.Parameter under BOTH role names — same 8192 frozen elements at
    tiny scale, so ``frozen_param_count`` alone is blind to the difference.
    The digest therefore enumerates parameters WITHOUT torch's shared-tensor
    dedup (``remove_duplicate=False``), which is also the honest reading:
    freezing both roles of one table is not freezing one role of two.
    """
    untied = build_model(tiny_model_config(embedding_freeze="input", tie_embeddings=False))
    tied = build_model(tiny_model_config(embedding_freeze="both", tie_embeddings=True))
    a = untied.parameter_attestation()
    b = tied.parameter_attestation()

    assert a["frozen_mask_digest"] != b["frozen_mask_digest"]
    for model, attestation in ((untied, a), (tied, b)):
        assert attestation["frozen_param_count"] == model.wte.weight.numel()
        assert (
            attestation["trainable_param_count"] + attestation["frozen_param_count"]
            == model.num_parameters()
        )


def test_attestation_digest_is_stable_across_builds() -> None:
    """Constant within an arm: the digest reads names and shapes, never the
    draw — so it cannot leak a seed and cannot vary across a seed sweep."""
    kwargs = {"embedding_freeze": "output", "tie_embeddings": False}
    first = build_model(tiny_model_config(**kwargs)).parameter_attestation()
    torch.manual_seed(1234)
    second = build_model(tiny_model_config(**kwargs)).parameter_attestation()
    assert first == second


def test_attestation_digest_ignores_weight_values() -> None:
    """A pure function of WHICH parameters are frozen. If weights entered the
    digest it would differ per seed and per step, and the packet's per-arm
    equality structure — the whole point — would dissolve into noise."""
    model = build_model(tiny_model_config(embedding_freeze="input", tie_embeddings=False))
    before = model.parameter_attestation()
    with torch.no_grad():
        model.wte.weight.add_(1.0)
    assert model.parameter_attestation() == before
