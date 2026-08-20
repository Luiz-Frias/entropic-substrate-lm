"""Config loading / merging and the shipped experiment YAML files."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from entropic_substrate.config import (
    EXPERIMENTS,
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_from_dict,
    configs_dir,
    is_execution_gated,
    load_config,
    merge_dicts,
    resolve_experiment_config,
)
from entropic_substrate.models.backbone import build_model, vocab_size_for

pytestmark = pytest.mark.unit

# E1.5 arms (docs/e1_5-instruction.md): exact total param counts, pinned.
# The extended formula must reproduce every one of these exactly, not just
# land the shared 80-100M band assertion (which the pre-extension formula
# already satisfies for every arm here, so the band alone is not
# RED-capable for this change).
E1_5_EXACT_TOTAL_PARAMS: dict[str, int] = {
    "e1-fisher": 85_131_264,
    "e1-5a-input": 85_327_872,
    "e1-5a-output": 85_327_872,
    "e1-5a-both": 85_131_264,
    "e1-5b-gram": 85_131_264,
    "e1-5c-field": 85_557_248,
    # P7 (FOR-924): H/P share e1-5c-field's p_phi (field_d_model=128,
    # field_n_layers=2) exactly, so e1-5c-prob (no sqrt, same lift, no new
    # params) lands on the SAME total as e1-5c-field. e1-5c-hidden adds the
    # one trainable lift W: Linear(field_d_model=128, d_model=768,
    # bias=False) = 128*768 = 98_304 params over that shared base.
    "e1-5c-hidden": 85_655_552,
    "e1-5c-prob": 85_557_248,
}


def test_merge_dicts_is_recursive() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert merge_dicts(base, {"a": {"y": 9}, "c": 4}) == {
        "a": {"x": 1, "y": 9},
        "b": 3,
        "c": 4,
    }
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}  # not mutated


def test_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown top-level"):
        config_from_dict({"modle": {}})
    with pytest.raises(ValueError, match="unknown keys for ModelConfig"):
        config_from_dict({"model": {"d_modle": 8}})


def test_every_experiment_config_exists_and_parses() -> None:
    for exp in EXPERIMENTS:
        path = configs_dir() / f"{exp}.yaml"
        assert path.exists(), path
        cfg = load_config(path)
        assert cfg.name == exp
        assert cfg.model.d_model % cfg.model.n_heads == 0


def test_tiny_yaml_exists() -> None:
    assert (configs_dir() / "tiny.yaml").exists()


def test_experiment_ladder_wiring() -> None:
    """Each rung must differ from its predecessor in exactly the intended way."""
    cfgs = {exp: resolve_experiment_config(exp) for exp in EXPERIMENTS}

    assert cfgs["e0-byte"].model.tokenizer == "byte"
    assert cfgs["e0-bpe"].model.tokenizer == "gpt2"

    fisher, random_arm = cfgs["e1-fisher"].model, cfgs["e1-random"].model
    assert fisher.embedding_init == "fisher_sqrt"
    assert random_arm.embedding_init == "random_sphere"
    for field in ("d_model", "n_layers", "n_heads", "block", "norm", "tokenizer"):
        assert getattr(fisher, field) == getattr(random_arm, field), field
    assert cfgs["e1-fisher"].train.seed == cfgs["e1-random"].train.seed

    assert cfgs["e2"].model.block == "ngpt"
    assert cfgs["e2"].model.norm == "ngpt_norm"

    e3 = cfgs["e3"].model
    assert e3.block == "ngpt"
    assert e3.attn_kernel == "kappa"
    assert cfgs["e2"].model.attn_kernel == "cosine"
    assert e3.kappa_init == [-1.0, 0.0, 1.0]
    # E3 is E2 plus the attention-kernel swap - nothing else may move,
    # including the norm (the annihilated kappa_moe swap is retired; see
    # docs/e-3-instruction.md).
    for field in ("d_model", "n_layers", "n_heads", "tokenizer", "embedding_init", "norm"):
        assert getattr(e3, field) == getattr(cfgs["e2"].model, field), field


def _total_params(cfg: Config) -> int:
    d, layers = cfg.model.d_model, cfg.model.n_layers
    per_layer = (
        4 * d * d + (3 if cfg.model.block == "ngpt" else 2) * int(cfg.model.mlp_ratio * d) * d
    )
    # Untied embeddings hold a separate input table AND output classifier,
    # so the vocab*d term is paid twice (E1.5a input/output variants).
    embedding_multiplier = 1 if cfg.model.tie_embeddings else 2
    embedding = vocab_size_for(cfg.model) * d * embedding_multiplier
    total = per_layer * layers + embedding
    if cfg.model.state_source == "field":
        # E1.5c: the causal density estimator p_phi is its own small
        # transformer encoder (256-way input embedding + field_n_layers
        # baseline-shaped blocks, 12*d^2/layer at mlp_ratio=4) sitting in
        # front of the backbone - its params are additive, not swapped in.
        fd = cfg.model.field_d_model
        total += 256 * fd + cfg.model.field_n_layers * 12 * fd * fd
        if cfg.model.field_state == "hidden":
            # P7 (FOR-924): H is the ONLY arm with a trainable lift - a
            # Linear(field_d_model, d_model, bias=False) replacing the fixed
            # semi_orthogonal_lift buffer that F/P share (buffers are not
            # parameters and are already excluded from this formula).
            total += fd * d
    return total


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_experiment_configs_are_parameter_matched(exp: str) -> None:
    """All rungs must land in the 80-100M band from the spec."""
    cfg = resolve_experiment_config(exp)
    total = _total_params(cfg)
    assert 80e6 <= total <= 100e6, f"{exp}: {total / 1e6:.1f}M"


def test_e1_5_arm_membership_matches_the_pinned_exact_totals() -> None:
    """TDD trip-wire: every arm this pin covers must actually be a
    registered experiment before the exact-total test below can mean
    anything (fails loud, not by silently not collecting cases)."""
    assert set(E1_5_EXACT_TOTAL_PARAMS) <= set(EXPERIMENTS)


@pytest.mark.parametrize("exp", sorted(E1_5_EXACT_TOTAL_PARAMS))
def test_e1_5_arms_hit_their_pinned_exact_param_total(exp: str) -> None:
    cfg = resolve_experiment_config(exp)
    assert _total_params(cfg) == E1_5_EXACT_TOTAL_PARAMS[exp], exp


# ---------------------------------------------------- P8 (FOR-925): matched-total
# Shrunken-backbone twins so N_backbone + N_field ~= e1-fisher's 85,131,264
# total (docs/e1_5-instruction.md "Comparison lanes": matched-BACKBONE vs
# matched-TOTAL-parameter — "is predictive-state construction a better use
# of a FIXED budget?"). Deliberately NOT registered in config.py's
# EXPERIMENTS (execution is prereg-time scope, build-plan Phase D — same
# code-ready-but-ungated posture as e3-anchor/e3-vantage) — loaded here by
# explicit path, mirroring resolve_experiment_config's own `--config` path,
# rather than by name.
_MATCHED_TOTAL_EXACT_PARAMS: dict[str, int] = {
    # 144*756^2 + 256*756 + (256*128 + 2*12*128^2) = 82,494,720 + 425,984.
    "e1-5c-field-matched-total": 82_920_704,
    # Field twin's total + H's own trainable state_lift (128*756 = 96,768).
    "e1-5c-hidden-matched-total": 83_017_472,
    # P shares F's p_phi exactly (no sqrt, same fixed lift, no new params).
    "e1-5c-prob-matched-total": 82_920_704,
}

# The matched-BACKBONE config each matched-total twin shrinks from.
_MATCHED_TOTAL_BASE: dict[str, str] = {
    "e1-5c-field-matched-total": "e1-5c-field",
    "e1-5c-hidden-matched-total": "e1-5c-hidden",
    "e1-5c-prob-matched-total": "e1-5c-prob",
}


@pytest.mark.parametrize("name", sorted(_MATCHED_TOTAL_EXACT_PARAMS))
def test_matched_total_configs_exist_and_parse(name: str) -> None:
    path = configs_dir() / f"{name}.yaml"
    assert path.exists(), path
    cfg = load_config(path)
    assert cfg.name == name
    assert cfg.model.d_model % cfg.model.n_heads == 0


@pytest.mark.parametrize("name", sorted(_MATCHED_TOTAL_EXACT_PARAMS))
def test_matched_total_configs_hit_their_pinned_exact_param_total(name: str) -> None:
    cfg = load_config(configs_dir() / f"{name}.yaml")
    assert _total_params(cfg) == _MATCHED_TOTAL_EXACT_PARAMS[name], name


@pytest.mark.parametrize("name", sorted(_MATCHED_TOTAL_EXACT_PARAMS))
def test_matched_total_configs_land_in_the_shared_band(name: str) -> None:
    """Same 80-100M band every registered rung is held to
    (test_experiment_configs_are_parameter_matched), applied here by hand
    since these configs are deliberately unregistered."""
    cfg = load_config(configs_dir() / f"{name}.yaml")
    total = _total_params(cfg)
    assert 80e6 <= total <= 100e6, f"{name}: {total / 1e6:.1f}M"


@pytest.mark.parametrize("name", sorted(_MATCHED_TOTAL_EXACT_PARAMS))
def test_matched_total_configs_only_change_d_model_and_fisher_init_path_from_their_base(
    name: str,
) -> None:
    """A matched-total twin is a DELIBERATE two-field change from its own
    matched-backbone base config: d_model (to shrink the total) and
    fisher_init_path (the checkpointed d768 table no longer matches a
    shrunk width, so this falls back to the corpus-computed path) —
    everything else, including the field_state axis that already
    distinguishes the base config from e1-fisher, must carry over
    unchanged. Mirrors test_e1_5_arms_extend_e1_fisher_only_on_their_axis's
    shape but against the base config rather than e1-fisher."""
    base = resolve_experiment_config(_MATCHED_TOTAL_BASE[name])
    twin = load_config(configs_dir() / f"{name}.yaml")
    for f in dataclasses.fields(ModelConfig):
        base_val = getattr(base.model, f.name)
        twin_val = getattr(twin.model, f.name)
        if f.name in ("d_model", "fisher_init_path"):
            assert twin_val != base_val, f"{name}: axis field {f.name!r} did not change"
        else:
            assert twin_val == base_val, f"{name}: unexpected change on model.{f.name}"
    assert twin.data == base.data
    assert twin.train == base.train


def test_tiny_overlay_shrinks_without_changing_the_experiment() -> None:
    full = resolve_experiment_config("e3")
    tiny = resolve_experiment_config("e3", tiny=True)
    assert tiny.name == "e3-tiny"
    assert tiny.model.d_model == 64
    assert tiny.model.n_layers == 2
    assert tiny.train.device == "cpu"
    assert tiny.data.source == "synthetic"
    # The things E3 *is* must survive the overlay.
    assert tiny.model.block == full.model.block
    assert tiny.model.norm == full.model.norm
    assert tiny.model.attn_kernel == full.model.attn_kernel
    assert tiny.model.kappa_init == full.model.kappa_init


def test_overrides_beat_the_tiny_overlay() -> None:
    cfg = resolve_experiment_config("e2", tiny=True, overrides={"train": {"max_steps": 3}})
    assert cfg.train.max_steps == 3


def test_resolve_requires_an_experiment_or_path() -> None:
    with pytest.raises(ValueError, match="must be given"):
        resolve_experiment_config()
    with pytest.raises(ValueError, match="unknown experiment"):
        resolve_experiment_config("e9")


def test_explicit_config_path_wins(tmp_path: Path) -> None:
    path = tmp_path / "custom.yaml"
    path.write_text("name: custom\nmodel:\n  d_model: 16\n  n_heads: 4\n", encoding="utf-8")
    cfg = resolve_experiment_config(config_path=path)
    assert cfg.name == "custom"
    assert cfg.model.d_model == 16


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_config(path)


def test_config_is_json_serialisable() -> None:
    import json

    payload = resolve_experiment_config("e3").to_dict()
    assert json.loads(json.dumps(payload))["model"]["attn_kernel"] == "kappa"


def test_tiny_e3_model_builds_from_config(sample_bytes: bytes) -> None:
    cfg = resolve_experiment_config("e3", tiny=True)
    model = build_model(cfg.model, fisher_source=sample_bytes)
    # The kappa_moe norm swap is retired (radially annihilated); curvature
    # now lives in the attention kernel, one module per layer.
    assert len(model.kappa_modules()) == 0
    assert len(model.kappa_attention_modules()) == cfg.model.n_layers


def test_anchor_arm_configs_extend_e3_only_on_the_anchor_axes() -> None:
    """The anchored arms change the anchor/granularity axes and nothing else."""
    cfgs = {e: resolve_experiment_config(e) for e in ("e3", "e3-anchor", "e3-vantage")}
    e3 = cfgs["e3"].model
    anchor = cfgs["e3-anchor"].model
    vantage = cfgs["e3-vantage"].model
    assert e3.anchor_mode == "none"
    assert anchor.anchor_mode == "global"
    assert anchor.kappa_granularity == "per_head"
    assert vantage.anchor_mode == "per_head"
    assert vantage.kappa_granularity == "global"
    for arm in (anchor, vantage):
        for field in ("d_model", "n_layers", "n_heads", "norm", "attn_kernel", "embedding_init"):
            assert getattr(arm, field) == getattr(e3, field), field


def test_execution_gate_follows_the_resolved_config_not_the_flag() -> None:
    """The gate is a property of what RUNS. Anchors are the gated feature, so
    the predicate reads them off the resolved config — an anchored arm reached
    through ``--config`` (where ``args.exp`` is None) is still gated."""
    for arm in ("e3-anchor", "e3-vantage"):
        assert is_execution_gated(resolve_experiment_config(arm))
    for ungated in ("e0-byte", "e1-fisher", "e1-random", "e2", "e3"):
        assert not is_execution_gated(resolve_experiment_config(ungated))


def test_execution_gate_catches_a_hand_rolled_anchored_config() -> None:
    """Not a name check: any config that turns anchors on is gated."""
    cfg = config_from_dict(
        {"name": "my-own-arm", "model": {"attn_kernel": "kappa", "anchor_mode": "global"}}
    )
    assert is_execution_gated(cfg)


# E1.5 arms declare exactly one axis change from e1-fisher (docs/e1_5-instruction.md).
# "input"/"output" freeze entails untying, since a tied table cannot be frozen on
# only one of its two roles - that entailment is part of the declared axis, not a
# stray extra change.
_E1_5_ARM_AXES: dict[str, frozenset[str]] = {
    "e1-5a-input": frozenset({"embedding_freeze", "tie_embeddings"}),
    "e1-5a-output": frozenset({"embedding_freeze", "tie_embeddings"}),
    "e1-5a-both": frozenset({"embedding_freeze"}),
    "e1-5b-gram": frozenset({"gram_lambda"}),
    "e1-5c-field": frozenset({"state_source"}),
    # P7 (FOR-924): H/P share e1-5c-field's declared state_source axis (a
    # field_encoder must exist for field_state to mean anything - see
    # backbone.py's state_source='table' guard) PLUS their own field_state
    # value away from the shared default "sqrt_prob" (== the F arm).
    "e1-5c-hidden": frozenset({"state_source", "field_state"}),
    "e1-5c-prob": frozenset({"state_source", "field_state"}),
}


@pytest.mark.parametrize("exp", sorted(_E1_5_ARM_AXES))
def test_e1_5_arms_extend_e1_fisher_only_on_their_axis(exp: str) -> None:
    """Every E1.5 arm must match e1-fisher on every ModelConfig/DataConfig/
    TrainConfig field except its declared axis - one variable per rung
    (docs/e1_5-instruction.md), mirroring
    test_anchor_arm_configs_extend_e3_only_on_the_anchor_axes but over the
    full field set rather than a hand-picked subset."""
    base = resolve_experiment_config("e1-fisher")
    cfg = resolve_experiment_config(exp)
    axis = _E1_5_ARM_AXES[exp]

    for section_name, section_cls in (
        ("model", ModelConfig),
        ("data", DataConfig),
        ("train", TrainConfig),
    ):
        base_section = getattr(base, section_name)
        cfg_section = getattr(cfg, section_name)
        for f in dataclasses.fields(section_cls):
            base_val = getattr(base_section, f.name)
            cfg_val = getattr(cfg_section, f.name)
            if section_name == "model" and f.name in axis:
                assert cfg_val != base_val, f"{exp}: axis field {f.name!r} did not change"
            else:
                assert cfg_val == base_val, f"{exp}: unexpected change on {section_name}.{f.name}"
