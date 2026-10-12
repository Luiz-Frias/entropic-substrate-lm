"""Config loading / merging and the shipped experiment YAML files."""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import pytest

from entropic_substrate.core.substrate.backbone import build_model, vocab_size_for
from entropic_substrate.kernel.runtime.config import (
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

pytestmark = pytest.mark.unit

# E1.5 arms (docs/e1_5-instruction.md): exact total param counts, pinned.
# The formula must reproduce every one of these exactly, not just land the
# shared 80-100M band assertion (which every arm here satisfies either way,
# so the band alone is not RED-capable for a counting change).
#
# CORRECTED 2026-08-20 (see `_total_params`' own retraction note): every
# value below was previously the formula's own output, and the formula
# omitted the LayerNorm gains — so each arm was pinned 19,200 (table arms)
# or 19,840 (field arms) below what `build_model` actually constructs. The
# numbers here are now REAL counts — e1-5c-hidden's is tied directly to a
# real build by test_pinned_exact_total_matches_a_real_build_at_full_width,
# and the formula that reproduces all of them is itself cross-checked
# against real builds of every arm by test_total_params_formula_matches_a_
# real_build.
E1_5_EXACT_TOTAL_PARAMS: dict[str, int] = {
    "e1-fisher": 85_150_464,
    "e1-5a-input": 85_347_072,
    "e1-5a-output": 85_347_072,
    "e1-5a-both": 85_150_464,
    "e1-5b-gram": 85_150_464,
    "e1-5c-field": 85_577_088,
    # P7 (FOR-924): H/P share e1-5c-field's p_phi (field_d_model=128,
    # field_n_layers=2) exactly, so e1-5c-prob (no sqrt, same lift, no new
    # params) lands on the SAME total as e1-5c-field. e1-5c-hidden adds the
    # one trainable lift W: Linear(field_d_model=128, d_model=768,
    # bias=False) = 128*768 = 98_304 params over that shared base — measured
    # against real builds as 85,675,392 - 85,577,088 = 98,304 exactly, i.e.
    # the H-arm delta was always right; only the shared base was low.
    "e1-5c-hidden": 85_675_392,
    "e1-5c-prob": 85_577_088,
    # Matched-total twins, registered in EXPERIMENTS by the E1.5c
    # preregistration commit (docs/e1-5c-preregistration.md — execution was
    # prereg-time scope). Exact totals measured against real builds when the
    # configs landed (P8, FOR-925); see each yaml's own arithmetic comment.
    # CORRECTED 2026-08-21 with the width fix below: these were 82,940,244 /
    # 83,037,012 / 82,940,244 at the retired d_model=756.
    "e1-5c-field-matched-total": 80_344_872,
    "e1-5c-hidden-matched-total": 80_440_104,
    # P shares F's p_phi exactly (no sqrt, same fixed lift, no new params).
    "e1-5c-prob-matched-total": 80_344_872,
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


def test_configs_dir_falls_back_to_cwd_when_repo_root_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Error case pinned at authoring (E-3/DN-8): a RepoLayoutError from the
    resolver must not propagate out of configs_dir() -- it must fall through
    to the pre-existing CWD-relative candidate, same as an unresolvable
    anchor did before this phase."""
    import entropic_substrate.kernel.runtime.config as config_module
    from entropic_substrate.kernel.runtime.paths import RepoLayoutError

    def _raise() -> Path:
        raise RepoLayoutError("mutated for test")

    monkeypatch.setattr(config_module, "repo_root", _raise)
    fake_configs = tmp_path / "configs"
    fake_configs.mkdir()
    monkeypatch.chdir(tmp_path)
    assert config_module.configs_dir() == fake_configs


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


def _norm_params(model: ModelConfig, width: int) -> int:
    """Parameters of ONE `build_norm(...)` layer at `width` (models/blocks.py).

    `nn.LayerNorm(width, bias=cfg.bias)` is weight-only while `bias=False`
    (guarded in `_total_params`); `UnitNorm` (`norm="ngpt_norm"`) is a pure
    function with no state at all.
    """
    if model.norm == "layernorm":
        return width
    if model.norm == "ngpt_norm":
        return 0
    # kappa_moe (GatedKappaNorm) carries a router + per-component curvatures.
    # It is measured radially-annihilated and no registered arm selects it
    # (CLAUDE.md architecture section), so it is refused rather than modelled
    # from memory — an unmodelled term must fail loud, never miscount quietly.
    msg = f"_total_params does not model norm={model.norm!r}"
    raise ValueError(msg)


def _block_params(model: ModelConfig) -> int:
    """Parameters of ONE transformer block, term for term against
    `models/blocks.py` (`BaselineBlock` / `NGPTBlock` + `CausalSelfAttention`
    + `MLP`)."""
    d = model.d_model
    hidden = int(model.mlp_ratio * d)
    # q/k/v/out projections, bias=False.
    total = 4 * d * d
    if model.block == "ngpt":
        # SwiGLU: fc is d -> 2*hidden, proj is hidden -> d.
        total += 3 * hidden * d
        # nGPT's learned scales: sqk (d), suv (2*hidden), and the two
        # eigen-learning-rate vectors attn_alpha/mlp_alpha (d each).
        total += d + 2 * hidden + 2 * d
    else:
        total += 2 * hidden * d
    # norm_attn + norm_mlp. NGPTBlock's third norm (`retract`) is always
    # UnitNorm and therefore always free.
    total += 2 * _norm_params(model, d)
    if model.attn_kernel == "kappa":
        # E3: one learnable curvature per head — or a single shared one on
        # the vantage ablation — plus one per-head temperature tau.
        total += 1 if model.kappa_granularity == "global" else model.n_heads
        total += model.n_heads
        if model.anchor_mode != "none":
            # Asymmetric Mobius anchors p_q, p_k, each (1, n_anchor, 1, head_dim).
            n_anchor = 1 if model.anchor_mode == "global" else model.n_heads
            total += 2 * n_anchor * (d // model.n_heads)
    return total


def _field_encoder_params(model: ModelConfig) -> int:
    """Parameters of the E1.5c estimator p_phi (models/field_encoder.py).

    Its blocks are pinned to a baseline/layernorm/cosine shape at
    `field_d_model` regardless of what the OUTER config selects (the
    `inner_cfg` replace in `FieldEncoder.__init__`), so this reads those
    constants rather than the outer arm's block/norm/kernel. The fixed
    `lift` is a buffer, not a parameter, and the tied 256-way head shares
    `wte.weight`.
    """
    fd, layers = model.field_d_model, model.field_n_layers
    per_layer = 4 * fd * fd + 2 * (4 * fd) * fd + 2 * fd  # attn + GELU MLP + 2 norms
    total = 256 * fd + layers * per_layer + fd  # wte (tied head) + blocks + norm_f
    if model.field_state == "hidden":
        # P7 (FOR-924): H is the ONLY arm with a trainable lift — a
        # Linear(field_d_model, d_model, bias=False) standing in for the
        # fixed semi_orthogonal_lift buffer that F/P share.
        total += fd * model.d_model
    return total


def _total_params(cfg: Config) -> int:
    """Total parameters implied by `cfg`, without building the model.

    Cross-checked against real `build_model(...)` counts by
    `test_total_params_formula_matches_a_real_build[_at_full_width]` below —
    that pair is what keeps this honest. RETRACTION (2026-08-20): this
    formula previously omitted every normalisation gain, so each arm it
    pinned came out low by exactly the LayerNorm weights. MEASURED against
    a real build of e1-5c-field: formula 85,557,248 vs real 85,577,088, a
    uniform -19,840 across the E1.5 arms, which decomposes as the backbone's
    (2 norms/block * 12 blocks + 1 final norm_f) * 768 = 25 * 768 = 19,200
    plus the field encoder's (2 * 2 + 1) * 128 = 640. The nGPT arms were
    additionally short their sqk/suv/alpha/sz scales (e2: 76,288) and the
    E3 arms their kappa/tau/anchors (e3: +216, e3-anchor: +1,152,
    e3-vantage: +13,941 over e2) — all now modelled and all verified
    against real builds.
    """
    model = cfg.model
    if model.bias:
        # Every shipped config is bias=False, so the Linear/LayerNorm bias
        # terms are deliberately unmodelled — refuse rather than under-count
        # a future config by exactly the amount this helper exists to get
        # right.
        msg = "_total_params models bias=False only; a bias=True config needs the bias terms added"
        raise ValueError(msg)
    d = model.d_model
    embedding_multiplier = 1 if model.tie_embeddings else 2
    total = vocab_size_for(model) * d * embedding_multiplier
    total += model.n_layers * _block_params(model)
    # LanguageModel.norm_f: UnitNorm (free) whenever the block is nGPT,
    # regardless of cfg.norm — backbone.py picks it off `hypersphere`, not
    # off the norm setting the blocks use.
    total += 0 if model.block == "ngpt" else _norm_params(model, d)
    if model.block == "ngpt":
        # nGPT's per-class logit temperature `sz`.
        total += vocab_size_for(model)
    if model.state_source == "field":
        total += _field_encoder_params(model)
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
# Shrunken-backbone twins so N_backbone + N_field ~= e1-fisher's 85,150,464
# total (docs/e1_5-instruction.md "Comparison lanes": matched-BACKBONE vs
# matched-TOTAL-parameter — "is predictive-state construction a better use
# of a FIXED budget?"). Registered in config.py's EXPERIMENTS by the E1.5c
# preregistration commit (docs/e1-5c-preregistration.md), so existence/
# parse, the 80-100M band, and the exact pins above are all carried by the
# EXPERIMENTS-wide tests now — what remains here is the axis pin no generic
# sweep covers: each twin is a deliberate two-field change from its own
# matched-backbone base.

# The matched-BACKBONE config each matched-total twin shrinks from.
_MATCHED_TOTAL_BASE: dict[str, str] = {
    "e1-5c-field-matched-total": "e1-5c-field",
    "e1-5c-hidden-matched-total": "e1-5c-hidden",
    "e1-5c-prob-matched-total": "e1-5c-prob",
}


@pytest.mark.parametrize("name", sorted(_MATCHED_TOTAL_BASE))
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


# ------------------------------------------- formula vs a REAL build (FOR-9xx)
# `_total_params` is a hand-rolled duplicate of the model's own parameter
# accounting, and a duplicate drifts silently: measured 2026-08-20, it was
# under-counting every arm by the LayerNorm gains (see its own comment), and
# nothing in the suite noticed because the only other check on it was an
# 80-100M band 20k parameters wide. These two tests remove the silence by
# comparing the formula against `build_model(...)` itself.
#
# Deterministic corpus stand-in for the `fisher_sqrt` arms: `build_fisher_init`
# only needs bytes to count transitions over, and Laplace smoothing keeps every
# row well-defined even on uniform noise. Parameter COUNT does not depend on
# the values, only on the shapes, so this never has to be the real corpus.
_FISHER_SOURCE: bytes = random.Random(0).randbytes(20_000)

# Exactly ONE full-width build, and the number below is why. MEASURED
# 2026-08-20 on the full `-m unit` suite: no cross-check 40.83s, + the
# 19-config shrunken sweep 39.65s (free, inside run-to-run noise), + five
# full-width builds 99.2s. A full-width 85M build costs ~1.8s alone but
# 3.2-5.6s once ~950 tests of allocation sit behind it, and the unit budget
# is 60s (pyproject's own marker text). The shrunken sweep is what carries
# the coverage — the `--tiny` overlay moves only WIDTHS, so every MECHANISM
# (block, norm, attn_kernel, anchor_mode, kappa_granularity, tie_embeddings,
# state_source, field_state, and e0-bpe's 50257-wide vocab, which `--tiny`
# does not shrink) is exercised against a real build there. This one arm
# exists to tie a PINNED LITERAL to a real count at the width it was pinned
# at; e1-5c-hidden is the richest such arm (baseline blocks + LayerNorm
# gains + tied byte tables + p_phi + H's trainable state_lift).
_FULL_WIDTH_CROSSCHECK: tuple[str, ...] = ("e1-5c-hidden",)


def _resolve_by_name(name: str, *, tiny: bool = False) -> Config:
    """Registered experiments by name; the unregistered matched-total twins by
    path (mirroring how their own tests above load them)."""
    if name in EXPERIMENTS:
        return resolve_experiment_config(name, tiny=tiny)
    return resolve_experiment_config(config_path=configs_dir() / f"{name}.yaml", tiny=tiny)


def _real_total_params(cfg: Config) -> int:
    """Ground truth: parameters of a model that was actually BUILT."""
    model = build_model(cfg.model, fisher_source=_FISHER_SOURCE)
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_total_params_formula_matches_a_real_build(name: str) -> None:
    """Every registered arm (matched-total twins included, now that the
    E1.5c preregistration registered them), shrunken by the repo's own
    `--tiny` overlay — which changes only WIDTHS (d_model, n_layers,
    n_heads, field_*) and leaves every mechanism switch alone, so a real
    build here constructs exactly the modules the full-size arm would."""
    cfg = _resolve_by_name(name, tiny=True)
    assert _total_params(cfg) == _real_total_params(cfg), name


@pytest.mark.parametrize("name", _FULL_WIDTH_CROSSCHECK)
def test_pinned_exact_total_matches_a_real_build_at_full_width(name: str) -> None:
    """The pinned literal, checked against a model that was actually built at
    the width it was pinned at — not against the formula (that comparison is
    `test_e1_5_arms_hit_their_pinned_exact_param_total`'s job, and a formula
    and a pin can be edited into agreement while both are wrong)."""
    cfg = _resolve_by_name(name)
    real = _real_total_params(cfg)
    assert E1_5_EXACT_TOTAL_PARAMS[name] == real, name
    assert _total_params(cfg) == real, name


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


def test_model_config_rejects_gram_lambda_with_non_byte_tokenizer() -> None:
    """Guard (FOR-915): gram_lambda != 0 with a non-byte tokenizer allocates
    O(V²) fp32 Gram memory — ≈20.2 GB at gpt2's V=50257."""
    with pytest.raises(ValueError, match="gram_lambda"):
        ModelConfig(gram_lambda=0.5, tokenizer="gpt2")


def test_gram_lambda_with_byte_tokenizer_is_allowed() -> None:
    cfg = ModelConfig(gram_lambda=0.5, tokenizer="byte")
    assert cfg.gram_lambda == 0.5


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_registered_arms_have_an_even_head_dim_at_full_width(exp: str) -> None:
    """RoPE needs an EVEN head_dim (`blocks.py::rotary_embeddings`), and the
    registered width is the one that runs on pods.

    MEASURED 2026-08-21, the reason this test exists: four
    `e1-5c-field-matched-total` pods burned ~4 GPU-hours each and produced
    nothing. `d_model=756 / n_heads=12 = 63` — odd — so the FIRST forward
    pass raised `ValueError: head_dim must be even for RoPE, got 63`. The
    config could never have trained.

    Nothing in the suite caught it because every other check either builds
    at `--tiny` width (`d_model: 64, n_heads: 4` -> head_dim 16, even) or
    counts parameters, which the head split does not affect. Construction
    succeeds too — the failure lives in the forward pass. So this pins the
    ARITHMETIC at the registered width, which is cheap, exhaustive, and
    needs no model build."""
    cfg = resolve_experiment_config(exp)
    d_model, n_heads = cfg.model.d_model, cfg.model.n_heads
    assert d_model % n_heads == 0, f"{exp}: d_model {d_model} not divisible by n_heads {n_heads}"
    head_dim = d_model // n_heads
    assert head_dim % 2 == 0, (
        f"{exp}: head_dim {head_dim} (d_model {d_model} / n_heads {n_heads}) is ODD — "
        "RoPE requires an even head_dim, so this arm cannot run at its registered width"
    )


# FOR-1585: TrainConfig owns its betas coercion. YAML parses ``[0.9, 0.95]``
# as a list; the dataclass declares ``tuple[float, float]``. Normalising in
# ``__post_init__`` makes the file path and the code path agree without a
# ``cls is TrainConfig`` special case in the generic ``_build``.
def test_yaml_list_betas_load_as_a_two_float_tuple(tmp_path: Path) -> None:
    path = tmp_path / "betas.yaml"
    path.write_text("train:\n  betas: [1, 2]\n", encoding="utf-8")
    betas = load_config(path).train.betas
    assert betas == (1.0, 2.0)
    assert type(betas) is tuple
    assert all(type(beta) is float for beta in betas)


@pytest.mark.parametrize("raw", [[1, 2], (1, 2), ("0.9", "0.95"), (True, 0)])
def test_code_constructed_train_config_normalises_betas_like_the_loader(
    raw: object,
) -> None:
    built = TrainConfig(betas=raw)  # type: ignore[arg-type]
    loaded = config_from_dict({"train": {"betas": raw}}).train
    assert built.betas == loaded.betas
    assert type(built.betas) is tuple
    assert all(type(beta) is float for beta in built.betas)


def test_train_config_default_betas_are_unchanged() -> None:
    assert TrainConfig().betas == (0.9, 0.95)


def test_train_config_rejects_betas_without_a_second_element() -> None:
    # Parent behaviour for the YAML loader path (the `_build` branch indexed
    # betas[1]); the code-constructed path now matches it (TrainConfig(betas=[0.9])
    # raised nothing at the parent and raises IndexError now, the intended
    # consequence of moving the coercion into __post_init__).
    with pytest.raises(IndexError):
        TrainConfig(betas=[0.9])  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        TrainConfig(betas=[])  # type: ignore[arg-type]
