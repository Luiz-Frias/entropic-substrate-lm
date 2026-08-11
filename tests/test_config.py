"""Config loading / merging and the shipped experiment YAML files."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.config import (
    EXPERIMENTS,
    config_from_dict,
    configs_dir,
    load_config,
    merge_dicts,
    resolve_experiment_config,
)
from entropic_substrate.models.backbone import build_model, vocab_size_for

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("exp", EXPERIMENTS)
def test_experiment_configs_are_parameter_matched(exp: str) -> None:
    """All rungs must land in the 80-100M band from the spec."""
    cfg = resolve_experiment_config(exp)
    d, layers = cfg.model.d_model, cfg.model.n_layers
    per_layer = (
        4 * d * d + (3 if cfg.model.block == "ngpt" else 2) * int(cfg.model.mlp_ratio * d) * d
    )
    embedding = vocab_size_for(cfg.model) * d
    total = per_layer * layers + embedding
    assert 80e6 <= total <= 100e6, f"{exp}: {total / 1e6:.1f}M"


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
