"""Typed experiment configuration, loaded from YAML.

One flat file per experiment in ``configs/``; every field has a default so a
config only needs to state what it changes. ``--tiny`` overlays
``configs/tiny.yaml`` on top of any experiment for CPU smoke runs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "load_config",
    "merge_dicts",
    "resolve_experiment_config",
]

EmbeddingInit = Literal["random_sphere", "fisher_sqrt", "default"]
NormKind = Literal["layernorm", "ngpt_norm", "kappa_moe"]
BlockKind = Literal["baseline", "ngpt"]
AttnKernel = Literal["cosine", "kappa"]
Tokenizer = Literal["byte", "gpt2"]
DataSource = Literal["enwik8", "fineweb", "synthetic"]

EXPERIMENTS: tuple[str, ...] = (
    "e0-byte",
    "e0-bpe",
    "e1-fisher",
    "e1-random",
    "e2",
    "e3",
)


@dataclass(slots=True)
class ModelConfig:
    """Architecture."""

    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    mlp_ratio: float = 4.0
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = False
    tokenizer: Tokenizer = "byte"
    vocab_size: int | None = None  # inferred from tokenizer when None
    block: BlockKind = "baseline"
    norm: NormKind = "layernorm"
    # E3: attention similarity kernel — "cosine" (E0–E2) or "kappa"
    # (per-head learnable curvature; kappa_init spreads over the heads).
    attn_kernel: AttnKernel = "cosine"
    embedding_init: EmbeddingInit = "default"
    fisher_init_path: str | None = None
    tie_embeddings: bool = True
    # kappa_moe options
    n_components: int = 3
    top_k: int = 1
    kappa_init: list[float] = field(default_factory=lambda: [-1.0, 0.0, 1.0])
    aux_loss_weight: float = 0.01
    router_noise: float = 0.0
    # nGPT options
    alpha_init: float = 0.05


@dataclass(slots=True)
class DataConfig:
    """Dataset selection and batching."""

    source: DataSource = "enwik8"
    path: str = "data/enwik8/enwik8"
    fineweb_name: str = "sample-10BT"
    fineweb_chars: int = 20_000_000
    val_fraction: float = 0.05
    batch_size: int = 8
    seq_len: int = 1024
    num_eval_batches: int = 20
    synthetic_bytes: int = 1_000_000


@dataclass(slots=True)
class TrainConfig:
    """Optimisation loop."""

    max_steps: int = 20_000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_interval: int = 500
    log_interval: int = 20
    seed: int = 42
    amp_dtype: Literal["bf16", "fp16", "off"] = "bf16"
    device: str = "auto"
    compile: bool = False
    results_dir: str = "results"
    models_dir: str = "models"
    save_interval: int = 1000
    resume: str | None = None


@dataclass(slots=True)
class Config:
    """Full experiment description."""

    name: str = "unnamed"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (JSON serialisable), used for run metadata."""
        return dataclasses.asdict(self)


def merge_dicts(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins)."""
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            out[key] = merge_dicts(current, value)
        else:
            out[key] = value
    return out


def _build(cls: type, payload: Mapping[str, Any]) -> Any:
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        msg = f"unknown keys for {cls.__name__}: {sorted(unknown)}"
        raise ValueError(msg)
    kwargs = dict(payload)
    if cls is TrainConfig and "betas" in kwargs:
        betas = kwargs["betas"]
        kwargs["betas"] = (float(betas[0]), float(betas[1]))
    return cls(**kwargs)


def config_from_dict(payload: Mapping[str, Any]) -> Config:
    """Build a :class:`Config` from a nested mapping (as parsed from YAML)."""
    known = {"name", "model", "data", "train"}
    unknown = set(payload) - known
    if unknown:
        msg = f"unknown top-level config keys: {sorted(unknown)}"
        raise ValueError(msg)
    return Config(
        name=str(payload.get("name", "unnamed")),
        model=_build(ModelConfig, payload.get("model", {}) or {}),
        data=_build(DataConfig, payload.get("data", {}) or {}),
        train=_build(TrainConfig, payload.get("train", {}) or {}),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"{path} must contain a YAML mapping"
        raise TypeError(msg)
    return data


def load_config(path: str | Path, overlay: Mapping[str, Any] | None = None) -> Config:
    """Load a single YAML config file, optionally merging an overlay mapping."""
    payload = _read_yaml(Path(path))
    if overlay:
        payload = merge_dicts(payload, overlay)
    return config_from_dict(payload)


def configs_dir() -> Path:
    """Locate the repository ``configs/`` directory.

    Works both from a source checkout (``src/entropic_substrate/config.py`` ->
    ``../../configs``) and from an installed package sitting next to a
    ``configs/`` folder in the current working directory.
    """
    candidate = Path(__file__).resolve().parents[2] / "configs"
    if candidate.is_dir():
        return candidate
    cwd_candidate = Path.cwd() / "configs"
    if cwd_candidate.is_dir():
        return cwd_candidate
    msg = "could not locate configs/ directory"
    raise FileNotFoundError(msg)


def resolve_experiment_config(
    experiment: str | None = None,
    config_path: str | Path | None = None,
    *,
    tiny: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Resolve ``--exp`` / ``--config`` / ``--tiny`` into a single :class:`Config`."""
    if config_path is not None:
        payload = _read_yaml(Path(config_path))
    elif experiment is not None:
        if experiment not in EXPERIMENTS:
            msg = f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}"
            raise ValueError(msg)
        payload = _read_yaml(configs_dir() / f"{experiment}.yaml")
    else:
        msg = "one of experiment or config_path must be given"
        raise ValueError(msg)

    if tiny:
        payload = merge_dicts(payload, _read_yaml(configs_dir() / "tiny.yaml"))
        payload["name"] = f"{payload.get('name', 'unnamed')}-tiny"
    if overrides:
        payload = merge_dicts(payload, overrides)
    return config_from_dict(payload)
