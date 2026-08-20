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
    "E15_PREREGISTRATIONS",
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "is_execution_gated",
    "load_config",
    "merge_dicts",
    "required_preregistrations",
    "resolve_experiment_config",
]

EmbeddingInit = Literal["random_sphere", "fisher_sqrt", "default"]
NormKind = Literal["layernorm", "ngpt_norm", "kappa_moe"]
BlockKind = Literal["baseline", "ngpt"]
AttnKernel = Literal["cosine", "kappa"]
AnchorMode = Literal["none", "global", "per_head"]
KappaGranularity = Literal["per_head", "global"]
Tokenizer = Literal["byte", "gpt2"]
DataSource = Literal["enwik8", "fineweb", "synthetic"]
# E1.5a: which half of the (possibly untied) embedding stays frozen at its
# init value while the rest of the backbone trains freely.
EmbeddingFreeze = Literal["none", "input", "output", "both"]
# E1.5b: which Gram matrix the geometry-preserving penalty pulls the live
# embedding's Gram matrix toward. Only "init" exists today; the Literal
# names the axis so a second reference point is a type-checked addition.
GramReference = Literal["init"]
# E1.5c: does the per-symbol state come from a static lookup table (E1) or
# a causal, context-conditioned field p_phi(.|x_<=t) (E1.5c)?
StateSource = Literal["table", "field"]
# E1.5c: architecture of the field estimator p_phi. Only the v1 causal
# transformer is registered; docs/e1_5-instruction.md's v0/v2 rungs add
# members here when they're registered, not before.
FieldEstimator = Literal["transformer_v1"]
# E1.5c Repair 1 (docs/e1_5-instruction.md, PR #73): does p_phi train jointly
# with the backbone under the calibration CE (field_loss_lambda), or does it
# load a checkpoint and freeze — the cleaner test, since it prevents p_phi
# from becoming a private code negotiated with the backbone.
FieldMode = Literal["joint", "frozen"]
# E1.5c Repair 2 (docs/e1_5-instruction.md, PR #73, FOR-924): which
# CONSTRUCTION of p_phi's output becomes the state z_t handed to the
# backbone, once state_source="field" — the four-arm table's F/P/H legs
# (T, the static-table arm, is state_source="table" and carries no
# field_state at all). "sqrt_prob" = F = sqrt(p_phi).Q, the full thesis
# (docs default, matches the arm merged pre-P7); "prob" = P = p_phi.Q, no
# sqrt — the control for the sqrt map itself; "hidden" = H = W.h_phi,t via
# a trainable lift — contextual preprocessing without probability geometry.
FieldState = Literal["sqrt_prob", "prob", "hidden"]

EXPERIMENTS: tuple[str, ...] = (
    "e0-byte",
    "e0-bpe",
    "e1-fisher",
    "e1-random",
    "e2",
    "e3",
    # Anchored arms — code-ready, execution gated on the E3 readout
    # (per-head kappa divergence); see docs/e-3-instruction.md.
    "e3-anchor",
    "e3-vantage",
    # E1.5 arms — code-ready; NOT in the gated ladder (scripts/run_ladder.sh)
    # until each arm's preregistration is committed; see
    # docs/e1_5-instruction.md.
    "e1-5a-input",
    "e1-5a-output",
    "e1-5a-both",
    "e1-5b-gram",
    "e1-5c-field",
    # P7 (FOR-924): H and P — the two other legs of the four-arm table, sharing
    # e1-5c-field's p_phi and differing only in field_state.
    "e1-5c-hidden",
    "e1-5c-prob",
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
    # Anchored arms (docs/e-3-instruction.md; execution gated on E3 readout).
    # anchor_mode: asymmetric Mobius anchors p_q != p_k — "global" (one shared
    # pair, the minimal symmetry break) or "per_head" (the vantage arm).
    # kappa_granularity "global" collapses curvature to one shared knob (the
    # curvature-vs-vantage attribution ablation).
    anchor_mode: AnchorMode = "none"
    kappa_granularity: KappaGranularity = "per_head"
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
    # E1.5a: freeze the named half of the embedding at its init value
    # (frozen Fisher geometry); "input"/"output" require tie_embeddings=False
    # since a tied table cannot be frozen on only one of its two roles.
    embedding_freeze: EmbeddingFreeze = "none"
    # E1.5b: weight on the geometry-preserving Gram penalty
    # L_CE + gram_lambda * ||G(E_t) - G(E_reference)||_F^2; 0.0 disables it.
    gram_lambda: float = 0.0
    # E1.5b: which Gram matrix gram_lambda penalizes divergence from.
    gram_reference: GramReference = "init"
    # E1.5c: symbol state is a static table lookup (E1) or the live output
    # of a causal field estimator conditioned on x_<=t (E1.5c).
    state_source: StateSource = "table"
    # E1.5c: architecture of the field estimator p_phi when state_source
    # is "field".
    field_estimator: FieldEstimator = "transformer_v1"
    # E1.5c: field estimator depth (number of causal transformer layers).
    field_n_layers: int = 2
    # E1.5c: field estimator width (d_model of the small causal encoder,
    # independent of the backbone's own d_model).
    field_d_model: int = 128
    # E1.5c: field estimator attention head count.
    field_n_heads: int = 4
    # E1.5c Repair 1: weight on the calibration CE that keeps p_phi an
    # honest next-byte predictor (L_total = L_backbone + field_loss_lambda *
    # L_field). 0.0 disables the seam entirely — the merged-but-uncalibrated
    # simplex-bottleneck configuration docs/e1_5-instruction.md's design
    # correction names as the flaw this repairs.
    field_loss_lambda: float = 0.0
    # E1.5c Repair 1: "joint" trains p_phi and the backbone together under
    # the calibration CE; "frozen" loads p_phi from field_ckpt_path and
    # freezes it before the backbone ever trains over sqrt(p_phi).Q.
    field_mode: FieldMode = "joint"
    # E1.5c Repair 1: checkpoint p_phi's frozen weights load from. Required
    # when field_mode="frozen" (refused loudly at build otherwise); must be
    # None when field_mode="joint" (a set path there would be silently
    # ignored, which is exactly the kind of quiet misconfiguration CLAUDE.md's
    # Errors principle refuses).
    field_ckpt_path: str | None = None
    # E1.5c Repair 2 (P7, FOR-924): which construction of p_phi's output
    # becomes z_t — see FieldState above. Only meaningful when
    # state_source="field"; refused loudly at build otherwise (backbone.py),
    # same posture as field_loss_lambda/field_ckpt_path/field_mode.
    field_state: FieldState = "sqrt_prob"

    def __post_init__(self) -> None:
        if self.gram_lambda != 0.0 and self.tokenizer != "byte":
            msg = (
                "gram_lambda != 0.0 requires tokenizer='byte': "
                "normalized_gram allocates an (n_symbols, n_symbols) fp32 matrix — "
                "trivial at the 256-byte vocab (256² × 4 B ≈ 262 KB × 2) but "
                f"O(V²) at tokenizer={self.tokenizer!r}; at gpt2 (V=50257) the "
                "reference + live matrices cost ≈20.2 GB before autograd storage"
            )
            raise ValueError(msg)


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


# FOR-914: each E1.5 axis maps to its committed preregistration, or None
# while unregistered. This in-code map — not a docs/ stat — is the RUNTIME
# half of the execution gate because the training image carries no docs/
# (Containerfile copies src/ configs/ scripts/ only), so a pod cannot check
# the file. The CI half (tests/test_execution_gate.py) pins every non-None
# entry to a file that actually exists in the repo, so the assertion and
# the artifact cannot drift. Flip an entry to a path ONLY in the same
# commit that adds the preregistration document (docs/rung-lifecycle-
# protocol.md: committed before any confirmatory run).
E15_PREREGISTRATIONS: dict[str, str | None] = {
    "embedding_freeze": "docs/e1-5a-preregistration.md",
    "gram_lambda": "docs/e1-5b-preregistration.md",
    "state_source": None,  # E1.5c preregistration not yet committed
}


def required_preregistrations(cfg: Config) -> tuple[tuple[str, str | None], ...]:
    """Every E1.5 axis ``cfg`` activates, with its registered preregistration.

    Read off the RESOLVED config, never off ``--exp``, for the same reason
    :func:`is_execution_gated` is: ``--config`` outranks ``--exp`` and a
    hand-rolled config never names an experiment. Returns ``(axis, path)``
    pairs; ``path is None`` means the axis is active but unregistered —
    the CLI preflight refuses that combination for non-tiny runs.
    """
    axes: list[tuple[str, str | None]] = []
    model = cfg.model
    if model.embedding_freeze != "none":
        axes.append(("embedding_freeze", E15_PREREGISTRATIONS.get("embedding_freeze")))
    if model.gram_lambda != 0.0:
        axes.append(("gram_lambda", E15_PREREGISTRATIONS.get("gram_lambda")))
    if model.state_source == "field":
        axes.append(("state_source", E15_PREREGISTRATIONS.get("state_source")))
    return tuple(axes)


def is_execution_gated(cfg: Config) -> bool:
    """Does ``cfg`` describe a run held behind the E3 kappa readout?

    Read off the RESOLVED config, never off ``--exp``: ``--config
    configs/e3-anchor.yaml`` is a supported invocation that leaves ``--exp``
    unset, and an explicit ``--config`` outranks ``--exp`` when both are given,
    so the flag does not say what runs. The gated feature is the asymmetric
    anchor pair (docs/e-3-instruction.md), which is exactly what
    ``anchor_mode != "none"`` selects — including in a hand-rolled config that
    never names an experiment.
    """
    return cfg.model.anchor_mode != "none"


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
