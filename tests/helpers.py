"""Test helpers shared across modules (importable as `helpers`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from entropic_substrate.config import Config, DataConfig, ModelConfig, TrainConfig

__all__ = [
    "load_compose",
    "mlflow_command",
    "mlflow_service_config",
    "tiny_config",
    "tiny_model_config",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_model_config(**overrides: object) -> ModelConfig:
    """A CPU-sized :class:`ModelConfig`."""
    cfg = ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        block_size=64,
        tokenizer="byte",
        tie_embeddings=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def load_compose() -> dict[str, Any]:
    """Parse ``compose.yaml`` fresh on every call (no caching — these are
    cheap, infrequent static-config checks, not a hot path).
    """
    text = (_REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        msg = (
            f"compose.yaml did not parse to a mapping (got {type(parsed).__name__}) — "
            "an empty or malformed compose.yaml would otherwise surface as an opaque "
            "KeyError/TypeError several calls downstream in mlflow_service_config()"
        )
        raise ValueError(msg)
    return parsed


def mlflow_service_config() -> dict[str, Any]:
    """``compose.yaml``'s ``services.mlflow`` block, parsed."""
    return load_compose()["services"]["mlflow"]


def mlflow_command() -> list[str]:
    """``compose.yaml``'s ``services.mlflow.command`` as a FLAT argv list.

    Parsed with a real YAML loader rather than string-sliced, so this stays
    correct regardless of how the surrounding comment block is reflowed.

    The list shape is itself the contract. An earlier revision wrapped this
    command in ``sh -c`` so the container's own shell could expand
    ``${VAR:+--flag "$VAR"}`` and make the allowlist flags OPTIONAL. That
    wrapper is gone: MLflow binds ``--allowed-hosts`` /
    ``--cors-allowed-origins`` to ``MLFLOW_SERVER_ALLOWED_HOSTS`` /
    ``MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`` natively (click ``envvar=``,
    mlflow/utils/cli_args.py:256,270 in the pinned image), so the flags are
    never passed and nothing needs a shell. Asserting the flat shape here
    means a regression to any shell wrapper fails LOUDLY in one place
    instead of silently reintroducing a command-text interpolation site.
    """
    service = mlflow_service_config()
    command = service.get("command")
    assert command, "mlflow service must declare a command"
    assert isinstance(command, list), (
        f"mlflow's command must be a flat argv LIST, not a shell string; got {command!r}"
    )
    assert command[:2] != ["sh", "-c"], (
        "mlflow's command is wrapped in `sh -c` again. That wrapper existed only to "
        "make the allowlist flags optional, and it is exactly the command-TEXT "
        "interpolation site that allowed a measured shell injection. MLflow reads "
        "MLFLOW_SERVER_ALLOWED_HOSTS/MLFLOW_SERVER_CORS_ALLOWED_ORIGINS natively — "
        f"pass them via `environment:` instead; got {command!r}"
    )
    return [str(entry) for entry in command]


def tiny_config(tmp_path: Path, **model_overrides: object) -> Config:
    """A complete CPU smoke-test :class:`Config` rooted at ``tmp_path``."""
    return Config(
        name="pytest-tiny",
        model=tiny_model_config(**model_overrides),
        data=DataConfig(
            source="synthetic",
            synthetic_bytes=80_000,
            batch_size=4,
            seq_len=32,
            val_fraction=0.1,
            num_eval_batches=2,
        ),
        train=TrainConfig(
            max_steps=5,
            lr=3e-3,
            warmup_steps=2,
            eval_interval=0,
            log_interval=1,
            save_interval=0,
            amp_dtype="off",
            device="cpu",
            results_dir=str(tmp_path / "results"),
            models_dir=str(tmp_path / "models"),
        ),
    )
