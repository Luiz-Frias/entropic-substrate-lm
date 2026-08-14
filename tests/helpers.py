"""Test helpers shared across modules (importable as `helpers`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from entropic_substrate.config import Config, DataConfig, ModelConfig, TrainConfig

__all__ = [
    "load_compose",
    "mlflow_command_script",
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
    return yaml.safe_load(text)


def mlflow_service_config() -> dict[str, Any]:
    """``compose.yaml``'s ``services.mlflow`` block, parsed."""
    return load_compose()["services"]["mlflow"]


def mlflow_command_script() -> str:
    """The literal ``sh -c`` script compose.yaml hands to the mlflow
    container — a single string (YAML folded scalar), NOT a flat argv list.
    Parsed with a real YAML loader rather than string-sliced, so this stays
    correct regardless of how the surrounding comment block is reflowed.

    Shared by tests/test_deploy_config.py and tests/test_compose_config.py:
    both need the same parsed script (worker count in one, allowlist flag
    shape in the other), and both need the SAME ``["sh", "-c"]`` prefix
    check — a test that skips it and reads ``command[-1]`` would silently
    keep passing against an unrelated element if the command ever became a
    flat argv list again.
    """
    service = mlflow_service_config()
    command = service.get("command")
    assert command, "mlflow service must declare a command"
    assert command[:2] == ["sh", "-c"], (
        "mlflow's command must stay an `sh -c` wrapper — that's what makes "
        "--allowed-hosts/--cors-allowed-origins OPTIONAL flags (see "
        f"compose.yaml's own HOST ALLOWLIST comment); got {command!r}"
    )
    return command[2]


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
