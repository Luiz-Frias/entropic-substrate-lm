"""Test helpers shared across modules (importable as `helpers`)."""

from __future__ import annotations

from pathlib import Path

from entropic_substrate.config import Config, DataConfig, ModelConfig, TrainConfig

__all__ = ["tiny_config", "tiny_model_config"]


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
