"""Shared deterministic fixtures for feature-local predictive-state tests."""

from __future__ import annotations

import torch
from torch import nn

from entropic_substrate.core.latent.future_law import MeanDisplacementBaseline
from entropic_substrate.core.latent.target_values import FutureTarget
from entropic_substrate.core.observation.contracts import CodecIdentity, EncodedObservations
from entropic_substrate.kernel.runtime.config import ModelConfig

__all__ = [
    "CompositeTargetBranch",
    "codec_identity",
    "encoded",
    "fit_mean_baseline",
    "model_config",
]


def codec_identity() -> CodecIdentity:
    return CodecIdentity(name="predictive-test", construction_digest="b" * 64, cardinality=17)


def model_config(**overrides: object) -> ModelConfig:
    cfg = ModelConfig(
        d_model=8,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2.0,
        block_size=16,
        dropout=0.0,
        tokenizer="byte",
        vocab_size=17,
        tie_embeddings=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def encoded(*, batch: int = 2, time: int = 6, width: int = 8) -> EncodedObservations:
    generator = torch.Generator().manual_seed(23)
    return EncodedObservations(
        values=torch.randn(batch, time, width, generator=generator),
        codec_identity=codec_identity(),
    )


def fit_mean_baseline(*, width: int, values: torch.Tensor) -> MeanDisplacementBaseline:
    baseline = MeanDisplacementBaseline(latent_width=width)
    baseline.fit(FutureTarget(values=values, codec_identity=codec_identity()))
    return baseline


class CompositeTargetBranch(nn.Module):
    """Target observation-adapter + encoder stand-in for policy ownership tests."""

    def __init__(self) -> None:
        super().__init__()
        self.adapter = nn.Linear(3, 4)
        self.encoder = nn.Linear(4, 4)
        self.forward_grad_enabled: bool | None = None

    def forward(self, values: torch.Tensor) -> FutureTarget:
        self.forward_grad_enabled = torch.is_grad_enabled()
        encoded_values = self.encoder(torch.tanh(self.adapter(values)))
        return FutureTarget(values=encoded_values, codec_identity=codec_identity())
