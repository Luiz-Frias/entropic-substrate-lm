"""Transformer backbones and blocks."""

from __future__ import annotations

from entropic_substrate.models.backbone import (
    LanguageModel,
    LMOutput,
    build_model,
    count_parameters,
)
from entropic_substrate.models.blocks import (
    BaselineBlock,
    CausalSelfAttention,
    NGPTBlock,
    build_block,
    build_norm,
    rotary_embeddings,
)

__all__ = [
    "BaselineBlock",
    "CausalSelfAttention",
    "LMOutput",
    "LanguageModel",
    "NGPTBlock",
    "build_block",
    "build_model",
    "build_norm",
    "count_parameters",
    "rotary_embeddings",
]
