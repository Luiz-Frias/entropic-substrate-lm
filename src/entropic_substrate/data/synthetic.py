"""Deterministic synthetic corpus used by CPU smoke tests and ``--tiny`` runs.

It must be *learnable fast*: the 5-step smoke test asserts a decreasing loss,
so the stream is a low-entropy templated text with strong local structure
rather than random bytes.
"""

from __future__ import annotations

import random

__all__ = ["synthetic_text"]

_SUBJECTS = ("the substrate", "a byte", "the sphere", "curvature", "the gate")
_VERBS = ("encodes", "retracts", "mixes", "normalises", "routes")
_OBJECTS = ("information", "geometry", "entropy", "the manifold", "a token")


def synthetic_text(num_chars: int = 200_000, *, seed: int = 0) -> str:
    """Generate a reproducible, highly-structured pseudo-English corpus."""
    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    while total < num_chars:
        sentence = (
            f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
            f"{rng.choice(_OBJECTS)} on the unit hypersphere. "
        )
        parts.append(sentence)
        total += len(sentence)
    return "".join(parts)[:num_chars]
