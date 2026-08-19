"""Normalized-Gram-matrix geometry, shared by E1.5b's regularizer and E1.5-D's
cross-arm diagnostics (`docs/e1_5-instruction.md`).

A symbol table's RELATIONAL geometry — the pairwise angles between its rows —
is what E1.5b's penalty and E1.5-D's cross-arm agreement both actually care
about, not the rows' absolute scale or their absolute position in ambient
space (which a random rotation of the whole table would change while leaving
every downstream prediction identical). The normalized Gram matrix
`G = E_hat @ E_hat.T` (rows L2-normalized first) is exactly that: it is
invariant to any orthogonal transform of the table and to any PER-ROW
rescale, and it collapses to the pairwise cosine similarity matrix — stronger
than row-normalization alone, which fixes each row's radius but says nothing
about the angles between rows.

Non-negotiable numerics (CLAUDE.md): both functions run in FP32 inside an
explicit ``torch.autocast(enabled=False)`` block regardless of the caller's
ambient autocast state — mirrors ``geometry/kappa.py``'s scalar reductions.
"""

from __future__ import annotations

import torch

__all__ = ["gram_frobenius_sq", "normalized_gram"]

# Row-norm floor: guards the divide against an all-zero row without moving
# any genuinely non-degenerate row's cosine values. Deliberately NOT
# torch.finfo(torch.float32).tiny (~1.1755e-38, the smallest floor that
# would still let a technically-nonzero row through) — MEASURED: a row at
# norm ~3e-20 (well above that floor, i.e. still "nonzero" by it) divided by
# a fp32-tiny floor produces a full unit-norm vector whose direction is
# pure fp32 rounding noise from values near the representable limit, not a
# recoverable direction; the same row under this 1e-12 floor instead comes
# out correctly suppressed (norm ~3e-8), signaling "no real direction here"
# rather than fabricating one. 1e-12 is chosen as "far enough above fp32
# rounding noise to trust the division, far enough below any real embedding
# row's norm (rows are ~unit-norm by construction under this repo's nGPT
# retraction) to never clip a legitimate row."
_NORM_EPS: float = 1e-12


def normalized_gram(embedding: torch.Tensor) -> torch.Tensor:
    """Row-normalized Gram matrix ``G[i, j] = cos(E[i], E[j])`` of ``embedding``.

    ``embedding`` is ``(n_symbols, d)``. Always computed in FP32 (promoted
    from whatever dtype the caller passes, including bf16 under autocast) and
    clamped to ``[-1, 1]``: unit rows dotted in floating point can exceed that
    range by O(1e-7) from rounding, and an unclamped value there would make a
    caller's ``arccos`` of it (E1.5-D's FR-init distance,
    ``docs/e1_5-instruction.md``) silently return NaN.

    Raises:
        ValueError: if ``embedding`` is not 2-D. A 1-D input would fail
            confusingly inside ``.norm(p=2, dim=1, ...)`` (no dim 1 to
            reduce); a 3-D+ input would not fail at all, silently
            normalizing per-slice instead of per-row against the
            documented ``(n_symbols, d)`` contract. Fail fast instead
            (CLAUDE.md Errors principle).
    """
    if embedding.ndim != 2:
        msg = f"normalized_gram: expected a 2-D embedding, got shape {tuple(embedding.shape)}"
        raise ValueError(msg)
    with torch.autocast(device_type=embedding.device.type, enabled=False):
        e = embedding.float()
        unit = e / e.norm(p=2, dim=1, keepdim=True).clamp_min(_NORM_EPS)
        gram = unit @ unit.T
        return gram.clamp(-1.0, 1.0)


def gram_frobenius_sq(gram_a: torch.Tensor, gram_b: torch.Tensor) -> torch.Tensor:
    """``||gram_a - gram_b||_F^2`` — the squared Frobenius norm of the difference.

    This IS the E1.5b penalty term, up to the ``gram_lambda`` scale:
    ``L = L_CE + gram_lambda * gram_frobenius_sq(G(E_t), G(E_init))``. Callers
    are expected to pass ``normalized_gram`` outputs (or anything already
    respecting its fp32-and-clamped contract); this function does not
    reclamp, only promotes to FP32 inside its own ``autocast(enabled=False)``
    block so the reduction itself stays well-conditioned under an ambient
    bf16 autocast.

    MEASURED (2026-08-18, CPU): this wrapper's own ops (subtract, multiply,
    ``.sum()``) are not on ``torch.autocast``'s CPU downcast list — a bf16
    input run through them under ``torch.autocast("cpu", dtype=bfloat16)``
    stays bf16 whether or not this ``enabled=False`` block is present.
    Callers' upstream ``.float()`` (this module's own ``normalized_gram``,
    and ``train.py``'s ``_gram_penalty_terms``) already guarantees fp32
    inputs by the time this function runs, so no CPU test can distinguish
    this wrapper's presence from its absence — see
    ``test_gram_primitives_are_fp32_finite_and_differentiable_under_bf16_autocast``
    in ``tests/test_gram_penalty.py``, which passes either way. Retained per
    CLAUDE.md's non-negotiable geometry-numerics invariant (defense in depth
    against a caller that skips the ``.float()`` upstream, and against a
    CUDA autocast op-list that may differ from CPU's) — this is a coverage
    gap in the test suite, not evidence the wrapper is unnecessary.

    Raises:
        ValueError: if the two Gram matrices' shapes differ. Both callers in
            this repo (``Trainer._gram_penalty_terms``, and E1.5-D's planned
            cross-arm agreement) always diff two ``(n_symbols, n_symbols)``
            matrices for the SAME table's own symbol count — a mismatch can
            only mean two tables of different vocabularies were compared,
            which is a caller bug. Failing fast here (CLAUDE.md Errors
            principle) refuses that rather than letting ``-`` silently
            broadcast (or, for shapes that happen to be broadcast-compatible,
            silently produce a meaningless value instead of erroring at all).
    """
    if gram_a.shape != gram_b.shape:
        msg = f"gram_frobenius_sq: shape mismatch {tuple(gram_a.shape)} vs {tuple(gram_b.shape)}"
        raise ValueError(msg)
    with torch.autocast(device_type=gram_a.device.type, enabled=False):
        diff = gram_a.float() - gram_b.float()
        return (diff * diff).sum()
