r""":math:`\sqrt{p}` Fisher-Rao initialisation of byte identity vectors.

Idea
----
The natural geometry of a categorical distribution is the Fisher-Rao metric.
Under the square-root parametrisation :math:`p \mapsto \sqrt{p}` the
probability simplex :math:`\Delta^{255}` maps isometrically (up to a factor 2)
onto the *positive orthant* of the unit sphere :math:`S^{255}`, and the
Fisher-Rao distance becomes the plain angular distance

.. math::
    d_{FR}(p, q) = 2\arccos\Big(\sum_b \sqrt{p_b q_b}\Big).

So: estimate, for each byte :math:`b`, its empirical next-byte distribution
:math:`p(\cdot\mid b)` on a corpus sample (Laplace smoothed), take the
element-wise square root, and you have a 256-dimensional unit vector per byte
whose geometry *is* the information geometry of that byte's successor
statistics. Those 256-dim vectors are then lifted into ``d_model`` by a fixed
semi-orthogonal matrix, which preserves all inner products (hence all angles
and all Fisher-Rao distances) exactly.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import torch

__all__ = [
    "build_fisher_init",
    "count_transitions",
    "fisher_sqrt_simplex_embedding",
    "load_init",
    "random_sphere_init",
    "save_init",
    "semi_orthogonal_lift",
]

N_BYTES = 256


def count_transitions(
    stream: Iterable[bytes] | bytes | bytearray,
    max_bytes: int | None = None,
) -> torch.Tensor:
    """Count the 256x256 next-byte transition matrix over ``stream``.

    Parameters
    ----------
    stream:
        Either a single ``bytes``/``bytearray`` blob or an iterable of chunks
        (e.g. a streaming dataset yielding utf-8 encoded documents).
    max_bytes:
        Stop after consuming this many bytes (``None`` = consume everything).

    Returns
    -------
    torch.Tensor
        ``(256, 256)`` float64 counts; ``counts[i, j]`` = #(byte ``j`` follows
        byte ``i``). Chunk boundaries are stitched so no transition is lost.
    """
    counts = torch.zeros(N_BYTES, N_BYTES, dtype=torch.float64)
    chunks: Iterable[bytes | bytearray]
    if isinstance(stream, (bytes, bytearray)):
        chunks = (stream,)
    else:
        chunks = stream

    consumed = 0
    carry: int | None = None
    for chunk in chunks:
        if not chunk:
            continue
        if max_bytes is not None:
            remaining = max_bytes - consumed
            if remaining <= 0:
                break
            chunk = chunk[:remaining]
        arr = torch.frombuffer(bytearray(chunk), dtype=torch.uint8).long()
        consumed += int(arr.numel())
        if carry is not None:
            counts[carry, int(arr[0])] += 1.0
        if arr.numel() >= 2:
            flat = arr[:-1] * N_BYTES + arr[1:]
            counts.view(-1).index_add_(0, flat, torch.ones(flat.numel(), dtype=torch.float64))
        carry = int(arr[-1])
        if max_bytes is not None and consumed >= max_bytes:
            break
    return counts


def fisher_sqrt_simplex_embedding(counts: torch.Tensor, smoothing: float = 1.0) -> torch.Tensor:
    """Map transition counts to unit vectors on the positive orthant of ``S^255``.

    Laplace(``smoothing``) smoothing guarantees strictly positive rows, hence
    strictly positive coordinates after the square root (open positive
    orthant), which also makes the map injective and the geometry well defined
    for bytes that never occur in the sample.
    """
    if counts.shape != (N_BYTES, N_BYTES):
        msg = f"expected (256, 256) counts, got {tuple(counts.shape)}"
        raise ValueError(msg)
    if smoothing <= 0:
        msg = "smoothing must be > 0 to keep rows in the open positive orthant"
        raise ValueError(msg)
    probs = counts.double() + smoothing
    probs = probs / probs.sum(dim=1, keepdim=True)
    root = probs.sqrt()
    # Exact re-normalisation kills the float error left by sqrt/sum.
    return root / root.norm(dim=1, keepdim=True)


def semi_orthogonal_lift(
    d_in: int, d_out: int, seed: int = 0, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Fixed ``(d_in, d_out)`` semi-orthogonal matrix built from a seeded QR.

    * ``d_out >= d_in`` (the normal case, e.g. 256 -> 768): ``Q @ Q.T == I_{d_in}``,
      so the map ``x -> x @ Q`` is an **exact isometry** - every inner product,
      hence every Fisher-Rao angle, survives the lift unchanged.
    * ``d_out < d_in`` (only used by the CPU-tiny configs): ``Q.T @ Q ==
      I_{d_out}``, i.e. a random orthogonal *projection*. Angles are then
      preserved only in the Johnson-Lindenstrauss sense, which is fine for the
      smoke configs but is not the regime the experiments run in.

    Deterministic for a given ``seed`` (sign-fixed so LAPACK's arbitrary QR
    sign convention cannot change the result across platforms).
    """
    if d_in <= 0 or d_out <= 0:
        msg = f"dimensions must be positive, got d_in={d_in}, d_out={d_out}"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, cols = (d_out, d_in) if d_out >= d_in else (d_in, d_out)
    gaussian = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(gaussian, mode="reduced")
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    matrix = q.T if d_out >= d_in else q
    return matrix.contiguous().to(device=device)


def build_fisher_init(
    stream: Iterable[bytes] | bytes | bytearray,
    d_model: int,
    *,
    max_bytes: int | None = 50_000_000,
    smoothing: float = 1.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""End-to-end :math:`\sqrt{p}` Fisher init: corpus -> ``(256, d_model)``.

    The returned rows are unit-norm to ~1e-6 and their pairwise inner products
    equal :math:`\sum_b \sqrt{p(b\mid i)p(b\mid j)}` (the Bhattacharyya
    coefficient), i.e. the Fisher-Rao angular geometry is preserved by the lift.
    """
    counts = count_transitions(stream, max_bytes=max_bytes)
    sqrt_p = fisher_sqrt_simplex_embedding(counts, smoothing=smoothing)
    lift = semi_orthogonal_lift(N_BYTES, d_model, seed=seed)
    lifted = sqrt_p @ lift
    lifted = lifted / lifted.norm(dim=1, keepdim=True)
    return lifted.to(dtype)


def random_sphere_init(
    n_symbols: int, d_model: int, *, seed: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Uniform-on-``S^{d_model-1}`` control initialisation (E1 ablation arm)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(n_symbols, d_model, generator=generator, dtype=torch.float64)
    raw = raw / raw.norm(dim=1, keepdim=True)
    return raw.to(dtype)


def save_init(tensor: torch.Tensor, path: str | os.PathLike[str]) -> Path:
    """Persist an initialisation tensor as ``.pt`` (creates parent dirs)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), out)
    return out


def load_init(path: str | os.PathLike[str]) -> torch.Tensor:
    """Load an initialisation tensor saved by :func:`save_init`."""
    loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(loaded, torch.Tensor):
        msg = f"{path} does not contain a tensor"
        raise TypeError(msg)
    return loaded
