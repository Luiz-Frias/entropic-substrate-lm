"""E1.5-D embedding-geometry diagnostics: does training preserve the Fisher
geometry an arm's table started at, and do two arms' tables agree on the
same relational structure?

Torch/NumPy only — **no matplotlib import anywhere in this module** (chart
RENDERING is ``entropic_substrate.analysis.charts``'s job, same "pure
transform, no I/O" boundary that module already draws for E1/E2/E3).

Five independent questions, five functions:

* :func:`angular_drift` — per-symbol angle (fp64) between a symbol's INIT row
  and its FINAL (trained) row. Zero means the geometry never moved.
* :func:`gram_retention` — does the *relational* structure (pairwise angles)
  survive training, even where individual rows drift? Correlates each pair's
  FR-init distance against its final cosine distance.
* :func:`cross_arm_gram_agreement` — do two arms' FINAL tables agree on the
  same relational geometry, independent of any shared rotation between them
  (``geometry/gram.py``'s ``normalized_gram`` is exactly rotation-invariant).
* :func:`procrustes_align` — the orthogonal transform that best aligns one
  table onto another (``torch.linalg.svd`` on ``AᵀB``), plus the residual
  it leaves — cross-arm agreement's diagnostic complement: agreement says
  *whether* two tables share geometry, this says *how* to overlay them.
* :func:`functional_divergence` — do two trained MODELS (not just their
  input tables) actually predict differently on the same held-out data?
  KL both ways plus Jensen-Shannon, deterministic eval-split iteration.

:func:`load_checkpoint_model` and :func:`reconstruct_init_table` are the
retrieval side: a checkpoint on an analysis machine that holds neither the
enwik8 corpus nor a cached fisher-init ``.pt`` file must still load and (for
non-default init arms) reconstruct the table its embedding started from.

Non-negotiable numerics (CLAUDE.md): every geometric reduction here runs in
FP32-or-better inside an explicit ``torch.autocast(enabled=False)`` block
regardless of ambient autocast, mirroring ``geometry/gram.py``/
``geometry/kappa.py``. Angular quantities (:func:`angular_drift`,
:func:`gram_retention`'s FR-init distance) additionally promote to float64 —
this module runs offline, off the training hot path, so the extra precision
costs nothing and buys headroom against ``arccos``'s steep derivative near
the poles.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from entropic_substrate.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.config import Config, config_from_dict
from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.data.build import build_corpus, corpus_byte_stream
from entropic_substrate.geometry.gram import gram_frobenius_sq, normalized_gram
from entropic_substrate.init.fisher import build_fisher_init, load_init, random_sphere_init
from entropic_substrate.logging import logger
from entropic_substrate.models.backbone import LanguageModel, build_model, vocab_size_for

__all__ = [
    "CheckpointModel",
    "FunctionalDivergence",
    "GramRetention",
    "ProcrustesResult",
    "angular_drift",
    "cross_arm_gram_agreement",
    "functional_divergence",
    "gram_retention",
    "load_checkpoint_model",
    "procrustes_align",
    "reconstruct_init_table",
]

# Matches backbone.py's own fisher-rebuild fallback (Trainer.__init__,
# train.py:168) exactly — a DIFFERENT number here would silently make the
# reconstruction not the-table-the-run-actually-used.
_ANALYSIS_FISHER_CORPUS_BYTES: int = 20_000_000

# Row-cosine clamp floor, shared by every arccos site in this module: fp64
# unit-row dot products still overshoot [-1, 1] by rounding (same hazard
# geometry/gram.py's _clamp_cosine documents for fp32), and an unclamped
# arccos(1 + 1e-10) is NaN.
_COS_CLAMP = 1.0
_ROW_NORM_EPS: float = 1e-12
# Same value as _ROW_NORM_EPS, kept as its own name: this one clamps a
# Pearson-correlation denominator (_pearson), a different hazard site than
# a row-norm floor even though the guard-magnitude happens to match.
_CORR_DENOM_EPS: float = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointModel:
    """A loaded, weight-restored model plus the run config that produced it."""

    model: LanguageModel
    config: Config
    step: int
    path: Path


def load_checkpoint_model(
    path: str | Path, *, device: torch.device | str = "cpu"
) -> CheckpointModel:
    """Load a checkpoint into a fully weight-restored, eval-mode model.

    ``embedding_init`` is forced to ``"default"`` for the PRE-load build only
    — the analysis machine may hold neither the enwik8 corpus nor a cached
    fisher-init ``.pt`` (``embedding_init="fisher_sqrt"`` would otherwise
    demand one or the other at construction time, see ``backbone.py``'s
    ``_fisher_weights``), and whatever values that forced-default build draws
    are immediately overwritten by ``load_state_dict`` with the checkpoint's
    REAL trained weights below. Every other architecture-defining field
    (``d_model``, ``block``, ``state_source``, ``field_*``, ...) is preserved
    from the checkpoint's own config, since those determine tensor SHAPES
    that ``load_state_dict`` must match exactly.

    Raises:
        CheckpointFormatError: the file is not a loadable torch checkpoint at
            all (corrupt/truncated/non-checkpoint bytes) — re-raised from
            whatever ``torch.load`` raises, naming the path, so a malformed
            capture reads as the same controlled refusal every other
            checkpoint-format problem in this repo does (``checkpoint.py``),
            not an opaque pickle/zip traceback. Also raised by
            :func:`~entropic_substrate.checkpoint.normalize_checkpoint` for
            an unversioned-with-tau or too-new checkpoint.
    """
    path = Path(path)
    try:
        # map_location="cpu": a checkpoint captured off a CUDA training pod
        # must still load on an analysis machine with no GPU at all.
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except CheckpointFormatError:
        raise
    except Exception as exc:  # re-raised as a named, typed refusal below (any load failure)
        msg = f"{path}: could not be loaded as a torch checkpoint ({exc})"
        raise CheckpointFormatError(msg) from exc
    state = normalize_checkpoint(raw, source=str(path))
    cfg = config_from_dict(state["config"])
    build_cfg = dataclasses.replace(cfg.model, embedding_init="default")
    model = build_model(build_cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return CheckpointModel(model=model, config=cfg, step=int(state.get("step", 0)), path=path)


def reconstruct_init_table(cfg: Config) -> torch.Tensor:
    """Rebuild the INIT table ``cfg``'s embedding started training from.

    Code-truth reconstruction (CLAUDE.md finding #2), not the doc's surface
    reading: ``backbone.py``'s ``_init_embeddings`` calls
    ``random_sphere_init(vocab_size, d_model)`` with **no seed argument** —
    every ``embedding_init="random_sphere"`` run, regardless of its OWN
    ``train.seed``, draws from that function's default ``seed=0``. Every
    random-arm checkpoint therefore shares exactly ONE init table; this
    reconstruction reproduces that sharing, not a per-seed-varied one.

    Raises:
        ValueError: ``embedding_init == "default"`` (no init table exists
            for that arm — nothing to reconstruct), a ``fisher_sqrt`` config
            on a non-byte tokenizer, or a shape mismatch between an on-disk
            fisher table and ``cfg.model.d_model``.
    """
    model_cfg = cfg.model
    mode = model_cfg.embedding_init
    if mode == "random_sphere":
        vocab = vocab_size_for(model_cfg)
        return random_sphere_init(vocab, model_cfg.d_model)
    if mode == "fisher_sqrt":
        if model_cfg.tokenizer != "byte":
            msg = (
                "reconstruct_init_table: fisher_sqrt init is defined over the "
                f"256-symbol byte alphabet; tokenizer={model_cfg.tokenizer!r} is not supported"
            )
            raise ValueError(msg)
        path = model_cfg.fisher_init_path
        if path is not None and Path(path).exists():
            weights = load_init(path)
            expected = (256, model_cfg.d_model)
            if tuple(weights.shape) != expected:
                msg = (
                    f"reconstruct_init_table: {path} has shape {tuple(weights.shape)}, "
                    f"expected {expected}"
                )
                raise ValueError(msg)
            return weights
        logger.warning(
            "reconstruct_init_table: no fisher_init_path on disk for this checkpoint's "
            f"config — rebuilding the Fisher init table from a fresh "
            f"{_ANALYSIS_FISHER_CORPUS_BYTES:,}-byte corpus stream (mirrors backbone."
            "_fisher_weights' own fallback exactly). This is NOT guaranteed bit-identical "
            "to the table the training run actually consumed if the corpus source has "
            "changed since."
        )
        stream = b"".join(corpus_byte_stream(cfg.data, _ANALYSIS_FISHER_CORPUS_BYTES))
        return build_fisher_init(stream, model_cfg.d_model)
    msg = f"reconstruct_init_table: no init table exists for embedding_init={mode!r}"
    raise ValueError(msg)


def _check_paired_shape(a: torch.Tensor, b: torch.Tensor, *, fn: str) -> None:
    if a.shape != b.shape:
        msg = f"{fn}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"
        raise ValueError(msg)
    if a.ndim != 2:
        msg = f"{fn}: expected 2-D (n_symbols, d) tensors, got shape {tuple(a.shape)}"
        raise ValueError(msg)


def _unit_rows_f64(table: torch.Tensor) -> torch.Tensor:
    e = table.double()
    return e / e.norm(p=2, dim=1, keepdim=True).clamp_min(_ROW_NORM_EPS)


def angular_drift(init_table: torch.Tensor, final_table: torch.Tensor) -> torch.Tensor:
    """Per-symbol angular distance (radians, fp64) between init and final rows.

    ``0.0`` for a symbol whose row never moved; ``pi`` for one that flipped
    to the antipode. Both inputs are row-renormalized before comparing, so a
    caller need not pre-verify unit norm (a frozen table is already unit-norm
    to ~1e-7, per ``backbone.py``'s ``normalize_weights`` guard comment, but
    this function does not assume it).
    """
    _check_paired_shape(init_table, final_table, fn="angular_drift")
    with torch.autocast(device_type=init_table.device.type, enabled=False):
        a = _unit_rows_f64(init_table)
        b = _unit_rows_f64(final_table)
        cos = (a * b).sum(dim=1).clamp(-_COS_CLAMP, _COS_CLAMP)
        return torch.arccos(cos)


@dataclasses.dataclass(frozen=True, slots=True)
class GramRetention:
    """Does a table's RELATIONAL geometry (pairwise angles) survive training?

    Correlates each symbol pair's FR-init distance (``2 * arccos`` of the
    init table's cosine, the Fisher-Rao angular distance under the √p map)
    against the same pair's final cosine distance (``1 - cos``). ``pearson_r``
    /``spearman_r`` near ``1.0`` mean pairs that started far apart (in
    Fisher-Rao terms) are still far apart (in plain cosine terms) after
    training — the relational structure was preserved even where individual
    rows (:func:`angular_drift`) moved.
    """

    pearson_r: float
    spearman_r: float
    n_pairs: int


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Argsort-based ranks (Spearman's definition), ties broken by position.

    No tie-correction term: the two distance families this module correlates
    (arccos of a Gram row, 1 - a Gram row) are continuous functions of
    continuous cosine values, so exact ties are a measure-zero event in
    practice — the same "argsort-Spearman" the plan names, not the
    tie-adjusted textbook estimator.
    """
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    # A degenerate (constant) side yields 0.0, not NaN — a deliberate choice.
    # Named separately from _ROW_NORM_EPS (same value, different site: this
    # clamps a Pearson denominator, not a row norm) so a reader never has to
    # check whether the two clamps are coupled on purpose.
    denom = (xc.norm() * yc.norm()).clamp_min(_CORR_DENOM_EPS)
    return float((xc * yc).sum() / denom)


def gram_retention(init_table: torch.Tensor, final_table: torch.Tensor) -> GramRetention:
    """Pearson + (argsort) Spearman correlation of FR-init vs final-cosine
    pairwise distance, over every off-diagonal symbol pair.

    A monotonic transform of either distance family (e.g. swapping ``1 - cos``
    for ``arccos(cos)`` on the final side) cannot change ``spearman_r`` — rank
    correlation is transform-invariant — which is why this function's docstring
    and the plan both single Pearson out as the transform-SENSITIVE readout.
    """
    _check_paired_shape(init_table, final_table, fn="gram_retention")
    n = init_table.shape[0]
    if n < 2:
        msg = f"gram_retention: need at least 2 symbols to form a pair, got {n}"
        raise ValueError(msg)
    g_init = normalized_gram(init_table)
    g_final = normalized_gram(final_table)
    iu0, iu1 = torch.triu_indices(n, n, offset=1)
    with torch.autocast(device_type=init_table.device.type, enabled=False):
        fr_init_dist = (
            2.0 * torch.arccos(g_init[iu0, iu1].clamp(-_COS_CLAMP, _COS_CLAMP))
        ).double()
        final_cos_dist = (1.0 - g_final[iu0, iu1]).double()
    return GramRetention(
        pearson_r=_pearson(fr_init_dist, final_cos_dist),
        spearman_r=_pearson(_rank(fr_init_dist), _rank(final_cos_dist)),
        n_pairs=int(fr_init_dist.numel()),
    )


def cross_arm_gram_agreement(table_a: torch.Tensor, table_b: torch.Tensor) -> float:
    """Rotation-invariant Gram-matrix disagreement between two symbol tables.

    ``0.0`` means the two tables encode EXACTLY the same relational geometry
    (pairwise angles), independent of any shared orthogonal transform between
    them — ``normalized_gram(E @ Q) == normalized_gram(E)`` for any orthogonal
    ``Q`` (``geometry/gram.py``'s own invariance), so two arms whose absolute
    embedding ORIENTATION differs but relational geometry agrees score
    identically to two arms whose tables are bitwise equal. Larger values mean
    more disagreement (this is ``gram_frobenius_sq``, a squared distance, not
    a bounded similarity score).
    """
    _check_paired_shape(table_a, table_b, fn="cross_arm_gram_agreement")
    return float(gram_frobenius_sq(normalized_gram(table_a), normalized_gram(table_b)))


@dataclasses.dataclass(frozen=True, slots=True)
class ProcrustesResult:
    """The orthogonal transform best aligning ``table_a`` onto ``table_b``."""

    rotation: torch.Tensor
    residual: float


def procrustes_align(table_a: torch.Tensor, table_b: torch.Tensor) -> ProcrustesResult:
    """Orthogonal Procrustes: ``R = argmin_{R orthogonal} ||A @ R - B||_F``.

    Closed form via SVD of ``AᵀB`` (``R = U @ Vᵀ``): fp64 throughout, inside
    an explicit ``autocast(enabled=False)`` block. Cross-arm agreement's
    diagnostic complement — :func:`cross_arm_gram_agreement` says WHETHER two
    tables share relational geometry; a small :attr:`ProcrustesResult.residual`
    here says the SAME thing from a different angle (literally: it names the
    rotation that overlays one onto the other), while a small
    ``cross_arm_gram_agreement`` and a LARGE residual together would mean the
    two tables share pairwise angles but no single global rotation aligns them
    — a signature worth flagging on its own.
    """
    _check_paired_shape(table_a, table_b, fn="procrustes_align")
    with torch.autocast(device_type=table_a.device.type, enabled=False):
        a = table_a.double()
        b = table_b.double()
        m = a.T @ b
        u, _, vh = torch.linalg.svd(m)
        r = u @ vh
        residual = float(torch.linalg.norm(a @ r - b))
    return ProcrustesResult(rotation=r, residual=residual)


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionalDivergence:
    """KL (both directions) + Jensen-Shannon divergence between two models'
    next-symbol predictive distributions, averaged over held-out positions.

    ``js`` is bounded above by ``ln(2)`` (nats) by construction — two models
    that agree nowhere still cannot exceed that bound; it is not a measured
    fact about these particular models, it is a property of JS divergence
    itself, pinned as a test invariant.
    """

    kl_a_to_b: float
    kl_b_to_a: float
    js: float
    n_positions: int


def functional_divergence(
    ckpt_a: CheckpointModel,
    ckpt_b: CheckpointModel,
    *,
    num_batches: int = 8,
    device: torch.device | str = "cpu",
) -> FunctionalDivergence:
    """Do two trained models actually predict differently on the same data?

    Deterministic eval-split iteration: a fresh ``SequenceSampler(train=False,
    seed=cfg.train.seed + 1)`` over ``ckpt_a``'s own corpus — the EXACT
    construction :class:`~entropic_substrate.train.Trainer` uses for its own
    ``val_sampler`` (``train.py``), so this reads the identical held-out grid
    a training run's own eval loop would have. ``train=False`` walks a fixed
    stride grid (``SequenceSampler._starts``) with no draw from the seeded
    generator at all, so the ``+ 1`` offset is cosmetic parity with the
    trainer's construction, not a source of nondeterminism here — determinism
    is structural, not seed-dependent.

    Raises:
        ValueError: ``ckpt_a``/``ckpt_b`` were trained on different data
            (``DataConfig`` mismatch) or different tokenizers/vocabularies —
            either would make "KL divergence between their output
            distributions" meaningless (mismatched supports), not merely
            unfair.
    """
    cfg_a, cfg_b = ckpt_a.config, ckpt_b.config
    if cfg_a.data != cfg_b.data:
        msg = (
            "functional_divergence: ckpt_a and ckpt_b were built from different "
            f"DataConfig ({cfg_a.data!r} vs {cfg_b.data!r}) — their output distributions "
            "are not comparable over the same held-out positions"
        )
        raise ValueError(msg)
    if cfg_a.model.tokenizer != cfg_b.model.tokenizer:
        msg = (
            "functional_divergence: ckpt_a and ckpt_b use different tokenizers "
            f"({cfg_a.model.tokenizer!r} vs {cfg_b.model.tokenizer!r}) — mismatched output "
            "supports, KL/JS would compare probabilities over different vocabularies"
        )
        raise ValueError(msg)

    corpus = build_corpus(cfg_a.data, cfg_a.model.tokenizer)
    sampler = SequenceSampler(
        corpus.val,
        cfg_a.data.batch_size,
        cfg_a.data.seq_len,
        train=False,
        seed=cfg_a.train.seed + 1,
        device=device,
    )
    ckpt_a.model.eval()
    ckpt_b.model.eval()

    kl_ab_sum = 0.0
    kl_ba_sum = 0.0
    js_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, _y in sampler.iter_batches(num_batches):
            out_a = ckpt_a.model(x)
            out_b = ckpt_b.model(x)
            with torch.autocast(device_type=x.device.type, enabled=False):
                log_p = torch.log_softmax(out_a.logits.float(), dim=-1)
                log_q = torch.log_softmax(out_b.logits.float(), dim=-1)
                p = log_p.exp()
                q = log_q.exp()
                kl_ab = (p * (log_p - log_q)).sum(dim=-1)
                kl_ba = (q * (log_q - log_p)).sum(dim=-1)
                m = 0.5 * (p + q)
                log_m = m.clamp_min(1e-12).log()
                js = 0.5 * (p * (log_p - log_m)).sum(dim=-1) + 0.5 * (q * (log_q - log_m)).sum(
                    dim=-1
                )
            kl_ab_sum += float(kl_ab.sum())
            kl_ba_sum += float(kl_ba.sum())
            js_sum += float(js.sum())
            n += kl_ab.numel()

    if n == 0:
        msg = "functional_divergence: num_batches produced zero positions"
        raise ValueError(msg)
    return FunctionalDivergence(
        kl_a_to_b=kl_ab_sum / n, kl_b_to_a=kl_ba_sum / n, js=js_sum / n, n_positions=n
    )
