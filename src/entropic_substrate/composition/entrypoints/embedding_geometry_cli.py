"""The ``embedding-geometry`` handler family of the entry-point CLI (FOR-920; moved here by FOR-1608).

``entrypoints.cli`` keeps the thin ``cmd_embedding_geometry`` shell that owns the one
machine-consumed ``print()`` (``--json``); everything else lives here. This module holds no
``print()`` (the print ban in tests/test_observability_conformance.py allowlists only named
entry-point modules), so the computation returns the payload instead of printing it.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from entropic_substrate.kernel.runtime.config import Config
from entropic_substrate.kernel.runtime.logging import logger

if TYPE_CHECKING:
    from entropic_substrate.ladder.analyze.embedding_geometry import CheckpointModel

__all__ = [
    "EmbeddingGeometryRun",
    "cached_reconstruct_init_table",
    "compute_embedding_geometry",
    "init_table_cache_key",
    "write_embedding_geometry_report",
]


@dataclass(frozen=True)
class EmbeddingGeometryRun:
    """One successful diagnostics run: the JSON payload plus the raw per-seed arrays the figures need."""

    payload: dict[str, Any]
    per_seed_raw: list[dict[str, Any]]


def _load_ckpts_by_seed(paths: list[Path], *, label: str) -> dict[int, CheckpointModel]:
    """Load each checkpoint in ``paths`` and key it by ``config.train.seed``.

    Refuses (ValueError, a controlled caller mistake — see ``compute_embedding_
    geometry``'s own catch) two checkpoints on the SAME side sharing a seed:
    a silent dict-comprehension clobber there would pick whichever path
    happened to load last with no signal to the operator that the other one
    was ever discarded.
    """
    from entropic_substrate.ladder.analyze.embedding_geometry import load_checkpoint_model

    loaded: dict[int, CheckpointModel] = {}
    for path in paths:
        ckpt = load_checkpoint_model(path)
        seed = ckpt.config.train.seed
        if seed in loaded:
            msg = (
                f"embedding-geometry: {label} has two checkpoints at seed {seed} "
                f"({loaded[seed].path} and {path}) — ambiguous pairing"
            )
            raise ValueError(msg)
        loaded[seed] = ckpt
    return loaded


def init_table_cache_key(cfg: Config) -> tuple[object, ...]:
    """Hashable identity for the subset of ``cfg`` that determines
    ``reconstruct_init_table``'s output (FOR-920).

    Two configs sharing this key are guaranteed to reconstruct to the SAME
    table: ``embedding_init``/``d_model``/``tokenizer``/``vocab_size`` (the
    ``vocab_size_for`` inputs the ``random_sphere`` arm draws against —
    seed-0-shared by construction, per ``reconstruct_init_table``'s own
    docstring), ``fisher_init_path`` (``None`` selects the corpus-rebuild
    fallback below), and the FULL ``DataConfig`` identity — the fallback
    reads ``corpus_byte_stream(cfg.data, ...)``, so any data-config field
    could in principle change the rebuilt corpus stream; including the
    whole tuple (``DataConfig`` is small and all-primitive) costs nothing
    and avoids a class of "the fallback grew a new data-dependent knob and
    nobody widened the cache key" bugs.

    ``cfg.train`` is deliberately EXCLUDED: nothing in
    ``reconstruct_init_table`` reads it — the random-arm seed-0-sharing pin
    (this module's own finding #2, pinned in
    ``test_reconstruct_init_table_random_sphere_shares_seed_0_across_
    train_seeds``) is exactly the guarantee that ``cfg.train.seed`` must
    NOT appear here, since two seeds of the same arm are the common case
    this cache exists to collapse.
    """
    model_cfg = cfg.model
    return (
        model_cfg.embedding_init,
        model_cfg.d_model,
        model_cfg.tokenizer,
        model_cfg.vocab_size,
        model_cfg.fisher_init_path,
        dataclasses.astuple(cfg.data),
    )


def cached_reconstruct_init_table(
    cache: dict[tuple[object, ...], torch.Tensor], cfg: Config
) -> torch.Tensor:
    """Memoized ``reconstruct_init_table``, scoped to the ``cache`` dict the
    caller owns (FOR-920).

    ``compute_embedding_geometry`` creates a FRESH ``cache`` per invocation —
    never a module-level or disk cache — so nothing here leaks reconstructed
    tables across separate CLI runs. A multi-seed sweep pairs N seeds of the
    SAME arm; every one of those seeds' configs shares an identical
    :func:`init_table_cache_key` (``train.seed`` is excluded by
    construction), so without this memoization every seed rebuilt the
    IDENTICAL table redundantly — measured 2026-08-19: 9 fisher-arm rebuilds
    x ~7.5 min each (the 20MB corpus-stream fallback in
    ``reconstruct_init_table``) = ~68 min duplicated work in one invocation.

    Returns a CLONE on every call, including the first (cache miss): two
    callers holding "the same" cached table must never be able to corrupt
    each other's view by mutating their own copy in place.
    """
    from entropic_substrate.ladder.analyze import embedding_geometry as eg

    key = init_table_cache_key(cfg)
    if key not in cache:
        cache[key] = eg.reconstruct_init_table(cfg).clone()
    return cache[key].clone()


def compute_embedding_geometry(args: argparse.Namespace) -> EmbeddingGeometryRun | None:
    """Run the ``embedding-geometry`` diagnostics: E1.5-D on captured checkpoints.

    Returns ``None`` after logging the reason on a controlled refusal (the caller exits 2).

    ``--ckpt-a``/``--ckpt-b`` are repeatable, seed-paired lists — the same
    shape ``charts``'s ``--e1-fisher``/``--e1-control`` already uses. A seed
    present on only one side is reported (never silently dropped) and
    excluded from every pairwise diagnostic; zero paired seeds is a
    controlled refusal (exit 2), same idiom as ``charts``'s "no chart
    inputs" refusal.
    """
    if not args.ckpt_a or not args.ckpt_b:
        logger.error("embedding-geometry: --ckpt-a and --ckpt-b must both be given (repeatable)")
        return None

    from entropic_substrate.ladder.analyze import embedding_geometry as eg

    try:
        loaded_a = _load_ckpts_by_seed(args.ckpt_a, label="--ckpt-a")
        loaded_b = _load_ckpts_by_seed(args.ckpt_b, label="--ckpt-b")
        seeds_a, seeds_b = set(loaded_a), set(loaded_b)
        common = sorted(seeds_a & seeds_b)
        unpaired = sorted(seeds_a ^ seeds_b)
        if unpaired:
            logger.warning(
                "embedding-geometry: seed(s) present on only one side — excluded from "
                f"pairing, reported not silently dropped: {unpaired}"
            )
        if not common:
            logger.error(
                "embedding-geometry: no seed-paired checkpoints between --ckpt-a and --ckpt-b"
            )
            return None

        per_seed_raw: list[dict[str, Any]] = []
        # In-memory, per-invocation only (FOR-920) — a fresh dict created
        # right here, never module-level or disk-backed, so nothing leaks
        # across separate CLI runs. See cached_reconstruct_init_table's
        # own docstring for why this exists.
        init_table_cache: dict[tuple[object, ...], torch.Tensor] = {}
        for seed in common:
            ckpt_a, ckpt_b = loaded_a[seed], loaded_b[seed]
            # L2 finding, 2026-08-19: under state_source="field" (E1.5c),
            # ckpt.model.wte.weight is only the output classifier — the
            # real input representation is FieldEncoder.wte
            # (field_encoder.py), never read by wte-based diagnostics
            # below. Mirrors backbone.py's own gram_lambda (~109-131) and
            # embedding_freeze (~223-237) guards for this exact axis:
            # fail loud here rather than silently emit a plausible-looking
            # but wrong diagnostic (a paired fisher-shaped field checkpoint
            # passes shape validation, so nothing else would catch this).
            for label, ckpt in (("--ckpt-a", ckpt_a), ("--ckpt-b", ckpt_b)):
                state_source = ckpt.config.model.state_source
                if state_source != "table":
                    msg = (
                        f"embedding-geometry: {label} seed {seed} has "
                        f"state_source={state_source!r} — the wte.weight-based "
                        "diagnostics (angular_drift, gram_retention, "
                        "cross_arm_gram_agreement, procrustes_align) require "
                        "state_source='table': under state_source='field' "
                        "self.wte.weight is not the input representation "
                        "(FieldEncoder.wte is), so these diagnostics would "
                        "silently measure the wrong tensor"
                    )
                    raise ValueError(msg)
            final_a = ckpt_a.model.wte.weight.detach()
            final_b = ckpt_b.model.wte.weight.detach()
            # Reconstruct once per arm PER DISTINCT CONFIG — not once per
            # diagnostic (angular_drift and gram_retention both want the
            # SAME table) and not once per SEED either (FOR-920): a
            # multi-seed sweep pairs N seeds of the same arm, whose configs
            # share an identical init-table identity regardless of
            # train.seed, so the fisher_sqrt/no-saved-path fallback's 20MB
            # corpus-stream rebuild (reconstruct_init_table's own
            # docstring) would otherwise redo the SAME rebuild once per
            # seed for nothing. cached_reconstruct_init_table memoizes on
            # init_table_cache_key within this one invocation's
            # init_table_cache — never across invocations.
            init_a = cached_reconstruct_init_table(init_table_cache, ckpt_a.config)
            init_b = cached_reconstruct_init_table(init_table_cache, ckpt_b.config)
            drift_a = eg.angular_drift(init_a, final_a)
            drift_b = eg.angular_drift(init_b, final_b)
            retention_a = eg.gram_retention(init_a, final_a)
            retention_b = eg.gram_retention(init_b, final_b)
            entry: dict[str, Any] = {
                "seed": seed,
                "drift_a": drift_a,
                "drift_b": drift_b,
                "retention_a": retention_a,
                "retention_b": retention_b,
                "agreement": eg.cross_arm_gram_agreement(final_a, final_b),
                "procrustes": eg.procrustes_align(final_a, final_b),
                "functional_divergence": (
                    None
                    if args.no_functional
                    else eg.functional_divergence(ckpt_a, ckpt_b, num_batches=args.num_eval_batches)
                ),
            }
            per_seed_raw.append(entry)
    except ValueError as exc:
        # A diagnostic-function input-validation refusal (mismatched shapes,
        # embedding_init="default" with no init table to reconstruct,
        # DataConfig/tokenizer mismatch in functional_divergence, or the
        # duplicate-seed guard above) — a caller mistake, not a defect in
        # this tool. Same idiom as cmd_charts's own bare-ValueError catch.
        logger.error(f"embedding-geometry: {exc}")
        return None

    def _arm_summary(drift: Any, retention: Any) -> dict[str, Any]:
        return {
            "angular_drift_mean": float(drift.mean()),
            "angular_drift_max": float(drift.max()),
            "gram_retention_pearson": retention.pearson_r,
            "gram_retention_spearman": retention.spearman_r,
        }

    results_payload: list[dict[str, Any]] = []
    for raw in per_seed_raw:
        payload_entry: dict[str, Any] = {
            "seed": raw["seed"],
            "arm_a": _arm_summary(raw["drift_a"], raw["retention_a"]),
            "arm_b": _arm_summary(raw["drift_b"], raw["retention_b"]),
            "cross_arm_gram_agreement": raw["agreement"],
            "procrustes_residual": raw["procrustes"].residual,
        }
        fd = raw["functional_divergence"]
        if fd is not None:
            payload_entry["functional_divergence"] = {
                "kl_a_to_b": fd.kl_a_to_b,
                "kl_b_to_a": fd.kl_b_to_a,
                "js": fd.js,
                "n_positions": fd.n_positions,
            }
        results_payload.append(payload_entry)

    payload = {"paired_seeds": common, "unpaired_seeds": unpaired, "results": results_payload}
    return EmbeddingGeometryRun(payload=payload, per_seed_raw=per_seed_raw)


def write_embedding_geometry_report(args: argparse.Namespace, run: EmbeddingGeometryRun) -> None:
    """Log each seed's summary and write the two E1.5-D figures under ``args.out_dir``."""
    for entry in run.payload["results"]:
        logger.bind(report=True).info(f"seed {entry['seed']}: {entry}")

    from entropic_substrate.ladder.analyze import charts as chart_fns

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drift_series = [(f"seed{raw['seed']}-a", raw["drift_a"]) for raw in run.per_seed_raw] + [
        (f"seed{raw['seed']}-b", raw["drift_b"]) for raw in run.per_seed_raw
    ]
    retention_points = [
        (f"seed{raw['seed']}-a", raw["retention_a"]) for raw in run.per_seed_raw
    ] + [(f"seed{raw['seed']}-b", raw["retention_b"]) for raw in run.per_seed_raw]
    figures = {
        "fig_e15d_angular_drift": chart_fns.fig_e15d_angular_drift(drift_series),
        "fig_e15d_gram_retention": chart_fns.fig_e15d_gram_retention(retention_points),
    }
    for name, figure in figures.items():
        out_path = args.out_dir / f"{name}.png"
        figure.savefig(out_path, dpi=150)
        logger.info(f"embedding-geometry: wrote {out_path}")
