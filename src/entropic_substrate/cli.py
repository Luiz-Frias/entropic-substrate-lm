"""Command line entry point: ``python -m entropic_substrate.cli``.

Subcommands
-----------
``train``            run one rung of the experiment ladder
``init-fisher``      precompute and cache the sqrt(p) Fisher embedding table
``eval``             evaluate a checkpoint on its held-out split
``capture-results``  pull finished runs' results JSONLs into DVC
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from entropic_substrate.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.config import (
    EXPERIMENTS,
    Config,
    is_execution_gated,
    resolve_experiment_config,
)
from entropic_substrate.data.build import build_corpus
from entropic_substrate.data.enwik8 import load_enwik8_bytes
from entropic_substrate.data.fineweb import stream_fineweb_bytes
from entropic_substrate.data.synthetic import synthetic_text
from entropic_substrate.init.fisher import build_fisher_init, save_init
from entropic_substrate.logging import configure_logging, logger
from entropic_substrate.mlflow_aggregate import aggregate
from entropic_substrate.sinks import (
    MLFLOW_TINY_ENV,
    MLFLOW_TRACKING_ENV,
    MlflowStartupError,
    build_sink,
)
from entropic_substrate.telemetry import EXIT_TELEMETRY_GAP, TelemetryGap, read_final_kappa
from entropic_substrate.train import Trainer, resolve_device, resolve_results_path

__all__ = ["build_parser", "main"]

UNGATED_ENV: str = "ENTROPIC_ALLOW_UNGATED"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="entropic-substrate",
        description="Entropic substrate LM experiments (Form F(x) Labs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train one experiment")
    train.add_argument("--exp", choices=EXPERIMENTS, help="experiment ladder rung")
    train.add_argument("--config", type=Path, help="explicit YAML config path")
    train.add_argument("--device", default=None, help="cpu | cuda | cuda:N | mps | auto")
    train.add_argument(
        "--tiny",
        action="store_true",
        help=(
            "overlay configs/tiny.yaml. MLflow tracking (when MLFLOW_TRACKING_URI is "
            f"set) is OFF for --tiny runs unless {MLFLOW_TINY_ENV}=1 is also set."
        ),
    )
    train.add_argument("--steps", type=int, default=None, help="override max_steps")
    train.add_argument("--seed", type=int, default=None, help="override the seed")
    train.add_argument("--resume", type=Path, default=None, help="checkpoint to resume")
    train.add_argument("--results", type=Path, default=None, help="metrics JSONL path")
    train.add_argument("--quiet", action="store_true", help="suppress stdout logging")

    fisher = sub.add_parser("init-fisher", help="build the sqrt(p) Fisher init table")
    fisher.add_argument(
        "--corpus",
        default="data/enwik8/enwik8",
        help="path to a byte corpus, or 'fineweb' / 'synthetic'",
    )
    fisher.add_argument("--d-model", type=int, default=768)
    fisher.add_argument("--max-bytes", type=int, default=50_000_000)
    fisher.add_argument("--seed", type=int, default=0)
    fisher.add_argument("--smoothing", type=float, default=1.0)
    # Default matches the path the e1-fisher/e2/e3 configs read
    # (models are d_model=768 at full scale).
    fisher.add_argument("--out", type=Path, default=Path("data/fisher_init_d768.pt"))

    evaluate = sub.add_parser("eval", help="evaluate a checkpoint")
    evaluate.add_argument("--ckpt", type=Path, required=True)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--batches", type=int, default=None)
    evaluate.add_argument("--json", action="store_true", help="print JSON only")

    capture = sub.add_parser(
        "capture-results",
        help="pull finished runs' results JSONLs from the tracking server into DVC",
    )
    capture.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=None,
        help="experiment name (repeatable; defaults to $MLFLOW_EXPERIMENT_NAME)",
    )
    capture.add_argument("--dest", type=Path, default=Path("results/captured"))
    capture.add_argument(
        "--include-tiny", action="store_true", help="also capture -tiny smoke runs"
    )
    capture.add_argument(
        "--no-dvc", action="store_true", help="download + validate only; skip dvc add/push"
    )

    aggregate_p = sub.add_parser(
        "mlflow-aggregate", help="idempotently roll seed runs into a parent run by config_hash"
    )
    aggregate_p.add_argument(
        "--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "entropic-substrate-lm")
    )
    aggregate_p.add_argument(
        "--tracking-uri",
        default=None,
        help=f"defaults to ${MLFLOW_TRACKING_ENV}",
    )
    aggregate_p.add_argument(
        "--n-seeds", type=int, default=3, help="minimum children before a parent is created"
    )

    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    train: dict[str, Any] = {}
    if args.device is not None:
        train["device"] = args.device
    if args.steps is not None:
        train["max_steps"] = args.steps
    if args.seed is not None:
        train["seed"] = args.seed
    if args.resume is not None:
        train["resume"] = str(args.resume)
    return {"train": train} if train else {}


def _anchored_arm_preflight(cfg: Config, *, tiny: bool) -> int:
    """Refuse to launch an anchored arm without the E3 readout it is gated on.

    SANITY ONLY: this checks the readout *exists and is admissible*, and prints
    it. Whether the per-head divergence justifies the GPU hours stays a human
    go/no-go call by design (docs/e-3-instruction.md) — no threshold is encoded
    here. Admissibility itself lives in :mod:`entropic_substrate.telemetry`,
    shared with the ladder runner so the two cannot drift apart.
    """
    path = Path(cfg.train.results_dir) / ("e3-tiny.jsonl" if tiny else "e3.jsonl")
    try:
        readout = read_final_kappa(path)
    except TelemetryGap as gap:
        # Missing and malformed are the same premise — no valid readout to gate
        # on — so the override covers both, but must never report a NaN-poisoned
        # run as a merely absent one.
        if os.environ.get(UNGATED_ENV) == "1":
            logger.warning(
                f"no admissible E3 kappa readout ({gap}), but {UNGATED_ENV}=1 "
                f"is set — launching {cfg.name} on an ungated premise."
            )
            return 0
        logger.error(
            "\n=============================== DECISION-NEEDED ==============================="
            f"\n{cfg.name} is execution-gated on the E3 per-head kappa readout, and it is not"
            f"\nusable:\n  {gap}"
            "\nRun E3 first, then read the divergence and decide whether the anchored arms are"
            "\nworth the GPU hours — that call is yours, not this script's"
            "\n(docs/e-3-instruction.md)."
            f"\nDeliberate ungated run: {UNGATED_ENV}=1"
            "\n==============================================================================="
        )
        return EXIT_TELEMETRY_GAP
    logger.bind(report=True).info(
        f"[{cfg.name}] E3 per-head kappa divergence (the execution gate) from {path}:"
    )
    for line in readout.report_lines():
        logger.bind(report=True).info(line)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Handle ``train``."""
    if args.exp is None and args.config is None:
        logger.error("one of --exp or --config is required")
        return 2
    cfg = resolve_experiment_config(
        args.exp, args.config, tiny=args.tiny, overrides=_overrides(args)
    )
    # Gate on what RUNS, not on the flag that asked for it: --config leaves
    # --exp unset and outranks it when both are given.
    if is_execution_gated(cfg):
        code = _anchored_arm_preflight(cfg, tiny=args.tiny)
        if code != 0:
            return code

    results_path = resolve_results_path(cfg, args.results)
    # STARTUP ABORT, RUNTIME CONTINUE: the MLflow startup probe runs BEFORE
    # Trainer (corpus/model construction, GPU allocation) — an unreachable or
    # misconfigured tracking server must fail here, not after minutes of setup.
    try:
        sink = build_sink(cfg, tiny=args.tiny, results_path=results_path)
    except MlflowStartupError as exc:
        logger.error(f"mlflow startup abort: {exc}")
        return 2

    trainer = Trainer(cfg, results_path=results_path, sink=sink)
    logger.info(
        f"[{cfg.name}] {trainer.raw_model.num_parameters() / 1e6:.2f}M params on "
        f"{trainer.device}; tokenizer={trainer.meta.tokenizer} "
        f"bytes/token={trainer.meta.bytes_per_token:.4f}"
    )
    trainer.fit(verbose=not args.quiet)
    logger.info(f"metrics -> {trainer.results_path}")
    return 0


def cmd_init_fisher(args: argparse.Namespace) -> int:
    """Handle ``init-fisher``."""
    if args.corpus == "fineweb":
        stream: Any = stream_fineweb_bytes(args.max_bytes)
    elif args.corpus == "synthetic":
        stream = synthetic_text(min(args.max_bytes, 5_000_000)).encode("utf-8")
    else:
        stream = bytes(load_enwik8_bytes(args.corpus, args.max_bytes).tobytes())

    table = build_fisher_init(
        stream,
        args.d_model,
        max_bytes=args.max_bytes,
        smoothing=args.smoothing,
        seed=args.seed,
    )
    path = save_init(table, args.out)
    norms = table.norm(dim=1)
    logger.info(
        f"wrote {tuple(table.shape)} Fisher init -> {path} "
        f"(norm min={norms.min():.6f} max={norms.max():.6f})"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Handle ``eval``."""
    device = resolve_device(args.device or "auto")
    # Checkpoints are self-produced and hold only tensors and primitives, so the
    # secure default loads them fine (bandit B614).
    state = normalize_checkpoint(
        torch.load(args.ckpt, map_location=device, weights_only=True),
        source=str(args.ckpt),
    )
    from entropic_substrate.config import config_from_dict

    cfg: Config = config_from_dict(state["config"])
    corpus = build_corpus(cfg.data, cfg.model.tokenizer)
    trainer = Trainer(cfg, corpus=corpus, device=device)
    trainer.raw_model.load_state_dict(state["model"])
    trainer.step = int(state.get("step", 0))
    result = trainer.evaluate(args.batches)

    payload = {
        "checkpoint": str(args.ckpt),
        "run": cfg.name,
        "step": trainer.step,
        "loss": result.loss,
        "nats_per_byte": result.nats_per_byte,
        "bits_per_byte": result.bits_per_byte,
        "perplexity": result.perplexity,
        "bytes_per_token": trainer.meta.bytes_per_token,
        "kappa": result.kappa,
        "gate_usage": result.gate_fraction,
    }
    if args.json:
        # MACHINE-CONSUMED STDOUT, NOT LOGS: tests/test_train_smoke.py reads
        # this exact line back via json.loads(capsys...out.strip()...[-1]).
        # Stays raw print(), never routed through this module's sinks — see
        # entropic_substrate.logging's module docstring.
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            logger.bind(report=True).info(f"{key}: {value}")
    return 0


def cmd_capture_results(args: argparse.Namespace) -> int:
    """Handle ``capture-results``."""
    tracking_uri = os.environ.get(MLFLOW_TRACKING_ENV)
    if not tracking_uri:
        logger.error(f"capture-results: ${MLFLOW_TRACKING_ENV} is not set")
        return 2
    import subprocess  # nosec B404 -- dvc add/push only, fixed argv at the call site below

    import mlflow.tracking

    from entropic_substrate.capture_results import capture_finished_runs, dvc_commands

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiments = args.experiments or [
        os.environ.get("MLFLOW_EXPERIMENT_NAME", "entropic-substrate-lm")
    ]
    captured = capture_finished_runs(client, experiments, args.dest, include_tiny=args.include_tiny)
    if not captured:
        logger.info("capture-results: nothing new to capture")
        return 0
    if args.no_dvc:
        logger.info(f"capture-results: {len(captured)} file(s) captured; dvc skipped (--no-dvc)")
        return 0
    for command in dvc_commands(args.dest):
        # nosec B603 B607 -- argv is the fixed list from dvc_commands(), no
        # shell; "dvc" deliberately resolved from PATH, same idiom as sinks'
        # git_sha().
        subprocess.run(command, check=True)  # nosec B603 B607
    logger.info(f"capture-results: {len(captured)} file(s) captured and pushed via dvc")
    return 0


def cmd_mlflow_aggregate(args: argparse.Namespace) -> int:
    """Handle ``mlflow-aggregate``."""
    tracking_uri = args.tracking_uri or os.environ.get(MLFLOW_TRACKING_ENV)
    if not tracking_uri:
        logger.error(
            f"mlflow-aggregate: no tracking URI (--tracking-uri or ${MLFLOW_TRACKING_ENV})"
        )
        return 2
    import mlflow.tracking
    from mlflow.exceptions import MlflowException

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    try:
        summaries = aggregate(client, experiment_name=args.experiment, n_seeds=args.n_seeds)
    except MlflowException as exc:
        # A controlled refusal, not a raw traceback — same gate idiom as
        # CheckpointFormatError below: an operator-actionable server-side
        # rejection must read as "here's what happened", not as a stack
        # trace that looks like a defect in this tool.
        # mlflow_aggregate.py's own fix (n_seeds as a tag, not a write-once
        # param) makes the specific growth-crash this was written for
        # unreachable; this catch is the net for whatever the NEXT
        # MlflowException-raising defect in this path turns out to be.
        logger.error(f"mlflow-aggregate: aggregation rejected by the tracking server: {exc}")
        return 2
    if not summaries:
        logger.info("mlflow-aggregate: no complete seed group to aggregate")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch. Returns a process exit code."""
    configure_logging()
    args = build_parser().parse_args(argv)
    handlers = {
        "train": cmd_train,
        "init-fisher": cmd_init_fisher,
        "eval": cmd_eval,
        "capture-results": cmd_capture_results,
        "mlflow-aggregate": cmd_mlflow_aggregate,
    }
    try:
        return handlers[args.command](args)
    except CheckpointFormatError as exc:
        # An ambiguous or too-new checkpoint is an expected, operator-actionable
        # condition — the guidance is in the message. Present it the way every
        # other gate here does (printed refusal, exit 2), not as a traceback
        # that reads like a defect in the trainer.
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
