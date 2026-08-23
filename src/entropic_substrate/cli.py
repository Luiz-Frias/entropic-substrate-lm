"""Command line entry point: ``python -m entropic_substrate.cli``.

Subcommands
-----------
``train``            run one rung of the experiment ladder
``init-fisher``      precompute and cache the sqrt(p) Fisher embedding table
``eval``             evaluate a checkpoint on its held-out split
``capture-results``    pull finished runs' results JSONLs (or, with
                       ``--checkpoints``, final checkpoints) into DVC
``charts``             render E1/E2/E3 interpretation charts, attach to MLflow
``embedding-geometry`` E1.5-D angular-drift / Gram-retention / functional-
                       divergence diagnostics on captured checkpoints
``field-pretrain``     train a standalone p_phi and save the checkpoint
                       ``field_mode="frozen"`` loads (E1.5c Repair 1)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

if TYPE_CHECKING:
    from entropic_substrate.analysis.embedding_geometry import CheckpointModel

from entropic_substrate.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.config import (
    EXPERIMENTS,
    Config,
    is_execution_gated,
    merge_dicts,
    required_preregistrations,
    resolve_experiment_config,
)
from entropic_substrate.data.build import build_corpus
from entropic_substrate.data.enwik8 import load_enwik8_bytes
from entropic_substrate.data.fineweb import stream_fineweb_bytes
from entropic_substrate.data.synthetic import synthetic_text
from entropic_substrate.field_pretrain import add_field_pretrain_parser, cmd_field_pretrain
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


# YAML-1.1 float-resolver gap: dotless scientific notation ("3e-4").
_DOTLESS_SCI_FLOAT = re.compile(r"[-+]?\d+[eE][-+]?\d+")


def _override_kv(item: str) -> tuple[str, str, Any]:
    """argparse ``type`` for ``--override``: ``section.key=value`` → parsed triple.

    Malformed spellings fail INSIDE argparse (usage error, exit 2) before any
    config is resolved. The value goes through ``yaml.safe_load`` so the flag
    types values by the SAME rules the config files use — ``3e-4`` arrives as
    a float, ``frozen`` as a string, ``null`` as None — instead of smuggling
    every override in as a string for a dataclass field to choke on later.
    """
    key, eq, raw = item.partition("=")
    section, dot, field_name = key.partition(".")
    if not eq or not dot or not section or not field_name:
        msg = f"expected SECTION.KEY=VALUE (e.g. model.field_loss_lambda=0.3), got {item!r}"
        raise argparse.ArgumentTypeError(msg)
    value = yaml.safe_load(raw)
    if isinstance(value, str) and _DOTLESS_SCI_FLOAT.fullmatch(value):
        # PyYAML implements YAML 1.1, whose float resolver demands a dot
        # before the exponent — MEASURED 2026-08-20: safe_load("3e-4")
        # returns the STRING "3e-4", which then explodes inside AdamW far
        # from the flag that caused it. The registered LR-diagnostic grid
        # (docs/e1-5c-preregistration.md §Pilots) is written in exactly
        # this dotless form, so resolve it here, narrowly (a dotless
        # mantissa with an exponent cannot be an int or anything else).
        value = float(value)
    return section, field_name, value


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
    train.add_argument(
        "--override",
        action="append",
        default=[],
        type=_override_kv,
        metavar="SECTION.KEY=VALUE",
        help=(
            "override one resolved-config knob (repeatable), e.g. "
            "model.field_loss_lambda=0.3 or train.lr=3e-4 — the registered sweep "
            "surface (docs/e1-5b-preregistration.md, docs/e1-5c-preregistration.md "
            "pilots). Values parse as YAML (0.3 is a float, frozen a string, null "
            "is None); unknown keys are refused by the config loader. The dedicated "
            "--steps/--seed/--device/--resume flags win over --override."
        ),
    )

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
    capture.add_argument(
        "--checkpoints",
        action="store_true",
        help="capture each run's final checkpoint (not results JSONLs) — writes to --ckpt-dest",
    )
    capture.add_argument(
        "--ckpt-dest",
        type=Path,
        default=Path("models/captured"),
        help="capture destination when --checkpoints is given",
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

    charts = sub.add_parser(
        "charts",
        help="render E1/E2/E3 interpretation charts and attach them to MLflow",
    )
    charts.add_argument(
        "--e1-fisher",
        action="append",
        dest="e1_fisher",
        default=[],
        type=Path,
        help="E1 fisher-arm results JSONL (repeatable, one per seed)",
    )
    charts.add_argument(
        "--e1-control",
        action="append",
        dest="e1_control",
        default=[],
        type=Path,
        help="E1 control-arm results JSONL (repeatable, one per seed)",
    )
    charts.add_argument(
        "--e1-epsilon",
        type=float,
        default=None,
        help="practical-relevance threshold reported alongside the CI (NOT preregistered)",
    )
    charts.add_argument(
        "--config-hash",
        default=None,
        help="attach the E1 seed-band charts to the mlflow-aggregate parent run for this hash",
    )
    charts.add_argument("--e2", type=Path, default=None, help="E2 results JSONL (single run)")
    charts.add_argument("--e2-run-id", default=None, help="MLflow run_id to attach the E2 chart to")
    charts.add_argument("--e3", type=Path, default=None, help="E3 results JSONL (single run)")
    charts.add_argument(
        "--e3-run-id", default=None, help="MLflow run_id to attach the E3 charts to"
    )
    charts.add_argument("--out-dir", type=Path, default=Path("results/charts"))
    charts.add_argument(
        "--experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "entropic-substrate-lm")
    )
    charts.add_argument(
        "--tracking-uri",
        default=None,
        help=f"defaults to ${MLFLOW_TRACKING_ENV}; omit both to render locally only",
    )

    geometry = sub.add_parser(
        "embedding-geometry",
        help="E1.5-D angular-drift / Gram-retention / functional-divergence diagnostics",
    )
    geometry.add_argument(
        "--ckpt-a",
        action="append",
        dest="ckpt_a",
        default=[],
        type=Path,
        help="arm-A checkpoint path (repeatable, one per seed)",
    )
    geometry.add_argument(
        "--ckpt-b",
        action="append",
        dest="ckpt_b",
        default=[],
        type=Path,
        help="arm-B checkpoint path (repeatable, one per seed)",
    )
    geometry.add_argument("--out-dir", type=Path, default=Path("results/charts"))
    geometry.add_argument("--json", action="store_true", help="print JSON only (no charts written)")
    geometry.add_argument(
        "--no-functional", action="store_true", help="skip the functional-divergence pass"
    )
    geometry.add_argument(
        "--num-eval-batches", type=int, default=8, help="batches sampled for functional_divergence"
    )

    add_field_pretrain_parser(sub)

    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    generic: dict[str, Any] = {}
    for section, field_name, value in args.override:
        generic.setdefault(section, {})[field_name] = value
    train: dict[str, Any] = {}
    if args.device is not None:
        train["device"] = args.device
    if args.steps is not None:
        train["max_steps"] = args.steps
    if args.seed is not None:
        train["seed"] = args.seed
    if args.resume is not None:
        train["resume"] = str(args.resume)
    # Overlay order = precedence: the dedicated flags are the more specific
    # spelling of the same intent, so they win over --override (merge_dicts:
    # overlay beats base).
    return merge_dicts(generic, {"train": train}) if train else generic


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


def _e15_prereg_preflight(cfg: Config, *, tiny: bool) -> int:
    """Refuse an E1.5 arm whose axis has no committed preregistration (FOR-914).

    SANITY ONLY, same posture as :func:`_anchored_arm_preflight`: the check is
    "a registration is committed", read from ``config.E15_PREREGISTRATIONS``
    (the runtime half of the gate — the training image carries no ``docs/``,
    so the file itself is CI's half to pin, tests/test_execution_gate.py).
    ``--tiny`` runs pass ungated: the capture lane excludes ``-tiny`` runs by
    construction, so a tiny smoke cannot enter a confirmatory set.
    """
    if tiny:
        return 0
    axes = required_preregistrations(cfg)
    unregistered = [axis for axis, path in axes if path is None]
    if not unregistered:
        for axis, path in axes:
            logger.bind(report=True).info(
                f"[{cfg.name}] axis {axis!r} runs under committed registration {path}"
            )
        return 0
    if os.environ.get(UNGATED_ENV) == "1":
        logger.warning(
            f"E1.5 axes {unregistered} have no committed preregistration, but "
            f"{UNGATED_ENV}=1 is set — launching {cfg.name} on an ungated premise."
        )
        return 0
    listing = "\n".join(f"  {axis}" for axis in unregistered)
    logger.error(
        "\n=============================== DECISION-NEEDED ==============================="
        f"\n{cfg.name} activates E1.5 axes with no committed preregistration:\n{listing}"
        "\nThe rung-lifecycle flywheel (docs/rung-lifecycle-protocol.md) commits each"
        "\narm's registration BEFORE its first confirmatory run; register the axis in"
        "\nconfig.E15_PREREGISTRATIONS in the same commit as the preregistration document."
        f"\nDeliberate ungated run: {UNGATED_ENV}=1"
        "\n==============================================================================="
    )
    return 2


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
    code = _e15_prereg_preflight(cfg, tiny=args.tiny)
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
    """Handle ``capture-results``.

    ``--checkpoints`` is a MODE SWITCH, not additive: results JSONLs (small,
    ~KB each) and checkpoints (~340MB each, own ``--ckpt-dest``) are two
    differently-sized DVC-tracked directories, each with its own ``dvc
    add``/``dvc push`` pair — running both in one invocation would need two
    separate DVC command pairs anyway, so one flag choosing which capture
    runs keeps this command's ``dvc_commands(dest)`` call singular and
    unambiguous about which ``dest`` it is adding.
    """
    tracking_uri = os.environ.get(MLFLOW_TRACKING_ENV)
    if not tracking_uri:
        logger.error(f"capture-results: ${MLFLOW_TRACKING_ENV} is not set")
        return 2
    import subprocess  # nosec B404 -- dvc add/push only, fixed argv at the call site below

    import mlflow.tracking

    from entropic_substrate.capture_results import (
        capture_checkpoints,
        capture_finished_runs,
        dvc_commands,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiments = args.experiments or [
        os.environ.get("MLFLOW_EXPERIMENT_NAME", "entropic-substrate-lm")
    ]
    try:
        if args.checkpoints:
            dest, label = args.ckpt_dest, "checkpoint(s)"
            captured = capture_checkpoints(
                client, experiments, dest, include_tiny=args.include_tiny
            )
        else:
            dest, label = args.dest, "file(s)"
            captured = capture_finished_runs(
                client, experiments, dest, include_tiny=args.include_tiny
            )
    except LookupError as exc:
        # An unknown experiment name (capture_finished_runs/
        # capture_checkpoints' shared _resolve_experiment_ids) is an
        # operator typo, not a defect — same controlled-refusal idiom as
        # the missing-tracking-URI check above, not an unhandled traceback.
        logger.error(f"capture-results: {exc}")
        return 2
    if not captured:
        logger.info("capture-results: nothing new to capture")
        return 0
    if args.no_dvc:
        logger.info(f"capture-results: {len(captured)} {label} captured; dvc skipped (--no-dvc)")
        return 0
    for command in dvc_commands(dest):
        # nosec B603 B607 -- argv is the fixed list from dvc_commands(), no
        # shell; "dvc" deliberately resolved from PATH, same idiom as sinks'
        # git_sha().
        subprocess.run(command, check=True)  # nosec B603 B607
    logger.info(f"capture-results: {len(captured)} {label} captured and pushed via dvc")
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


# entropic_substrate.sinks.config_hash() is always
# hashlib.sha256(...).hexdigest()[:12] — 12 lowercase hex chars, exactly.
# Every OTHER --config-hash-shaped value this repo produces (mlflow-aggregate
# groups runs by reading this same tag back off already-tagged runs) comes
# from that one function, so this is the true contract, not a guess.
#
# ``--config-hash`` here is the first place that contract meets RAW,
# operator-typed input rather than a value the code computed for itself —
# mlflow-aggregate never took it as a CLI flag before this subcommand. A
# typo'd value used to fall straight into attach_seed_band_to_aggregate's
# best-effort ``except Exception`` net and get logged as "no aggregate
# parent yet" — indistinguishable from the legitimate case where
# mlflow-aggregate genuinely has not completed. Validating the SHAPE here,
# before that best-effort path ever sees it, turns a caller mistake into a
# controlled refusal instead of a silent no-op (repo NON-NEGOTIABLE: "STOP
# and ask > silent fallback").
_CONFIG_HASH_RE = re.compile(r"^[0-9a-f]{12}$")

_E1_METRIC = "nats_per_byte"
# Which --e2-run-id / --e3-run-id a per-run chart attaches to. E1's two charts
# are multi-seed by construction and attach elsewhere (the aggregate parent
# run per --config-hash, via attach_seed_band_to_aggregate) — they are
# deliberately absent from this table.
_RUN_ID_ARG_FOR_CHART: dict[str, str] = {
    "e2_spike_variance_texture": "e2_run_id",
    "e3_loss_separation_vs_e2": "e3_run_id",
    "e3_kappa_trajectories": "e3_run_id",
    "e3_kappa_divergence_vs_loss_gap": "e3_run_id",
}


def cmd_charts(args: argparse.Namespace) -> int:
    """Handle ``charts``: render, then best-effort attach to MLflow.

    Lazy-import idiom (mirrors ``cmd_capture_results``): matplotlib/mlflow
    stay out of every other subcommand's import graph. Rendering always
    happens first and always succeeds independent of MLflow (SACRED
    INVARIANTS: local rendering never depends on the tracking server) —
    only the attach step below is gated on a reachable tracking URI.
    """
    from entropic_substrate.analysis import charts as chart_fns
    from entropic_substrate.analysis.series import (
        load_eval_run,
        load_train_run,
        pair_final_metrics,
        paired_verdict,
    )

    figures: dict[str, Any] = {}
    try:
        if args.e1_fisher and args.e1_control:
            fisher_runs = [load_eval_run(p, label=f"fisher-{p.stem}") for p in args.e1_fisher]
            control_runs = [load_eval_run(p, label=f"control-{p.stem}") for p in args.e1_control]
            pairs, excluded = pair_final_metrics(args.e1_fisher, args.e1_control, metric=_E1_METRIC)
            if excluded:
                logger.warning(
                    f"charts: E1 seeds present on only one side — excluded from the "
                    f"verdict, reported not silently dropped: {excluded}"
                )
            verdict = paired_verdict(pairs)
            figures["e1_paired_loss_curves"] = chart_fns.fig_e1_paired_loss_curves(
                fisher_runs, control_runs, verdict, epsilon=args.e1_epsilon
            )
            figures["e1_log_steps_washout"] = chart_fns.fig_e1_log_steps_washout(
                fisher_runs, control_runs
            )
        elif args.e1_fisher or args.e1_control:
            logger.error("charts: --e1-fisher and --e1-control must both be given to pair E1")
            return 2

        e2_eval_run = load_eval_run(args.e2, label="e2") if args.e2 else None
        e3_eval_run = load_eval_run(args.e3, label="e3") if args.e3 else None
        if args.e2:
            e2_train_run = load_train_run(args.e2, label="e2")
            figures["e2_spike_variance_texture"] = chart_fns.fig_e2_spike_variance_texture(
                [e2_train_run]
            )
        if e2_eval_run is not None and e3_eval_run is not None:
            figures["e3_loss_separation_vs_e2"] = chart_fns.fig_e3_loss_separation_vs_e2(
                e2_eval_run, e3_eval_run
            )
            figures["e3_kappa_divergence_vs_loss_gap"] = (
                chart_fns.fig_e3_kappa_divergence_vs_loss_gap(e3_eval_run, e2_eval_run)
            )
        if e3_eval_run is not None:
            figures["e3_kappa_trajectories"] = chart_fns.fig_e3_kappa_trajectories(e3_eval_run)
    except TelemetryGap as gap:
        # Measurement path: fail LOUD, never a blank or partial chart from bad
        # telemetry — same controlled-refusal shape as the anchored-arm
        # preflight (entropic_substrate.telemetry.EXIT_TELEMETRY_GAP).
        logger.error(f"charts: telemetry gap: {gap}")
        return EXIT_TELEMETRY_GAP
    except ValueError as exc:
        # A chart-builder / series-loader input-validation refusal (e.g.
        # pair_final_metrics's same-arm duplicate-seed check) — a caller
        # mistake, not a defect in this command. MEASURED gap this closes:
        # main() catches no bare ValueError, so this used to reach the
        # operator as an unhandled traceback instead of the controlled
        # exit-2 idiom every OTHER input-validation refusal here uses (see
        # tests/test_cli_charts.py::
        # test_charts_with_a_duplicate_seed_within_one_e1_arm_is_a_controlled_refusal).
        logger.error(f"charts: {exc}")
        return 2

    if not figures:
        logger.error(
            "charts: no chart inputs given (need --e1-fisher/--e1-control, --e2, and/or --e3)"
        )
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        out_path = args.out_dir / f"{name}.png"
        figure.savefig(out_path, dpi=150)
        logger.info(f"charts: wrote {out_path}")

    tracking_uri = args.tracking_uri or os.environ.get(MLFLOW_TRACKING_ENV)
    if not tracking_uri:
        logger.info("charts: no tracking URI — rendered locally only, nothing attached")
        return 0

    import mlflow.tracking

    from entropic_substrate.analysis.mlflow_attach import (
        attach_figures_to_run,
        attach_seed_band_to_aggregate,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    by_run_id: dict[str, dict[str, Any]] = {}
    for name, figure in figures.items():
        run_id_arg = _RUN_ID_ARG_FOR_CHART.get(name)
        run_id = getattr(args, run_id_arg, None) if run_id_arg else None
        if run_id:
            by_run_id.setdefault(run_id, {})[name] = figure
    for run_id, run_figures in by_run_id.items():
        attach_figures_to_run(client, run_id, run_figures)

    seed_band_figures = {name: fig for name, fig in figures.items() if name.startswith("e1_")}
    if seed_band_figures and args.config_hash:
        if not _CONFIG_HASH_RE.fullmatch(args.config_hash):
            # fullmatch, not match: re.match's trailing "$" anchor matches
            # end-of-string OR just before a SINGLE trailing "\n" — MEASURED
            # regression, see tests/test_cli_charts.py::
            # test_charts_with_a_trailing_newline_on_config_hash_is_a_controlled_refusal
            # (a value like "deadbeefcafe\n" — an unstripped $(cat file) or
            # an env var with a trailing newline — satisfied .match() despite
            # not being a clean 12-char hex digest).
            # A shape check, not an existence check: this fires on a
            # malformed value BEFORE the best-effort MLflow path ever sees
            # it, so it cannot be confused with "no aggregate parent yet" —
            # that legitimate case is attach_seed_band_to_aggregate's to
            # report; this is cmd_charts's own controlled refusal of a
            # caller mistake. Local PNGs are already written by this point
            # and are not withheld — this refusal is scoped to the attach
            # step only, and only fires when the seed-band attach would
            # otherwise actually be attempted (an unused/absent
            # --config-hash is never validated — nothing would consume it).
            logger.error(
                f"charts: --config-hash {args.config_hash!r} is not a 12-char lowercase "
                "hex digest (entropic_substrate.sinks.config_hash()'s exact shape) — "
                "refusing to guess which aggregate parent this was meant to reach"
            )
            return 2
        attach_seed_band_to_aggregate(
            client,
            experiment_name=args.experiment,
            config_hash=args.config_hash,
            figures=seed_band_figures,
        )
    return 0


def _load_ckpts_by_seed(paths: list[Path], *, label: str) -> dict[int, CheckpointModel]:
    """Load each checkpoint in ``paths`` and key it by ``config.train.seed``.

    Refuses (ValueError, a controlled caller mistake — see ``cmd_embedding_
    geometry``'s own catch) two checkpoints on the SAME side sharing a seed:
    a silent dict-comprehension clobber there would pick whichever path
    happened to load last with no signal to the operator that the other one
    was ever discarded.
    """
    from entropic_substrate.analysis.embedding_geometry import load_checkpoint_model

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


def _init_table_cache_key(cfg: Config) -> tuple[object, ...]:
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


def _cached_reconstruct_init_table(
    cache: dict[tuple[object, ...], torch.Tensor], cfg: Config
) -> torch.Tensor:
    """Memoized ``reconstruct_init_table``, scoped to the ``cache`` dict the
    caller owns (FOR-920).

    ``cmd_embedding_geometry`` creates a FRESH ``cache`` per invocation —
    never a module-level or disk cache — so nothing here leaks reconstructed
    tables across separate CLI runs. A multi-seed sweep pairs N seeds of the
    SAME arm; every one of those seeds' configs shares an identical
    :func:`_init_table_cache_key` (``train.seed`` is excluded by
    construction), so without this memoization every seed rebuilt the
    IDENTICAL table redundantly — measured 2026-08-19: 9 fisher-arm rebuilds
    x ~7.5 min each (the 20MB corpus-stream fallback in
    ``reconstruct_init_table``) = ~68 min duplicated work in one invocation.

    Returns a CLONE on every call, including the first (cache miss): two
    callers holding "the same" cached table must never be able to corrupt
    each other's view by mutating their own copy in place.
    """
    from entropic_substrate.analysis import embedding_geometry as eg

    key = _init_table_cache_key(cfg)
    if key not in cache:
        cache[key] = eg.reconstruct_init_table(cfg).clone()
    return cache[key].clone()


def cmd_embedding_geometry(args: argparse.Namespace) -> int:
    """Handle ``embedding-geometry``: E1.5-D diagnostics on captured checkpoints.

    ``--ckpt-a``/``--ckpt-b`` are repeatable, seed-paired lists — the same
    shape ``charts``'s ``--e1-fisher``/``--e1-control`` already uses. A seed
    present on only one side is reported (never silently dropped) and
    excluded from every pairwise diagnostic; zero paired seeds is a
    controlled refusal (exit 2), same idiom as ``charts``'s "no chart
    inputs" refusal.
    """
    if not args.ckpt_a or not args.ckpt_b:
        logger.error("embedding-geometry: --ckpt-a and --ckpt-b must both be given (repeatable)")
        return 2

    from entropic_substrate.analysis import embedding_geometry as eg

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
            return 2

        per_seed_raw: list[dict[str, Any]] = []
        # In-memory, per-invocation only (FOR-920) — a fresh dict created
        # right here, never module-level or disk-backed, so nothing leaks
        # across separate CLI runs. See _cached_reconstruct_init_table's
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
            # seed for nothing. _cached_reconstruct_init_table memoizes on
            # _init_table_cache_key within this one invocation's
            # init_table_cache — never across invocations.
            init_a = _cached_reconstruct_init_table(init_table_cache, ckpt_a.config)
            init_b = _cached_reconstruct_init_table(init_table_cache, ckpt_b.config)
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
        return 2

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
    if args.json:
        # MACHINE-CONSUMED STDOUT, NOT LOGS — same idiom as cmd_eval's --json.
        print(json.dumps(payload, sort_keys=True))
        return 0

    for entry in results_payload:
        logger.bind(report=True).info(f"seed {entry['seed']}: {entry}")

    from entropic_substrate.analysis import charts as chart_fns

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drift_series = [(f"seed{raw['seed']}-a", raw["drift_a"]) for raw in per_seed_raw] + [
        (f"seed{raw['seed']}-b", raw["drift_b"]) for raw in per_seed_raw
    ]
    retention_points = [(f"seed{raw['seed']}-a", raw["retention_a"]) for raw in per_seed_raw] + [
        (f"seed{raw['seed']}-b", raw["retention_b"]) for raw in per_seed_raw
    ]
    figures = {
        "fig_e15d_angular_drift": chart_fns.fig_e15d_angular_drift(drift_series),
        "fig_e15d_gram_retention": chart_fns.fig_e15d_gram_retention(retention_points),
    }
    for name, figure in figures.items():
        out_path = args.out_dir / f"{name}.png"
        figure.savefig(out_path, dpi=150)
        logger.info(f"embedding-geometry: wrote {out_path}")
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
        "charts": cmd_charts,
        "embedding-geometry": cmd_embedding_geometry,
        "field-pretrain": cmd_field_pretrain,
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
