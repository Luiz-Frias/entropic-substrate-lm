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
``d1-*``               dispatch one strict D1 application manifest
``d-probe``            run the D-ladder probe (handlers table)
``d-score-gain``       score the D-ladder probe gain (handlers table)
``h-*``                the H sentence, capture, span and source families
                       (``h-train-probe``, ``h-inspect-training``,
                       ``h-generate-probe``, ``h-inspect-generation``,
                       ``h-capture-model``, ``h-inspect-model``,
                       ``h-inspect-custody``, ``h-diagnose-spans``,
                       ``h-inspect-spans``, ``h-prepare-source``,
                       ``h-inspect-source``), routed through the handlers table
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

if TYPE_CHECKING:
    from entropic_substrate.composition.wiring.d1_cli import D1CliApplication

from entropic_substrate.composition.entrypoints.embedding_geometry_cli import (
    cached_reconstruct_init_table,
    compute_embedding_geometry,
    init_table_cache_key,
    write_embedding_geometry_report,
)
from entropic_substrate.composition.entrypoints.h_analysis_cli import (
    H_LANE,
    add_h_analysis_lanes,
    cmd_aggregate_h,
    cmd_charts_h,
    cmd_eval_h,
)
from entropic_substrate.core.substrate.fisher import build_fisher_init, save_init
from entropic_substrate.historical.e_series.field_pretrain import (
    add_field_pretrain_parser,
    cmd_field_pretrain,
)
from entropic_substrate.historical.e_series.train import (
    Trainer,
    resolve_device,
    resolve_results_path,
)
from entropic_substrate.kernel.runtime.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.kernel.runtime.config import (
    EXPERIMENTS,
    Config,
    is_execution_gated,
    merge_dicts,
    required_preregistrations,
    resolve_experiment_config,
)
from entropic_substrate.kernel.runtime.logging import configure_logging, logger
from entropic_substrate.ladder.capture.custody.build import build_corpus
from entropic_substrate.ladder.capture.custody.enwik8 import load_enwik8_bytes
from entropic_substrate.ladder.capture.custody.fineweb import stream_fineweb_bytes
from entropic_substrate.ladder.capture.custody.synthetic import synthetic_text
from entropic_substrate.ladder.capture.reader.mlflow_aggregate import aggregate
from entropic_substrate.ladder.capture.reader.sinks import (
    MLFLOW_TINY_ENV,
    MLFLOW_TRACKING_ENV,
    MlflowStartupError,
    build_sink,
)
from entropic_substrate.ladder.capture.reader.telemetry import (
    EXIT_TELEMETRY_GAP,
    TelemetryGap,
    read_final_kappa,
)

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
    # DN-1648-B / K2: --ckpt is no longer argparse-required, because a required option
    # of the host parser would make `eval h ...` unreachable. cmd_eval refuses a bare
    # `eval` (neither --ckpt nor `h`) itself, with exit 2, before any checkpoint I/O.
    evaluate.add_argument("--ckpt", type=Path, default=None)
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

    add_h_analysis_lanes(evaluate, aggregate_p, charts)

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

    from entropic_substrate.composition.wiring.d_ladder_cli import add_d_ladder_parser

    add_d_ladder_parser(sub)

    from entropic_substrate.composition.wiring.d_ladder_probe_cli import add_score_gain_parser

    add_score_gain_parser(sub)

    from entropic_substrate.composition.cli import add_sentence_parsers

    add_sentence_parsers(sub)

    from entropic_substrate.composition.capture_cli import add_capture_parsers

    add_capture_parsers(sub)

    from entropic_substrate.composition.span_cli import add_span_parsers

    add_span_parsers(sub)

    from entropic_substrate.composition.source_cli import add_source_parsers

    add_source_parsers(sub)

    from entropic_substrate.composition.wiring.d1_cli import D1CliCommand

    for command in D1CliCommand:
        d1 = sub.add_parser(
            command.value,
            help=f"execute the {command.value} application manifest",
        )
        d1.add_argument(
            "--manifest",
            type=Path,
            required=True,
            help="strict d1-cli-command-v1 JSON manifest",
        )

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
    if getattr(args, "eval_lane", None) == H_LANE:
        return cmd_eval_h(args, _emit_h_document)
    if args.ckpt is None:
        # K2: the compensation for --ckpt no longer being argparse-required. Same exit
        # code (2) as argparse's own refusal; only the text and mechanism differ.
        logger.error("eval: --ckpt is required (or use the `h` lane: eval h --run ...)")
        return 2
    device = resolve_device(args.device or "auto")
    # Checkpoints are self-produced and hold only tensors and primitives, so the
    # secure default loads them fine (bandit B614).
    state = normalize_checkpoint(
        torch.load(args.ckpt, map_location=device, weights_only=True),
        source=str(args.ckpt),
    )
    from entropic_substrate.kernel.runtime.config import config_from_dict

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

    import entropic_substrate.ladder.capture.reader.capture_results as capture_results_mod
    from entropic_substrate.ladder.capture.adapters.mlflow import build_mlflow_client
    from entropic_substrate.ladder.capture.reader.capture_results import (
        capture_checkpoints,
        capture_finished_runs,
        dvc_commands,
    )

    client = build_mlflow_client(tracking_uri)
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
    except TelemetryGap as gap:
        # An inadmissible results file (either lane's predicate): the file is
        # kept on disk as evidence and named in the message. Controlled
        # refusal, never a traceback; nothing is captured silently.
        logger.error(f"capture-results: {gap}")
        return 2
    if not captured:
        # Reached only when a fully admissible pass found nothing new: since
        # FOR-1650 a refused pass raises above and moves nothing into dest, so
        # no pass AFTER that fix can strand admissible files here. Files a
        # pre-FOR-1650 checkout stranded in dest are still skipped as
        # "already captured" and need one manual `dvc add <dest>` to be tracked.
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
    # Module-attribute call (not a from-import) so the gate stays patchable
    # in tests; verify_cloud_sync raises when the remote does not verifiably
    # hold what the freshly rewritten pointer names — exit 0 from this
    # command MEANS "remote verified", never just "push did not error".
    capture_results_mod.verify_cloud_sync(dest)
    logger.info(f"capture-results: {len(captured)} {label} captured, pushed, and cloud-verified")
    return 0


def cmd_mlflow_aggregate(args: argparse.Namespace) -> int:
    """Handle ``mlflow-aggregate``."""
    if getattr(args, "aggregate_lane", None) == H_LANE:
        return cmd_aggregate_h(args, _emit_h_document)
    tracking_uri = args.tracking_uri or os.environ.get(MLFLOW_TRACKING_ENV)
    if not tracking_uri:
        logger.error(
            f"mlflow-aggregate: no tracking URI (--tracking-uri or ${MLFLOW_TRACKING_ENV})"
        )
        return 2
    from entropic_substrate.ladder.capture.adapters.mlflow import (
        MlflowException,
        build_mlflow_client,
    )

    client = build_mlflow_client(tracking_uri)
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
    if getattr(args, "charts_lane", None) == H_LANE:
        return cmd_charts_h(args, _emit_h_document)
    from entropic_substrate.ladder.analyze import charts as chart_fns
    from entropic_substrate.ladder.analyze.series import (
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

    from entropic_substrate.ladder.analyze.mlflow_attach import (
        attach_figures_to_run,
        attach_seed_band_to_aggregate,
    )
    from entropic_substrate.ladder.capture.adapters.mlflow import build_mlflow_client

    client = build_mlflow_client(tracking_uri)

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
                "hex digest (entropic_substrate.ladder.capture.reader.sinks.config_hash()'s exact shape) — "
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


def cmd_embedding_geometry(args: argparse.Namespace) -> int:
    """Handle ``embedding-geometry``; the diagnostics live in ``embedding_geometry_cli``.

    This shell owns the one machine-consumed ``--json`` print, which the print ban
    allows only in this module.
    """
    run = compute_embedding_geometry(args)
    if run is None:
        return 2
    if args.json:
        # MACHINE-CONSUMED STDOUT, NOT LOGS — same idiom as cmd_eval's --json.
        print(json.dumps(run.payload, sort_keys=True))
        return 0
    write_embedding_geometry_report(args, run)
    return 0


# Re-exported under their historical names: tests/test_embedding_geometry.py takes both
# from this module (tests/test_cli_surface.py pins them).
_init_table_cache_key = init_table_cache_key
_cached_reconstruct_init_table = cached_reconstruct_init_table


def _emit_h_document(document: Any) -> None:
    # MACHINE-CONSUMED STDOUT, NOT LOGS: raw print(), one JSON document, never routed
    # through this module's sinks (see entropic_substrate.logging's module docstring).
    # The H analysis handlers (``h_analysis_cli``, which holds no print) receive this as ``emit``.
    print(json.dumps(document, sort_keys=True, allow_nan=False))


type _Handler = Callable[[argparse.Namespace], int]


def main(
    argv: Sequence[str] | None = None,
    *,
    d1_application: D1CliApplication | None = None,
) -> int:
    """Parse ``argv`` and dispatch. Returns a process exit code."""
    configure_logging()
    args = build_parser().parse_args(argv)
    from entropic_substrate.composition.wiring.d1_cli import D1CliCommand

    if args.command in {command.value for command in D1CliCommand}:
        command = D1CliCommand(args.command)
        if d1_application is None:
            from entropic_substrate.composition.wiring.d1_cli_composition import (
                build_default_d1_cli_application,
            )

            application = build_default_d1_cli_application()
        else:
            application = d1_application
        outcome = application.execute(command, args.manifest)
        print(
            json.dumps(
                outcome.manifest,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return outcome.exit_code

    from entropic_substrate.composition.capture_cli import cmd_capture
    from entropic_substrate.composition.cli import cmd_sentence
    from entropic_substrate.composition.source_cli import cmd_source
    from entropic_substrate.composition.span_cli import cmd_span
    from entropic_substrate.composition.wiring.d_ladder_cli import cmd_d_probe
    from entropic_substrate.composition.wiring.d_ladder_probe_cli import cmd_score_gain

    handlers: dict[str, _Handler] = {
        "train": cmd_train,
        "init-fisher": cmd_init_fisher,
        "eval": cmd_eval,
        "capture-results": cmd_capture_results,
        "mlflow-aggregate": cmd_mlflow_aggregate,
        "charts": cmd_charts,
        "embedding-geometry": cmd_embedding_geometry,
        "field-pretrain": cmd_field_pretrain,
        "d-probe": cmd_d_probe,
        "d-score-gain": cmd_score_gain,
        "h-train-probe": cmd_sentence,
        "h-inspect-training": cmd_sentence,
        "h-generate-probe": cmd_sentence,
        "h-inspect-generation": cmd_sentence,
        "h-capture-model": cmd_capture,
        "h-inspect-model": cmd_capture,
        "h-inspect-custody": cmd_capture,
        "h-diagnose-spans": cmd_span,
        "h-inspect-spans": cmd_span,
        "h-prepare-source": cmd_source,
        "h-inspect-source": cmd_source,
    }
    try:
        return handlers[args.command](args)
    except CheckpointFormatError as exc:
        # Scope: this fires only for the E/D commands whose import closure
        # raises CheckpointFormatError; the h-* handlers do not reach it.
        # An ambiguous or too-new checkpoint is an expected, operator-actionable
        # condition — the guidance is in the message. Present it the way every
        # other gate here does (printed refusal, exit 2), not as a traceback
        # that reads like a defect in the trainer.
        logger.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
