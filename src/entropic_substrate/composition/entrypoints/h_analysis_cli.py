"""The H analysis lanes of the entry-point CLI (FOR-1648, DN-1648-B; moved here by FOR-1608).

``eval h``, ``charts h`` and ``mlflow-aggregate h`` are optional nested sub-subparsers on the
EXISTING host parsers, so the frozen top-level subcommand set (tests/test_cli_surface.py) is
untouched. ``entrypoints.cli`` routes to the handlers below BEFORE any E gate (a tracking-URI
check, an E input check), so an offline H aggregate never demands a server.

Machine-consumed stdout is exactly ONE JSON document per invocation. This module holds no
``print()`` (the print ban in tests/test_observability_conformance.py allowlists only named
entry-point modules): every handler takes the ``emit`` callable ``entrypoints.cli`` owns and
hands its document to it. Every H module is imported inside the handlers: this module keeps no
module-level import of ladder.analyze.h_*, matplotlib or mlflow.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.reader.sinks import MLFLOW_TRACKING_ENV
from entropic_substrate.ladder.capture.reader.telemetry import EXIT_TELEMETRY_GAP, TelemetryGap

__all__ = [
    "EXIT_H_ATTACH_FAILED",
    "H_LANE",
    "HEmit",
    "add_h_analysis_lanes",
    "cmd_aggregate_h",
    "cmd_charts_h",
    "cmd_eval_h",
]

H_LANE = "h"
EXIT_H_ATTACH_FAILED = 1

# Writes the one JSON document an H handler produced to stdout (``cli._emit_h_document``).
type HEmit = Callable[[Any], None]


def add_h_analysis_lanes(
    evaluate: argparse.ArgumentParser,
    aggregate_p: argparse.ArgumentParser,
    charts: argparse.ArgumentParser,
) -> None:
    """The H analysis surface (FOR-1648, operator door ruling B, conductor DN-1648-B).

    Three optional nested sub-subparsers named ``h``, one on each EXISTING host parser, so
    the frozen top-level subcommand set (tests/test_cli_surface.py) is untouched. Every
    option an ``h`` parser declares has a dest prefixed ``h_``, disjoint from every dest of
    its host: argparse writes a child's defaults over the parent's parsed values, so a
    shared dest would silently clobber a host option (MEASURED on 3.12:
    ``charts --out-dir X h`` came back with the child's default).
    """

    def add_run_inputs(parser: argparse.ArgumentParser, *, many: bool) -> None:
        parser.add_argument(
            "--run",
            dest="h_run",
            nargs=2,
            metavar=("METRICS_JSONL", "WORKER_LOG_JSONL"),
            type=Path,
            required=True,
            **({"action": "append"} if many else {}),
            help=(
                "one captured H run: its metrics.jsonl and its worker-log.jsonl; the two are "
                "bound by request_sha256, never by file name"
                + (" (repeatable, one per arm)" if many else "")
            ),
        )

    def add_arms(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--arm",
            dest="h_arm",
            action="append",
            required=True,
            metavar="IDENTITY=LABEL",
            help=(
                "label one run by its full 64-hex profile identity (repeatable); an unknown "
                "identity or an unlabelled run is refused, labels are never inferred"
            ),
        )

    aggregate_lane = aggregate_p.add_subparsers(
        dest="aggregate_lane", required=False, metavar="LANE"
    )
    h_aggregate = aggregate_lane.add_parser(
        "h", help="aggregate captured H runs, paired by global step (offline, no MLflow)"
    )
    add_run_inputs(h_aggregate, many=True)
    add_arms(h_aggregate)

    charts_lane = charts.add_subparsers(dest="charts_lane", required=False, metavar="LANE")
    h_charts = charts_lane.add_parser(
        "h", help="render the H chart set over captured runs; --attach publishes to MLflow"
    )
    add_run_inputs(h_charts, many=True)
    add_arms(h_charts)
    h_charts.add_argument(
        "--out-dir", dest="h_out_dir", type=Path, default=Path("results/h-charts")
    )
    h_charts.add_argument(
        "--attach",
        dest="h_attach",
        action="store_true",
        help="attach the charts to each arm's MLflow run (needs a tracking URI; fails loud)",
    )
    h_charts.add_argument(
        "--tracking-uri",
        dest="h_tracking_uri",
        default=None,
        help=f"defaults to ${MLFLOW_TRACKING_ENV}; only read with --attach",
    )

    eval_lane = evaluate.add_subparsers(dest="eval_lane", required=False, metavar="LANE")
    h_eval = eval_lane.add_parser(
        "h", help="report one captured H run (offline, no MLflow); prints one JSON document"
    )
    add_run_inputs(h_eval, many=False)
    h_eval.add_argument(
        "--receipt-root",
        dest="h_receipt_root",
        type=Path,
        default=None,
        help="with --receipt-manifest: also verify the training receipt via h-inspect-training",
    )
    h_eval.add_argument("--receipt-manifest", dest="h_receipt_manifest", default=None)


def _h_host_conflict(args: argparse.Namespace, command: str, lane_dest: str) -> str | None:
    """The first host-parser option, set away from its default, that the `h` lane would
    silently ignore (``None`` when there is none)."""
    # Function-local: ``cli`` imports this module at load, so a module-level import back
    # would be a cycle. Resolved at call time, so a patched ``cli.build_parser`` is seen.
    from entropic_substrate.composition.entrypoints.cli import build_parser

    defaults = vars(build_parser().parse_args([command]))
    for dest, default in defaults.items():
        if dest in ("command", lane_dest):
            continue
        if getattr(args, dest, default) != default:
            return "--" + dest.replace("_", "-")
    return None


def _h_refuse_host_args(args: argparse.Namespace, command: str, lane_dest: str) -> int | None:
    conflict = _h_host_conflict(args, command, lane_dest)
    if conflict is None:
        return None
    logger.error(
        f"{command} h: {conflict} belongs to the E lane, which `h` does not consume; "
        f"give every H option after `h`"
    )
    return 2


def _h_aggregate_document(args: argparse.Namespace) -> dict[str, Any]:
    from entropic_substrate.ladder.analyze.h_aggregate import (
        aggregate_h_runs,
        load_h_run,
        parse_arm_labels,
    )

    labels = parse_arm_labels(args.h_arm)
    runs = [load_h_run(metrics, worker_log) for metrics, worker_log in args.h_run]
    return aggregate_h_runs(runs, labels)


def cmd_aggregate_h(args: argparse.Namespace, emit: HEmit) -> int:
    """Handle ``mlflow-aggregate h``: offline, no MLflow client, no tracking URI."""
    refused = _h_refuse_host_args(args, "mlflow-aggregate", "aggregate_lane")
    if refused is not None:
        return refused
    try:
        document = _h_aggregate_document(args)
    except TelemetryGap as gap:
        logger.error(f"mlflow-aggregate h: telemetry gap: {gap}")
        return EXIT_TELEMETRY_GAP
    except ValueError as exc:
        logger.error(f"mlflow-aggregate h: {exc}")
        return 2
    emit(document)
    return 0


def cmd_charts_h(args: argparse.Namespace, emit: HEmit) -> int:
    """Handle ``charts h``: render locally first, attach only with ``--attach``."""
    refused = _h_refuse_host_args(args, "charts", "charts_lane")
    if refused is not None:
        return refused
    attach_uri: str | None = None
    if args.h_attach:
        attach_uri = args.h_tracking_uri or os.environ.get(MLFLOW_TRACKING_ENV)
        if not attach_uri:
            logger.error(f"charts h: --attach needs --tracking-uri or ${MLFLOW_TRACKING_ENV}")
            return 2
    from entropic_substrate.ladder.analyze.h_charts import build_h_figures

    try:
        document = _h_aggregate_document(args)
        figures, absent = build_h_figures(document)
    except TelemetryGap as gap:
        logger.error(f"charts h: telemetry gap: {gap}")
        return EXIT_TELEMETRY_GAP
    except ValueError as exc:
        logger.error(f"charts h: {exc}")
        return 2
    args.h_out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, figure in figures.items():
        out_path = args.h_out_dir / f"{name}.png"
        figure.savefig(out_path, dpi=150)
        written.append(str(out_path))
    for name, reason in absent.items():
        logger.warning(f"charts h: {name} not rendered: {reason}")
    result: dict[str, Any] = {"absent": absent, "attached": {}, "written": sorted(written)}
    if attach_uri is None:
        emit(result)
        return 0

    from entropic_substrate.ladder.analyze.h_attach import HAttachError, attach_h_figures
    from entropic_substrate.ladder.capture.adapters.mlflow import build_mlflow_client

    client = build_mlflow_client(attach_uri)
    targets = {label: arm["mlflow_run_id"] for label, arm in document["arms"].items()}
    try:
        result["attached"] = attach_h_figures(client, targets, figures)
    except HAttachError as exc:
        result["attached"] = exc.attached
        result["attach_failures"] = exc.failures
        logger.error(f"charts h: {exc}")
        emit(result)
        return EXIT_H_ATTACH_FAILED
    emit(result)
    return 0


def _h_receipt_verdict(root: Path, manifest: str) -> tuple[dict[str, Any], int]:
    """Delegate to the unedited ``h-inspect-training`` handler and capture its one JSON line."""
    import contextlib
    import io

    from entropic_substrate.composition.cli import cmd_sentence

    captured = io.StringIO()
    delegate_args = argparse.Namespace(command="h-inspect-training", root=root, manifest=manifest)
    with contextlib.redirect_stdout(captured):
        code = cmd_sentence(delegate_args)
    verdict = json.loads(captured.getvalue())
    if not isinstance(verdict, dict):
        msg = "h-inspect-training printed a non-object verdict"
        raise ValueError(msg)
    return verdict, code  # pyright: ignore[reportUnknownVariableType]


def cmd_eval_h(args: argparse.Namespace, emit: HEmit) -> int:
    """Handle ``eval h``: one captured run, offline, one JSON document."""
    refused = _h_refuse_host_args(args, "eval", "eval_lane")
    if refused is not None:
        return refused
    if (args.h_receipt_root is None) != (args.h_receipt_manifest is None):
        logger.error("eval h: --receipt-root and --receipt-manifest must be given together")
        return 2
    from entropic_substrate.ladder.analyze.h_report import build_h_report

    try:
        report = build_h_report(*args.h_run)
        code = 0
        if args.h_receipt_root is not None:
            verdict, code = _h_receipt_verdict(args.h_receipt_root, args.h_receipt_manifest)
            report["receipt_exit_code"] = code
            report["receipt_verdict"] = verdict
    except TelemetryGap as gap:
        logger.error(f"eval h: telemetry gap: {gap}")
        return EXIT_TELEMETRY_GAP
    except ValueError as exc:
        logger.error(f"eval h: {exc}")
        return 2
    emit(report)
    return code
