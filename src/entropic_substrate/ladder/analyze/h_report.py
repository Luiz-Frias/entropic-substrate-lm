"""Single-run H report: lineage, status, step range, evaluations, E-LM and the final leg.

A pure function of one run's captured ``metrics.jsonl`` and ``worker-log.jsonl``,
admitted through :func:`~entropic_substrate.ladder.analyze.h_aggregate.load_h_run`
(and so through the existing H reader). It never touches MLflow: the evidence is the
captured file. Optional receipt verification (``h-inspect-training``) is delegated by
the CLI, which hands the delegate's verdict in verbatim; this module reimplements none
of it and prints nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from entropic_substrate.ladder.analyze.h_aggregate import HRun, load_h_run, run_summary

__all__ = ["REPORT_SCHEMA", "build_h_report", "h_report_from_run"]

REPORT_SCHEMA = "h-report-v1"


def h_report_from_run(
    run: HRun, *, receipt_verdict: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report document for an admitted run (JSON-safe, every number captured)."""
    summary = run_summary(run)
    report: dict[str, Any] = {
        "checkpoints": {
            "publication_seconds": {
                str(step): s for step, s in sorted(run.publication_seconds.items())
            },
            "seal_steps": list(run.seal_steps),
        },
        "elm": {
            "denominator": summary["elm"]["denominator"],
            "headline_samples": summary["elm"]["headline_samples"],
            "steps": {str(step): projection for step, projection in run.elm},
        },
        "evaluation": {
            "metrics": {str(step): metrics for step, metrics in run.evaluations},
            "steps": summary["evaluation_steps"],
        },
        "final_leg": {
            "admissions": [dict(a) for a in run.admissions],
            "stages": [dict(s) for s in run.stages],
        },
        "future_nll_per_dimension": summary["future_nll_per_dimension"],
        "lineage": {
            **summary["lineage"],
            "mlflow_run_id": run.mlflow_run_id,
            "profile_identity": run.profile_identity,
            "request_sha256": run.request_sha256,
            "runtime": _runtime(run),
        },
        "schema": REPORT_SCHEMA,
        "status": {
            "completed": summary["completed"],
            "result_status": summary["run_end"]["result_status"],
            "status": summary["run_end"]["status"],
            "successful_optimizer_steps": summary["run_end"]["successful_optimizer_steps"],
            "wall_seconds": summary["run_end"]["wall_seconds"],
        },
        "steps": {
            "committed_rows": summary["committed_step_rows"],
            "first_prediction_step": summary["first_prediction_step"],
            "range": summary["step_range"],
            "uncommitted_rows": summary["uncommitted_step_rows"],
            "warmup_steps": summary["warmup_steps"],
        },
    }
    if receipt_verdict is not None:
        report["receipt_verdict"] = dict(receipt_verdict)
    return report


def _runtime(run: HRun) -> dict[str, Any]:
    return dict(run.runtime)


def build_h_report(
    metrics_path: str | Path,
    worker_log_path: str | Path,
    *,
    receipt_verdict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit one captured run and report it; raises ``TelemetryGap`` when inadmissible."""
    return h_report_from_run(
        load_h_run(metrics_path, worker_log_path), receipt_verdict=receipt_verdict
    )
