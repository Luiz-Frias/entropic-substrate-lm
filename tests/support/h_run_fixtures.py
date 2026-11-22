"""Synthetic H run streams for loader tests (moved from tests/integration/test_h_aggregate.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

IDENTITY_A = "0" * 63 + "1"


REQUEST_A = "1" * 63 + "a"


RUN_ID_A = "a" * 32


# ------------------------------------------------------------------ synthetic builder
def _step_row(
    identity: str, phase: str, step: int, scale: float, *, committed: bool = True
) -> dict[str, Any]:
    glob = step if phase == "codec_warmup" else step + 2  # warm-up of 2 in the default builder
    return {
        "aggregate_norm": 1.0 + 0.01 * glob,
        "decoder_ce": scale * (6.0 - 0.1 * glob),
        "future_nll_per_dimension": None if phase == "codec_warmup" else scale * (1.0 - 0.4 * step),
        "objective": scale * (7.0 - 0.1 * glob),
        "phase": phase,
        "run_identity": identity,
        "schema": "h-training-step-v2",
        "status": "committed" if committed else "failed",
        "step": step,
        "wall_seconds": 0.5,
    }


def _eval_row(identity: str, step: int, scale: float) -> dict[str, Any]:
    def entry(estimate: float, ess: float) -> dict[str, Any]:
        return {
            "effective_sample_size": ess,
            "gap": None,
            "likelihood_relative_mcse": 0.5,
            "log_estimate": estimate,
        }

    return {
        "checkpoint_seal": {"path": f"seal-{step}.json", "sha256": "2" * 64},
        "cuda_seconds": 0.25,
        "examples": [
            {
                "boundary_symbols": [1, 1],
                "source_bytes": [100, 200],
                "by_sample_count": {
                    "2": [entry(-100.0 * scale - step, 1.0), entry(-110.0 * scale - step, 1.5)],
                    "4": [entry(-99.0 * scale - step, 2.0), entry(-98.0 * scale - step, 2.5)],
                },
            }
        ],
        "meaning": "importance_weighted_log_marginal_estimate",
        "run_identity": identity,
        "schema": "h-predictive-evaluation-v1",
        "step": step,
        "wall_seconds": 0.75,
    }


def _annotate(global_step: int, seconds: str) -> dict[str, Any]:
    prefix = f"checkpoint/{global_step:08d}"
    return {
        "event": "annotate",
        f"{prefix}/key": f"snapshots/checkpoint-{global_step:08d}.json",
        f"{prefix}/publication_seconds": seconds,
        f"{prefix}/sha256": "3" * 64,
        f"{prefix}/size_bytes": "100",
    }


def _log(event: str, **fields: Any) -> dict[str, Any]:
    return {"event": event, "level": "INFO", "message": event, **fields}


def build_h_run_rows(
    identity: str,
    request: str,
    *,
    run_id: str | None = RUN_ID_A,
    warmup: int = 2,
    steps: int = 4,
    interval: int = 2,
    scale: float = 1.0,
    status: str = "ok",
    result_status: str = "probe_workload_complete",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(metrics rows, worker-log rows)`` of a tiny admissible run (warm-up 2, 4 steps)."""
    assert warmup == 2, "the step-row scalar formulas assume a warm-up of 2"
    metrics: list[dict[str, Any]] = [
        {
            "code_commit": "c" * 40,
            "event": "run_start",
            "image": "example.invalid/image@sha256:" + "4" * 64,
            "profile_identity": identity,
            "request_sha256": request,
            "run_class": "remote_tiny",
            "runtime": {"accelerator": {"count": 1, "model": "synthetic"}, "pod_id": "pod"},
            "schema": "h-run-lifecycle-v1",
        }
    ]
    log: list[dict[str, Any]] = [_log("worker.start", request_sha256=request)]
    if run_id is not None:
        log.append(_log("mlflow.run", run_id=run_id))
    metrics.extend(_step_row(identity, "codec_warmup", s, scale) for s in range(1, warmup + 1))
    metrics.append(_annotate(warmup, "1.5"))
    log.append(_log("checkpoint.seal", phase="codec_warmup", step=warmup, global_step=warmup))
    for p in range(1, steps + 1):
        metrics.append(_step_row(identity, "prediction", p, scale))
        if p % interval == 0:
            metrics.append(_eval_row(identity, p, scale))
            metrics.append(_annotate(warmup + p, "2.5"))
            log.append(_log("checkpoint.seal", phase="prediction", step=p, global_step=warmup + p))
            log.append(_log("evaluation.start", global_step=warmup + p))
    log.append(_log("stage.end", stage="training_readback", seconds=2.0))
    log.append(
        _log(
            "admission.end",
            caller="completion",
            seconds=3.0,
            seals_walked=2,
            evaluations=2,
            journals_read=1,
        )
    )
    metrics.append(
        {
            "event": "run_end",
            "result_status": result_status,
            "schema": "h-run-lifecycle-v1",
            "status": status,
            "successful_optimizer_steps": warmup + steps,
            "wall_seconds": 12.5,
        }
    )
    return metrics, log


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def write_h_run(
    directory: Path,
    identity: str = IDENTITY_A,
    request: str = REQUEST_A,
    *,
    tag: str = "a",
    mutate_metrics: Callable[[list[dict[str, Any]]], None] | None = None,
    mutate_log: Callable[[list[dict[str, Any]]], None] | None = None,
    **kwargs: Any,
) -> tuple[Path, Path]:
    metrics, log = build_h_run_rows(identity, request, **kwargs)
    if mutate_metrics is not None:
        mutate_metrics(metrics)
    if mutate_log is not None:
        mutate_log(log)
    directory.mkdir(parents=True, exist_ok=True)
    return (
        _dump(directory / f"{tag}-metrics.jsonl", metrics),
        _dump(directory / f"{tag}-worker-log.jsonl", log),
    )
