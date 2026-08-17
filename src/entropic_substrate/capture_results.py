"""Retrieval-side capture of run results JSONLs into DVC.

Pods run credential-free (operator ruling 2026-08-17): a training pod holds
no R2/DVC identity and uploads its authoritative ``results/*.jsonl`` to the
tracking server as a run artifact (``MlflowSink.finish()``). This module is
the other half of that contract — it runs on an operator machine that DOES
hold R2 credentials, pulls those artifacts down, validates each one through
:func:`entropic_substrate.telemetry.final_eval_record` (the same admissibility
predicate every other JSONL consumer uses), and hands the capture directory to
DVC.

Measurement path: it fails LOUD. An unknown experiment raises, an
inadmissible file raises :class:`~entropic_substrate.telemetry.TelemetryGap`
(the offending file is kept on disk as evidence, named in the exception), and
nothing is ever captured silently. ``-tiny`` smoke runs are plumbing checks,
not science — excluded unless explicitly asked for.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from entropic_substrate.logging import logger
from entropic_substrate.telemetry import final_eval_record

__all__ = ["capture_finished_runs", "dvc_commands"]

_RESULTS_PREFIX = "results"
_TINY_SUFFIX = "-tiny"


def capture_finished_runs(
    client: Any,
    experiment_names: Sequence[str],
    dest: str | Path,
    *,
    include_tiny: bool = False,
) -> list[Path]:
    """Download every FINISHED run's results JSONLs into ``dest``.

    Returns the newly captured paths; files already present in ``dest`` are
    skipped (idempotent — re-running after new runs finish captures only the
    delta, and ``dvc add`` on the directory picks up exactly that delta).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    experiment_ids: list[str] = []
    for name in experiment_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            msg = f"experiment {name!r} not found on the tracking server"
            raise LookupError(msg)
        experiment_ids.append(experiment.experiment_id)

    captured: list[Path] = []
    for run in client.search_runs(experiment_ids, max_results=500):
        info = run.info
        if info.status != "FINISHED":
            continue
        if not include_tiny and info.run_name.endswith(_TINY_SUFFIX):
            continue
        for artifact in client.list_artifacts(info.run_id, _RESULTS_PREFIX):
            if not artifact.path.endswith(".jsonl"):
                continue
            target = dest / f"{info.run_name}-{info.run_id[:8]}-{Path(artifact.path).name}"
            if target.exists():
                continue
            # download_artifacts recreates the artifact tree under its dst;
            # stage in a scratch dir so dest holds only the flat capture set.
            with tempfile.TemporaryDirectory(dir=dest) as scratch:
                local = Path(client.download_artifacts(info.run_id, artifact.path, scratch))
                shutil.move(str(local), target)
            # Admissibility gate — raises TelemetryGap on a file the readers
            # would refuse; target stays on disk as evidence, named in the
            # exception message.
            final_eval_record(target)
            captured.append(target)
            logger.info(f"captured {target}")
    return captured


def dvc_commands(dest: str | Path) -> list[list[str]]:
    """The exact DVC invocation pair the CLI runs after a capture."""
    return [["dvc", "add", str(Path(dest))], ["dvc", "push"]]
