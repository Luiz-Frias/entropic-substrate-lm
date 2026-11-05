"""Read side of an H run's ``metrics.jsonl`` -- the H lane's admissibility gate.

The E-series gate (:func:`~entropic_substrate.ladder.capture.reader.telemetry.final_eval_record`)
asks for a final ``event == "eval"`` row; an H worker's stream carries
``h-training-step-v2`` step rows, ``h-predictive-evaluation-v1`` or ``-v2`` reports (FOR-1704;
this module gates on neither schema, only on the lifecycle pair and a committed step row) and
two ``h-run-lifecycle-v1`` records (``run_start`` / ``run_end``) and no
``eval`` event, so the E predicate refuses every H run. This module is the H
counterpart, with the same posture: an inadmissible file raises
:class:`~entropic_substrate.ladder.capture.reader.telemetry.TelemetryGap`
naming the file and the reason, and no empty value is ever returned.

A ``metrics.jsonl`` is admissible iff it holds exactly one ``run_start``,
exactly one ``run_end`` and at least one committed ``h-training-step-v2``
row. There is deliberately no ``status == "ok"`` condition: a failed run
that committed a step is still evidence. Whether the run succeeded is the
returned ``run_end`` record's ``status``, for the caller to read.

AMENDMENT (house rule 4, FOR-1693, DH-46, 2026-11-01): the committed-row condition has ONE
exception. A stream whose ``run_end`` records a scientific stop (``status == failed`` and
``result_status == scientific_gap``) is admissible with ZERO committed rows: a codec warm-up
that diverged at its first step is a recorded scientific outcome, not a run that never
trained. The exception is keyed on that record and never on a row count, so an
execution-truncated stream (no such record) is refused exactly as above. The text above stands
for every stream that does not carry the record.

RETRACTION (house rule 4, FOR-1642, conductor 2026-10-15, DN-1642-FAILED-CAPTURE): this
docstring used to give, as the reason for that rule, "MLflow marks every H run FINISHED
(``MlflowSink.finish`` is unconditional), so the stream is the only place a failed run can
be told from a complete one". That premise was true when written (slice 3) and is FALSE
since DN-1643-PORT: ``MetricSink.finish`` takes a status and the H worker and the local
probe terminate their MLflow run ``FAILED`` on every raised exit. The rule it justified
stands for a different reason, the one above; the run_end record, not the MLflow status,
remains the evidence of how a run ended, and ``capture_finished_runs`` now admits an
H-lane ``FAILED`` run through this same predicate.

The schema strings are this module's own literals: the capture layer never
imports from ``ladder/run``. Emitter-to-reader agreement is pinned by a test
that runs the real worker and feeds its file back through this predicate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

__all__ = ["final_h_run_end", "is_scientific_stop", "require_json_object_lines"]

_LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"
_STEP_SCHEMA = "h-training-step-v2"
_COMMITTED = "committed"
# The recorded scientific stop (DH-46): the ``run_end`` pair the worker writes when a run's
# training raised a scientific gap (``object_worker``'s failure arm). These two literals are
# this module's own; the capture layer never imports from ``ladder/run``, and
# tests/test_h_pair_request.py pins them equal to the emitter by running the real worker.
_FAILED_STATUS = "failed"
_SCIENTIFIC_GAP = "scientific_gap"


def is_scientific_stop(run_end: dict[str, Any]) -> bool:
    """True iff ``run_end`` records a scientific stop: ``status == failed`` AND
    ``result_status == scientific_gap``. Keyed on the record, never on row counts."""
    return (
        run_end.get("status") == _FAILED_STATUS and run_end.get("result_status") == _SCIENTIFIC_GAP
    )


def _json_object_lines(path: Path) -> list[dict[str, Any]]:
    """Every non-blank line of ``path`` as a JSON object, or a TelemetryGap.

    Unlike the E reader's silent skip of non-dict values, a line that parses
    to anything but an object is a defect in the stream and raises. The file
    is read as bytes and decoded per line so a corrupt byte is refused as a
    ``TelemetryGap`` naming its line. Measured 2026-10-15 (RED run of
    ``tests/test_h_telemetry.py``): a text-mode read raised
    ``UnicodeDecodeError`` out of the iteration, which is not a
    ``TelemetryGap`` and so is not caught by ``cmd_capture_results``.
    """
    if not path.exists():
        msg = f"{path}: no such file — the H run never wrote telemetry here"
        raise TelemetryGap(msg)
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                msg = f"{path}: line {number} is not valid UTF-8 ({exc}) — corrupt write?"
                raise TelemetryGap(msg) from exc
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}: line {number} is not valid JSON ({exc}) — truncated write?"
                raise TelemetryGap(msg) from exc
            if not isinstance(record, dict):
                msg = (
                    f"{path}: line {number} is not a JSON object "
                    f"(got {type(record).__name__}) — the H stream is JSON-object lines"
                )
                raise TelemetryGap(msg)
            records.append(record)
    return records


def require_json_object_lines(path: str | Path) -> None:
    """Refuse an H-run artifact (e.g. ``worker-log.jsonl``) that is not well-formed.

    Well-formedness only: every non-blank line must parse to a JSON object.
    An empty file passes (a quiet worker is not a malformed one).
    """
    _json_object_lines(Path(path))


def final_h_run_end(path: str | Path) -> dict[str, Any]:
    """The ``run_end`` lifecycle record of an admissible H ``metrics.jsonl``.

    Raises :class:`TelemetryGap`, naming ``path``, unless the file holds
    exactly one ``run_start``, exactly one ``run_end`` and at least one
    committed ``h-training-step-v2`` row. AMENDMENT (DH-46): the committed-row condition
    is waived when that ``run_end`` records a scientific stop (:func:`is_scientific_stop`).
    """
    path = Path(path)
    records = _json_object_lines(path)
    lifecycle = [r for r in records if r.get("schema") == _LIFECYCLE_SCHEMA]
    starts = [r for r in lifecycle if r.get("event") == "run_start"]
    ends = [r for r in lifecycle if r.get("event") == "run_end"]
    committed = [
        r for r in records if r.get("schema") == _STEP_SCHEMA and r.get("status") == _COMMITTED
    ]
    if len(starts) != 1:
        msg = (
            f"{path}: an admissible H run holds exactly one run_start "
            f"({_LIFECYCLE_SCHEMA}), found {len(starts)}"
        )
        raise TelemetryGap(msg)
    if len(ends) != 1:
        msg = (
            f"{path}: an admissible H run holds exactly one run_end "
            f"({_LIFECYCLE_SCHEMA}), found {len(ends)} — the worker never reached its end"
        )
        raise TelemetryGap(msg)
    if not committed and not is_scientific_stop(ends[0]):
        msg = (
            f"{path}: an admissible H run holds at least one committed {_STEP_SCHEMA} "
            "row, found none — the run never trained"
        )
        raise TelemetryGap(msg)
    return ends[0]
