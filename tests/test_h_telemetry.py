"""Admissibility of an H run's ``metrics.jsonl`` -- the read side of the
``h-run-lifecycle-v1`` records (FOR-1639).

``final_h_run_end`` is the H lane's counterpart of ``final_eval_record``: a
file is admissible iff it holds exactly one ``run_start``, exactly one
``run_end`` and at least one committed ``h-training-step-v2`` row. Anything
else raises :class:`TelemetryGap` naming the file and the H-specific reason;
no empty value is ever returned. These tests pin the predicate over
hand-built fixtures; the emitter-to-reader agreement over the real worker's
output lives in ``tests/test_h_worker_command_observability.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.ladder.capture.reader.h_telemetry import (
    final_h_run_end,
    require_json_object_lines,
)
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

pytestmark = pytest.mark.unit

LIFECYCLE = "h-run-lifecycle-v1"


def _start() -> dict[str, Any]:
    return {
        "schema": LIFECYCLE,
        "event": "run_start",
        "request_sha256": "a" * 64,
        "image": "img",
        "code_commit": "c" * 40,
        "profile_identity": "h0-x",
        "run_class": "local_tiny",
        "runtime": {"kind": "local_tiny"},
    }


def _end(status: str = "ok") -> dict[str, Any]:
    return {
        "schema": LIFECYCLE,
        "event": "run_end",
        "status": status,
        "result_status": "probe_workload_complete" if status == "ok" else "execution_defect",
        "wall_seconds": 1.5,
        "successful_optimizer_steps": 6 if status == "ok" else None,
    }


def _step(status: str = "committed", step: int = 1) -> dict[str, Any]:
    return {"schema": "h-training-step-v2", "status": status, "phase": "prediction", "step": step}


def _write(path: Path, rows: Sequence[dict[str, Any] | str]) -> Path:
    lines = [row if isinstance(row, str) else json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_admissible_file_returns_its_run_end_record(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), _step(step=2), _end()])
    assert final_h_run_end(path) == _end()


def test_failed_run_with_a_committed_step_is_still_admissible(tmp_path: Path) -> None:
    """The predicate carries no ``status == ok`` condition: a failed run that
    committed at least one step is evidence, and its run_end is returned."""
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), _end("failed")])
    assert final_h_run_end(path)["status"] == "failed"


def test_foreign_rows_between_the_records_are_tolerated(tmp_path: Path) -> None:
    evaluation = {"schema": "h-predictive-evaluation-v1", "status": "sealed"}
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), evaluation, _end()])
    assert final_h_run_end(path)["event"] == "run_end"


def test_missing_run_end_raises_naming_file_and_reason(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step()])
    with pytest.raises(TelemetryGap, match=r"metrics\.jsonl.*exactly one run_end.*found 0"):
        final_h_run_end(path)


def test_missing_run_start_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_step(), _end()])
    with pytest.raises(TelemetryGap, match=r"metrics\.jsonl.*exactly one run_start.*found 0"):
        final_h_run_end(path)


def test_duplicate_run_end_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), _end(), _end("failed")])
    with pytest.raises(TelemetryGap, match=r"exactly one run_end.*found 2"):
        final_h_run_end(path)


def test_duplicate_run_start_raises(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_start(), _start(), _step(), _end()])
    with pytest.raises(TelemetryGap, match=r"exactly one run_start.*found 2"):
        final_h_run_end(path)


@pytest.mark.parametrize("rows", [[], [_step("started")]])
def test_no_committed_step_raises(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    """Zero rows and a started-but-never-committed step both lack the evidence
    that the run trained at all."""
    path = _write(tmp_path / "metrics.jsonl", [_start(), *rows, _end()])
    with pytest.raises(TelemetryGap, match=r"at least one committed h-training-step-v2"):
        final_h_run_end(path)


def test_a_foreign_committed_row_is_not_a_training_step(tmp_path: Path) -> None:
    """The step filter checks the schema as well as the status: a committed row
    of another schema does not stand in for an h-training-step-v2 row."""
    foreign = {"schema": "h-predictive-evaluation-v1", "status": "committed"}
    path = _write(tmp_path / "metrics.jsonl", [_start(), foreign, _end()])
    with pytest.raises(TelemetryGap, match=r"at least one committed h-training-step-v2"):
        final_h_run_end(path)


def test_a_wrong_schema_lifecycle_row_does_not_count(tmp_path: Path) -> None:
    impostor = {**_end(), "schema": "h-run-lifecycle-v0"}
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), impostor])
    with pytest.raises(TelemetryGap, match=r"exactly one run_end.*found 0"):
        final_h_run_end(path)


def test_an_e_series_eval_row_is_not_a_lifecycle_or_step_row(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [{"event": "run_start"}, {"event": "eval"}])
    with pytest.raises(TelemetryGap, match=r"exactly one run_start.*found 0"):
        final_h_run_end(path)


def test_malformed_json_line_raises_with_its_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path / "metrics.jsonl", [_start(), '{"truncated": ', _end()])
    with pytest.raises(TelemetryGap, match=r"metrics\.jsonl.*line 2.*not valid JSON"):
        final_h_run_end(path)


def test_a_non_object_line_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    """telemetry._iter_records silently skips non-dict values; the H lane must
    not -- a bare number or list in the stream is a defect, not noise."""
    path = _write(tmp_path / "metrics.jsonl", [_start(), _step(), "[1, 2]", _end()])
    with pytest.raises(TelemetryGap, match=r"line 3.*not a JSON object"):
        final_h_run_end(path)


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join([json.dumps(_start()), "", json.dumps(_step()), "   ", json.dumps(_end()), ""]),
        encoding="utf-8",
    )
    assert final_h_run_end(path)["event"] == "run_end"


def test_absent_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TelemetryGap, match=r"no such file"):
        final_h_run_end(tmp_path / "metrics.jsonl")


def test_empty_file_raises_never_returns_an_empty_value(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(TelemetryGap, match=r"exactly one run_start"):
        final_h_run_end(path)


# ---------------------------------------------------- non-metrics H artifacts
def test_json_object_lines_accepts_a_well_formed_worker_log(tmp_path: Path) -> None:
    path = _write(tmp_path / "worker-log.jsonl", [{"event": "worker.start"}, {"level": "INFO"}])
    require_json_object_lines(path)


def test_json_object_lines_accepts_an_empty_file(tmp_path: Path) -> None:
    """Well-formedness only: an empty log is a quiet worker, not a malformed one."""
    path = tmp_path / "worker-log.jsonl"
    path.write_text("", encoding="utf-8")
    require_json_object_lines(path)


def test_json_object_lines_rejects_a_non_json_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "worker-log.jsonl", [{"event": "ok"}, "not json at all"])
    with pytest.raises(TelemetryGap, match=r"worker-log\.jsonl.*line 2.*not valid JSON"):
        require_json_object_lines(path)


def test_json_object_lines_rejects_a_non_object_line(tmp_path: Path) -> None:
    path = _write(tmp_path / "worker-log.jsonl", ['"just a string"'])
    with pytest.raises(TelemetryGap, match=r"line 1.*not a JSON object"):
        require_json_object_lines(path)
