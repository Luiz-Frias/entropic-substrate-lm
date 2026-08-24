"""Public facade for authoritative D1 telemetry values and queries."""

from __future__ import annotations

from pathlib import Path

from entropic_substrate.d1_numerical_defect import D1TerminalNumericalDefectRecord
from entropic_substrate.evidence.d1_telemetry_query import (
    final_d1_eval_metric,
    read_d1_eval_series,
    read_d1_run_evidence,
)
from entropic_substrate.evidence.d1_telemetry_values import (
    D1EvalEvidence,
    D1QualityEvidence,
    D1RunEvidence,
    D1RunStatus,
    D1TelemetryGap,
)
from entropic_substrate.evidence.d1_terminal_defect_query import (
    validate_d1_terminal_defect,
)

__all__ = [
    "D1EvalEvidence",
    "D1QualityEvidence",
    "D1RunEvidence",
    "D1RunStatus",
    "D1TelemetryGap",
    "final_d1_eval_metric",
    "read_d1_eval_series",
    "read_d1_run_evidence",
    "read_d1_terminal_defect",
]


def read_d1_terminal_defect(
    path: str | Path, **expected: object
) -> D1TerminalNumericalDefectRecord:
    run = read_d1_run_evidence(path, allow_incomplete=True, **expected)  # type: ignore[arg-type]
    return validate_d1_terminal_defect(path, run)
