"""Public facade for authoritative D1 telemetry values and queries."""

from __future__ import annotations

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

__all__ = [
    "D1EvalEvidence",
    "D1QualityEvidence",
    "D1RunEvidence",
    "D1RunStatus",
    "D1TelemetryGap",
    "final_d1_eval_metric",
    "read_d1_eval_series",
    "read_d1_run_evidence",
]
