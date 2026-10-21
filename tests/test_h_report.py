"""The single-run H report (FOR-1648): one captured run, read from its files and nothing else.

``unit`` tests use the synthetic builder of tests/test_h_aggregate.py; ``integration`` tests
read the real HEpP7g capture and raise when it is absent (no skip).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_aggregate import (
    IDENTITY_A,
    REQUEST_A,
    RUN_ID_A,
    fixture_arguments,
    write_h_run,
)

from entropic_substrate.ladder.analyze.h_report import build_h_report
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

MLFLOW_BUILDER = "entropic_substrate.ladder.capture.adapters.mlflow.build_mlflow_client"


@pytest.mark.unit
class TestReportFromACapturedRun:
    def test_lineage_status_steps_and_evaluation_steps(self, tmp_path: Path) -> None:
        report = build_h_report(*write_h_run(tmp_path, IDENTITY_A, REQUEST_A))
        assert report["schema"] == "h-report-v1"
        lineage = report["lineage"]
        assert lineage["profile_identity"] == IDENTITY_A
        assert lineage["request_sha256"] == REQUEST_A
        assert lineage["mlflow_run_id"] == RUN_ID_A
        assert lineage["run_class"] == "remote_tiny"
        assert lineage["code_commit"] == "c" * 40
        assert lineage["runtime"]["pod_id"] == "pod"
        assert report["status"] == {
            "completed": True,
            "result_status": "probe_workload_complete",
            "status": "ok",
            "successful_optimizer_steps": 6,
            "wall_seconds": 12.5,
        }
        assert report["steps"] == {
            "committed_rows": 6,
            "first_prediction_step": 3,
            "range": [1, 6],
            "uncommitted_rows": 0,
            "warmup_steps": 2,
        }
        assert report["evaluation"]["steps"] == [4, 6]
        assert set(report["evaluation"]["metrics"]) == {"4", "6"}
        assert "eval/log_marginal_k2" in report["evaluation"]["metrics"]["4"]
        assert report["checkpoints"]["seal_steps"] == [2, 4, 6]
        assert report["checkpoints"]["publication_seconds"] == {"2": 1.5, "4": 2.5, "6": 2.5}

    def test_final_leg_stage_table_is_read_from_the_worker_log(self, tmp_path: Path) -> None:
        report = build_h_report(*write_h_run(tmp_path))
        assert report["final_leg"]["stages"] == [{"seconds": 2.0, "stage": "training_readback"}]
        assert report["final_leg"]["admissions"] == [
            {
                "caller": "completion",
                "evaluations": 2,
                "journals_read": 1,
                "seals_walked": 2,
                "seconds": 3.0,
            }
        ]

    def test_future_nll_is_reported_as_captured(self, tmp_path: Path) -> None:
        report = build_h_report(*write_h_run(tmp_path))
        nll = report["future_nll_per_dimension"]
        assert nll["null_rows"] == 2
        assert nll["negative_rows"] == 2
        assert nll["min"] == pytest.approx(-0.6)

    def test_json_is_deterministic_and_strict(self, tmp_path: Path) -> None:
        paths = write_h_run(tmp_path)
        first = json.dumps(build_h_report(*paths), sort_keys=True, allow_nan=False)
        second = json.dumps(build_h_report(*paths), sort_keys=True, allow_nan=False)
        assert first == second

    def test_a_failed_run_is_not_presented_as_complete(self, tmp_path: Path) -> None:
        report = build_h_report(*write_h_run(tmp_path, status="failed"))
        assert report["status"]["completed"] is False
        assert report["status"]["status"] == "failed"

    def test_a_receipt_verdict_is_embedded_verbatim(self, tmp_path: Path) -> None:
        verdict = {"status": "complete", "scientific_authority": "none"}
        report = build_h_report(*write_h_run(tmp_path), receipt_verdict=verdict)
        assert report["receipt_verdict"] == verdict
        assert "receipt_verdict" not in build_h_report(*write_h_run(tmp_path / "n"))

    def test_an_inadmissible_run_raises_and_names_the_file(self, tmp_path: Path) -> None:
        def drop_run_end(rows: list[dict[str, Any]]) -> None:
            rows[:] = [r for r in rows if r.get("event") != "run_end"]

        metrics, log = write_h_run(tmp_path, mutate_metrics=drop_run_end)
        with pytest.raises(TelemetryGap) as excinfo:
            build_h_report(metrics, log)
        assert str(metrics) in str(excinfo.value)

    def test_the_report_reads_the_captured_file_never_mlflow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load-bearing: a dummy tracking URI is set AND the client builder raises, so a
        report that consulted MLflow would fail here rather than pass vacuously."""
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")

        def boom(*_: Any, **__: Any) -> Any:
            raise AssertionError("the report consulted MLflow")

        monkeypatch.setattr(MLFLOW_BUILDER, boom)
        paths = write_h_run(tmp_path)
        assert build_h_report(*paths)["lineage"]["profile_identity"] == IDENTITY_A


@pytest.mark.integration
class TestReportOnTheCapturedRuns:
    @pytest.mark.parametrize("which", [0, 1])
    def test_real_run(self, which: int) -> None:
        pairs, labels = fixture_arguments()
        metrics, log = pairs[which]
        report = build_h_report(metrics, log)
        assert report["lineage"]["profile_identity"] in labels
        # the 8-hex fragment in the capture's file name is a prefix of the 32-hex run id the
        # worker log names; the report carries the full id, never the fragment
        fragment = metrics.name.split("-")[-2]
        assert len(report["lineage"]["mlflow_run_id"]) == 32
        assert report["lineage"]["mlflow_run_id"].startswith(fragment)
        assert report["steps"]["range"] == [1, 1000]
        assert report["steps"]["warmup_steps"] == 250
        assert report["evaluation"]["steps"] == [500, 750, 1000]
        assert report["checkpoints"]["seal_steps"] == [250, 500, 750, 1000]
        assert sorted(report["checkpoints"]["publication_seconds"], key=int) == [
            "250",
            "500",
            "750",
            "1000",
        ]
        stages = {s["stage"] for s in report["final_leg"]["stages"]}
        assert {
            "materialization",
            "runtime_admission",
            "training_readback",
            "generation",
            "span_diagnostics",
            "trace_capture",
            "native_verification",
            "result_admission",
        } == stages
        callers = {a["caller"] for a in report["final_leg"]["admissions"]}
        assert "completion" in callers
        assert report["future_nll_per_dimension"]["min"] < 0
