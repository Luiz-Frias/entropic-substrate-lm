"""The H0 final-probe checks over the real HEpP7g capture (integration).

``results/h0-plumbing-1k-v3-HEpP7g/`` holds, per arm, only ``metrics.jsonl`` and
``worker-log.jsonl``: no controller records, receipts or profile manifests. So K1, K2 and
K5 are not exercised here, and K6's expected k-set is passed explicitly as {2, 4} (the
pre-FOR-1664 image recorded no diagnostic k and its manifests were not captured).

HEpP7g is the NEGATIVE exemplar for K4: its five shared-memo final-leg stages re-read
evaluations from disk (pre-FOR-1670 image). That FAIL is pinned here as a passing test.
A missing fixture RAISES (never skips: a skip is a false green).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from entropic_substrate.composition.entrypoints.cli import main as cli_main
from entropic_substrate.ladder.analyze.h_final_probe_checks import (
    check_analysis_surface,
    check_capture,
    check_importance_curve,
    check_read_floor,
)

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "results" / "h0-plumbing-1k-v3-HEpP7g"
ARMS = {"a163161f": "magnitude", "988d8388": "sphere"}


def _runs() -> dict[str, tuple[Path, Path]]:
    if not FIXTURE_DIR.is_dir():
        raise AssertionError(
            f"{FIXTURE_DIR} is absent: run `dvc pull results/h0-plumbing-1k-v3-HEpP7g.dvc` "
            "(this test fails rather than skips: a skip is a false green)"
        )
    runs: dict[str, tuple[Path, Path]] = {}
    for metrics in sorted(FIXTURE_DIR.glob("*-metrics.jsonl")):
        log = metrics.with_name(metrics.name.replace("-metrics.jsonl", "-worker-log.jsonl"))
        runs[ARMS[metrics.name.split("-")[-2]]] = (metrics, log)
    assert sorted(runs) == ["magnitude", "sphere"], sorted(runs)
    return runs


def _worker_start_digest(log: Path) -> str:
    """The request digest from the worker log's own ``worker.start`` (an independent file
    from the metrics stream K3 binds)."""
    for line in log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "worker.start":
            return row["request_sha256"]
    raise AssertionError(f"{log} has no worker.start")


def test_k3_both_arms_are_admissible_and_bound_to_their_requests() -> None:
    runs = _runs()
    digests = [_worker_start_digest(log) for _, log in runs.values()]
    result = check_capture([metrics for metrics, _ in runs.values()], digests)
    assert result.verdict == "PASS", result.to_json()


def test_k4_fails_on_hepp7g_naming_the_five_shared_memo_disk_readers() -> None:
    runs = _runs()
    result = check_read_floor([log for _, log in runs.values()])
    assert result.verdict == "FAIL", result.to_json()
    expected = {
        "training_readback read 3",
        "generation read 3",
        "span_diagnostics read 6",
        "trace_capture read 9",
        "result_admission read 3",
    }
    for log_facts in result.facts["logs"]:
        shared = {
            f"{row['caller']} read {row['evaluations']}" for row in log_facts["shared_disk_reads"]
        }
        assert shared == expected, shared
        fresh = {(row["caller"], row["root"]) for row in log_facts["fresh_walks"]}
        assert fresh == {("completion", "run"), ("native_verification", "capture")}
    assert len(result.reasons) == 10, result.reasons


def test_k6_every_evaluation_row_carries_exactly_k_2_and_4_with_ess_within_k() -> None:
    runs = _runs()
    result = check_importance_curve([metrics for metrics, _ in runs.values()], (2, 4))
    assert result.verdict == "PASS", result.to_json()
    assert result.facts["rows"] == 6


def test_k6_the_current_development_k_set_is_refused_on_the_old_capture() -> None:
    runs = _runs()
    result = check_importance_curve([runs["magnitude"][0]], (2, 4, 8))
    assert result.verdict == "FAIL", result.to_json()


def test_k7_the_analysis_surface_admits_both_arms() -> None:
    runs = _runs()
    result = check_analysis_surface({label: [pair] for label, pair in runs.items()})
    assert result.verdict == "PASS", result.to_json()
    assert result.facts["paired_training_steps"] == 1000


def test_k7_a_truncated_copy_fails_naming_the_file(tmp_path: Path) -> None:
    runs = _runs()
    metrics, log = runs["magnitude"]
    copy = tmp_path / metrics.name
    data = metrics.read_bytes()
    copy.write_bytes(data[: len(data) // 2])
    shutil.copyfile(log, tmp_path / log.name)
    result = check_analysis_surface({"magnitude": [(copy, tmp_path / log.name)]})
    assert result.verdict == "FAIL", result.to_json()
    assert any(str(copy) in reason for reason in result.reasons), result.reasons


def test_k7_the_cli_handlers_exit_zero_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    runs = _runs()
    identities = {
        label: json.loads(metrics.read_text(encoding="utf-8").splitlines()[0])["profile_identity"]
        for label, (metrics, _) in runs.items()
    }
    aggregate = ["mlflow-aggregate", "h"]
    for label, (metrics, log) in runs.items():
        aggregate += ["--run", str(metrics), str(log), "--arm", f"{identities[label]}={label}"]
    assert cli_main(aggregate) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "h-aggregate-v1"
    metrics, log = runs["sphere"]
    assert cli_main(["eval", "h", "--run", str(metrics), str(log)]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "h-report-v1"
