"""Contracts for the retrieval-side results capture (`capture-results`).

Pods run credential-free (operator ruling 2026-08-17): they upload the
authoritative results JSONL to the tracking server as a run artifact, and the
capture module — running on an operator machine that holds R2 credentials —
pulls those artifacts down, validates admissibility through telemetry.py, and
hands the set to DVC. Measurement path: inadmissible telemetry raises, it is
never captured silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.capture_results import capture_finished_runs, dvc_commands
from entropic_substrate.telemetry import TelemetryGap

pytestmark = pytest.mark.unit

ADMISSIBLE = json.dumps({"event": "eval", "step": 200, "nats_per_byte": 0.77}) + "\n"
INADMISSIBLE = json.dumps({"event": "train", "step": 10}) + "\n"


@dataclass
class _Info:
    run_id: str
    run_name: str
    status: str


@dataclass
class _Run:
    info: _Info


@dataclass
class _FileInfo:
    path: str


@dataclass
class _Experiment:
    experiment_id: str


@dataclass
class FakeCaptureClient:
    """The slice of ``MlflowClient`` the capture path uses."""

    experiments: dict[str, str]
    runs: list[_Run]
    artifacts: dict[str, list[str]]
    payloads: dict[tuple[str, str], str]
    download_calls: list[tuple[str, str]] = field(default_factory=list)

    def get_experiment_by_name(self, name: str) -> _Experiment | None:
        if name not in self.experiments:
            return None
        return _Experiment(experiment_id=self.experiments[name])

    def search_runs(self, experiment_ids: list[str], max_results: int = 500) -> list[_Run]:
        return list(self.runs)

    def list_artifacts(self, run_id: str, path: str) -> list[_FileInfo]:
        return [_FileInfo(path=p) for p in self.artifacts.get(run_id, []) if p.startswith(path)]

    def download_artifacts(self, run_id: str, path: str, dst_path: str) -> str:
        self.download_calls.append((run_id, path))
        target = Path(dst_path) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.payloads[(run_id, path)], encoding="utf-8")
        return str(target)


def _client(payload: str = ADMISSIBLE) -> FakeCaptureClient:
    return FakeCaptureClient(
        experiments={"entropic-substrate-lm": "exp-1"},
        runs=[
            _Run(_Info(run_id="run-full-0001", run_name="e1-fisher", status="FINISHED")),
            _Run(_Info(run_id="run-tiny-0002", run_name="e1-fisher-tiny", status="FINISHED")),
            _Run(_Info(run_id="run-live-0003", run_name="e1-fisher", status="RUNNING")),
        ],
        artifacts={
            "run-full-0001": ["results/e1-fisher.jsonl"],
            "run-tiny-0002": ["results/e1-fisher-tiny.jsonl"],
            "run-live-0003": ["results/e1-fisher.jsonl"],
        },
        payloads={
            ("run-full-0001", "results/e1-fisher.jsonl"): payload,
            ("run-tiny-0002", "results/e1-fisher-tiny.jsonl"): ADMISSIBLE,
        },
    )


def test_captures_finished_nontiny_runs_with_run_scoped_names(tmp_path: Path) -> None:
    client = _client()
    captured = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert [p.name for p in captured] == ["e1-fisher-run-full-e1-fisher.jsonl"]
    assert captured[0].read_text(encoding="utf-8") == ADMISSIBLE
    # RUNNING and -tiny runs were never downloaded
    assert client.download_calls == [("run-full-0001", "results/e1-fisher.jsonl")]


def test_include_tiny_opt_in_captures_the_smoke_run_too(tmp_path: Path) -> None:
    client = _client()
    captured = capture_finished_runs(
        client, ["entropic-substrate-lm"], tmp_path / "cap", include_tiny=True
    )
    assert len(captured) == 2


def test_inadmissible_telemetry_raises_and_is_never_captured_silently(tmp_path: Path) -> None:
    client = _client(payload=INADMISSIBLE)
    with pytest.raises(TelemetryGap):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")


def test_capture_is_idempotent_across_invocations(tmp_path: Path) -> None:
    client = _client()
    first = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    again = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert len(first) == 1
    assert again == []
    assert len(client.download_calls) == 1


def test_unknown_experiment_name_fails_loud(tmp_path: Path) -> None:
    client = _client()
    with pytest.raises(LookupError):
        capture_finished_runs(client, ["no-such-experiment"], tmp_path / "cap")


def test_dvc_commands_add_then_push_the_capture_dir(tmp_path: Path) -> None:
    dest = tmp_path / "cap"
    assert dvc_commands(dest) == [["dvc", "add", str(dest)], ["dvc", "push"]]
