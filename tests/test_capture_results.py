"""Contracts for the retrieval-side results capture (`capture-results`).

Pods run credential-free (operator ruling 2026-08-17): they upload the
authoritative results JSONL to the tracking server as a run artifact, and the
capture module — running on an operator machine that holds R2 credentials —
pulls those artifacts down, validates admissibility through telemetry.py, and
hands the set to DVC. Measurement path: inadmissible telemetry raises, it is
never captured silently.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from entropic_substrate.kernel.runtime.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointFormatError,
)
from entropic_substrate.ladder.capture.reader.capture_results import (
    capture_checkpoints,
    capture_finished_runs,
    dvc_commands,
)
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

pytestmark = pytest.mark.unit

ADMISSIBLE = json.dumps({"event": "eval", "step": 200, "nats_per_byte": 0.77}) + "\n"
INADMISSIBLE = json.dumps({"event": "train", "step": 10}) + "\n"


def _checkpoint_bytes(*, step: int = 100, seed: int = 0, with_optimizer: bool = True) -> bytes:
    """A minimal, VALID (format_version-stamped, no tau) checkpoint blob —
    the same shape :meth:`entropic_substrate.train.Trainer.state_dict` writes."""
    state: dict[str, object] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model": {"wte.weight": torch.randn(4, 3)},
        "config": {
            "name": "pytest-ckpt",
            "model": {},
            "data": {},
            "train": {"seed": seed},
        },
    }
    if with_optimizer:
        state["optimizer"] = {"state": {}, "param_groups": []}
        state["sampler"] = {"generator": torch.get_rng_state(), "eval_cursor": 0}
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


@dataclass
class _Info:
    run_id: str
    run_name: str
    status: str


@dataclass
class _Data:
    """The slice of ``mlflow.entities.RunData`` the capture partition reads."""

    tags: dict[str, object] = field(default_factory=dict)


@dataclass
class _Run:
    info: _Info
    data: _Data = field(default_factory=_Data)


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
    # str payloads write as text (results JSONLs); bytes payloads write raw
    # (checkpoint .pt blobs) — capture_checkpoints' downloads are binary, so
    # a single str-only payload type could not represent both capture lanes'
    # fixtures in one fake.
    payloads: dict[tuple[str, str], str | bytes]
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
        payload = self.payloads[(run_id, path)]
        if isinstance(payload, bytes):
            target.write_bytes(payload)
        else:
            target.write_text(payload, encoding="utf-8")
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


# ------------------------------------------------- H lane (FOR-1639, FOR-1641)
H_SCHEMA_TAG = "h-probe-worker-result-v1"
H_RUN = "h-worker-h0-fisher"
LIFECYCLE = "h-run-lifecycle-v1"


class _EPredicateApplied(Exception):
    """Distinct sentinel: ``final_eval_record`` was applied to an H artifact."""


def _h_lines(*, with_end: bool = True, with_step: bool = True) -> str:
    rows: list[dict[str, object]] = [
        {"schema": LIFECYCLE, "event": "run_start", "request_sha256": "a" * 64}
    ]
    if with_step:
        rows.append(
            {
                "schema": "h-training-step-v2",
                "status": "committed",
                "phase": "prediction",
                "step": 1,
            }
        )
    if with_end:
        rows.append(
            {
                "schema": LIFECYCLE,
                "event": "run_end",
                "status": "ok",
                "wall_seconds": 1.0,
                "successful_optimizer_steps": 1,
            }
        )
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


H_WORKER_LOG = json.dumps({"event": "worker.start", "level": "INFO"}) + "\n"


def _h_run(
    run_id: str = "run-h-0004", *, status: str = "FINISHED", tag: object = H_SCHEMA_TAG
) -> _Run:
    return _Run(
        _Info(run_id=run_id, run_name=H_RUN, status=status),
        _Data(tags={"schema": tag}),
    )


def _h_client(
    *,
    metrics: str | None = None,
    worker_log: str = H_WORKER_LOG,
    artifacts: list[str] | None = None,
    extra_runs: list[_Run] | None = None,
) -> FakeCaptureClient:
    """An experiment holding one FINISHED H worker run (and optionally more)."""
    metrics = _h_lines() if metrics is None else metrics
    listed = (
        artifacts
        if artifacts is not None
        else [
            "results/metrics.jsonl",
            "results/worker-log.jsonl",
        ]
    )
    return FakeCaptureClient(
        experiments={"entropic-substrate-lm": "exp-1"},
        runs=[_h_run(), *(extra_runs or [])],
        artifacts={"run-h-0004": listed},
        payloads={
            ("run-h-0004", "results/metrics.jsonl"): metrics,
            ("run-h-0004", "results/worker-log.jsonl"): worker_log,
        },
    )


@pytest.fixture
def e_predicate_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``final_eval_record`` raise a distinct sentinel on any H artifact.

    Without this, an H file wrongly routed to the E predicate raises a plain
    ``TelemetryGap`` ("no eval record") that a loose assertion cannot tell
    apart from the H predicate's own refusal.
    """
    from entropic_substrate.ladder.capture.reader import capture_results as capture_mod

    real = capture_mod.final_eval_record

    def guarded(path: str | Path) -> dict[str, object]:
        if H_RUN in Path(path).name:
            raise _EPredicateApplied(str(path))
        return real(path)

    monkeypatch.setattr(capture_mod, "final_eval_record", guarded)


def test_h_run_is_captured_under_run_scoped_names_via_the_h_predicate(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    client = _h_client()
    captured = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert sorted(p.name for p in captured) == [
        f"{H_RUN}-run-h-00-metrics.jsonl",
        f"{H_RUN}-run-h-00-worker-log.jsonl",
    ]
    assert (tmp_path / "cap" / f"{H_RUN}-run-h-00-metrics.jsonl").read_text() == _h_lines()


def test_h_run_without_run_end_raises_the_h_predicates_refusal(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    client = _h_client(metrics=_h_lines(with_end=False))
    with pytest.raises(TelemetryGap, match=r"metrics\.jsonl.*exactly one run_end.*found 0"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    # evidence stays on disk, named in the exception (same idiom as the E lane)
    assert (tmp_path / "cap" / f"{H_RUN}-run-h-00-metrics.jsonl").exists()


def test_h_run_with_a_malformed_worker_log_raises(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    client = _h_client(worker_log=H_WORKER_LOG + "this is not json\n")
    with pytest.raises(TelemetryGap, match=r"worker-log\.jsonl.*line 2.*not valid JSON"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")


def test_h_run_without_a_metrics_artifact_raises_instead_of_capturing_the_log_alone(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    """A worker that died before its first sink.log leaves only worker-log.jsonl;
    MLflow still marks the run FINISHED. Capturing that alone would read as
    success, so the run is refused before anything is downloaded."""
    client = _h_client(artifacts=["results/worker-log.jsonl"])
    with pytest.raises(TelemetryGap, match=r"run-h-0004.*results/metrics\.jsonl"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert client.download_calls == []


def test_h_run_with_a_duplicated_metrics_artifact_raises(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    client = _h_client(
        artifacts=["results/metrics.jsonl", "results/metrics.jsonl", "results/worker-log.jsonl"]
    )
    with pytest.raises(TelemetryGap, match=r"run-h-0004.*results/metrics\.jsonl"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert client.download_calls == []


def test_a_refused_h_file_kept_on_disk_is_refused_again_on_rerun(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    """The refused file stays on disk as evidence; a second invocation must not
    skip it as 'already captured' and report a clean nothing-to-do."""
    client = _h_client(metrics=_h_lines(with_end=False))
    for _ in range(2):
        with pytest.raises(TelemetryGap, match=r"exactly one run_end"):
            capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert len(client.download_calls) == 1  # re-validated from disk, not re-downloaded


def test_h_capture_is_idempotent_once_admitted(tmp_path: Path, e_predicate_tripwire: None) -> None:
    client = _h_client()
    first = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    again = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert len(first) == 2 and again == []
    assert len(client.download_calls) == 2


def test_e_run_beside_an_h_run_is_still_captured_through_the_e_predicate(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    e_run = _Run(_Info(run_id="run-full-0001", run_name="e1-fisher", status="FINISHED"))
    client = _h_client(extra_runs=[e_run])
    client.artifacts["run-full-0001"] = ["results/e1-fisher.jsonl"]
    client.payloads[("run-full-0001", "results/e1-fisher.jsonl")] = ADMISSIBLE
    captured = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert sorted(p.name for p in captured) == [
        "e1-fisher-run-full-e1-fisher.jsonl",
        f"{H_RUN}-run-h-00-metrics.jsonl",
        f"{H_RUN}-run-h-00-worker-log.jsonl",
    ]


def test_h_runs_still_obey_the_finished_gate(tmp_path: Path, e_predicate_tripwire: None) -> None:
    # A distinct 8-char prefix: run-h-0005 would share run-h-00 with run-h-0004,
    # so its target file would already exist and the download would be skipped.
    client = _h_client()
    running = "bbbbbbbb-0005"
    client.runs.append(_h_run(running, status="RUNNING"))
    client.artifacts[running] = ["results/metrics.jsonl"]
    client.payloads[(running, "results/metrics.jsonl")] = _h_lines()
    capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert {run_id for run_id, _ in client.download_calls} == {"run-h-0004"}


@pytest.mark.parametrize("tag", ["e1-result", "", "H-upper", "hello", "h", "hx-1"])
def test_a_non_h_schema_tag_stays_on_the_e_lane(tmp_path: Path, tag: str) -> None:
    run = _Run(
        _Info(run_id="run-full-0001", run_name="e1-fisher", status="FINISHED"),
        _Data(tags={"schema": tag}),
    )
    client = _client()
    client.runs[0] = run
    captured = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert [p.name for p in captured] == ["e1-fisher-run-full-e1-fisher.jsonl"]


def test_an_h_tiny_trace_run_without_lifecycle_records_is_refused_on_the_h_lane(
    tmp_path: Path, e_predicate_tripwire: None
) -> None:
    """Ruling 7: the h- prefix stands, so the local tracer's h-tiny-trace-v1 runs
    take the H lane and are refused for carrying no lifecycle records."""
    step_only = json.dumps(
        {"schema": "h-training-step-v2", "status": "committed", "phase": "prediction", "step": 1},
        sort_keys=True,
    )
    client = _h_client(metrics=step_only + "\n")
    client.runs[0] = _h_run(tag="h-tiny-trace-v1")
    with pytest.raises(TelemetryGap, match=r"metrics\.jsonl.*exactly one run_start.*found 0"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert client.download_calls == [("run-h-0004", "results/metrics.jsonl")]


def test_a_run_object_without_data_is_the_e_lane(tmp_path: Path) -> None:
    @dataclass
    class _BareRun:
        info: _Info

    client = _client()
    client.runs[0] = _BareRun(
        _Info(run_id="run-full-0001", run_name="e1-fisher", status="FINISHED")
    )  # type: ignore[assignment]
    captured = capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert len(captured) == 1


def test_a_non_string_schema_tag_raises_naming_the_run(tmp_path: Path) -> None:
    client = _h_client()
    client.runs[0] = _h_run(tag=7)
    with pytest.raises(TypeError, match=r"run-h-0004.*schema"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "cap")
    assert client.download_calls == []


def test_dvc_commands_add_then_push_the_capture_dir(tmp_path: Path) -> None:
    dest = tmp_path / "cap"
    assert dvc_commands(dest) == [["dvc", "add", str(dest)], ["dvc", "push"]]


# ------------------------------------------------------------- checkpoints
def _ckpt_client(payload: bytes | None = None) -> FakeCaptureClient:
    payload = payload if payload is not None else _checkpoint_bytes(seed=6)
    return FakeCaptureClient(
        experiments={"entropic-substrate-lm": "exp-1"},
        runs=[
            _Run(_Info(run_id="run-full-0001", run_name="e1-fisher", status="FINISHED")),
            _Run(_Info(run_id="run-tiny-0002", run_name="e1-fisher-tiny", status="FINISHED")),
            _Run(_Info(run_id="run-live-0003", run_name="e1-fisher", status="RUNNING")),
        ],
        artifacts={
            # -best.pt lives under the SAME checkpoints/ prefix as -final.pt
            # — only -final.pt is ever captured (see capture_checkpoints'
            # own docstring), so its presence here is itself a guard-tooth:
            # a filter bug that captured both would surface as len==2.
            "run-full-0001": ["checkpoints/e1-fisher-best.pt", "checkpoints/e1-fisher-final.pt"],
            "run-tiny-0002": ["checkpoints/e1-fisher-tiny-final.pt"],
            "run-live-0003": ["checkpoints/e1-fisher-final.pt"],
        },
        payloads={
            ("run-full-0001", "checkpoints/e1-fisher-final.pt"): payload,
            ("run-tiny-0002", "checkpoints/e1-fisher-tiny-final.pt"): _checkpoint_bytes(seed=7),
        },
    )


def test_capture_checkpoints_downloads_only_final_finished_nontiny(tmp_path: Path) -> None:
    client = _ckpt_client()
    captured = capture_checkpoints(client, ["entropic-substrate-lm"], tmp_path / "ckpts")
    assert [p.name for p in captured] == ["e1-fisher-run-full-e1-fisher-final.pt"]
    # RUNNING and -tiny runs, and the -best.pt twin, were never downloaded.
    assert client.download_calls == [("run-full-0001", "checkpoints/e1-fisher-final.pt")]


def test_capture_checkpoints_include_tiny_opt_in(tmp_path: Path) -> None:
    client = _ckpt_client()
    captured = capture_checkpoints(
        client, ["entropic-substrate-lm"], tmp_path / "ckpts", include_tiny=True
    )
    assert sorted(p.name for p in captured) == [
        "e1-fisher-run-full-e1-fisher-final.pt",
        "e1-fisher-tiny-run-tiny-e1-fisher-tiny-final.pt",
    ]
    # The -best.pt twin and the RUNNING run stay excluded on this arm too.
    assert sorted(client.download_calls) == [
        ("run-full-0001", "checkpoints/e1-fisher-final.pt"),
        ("run-tiny-0002", "checkpoints/e1-fisher-tiny-final.pt"),
    ]


def test_capture_checkpoints_strips_optimizer_by_default(tmp_path: Path) -> None:
    original = _checkpoint_bytes(seed=6, with_optimizer=True)
    expected = torch.load(io.BytesIO(original), map_location="cpu", weights_only=True)
    client = _ckpt_client(payload=original)
    (captured,) = capture_checkpoints(client, ["entropic-substrate-lm"], tmp_path / "ckpts")
    reloaded = torch.load(captured, map_location="cpu", weights_only=True)
    assert set(reloaded) == {"format_version", "step", "model", "config"}
    assert "optimizer" not in reloaded
    assert "sampler" not in reloaded
    # Stripping drops resume machinery ONLY — the trained weights and the
    # step they belong to must survive the re-save byte-for-byte, on CPU.
    assert reloaded["step"] == expected["step"]
    torch.testing.assert_close(reloaded["model"]["wte.weight"], expected["model"]["wte.weight"])
    assert reloaded["model"]["wte.weight"].device.type == "cpu"


def test_capture_checkpoints_strip_write_failure_leaves_no_corrupt_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strip-resave that fails PARTWAY must not leave ``target`` — the
    idempotency-checked final path — holding truncated bytes: a future
    invocation's ``target.exists()`` check would otherwise treat that
    corrupt file as already-captured forever, and re-running the capture
    would never repair it.

    RED against the pre-fix ``torch.save(stripped, target)`` (writes
    straight to the idempotency-checked path — a failure partway through
    truncates and corrupts it in place); the atomic tmp-then-``os.replace``
    fix writes the stripped payload to a sibling ``.partial`` file first, so
    a failure there never touches ``target`` at all and the ORIGINAL
    (unstripped, still fully valid) download survives.
    """
    client = _ckpt_client()
    dest = tmp_path / "ckpts"
    real_save = torch.save

    def _flaky_save(obj: object, f: object, *args: object, **kwargs: object) -> None:
        Path(str(f)).write_bytes(b"PARTIAL-GARBAGE-FROM-INTERRUPTED-WRITE")
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", _flaky_save)
    with pytest.raises(RuntimeError, match="simulated interrupted write"):
        capture_checkpoints(client, ["entropic-substrate-lm"], dest)
    monkeypatch.setattr(torch, "save", real_save)

    (target,) = dest.glob("*-final.pt")
    # target must still be the intact ORIGINAL download, loadable and
    # carrying its optimizer state — the strip attempt never completed.
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    assert "optimizer" in reloaded
    # Other arm: the failed strip must leave no residue either. `dest` is
    # the directory cmd_capture_results hands to `dvc add`, so a surviving
    # .partial sibling would be committed as tracked content.
    assert list(dest.glob("*.partial")) == []


def test_capture_checkpoints_missing_required_key_raises_named_error(tmp_path: Path) -> None:
    """A checkpoint dict that normalizes without raising (format_version OK,
    no tau) but is missing "step"/"model"/"config" is not a shape
    Trainer.state_dict() ever writes — must raise the repo's own NAMED
    CheckpointFormatError, not a bare unhelpful KeyError."""
    buf = io.BytesIO()
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "model": {}}, buf)
    client = _ckpt_client(payload=buf.getvalue())
    with pytest.raises(CheckpointFormatError, match="missing required key"):
        capture_checkpoints(client, ["entropic-substrate-lm"], tmp_path / "ckpts")


def test_capture_checkpoints_keeps_optimizer_when_strip_optimizer_is_false(tmp_path: Path) -> None:
    client = _ckpt_client(payload=_checkpoint_bytes(seed=6, with_optimizer=True))
    (captured,) = capture_checkpoints(
        client, ["entropic-substrate-lm"], tmp_path / "ckpts", strip_optimizer=False
    )
    reloaded = torch.load(captured, map_location="cpu", weights_only=True)
    assert "optimizer" in reloaded
    assert "sampler" in reloaded


def test_capture_checkpoints_is_idempotent_across_invocations(tmp_path: Path) -> None:
    client = _ckpt_client()
    dest = tmp_path / "ckpts"
    first = capture_checkpoints(client, ["entropic-substrate-lm"], dest)
    again = capture_checkpoints(client, ["entropic-substrate-lm"], dest)
    assert len(first) == 1
    assert again == []
    assert len(client.download_calls) == 1


def test_capture_checkpoints_malformed_bytes_raise_checkpoint_format_error_loud(
    tmp_path: Path,
) -> None:
    client = _ckpt_client(payload=b"not a torch checkpoint, just garbage bytes")
    dest = tmp_path / "ckpts"
    with pytest.raises(CheckpointFormatError):
        capture_checkpoints(client, ["entropic-substrate-lm"], dest)
    # kept on disk as evidence, same idiom as the results lane's TelemetryGap
    (kept,) = dest.glob("*-final.pt")
    assert kept.exists()


def test_capture_checkpoints_unknown_experiment_name_fails_loud(tmp_path: Path) -> None:
    client = _ckpt_client()
    with pytest.raises(LookupError):
        capture_checkpoints(client, ["no-such-experiment"], tmp_path / "ckpts")


# ------------------------------------------------------------- CLI wiring
def test_cli_capture_results_unknown_experiment_is_a_controlled_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CodeRabbit finding (FOR-909 P4 follow-up): cmd_capture_results called
    capture_finished_runs/capture_checkpoints with no try/except around
    them at all — an unknown experiment name (LookupError, raised by their
    shared _resolve_experiment_ids) propagated straight past main()'s own
    handler (which only catches CheckpointFormatError) as an unhandled
    traceback, instead of the same controlled exit-2 refusal every other
    named-input mistake in this command already gets (e.g. the missing-
    tracking-URI check just above it)."""
    import mlflow.tracking

    from entropic_substrate.composition.entrypoints.cli import main
    from entropic_substrate.ladder.capture.reader.sinks import MLFLOW_TRACKING_ENV

    class _FakeMlflowClient:
        def __init__(self, *, tracking_uri: str) -> None:
            del tracking_uri

        def get_experiment_by_name(self, name: str) -> None:
            del name
            return None

    monkeypatch.setattr(mlflow.tracking, "MlflowClient", _FakeMlflowClient)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake-tracking-server.invalid")

    code = main(
        [
            "capture-results",
            "--dest",
            str(tmp_path / "cap"),
            "--experiment",
            "no-such-experiment",
        ]
    )
    assert code == 2
    assert "not found on the tracking server" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("lane", "reason"),
    [("e", "no eval record"), ("h", "exactly one run_end")],
)
def test_cli_capture_results_telemetry_gap_is_a_controlled_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lane: str,
    reason: str,
) -> None:
    """FOR-1639: an inadmissible results file (either lane) makes the capture
    lane raise TelemetryGap; cmd_capture_results must turn it into the same
    controlled exit-2 refusal as an unknown experiment -- the file named on
    stderr, no traceback -- instead of letting it escape main()."""
    import mlflow.tracking

    from entropic_substrate.composition.entrypoints.cli import main
    from entropic_substrate.ladder.capture.reader.sinks import MLFLOW_TRACKING_ENV

    fake = (
        _client(payload=INADMISSIBLE)
        if lane == "e"
        else _h_client(metrics=_h_lines(with_end=False))
    )
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *, tracking_uri: fake)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake-tracking-server.invalid")

    dest = tmp_path / "cap"
    code = main(["capture-results", "--dest", str(dest)])

    err = capsys.readouterr().err
    assert code == 2
    assert reason in err
    assert ".jsonl" in err and str(dest) in err
    assert "Traceback" not in err


@pytest.mark.parametrize("hostile", ["../escape", "nested/inner"])
def test_capture_checkpoints_refuses_path_escaping_run_name(tmp_path: Path, hostile: str) -> None:
    """``run_name`` is REMOTE data interpolated into a filesystem path:
    ``dest / name`` silently REPLACES dest for an absolute name and ``..``
    segments traverse out of it — a rename on the MLflow server must never
    steer where capture writes. Refused loudly, never skipped silently."""
    client = FakeCaptureClient(
        experiments={"entropic-substrate-lm": "exp-1"},
        runs=[_Run(_Info(run_id="run-evil-0001", run_name=hostile, status="FINISHED"))],
        artifacts={},
        payloads={},
    )
    with pytest.raises(ValueError, match="not a safe filename component"):
        capture_checkpoints(client, ["entropic-substrate-lm"], tmp_path / "ckpts")


@pytest.mark.parametrize("hostile", ["../escape", "nested/inner"])
def test_capture_finished_runs_refuses_path_escaping_run_name(tmp_path: Path, hostile: str) -> None:
    """The results lane builds its target from the same remote field — the
    guard binds both capture lanes, not only the new checkpoint one."""
    client = FakeCaptureClient(
        experiments={"entropic-substrate-lm": "exp-1"},
        runs=[_Run(_Info(run_id="run-evil-0001", run_name=hostile, status="FINISHED"))],
        artifacts={},
        payloads={},
    )
    with pytest.raises(ValueError, match="not a safe filename component"):
        capture_finished_runs(client, ["entropic-substrate-lm"], tmp_path / "results")


def test_capture_checkpoints_non_mapping_payload_raises_named_error(tmp_path: Path) -> None:
    """torch.load can succeed on a payload that is not a checkpoint mapping
    at all (a bare list saves and loads fine) — that must surface as the
    repo's NAMED CheckpointFormatError, not an AttributeError from deep
    inside normalize_checkpoint."""
    buf = io.BytesIO()
    torch.save([1, 2, 3], buf)
    client = _ckpt_client(payload=buf.getvalue())
    with pytest.raises(CheckpointFormatError, match="not a mapping"):
        capture_checkpoints(client, ["entropic-substrate-lm"], tmp_path / "ckpts")


# ---------------------------------------------------------- cloud-sync gate
@dataclass
class _FakeDvcProc:
    stdout: str
    stderr: str = ""


def test_verify_cloud_sync_accepts_an_empty_status(tmp_path: Path) -> None:
    from entropic_substrate.ladder.capture.reader.capture_results import verify_cloud_sync

    dest = tmp_path / "captured"
    Path(f"{dest}.dvc").write_text("outs: []\n", encoding="utf-8")
    calls: list[list[str]] = []

    def runner(argv: list[str], **_: object) -> _FakeDvcProc:
        calls.append(argv)
        return _FakeDvcProc(stdout="{}")

    verify_cloud_sync(dest, runner=runner)
    assert calls == [["dvc", "status", "--cloud", "--json", f"{dest}.dvc"]]


def test_verify_cloud_sync_raises_on_out_of_sync_remote(tmp_path: Path) -> None:
    from entropic_substrate.ladder.capture.reader.capture_results import verify_cloud_sync

    dest = tmp_path / "captured"
    Path(f"{dest}.dvc").write_text("outs: []\n", encoding="utf-8")

    def runner(argv: list[str], **_: object) -> _FakeDvcProc:
        return _FakeDvcProc(stdout=json.dumps({f"{dest}.dvc": [{"changed outs": {"x": "new"}}]}))

    with pytest.raises(RuntimeError, match="not safe to commit"):
        verify_cloud_sync(dest, runner=runner)


def test_verify_cloud_sync_raises_on_unparseable_dvc_output(tmp_path: Path) -> None:
    from entropic_substrate.ladder.capture.reader.capture_results import verify_cloud_sync

    dest = tmp_path / "captured"
    Path(f"{dest}.dvc").write_text("outs: []\n", encoding="utf-8")

    def runner(argv: list[str], **_: object) -> _FakeDvcProc:
        return _FakeDvcProc(stdout="ERROR: unexpected error", stderr="boom")

    with pytest.raises(RuntimeError, match="unparseable"):
        verify_cloud_sync(dest, runner=runner)


def test_verify_cloud_sync_requires_the_pointer_file(tmp_path: Path) -> None:
    from entropic_substrate.ladder.capture.reader.capture_results import verify_cloud_sync

    with pytest.raises(RuntimeError, match="\\.dvc"):
        verify_cloud_sync(tmp_path / "never-added", runner=lambda *a, **k: _FakeDvcProc("{}"))


def test_cli_success_path_runs_the_cloud_gate_after_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import mlflow.tracking

    import entropic_substrate.ladder.capture.reader.capture_results as capture_mod
    from entropic_substrate.composition.entrypoints.cli import main
    from entropic_substrate.ladder.capture.reader.sinks import MLFLOW_TRACKING_ENV

    fake = _client()
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda *, tracking_uri: fake)
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake-tracking-server.invalid")

    events: list[str] = []
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_: events.append(" ".join(argv[:2])) or None
    )
    monkeypatch.setattr(
        capture_mod, "verify_cloud_sync", lambda dest, **_: events.append(f"verify {dest}")
    )

    dest = tmp_path / "cap"
    code = main(["capture-results", "--dest", str(dest)])
    assert code == 0
    # the affirmative remote check runs AFTER add+push, before success
    assert events == ["dvc add", "dvc push", f"verify {dest}"]
