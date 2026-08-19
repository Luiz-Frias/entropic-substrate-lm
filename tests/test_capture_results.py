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

from entropic_substrate.capture_results import (
    capture_checkpoints,
    capture_finished_runs,
    dvc_commands,
)
from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION, CheckpointFormatError
from entropic_substrate.telemetry import TelemetryGap

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
