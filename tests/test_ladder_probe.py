"""Checkpoint probes use the actual observation-only model path and bounded windows."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from entropic_substrate.analysis.ladder_probe import ProbeSource, run_checkpoint_probe
from entropic_substrate.analysis.predictive_probe import ProbeConfig
from entropic_substrate.d_ladder_cli import tiny_rung_spec
from entropic_substrate.d_ladder_execution import SmokeRequest, run_local_smoke
from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.predictive_state.ladder_runtime import LadderPredictiveSystem

pytestmark = pytest.mark.unit


def source(root: Path, split: str) -> ProbeSource:
    values = bytes((index * 73 + 19) % 256 for index in range(256))
    (root / split).write_bytes(values)
    return ProbeSource(
        root,
        split,
        ContentHash("sha256", hashlib.sha256(values).hexdigest()),
        split,
        split,
        "corpus",
        0 if split == "train" else 1000,
    )


def config() -> ProbeConfig:
    return ProbeConfig(lags=(2, 4), train_positions=8, eval_positions=4, updates=2, batch_size=4)


@pytest.mark.parametrize("experiment", ["d2-s", "d4-fixed", "d5-permuted"])
def test_checkpoint_probe_captures_real_states_and_report(
    experiment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = SmokeRequest(tiny_rung_spec(experiment, seed=3), max_steps=1)
    run = run_local_smoke(request, tmp_path / "run")
    train, heldout = source(tmp_path, "train"), source(tmp_path, "heldout")
    calls: list[int] = []
    original = LadderPredictiveSystem.encode_observations

    def encode(self: LadderPredictiveSystem, ids: torch.Tensor):
        calls.append(ids.shape[1])
        assert not self.training
        return original(self, ids)

    def forbidden(*args, **kwargs):
        raise AssertionError("probe must not invoke target-scoring forward")

    monkeypatch.setattr(LadderPredictiveSystem, "encode_observations", encode)
    monkeypatch.setattr(LadderPredictiveSystem, "forward", forbidden)
    capture = run_checkpoint_probe(
        request, Path(run["checkpoint"]), train, heldout, tmp_path / "probe", config()
    )
    receipt = json.loads(Path(capture.receipt.path).read_text())
    report = json.loads(Path(capture.report.path).read_text())
    assert receipt["scientific_authority"] == "none"
    assert receipt["diagnostic_only"] is True
    assert receipt["training_source"]["content_hash"] == train.content_hash.digest
    assert receipt["extraction_seconds"] >= 0
    assert receipt["extracted_online_positions"] == sum(calls)
    assert max(calls) <= request.spec.base.encoder_pair.online.block_size
    assert len(report["primary"]) == 8
    for pair in report["primary"]:
        before = pair["before"]
        expected = ((before["target_position"] - heldout.source_offset) * 73 + 19) % 256
        assert before["target_byte"] == expected
    assert (
        hashlib.sha256(Path(capture.weights.path).read_bytes()).hexdigest()
        == capture.weights.content_hash.digest
    )


def test_changed_source_and_checkpoint_fail_before_probe_publication(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=3), max_steps=1)
    run = run_local_smoke(request, tmp_path / "run")
    train, heldout = source(tmp_path, "train"), source(tmp_path, "heldout")
    (tmp_path / "train").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="digest"):
        run_checkpoint_probe(
            request, Path(run["checkpoint"]), train, heldout, tmp_path / "probe", config()
        )
    assert not (tmp_path / "probe" / "receipt.json").exists()


def test_source_symlink_and_wrong_split_refused(tmp_path: Path) -> None:
    original = source(tmp_path, "train")
    (tmp_path / "alias").symlink_to(tmp_path / "train")
    with pytest.raises(ValueError, match="symlink"):
        replace(original, relative_path="alias").read()
    with pytest.raises(ValueError, match="split"):
        replace(original, split="test")


def test_window_capacity_failure_cannot_change_lags(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=3), max_steps=1)
    run = run_local_smoke(request, tmp_path / "run")
    train, heldout = source(tmp_path, "train"), source(tmp_path, "heldout")
    with pytest.raises(ValueError, match="lags exceed"):
        run_checkpoint_probe(
            request,
            Path(run["checkpoint"]),
            train,
            heldout,
            tmp_path / "probe",
            replace(config(), lags=(128,)),
        )
    assert not (tmp_path / "probe" / "receipt.json").exists()
