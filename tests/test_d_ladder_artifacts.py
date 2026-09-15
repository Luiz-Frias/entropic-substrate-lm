"""Immutable smoke artifacts remain tied to their model and execution request."""

import json
from pathlib import Path

import pytest
import torch

from entropic_substrate.d_ladder_artifacts import load_smoke_model, read_smoke_capture
from entropic_substrate.d_ladder_cli import tiny_rung_spec
from entropic_substrate.d_ladder_execution import SmokeRequest, run_local_smoke

pytestmark = pytest.mark.unit


def test_artifact_reader_loads_same_verified_model_bytes(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=1), max_steps=1)
    receipt = run_local_smoke(request, tmp_path)
    model = load_smoke_model(request, Path(receipt["checkpoint"]))
    saved = torch.load(receipt["checkpoint"], weights_only=True)
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    capture = read_smoke_capture(request, tmp_path)
    assert capture["run_class"] == "smoke"
    assert capture["scientific_authority"] == "none"
    assert len(capture["steps"]) == 1


def test_reader_rejects_changed_authority_and_symlink(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=1), max_steps=1)
    receipt = run_local_smoke(request, tmp_path)
    event_path = Path(receipt["checkpoint"]).with_suffix(".json")
    event = json.loads(event_path.read_text())
    event["run_class"] = "confirmatory"
    event_path.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="smoke"):
        read_smoke_capture(request, tmp_path)
    event["run_class"] = "smoke"
    event_path.write_text(json.dumps(event))
    checkpoint = Path(receipt["checkpoint"])
    original = checkpoint.with_name("original.pt")
    checkpoint.rename(original)
    checkpoint.symlink_to(original)
    with pytest.raises(ValueError, match=r"symlink|regular|opened"):
        load_smoke_model(request, checkpoint)


def test_reader_rejects_forged_measurements(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=1), max_steps=1)
    receipt = run_local_smoke(request, tmp_path)
    checkpoint = Path(receipt["checkpoint"])
    sidecar = checkpoint.with_suffix(".json")
    event = json.loads(sidecar.read_text())
    event["metrics"]["language_nats"] = 0.0
    sidecar.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="measurement digest"):
        load_smoke_model(request, checkpoint)
    with pytest.raises(ValueError, match="digest"):
        load_smoke_model(request, checkpoint, expected_sha256="0" * 64)
