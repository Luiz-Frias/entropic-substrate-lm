"""Real local tiny execution and checkpoint continuation contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from entropic_substrate.composition.wiring.d_ladder_cli import tiny_rung_spec
from entropic_substrate.historical.d_ladder.d_ladder_execution import SmokeRequest, run_local_smoke

pytestmark = pytest.mark.unit


def test_smoke_continuation_matches_uninterrupted_training(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-s", seed=7), max_steps=2)
    full = run_local_smoke(request, tmp_path / "full")
    first = run_local_smoke(request, tmp_path / "split", until_step=1)
    resumed = run_local_smoke(request, tmp_path / "split", resume=Path(first["checkpoint"]))
    left = torch.load(full["checkpoint"], weights_only=True)
    right = torch.load(resumed["checkpoint"], weights_only=True)
    for name, value in left["model"].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, right["model"][name], rtol=0, atol=0)
    event = json.loads((tmp_path / "split" / "step-000002.json").read_text())
    assert event["run_class"] == "smoke"
    assert event["metrics"]["language_nats"] > 0
    assert event["metrics"]["online_numerical_rank"] >= 0
    assert event["train_online_positions"] == 32
    assert event["scientific_authority"] == "none"


@pytest.mark.parametrize(
    "experiment", ["d2-g", "d3-h64", "d3-composite", "d4-fixed", "d4-learned", "d5-permuted"]
)
def test_all_rungs_run_their_real_mechanism(experiment: str, tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec(experiment, seed=7), max_steps=1)
    result = run_local_smoke(request, tmp_path / experiment)
    assert result["step"] == 1
    manifest = json.loads((tmp_path / experiment / "manifest.json").read_text())
    assert manifest["construction"]["rung"] == experiment[:2]
    assert manifest["dataset"] == "synthetic-local-smoke-v1"


def test_resume_refuses_changed_identity_or_checkpoint_corruption(tmp_path: Path) -> None:
    request = SmokeRequest(tiny_rung_spec("d2-l", seed=7), max_steps=2)
    first = run_local_smoke(request, tmp_path, until_step=1)
    with pytest.raises(ValueError, match="manifest"):
        run_local_smoke(replace(request, max_steps=3), tmp_path, resume=Path(first["checkpoint"]))
    with Path(first["checkpoint"]).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="digest"):
        run_local_smoke(request, tmp_path, resume=Path(first["checkpoint"]))


def test_smoke_request_refuses_invalid_budget_and_fresh_overwrite(tmp_path: Path) -> None:
    spec = tiny_rung_spec("d2-l", seed=7)
    with pytest.raises(ValueError, match="positive integer"):
        SmokeRequest(spec, max_steps=True)
    request = SmokeRequest(spec, max_steps=1)
    run_local_smoke(request, tmp_path)
    with pytest.raises(FileExistsError):
        run_local_smoke(request, tmp_path)


def test_scientific_execution_refuses_missing_resources_before_touching_the_request(
    tmp_path: Path,
) -> None:
    """``resources is None`` stays load-bearing: scientific mode must name its meter.

    The request is deliberately uninitialised: the guard has to fire before any
    attribute of it is read, so no fixture can mask a regression that moved the
    check later (or replaced ``None`` with a stand-in meter).
    """
    from entropic_substrate.historical.d_ladder.d_ladder_execution import run_ladder
    from entropic_substrate.historical.d_ladder.d_ladder_request import ScientificRequest

    request = ScientificRequest.__new__(ScientificRequest)
    with pytest.raises(TypeError, match="scientific execution requires explicit LadderResources"):
        run_ladder(request, tmp_path, resources=None)
    assert not any(tmp_path.iterdir())
