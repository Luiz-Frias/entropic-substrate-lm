"""First-rung probes admit real trainer checkpoints and never execute targets."""

import hashlib
import json
from pathlib import Path

import pytest

from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.historical.d_ladder.d1_probe import run_d1_checkpoint_probe
from entropic_substrate.historical.d_ladder.ladder_probe import ProbeSource
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.register.d1_plan import create_d1_excluded_pilot_plan
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin

pytestmark = pytest.mark.unit


def source(root: Path, split: str) -> ProbeSource:
    payload = bytes(range(128))
    (root / split).write_bytes(payload)
    return ProbeSource(
        root,
        split,
        ContentHash("sha256", hashlib.sha256(payload).hexdigest()),
        split,
        split,
        "synthetic",
        0 if split == "train" else 1000,
    )


def checkpoint(tmp_path: Path, *, pilot: bool = False):
    plan = micro_plan(D1Arm.BOTH)
    if pilot:
        plan = create_d1_excluded_pilot_plan(
            plan.profile,
            logical_run_id="d1-pilot-both",
            seed=plan.seed,
            pairing_id="pilot-pair",
            evidence=plan.run.evidence,
        )
    completion = run_d1_fresh(
        plan=plan,
        start=run_start(plan),
        origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
        locations=D1FreshRunLocations(
            journal_path=tmp_path / "run.jsonl", checkpoint_path=tmp_path / "model.pt"
        ),
        clock=DeterministicTrainerClock(),
    )
    return plan, completion.final_checkpoint


@pytest.mark.parametrize("pilot", [False, True])
def test_d1_completed_checkpoint_reaches_shared_probe_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pilot: bool
) -> None:
    plan, saved = checkpoint(tmp_path, pilot=pilot)
    train, heldout = source(tmp_path, "train"), source(tmp_path, "heldout")

    def forbidden(*args, **kwargs):
        raise AssertionError("score diagnostic must not run D1 target forward")

    monkeypatch.setattr(D1PredictiveSystem, "forward", forbidden)
    config = ProbeConfig(lags=(2, 4), train_positions=8, eval_positions=4, updates=1, batch_size=4)
    capture = run_d1_checkpoint_probe(
        plan, saved.path, saved.sha256, train, heldout, tmp_path / "probe", config
    )
    receipt = json.loads(Path(capture.receipt.path).read_text())
    assert receipt["run_class"] == ("pilot" if pilot else "smoke")
    assert receipt["scientific_authority"] == "none"
    assert receipt["execution_digest"] == plan.run.execution_digest
    assert receipt["checkpoint_sha256"] == saved.sha256
    report = json.loads(Path(capture.report.path).read_text())
    assert len(report["primary"]) == 8
    for pair in report["primary"]:
        assert pair["before"]["binding"]["checkpoint_id"] == saved.sha256


def test_d1_probe_rejects_wrong_sha_or_construction(tmp_path: Path) -> None:
    plan, saved = checkpoint(tmp_path)
    train, heldout = source(tmp_path, "train"), source(tmp_path, "heldout")
    config = ProbeConfig(lags=(2,), train_positions=4, eval_positions=2, updates=1)
    with pytest.raises(RuntimeError, match="SHA-256"):
        run_d1_checkpoint_probe(
            plan, saved.path, "0" * 64, train, heldout, tmp_path / "bad", config
        )
    with pytest.raises(RuntimeError, match="identity"):
        run_d1_checkpoint_probe(
            micro_plan(D1Arm.NONE),
            saved.path,
            saved.sha256,
            train,
            heldout,
            tmp_path / "bad",
            config,
        )
    assert not (tmp_path / "bad" / "receipt.json").exists()
