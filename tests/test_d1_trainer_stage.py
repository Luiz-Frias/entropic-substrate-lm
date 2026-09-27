"""Both operator lanes consume one attested stage/start transaction."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.ladder.run.campaign.d1_operator_evidence import D1ExportedTrainerHandoff
from entropic_substrate.ladder.run.trainer.d1_trainer_stage import stage_and_start_d1_trainer

from .test_d1_operator_controller import _case, _Trainer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("drift", [None, "plan", "artifact_digest", "runtime_version"])
def test_shared_stage_rejects_mutation_before_trainer_start(
    tmp_path: Path, drift: str | None
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    trainer = _Trainer(
        D1ExportedTrainerHandoff(case.handoff, case.journal_path),
        events,
        case.start,
        stage_drift=drift,
    )
    runtime = trainer.bootstrap_runtime("pod-123", "a" * 64)
    if drift is None:
        stage_and_start_d1_trainer(
            trainer, "pod-123", case.intent.plan, case.start, runtime, "a" * 64
        )
        assert "start_trainer" in events
    else:
        with pytest.raises(ValueError):
            stage_and_start_d1_trainer(
                trainer, "pod-123", case.intent.plan, case.start, runtime, "a" * 64
            )
        assert "start_trainer" not in events
