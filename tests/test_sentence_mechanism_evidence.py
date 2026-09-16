"""Active dynamics must leave validated, phase-specific numerical evidence."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.sentence_prediction import execution as runtime
from entropic_substrate.sentence_prediction import mechanism_evidence
from entropic_substrate.sentence_prediction.checkpoint import save_checkpoint
from entropic_substrate.sentence_prediction.execution import SentenceExecution
from entropic_substrate.sentence_prediction.journal import (
    artifact_reference,
    read_journal,
    read_json,
    write_journal,
    write_json,
)
from entropic_substrate.sentence_prediction.model import SentencePredictor
from entropic_substrate.sentence_prediction.seals import read_seal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("consumer", ["fisher", "isotropic"])
def test_step_records_actual_dynamics_and_journal_replays_it(tmp_path: Path, consumer: str) -> None:
    template = engine(tmp_path)
    run = SentenceExecution(
        replace(template.profile, consumer=consumer),
        template.source_root,
        max_source_bytes=1_000_000,
    )
    assert run.advance()["mechanism"]["reason"] == "codec_warmup"
    run.advance()
    run.begin_prediction()
    row = run.advance()
    measured = row["mechanism"]
    assert measured["solver"]["method"] == (
        "conjugate_gradient" if consumer == "fisher" else "analytical_isotropic"
    )
    assert measured["solver"]["max_residual_ratio"] <= 1
    assert measured["states"]["dtype"] == "torch.float32"
    assert measured["sample"]["count"] == run.profile.schedule.batch_size
    artifact = write_journal(
        tmp_path, "measured.json", run, (row,), parent=None, max_bytes=1_000_000
    )
    loaded = read_journal(
        tmp_path,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=1_000_000,
        max_source_bytes=1_000_000,
    )
    assert loaded["schema"] == "h-journal-shard-v2"
    assert loaded["rows"][0]["mechanism"] == measured
    for index, (field, message) in enumerate(
        (("solver", "convergence"), ("states", "position count"), ("residual", "residual"))
    ):
        forged = copy.deepcopy(row)
        if field == "solver":
            forged["mechanism"]["solver"]["max_residual_ratio"] = 2
        elif field == "states":
            forged["mechanism"]["states"]["count"] += 1
        else:
            excessive = forged["mechanism"]["solver"]["tolerance"][2] + 1
            forged["mechanism"]["solver"]["residual"] = [excessive] * 3
            forged["mechanism"]["solver"]["max_residual_ratio"] = 0
        with pytest.raises(ValueError, match=message):
            write_journal(
                tmp_path, f"forged-{index}.json", run, (forged,), parent=None, max_bytes=1_000_000
            )


def test_missing_active_diagnostics_refuse_before_optimizer_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    original = SentencePredictor.forward

    def missing(self: SentencePredictor, history: torch.Tensor, prefix: torch.Tensor):
        result = original(self, history, prefix)
        return replace(result, prediction=replace(result.prediction, diagnostics=None))

    monkeypatch.setattr(SentencePredictor, "forward", missing)
    with pytest.raises(ValueError, match="solver"):
        run.advance()
    assert run.failed and run.step == 0 and not run.optimizer.state


def test_live_consumer_drift_is_not_labelled_as_the_registered_arm(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    run.model.consumer_mode = "isotropic"
    with pytest.raises(ValueError, match="construction"):
        run.advance()
    assert run.failed and run.step == 0


def test_missing_prediction_readout_is_not_misclassified_as_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    original = runtime.prediction_loss

    def missing(*args: object, **kwargs: object):
        return replace(original(*args, **kwargs), readout=None)

    monkeypatch.setattr(runtime, "prediction_loss", missing)
    with pytest.raises(ValueError, match="readout"):
        run.advance()
    assert run.failed and run.step == 0 and not run.optimizer.state


def test_summary_admission_preserves_real_float_reduction_roundoff() -> None:
    values = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float64)
    summary = mechanism_evidence._range(values)
    assert mechanism_evidence._read_range(summary)[1] == summary[1]


def test_zero_carrier_measurements_are_retained_without_inventing_unit_norm() -> None:
    zero = mechanism_evidence._norms(torch.zeros(2, 8))
    mechanism_evidence._read_norms(zero, 2, spherical=False)
    assert zero["zero_count"] == 2 and zero["range"] == [0, 0, 0]
    with pytest.raises(ValueError, match="spherical"):
        mechanism_evidence._read_norms(zero, 2, spherical=True)


def test_legacy_journal_is_readable_but_cannot_authorize_current_resume(tmp_path: Path) -> None:
    run = engine(tmp_path)
    rows = (run.advance(), run.advance())
    current = write_journal(tmp_path, "current.json", run, rows, parent=None, max_bytes=1_000_000)
    value = read_json(tmp_path, current, max_bytes=1_000_000)
    value["schema"] = "h-journal-shard-v1"
    for row in value["rows"]:
        row["schema"] = "h-training-step-v1"
        del row["mechanism"]
    legacy = write_json(tmp_path, "legacy.json", value, max_bytes=1_000_000)
    assert (
        read_journal(
            tmp_path,
            legacy,
            run.profile,
            run.source_root,
            max_bytes=1_000_000,
            max_source_bytes=1_000_000,
        )["schema"]
        == "h-journal-shard-v1"
    )
    checkpoint = save_checkpoint(
        tmp_path,
        "legacy-bound.pt",
        run.model,
        run.optimizer,
        run.sampler,
        phase=run.phase,
        step=run.step,
        run_identity=run.profile.identity,
        measurement=legacy.content_hash,
        max_bytes=10_000_000,
    )
    seal = write_json(
        tmp_path,
        "mixed-seal.json",
        {
            "schema": "h-checkpoint-seal-v1",
            "run_identity": run.profile.identity,
            "phase": run.phase,
            "step": run.step,
            "checkpoint": artifact_reference(tmp_path, checkpoint),
            "journal": artifact_reference(tmp_path, legacy),
        },
        max_bytes=1_000_000,
    )
    with pytest.raises(ValueError, match="journal v2"):
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=10_000_000,
            max_source_bytes=1_000_000,
        )
