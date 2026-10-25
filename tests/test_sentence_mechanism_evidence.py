"""Active dynamics must leave validated, phase-specific numerical evidence."""

import copy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.spine import CarrierBlock
from entropic_substrate.ladder.run import execution as runtime
from entropic_substrate.ladder.run import mechanism_evidence
from entropic_substrate.ladder.run.checkpoint import save_checkpoint
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import (
    artifact_reference,
    read_journal,
    read_json,
    write_journal,
    write_json,
)
from entropic_substrate.ladder.run.seals import read_seal

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


@pytest.mark.parametrize(
    "component", ["none", "consumer", "solver", "spine", "codec", "law", "head"]
)
def test_live_consumer_drift_is_not_labelled_as_the_registered_arm(
    tmp_path: Path, component: str
) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    if component == "none":
        assert run.advance()["status"] == "committed"
        assert not run.failed and run.step == 1
        return
    if component == "consumer":
        run.model.consumer_mode = "isotropic"
    elif component == "solver":
        run.model.solver = replace(run.model.solver, damping=2)
    elif component == "spine":
        run.model.spine.spec = replace(run.model.spine.spec, context_sentences=5)
    elif component == "codec":
        run.model.codec.spec = replace(run.model.codec.spec, max_sentence_bytes=65)
    elif component == "law":
        run.model.codec.law_spec = replace(run.model.codec.law_spec, log_radius_std=0.2)
    else:
        run.model.head.spec = replace(run.model.head.spec, log_radius_std=0.2)
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


# FOR-1673: a zero-sentence mechanism row says WHY it has no norm range.
# Read-old literals below are verbatim from capture h0-plumbing-1k-v3.HEpP7g
# (both arms, measured: 250 warm-up rows, 732 non-empty and 18 zero-sentence
# prediction rows per arm, every zero-sentence entry byte-identical to
# ABSENT_V1). Unit tests never read results/.
SCHEMA_V1 = "h-mechanism-evidence-v1"
SCHEMA_V2 = "h-mechanism-evidence-v2"
WARMUP_V1 = {"schema": SCHEMA_V1, "status": "structurally_absent", "reason": "codec_warmup"}
WARMUP_V2 = {"schema": SCHEMA_V2, "status": "structurally_absent", "reason": "codec_warmup"}
ABSENT_V1 = {"count": 0, "dtype": "torch.float32", "range": None, "zero_count": 0}
ABSENT_V2 = {**ABSENT_V1, "status": "structurally_absent", "reason": "no_sentence_states"}


def zero_sentence_row(tmp_path: Path) -> tuple[SentenceExecution, dict[str, object]]:
    """Real SentencePredictor forward over a (B, 0, W) history, never a swapped tensor."""
    run = engine(tmp_path)
    history = torch.zeros(1, 0, 1, dtype=torch.int64)
    readout = run.model(history, torch.tensor([[258, 67]]))
    assert readout.prediction.sentences.shape[:2] == (1, 0)
    row = mechanism_evidence.collect_mechanism(
        readout,
        run.profile,
        phase="prediction",
        zero_views=[cast(CarrierBlock, b).zero_views for b in run.model.spine.blocks],
    )
    return run, row


def admit(run: SentenceExecution, row: object, *, phase: str = "prediction") -> None:
    mechanism_evidence.admit_mechanism(row, run.profile, phase=phase, batch=1, history=0)


def test_zero_sentence_readout_states_why_its_range_is_absent(tmp_path: Path) -> None:
    run, row = zero_sentence_row(tmp_path)
    assert row["sentences"] == ABSENT_V2
    assert row["schema"] == SCHEMA_V2 and row["status"] == "measured"
    assert set(row) == {
        "schema",
        "status",
        "consumer",
        "sentences",
        "states",
        "sample",
        "solver",
        "zero_views",
    }
    # Only sentences can be empty; states and sample are measured even here.
    for field in ("states", "sample"):
        assert "status" not in row[field] and "reason" not in row[field]  # type: ignore[operator]
        assert row[field]["range"] is not None  # type: ignore[index]
    admit(run, row)


def test_emitted_rows_carry_the_v2_label_in_both_phases_without_absence_on_measured_norms(
    tmp_path: Path,
) -> None:
    run = engine(tmp_path)
    assert run.advance()["mechanism"] == WARMUP_V2
    run.advance()
    run.begin_prediction()
    mechanism = run.advance()["mechanism"]
    assert mechanism["schema"] == SCHEMA_V2
    for field in ("sentences", "states", "sample"):
        assert set(mechanism[field]) == {"count", "zero_count", "dtype", "range"}
        assert mechanism[field]["range"] is not None


def test_read_old_hepp7g_rows_still_admit_through_the_journal_entry(tmp_path: Path) -> None:
    run, row = zero_sentence_row(tmp_path)
    legacy = copy.deepcopy(row)
    legacy["schema"] = SCHEMA_V1
    legacy["sentences"] = dict(ABSENT_V1)
    admit(run, legacy)
    mechanism_evidence.admit_mechanism(
        WARMUP_V1, run.profile, phase="codec_warmup", batch=1, history=0
    )
    mechanism_evidence.admit_mechanism(
        WARMUP_V2, run.profile, phase="codec_warmup", batch=1, history=0
    )


@pytest.mark.parametrize(
    ("sentences", "message"),
    [
        ({**ABSENT_V2, "range": [0.0, 0.0, 0.0]}, "invent"),
        ({**ABSENT_V1, "range": [0.0, 0.0, 0.0]}, "fields|invent"),
        (dict(ABSENT_V1), "fields"),
        ({**ABSENT_V2, "reason": "codec_warmup"}, "absence"),
        ({**ABSENT_V2, "status": "measured"}, "absence"),
        ({**ABSENT_V2, "reason": None}, "absence"),
        ({k: v for k, v in ABSENT_V2.items() if k != "reason"}, "fields"),
        ({k: v for k, v in ABSENT_V2.items() if k != "status"}, "fields"),
        ({**ABSENT_V2, "count": 1}, "position count"),
        ({**ABSENT_V2, "zero_count": 1}, "position count"),
        ({**ABSENT_V2, "dtype": "torch.float64"}, "position count"),
    ],
)
def test_v2_zero_sentence_entry_is_closed_and_exact(
    tmp_path: Path, sentences: dict[str, object], message: str
) -> None:
    run, row = zero_sentence_row(tmp_path)
    forged = copy.deepcopy(row)
    forged["sentences"] = sentences
    with pytest.raises(ValueError, match=message):
        admit(run, forged)


def test_v1_label_cannot_carry_absence_keys_and_unknown_labels_are_refused(
    tmp_path: Path,
) -> None:
    run, row = zero_sentence_row(tmp_path)
    mixed = copy.deepcopy(row)
    mixed["schema"] = SCHEMA_V1
    with pytest.raises(ValueError, match="fields"):
        admit(run, mixed)
    unknown = copy.deepcopy(row)
    unknown["schema"] = "h-mechanism-evidence-v3"
    with pytest.raises(ValueError, match="construction"):
        admit(run, unknown)
    for forged in (
        {**WARMUP_V2, "schema": "h-mechanism-evidence-v3"},
        {**WARMUP_V2, "reason": "no_sentence_states"},
        {**WARMUP_V1, "extra": 1},
    ):
        with pytest.raises(ValueError, match="warm-up"):
            mechanism_evidence.admit_mechanism(
                forged, run.profile, phase="codec_warmup", batch=1, history=0
            )


def test_absence_keys_on_a_nonempty_or_unlabelled_entry_are_refused() -> None:
    measured = mechanism_evidence._norms(torch.ones(2, 8))
    with pytest.raises(ValueError, match="fields"):
        mechanism_evidence._read_norms(
            {**measured, "status": "structurally_absent", "reason": "no_sentence_states"},
            2,
            spherical=False,
        )
    with pytest.raises(ValueError, match="fields"):
        mechanism_evidence._read_norms(
            {**measured, "status": "structurally_absent", "reason": "no_sentence_states"},
            2,
            spherical=False,
            labelled=True,
        )
    # An empty states or sample tensor stays an execution defect, never labelled.
    with pytest.raises(ValueError, match="fields"):
        mechanism_evidence._read_norms(dict(ABSENT_V2), 0, spherical=False)
    labelled = mechanism_evidence._norms(torch.zeros(0, 8), absent_reason="no_sentence_states")
    assert labelled == ABSENT_V2
    assert mechanism_evidence._norms(torch.zeros(0, 8)) == ABSENT_V1
