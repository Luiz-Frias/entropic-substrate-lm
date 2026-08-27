from __future__ import annotations

import copy
import hashlib
import json
import pickle
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple, cast

import pytest

from d1_billing_finalizer_fixtures import billing_gap, provider_bill
from entropic_substrate.d1_billing_finalization_values import D1BillingFinalization
from entropic_substrate.d1_billing_finalizer import finalize_d1_billing
from entropic_substrate.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyError,
    D1FinalizedCellCustodyRecord,
    attest_d1_finalized_cell_custody,
    parse_d1_finalized_cell_custody,
    serialize_d1_finalized_cell_custody,
)
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.d1_trainer_handoff import D1FreshRunCompletion
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.evidence.resource_billing import ExecutionSegmentTermination
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    SegmentStartMode,
    execution_segment_id,
)
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import RunClass, canonical_json

from .test_d1_operator_controller import _case

pytestmark = pytest.mark.unit


class _FinalizedCase(NamedTuple):
    intent: D1OperatorLaunchIntent
    starts: tuple[RunStartProvenance, ...]
    ledger: D1LaunchAttemptLedger
    finalization: D1BillingFinalization
    journal: ArtifactIdentity
    checkpoint: ArtifactIdentity
    quarantined_siblings: tuple[ArtifactIdentity, ...]


def _finalized_case(tmp_path: Path, *, missing_cost: bool = False) -> _FinalizedCase:
    tmp_path.mkdir(parents=True, exist_ok=True)
    case = _case(tmp_path)
    intent = case.intent
    start = cast(RunStartProvenance, case.start)
    handoff = cast(D1FreshRunCompletion, case.handoff)
    export = verify_d1_trainer_export(D1ExportedTrainerHandoff(handoff, case.journal_path))
    assert isinstance(export.checkpoint, ArtifactIdentity)
    ended = handoff.termination.trainer_ended_at
    termination = ExecutionSegmentTermination(
        start.logical_run_id,
        0,
        start.execution_segment_id,
        ended,
        ended + timedelta(seconds=1),
    )
    provider = billing_gap(start) if missing_cost else provider_bill(start)
    finalization = finalize_d1_billing(
        plan=intent.plan,
        segment_starts=(start,),
        segment_terminations=(termination,),
        provider_billings=(provider,),
        handoff=handoff,
        journal_path=case.journal_path,
        expected_journal_state=handoff.journal_state,
    )
    attempt = D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        intent.plan.profile.family.family_digest,
        intent.plan.profile.arm,
        intent.plan.seed,
        intent.plan.logical_run_id,
        cast(str, intent.registered_design_digest),
    )
    ledger = D1LaunchAttemptLedger().record(
        attempt,
        observed_at=start.trainer_started_at - timedelta(seconds=1),
    )
    ledger = ledger.transition(
        attempt,
        status=D1LaunchAttemptStatus.TRAINER_STARTED,
        observed_at=start.trainer_started_at,
    )
    ledger = ledger.transition(
        attempt,
        status=D1LaunchAttemptStatus.COMPLETED,
        observed_at=termination.pod_terminated_at,
    )
    journal = ArtifactIdentity(
        str(case.journal_path),
        ContentHash("sha256", hashlib.sha256(case.journal_path.read_bytes()).hexdigest()),
    )
    return _FinalizedCase(
        intent,
        (start,),
        ledger,
        finalization,
        journal,
        export.checkpoint,
        (),
    )


def _attest(case: _FinalizedCase) -> D1FinalizedCellCustodyRecord:
    return attest_d1_finalized_cell_custody(
        intent=case.intent,
        segment_starts=case.starts,
        attempt_ledger=case.ledger,
        finalization=case.finalization,
        journal=case.journal,
        checkpoint=case.checkpoint,
        quarantined_siblings=case.quarantined_siblings,
    )


def test_finalized_cell_round_trips_exact_canonical_evidence(tmp_path: Path) -> None:
    case = _finalized_case(tmp_path)
    record = _attest(case)

    record.require_factory_attested()
    assert (record.billing_posture, record.quality_export_ready, record.trainer_outcome) == (
        "reconciled",
        True,
        "completed",
    )
    assert record.checkpoint == case.checkpoint
    assert record.quarantined_siblings == ()
    assert record.schema_version == "d1-finalized-cell-custody-v3"
    payload = serialize_d1_finalized_cell_custody(record)
    restored = parse_d1_finalized_cell_custody(payload)

    assert payload == serialize_d1_finalized_cell_custody(restored)
    assert restored.manifest == record.manifest
    restored.require_factory_attested()


def test_factory_requires_completed_ledger_reconciled_billing_and_final_journal(
    tmp_path: Path,
) -> None:
    case = _finalized_case(tmp_path / "valid")
    with pytest.raises(ValueError, match="completed attempt"):
        _attest(case._replace(ledger=D1LaunchAttemptLedger(case.ledger.events[:-1])))

    gap = _finalized_case(tmp_path / "gap", missing_cost=True)
    with pytest.raises(ValueError, match="billing"):
        _attest(gap)

    wrong_journal = replace(
        case.journal,
        content_hash=ContentHash("sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="journal"):
        _attest(case._replace(journal=wrong_journal))

    wrong_checkpoint = replace(
        case.checkpoint,
        content_hash=ContentHash("sha256", "0" * 64),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        _attest(case._replace(checkpoint=wrong_checkpoint))

    wrong_design = replace(case.intent, registered_design_digest="1" * 64)
    with pytest.raises(ValueError, match="attempt identity"):
        _attest(case._replace(intent=wrong_design))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("control", ""),
        ("intent_digest", "intent digest"),
        ("plan_digest", "intent and finalized-cell identity"),
        ("start_identity", "finalized-cell evidence is invalid"),
        ("ledger_digest", "attempt_ledger digest"),
        ("ledger_terminal", "finalized-cell evidence is invalid"),
        ("finalization_digest", "finalization digest"),
        ("journal_digest", "final journal identity"),
        ("checkpoint_digest", "checkpoint"),
        ("quarantine_unknown", "quarantined sibling"),
        ("billing_posture", "billing and quality posture"),
        ("quality_ready", "billing and quality posture"),
        ("unknown_field", "fields must be exact"),
    ],
)
def test_codec_rejects_each_substituted_evidence_edge(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    payload = serialize_d1_finalized_cell_custody(_attest(_finalized_case(tmp_path)))
    value = json.loads(payload)
    if mutation == "control":
        assert parse_d1_finalized_cell_custody(payload).manifest == value
        return
    if mutation == "intent_digest":
        value["intent"]["sha256"] = "0" * 64
    elif mutation == "plan_digest":
        value["plan_digest"] = "0" * 64
    elif mutation == "start_identity":
        value["segment_starts"][0]["logical_run_id"] = "substituted"
    elif mutation == "ledger_digest":
        value["attempt_ledger"]["sha256"] = "0" * 64
    elif mutation == "ledger_terminal":
        manifest = value["attempt_ledger"]["manifest"]
        manifest["events"][-1]["status"] = "recorded"
        value["attempt_ledger"]["sha256"] = hashlib.sha256(
            canonical_json(manifest).encode()
        ).hexdigest()
    elif mutation == "finalization_digest":
        value["finalization"]["sha256"] = "0" * 64
    elif mutation == "journal_digest":
        value["journal"]["sha256"] = "0" * 64
    elif mutation == "checkpoint_digest":
        value["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "quarantine_unknown":
        value["quarantined_siblings"] = [value["checkpoint"]]
    elif mutation == "billing_posture":
        value["billing_posture"] = "missing_cost"
    elif mutation == "quality_ready":
        value["quality_export_ready"] = False
    else:
        value["unexpected"] = True
    changed = canonical_json(value)

    with pytest.raises(D1FinalizedCellCustodyError, match=message):
        parse_d1_finalized_cell_custody(changed)


def test_finalized_cell_capability_cannot_be_constructed_copied_or_replayed(
    tmp_path: Path,
) -> None:
    record = _attest(_finalized_case(tmp_path))
    with pytest.raises(TypeError, match="factory-only"):
        D1FinalizedCellCustodyRecord()
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError, match="cannot be copied"):
            copier(record)


def test_resumed_cell_custody_retains_verified_quarantined_sibling(tmp_path: Path) -> None:
    case = _finalized_case(tmp_path)
    start0 = case.starts[0]
    start1 = replace(
        start0,
        execution_segment_index=1,
        execution_segment_id=execution_segment_id(start0.logical_run_id, 1),
        segment_start_mode=SegmentStartMode.RESUME,
        resume_checkpoint_sha256=case.checkpoint.content_hash.digest,
        pod_created_at=start0.pod_created_at + timedelta(seconds=2),
        trainer_started_at=start0.trainer_started_at + timedelta(seconds=2),
    )
    quarantine_path = tmp_path / "run.quarantine.jsonl"
    quarantine_path.write_bytes(b"quarantined tail\n")
    quarantine = ArtifactIdentity(
        str(quarantine_path),
        ContentHash("sha256", hashlib.sha256(quarantine_path.read_bytes()).hexdigest()),
    )
    resumed = case._replace(
        intent=replace(
            case.intent,
            execution_segment_index=1,
            resume_checkpoint_sha256=case.checkpoint.content_hash.digest,
        ),
        starts=(start0, start1),
        quarantined_siblings=(quarantine,),
    )

    record = _attest(resumed)

    assert record.quarantined_siblings == (quarantine,)
    wrong = replace(quarantine, content_hash=ContentHash("sha256", "0" * 64))
    with pytest.raises(ValueError, match="quarantined sibling"):
        _attest(resumed._replace(quarantined_siblings=(wrong,)))


def test_attestation_rejects_unminted_and_mutated_records(tmp_path: Path) -> None:
    forged = object.__new__(D1FinalizedCellCustodyRecord)
    with pytest.raises(TypeError, match="not factory-minted"):
        forged.require_factory_attested()

    record = _attest(_finalized_case(tmp_path))
    object.__setattr__(record, "logical_run_id", "substituted")
    with pytest.raises(TypeError, match="changed after minting"):
        record.require_factory_attested()
