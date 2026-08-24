"""Pilot and aborted D1 evidence survive transport outside the decision set."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    D1DvcBindingState,
    identify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_bundle import D1AttemptEvidence
from entropic_substrate.evidence.d1_capture_roster import (
    D1AttemptLane,
    D1DeclaredAttempt,
    D1LaunchRoster,
)
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass, canonical_digest

from .d1_capture_fixtures import _events_for, capture_family
from .d1_telemetry_fixtures import write_events
from .test_d1_capture_service import _POINTER_PATH, _capture_pointer, _service

pytestmark = pytest.mark.unit


def _attempt_events(
    *, logical_run_id: str, run_class: RunClass, aborted: bool
) -> list[D1JournalEvent]:
    events, _ = _events_for(
        D1Arm.SG,
        logical_run_id=logical_run_id,
        construction_digest="2" * 64,
        execution_digest="6" * 64,
        aggregation_digest="7" * 64,
        checkpoint_sha256="f" * 64,
    )
    rewritten: list[D1JournalEvent] = []
    for event in events:
        payload = deepcopy(cast(dict[str, object], event.manifest["payload"]))
        if event.event_kind is JournalEventKind.RUN_START:
            plan = cast(dict[str, object], payload["plan"])
            plan["run_class"] = run_class.value
            payload["plan_digest"] = canonical_digest(plan)
        rewritten.append(
            replace(
                event,
                logical_run_id=logical_run_id,
                run_class=run_class,
                payload=payload,
            )
        )
    if aborted:
        rewritten[-1] = replace(
            rewritten[-1],
            event_kind=JournalEventKind.RUN_ABORT,
            payload={"error": "registered attempt aborted"},
        )
    return rewritten


def _attempt(
    root: Path,
    *,
    logical_run_id: str,
    run_class: RunClass,
    lane: D1AttemptLane,
) -> tuple[D1DeclaredAttempt, D1AttemptEvidence]:
    path = root / "attempts" / f"{logical_run_id}.jsonl"
    path.parent.mkdir(exist_ok=True)
    write_events(
        path,
        _attempt_events(
            logical_run_id=logical_run_id,
            run_class=run_class,
            aborted=lane is D1AttemptLane.ABORTED,
        ),
    )
    declared = D1DeclaredAttempt(
        logical_run_id=logical_run_id,
        arm=D1Arm.SG,
        seed=41,
        pairing_id="attempt-41",
        run_class=run_class,
        lane=lane,
        disposition="retained non-decision evidence",
    )
    artifact = identify_d1_artifact(
        root,
        path,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path=_POINTER_PATH,
    )
    return declared, D1AttemptEvidence(declared, (artifact,), None)


def test_excluded_pilot_and_aborted_attempts_are_bound_but_never_enter_set_proof(
    tmp_path: Path,
) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    pilot = _attempt(
        source_root,
        logical_run_id="d1-pilot-sg-s41",
        run_class=RunClass.PILOT,
        lane=D1AttemptLane.EXCLUDED_PILOT,
    )
    aborted = _attempt(
        source_root,
        logical_run_id="d1-sg-s41-attempt-0",
        run_class=RunClass.CONFIRMATORY,
        lane=D1AttemptLane.ABORTED,
    )
    roster = D1LaunchRoster(
        preregistration_path=roster.preregistration_path,
        preregistration_commit=roster.preregistration_commit,
        family_digest=roster.family_digest,
        expected_confirmatory=roster.expected_confirmatory,
        attempts=(pilot[0], aborted[0]),
    )
    service, _, _ = _service(tmp_path, source_root)

    receipt = service.capture(
        roster=roster,
        capture_inputs=tuple(_capture_pointer(item) for item in raw_inputs),
        attempt_evidence=(pilot[1], aborted[1]),
    )

    assert [item.attempt.lane for item in receipt.bundle.attempts] == [
        D1AttemptLane.EXCLUDED_PILOT,
        D1AttemptLane.ABORTED,
    ]
    assert all(
        artifact.dvc.state is D1DvcBindingState.BOUND
        for item in receipt.bundle.attempts
        for artifact in item.artifacts
    )
    assert receipt.bundle.set_proof.expected_logical_run_ids == roster.expected_run_ids
    assert not receipt.bundle.set_proof.expected_logical_run_ids.intersection(
        {pilot[0].logical_run_id, aborted[0].logical_run_id}
    )
