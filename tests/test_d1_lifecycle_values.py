"""Contract tests for canonical D1 lifecycle persistence values."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.evidence.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1LifecycleValueError,
    D1ProblemKind,
    D1ProcessEventKind,
    canonical_lifecycle_event_line,
    parse_lifecycle_event_line,
)
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    SuccessorAuthorization,
)


def _artifact(index: int) -> ArtifactIdentity:
    return ArtifactIdentity(
        path=f"evidence/{index}.json",
        content_hash=ContentHash(algorithm="sha256", digest=f"{index:064x}"),
    )


def _actor(role: ActorRole = ActorRole.OPERATOR) -> ActorIdentity:
    return ActorIdentity(actor_id=f"actor-{role.value}", session_id="session-1", role=role)


def _event(**overrides: object) -> D1LifecycleEvent:
    values: dict[str, object] = {
        "event_seq": 0,
        "event_id": "event-0",
        "event_kind": D1ProcessEventKind.PREREGISTRATION_FILED,
        "actor": _actor(),
        "artifacts": (_artifact(1),),
    }
    values.update(overrides)
    return D1LifecycleEvent(**values)  # type: ignore[arg-type]


def test_canonical_lifecycle_event_round_trips_exact_bytes() -> None:
    event = _event()

    line = canonical_lifecycle_event_line(event)

    assert line.endswith(b"\n")
    assert parse_lifecycle_event_line(line) == event
    assert canonical_lifecycle_event_line(parse_lifecycle_event_line(line)) == line
    payload = json.loads(line)
    assert payload["schema_version"] == "d1-lifecycle-event/v1"
    assert payload["artifacts"] == [
        {
            "content_hash": {"algorithm": "sha256", "digest": f"{1:064x}"},
            "path": "evidence/1.json",
        }
    ]


def test_operational_events_require_subject_identity() -> None:
    with pytest.raises(D1LifecycleValueError, match="subject_id") as error:
        _event(event_kind=D1ProcessEventKind.RUNNING)

    assert error.value.code == "missing_subject"


def test_nonoperational_events_refuse_subject_identity() -> None:
    with pytest.raises(D1LifecycleValueError, match="subject_id") as error:
        _event(subject_id="run-1")

    assert error.value.code == "unexpected_subject"


@pytest.mark.parametrize(
    ("kind", "authorization", "strategies"),
    [
        (
            D1ProcessEventKind.GATE_AUTHORIZED,
            SuccessorAuthorization.AUTHORIZED,
            ("d2-pullback",),
        ),
        (
            D1ProcessEventKind.GATE_VALID_NEGATIVE,
            SuccessorAuthorization.VALID_NEGATIVE,
            (),
        ),
        (
            D1ProcessEventKind.GATE_RETIRED,
            SuccessorAuthorization.RETIRED,
            (),
        ),
    ],
)
def test_gate_events_bind_exact_outcome(
    kind: D1ProcessEventKind,
    authorization: SuccessorAuthorization,
    strategies: tuple[str, ...],
) -> None:
    event = _event(
        event_kind=kind,
        gate_authorization=authorization,
        eligible_strategies=strategies,
    )

    assert parse_lifecycle_event_line(canonical_lifecycle_event_line(event)) == event


def test_authorized_gate_requires_a_unique_strategy() -> None:
    with pytest.raises(D1LifecycleValueError, match="eligible_strategies") as error:
        _event(
            event_kind=D1ProcessEventKind.GATE_AUTHORIZED,
            gate_authorization=SuccessorAuthorization.AUTHORIZED,
            eligible_strategies=(),
        )
    assert error.value.code == "invalid_gate_strategy"

    with pytest.raises(D1LifecycleValueError, match="eligible_strategies"):
        _event(
            event_kind=D1ProcessEventKind.GATE_AUTHORIZED,
            gate_authorization=SuccessorAuthorization.AUTHORIZED,
            eligible_strategies=("d2", "d2"),
        )


def test_problem_event_requires_typed_problem_and_other_events_refuse_it() -> None:
    problem = _event(
        event_kind=D1ProcessEventKind.PROBLEM_RECORDED,
        problem_kind=D1ProblemKind.EVIDENCE_GAP,
    )
    assert parse_lifecycle_event_line(canonical_lifecycle_event_line(problem)) == problem

    with pytest.raises(D1LifecycleValueError, match="problem_kind"):
        _event(event_kind=D1ProcessEventKind.PROBLEM_RECORDED)
    with pytest.raises(D1LifecycleValueError, match="problem_kind"):
        _event(problem_kind=D1ProblemKind.EVIDENCE_GAP)


def test_event_refuses_duplicate_artifact_paths_and_non_artifact_payloads() -> None:
    with pytest.raises(D1LifecycleValueError, match="unique paths"):
        _event(artifacts=(_artifact(1), _artifact(1)))

    with pytest.raises(TypeError, match="ArtifactIdentity"):
        _event(artifacts=({"path": "live-object"},))


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda line: line.rstrip(b"\n"), "partial_line"),
        (
            lambda line: line.replace(b'"event_seq":0', b'"event_seq":0,"event_seq":0'),
            "duplicate_field",
        ),
        (lambda line: b" " + line, "noncanonical_json"),
        (
            lambda line: line.replace(
                b'"schema_version":"d1-lifecycle-event/v1"', b'"schema_version":"v0"'
            ),
            "unsupported_schema",
        ),
    ],
)
def test_parser_fails_loudly_on_corrupt_or_noncanonical_lines(
    mutator: object,
    code: str,
) -> None:
    line = canonical_lifecycle_event_line(_event())

    with pytest.raises(D1LifecycleValueError) as error:
        parse_lifecycle_event_line(mutator(line))  # type: ignore[operator]

    assert error.value.code == code


def test_parser_rejects_unknown_fields() -> None:
    payload = json.loads(canonical_lifecycle_event_line(_event()))
    payload["live_client"] = "forbidden"
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode() + b"\n"

    with pytest.raises(D1LifecycleValueError) as error:
        parse_lifecycle_event_line(line)

    assert error.value.code == "unknown_field"
