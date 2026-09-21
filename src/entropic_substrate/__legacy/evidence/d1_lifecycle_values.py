"""Canonical artifact-only values for the persisted D1 process lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from entropic_substrate.evidence.d1_types import D1_STABILIZATION_ARM_IDS
from entropic_substrate.evidence.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
    LifecycleContractError,
    SuccessorAuthorization,
)

D1_LIFECYCLE_EVENT_SCHEMA: Final = "d1-lifecycle-event/v2"
_EVENT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "rung",
        "event_seq",
        "event_id",
        "event_kind",
        "actor",
        "subject_id",
        "artifacts",
        "problem_kind",
        "gate_authorization",
        "eligible_arms",
        "eligible_strategies",
    }
)
_ACTOR_FIELDS: Final = frozenset({"actor_id", "session_id", "role"})
_ARTIFACT_FIELDS: Final = frozenset({"path", "content_hash"})
_HASH_FIELDS: Final = frozenset({"algorithm", "digest"})


class D1LifecycleValueError(ValueError):
    """A malformed persisted lifecycle value with a stable error code."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": path}
        super().__init__(f"D1 lifecycle {code} at {path}: {detail}")


def _error(code: str, path: str, detail: str) -> D1LifecycleValueError:
    return D1LifecycleValueError(code=code, path=path, detail=detail)


class D1ProcessState(StrEnum):
    """Derived states in the D1 flywheel process manager."""

    CONTRACT_VERIFIED = "contract_verified"
    PILOT_FILED = "pilot_filed"
    PREREGISTERED = "preregistered"
    LAUNCH_ADMITTED = "launch_admitted"
    RUNNING = "running"
    TRAINER_STOPPED = "trainer_stopped"
    OPERATOR_FINALIZED = "operator_finalized"
    CAPTURED = "captured"
    ANALYSIS_FILED = "analysis_filed"
    ALIAS_COMMITTED = "alias_committed"
    REVIEW_FILED = "review_filed"
    ALIAS_REVEALED = "alias_revealed"
    ADJUDICATED = "adjudicated"
    VERDICT_FILED = "verdict_filed"
    GATE_AUTHORIZED = "gate_authorized"
    VALID_NEGATIVE = "valid_negative"
    RETIRED = "retired"
    PROBLEM = "problem"


class D1ProcessEventKind(StrEnum):
    """Complete persisted event vocabulary for the D1 rung flywheel."""

    CONTRACT_VERIFIED = "contract_verified"
    PILOT_FILED = "pilot_filed"
    PREREGISTRATION_FILED = "preregistration_filed"
    LAUNCH_ADMITTED = "launch_admitted"
    RUNNING = "running"
    RESUMED = "resumed"
    TRAINER_STOPPED = "trainer_stopped"
    OPERATOR_FINALIZED = "operator_finalized"
    CAPTURED = "captured"
    ANALYSIS_FILED = "analysis_filed"
    ALIAS_COMMITMENT_FILED = "alias_commitment_filed"
    REVIEW_FILED = "review_filed"
    ALIAS_REVEALED = "alias_revealed"
    ADJUDICATED = "adjudicated"
    VERDICT_FILED = "verdict_filed"
    GATE_AUTHORIZED = "gate_authorized"
    GATE_VALID_NEGATIVE = "gate_valid_negative"
    GATE_RETIRED = "gate_retired"
    PROBLEM_RECORDED = "problem_recorded"


class D1ProblemKind(StrEnum):
    """Bounded scientific/operational problems, never parser corruption."""

    INVALIDATED = "invalidated"
    NUMERICAL_DEFECT = "numerical_defect"
    EVIDENCE_GAP = "evidence_gap"
    OPERATOR_DECISION_NEEDED = "operator_decision_needed"


_SUBJECT_EVENTS: Final = frozenset(
    {
        D1ProcessEventKind.LAUNCH_ADMITTED,
        D1ProcessEventKind.RUNNING,
        D1ProcessEventKind.RESUMED,
        D1ProcessEventKind.TRAINER_STOPPED,
        D1ProcessEventKind.OPERATOR_FINALIZED,
    }
)
_GATE_OUTCOME: Final = {
    D1ProcessEventKind.GATE_AUTHORIZED: SuccessorAuthorization.AUTHORIZED,
    D1ProcessEventKind.GATE_VALID_NEGATIVE: SuccessorAuthorization.VALID_NEGATIVE,
    D1ProcessEventKind.GATE_RETIRED: SuccessorAuthorization.RETIRED,
}


def _trimmed(value: object, *, path: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise _error("invalid_string", path, "must be a nonempty, trimmed string")
    return value


def _nonnegative_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise _error("invalid_integer", path, "must be a nonnegative integer")
    return value


def _exact_object(value: object, *, path: str, fields: frozenset[str]) -> Mapping[str, object]:
    if type(value) is not dict:
        raise _error("invalid_type", path, "must be a JSON object")
    mapping = cast(dict[object, object], value)
    if any(type(key) is not str for key in mapping):
        raise _error("invalid_field_name", path, "field names must be strings")
    actual = cast(set[str], set(mapping))
    if missing := sorted(fields - actual):
        raise _error("missing_field", path, f"missing required field(s): {', '.join(missing)}")
    if unknown := sorted(actual - fields):
        raise _error("unknown_field", path, f"unknown field(s): {', '.join(unknown)}")
    return cast(Mapping[str, object], mapping)


@dataclass(frozen=True, slots=True)
class D1LifecycleEvent:
    """One immutable fact; artifacts are identities, never opened objects."""

    event_seq: int
    event_id: str
    event_kind: D1ProcessEventKind
    actor: ActorIdentity
    artifacts: tuple[ArtifactIdentity, ...]
    subject_id: str | None = None
    problem_kind: D1ProblemKind | None = None
    gate_authorization: SuccessorAuthorization | None = None
    eligible_arms: tuple[str, ...] = ()
    eligible_strategies: tuple[str, ...] = ()
    rung: str = "d1"

    def __post_init__(self) -> None:
        _nonnegative_int(self.event_seq, path="event_seq")
        _trimmed(self.event_id, path="event_id")
        if self.rung != "d1":
            raise _error("invalid_rung", "rung", "D1 lifecycle events require rung d1")
        if type(self.event_kind) is not D1ProcessEventKind:
            raise TypeError("event_kind must be a D1ProcessEventKind")
        if type(self.actor) is not ActorIdentity:
            raise TypeError("actor must be an ActorIdentity")
        if type(self.artifacts) is not tuple or not self.artifacts:
            raise _error("missing_artifact", "artifacts", "requires at least one identity")
        if any(type(item) is not ArtifactIdentity for item in self.artifacts):
            raise TypeError("artifacts must contain only ArtifactIdentity values")
        paths = tuple(item.path for item in self.artifacts)
        if len(set(paths)) != len(paths):
            raise _error("duplicate_artifact", "artifacts", "artifacts require unique paths")
        self._validate_subject()
        self._validate_problem()
        self._validate_gate()

    def _validate_subject(self) -> None:
        if self.event_kind in _SUBJECT_EVENTS:
            if self.subject_id is None:
                raise _error(
                    "missing_subject", "subject_id", "operational event requires subject_id"
                )
            _trimmed(self.subject_id, path="subject_id")
        elif self.subject_id is not None:
            raise _error(
                "unexpected_subject", "subject_id", "nonoperational event forbids subject_id"
            )

    def _validate_problem(self) -> None:
        if self.event_kind is D1ProcessEventKind.PROBLEM_RECORDED:
            if type(self.problem_kind) is not D1ProblemKind:
                raise _error("missing_problem", "problem_kind", "problem event requires typed kind")
        elif self.problem_kind is not None:
            raise _error("unexpected_problem", "problem_kind", "only problem events carry a kind")

    def _validate_gate(self) -> None:
        expected = _GATE_OUTCOME.get(self.event_kind)
        if expected is None:
            if (
                self.gate_authorization is not None
                or self.eligible_arms
                or self.eligible_strategies
            ):
                raise _error(
                    "unexpected_gate", "gate_authorization", "only gate events carry outcome"
                )
            return
        if self.gate_authorization is not expected:
            raise _error("invalid_gate_outcome", "gate_authorization", f"must be {expected.value}")
        strategies = self.eligible_strategies
        arms = self.eligible_arms
        if type(arms) is not tuple or any(
            type(item) is not str or not item or item.strip() != item for item in arms
        ):
            raise _error("invalid_gate_arm", "eligible_arms", "must be trimmed identifiers")
        canonical_arms = tuple(arm for arm in D1_STABILIZATION_ARM_IDS if arm in arms)
        if arms != canonical_arms:
            raise _error("invalid_gate_arm", "eligible_arms", "must be unique canonical arms")
        if type(strategies) is not tuple or any(
            type(item) is not str or not item or item.strip() != item for item in strategies
        ):
            raise _error(
                "invalid_gate_strategy", "eligible_strategies", "must be trimmed identifiers"
            )
        if len(set(strategies)) != len(strategies):
            raise _error("invalid_gate_strategy", "eligible_strategies", "must be unique")
        if expected is SuccessorAuthorization.AUTHORIZED and not strategies:
            raise _error(
                "invalid_gate_strategy", "eligible_strategies", "AUTHORIZED requires a strategy"
            )
        if expected is SuccessorAuthorization.AUTHORIZED and not arms:
            raise _error("invalid_gate_arm", "eligible_arms", "AUTHORIZED requires an arm")
        if expected is not SuccessorAuthorization.AUTHORIZED and strategies:
            raise _error(
                "invalid_gate_strategy",
                "eligible_strategies",
                f"{expected.value} forbids strategies",
            )
        if expected is not SuccessorAuthorization.AUTHORIZED and arms:
            raise _error("invalid_gate_arm", "eligible_arms", f"{expected.value} forbids arms")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": D1_LIFECYCLE_EVENT_SCHEMA,
            "rung": self.rung,
            "event_seq": self.event_seq,
            "event_id": self.event_id,
            "event_kind": self.event_kind.value,
            "actor": {
                "actor_id": self.actor.actor_id,
                "session_id": self.actor.session_id,
                "role": self.actor.role.value,
            },
            "subject_id": self.subject_id,
            "artifacts": [
                {
                    "path": item.path,
                    "content_hash": {
                        "algorithm": item.content_hash.algorithm,
                        "digest": item.content_hash.digest,
                    },
                }
                for item in self.artifacts
            ],
            "problem_kind": None if self.problem_kind is None else self.problem_kind.value,
            "gate_authorization": (
                None if self.gate_authorization is None else self.gate_authorization.value
            ),
            "eligible_arms": list(self.eligible_arms),
            "eligible_strategies": list(self.eligible_strategies),
        }


def canonical_lifecycle_event_line(event: D1LifecycleEvent) -> bytes:
    """Return the sole admitted UTF-8 JSONL representation."""

    if type(event) is not D1LifecycleEvent:
        raise TypeError("event must be a D1LifecycleEvent")
    return (
        json.dumps(
            event.manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _error("duplicate_field", key, "duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise _error("nonfinite_json", "json", f"{value} is not finite")


def _parse_actor(value: object) -> ActorIdentity:
    actor = _exact_object(value, path="actor", fields=_ACTOR_FIELDS)
    try:
        role = ActorRole(actor["role"])
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "actor.role", "unsupported actor role") from exc
    return ActorIdentity(
        actor_id=_trimmed(actor["actor_id"], path="actor.actor_id"),
        session_id=_trimmed(actor["session_id"], path="actor.session_id"),
        role=role,
    )


def _parse_artifact(value: object, *, index: int) -> ArtifactIdentity:
    path = f"artifacts[{index}]"
    artifact = _exact_object(value, path=path, fields=_ARTIFACT_FIELDS)
    content_hash = _exact_object(
        artifact["content_hash"], path=f"{path}.content_hash", fields=_HASH_FIELDS
    )
    try:
        digest = ContentHash(
            algorithm=_trimmed(content_hash["algorithm"], path=f"{path}.content_hash.algorithm"),
            digest=_trimmed(content_hash["digest"], path=f"{path}.content_hash.digest"),
        )
        return ArtifactIdentity(
            path=_trimmed(artifact["path"], path=f"{path}.path"), content_hash=digest
        )
    except LifecycleContractError as exc:
        raise _error("invalid_artifact", path, str(exc)) from exc


def parse_lifecycle_event_line(line: bytes) -> D1LifecycleEvent:
    """Parse one event and reject permissive or noncanonical JSON."""

    if type(line) is not bytes:
        raise TypeError("line must be bytes")
    if not line.endswith(b"\n"):
        raise _error("partial_line", "line", "must end with exactly one newline")
    if b"\n" in line[:-1]:
        raise _error("multiple_events", "line", "must contain exactly one event")
    try:
        raw = json.loads(
            line[:-1].decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except D1LifecycleValueError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _error("invalid_json", "line", "must be valid UTF-8 JSON") from exc
    data = _exact_object(raw, path="event", fields=_EVENT_FIELDS)
    if data["schema_version"] != D1_LIFECYCLE_EVENT_SCHEMA:
        raise _error("unsupported_schema", "schema_version", "unsupported lifecycle schema")
    try:
        kind = D1ProcessEventKind(data["event_kind"])
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "event_kind", "unsupported event kind") from exc
    artifacts_value = data["artifacts"]
    if type(artifacts_value) is not list:
        raise _error("invalid_type", "artifacts", "must be a JSON array")
    artifacts = tuple(
        _parse_artifact(item, index=index)
        for index, item in enumerate(cast(Sequence[object], artifacts_value))
    )
    problem_value = data["problem_kind"]
    try:
        problem = None if problem_value is None else D1ProblemKind(problem_value)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "problem_kind", "unsupported problem kind") from exc
    gate_value = data["gate_authorization"]
    try:
        gate = None if gate_value is None else SuccessorAuthorization(gate_value)
    except (TypeError, ValueError) as exc:
        raise _error("invalid_enum", "gate_authorization", "unsupported gate outcome") from exc
    strategy_values = data["eligible_strategies"]
    if type(strategy_values) is not list:
        raise _error("invalid_type", "eligible_strategies", "must be a JSON array")
    strategies = tuple(
        _trimmed(item, path=f"eligible_strategies[{index}]")
        for index, item in enumerate(cast(Sequence[object], strategy_values))
    )
    arm_values = data["eligible_arms"]
    if type(arm_values) is not list:
        raise _error("invalid_type", "eligible_arms", "must be a JSON array")
    arms = tuple(
        _trimmed(item, path=f"eligible_arms[{index}]")
        for index, item in enumerate(cast(Sequence[object], arm_values))
    )
    subject_value = data["subject_id"]
    subject = None if subject_value is None else _trimmed(subject_value, path="subject_id")
    event = D1LifecycleEvent(
        rung=_trimmed(data["rung"], path="rung"),
        event_seq=_nonnegative_int(data["event_seq"], path="event_seq"),
        event_id=_trimmed(data["event_id"], path="event_id"),
        event_kind=kind,
        actor=_parse_actor(data["actor"]),
        artifacts=artifacts,
        subject_id=subject,
        problem_kind=problem,
        gate_authorization=gate,
        eligible_arms=arms,
        eligible_strategies=strategies,
    )
    if canonical_lifecycle_event_line(event) != line:
        raise _error("noncanonical_json", "line", "must use canonical lifecycle JSON")
    return event
