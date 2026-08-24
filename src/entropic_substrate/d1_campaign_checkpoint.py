"""Canonical campaign restart state above the D1 attempt and lifecycle ledgers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn, cast

from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.evidence.resource_provenance_codec import (
    run_start_from_mapping,
    run_start_to_mapping,
)
from entropic_substrate.experiments.d1_launch_attempt_codec import (
    launch_attempt_ledger_from_mapping,
    launch_attempt_ledger_to_mapping,
)
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, require_sha256

_SCHEMA = "d1-campaign-checkpoint-v1"
_ROOT_KEYS = frozenset(
    {
        "attempt_ledger",
        "completed_cells",
        "decision_contract_digest",
        "design_digest",
        "disposition",
        "halt_reason",
        "lifecycle",
        "next_index",
        "schema_version",
    }
)
_CELL_KEYS = frozenset(
    {
        "arm",
        "index",
        "intent_digest",
        "ledger_digest_after",
        "lifecycle_event_count_after",
        "lifecycle_prefix_after",
        "logical_run_id",
        "plan_digest",
        "seed",
        "start",
    }
)
_LIFECYCLE_KEYS = frozenset({"active_subject", "event_count", "prefix_sha256", "process_state"})


class D1CampaignCheckpointError(ValueError):
    """Persisted campaign state is malformed, forked, or noncanonical."""


class D1CampaignDisposition(StrEnum):
    READY = "ready"
    HARD_HALT = "hard_halt"


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _sha(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    require_sha256(text, label=label)
    return text


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class D1CampaignCompletedCell:
    index: int
    logical_run_id: str
    arm: D1Arm
    seed: int
    plan_digest: str
    intent_digest: str
    start: RunStartProvenance
    ledger_digest_after: str
    lifecycle_prefix_after: str
    lifecycle_event_count_after: int

    def __post_init__(self) -> None:
        _integer(self.index, label="completed cell index")
        _text(self.logical_run_id, label="completed cell logical_run_id")
        if type(self.arm) is not D1Arm:
            raise TypeError("completed cell arm must be D1Arm")
        _integer(self.seed, label="completed cell seed")
        for label, value in (
            ("plan_digest", self.plan_digest),
            ("intent_digest", self.intent_digest),
            ("ledger_digest_after", self.ledger_digest_after),
            ("lifecycle_prefix_after", self.lifecycle_prefix_after),
        ):
            _sha(value, label=label)
        if type(self.start) is not RunStartProvenance:
            raise TypeError("completed cell start must be RunStartProvenance")
        if self.start.logical_run_id != self.logical_run_id:
            raise ValueError("completed cell start logical identity differs")
        if self.start.execution_segment_index != 0:
            raise ValueError("completed campaign cell must begin with fresh segment zero")
        if self.lifecycle_event_count_after < 1:
            raise ValueError("completed cell lifecycle event count must be positive")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "index": self.index,
            "intent_digest": self.intent_digest,
            "ledger_digest_after": self.ledger_digest_after,
            "lifecycle_event_count_after": self.lifecycle_event_count_after,
            "lifecycle_prefix_after": self.lifecycle_prefix_after,
            "logical_run_id": self.logical_run_id,
            "plan_digest": self.plan_digest,
            "seed": self.seed,
            "start": run_start_to_mapping(self.start),
        }


@dataclass(frozen=True, slots=True)
class D1CampaignCheckpoint:
    design_digest: str
    decision_contract_digest: str
    completed_cells: tuple[D1CampaignCompletedCell, ...]
    attempt_ledger: D1LaunchAttemptLedger
    process_state: D1ProcessState
    lifecycle_prefix: str
    lifecycle_event_count: int
    active_subject: str | None
    disposition: D1CampaignDisposition
    halt_reason: str | None

    def __post_init__(self) -> None:
        _sha(self.design_digest, label="campaign design_digest")
        _sha(self.decision_contract_digest, label="campaign decision_contract_digest")
        if type(self.completed_cells) is not tuple or any(
            type(cell) is not D1CampaignCompletedCell for cell in self.completed_cells
        ):
            raise TypeError("campaign completed_cells must be a nominal tuple")
        if tuple(cell.index for cell in self.completed_cells) != tuple(
            range(len(self.completed_cells))
        ):
            raise ValueError("campaign completed cell indices must be consecutive from zero")
        if len({cell.logical_run_id for cell in self.completed_cells}) != len(self.completed_cells):
            raise ValueError("campaign completed logical identities must be unique")
        if type(self.attempt_ledger) is not D1LaunchAttemptLedger:
            raise TypeError("campaign checkpoint requires D1LaunchAttemptLedger")
        if type(self.process_state) is not D1ProcessState:
            raise TypeError("campaign process_state must be D1ProcessState")
        _sha(self.lifecycle_prefix, label="campaign lifecycle_prefix")
        if type(self.lifecycle_event_count) is not int or self.lifecycle_event_count < 1:
            raise ValueError("campaign lifecycle_event_count must be positive")
        if self.active_subject is not None:
            _text(self.active_subject, label="campaign active_subject")
        if type(self.disposition) is not D1CampaignDisposition:
            raise TypeError("campaign disposition must be nominal")
        self._validate_completed_prefix()
        self._validate_disposition()

    def _validate_completed_prefix(self) -> None:
        prior_count = 0
        for cell in self.completed_cells:
            history = tuple(
                event
                for event in self.attempt_ledger.events
                if event.identity.logical_run_id == cell.logical_run_id
            )
            if tuple(event.status for event in history) != (
                D1LaunchAttemptStatus.RECORDED,
                D1LaunchAttemptStatus.TRAINER_STARTED,
                D1LaunchAttemptStatus.COMPLETED,
            ):
                raise ValueError("completed cell lacks one terminal completed attempt history")
            identity = history[0].identity
            if (
                identity.arm is not cell.arm
                or identity.seed != cell.seed
                or identity.registered_design_digest != self.design_digest
            ):
                raise ValueError("completed cell identity differs from its attempt history")
            prefix = D1LaunchAttemptLedger(self.attempt_ledger.events[: history[-1].event_seq + 1])
            if prefix.ledger_digest != cell.ledger_digest_after:
                raise ValueError("completed cell ledger prefix digest differs")
            if cell.lifecycle_event_count_after <= prior_count:
                raise ValueError("completed cell lifecycle counts must increase")
            prior_count = cell.lifecycle_event_count_after
        if self.completed_cells:
            last = self.completed_cells[-1]
            if (
                last.lifecycle_prefix_after != self.lifecycle_prefix
                or last.lifecycle_event_count_after != self.lifecycle_event_count
            ):
                raise ValueError("campaign lifecycle head differs from completed prefix")

    def _validate_disposition(self) -> None:
        completed_ids = {cell.logical_run_id for cell in self.completed_cells}
        ledger_ids = {event.identity.logical_run_id for event in self.attempt_ledger.events}
        if self.disposition is D1CampaignDisposition.READY:
            expected_state = (
                D1ProcessState.OPERATOR_FINALIZED
                if self.completed_cells
                else D1ProcessState.PREREGISTERED
            )
            if (
                ledger_ids != completed_ids
                or self.process_state is not expected_state
                or self.active_subject is not None
                or self.halt_reason is not None
            ):
                raise ValueError("READY checkpoint requires an exact terminal completed prefix")
            return
        if self.halt_reason is None:
            raise ValueError("HARD_HALT checkpoint requires a halt reason")
        _text(self.halt_reason, label="campaign halt_reason")

    @property
    def next_index(self) -> int:
        return len(self.completed_cells)

    @property
    def next_action(self) -> str:
        if self.disposition is D1CampaignDisposition.HARD_HALT:
            return "hard_halt"
        return f"launch_registered_cell_{self.next_index}"

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "attempt_ledger": launch_attempt_ledger_to_mapping(self.attempt_ledger),
            "completed_cells": [cell.manifest for cell in self.completed_cells],
            "decision_contract_digest": self.decision_contract_digest,
            "design_digest": self.design_digest,
            "disposition": self.disposition.value,
            "halt_reason": self.halt_reason,
            "lifecycle": {
                "active_subject": self.active_subject,
                "event_count": self.lifecycle_event_count,
                "prefix_sha256": self.lifecycle_prefix,
                "process_state": self.process_state.value,
            },
            "next_index": self.next_index,
            "schema_version": _SCHEMA,
        }


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CampaignCheckpointError(message)
    raise D1CampaignCheckpointError(message) from error


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], *, label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} keys must be exact")


def _cell(value: object) -> D1CampaignCompletedCell:
    item = _mapping(value, label="completed cell")
    _exact(item, _CELL_KEYS, label="completed cell")
    try:
        return D1CampaignCompletedCell(
            _integer(item["index"], label="completed cell index"),
            _text(item["logical_run_id"], label="logical_run_id"),
            D1Arm(_text(item["arm"], label="arm")),
            _integer(item["seed"], label="seed"),
            _sha(item["plan_digest"], label="plan_digest"),
            _sha(item["intent_digest"], label="intent_digest"),
            run_start_from_mapping(item["start"]),
            _sha(item["ledger_digest_after"], label="ledger_digest_after"),
            _sha(item["lifecycle_prefix_after"], label="lifecycle_prefix_after"),
            _integer(item["lifecycle_event_count_after"], label="lifecycle event count"),
        )
    except (TypeError, ValueError) as error:
        _fail(f"completed cell is invalid: {error}", error)


def campaign_checkpoint_to_mapping(checkpoint: D1CampaignCheckpoint) -> dict[str, object]:
    if type(checkpoint) is not D1CampaignCheckpoint:
        raise TypeError("campaign checkpoint codec requires D1CampaignCheckpoint")
    return checkpoint.manifest


def campaign_checkpoint_from_mapping(value: object) -> D1CampaignCheckpoint:
    root = _mapping(value, label="campaign checkpoint")
    _exact(root, _ROOT_KEYS, label="campaign checkpoint")
    if root["schema_version"] != _SCHEMA:
        _fail("campaign checkpoint schema_version is unsupported")
    raw_cells = root["completed_cells"]
    if type(raw_cells) is not list:
        _fail("campaign completed_cells must be a JSON array")
    lifecycle = _mapping(root["lifecycle"], label="campaign lifecycle")
    _exact(lifecycle, _LIFECYCLE_KEYS, label="campaign lifecycle")
    active = lifecycle["active_subject"]
    halt = root["halt_reason"]
    if active is not None and type(active) is not str:
        _fail("campaign active_subject must be a string or null")
    if halt is not None and type(halt) is not str:
        _fail("campaign halt_reason must be a string or null")
    try:
        checkpoint = D1CampaignCheckpoint(
            _sha(root["design_digest"], label="design_digest"),
            _sha(root["decision_contract_digest"], label="decision_contract_digest"),
            tuple(_cell(item) for item in raw_cells),
            launch_attempt_ledger_from_mapping(root["attempt_ledger"]),
            D1ProcessState(_text(lifecycle["process_state"], label="process_state")),
            _sha(lifecycle["prefix_sha256"], label="lifecycle prefix"),
            _integer(lifecycle["event_count"], label="lifecycle event_count"),
            active,
            D1CampaignDisposition(_text(root["disposition"], label="disposition")),
            halt,
        )
    except D1CampaignCheckpointError:
        raise
    except (TypeError, ValueError) as error:
        _fail(f"campaign checkpoint is invalid: {error}", error)
    if root["next_index"] != checkpoint.next_index:
        _fail("campaign checkpoint next_index differs from completed prefix")
    return checkpoint


def serialize_campaign_checkpoint(checkpoint: D1CampaignCheckpoint) -> str:
    return json.dumps(
        campaign_checkpoint_to_mapping(checkpoint),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate campaign checkpoint key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite campaign checkpoint constant {value!r}")


def parse_campaign_checkpoint(payload: str) -> D1CampaignCheckpoint:
    if type(payload) is not str or not payload:
        _fail("campaign checkpoint payload must be nonempty text")
    try:
        value: object = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except D1CampaignCheckpointError:
        raise
    except json.JSONDecodeError as error:
        _fail("campaign checkpoint payload is invalid JSON", error)
    checkpoint = campaign_checkpoint_from_mapping(value)
    if serialize_campaign_checkpoint(checkpoint) != payload:
        _fail("campaign checkpoint JSON is not canonical")
    return checkpoint
