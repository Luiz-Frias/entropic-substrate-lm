from __future__ import annotations

import hashlib
import json
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, SupportsIndex, cast

from entropic_substrate.d1_billing_finalization_values import D1BillingFinalization
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.evidence.resource_billing_reconciliation import (
    LogicalBillingReconciliation,
)
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    validate_execution_segments,
)
from entropic_substrate.evidence.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
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
from entropic_substrate.experiments.d1_primitives import (
    D1Arm,
    RunClass,
    canonical_digest,
    canonical_json,
    require_sha256,
)

_SCHEMA = "d1-finalized-cell-custody-v1"
_FINALIZATION_SCHEMA = "d1-finalized-cell-evidence-v1"
_ROOT_KEYS = frozenset(
    (
        "arm attempt_ledger billing_posture finalization intent journal logical_run_id "
        "plan_digest quality_export_ready registered_design_digest schema_version seed "
        "segment_starts trainer_outcome"
    ).split()
)
_DIGESTED_KEYS = frozenset({"manifest", "sha256"})
_JOURNAL_KEYS = frozenset({"path", "sha256"})
_MINTED: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()


class D1FinalizedCellCustodyError(ValueError):
    """Finalized-cell bytes are malformed, substituted, or internally forked."""


def _fail(message: str, cause: Exception | None = None) -> NoReturn:
    error = D1FinalizedCellCustodyError(message)
    if cause is None:
        raise error
    raise error from cause


def _sha(value: object, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label} must be SHA-256 text")
    try:
        require_sha256(value, label=label)
    except (TypeError, ValueError) as error:
        _fail(f"{label} must be lowercase SHA-256", error)
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        _fail(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _exact(value: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    if frozenset(value) != keys:
        _fail(f"{label} fields must be exact")


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        _fail(f"{label} must be nonempty trimmed text")
    return value


def _digest_envelope(text: str) -> dict[str, object]:
    return {
        "manifest": json.loads(text),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _decode_envelope(value: object, label: str) -> tuple[dict[str, object], str]:
    envelope = _mapping(value, label)
    _exact(envelope, _DIGESTED_KEYS, label)
    manifest = _mapping(envelope["manifest"], f"{label} manifest")
    text = canonical_json(manifest)
    if _sha(envelope["sha256"], f"{label}.sha256") != hashlib.sha256(text.encode()).hexdigest():
        _fail(f"{label} digest differs from its manifest")
    return manifest, text


def _finalization_manifest(finalization: D1BillingFinalization) -> dict[str, object]:
    return {
        "billing": finalization.billing.manifest,
        "journal_state": finalization.journal_state.manifest,
        "quality_export_ready": finalization.quality_export_ready,
        "schema_version": _FINALIZATION_SCHEMA,
        "terminal_event": finalization.terminal_event.manifest,
        "trainer_outcome": finalization.trainer_outcome,
    }


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1FinalizedCellCustodyRecord:
    logical_run_id: str
    arm: D1Arm
    seed: int
    registered_design_digest: str
    plan_digest: str
    intent_json: str
    segment_starts: tuple[RunStartProvenance, ...]
    attempt_ledger: D1LaunchAttemptLedger
    finalization_json: str
    journal: ArtifactIdentity
    billing_posture: str
    quality_export_ready: bool
    trainer_outcome: str
    _mint_digest: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1FinalizedCellCustodyRecord is factory-only")

    def __copy__(self) -> NoReturn:
        raise TypeError("D1 finalized-cell custody cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("D1 finalized-cell custody cannot be copied")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("D1 finalized-cell custody cannot be copied")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "arm": self.arm.value,
            "attempt_ledger": _digest_envelope(
                canonical_json(launch_attempt_ledger_to_mapping(self.attempt_ledger))
            ),
            "billing_posture": self.billing_posture,
            "finalization": _digest_envelope(self.finalization_json),
            "intent": _digest_envelope(self.intent_json),
            "journal": {
                "path": self.journal.path,
                "sha256": self.journal.content_hash.digest,
            },
            "logical_run_id": self.logical_run_id,
            "plan_digest": self.plan_digest,
            "quality_export_ready": self.quality_export_ready,
            "registered_design_digest": self.registered_design_digest,
            "schema_version": _SCHEMA,
            "seed": self.seed,
            "segment_starts": [run_start_to_mapping(item) for item in self.segment_starts],
            "trainer_outcome": self.trainer_outcome,
        }

    @property
    def record_digest(self) -> str:
        return canonical_digest(self.manifest)

    def require_factory_attested(self) -> None:
        if _MINTED.get(id(self)) is not self:
            raise TypeError("D1 finalized-cell custody was not factory-minted")
        if self.record_digest != self._mint_digest:
            raise TypeError("D1 finalized-cell custody changed after minting")


def _validate_record(record: D1FinalizedCellCustodyRecord) -> None:
    _text(record.logical_run_id, "logical_run_id")
    if type(record.arm) is not D1Arm or type(record.seed) is not int or record.seed < 0:
        _fail("finalized cell requires nominal arm and nonnegative seed")
    _sha(record.registered_design_digest, "registered_design_digest")
    _sha(record.plan_digest, "plan_digest")
    intent = _mapping(json.loads(record.intent_json), "intent manifest")
    if canonical_json(intent) != record.intent_json:
        _fail("intent manifest must use canonical JSON")
    if hashlib.sha256(record.intent_json.encode()).hexdigest() != canonical_digest(intent):
        _fail("intent digest derivation differs")
    plan = _mapping(intent.get("plan"), "intent plan")
    if (
        intent.get("plan_digest") != record.plan_digest
        or intent.get("registered_design_digest") != record.registered_design_digest
        or plan.get("logical_run_id") != record.logical_run_id
        or plan.get("arm") != record.arm.value
        or plan.get("seed") != record.seed
    ):
        _fail("intent and finalized-cell identity differ")
    validate_execution_segments(record.segment_starts)
    if record.segment_starts[0].logical_run_id != record.logical_run_id or record.segment_starts[
        -1
    ].execution_segment_index != intent.get("execution_segment_index"):
        _fail("segment starts differ from the finalized intent")
    history = tuple(
        event
        for event in record.attempt_ledger.events
        if event.identity.logical_run_id == record.logical_run_id
    )
    if (
        tuple(item.status for item in history)
        != (
            D1LaunchAttemptStatus.RECORDED,
            D1LaunchAttemptStatus.TRAINER_STARTED,
            D1LaunchAttemptStatus.COMPLETED,
        )
        or history[-1] is not record.attempt_ledger.events[-1]
    ):
        _fail("finalized cell requires one terminal completed attempt")
    identity = history[0].identity
    if (
        identity.run_class is not RunClass.CONFIRMATORY
        or identity.arm is not record.arm
        or identity.seed != record.seed
        or identity.registered_design_digest != record.registered_design_digest
    ):
        _fail("completed attempt identity differs from finalized intent")
    finalization = _mapping(json.loads(record.finalization_json), "finalization manifest")
    if canonical_json(finalization) != record.finalization_json:
        _fail("finalization manifest must use canonical JSON")
    if (
        finalization.get("schema_version") != _FINALIZATION_SCHEMA
        or record.billing_posture != "reconciled"
        or _mapping(finalization.get("billing"), "billing").get("kind") != "reconciled"
        or record.trainer_outcome != "completed"
        or finalization.get("trainer_outcome") != "completed"
        or record.quality_export_ready is not True
        or finalization.get("quality_export_ready") is not True
    ):
        _fail("finalized cell requires completed, reconciled, export-ready billing")
    state = _mapping(finalization.get("journal_state"), "finalization journal_state")
    terminal = _mapping(finalization.get("terminal_event"), "finalization terminal_event")
    if (
        state.get("logical_run_id") != record.logical_run_id
        or state.get("prefix_sha256") != record.journal.content_hash.digest
        or terminal.get("event_kind") != JournalEventKind.RUN_END.value
        or terminal.get("logical_run_id") != record.logical_run_id
        or state.get("last_event_sequence") != terminal.get("event_seq")
    ):
        _fail("final journal identity differs from successful finalization")
    if record.journal.content_hash.algorithm != "sha256":
        _fail("final journal identity must use SHA-256")


def _mint(
    *,
    logical_run_id: str,
    arm: D1Arm,
    seed: int,
    registered_design_digest: str,
    plan_digest: str,
    intent_json: str,
    segment_starts: tuple[RunStartProvenance, ...],
    attempt_ledger: D1LaunchAttemptLedger,
    finalization_json: str,
    journal: ArtifactIdentity,
) -> D1FinalizedCellCustodyRecord:
    record = object.__new__(D1FinalizedCellCustodyRecord)
    source = locals()
    for name in (
        "logical_run_id arm seed registered_design_digest plan_digest intent_json "
        "segment_starts attempt_ledger finalization_json journal"
    ).split():
        object.__setattr__(record, name, source[name])
    object.__setattr__(record, "billing_posture", "reconciled")
    object.__setattr__(record, "quality_export_ready", True)
    object.__setattr__(record, "trainer_outcome", "completed")
    object.__setattr__(record, "_mint_digest", "0" * 64)
    _validate_record(record)
    object.__setattr__(record, "_mint_digest", record.record_digest)
    _MINTED[id(record)] = record
    return record


def attest_d1_finalized_cell_custody(
    *,
    intent: D1OperatorLaunchIntent,
    segment_starts: tuple[RunStartProvenance, ...],
    attempt_ledger: D1LaunchAttemptLedger,
    finalization: D1BillingFinalization,
    journal: ArtifactIdentity,
) -> D1FinalizedCellCustodyRecord:
    if (
        type(intent) is not D1OperatorLaunchIntent
        or type(segment_starts) is not tuple
        or type(attempt_ledger) is not D1LaunchAttemptLedger
        or type(finalization) is not D1BillingFinalization
        or type(journal) is not ArtifactIdentity
    ):
        raise TypeError("finalized-cell custody requires nominal source evidence")
    if type(finalization.billing) is not LogicalBillingReconciliation:
        raise ValueError("finalized-cell custody requires reconciled provider billing")
    if finalization.trainer_outcome != "completed" or not finalization.quality_export_ready:
        raise ValueError("finalized-cell custody requires successful quality finalization")
    return _mint(
        logical_run_id=intent.plan.logical_run_id,
        arm=intent.plan.profile.arm,
        seed=intent.plan.seed,
        registered_design_digest=cast(str, intent.registered_design_digest),
        plan_digest=intent.plan.plan_digest,
        intent_json=canonical_json(intent.manifest),
        segment_starts=segment_starts,
        attempt_ledger=attempt_ledger,
        finalization_json=canonical_json(_finalization_manifest(finalization)),
        journal=journal,
    )


def serialize_d1_finalized_cell_custody(record: D1FinalizedCellCustodyRecord) -> str:
    if type(record) is not D1FinalizedCellCustodyRecord:
        raise TypeError("finalized-cell serialization requires its nominal record")
    record.require_factory_attested()
    return canonical_json(record.manifest)


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate finalized-cell field {key!r}")
        result[key] = value
    return result


def parse_d1_finalized_cell_custody(payload: str) -> D1FinalizedCellCustodyRecord:
    if type(payload) is not str or not payload:
        _fail("finalized-cell payload must be nonempty text")
    try:
        root = _mapping(json.loads(payload, object_pairs_hook=_pairs), "finalized cell")
    except D1FinalizedCellCustodyError:
        raise
    except (RecursionError, ValueError) as error:
        _fail("finalized-cell payload must be strict JSON", error)
    _exact(root, _ROOT_KEYS, "finalized cell")
    if root["schema_version"] != _SCHEMA:
        _fail("finalized-cell schema is unsupported")
    _, intent_text = _decode_envelope(root["intent"], "intent")
    ledger_manifest, _ = _decode_envelope(root["attempt_ledger"], "attempt_ledger")
    _, finalization_text = _decode_envelope(root["finalization"], "finalization")
    journal = _mapping(root["journal"], "journal")
    _exact(journal, _JOURNAL_KEYS, "journal")
    starts_raw = root["segment_starts"]
    if type(starts_raw) is not list or not starts_raw:
        _fail("segment_starts must be a nonempty array")
    try:
        attempt_ledger = launch_attempt_ledger_from_mapping(ledger_manifest)
        if canonical_digest(ledger_manifest) != attempt_ledger.ledger_digest:
            _fail("attempt ledger digest differs from its manifest")
        starts = tuple(run_start_from_mapping(item) for item in starts_raw)
        record = _mint(
            logical_run_id=_text(root["logical_run_id"], "logical_run_id"),
            arm=D1Arm(_text(root["arm"], "arm")),
            seed=cast(int, root["seed"]),
            registered_design_digest=_sha(
                root["registered_design_digest"], "registered_design_digest"
            ),
            plan_digest=_sha(root["plan_digest"], "plan_digest"),
            intent_json=intent_text,
            segment_starts=starts,
            attempt_ledger=attempt_ledger,
            finalization_json=finalization_text,
            journal=ArtifactIdentity(
                _text(journal["path"], "journal.path"),
                ContentHash("sha256", _sha(journal["sha256"], "journal.sha256")),
            ),
        )
    except D1FinalizedCellCustodyError:
        raise
    except (ResourceProvenanceCodecGap, TypeError, ValueError) as error:
        _fail(f"finalized-cell evidence is invalid: {error}", error)
    if root["billing_posture"] != "reconciled" or root["quality_export_ready"] is not True:
        _fail("finalized-cell billing and quality posture differ")
    if root["trainer_outcome"] != "completed":
        _fail("finalized-cell trainer outcome differs")
    if serialize_d1_finalized_cell_custody(record) != payload:
        _fail("finalized-cell payload must use exact canonical JSON")
    return record
