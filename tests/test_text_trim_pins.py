"""Pin each codec's own `_text` trim boundary at today's semantics (FOR-1614).

No semantic change lands in this rung (operator ruling 12, 2026-09-28): these
pins record what `d1_launch_attempt_codec._text` (permissive: untrimmed but
nonblank passes and round-trips byte for byte), `d1_registration_codec._text`
(strict: any value whose `.strip()` differs is rejected) and
`ladder_campaign._text` (strict, same rule) each do RIGHT NOW, through their
own public parse entry point, on a field whose only admission gate is that
codec's `_text` (verified against the source at
`refactor/p14-consolidation`). A guard-teeth mutation (flip the boundary,
show RED, revert, show GREEN) is demonstrated for each pin in the PR body
rather than in this module, per the brief's "no `_text` semantics change in
commit 1" rule and the sacred invariant that the committed module never
monkeypatches `_text` itself.

Fixture note for the conductor: `decision_contract()` is imported from
`tests/d1_verdict_v2_fixtures.py`, which is in `p14-t-verdict-vocab`'s touch
set, not this rung's. It is a pure builder with no side effects; nothing
here mutates or depends on that phase's later changes to it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from d1_verdict_v2_fixtures import decision_contract
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.gate.ladder_campaign import read_campaign_design
from entropic_substrate.ladder.register.d1_launch_attempt_codec import (
    D1LaunchAttemptCodecError,
    parse_launch_attempt_ledger,
    serialize_launch_attempt_ledger,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1RegistrationCodecError,
    decision_contract_to_mapping,
    parse_decision_contract,
)

pytestmark = pytest.mark.unit

_UNTRIMMED = (" run-1", "run-1 ", "run-1\n")


# --- launch-attempt: permissive `_text`, pinned via the public parse function ---


def _launch_attempt_identity_manifest() -> dict[str, object]:
    return {
        "arm": "none",
        "family_digest": "a" * 64,
        "registered_design_digest": None,
        "run_class": "pilot",
        "schema_version": "d1-launch-attempt-v1",
        "seed": 1,
    }


def _launch_ledger_payload(*, logical_run_id: str = "run-1", status: str = "recorded") -> str:
    manifest = _launch_attempt_identity_manifest()
    document = {
        "events": [
            {
                "event_seq": 0,
                "identity": {
                    **manifest,
                    "logical_run_id": logical_run_id,
                    "optional_stopping_digest": canonical_digest(manifest),
                },
                "observed_at": "2026-09-30T00:00:00+00:00",
                "reason": None,
                "status": status,
            }
        ],
        "schema_version": "d1-launch-ledger-v1",
    }
    return json.dumps(
        document, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


@pytest.mark.parametrize("value", _UNTRIMMED)
def test_launch_attempt_text_accepts_untrimmed_logical_run_id_and_round_trips(
    value: str,
) -> None:
    payload = _launch_ledger_payload(logical_run_id=value)

    ledger = parse_launch_attempt_ledger(payload)

    assert ledger.events[0].identity.logical_run_id == value
    assert serialize_launch_attempt_ledger(ledger) == payload


def test_launch_attempt_text_rejects_blank_logical_run_id() -> None:
    payload = _launch_ledger_payload(logical_run_id="")

    with pytest.raises(D1LaunchAttemptCodecError) as excinfo:
        parse_launch_attempt_ledger(payload)

    assert type(excinfo.value) is D1LaunchAttemptCodecError
    assert (
        str(excinfo.value)
        == "attempt identity is invalid: logical_run_id must be a nonempty string"
    )


def test_launch_attempt_text_rejects_blank_status_unwrapped_at_event_level() -> None:
    """Row-4 vehicle (guard-teeth table): an EVENT-level field, not identity-level.

    `_identity` catches `(TypeError, ValueError)` and rewraps every `_text`
    failure as "attempt identity is invalid: {error}" regardless of whether
    `_text` raises `D1LaunchAttemptCodecError` or a bare `TypeError`, so an
    identity-level field (`logical_run_id`, `family_digest`, `arm`,
    `run_class`) cannot distinguish the "raise TypeError instead of the
    site's own error" mutation from today's code: only `__cause__` differs,
    which this pin never asserts. `_event` instead re-raises
    `D1LaunchAttemptCodecError` UNWRAPPED (`except D1LaunchAttemptCodecError:
    raise`, checked before the rewrap clause), so today's bare message
    ("status must be a nonempty string") is caller-observably different from
    the mutation's wrapped one ("attempt event is invalid: ..."). Measured
    directly against both arms before authoring this pin.
    """
    payload = _launch_ledger_payload(status="")

    with pytest.raises(D1LaunchAttemptCodecError) as excinfo:
        parse_launch_attempt_ledger(payload)

    assert type(excinfo.value) is D1LaunchAttemptCodecError
    assert str(excinfo.value) == "status must be a nonempty string"


# --- registration: strict `_text`, pinned on `verdict_vocabulary` ---


def _decision_contract_payload(*, verdict_vocabulary: str) -> str:
    mapping = decision_contract_to_mapping(decision_contract())
    mapping["verdict_vocabulary"] = verdict_vocabulary
    return json.dumps(
        mapping, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


@pytest.mark.parametrize("value", _UNTRIMMED)
def test_registration_text_rejects_untrimmed_verdict_vocabulary(value: str) -> None:
    payload = _decision_contract_payload(verdict_vocabulary=value)

    with pytest.raises(D1RegistrationCodecError) as excinfo:
        parse_decision_contract(payload)

    assert type(excinfo.value) is D1RegistrationCodecError
    assert str(excinfo.value) == "verdict_vocabulary must be a nonempty trimmed string"


def test_registration_text_rejects_blank_verdict_vocabulary() -> None:
    payload = _decision_contract_payload(verdict_vocabulary="")

    with pytest.raises(D1RegistrationCodecError) as excinfo:
        parse_decision_contract(payload)

    assert type(excinfo.value) is D1RegistrationCodecError
    assert str(excinfo.value) == "verdict_vocabulary must be a nonempty trimmed string"


# --- campaign: strict `_text`, pinned on `cells[i].logical_run_id` ---


def _artifact_file(root: Path, name: str, value: object) -> ArtifactIdentity:
    data = json.dumps(value, sort_keys=True).encode()
    (root / name).write_bytes(data)
    return ArtifactIdentity(name, ContentHash("sha256", hashlib.sha256(data).hexdigest()))


def _campaign_design_artifact(root: Path, *, none_seed_one_logical_run_id: str) -> ArtifactIdentity:
    statistical_design = {
        "schema": "d-ladder-statistical-design-v1",
        "run_class": "pilot",
        "registered_design_digest": None,
        "family_digest": "4" * 64,
        "construction_digests": {
            arm: str(index + 1) * 64 for index, arm in enumerate(("none", "sg", "ac", "both"))
        },
        "seeds": [1, 2],
        "contrasts": [
            {
                "contrast_id": "sg-vs-none",
                "terms": {"none": -1.0, "sg": 1.0, "ac": -1.0, "both": 1.0},
                "metric": "nats_per_byte",
                "units": "nats/byte",
                "eval_steps": [1, 2],
                "source_bytes": 8,
                "kind": "superiority",
                "expected_effect": 0.01,
                "lower_bound": 0.0,
                "upper_bound": None,
            }
        ],
    }
    stats = _artifact_file(root, "statistics.json", statistical_design)
    cells = []
    for arm in ("none", "sg", "ac", "both"):
        for seed in (1, 2):
            logical_run_id = (
                none_seed_one_logical_run_id if (arm, seed) == ("none", 1) else f"{arm}-{seed}"
            )
            cells.append(
                {
                    "logical_run_id": logical_run_id,
                    "arm": arm,
                    "seed": seed,
                    "pairing_id": f"pair-{seed}",
                    "spec_digest": "c" * 64,
                    "predecessor_strategy": "supervised",
                }
            )
    design = {
        "schema": "d-ladder-campaign-design-v1",
        "rung": "d1",
        "d4_phase": None,
        "statistics": {"path": stats.path, "sha256": stats.content_hash.digest},
        "preregistration": None,
        "registration_git": None,
        "implementation": {
            "actor_id": "builder",
            "session_id": "build-1",
            "role": "implementation",
        },
        "operator": {"actor_id": "operator", "session_id": "op-1", "role": "operator"},
        "execution_protocol_digest": "a" * 64,
        "source_sha256": "b" * 64,
        "source_split": {
            "train_start": 0,
            "train_end": 100,
            "eval_start": 100,
            "eval_end": 120,
            "source_total_bytes": 120,
        },
        "reader_identity": "scientific-reader-v1",
        "device": "cpu",
        "cells": cells,
    }
    return _artifact_file(root, "design.json", design)


@pytest.mark.parametrize("value", _UNTRIMMED)
def test_campaign_text_rejects_untrimmed_logical_run_id(value: str) -> None:
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root).resolve()
        design = _campaign_design_artifact(root, none_seed_one_logical_run_id=value)

        with pytest.raises(ValueError) as excinfo:
            read_campaign_design(root, design)

        assert type(excinfo.value) is ValueError
        assert str(excinfo.value) == "ladder identifier must be nonempty trimmed text"


def test_campaign_text_rejects_blank_logical_run_id() -> None:
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root).resolve()
        design = _campaign_design_artifact(root, none_seed_one_logical_run_id="")

        with pytest.raises(ValueError) as excinfo:
            read_campaign_design(root, design)

        assert type(excinfo.value) is ValueError
        assert str(excinfo.value) == "ladder identifier must be nonempty trimmed text"
