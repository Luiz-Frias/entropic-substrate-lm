"""Scientific authorization is bound to bytes and complete paired rosters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.ladder.gate.ladder_campaign import (
    authorize_ladder_run,
    validate_ladder_phase_transition,
)

pytestmark = pytest.mark.unit


def test_d4_learned_requires_filed_fixed_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        validate_ladder_phase_transition("d3", None, "d4", "learned")
    validate_ladder_phase_transition("d3", None, "d4", "fixed")
    validate_ladder_phase_transition("d4", "fixed", "d4", "learned")
    validate_ladder_phase_transition("d4", "learned", "d5", None)
    with pytest.raises(ValueError, match="phase"):
        validate_ladder_phase_transition("d4", "learned", "d4", "learned")


def artifact(root: Path, name: str, value: object) -> ArtifactIdentity:
    data = json.dumps(value, sort_keys=True).encode()
    (root / name).write_bytes(data)
    return ArtifactIdentity(name, ContentHash("sha256", hashlib.sha256(data).hexdigest()))


def campaign(root: Path, *, rung: str = "d1", run_class: str = "pilot") -> ArtifactIdentity:
    from .test_ladder_registration import specification

    rules = specification()
    rules["run_class"] = run_class
    arms = {arm: str(i + 1) * 64 for i, arm in enumerate(("none", "sg", "ac", "both"))}
    rules["construction_digests"] = arms
    original = rules["contrasts"][0]
    original["terms"] = {"none": -1.0, "sg": 1.0, "ac": -1.0, "both": 1.0}
    rules["seeds"] = [1, 2]
    stats = artifact(root, "statistics.json", rules)
    value = {
        "schema": "d-ladder-campaign-design-v1",
        "rung": rung,
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
        "cells": [
            {
                "logical_run_id": f"{arm}-{seed}",
                "arm": arm,
                "seed": seed,
                "pairing_id": f"pair-{seed}",
                "spec_digest": "c" * 64,
                "predecessor_strategy": "supervised",
            }
            for arm in arms
            for seed in (1, 2)
        ],
    }
    return artifact(root, "design.json", value)


def test_pilot_authorization_is_hash_bound_and_cannot_claim_confirmation(tmp_path: Path) -> None:
    design = campaign(tmp_path)
    operator = ActorIdentity("operator", "op-1", ActorRole.OPERATOR)
    auth = authorize_ladder_run(tmp_path, design, "none-1", operator=operator)
    assert auth.run_class == "pilot" and auth.arm == "none" and auth.seed == 1
    assert auth.manifest["schema"] == "d-ladder-run-authorization-v1"
    (tmp_path / "statistics.json").write_text("{}")
    with pytest.raises(ValueError, match="digest"):
        authorize_ladder_run(tmp_path, design, "none-1", operator=operator)


def test_successor_and_operator_authority_are_required(tmp_path: Path) -> None:
    design = campaign(tmp_path, rung="d2")
    operator = ActorIdentity("operator", "op-1", ActorRole.OPERATOR)
    with pytest.raises(ValueError, match="predecessor"):
        authorize_ladder_run(tmp_path, design, "none-1", operator=operator)
    with pytest.raises(ValueError, match="operator"):
        authorize_ladder_run(
            tmp_path, design, "none-1", operator=ActorIdentity("other", "x", ActorRole.OPERATOR)
        )


def test_incomplete_or_aliased_seed_roster_is_refused(tmp_path: Path) -> None:
    identity = campaign(tmp_path)
    value = json.loads((tmp_path / identity.path).read_bytes())
    value["cells"].pop()
    bad = artifact(tmp_path, "bad-design.json", value)
    with pytest.raises(ValueError, match="roster"):
        authorize_ladder_run(
            tmp_path, bad, "none-1", operator=ActorIdentity("operator", "op-1", ActorRole.OPERATOR)
        )
