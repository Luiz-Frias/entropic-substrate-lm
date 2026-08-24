"""Typed construction contracts for the D1 exact-resume checkpoint value."""

from __future__ import annotations

from typing import cast

import pytest

from d1_checkpoint_fixtures import build_checkpoint, checkpoint_case, checkpoint_state
from entropic_substrate.evidence.resource_ledger_codec import resource_ledger_state_to_mapping
from entropic_substrate.experiments.d1_runs import D1RunSpec
from entropic_substrate.predictive_state.checkpoint_values import (
    D1_CHECKPOINT_KIND,
    D1_CHECKPOINT_SCHEMA_VERSION,
)

pytestmark = pytest.mark.unit


def test_builder_captures_typed_evidence_manifests_and_exact_identity() -> None:
    state, case = checkpoint_state()

    assert state["kind"] == D1_CHECKPOINT_KIND
    assert state["schema_version"] == D1_CHECKPOINT_SCHEMA_VERSION
    identity = cast(dict[str, object], state["identity"])
    assert identity["construction_digest"] == case.run.construction_digest
    assert identity["family_digest"] == case.run.family_digest
    assert identity["aggregation_digest"] == case.run.aggregation_digest
    assert identity["execution_digest"] == case.run.execution_digest
    assert identity["execution_protocol_digest"] == case.protocol.protocol_digest
    assert state["baseline_calibration"] == case.baseline.manifest
    assert state["resource_ledger"] == resource_ledger_state_to_mapping(case.ledger)
    readings = cast(dict[str, object], state["resource_ledger"])["last_optional_readings"]
    assert cast(dict[str, object], readings)["gpu_train_device_seconds"]
    assert state["journal"] == case.journal.manifest


def test_baseline_and_ema_travel_only_through_normal_model_state() -> None:
    state, case = checkpoint_state()
    model = cast(dict[str, object], state["model"])
    extra = cast(dict[str, object], model["_extra_state"])

    assert "mean_displacement_baseline._mean" in model
    assert "mean_displacement_baseline._fitted" in model
    assert extra["target_policy"] == case.system.target_policy_state()
    assert "baseline" not in state and "target_policy" not in state


def test_builder_accepts_a_nominal_d1_run_subtype() -> None:
    case = checkpoint_case()

    assert isinstance(case.run, D1RunSpec)
    assert type(case.run) is not D1RunSpec
    assert build_checkpoint(case)["successful_step"] == 0
