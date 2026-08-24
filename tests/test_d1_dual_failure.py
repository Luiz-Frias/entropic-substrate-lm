"""Primary trainer defects and secondary meter defects remain independently replayable."""

from __future__ import annotations

import pytest

from entropic_substrate.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
    secondary_failure_from_mapping,
)

pytestmark = pytest.mark.unit


def _secondary() -> D1SecondaryFailureEvidence:
    return D1SecondaryFailureEvidence.from_error(
        D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
        RuntimeError("meter snapshot failed"),
    )


def test_secondary_failure_has_closed_canonical_round_trip() -> None:
    evidence = _secondary()

    assert evidence.manifest == {
        "error_message": "meter snapshot failed",
        "error_type": "RuntimeError",
        "schema_version": "d1-secondary-failure/v1",
        "stage": "terminal-meter-refresh",
    }
    assert secondary_failure_from_mapping(evidence.manifest) == evidence


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"invented": True}),
        lambda value: value.pop("error_type"),
        lambda value: value.update({"stage": "somewhere-else"}),
        lambda value: value.update({"error_message": ""}),
    ],
)
def test_secondary_failure_replay_refuses_unknown_missing_or_open_vocabulary(
    mutation: object,
) -> None:
    mapping = dict(_secondary().manifest)
    mutation(mapping)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        secondary_failure_from_mapping(mapping)


def test_nominal_dual_failure_retains_both_exception_objects_and_primary_causality() -> None:
    primary = ValueError("scientific defect")
    secondary = RuntimeError("meter snapshot failed")
    combined = D1PrimarySecondaryFailure(primary, secondary)

    assert combined.primary is primary
    assert combined.secondary is secondary
    assert combined.exceptions == (primary, secondary)
    assert combined.secondary_evidence == _secondary()
    assert "scientific defect" in str(combined)
    assert "meter snapshot failed" in str(combined)
