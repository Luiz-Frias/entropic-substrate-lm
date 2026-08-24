"""One durable service handoff binds every D1 complete and defect artifact."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_service import (
    D1HeterogeneousCaptureReceipt,
)

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_capture_service import _OBJECT_ID, _service

pytestmark = pytest.mark.unit


def test_heterogeneous_service_returns_one_bound_analysis_handoff(tmp_path: Path) -> None:
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, events, repo_root = _service(tmp_path, source_root)

    receipt = service.capture_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )

    assert type(receipt) is D1HeterogeneousCaptureReceipt
    assert events == ["export", "dvc_add", "dvc_push", "cloud", "git"]
    assert receipt.pointer_commit == "abcdef1"
    assert receipt.pointer.object_id == _OBJECT_ID
    assert receipt.aggregate.dvc_handoff.analysis_ready is True
    assert len(receipt.aggregate.complete_runs) == 3
    assert len(receipt.aggregate.terminal_defects) == 1
    assert len(receipt.aggregate.attempts) == 1
    assert len(receipt.aggregate.pair_exclusions) == 1
    assert all(
        artifact.dvc.object_id == _OBJECT_ID for artifact in receipt.aggregate.dvc_handoff.artifacts
    )
    assert (
        len([path for path in (repo_root / "results" / "captured").rglob("*") if path.is_file()])
        == 12
    )
    assert receipt.manifest["analysis_ready"] is True
    assert receipt.manifest["aggregate_digest"] == receipt.aggregate.manifest_digest


def test_missing_partition_cell_fails_before_export_or_dvc(tmp_path: Path) -> None:
    roster, complete, _, attempts, source_root = heterogeneous_family(tmp_path)
    service, events, _ = _service(tmp_path, source_root)

    with pytest.raises(D1CaptureTransportError) as captured:
        service.capture_heterogeneous(
            roster=roster,
            complete_inputs=complete,
            terminal_defect_inputs=(),
            attempt_evidence=attempts,
        )

    assert captured.value.code == "capture_partition"
    assert events == []


@pytest.mark.parametrize(
    ("fault", "code", "events"),
    [
        ("cloud", "cloud_mismatch", ["export", "dvc_add", "dvc_push", "cloud"]),
        (
            "git",
            "pointer_uncommitted",
            ["export", "dvc_add", "dvc_push", "cloud", "git"],
        ),
    ],
)
def test_heterogeneous_transport_never_returns_partial_analysis_readiness(
    tmp_path: Path, fault: str, code: str, events: list[str]
) -> None:
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, observed, _ = _service(
        tmp_path,
        source_root,
        cloud_present=fault != "cloud",
        git_commit=None if fault == "git" else "abcdef1",
    )

    with pytest.raises(D1CaptureTransportError) as captured:
        service.capture_heterogeneous(
            roster=roster,
            complete_inputs=complete,
            terminal_defect_inputs=defects,
            attempt_evidence=attempts,
        )

    assert captured.value.code == code
    assert observed == events


def test_receipt_refuses_a_pending_aggregate_even_with_plausible_pointer_identity(
    tmp_path: Path,
) -> None:
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, _, _ = _service(tmp_path, source_root)
    ready = service.capture_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )

    with pytest.raises(ValueError, match="pending"):
        D1HeterogeneousCaptureReceipt(
            aggregate=build_d1_capture_aggregate(
                source_root,
                roster=roster,
                complete_inputs=complete,
                terminal_defect_inputs=defects,
                attempt_evidence=attempts,
            ),
            pointer=ready.pointer,
            pointer_commit=ready.pointer_commit,
        )
