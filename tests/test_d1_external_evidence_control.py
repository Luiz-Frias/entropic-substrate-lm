"""Strict external-worker controls for capture custody and preregistration bytes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_codec import (
    parse_d1_capture_aggregate_declaration,
    serialize_d1_capture_aggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    D1PreregistrationByteEvidence,
    parse_registered_design_declaration,
    serialize_registered_design,
)
from entropic_substrate.ladder.review.d1_external_analysis_control import (
    D1ExternalAnalysisControlError,
    parse_d1_capture_receipt_control,
    parse_d1_preregistration_control,
)

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_capture_service import _service
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION, _design

pytestmark = pytest.mark.unit


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _capture_case(tmp_path: Path):
    roster, complete, defects, attempts, source_root = heterogeneous_family(tmp_path)
    service, _, _ = _service(tmp_path, source_root)
    receipt = service.capture_heterogeneous(
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    declaration = parse_d1_capture_aggregate_declaration(
        serialize_d1_capture_aggregate(receipt.aggregate)
    )
    return receipt, declaration


def _preregistration_case():
    design = _design()
    declaration = parse_registered_design_declaration(serialize_registered_design(design))
    evidence = D1PreregistrationByteEvidence.from_bytes(_RAW_PREREGISTRATION)
    manifest = {
        "commit": declaration.preregistration.commit,
        "path": declaration.preregistration.path,
        "schema_version": "d1-preregistration-control-v1",
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }
    return declaration, evidence, manifest


def test_bound_capture_receipt_control_matches_aggregate_and_every_dvc_artifact(
    tmp_path: Path,
) -> None:
    receipt, aggregate = _capture_case(tmp_path)
    payload = _bytes(receipt.manifest)

    control = parse_d1_capture_receipt_control(payload, aggregate=aggregate)

    assert control.canonical_bytes == payload
    assert control.aggregate_digest == aggregate.manifest_digest
    assert control.analysis_ready is True
    assert control.pointer == receipt.pointer
    assert control.pointer_commit == receipt.pointer_commit


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("analysis_ready", False),
        ("analysis_ready", 1),
        ("aggregate_digest", "0" * 64),
        ("pointer_commit", "ABCDEF0"),
    ],
)
def test_capture_receipt_rejects_readiness_digest_or_commit_drift(
    tmp_path: Path, field: str, replacement: object
) -> None:
    receipt, aggregate = _capture_case(tmp_path)
    manifest = receipt.manifest
    manifest[field] = replacement

    with pytest.raises(D1ExternalAnalysisControlError):
        parse_d1_capture_receipt_control(_bytes(manifest), aggregate=aggregate)


def test_capture_receipt_rejects_pointer_identity_and_pending_aggregate_drift(
    tmp_path: Path,
) -> None:
    receipt, aggregate = _capture_case(tmp_path)
    for field, replacement in (
        ("pointer_path", "results/other.dvc"),
        ("pointer_sha256", "A" * 64),
        ("object_id", "f" * 32 + ".dir"),
    ):
        manifest = receipt.manifest
        manifest["pointer_identity"][field] = replacement  # type: ignore[index]
        with pytest.raises(D1ExternalAnalysisControlError, match="pointer"):
            parse_d1_capture_receipt_control(_bytes(manifest), aggregate=aggregate)

    pending_root = tmp_path / "pending"
    pending_root.mkdir()
    roster, complete, defects, attempts, root = heterogeneous_family(pending_root)
    pending = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    pending_declaration = parse_d1_capture_aggregate_declaration(
        serialize_d1_capture_aggregate(pending)
    )
    pending_receipt = receipt.manifest
    pending_receipt["aggregate_digest"] = pending_declaration.manifest_digest
    with pytest.raises(D1ExternalAnalysisControlError, match="bind"):
        parse_d1_capture_receipt_control(_bytes(pending_receipt), aggregate=pending_declaration)


def test_preregistration_control_matches_design_and_measured_byte_evidence() -> None:
    design, evidence, manifest = _preregistration_case()
    payload = _bytes(manifest)

    control = parse_d1_preregistration_control(
        payload,
        design=design,
        evidence=evidence,
    )

    assert control.canonical_bytes == payload
    assert control.path == design.preregistration.path
    assert control.commit == design.preregistration.commit
    assert control.size_bytes == len(_RAW_PREREGISTRATION)
    assert control.sha256 == evidence.sha256


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("path", "docs/other.md"),
        ("path", "../escape.md"),
        ("commit", "1" * 40),
        ("commit", "A" * 40),
        ("size_bytes", 1),
        ("size_bytes", True),
        ("sha256", "2" * 64),
        ("sha256", "A" * 64),
    ],
)
def test_preregistration_control_rejects_identity_size_or_hash_drift(
    field: str, replacement: object
) -> None:
    design, evidence, manifest = _preregistration_case()
    manifest[field] = replacement

    with pytest.raises(D1ExternalAnalysisControlError):
        parse_d1_preregistration_control(_bytes(manifest), design=design, evidence=evidence)


@pytest.mark.parametrize(
    ("parser", "payload"),
    [
        (
            "capture",
            b'{"schema_version":"d1-heterogeneous-capture-receipt-v1","schema_version":"x"}',
        ),
        ("capture", b'{"analysis_ready":NaN}'),
        ("preregistration", b" {}"),
        ("preregistration", b"{}\n"),
        ("preregistration", b"[]"),
    ],
)
def test_evidence_controls_reject_duplicate_nonfinite_noncanonical_or_nonobject_bytes(
    tmp_path: Path, parser: str, payload: bytes
) -> None:
    with pytest.raises(D1ExternalAnalysisControlError):
        if parser == "capture":
            _, aggregate = _capture_case(tmp_path)
            parse_d1_capture_receipt_control(payload, aggregate=aggregate)
        else:
            design, evidence, _ = _preregistration_case()
            parse_d1_preregistration_control(payload, design=design, evidence=evidence)
