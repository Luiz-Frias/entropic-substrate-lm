"""CLI-neutral orchestration of the authoritative D1 capture transaction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1DvcBinding,
    D1DvcBindingState,
)
from entropic_substrate.evidence.d1_capture_bundle import D1CaptureInput
from entropic_substrate.evidence.d1_capture_filesystem import LocalD1EvidenceExportSource
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.evidence.d1_capture_service import D1CaptureService

from .d1_capture_fixtures import capture_family

pytestmark = pytest.mark.unit

_CAPTURE_PATH = "results/captured"
_POINTER_PATH = "results/captured.dvc"
_OBJECT_ID = "8" * 32 + ".dir"


def _artifact_pointer(artifact: D1ArtifactIdentity) -> D1ArtifactIdentity:
    return replace(artifact, dvc=D1DvcBinding.pending(_POINTER_PATH))


def _capture_pointer(value: D1CaptureInput) -> D1CaptureInput:
    return replace(
        value,
        journal=_artifact_pointer(value.journal),
        checkpoint=_artifact_pointer(value.checkpoint),
        billing_sidecars=tuple(_artifact_pointer(item) for item in value.billing_sidecars),
        quarantined_siblings=tuple(_artifact_pointer(item) for item in value.quarantined_siblings),
    )


@dataclass
class _ExportSource:
    source_root: Path
    events: list[str]

    def export(
        self,
        *,
        artifacts: tuple[D1ArtifactIdentity, ...],
        destination_root: Path,
    ) -> tuple[D1ArtifactIdentity, ...]:
        self.events.append("export")
        return LocalD1EvidenceExportSource(self.source_root).export(
            artifacts=artifacts,
            destination_root=destination_root,
        )


@dataclass
class _Dvc:
    repo_root: Path
    events: list[str]
    fail_at: str | None = None
    pointer_target: str = "captured"

    def add(self, *, tracked_path: Path, pointer_path: Path) -> None:
        self.events.append("dvc_add")
        assert tracked_path == self.repo_root / _CAPTURE_PATH
        assert pointer_path == self.repo_root / _POINTER_PATH
        if self.fail_at == "add":
            raise RuntimeError("simulated dvc add failure")
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(
            f"outs:\n- md5: {_OBJECT_ID}\n  size: 99\n  nfiles: 12\n  path: {self.pointer_target}\n",
            encoding="utf-8",
        )

    def push(self, *, pointer_path: Path) -> None:
        self.events.append("dvc_push")
        assert pointer_path == self.repo_root / _POINTER_PATH
        if self.fail_at == "push":
            raise RuntimeError("simulated dvc push failure")


@dataclass
class _Cloud:
    events: list[str]
    present: bool = True

    def object_is_present(self, *, pointer_path: Path, object_id: str) -> bool:
        self.events.append("cloud")
        assert pointer_path.name == "captured.dvc"
        assert object_id == _OBJECT_ID
        return self.present


@dataclass
class _Git:
    events: list[str]
    commit: str | None = "abcdef1"
    observed_sha256: str | None = field(default=None, init=False)

    def committed_revision(self, *, pointer_path: Path, pointer_sha256: str) -> str | None:
        self.events.append("git")
        assert pointer_sha256 == hashlib.sha256(pointer_path.read_bytes()).hexdigest()
        self.observed_sha256 = pointer_sha256
        return self.commit


def _service(
    tmp_path: Path,
    source_root: Path,
    *,
    dvc_fail_at: str | None = None,
    cloud_present: bool = True,
    git_commit: str | None = "abcdef1",
    pointer_target: str = "captured",
):
    events: list[str] = []
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    return (
        D1CaptureService(
            repo_root=repo_root,
            evidence_source=_ExportSource(source_root, events),
            dvc=_Dvc(repo_root, events, dvc_fail_at, pointer_target),
            cloud=_Cloud(events, cloud_present),
            git=_Git(events, git_commit),
        ),
        events,
        repo_root,
    )


def test_service_binds_filesystem_dvc_cloud_and_git_identity_in_order(tmp_path: Path) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    inputs = tuple(_capture_pointer(item) for item in raw_inputs)
    service, events, repo_root = _service(tmp_path, source_root)

    receipt = service.capture(roster=roster, capture_inputs=inputs)

    assert events == ["export", "dvc_add", "dvc_push", "cloud", "git"]
    assert receipt.pointer.pointer_path == _POINTER_PATH
    assert receipt.pointer.object_id == _OBJECT_ID
    assert receipt.pointer_commit == "abcdef1"
    assert receipt.bundle.dvc_handoff.analysis_ready is True
    assert all(
        artifact.dvc.state is D1DvcBindingState.BOUND
        for artifact in receipt.bundle.dvc_handoff.artifacts
    )
    assert {artifact.dvc.object_id for artifact in receipt.bundle.dvc_handoff.artifacts} == {
        _OBJECT_ID
    }
    assert len(list((repo_root / _CAPTURE_PATH).rglob("*"))) == 12


@pytest.mark.parametrize(
    ("fault", "code", "expected_events"),
    [
        ("dvc_add", "dvc_failure", ["export", "dvc_add"]),
        ("dvc_push", "dvc_failure", ["export", "dvc_add", "dvc_push"]),
        ("cloud", "cloud_mismatch", ["export", "dvc_add", "dvc_push", "cloud"]),
        (
            "git",
            "pointer_uncommitted",
            ["export", "dvc_add", "dvc_push", "cloud", "git"],
        ),
        (
            "pointer",
            "pointer_identity",
            ["export", "dvc_add", "dvc_push"],
        ),
    ],
)
def test_service_failures_are_typed_and_never_claim_analysis_ready(
    tmp_path: Path, fault: str, code: str, expected_events: list[str]
) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    inputs = tuple(_capture_pointer(item) for item in raw_inputs)
    service, events, _ = _service(
        tmp_path,
        source_root,
        dvc_fail_at=fault.removeprefix("dvc_") if fault.startswith("dvc_") else None,
        cloud_present=fault != "cloud",
        git_commit=None if fault == "git" else "abcdef1",
        pointer_target="wrong-directory" if fault == "pointer" else "captured",
    )

    with pytest.raises(D1CaptureTransportError) as captured:
        service.capture(roster=roster, capture_inputs=inputs)

    assert captured.value.code == code
    assert events == expected_events


def test_missing_roster_cell_and_reader_refusal_have_no_export_or_live_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    inputs = tuple(_capture_pointer(item) for item in raw_inputs)
    service, events, _ = _service(tmp_path, source_root)

    with pytest.raises(D1CaptureTransportError) as missing:
        service.capture(roster=roster, capture_inputs=inputs[:-1])
    assert missing.value.code == "capture_input_set"
    assert events == []

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise ValueError("public reader refused staged evidence")

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", refuse)
    with pytest.raises(D1CaptureTransportError) as inadmissible:
        service.capture(roster=roster, capture_inputs=inputs)
    assert inadmissible.value.code == "local_admission"
    assert events == ["export"]
