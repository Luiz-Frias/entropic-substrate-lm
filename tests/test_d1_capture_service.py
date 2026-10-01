"""CLI-neutral orchestration of the authoritative D1 capture transaction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

import entropic_substrate.ladder.capture.reader.telemetry as telemetry_api
from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1DvcBinding,
    D1DvcBindingState,
)
from entropic_substrate.ladder.capture.custody.d1_capture_bundle import D1CaptureInput
from entropic_substrate.ladder.capture.custody.d1_capture_filesystem import (
    LocalD1EvidenceExportSource,
)
from entropic_substrate.ladder.capture.custody.d1_capture_ports import (
    D1CaptureTransportError,
    D1DvcPointerIdentity,
    D1GitPointerCommitEvidence,
)
from entropic_substrate.ladder.capture.custody.d1_capture_service import (
    D1CapturePreparationStatus,
    D1CaptureService,
)

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
    pointer_sha256_override: str | None = None
    object_id_override: str | None = None
    observed_sha256: str | None = field(default=None, init=False)

    def committed_pointer(
        self,
        *,
        pointer_path: Path,
        pointer_bytes: bytes,
        pointer_sha256: str,
        object_id: str,
    ) -> D1GitPointerCommitEvidence | None:
        self.events.append("git")
        assert pointer_sha256 == hashlib.sha256(pointer_bytes).hexdigest()
        self.observed_sha256 = pointer_sha256
        if self.commit is None:
            return None
        return D1GitPointerCommitEvidence(
            self.commit,
            _POINTER_PATH,
            self.pointer_sha256_override or pointer_sha256,
            self.object_id_override or object_id,
        )


def _service(
    tmp_path: Path,
    source_root: Path,
    *,
    dvc_fail_at: str | None = None,
    cloud_present: bool = True,
    git_commit: str | None = "abcdef1",
    git_pointer_sha256: str | None = None,
    git_object_id: str | None = None,
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
            git=_Git(events, git_commit, git_pointer_sha256, git_object_id),
        ),
        events,
        repo_root,
    )


def test_service_binds_filesystem_dvc_cloud_and_git_identity_in_order(tmp_path: Path) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    inputs = tuple(_capture_pointer(item) for item in raw_inputs)
    service, events, repo_root = _service(tmp_path, source_root)

    preparation = service.prepare(roster=roster, capture_inputs=inputs)

    assert preparation.status is D1CapturePreparationStatus.COMMIT_REQUIRED
    assert events == ["export", "dvc_add", "dvc_push", "cloud"]
    assert preparation.pointer_bytes == (repo_root / _POINTER_PATH).read_bytes()
    receipt = service.finalize(preparation)

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


@pytest.mark.parametrize("mutation", ["pointer_bytes", "commit_hash", "object_id"])
def test_finalize_rejects_pointer_or_commit_evidence_substitution(
    tmp_path: Path,
    mutation: str,
) -> None:
    roster, raw_inputs, source_root = capture_family(tmp_path)
    inputs = tuple(_capture_pointer(item) for item in raw_inputs)
    service, events, repo_root = _service(
        tmp_path,
        source_root,
        git_pointer_sha256="0" * 64 if mutation == "commit_hash" else None,
        git_object_id="0" * 32 + ".dir" if mutation == "object_id" else None,
    )
    preparation = service.prepare(roster=roster, capture_inputs=inputs)
    if mutation == "pointer_bytes":
        (repo_root / _POINTER_PATH).write_bytes(preparation.pointer_bytes + b"\n")

    with pytest.raises(D1CaptureTransportError):
        service.finalize(preparation)

    assert events[-1] == ("cloud" if mutation == "pointer_bytes" else "git")


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


# FOR-1602: each class validates through ``self.binding`` so one site knows the
# D1DvcBinding constructor shape. Construction must still refuse eagerly, with
# the exceptions D1DvcBinding itself raises (messages measured at the parent).
_GOOD_SHA = "a" * 64
_GOOD_COMMIT = "b" * 40


def _pointer_identity(pointer_path: str, pointer_sha256: str, object_id: str) -> object:
    return D1DvcPointerIdentity(pointer_path, pointer_sha256, object_id)


def _commit_evidence(pointer_path: str, pointer_sha256: str, object_id: str) -> object:
    return D1GitPointerCommitEvidence(_GOOD_COMMIT, pointer_path, pointer_sha256, object_id)


_INVALID_BINDINGS = [
    ("results/captured.txt", _GOOD_SHA, _OBJECT_ID, "DVC pointer_path must name a .dvc file"),
    (
        "/abs/captured.dvc",
        _GOOD_SHA,
        _OBJECT_ID,
        "DVC pointer_path must be a normalized repository-relative POSIX path",
    ),
    ("", _GOOD_SHA, _OBJECT_ID, "DVC pointer_path must be a nonempty path"),
    (
        _POINTER_PATH,
        "A" * 64,
        _OBJECT_ID,
        "DVC pointer_sha256 must be 64 lowercase hexadecimal characters",
    ),
    (
        _POINTER_PATH,
        "a" * 63,
        _OBJECT_ID,
        "DVC pointer_sha256 must be 64 lowercase hexadecimal characters",
    ),
    (
        _POINTER_PATH,
        _GOOD_SHA,
        "8" * 31 + ".dir",
        "DVC object_id must be 32 lowercase hexadecimal characters",
    ),
]


@pytest.mark.parametrize("build", [_pointer_identity, _commit_evidence])
@pytest.mark.parametrize(("pointer_path", "sha", "object_id", "message"), _INVALID_BINDINGS)
def test_invalid_binding_still_raises_at_construction_for_both_classes(
    build: object, pointer_path: str, sha: str, object_id: str, message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        build(pointer_path, sha, object_id)  # type: ignore[operator]


@pytest.mark.parametrize("build", [_pointer_identity, _commit_evidence])
def test_valid_binding_round_trips_through_the_property_for_both_classes(build: object) -> None:
    value = build(_POINTER_PATH, _GOOD_SHA, _OBJECT_ID)  # type: ignore[operator]
    assert value.binding == D1DvcBinding(  # type: ignore[attr-defined]
        state=D1DvcBindingState.BOUND,
        pointer_path=_POINTER_PATH,
        pointer_sha256=_GOOD_SHA,
        object_id=_OBJECT_ID,
    )


def test_commit_evidence_still_refuses_a_non_lowercase_commit_before_the_binding() -> None:
    with pytest.raises(ValueError, match=r"^pointer commit must be lowercase hexadecimal$"):
        D1GitPointerCommitEvidence("B" * 40, "results/x", "A" * 64, "bad")
