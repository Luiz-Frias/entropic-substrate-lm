"""D1 capture artifacts are content-addressed before a DVC adapter may bind them."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
    D1ArtifactRole,
    D1CaptureArtifactError,
    D1DvcBinding,
    D1DvcBindingState,
    identify_d1_artifact,
    verify_d1_artifact,
)

pytestmark = pytest.mark.unit


def test_identity_records_normalized_path_hashes_and_pending_dvc_handoff(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    target = root / "journals" / "d1.jsonl"
    target.parent.mkdir(parents=True)
    target.write_bytes(b'{"event":"run_end"}\n')

    identity = identify_d1_artifact(
        root,
        target,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path="results/d1-captured.dvc",
    )

    assert identity.normalized_path == "journals/d1.jsonl"
    assert identity.size_bytes == len(target.read_bytes())
    assert identity.md5 == "914111ee5d4dae808f55b630c974b97c"  # pragma: allowlist secret
    assert (
        identity.sha256
        == (
            "578c897930863b902b35b20401e6ddf552342694bcd147c30f7dea49318116e9"  # pragma: allowlist secret
        )
    )
    assert identity.dvc.state is D1DvcBindingState.PENDING
    assert identity.dvc.pointer_sha256 is None
    assert identity.dvc.object_id is None
    assert verify_d1_artifact(root, identity) == identity


def test_dvc_binding_is_a_typed_later_adapter_handoff() -> None:
    pending = D1DvcBinding.pending("results/d1-captured.dvc")
    bound = pending.bind(pointer_sha256="1" * 64, object_id="2" * 32 + ".dir")

    assert bound.state is D1DvcBindingState.BOUND
    assert bound.pointer_path == "results/d1-captured.dvc"
    with pytest.raises(ValueError, match="pending"):
        replace(pending, pointer_sha256="1" * 64)
    with pytest.raises(ValueError, match="already bound"):
        bound.bind(pointer_sha256="3" * 64, object_id="4" * 32)


def test_artifact_drift_missing_file_and_skipped_existing_shortcut_fail_loudly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    target = root / "billing.json"
    target.write_text("{}\n", encoding="utf-8")
    identity = identify_d1_artifact(
        root,
        target,
        role=D1ArtifactRole.BILLING_SIDECAR,
        dvc_pointer_path="results/d1-captured.dvc",
    )

    target.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(D1CaptureArtifactError, match="identity changed"):
        verify_d1_artifact(root, identity)
    target.unlink()
    with pytest.raises(D1CaptureArtifactError, match="does not exist"):
        verify_d1_artifact(root, identity)
    with pytest.raises(D1CaptureArtifactError, match="does not exist"):
        identify_d1_artifact(
            root,
            target,
            role=D1ArtifactRole.BILLING_SIDECAR,
            dvc_pointer_path="results/d1-captured.dvc",
        )


def test_paths_outside_root_symlinks_empty_files_and_non_normalized_pointer_are_refused(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("x", encoding="utf-8")
    link = root / "link.jsonl"
    link.symlink_to(outside)
    empty = root / "empty.jsonl"
    empty.touch()

    for target, message in ((outside, "outside"), (link, "symlink"), (empty, "empty")):
        with pytest.raises(D1CaptureArtifactError, match=message):
            identify_d1_artifact(
                root,
                target,
                role=D1ArtifactRole.JOURNAL,
                dvc_pointer_path="results/d1-captured.dvc",
            )
    with pytest.raises(ValueError, match="normalized"):
        D1DvcBinding.pending("results/../captured.dvc")
    assert not os.path.isabs("results/d1-captured.dvc")
