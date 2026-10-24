"""Portable capture contains the referenced model trace, not caller summaries."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from tests.test_sentence_generation import trained
from tests.test_sentence_rollout import scripted_tokens

import entropic_substrate.ladder.run.trace_capture as trace_capture_module
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.trace_capture import (
    capture_model_trace,
    read_model_trace,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", [1, 2])
def test_portable_capture_rehydrates_a_complete_model_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 257])
    generation = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    destination = tmp_path / "capture"
    destination.mkdir()
    artifact = capture_model_trace(
        output,
        training,
        generation.receipt,
        template.profile,
        template.source_root,
        destination,
        max_bytes=10_000_000,
        max_source_bytes=1_000_000,
        max_total_bytes=50_000_000,
        max_artifacts=100,
    )
    if version == 1:
        # Test-only v1 wire fixture: no diagnostic roots existed in that format.
        path = Path(artifact.path)
        payload = json.loads(path.read_text())
        payload["schema"] = "h-model-trace-capture-v1"
        del payload["diagnostic_roots"]
        data = json.dumps(payload, sort_keys=True).encode()
        path.write_bytes(data)
        artifact = ArtifactIdentity(str(path), ContentHash("sha256", sha256(data).hexdigest()))
    captured = read_model_trace(
        destination,
        artifact,
        max_bytes=10_000_000,
        max_source_bytes=1_000_000,
        max_total_bytes=50_000_000,
        max_artifacts=100,
    )
    assert captured["kind"] == "model_trace"
    assert captured["scientific_authority"] == "none"
    assert captured["run_identity"] == template.profile.identity
    assert captured["file_count"] == len(captured["artifacts"])
    assert captured["total_bytes"] > 0
    assert (destination / "source" / "prepared-corpus.json").is_file()
    (destination / "unlisted.txt").write_text("not a captured artifact")
    with pytest.raises(ValueError, match="unlisted"):
        read_model_trace(
            destination,
            artifact,
            max_bytes=10_000_000,
            max_source_bytes=1_000_000,
            max_total_bytes=50_000_000,
            max_artifacts=100,
        )


def test_capture_refuses_budgets_and_occupied_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 257])
    generation = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    for name, limit in (("too-small", 1), ("too-few", 50_000_000)):
        destination = tmp_path / name
        destination.mkdir()
        with pytest.raises(ValueError, match="budget"):
            capture_model_trace(
                output,
                training,
                generation.receipt,
                template.profile,
                template.source_root,
                destination,
                max_bytes=10_000_000,
                max_source_bytes=1_000_000,
                max_total_bytes=limit,
                max_artifacts=1 if name == "too-few" else 100,
            )
        assert not list(destination.iterdir())
    with pytest.raises(ValueError, match="empty"):
        capture_model_trace(
            output,
            training,
            generation.receipt,
            template.profile,
            template.source_root,
            output,
            max_bytes=10_000_000,
            max_source_bytes=1_000_000,
            max_total_bytes=50_000_000,
            max_artifacts=100,
        )


def test_capture_reader_detects_a_missing_referenced_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 257])
    generation = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    destination = tmp_path / "capture"
    destination.mkdir()
    artifact = capture_model_trace(
        output,
        training,
        generation.receipt,
        template.profile,
        template.source_root,
        destination,
        max_bytes=10_000_000,
        max_source_bytes=1_000_000,
        max_total_bytes=50_000_000,
        max_artifacts=100,
    )
    next(destination.glob("*-prediction-*.pt")).unlink()
    with pytest.raises(ValueError):
        read_model_trace(
            destination,
            artifact,
            max_bytes=10_000_000,
            max_source_bytes=1_000_000,
            max_total_bytes=50_000_000,
            max_artifacts=100,
        )


def test_capture_includes_the_paused_generation_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run.journal import read_json

    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 46, 257])
    paused = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
        pause_after=1,
    )
    completed = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
        resume=paused.receipt,
    )
    destination = tmp_path / "capture"
    destination.mkdir()
    capture = capture_model_trace(
        output,
        training,
        completed.receipt,
        template.profile,
        template.source_root,
        destination,
        max_bytes=10_000_000,
        max_source_bytes=1_000_000,
        max_total_bytes=50_000_000,
        max_artifacts=100,
    )
    manifest = read_json(destination, capture, max_bytes=10_000_000)
    assert paused.receipt.path in {item["path"] for item in manifest["artifacts"]}


def test_capture_schema_vocabulary_moves_the_profile_and_keeps_every_receipt_name() -> None:
    """FOR-1674: the profile literal is v3 only; v1, v2 and v3 receipts stay named."""
    schemas = trace_capture_module._SCHEMAS  # pyright: ignore[reportPrivateUsage]
    assert "h-run-profile-v3" in schemas and "h-run-profile-v2" not in schemas
    for name in ("h-generation-receipt-v1", "h-generation-receipt-v2", "h-generation-receipt-v3"):
        assert name in schemas


@pytest.mark.parametrize("old", ["h-generation-receipt-v1", "h-generation-receipt-v2"])
def test_an_old_receipt_bound_to_the_run_is_refused_by_name_never_admitted_unread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old: str
) -> None:
    """Ruling D3: the re-read loop used to skip every schema but the current one."""
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 257])
    generation = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    receipt = read_json(output, generation.receipt, max_bytes=10_000_000)
    receipt["schema"] = old
    del receipt["continuations"], receipt["sampling"]
    stale = write_json(output, "stale-receipt.json", receipt, max_bytes=10_000_000)
    destination = tmp_path / "capture"
    destination.mkdir()
    limits = {"max_bytes": 10_000_000, "max_source_bytes": 1_000_000, "max_artifacts": 100}
    with pytest.raises(ValueError, match="requires its original reader"):
        artifact = capture_model_trace(
            output,
            training,
            generation.receipt,
            template.profile,
            template.source_root,
            destination,
            max_total_bytes=50_000_000,
            forensic_roots=(stale,),
            **limits,
        )
        read_model_trace(destination, artifact, max_total_bytes=50_000_000, **limits)
