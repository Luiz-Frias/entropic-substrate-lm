"""Byte custody is portable and never presented as native model admission."""

import json
from pathlib import Path

import pytest
from tests.test_sentence_generation import trained
from tests.test_sentence_rollout import scripted_tokens

from entropic_substrate.cli import main
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.sentence_prediction import trace_capture
from entropic_substrate.sentence_prediction.generation import run_probe_generation
from entropic_substrate.sentence_prediction.journal import artifact_reference, write_json

pytestmark = pytest.mark.unit
LIMITS = {
    "max_bytes": 10_000_000,
    "max_source_bytes": 1_000_000,
    "max_total_bytes": 50_000_000,
    "max_artifacts": 100,
}


def build_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ArtifactIdentity]:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 257])
    generated = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    root = tmp_path / "capture"
    root.mkdir()
    artifact = trace_capture.capture_model_trace(
        output, training, generated.receipt, template.profile, template.source_root, root, **LIMITS
    )
    return root, artifact


@pytest.fixture
def captured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ArtifactIdentity]:
    return build_capture(tmp_path, monkeypatch)


def test_custody_does_not_call_native_admission_or_weaken_it(
    captured: tuple[Path, ArtifactIdentity], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, artifact = captured

    def native(*args: object, **kwargs: object) -> None:
        raise RuntimeError("native admission required")

    monkeypatch.setattr(trace_capture, "read_training_progress", native)
    result = trace_capture.inspect_model_custody(root, artifact, **LIMITS)
    assert result["status"] == "byte_custody_verified"
    assert result["native_validation"] == "not_performed"
    assert result["scientific_authority"] == "none"
    with pytest.raises(RuntimeError, match="native admission required"):
        trace_capture.read_model_trace(root, artifact, **LIMITS)
    tensor_file = next(root.glob("*.pt"))
    with tensor_file.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(ValueError, match="hash"):
        trace_capture.inspect_model_custody(root, artifact, **LIMITS)


def test_public_custody_command_names_its_limited_scope(
    tmp_path: Path, captured: tuple[Path, ArtifactIdentity], capsys: pytest.CaptureFixture[str]
) -> None:
    root, artifact = captured
    command = {
        "schema": "h-custody-inspect-command-v1",
        "capture_root": "capture",
        "capture": artifact_reference(root, artifact),
        **LIMITS,
    }
    write_json(tmp_path, "custody.json", command, max_bytes=1_000_000)
    assert main(["h-inspect-custody", "--root", str(tmp_path), "--manifest", "custody.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["native_validation"] == "not_performed"
    assert result["total_bytes"] > 0 and result["file_count"] > 0
