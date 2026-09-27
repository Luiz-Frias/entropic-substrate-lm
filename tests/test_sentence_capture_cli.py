"""Public capture commands preserve a complete referenced model trace."""

import json
from pathlib import Path

import pytest
from tests.test_sentence_generation import trained
from tests.test_sentence_rollout import scripted_tokens

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference, write_json

pytestmark = pytest.mark.unit


def test_public_capture_and_relocated_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
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
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    (tmp_path / "capture").mkdir()
    command = {
        "schema": "h-model-capture-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "destination": "capture",
        "training": artifact_reference(output, training),
        "generation": artifact_reference(output, generation.receipt),
        "forensic_roots": [],
        "max_source_bytes": 1_000_000,
        "max_bytes": 10_000_000,
        "max_total_bytes": 50_000_000,
        "max_artifacts": 100,
    }
    write_json(tmp_path, "capture-command.json", command, max_bytes=1_000_000)
    assert (
        main(["h-capture-model", "--root", str(tmp_path), "--manifest", "capture-command.json"])
        == 0
    )
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "model_trace_captured"
    inspection = {
        "schema": "h-model-inspect-command-v1",
        "capture_root": "capture",
        "capture": outcome["capture"],
        "max_source_bytes": 1_000_000,
        "max_bytes": 10_000_000,
        "max_total_bytes": 50_000_000,
        "max_artifacts": 100,
    }
    write_json(tmp_path, "inspect-capture.json", inspection, max_bytes=1_000_000)
    assert (
        main(["h-inspect-model", "--root", str(tmp_path), "--manifest", "inspect-capture.json"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["scientific_authority"] == "none"


@pytest.mark.parametrize("root,limit", [("../outside", 100), ("capture", 0)])
def test_capture_command_refuses_unsafe_root_or_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], root: str, limit: int
) -> None:
    from entropic_substrate.core.sentence.records import content_identity

    (tmp_path / "capture").mkdir()
    command = {
        "schema": "h-model-inspect-command-v1",
        "capture_root": root,
        "capture": {"path": "capture.json", "sha256": content_identity(("missing fixture",))},
        "max_source_bytes": 1_000_000,
        "max_bytes": 1_000_000,
        "max_total_bytes": 50_000_000,
        "max_artifacts": limit,
    }
    write_json(tmp_path, "invalid.json", command, max_bytes=1_000_000)
    assert main(["h-inspect-model", "--root", str(tmp_path), "--manifest", "invalid.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "capture_refused"
