"""Public commands use the same sealed-model diagnostic path as production."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_sentence_rollout import scripted_tokens
from tests.test_span_probe import config
from tests.test_span_state_collection import model

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.trace_capture import (
    capture_model_trace,
    read_model_trace,
)

pytestmark = pytest.mark.unit


def test_public_span_diagnostic_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    template = model(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    trained = run_probe_training(
        template.profile,
        template.source_root,
        root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
        latent_usage=POSTERIOR_MEAN,
    )
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-span-diagnostics-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "training": artifact_reference(root, trained.receipt),
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "config": asdict(config()),
    }
    write_json(tmp_path, "diagnose.json", command, max_bytes=1_000_000)
    assert main(["h-diagnose-spans", "--root", str(tmp_path), "--manifest", "diagnose.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "diagnostics_complete"
    inspection = {k: v for k, v in command.items() if k != "config"}
    inspection.update(schema="h-span-inspect-command-v1", receipt=result["receipt"])
    write_json(tmp_path, "inspect.json", inspection, max_bytes=1_000_000)
    assert main(["h-inspect-spans", "--root", str(tmp_path), "--manifest", "inspect.json"]) == 0
    assert json.loads(capsys.readouterr().out)["scientific_authority"] == "none"
    scripted_tokens(monkeypatch, [65, 46, 257])
    generation = run_probe_generation(
        root,
        trained.receipt,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    destination = tmp_path / "capture"
    destination.mkdir()
    limits = {
        "max_bytes": 10_000_000,
        "max_source_bytes": 1_000_000,
        "max_total_bytes": 50_000_000,
        "max_artifacts": 100,
    }
    capture_command = {
        "schema": "h-model-capture-command-v2",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "destination": "capture",
        "training": artifact_reference(root, trained.receipt),
        "generation": artifact_reference(root, generation.receipt),
        "forensic_roots": [],
        "diagnostic_roots": [result["receipt"]],
        **limits,
    }
    write_json(tmp_path, "capture-command.json", capture_command, max_bytes=1_000_000)
    assert (
        main(["h-capture-model", "--root", str(tmp_path), "--manifest", "capture-command.json"])
        == 0
    )
    captured = artifact_from_mapping(json.loads(capsys.readouterr().out)["capture"])
    relocated = read_model_trace(destination, captured, **limits)
    assert len(relocated["diagnostic_roots"]) == 1
    assert relocated["schema"] == "h-model-trace-capture-v2"
    forged = read_json(root, artifact_from_mapping(result["receipt"]), max_bytes=10_000_000)
    forged["collection"]["train"]["encoder_source_bytes"] += 1
    invalid_receipt = write_json(root, "forged-span.json", forged, max_bytes=10_000_000)
    refused = tmp_path / "refused-capture"
    refused.mkdir()
    with pytest.raises(ValueError, match="exposure"):
        capture_model_trace(
            root,
            trained.receipt,
            generation.receipt,
            template.profile,
            template.source_root,
            refused,
            diagnostic_roots=(invalid_receipt,),
            **limits,
        )
    assert not list(refused.iterdir())
    del command["config"]["learning_rate"]
    write_json(tmp_path, "missing.json", command, max_bytes=1_000_000)
    assert main(["h-diagnose-spans", "--root", str(tmp_path), "--manifest", "missing.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    for index, (field, value) in enumerate((("learning_rate", True), ("lags", [64, 2]))):
        invalid = command | {"config": asdict(config()) | {field: value}}
        name = f"invalid-{index}.json"
        write_json(tmp_path, name, invalid, max_bytes=1_000_000)
        assert main(["h-diagnose-spans", "--root", str(tmp_path), "--manifest", name]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
