"""The public command executes and inspects the same file-bound training path."""

import json
from pathlib import Path

import pytest
from tests.test_sentence_execution import engine

from entropic_substrate.cli import main
from entropic_substrate.sentence_prediction.journal import artifact_reference, write_json

pytestmark = pytest.mark.unit


def test_public_training_and_inspection_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": None,
    }
    write_json(tmp_path, "train.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "train.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "training_complete" and result["scientific_authority"] == "none"
    inspect = {key: value for key, value in command.items() if key not in ("resume", "pause_after")}
    inspect.update(schema="h-probe-inspect-command-v1", receipt=result["receipt"])
    write_json(tmp_path, "inspect.json", inspect, max_bytes=1_000_000)
    assert main(["h-inspect-training", "--root", str(tmp_path), "--manifest", "inspect.json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["successful_optimizer_steps"] == 6


def test_command_rejects_unknown_fields_and_source_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    for index, source in enumerate(("../outside", "source")):
        command = {
            "schema": "h-probe-train-command-v1",
            "profile": artifact_reference(tmp_path, profile),
            "source_root": source,
            "output_root": "run",
            "max_source_bytes": 1_000_000,
            "max_artifact_bytes": 10_000_000,
            "resume": None,
            "pause_after": None,
        }
        if index:
            command["steps_override"] = 1
        name = f"bad-{index}.json"
        write_json(tmp_path, name, command, max_bytes=1_000_000)
        assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", name]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert not list((tmp_path / "run").iterdir())


def test_public_pause_resume_preserves_the_training_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": {"phase": "prediction", "step": 2},
    }
    write_json(tmp_path, "pause.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "pause.json"]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "paused"
    command.update(resume=paused["receipt"], pause_after=None)
    write_json(tmp_path, "resume.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "resume.json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "training_complete"


def test_pause_at_an_unsealed_step_is_not_silently_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": {"phase": "prediction", "step": 1},
    }
    write_json(tmp_path, "bad-pause.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "bad-pause.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert not list((tmp_path / "run").iterdir())
