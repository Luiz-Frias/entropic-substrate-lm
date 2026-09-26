"""The shared diagnostic command preserves D1's native request admission."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

import entropic_substrate.historical.d_ladder.d1_probe as native_probe
import entropic_substrate.ladder.register.d1_config as config_module
from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    serialize_d1_trainer_request,
)

from .test_d1_trainer_request_codec import _fresh

pytestmark = pytest.mark.unit


def test_native_d1_command_uses_its_bound_request(tmp_path: Path, monkeypatch, capsys) -> None:
    request, catalog = _fresh()
    monkeypatch.setattr(config_module, "load_d1_execution_catalog", lambda: catalog)

    def artifact(name, payload):
        (tmp_path / name).write_bytes(payload)
        return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}

    request_ref = artifact("request.json", serialize_d1_trainer_request(request))
    checkpoint = artifact("model.pt", b"transport-only-fixture")
    sources = {}
    for split, offset in (("train", 0), ("heldout", 1000)):
        sources[split] = {
            "artifact": artifact(split + ".bin", bytes(range(128))),
            "split": split,
            "split_id": split,
            "source_id": "fixture",
            "source_offset": offset,
        }
    config = ProbeConfig(train_positions=4, eval_positions=2, updates=1, batch_size=2)
    command = {
        "schema": "d1-score-gain-command-v1",
        "request": request_ref,
        "checkpoint": checkpoint,
        "training": sources["train"],
        "heldout": sources["heldout"],
        "probe_config": asdict(config),
        "output": "probe",
    }
    observed = []

    def probe(plan, path, digest, training, heldout, output, envelope):
        observed.append((plan, path, digest, training.split, heldout.split, output, envelope))
        result = ArtifactIdentity("diagnostic.json", ContentHash("sha256", "a" * 64))
        return SimpleNamespace(receipt=result, weights=result, report=result)

    monkeypatch.setattr(native_probe, "run_d1_checkpoint_probe", probe)
    artifact("command.json", json.dumps(command).encode())
    assert main(["d-score-gain", "--root", str(tmp_path), "--manifest", "command.json"]) == 0
    assert observed == [
        (
            request.plan,
            tmp_path / "model.pt",
            checkpoint["sha256"],
            "train",
            "heldout",
            tmp_path / "probe",
            config,
        )
    ]
    assert "receipt" in json.loads(capsys.readouterr().out)
    (tmp_path / "request.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        main(["d-score-gain", "--root", str(tmp_path), "--manifest", "command.json"])
    assert len(observed) == 1
