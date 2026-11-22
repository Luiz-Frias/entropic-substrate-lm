"""Public score-gain command uses verified profile, model and source identities."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.historical.d_ladder.d_ladder_execution import SmokeRequest, run_local_smoke
from entropic_substrate.historical.d_ladder.ladder_profile import rung_profile_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.config import configs_dir
from entropic_substrate.ladder.register.d1_config import load_d1_profile
from entropic_substrate.ladder.register.ladder import Rung, RungSpec, Treatment

pytestmark = pytest.mark.unit


def test_public_score_command_preserves_diagnostic_identity(tmp_path: Path, capsys) -> None:
    def artifact(name, content):
        (tmp_path / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    payload = yaml.safe_load((configs_dir() / "d1-both.yaml").read_text())
    payload["encoder"].update(latent_width=8, n_layers=1, n_heads=2, block_size=128)
    payload["future_law"]["predictor_hidden_width"] = 8
    payload["data"].update(seq_len=8, batch_size=2)
    base = artifact("base.yaml", yaml.safe_dump(payload).encode())
    spec = RungSpec(load_d1_profile(tmp_path / "base.yaml").draft, Rung.D2, Treatment.S, 7)
    profile = artifact(
        "profile.json",
        json.dumps(
            rung_profile_mapping(
                spec,
                base_profile=ArtifactIdentity(base["path"], ContentHash("sha256", base["sha256"])),
                base_scale="full",
                predecessor=None,
            )
        ).encode(),
    )
    run = run_local_smoke(SmokeRequest(spec, max_steps=1), tmp_path / "run")
    checkpoint = Path(run["checkpoint"])
    sources = {}
    for split, offset in (("train", 0), ("heldout", 1024)):
        source = artifact(split + ".bin", bytes(range(256)))
        sources[split] = {
            "artifact": source,
            "split": split,
            "split_id": split,
            "source_id": "fixture",
            "source_offset": offset,
        }
    config = ProbeConfig(train_positions=8, eval_positions=4, updates=1, batch_size=4)
    command = {
        "schema": "d-score-gain-command-v1",
        "profile": profile,
        "checkpoint": {
            "path": "run/" + checkpoint.name,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "training": sources["train"],
        "heldout": sources["heldout"],
        "probe_config": asdict(config),
        "output": "probe",
    }
    artifact("command.json", json.dumps(command).encode())
    assert main(["d-score-gain", "--root", str(tmp_path), "--manifest", "command.json"]) == 0
    outcome = json.loads(capsys.readouterr().out)
    receipt = json.loads(Path(outcome["receipt"]["path"]).read_text())
    assert receipt["run_class"] == "smoke" and receipt["diagnostic_only"]
    assert receipt["checkpoint_sha256"] == command["checkpoint"]["sha256"]
