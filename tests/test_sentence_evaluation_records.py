"""Fresh evaluation admission reconstructs statistics and held-out source identities."""

import copy
from pathlib import Path

import pytest
from tests.test_sentence_execution import engine

from entropic_substrate.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.sentence_prediction.evaluation_records import read_evaluation
from entropic_substrate.sentence_prediction.journal import read_json, write_json
from entropic_substrate.sentence_prediction.runner import run_probe_training

pytestmark = pytest.mark.unit


def evaluation(root: Path):
    run = engine(root)
    output = root / "run"
    output.mkdir()
    result = run_probe_training(
        run.profile,
        run.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=10_000_000,
    )
    progress = read_json(output, result.receipt, max_bytes=10_000_000)
    artifact = artifact_from_mapping(progress["evaluations"][0]["artifact"])
    return run, output, artifact


def test_read_evaluation_reconstructs_the_recorded_sample_statistics(tmp_path: Path) -> None:
    run, output, artifact = evaluation(tmp_path)
    observed = read_evaluation(
        output,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=10_000_000,
        max_source_bytes=1_000_000,
    )
    assert observed["step"] == 2
    assert len(observed["examples"]) == run.profile.schedule.evaluation_batches


@pytest.mark.parametrize(
    "mutation",
    ["source", "weights", "summary", "quality_kind", "bytes", "missing_samples", "bool", "timing"],
)
def test_rehashed_invalid_evaluation_cannot_be_admitted(tmp_path: Path, mutation: str) -> None:
    run, output, artifact = evaluation(tmp_path)
    payload = copy.deepcopy(read_json(output, artifact, max_bytes=10_000_000))
    batch = payload["examples"][0]
    if mutation == "source":
        batch["source_examples"][0]["target"][0] += 1
    elif mutation == "weights":
        batch["log_weights"][0][0] += 1
    elif mutation == "summary":
        batch["by_sample_count"]["2"][0]["log_estimate"] += 1
    elif mutation == "quality_kind":
        payload["meaning"] = "oracle_reconstruction_training"
    elif mutation == "bytes":
        batch["source_bytes"][0] += 1
    elif mutation == "missing_samples":
        batch["proposal_log_prob"].pop()
    elif mutation == "bool":
        batch["log_weights"][0][0] = True
    else:
        payload["wall_seconds"] = -1
    altered = write_json(output, "altered-evaluation.json", payload, max_bytes=10_000_000)
    with pytest.raises(ValueError):
        read_evaluation(
            output,
            altered,
            run.profile,
            run.source_root,
            max_bytes=10_000_000,
            max_source_bytes=1_000_000,
        )
