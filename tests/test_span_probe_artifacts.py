"""Captured probe bytes are re-admitted by bindings and numerical replay."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import torch
from tests.test_span_probe import Samples, config, probe_samples

from entropic_substrate.d1_artifact_snapshot import D1ArtifactSnapshotError, publish_d1_atomic_bytes
from entropic_substrate.sentence_prediction.span_probe_artifacts import (
    capture_span_probes,
    read_span_probes,
)

pytestmark = pytest.mark.unit
LIMIT = 1_000_000


@pytest.fixture(name="samples")
def span_samples(tmp_path: Path) -> Samples:
    return probe_samples(tmp_path)


def test_probe_capture_replays_both_controls_without_mutating_rng(
    tmp_path: Path, samples: Samples
) -> None:
    _, train, heldout = samples
    artifact = capture_span_probes(tmp_path, "probe.pt", train, heldout, config(), max_bytes=LIMIT)
    rng = torch.get_rng_state().clone()
    bundle = read_span_probes(
        tmp_path,
        artifact,
        expected_checkpoint=train.checkpoint_id,
        expected_codec=train.codec_id,
        expected_source=train.source_digest,
        max_bytes=LIMIT,
    )
    assert [p.manifest.control for p in bundle.probes] == ["observed", "shuffled"]
    assert all(report.summary.count == 8 for report in bundle.reports)
    assert bundle.heldout.positions == heldout.positions
    torch.testing.assert_close(bundle.train.before, train.before, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    with pytest.raises(D1ArtifactSnapshotError, match="occupied"):
        capture_span_probes(tmp_path, "probe.pt", train, heldout, config(), max_bytes=LIMIT)


@pytest.mark.parametrize("field", ["expected_checkpoint", "expected_codec", "expected_source"])
def test_reader_rejects_another_lineage(tmp_path: Path, samples: Samples, field: str) -> None:
    _, train, heldout = samples
    artifact = capture_span_probes(tmp_path, "probe.pt", train, heldout, config(), max_bytes=LIMIT)
    expected = {
        "expected_checkpoint": train.checkpoint_id,
        "expected_codec": train.codec_id,
        "expected_source": train.source_digest,
    }
    expected[field] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        read_span_probes(tmp_path, artifact, **expected, max_bytes=LIMIT)


@pytest.mark.parametrize(
    "mutation", ["summary", "weight", "label", "training", "config", "permutation", "duration"]
)
def test_rehashed_forgery_is_not_admitted(tmp_path: Path, samples: Samples, mutation: str) -> None:
    _, train, heldout = samples
    artifact = capture_span_probes(tmp_path, "probe.pt", train, heldout, config(), max_bytes=LIMIT)
    value = torch.load(Path(artifact.path), weights_only=True)
    if mutation == "summary":
        value["reports"][0]["summary"]["mean_gain"] += 0.5
    elif mutation == "weight":
        value["probes"][0]["weights"]["weight"].add_(1)
    elif mutation == "label":
        value["heldout"]["labels"][0] ^= 1
    elif mutation == "training":
        value["train"]["labels"][0] ^= 1
    elif mutation == "config":
        del value["probes"][0]["manifest"]["config"]["learning_rate"]
    elif mutation == "permutation":
        value["reports"][1]["state_permutation"] = (0,) * 16
    else:
        value["reports"][0]["score_seconds"] = -1.0
    stream = BytesIO()
    torch.save(value, stream)
    forged = publish_d1_atomic_bytes(tmp_path, "forged.pt", stream.getvalue(), label="test forgery")
    with pytest.raises(ValueError):
        read_span_probes(
            tmp_path,
            forged,
            expected_checkpoint=train.checkpoint_id,
            expected_codec=train.codec_id,
            expected_source=train.source_digest,
            max_bytes=LIMIT,
        )


def test_capture_refuses_invalid_budget_and_unmatched_inputs(
    tmp_path: Path, samples: Samples
) -> None:
    _, train, heldout = samples
    with pytest.raises(ValueError, match="byte"):
        capture_span_probes(tmp_path, "probe.pt", train, heldout, config(), max_bytes=1)
    assert not (tmp_path / "probe.pt").exists()
    with pytest.raises(ValueError, match="binding"):
        capture_span_probes(
            tmp_path,
            "probe.pt",
            train,
            replace(heldout, codec_id="other"),
            config(),
            max_bytes=LIMIT,
        )
