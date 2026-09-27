"""Captured probe bytes are re-admitted by bindings and numerical replay."""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import torch
from tests.test_span_probe import Samples, config, probe_samples

from entropic_substrate.kernel.artifacts import D1ArtifactSnapshotError, publish_d1_atomic_bytes
from entropic_substrate.ladder.run.span_probe_artifacts import (
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
    ("mutation", "error"),
    [
        ("summary", "numerical report"),
        ("weight", "identity mismatch"),
        ("label", "differs from replay"),
        ("training", "training binding"),
        ("config", "versioned schema"),
        ("permutation", "coordinate differs"),
        ("duration", "nonnegative"),
    ],
)
def test_rehashed_forgery_is_not_admitted(
    tmp_path: Path, samples: Samples, mutation: str, error: str
) -> None:
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
    with pytest.raises(ValueError, match=error):
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


def test_too_small_budget_refuses_before_fitting(
    tmp_path: Path, samples: Samples, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import span_probe_artifacts as artifacts

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fitting started despite impossible byte budget")

    monkeypatch.setattr(artifacts, "fit_span_probe", forbidden)
    _, train, heldout = samples
    with pytest.raises(ValueError, match="byte"):
        capture_span_probes(tmp_path, "too-small.pt", train, heldout, config(), max_bytes=1)


def test_serialization_limit_keeps_partial_output_unpublished(
    tmp_path: Path, samples: Samples
) -> None:
    _, train, heldout = samples
    artifact = capture_span_probes(
        tmp_path, "complete.pt", train, heldout, config(), max_bytes=LIMIT
    )
    limit = Path(artifact.path).stat().st_size - 1000
    with pytest.raises(ValueError, match="byte"):
        capture_span_probes(tmp_path, "partial.pt", train, heldout, config(), max_bytes=limit)
    assert not (tmp_path / "partial.pt").exists()


def test_output_buffer_never_accepts_bytes_past_its_limit() -> None:
    from entropic_substrate.ladder.run.span_probe_artifacts import _LimitedBuffer

    with _LimitedBuffer(4) as stream:
        assert stream.write(b"1234") == 4
        with pytest.raises(ValueError, match="byte"):
            stream.write(b"5")
        assert stream.getvalue() == b"1234"
