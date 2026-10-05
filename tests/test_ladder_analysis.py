"""Shared analysis adapters cannot promote smoke, alias seeds or invent gates."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.kernel.artifacts import snapshot_d1_artifact
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1CosineGap
from entropic_substrate.ladder.analyze.ladder_analysis import (
    D1RunReader,
    read_ladder_run,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1TelemetryGap

from .d1_telemetry_fixtures import valid_events, write_events

pytestmark = pytest.mark.unit


def test_d1_adapter_uses_authoritative_reader_and_actual_persisted_class(tmp_path: Path) -> None:
    path = tmp_path / "d1.jsonl"
    write_events(path, valid_events())
    _, _, digest = snapshot_d1_artifact(tmp_path, path.name, label="test")
    artifact = ArtifactIdentity(path.name, ContentHash("sha256", digest))
    observed = D1RunReader().read_run(tmp_path, artifact)
    assert observed.run_class == "confirmatory"
    assert observed.evals[-1].source_bytes > 0
    assert "nats_per_byte" in dict(observed.evals[-1].metrics)
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="digest"):
        D1RunReader().read_run(tmp_path, artifact)


def test_reader_entrypoint_rejects_smoke_and_pilot_for_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "d1.jsonl"
    write_events(path, valid_events())
    _, _, digest = snapshot_d1_artifact(tmp_path, path.name, label="test")
    artifact = ArtifactIdentity(path.name, ContentHash("sha256", digest))
    actual = D1RunReader().read_run(tmp_path, artifact)

    class WrongClassReader:
        def read_run(self, root: Path, source: ArtifactIdentity):
            return replace(actual, run_class="pilot")

    with pytest.raises(ValueError, match="class"):
        read_ladder_run(tmp_path, artifact, WrongClassReader(), expected_class="confirmatory")


def test_d1_adapter_fails_loud_on_cosine_gap_metric(tmp_path: Path) -> None:
    """FOR-1579 (ruling 11): a gap-valued scalar metric is never silently dropped."""
    events = valid_events()
    for index, event in enumerate(events):
        if event.event_kind is not JournalEventKind.EVAL:
            continue
        payload = cast(dict[str, object], event.manifest["payload"])
        quality = cast(dict[str, object], payload["quality"])
        positions = cast(int, quality["online_positions"])
        width = len(cast(list[float], quality["state_mean"]))
        quality.update(
            {
                "state_mean": [0.0] * width,
                "state_variance": [0.0] * width,
                "state_covariance_eigenvalues": [0.0] * width,
                "state_rms_l2_norm": 0.0,
                "numerical_rank": 0,
                "effective_rank": 0.0,
                "mean_pairwise_cosine_distance": D1CosineGap(positions, positions).manifest,
            }
        )
        events[index] = replace(event, payload=payload)
    path = tmp_path / "collapsed.jsonl"
    write_events(path, events)
    _, _, digest = snapshot_d1_artifact(tmp_path, path.name, label="test")
    artifact = ArtifactIdentity(path.name, ContentHash("sha256", digest))
    with pytest.raises(D1TelemetryGap, match="mean_pairwise_cosine_distance"):
        D1RunReader().read_run(tmp_path, artifact)
