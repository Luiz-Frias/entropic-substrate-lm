"""The diagnostic producer must use the sealed model and captured source."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_span_probe import config
from tests.test_span_state_collection import model

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.span_diagnostics import (
    read_span_diagnostics,
    run_span_diagnostics,
)

pytestmark = pytest.mark.unit
LIMIT = 10_000_000
type Lineage = tuple[HProfile, Path, Path, ArtifactIdentity]


@pytest.fixture
def lineage(tmp_path: Path) -> Lineage:
    template = model(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    trained = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    return template.profile, template.source_root, output, trained.receipt


def test_diagnostic_follows_the_actual_checkpoint_and_retains_controls(lineage: Lineage) -> None:
    profile, source, root, training = lineage
    rng = torch.get_rng_state().clone()
    artifact = run_span_diagnostics(
        root,
        training,
        profile,
        source,
        config(),
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    report = read_span_diagnostics(
        root,
        artifact,
        training,
        profile,
        source,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    assert report["status"] == "diagnostics_complete"
    assert report["scientific_authority"] == "none"
    assert report["collection"]["train"]["encoder_source_bytes"] > 0
    assert report["collection"]["validation"]["cuda_seconds"] is None
    assert [r["control"] for r in report["summaries"]] == ["observed", "shuffled"]
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("checkpoint", "lineage binding"),
        ("training", "lineage binding"),
        ("collection", "exposure"),
        ("summaries", "numerical report"),
    ],
)
def test_rehashed_diagnostic_receipt_is_not_enough(
    lineage: Lineage, field: str, error: str
) -> None:
    profile, source, root, training = lineage
    artifact = run_span_diagnostics(
        root,
        training,
        profile,
        source,
        config(),
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    value = read_json(root, artifact, max_bytes=LIMIT)
    if field in ("checkpoint", "training"):
        value[field]["sha256"] = "0" * 64
    elif field == "collection":
        value[field]["train"]["encoder_source_bytes"] += 1
    else:
        value[field][0]["gain"]["mean_gain"] += 1
    forged = write_json(root, "forged-diagnostic.json", value, max_bytes=LIMIT)
    with pytest.raises(ValueError, match=error):
        read_span_diagnostics(
            root,
            forged,
            training,
            profile,
            source,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )


def test_insufficient_lag_population_keeps_a_failure_receipt(lineage: Lineage) -> None:
    profile, source, root, training = lineage
    with pytest.raises(ValueError, match="insufficient"):
        run_span_diagnostics(
            root,
            training,
            profile,
            source,
            replace(config(), train_positions=100000),
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )
    failures = list(root.glob("*-span-failure.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["status"] == "diagnostic_gap"
    assert failure["error_code"] == "insufficient_span_probe_population"
    assert not list(root.glob("*-span-diagnostics.json"))


def test_outer_receipt_preserves_the_registered_replay_tolerance(lineage: Lineage) -> None:
    profile, source, root, training = lineage
    artifact = run_span_diagnostics(
        root,
        training,
        profile,
        source,
        config(),
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    value = read_json(root, artifact, max_bytes=LIMIT)
    value["summaries"][0]["gain"]["mean_gain"] += 1e-6
    rounded = write_json(root, "rounded-diagnostic.json", value, max_bytes=LIMIT)
    read_span_diagnostics(
        root,
        rounded,
        training,
        profile,
        source,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )


@pytest.mark.parametrize("zero", [True, False])
def test_failure_receipts_distinguish_zero_geometry_from_defects(
    lineage: Lineage, monkeypatch: pytest.MonkeyPatch, zero: bool
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run import span_diagnostics

    error = UndefinedDirection(3) if zero else RuntimeError("injected observer defect")

    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(span_diagnostics, "collect_span_states", fail)
    profile, source, root, training = lineage
    with pytest.raises(type(error), match=r"undefined_zero_norm|injected observer defect"):
        run_span_diagnostics(
            root,
            training,
            profile,
            source,
            config(),
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )
    failure = json.loads(next(root.glob("*-span-failure.json")).read_text())
    assert failure["status"] == ("scientific_gap" if zero else "diagnostic_defect")
    assert failure["error_type"] == type(error).__name__
    assert failure["error_code"] == ("undefined_zero_norm" if zero else "RuntimeError")
    if zero:
        assert failure["affected_positions"] == 3
    assert not list(root.glob("*-span-diagnostics.json"))
