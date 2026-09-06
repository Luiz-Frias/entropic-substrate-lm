"""One admitted scientific cell traverses the same engine as local smokes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.ladder_campaign_fixtures import authorization

from d1_telemetry_fixtures import start_provenance
from entropic_substrate.analysis.ladder_analysis import ScientificRunReader
from entropic_substrate.analysis.ladder_probe import ProbeSource, run_checkpoint_probe
from entropic_substrate.analysis.predictive_probe import ProbeConfig
from entropic_substrate.d1_resource_activation import UnavailableResourceActivationFactory
from entropic_substrate.d_ladder_cli import tiny_rung_spec
from entropic_substrate.d_ladder_execution import run_ladder
from entropic_substrate.d_ladder_request import ScientificRequest
from entropic_substrate.d_ladder_resources import LadderResources
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.evidence.resource_provenance import execution_segment_id

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("interrupted", [False, True])
def test_registered_pilot_records_step_zero_terminal_and_cumulative_work(
    tmp_path, monkeypatch, interrupted
):
    spec = tiny_rung_spec("d2-s", seed=7)
    protocol = spec.base.execution_protocol
    protocol = replace(
        protocol,
        optimization=replace(protocol.optimization, max_steps=2, warmup_steps=0),
        evaluation=replace(
            protocol.evaluation, num_batches=2, eval_interval=1, log_interval=1, save_interval=1
        ),
    )
    spec = replace(
        spec,
        base=replace(
            spec.base,
            execution_protocol=protocol,
        ),
    )
    source = bytes(range(256)) * 8
    (tmp_path / "source.bin").write_bytes(source)
    split = {
        "train_start": 0,
        "train_end": 1024,
        "eval_start": 1024,
        "eval_end": 2048,
        "source_total_bytes": 2048,
    }
    auth = authorization(
        tmp_path / "authority", spec, hashlib.sha256(source).hexdigest(), split, monkeypatch
    )
    request = ScientificRequest(spec, auth, tmp_path, "source.bin")
    start = replace(
        start_provenance(),
        logical_run_id=auth.logical_run_id,
        execution_segment_id=execution_segment_id(auth.logical_run_id, 0),
    )
    first = run_ladder(
        request,
        tmp_path / "run",
        resources=LadderResources(start, UnavailableResourceActivationFactory()),
        until_step=1 if interrupted else None,
    )
    result = first
    if interrupted:
        checkpoint = Path(first["checkpoint"])
        resume_start = replace(
            start_provenance(1, hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
            logical_run_id=auth.logical_run_id,
            execution_segment_id=execution_segment_id(auth.logical_run_id, 1),
        )
        result = run_ladder(
            request,
            tmp_path / "run",
            resume=checkpoint,
            resources=LadderResources(resume_start, UnavailableResourceActivationFactory()),
        )
    assert result["run_class"] == "pilot"
    capture = json.loads(Path(result["capture"]).read_text())
    assert capture["completed_step"] == 2
    assert len(capture["records"]) == 3
    terminal = json.loads((tmp_path / "run" / "step-000002.json").read_text())
    assert terminal["source_bytes"] == 32
    assert terminal["resource_snapshot"]["baseline_calibration_positions"] == 576
    assert terminal["resource_snapshot"]["committed_quality_source_bytes"] == 32
    metrics = terminal["metrics"]
    assert metrics["future_nll"] == pytest.approx(
        sum(batch["future_nll"] for batch in metrics["batches"]) / 2
    )
    assert metrics["future_zero_nll"] > 0 and metrics["future_mean_nll"] > 0
    artifact = ArtifactIdentity(
        "run/capture.json",
        ContentHash("sha256", hashlib.sha256(Path(result["capture"]).read_bytes()).hexdigest()),
    )
    admitted = ScientificRunReader().read_run(tmp_path, artifact)
    assert [evaluation.step for evaluation in admitted.evals] == [0, 1, 2]
    if not interrupted:
        (tmp_path / "train.bin").write_bytes(source[:1024])
        (tmp_path / "heldout.bin").write_bytes(source[1024:])
        inputs = [
            ProbeSource(
                tmp_path,
                name + ".bin",
                ContentHash("sha256", hashlib.sha256(source[offset : offset + 1024]).hexdigest()),
                name,
                name,
                "registered-source",
                offset,
            )
            for name, offset in (("train", 0), ("heldout", 1024))
        ]
        probe = run_checkpoint_probe(
            request,
            Path(result["checkpoint"]),
            *inputs,
            tmp_path / "probe",
            ProbeConfig(lags=(2, 4), train_positions=8, eval_positions=4, updates=1, batch_size=4),
        )
        receipt = json.loads(Path(probe.receipt.path).read_text())
        assert receipt["run_class"] == "pilot" and receipt["diagnostic_only"]
    (tmp_path / "source.bin").write_bytes(bytes(reversed(source)))
    with pytest.raises(ValueError, match="digest"):
        request.load_splits()
