"""Hash-bound generic remote requests and existing immutable-object transport."""

import json
from pathlib import Path

import pytest
import yaml

from entropic_substrate.historical.d_ladder.d_ladder_remote import (
    materialize_remote_output,
    materialize_remote_request,
    publish_remote_request,
    run_remote_worker,
)
from entropic_substrate.historical.d_ladder.d_ladder_remote_request import RemoteLadderRequest
from entropic_substrate.historical.d_ladder.ladder_profile import rung_profile_mapping
from entropic_substrate.kernel.evidence_class.resource_provenance import execution_segment_id
from entropic_substrate.kernel.lifecycle import ActorIdentity, ActorRole
from entropic_substrate.ladder.register.d1_config import d1_configs_dir, load_d1_profile
from entropic_substrate.ladder.register.ladder import Rung, RungSpec, Treatment
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    UnavailableResourceActivationFactory,
)

from . import ladder_campaign_fixtures
from .test_d1_trainer_object_transport import _ObjectStore
from .test_d_ladder_remote_request import request_fixture
from .test_ladder_profile import write_artifact
from .test_resource_provenance import _start

pytestmark = pytest.mark.unit


def test_object_roundtrip_reuses_exact_snapshot_and_request_is_published_last(
    tmp_path: Path,
) -> None:
    source, destination = tmp_path / "source", tmp_path / "worker"
    source.mkdir()
    request = request_fixture(source)
    objects = _ObjectStore()
    identity = publish_remote_request(objects, request, source)
    assert objects.publishes[-1] == identity.key
    observed = materialize_remote_request(objects, identity, destination)
    assert observed == request
    assert (destination / "enwik8").read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        materialize_remote_request(objects, identity, destination)


def test_input_object_tampering_prevents_materialization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    request = request_fixture(source)
    objects = _ObjectStore()
    identity = publish_remote_request(objects, request, source)
    objects.values[f"{request.prefix}/inputs/enwik8"] = b"changed"
    with pytest.raises(ValueError, match="identity"):
        materialize_remote_request(objects, identity, tmp_path / "worker")


def test_worker_refuses_runtime_drift_before_dispatch_and_emits_no_ready_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    request = request_fixture(source)
    objects = _ObjectStore()
    identity = publish_remote_request(objects, request, source)
    with pytest.raises(ValueError, match="runtime"):
        run_remote_worker(
            objects,
            identity,
            tmp_path / "worker",
            observed_start=request.start,
            observed_code_commit="d" * 40,
            factory=UnavailableResourceActivationFactory(),
            predecessor_loader=lambda _: pytest.fail("must not load gate after runtime drift"),
        )
    assert f"{request.prefix}/ready.json" not in objects.values
    assert f"{request.prefix}/failure.json" in objects.values


def test_worker_dispatches_real_scientific_tiny_with_actual_filing_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, worker = tmp_path / "source", tmp_path / "worker"
    source.mkdir()
    payload = yaml.safe_load((d1_configs_dir() / "d1-both.yaml").read_bytes())
    payload["encoder"].update(latent_width=8, n_layers=1, n_heads=2, block_size=128)
    payload["future_law"].update(predictor_hidden_width=8)
    payload["data"].update(batch_size=2, seq_len=8, num_eval_batches=1)
    payload["train"].update(
        max_steps=2,
        warmup_steps=0,
        eval_interval=1,
        log_interval=1,
        save_interval=1,
        amp_dtype="off",
        device="cpu",
    )
    base = write_artifact(source, "base.yaml", yaml.safe_dump(payload).encode())
    spec = RungSpec(load_d1_profile(source / "base.yaml").draft, Rung.D2, Treatment.S, 7)
    corpus = write_artifact(source, "enwik8", bytes(range(256)) * 4)
    captured = []
    real_gate = ladder_campaign_fixtures.d1_gate

    def capture_gate(root: Path, patch: pytest.MonkeyPatch):
        gate = real_gate(root, patch)
        captured.append(gate)
        return gate

    monkeypatch.setattr(ladder_campaign_fixtures, "d1_gate", capture_gate)
    auth = ladder_campaign_fixtures.authorization(
        source,
        spec,
        corpus.content_hash.digest,
        {
            "train_start": 0,
            "train_end": 512,
            "eval_start": 512,
            "eval_end": 1024,
            "source_total_bytes": 1024,
        },
        monkeypatch,
    )
    profile = write_artifact(
        source,
        "profile.json",
        json.dumps(
            rung_profile_mapping(spec, base_profile=base, base_scale="full", predecessor=None)
        ).encode(),
    )
    from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
    from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
    from entropic_substrate.ladder.gate.ladder_campaign import authorize_ladder_run

    design = json.loads((source / "campaign.json").read_bytes())
    design["statistics"]["path"] = "statistics.json"
    campaign = write_artifact(source, "campaign.json", json.dumps(design).encode())
    operator = ActorIdentity("operator", "op-fixture", ActorRole.OPERATOR)
    auth = authorize_ladder_run(
        source, campaign, auth.logical_run_id, operator=operator, predecessor=captured[0]
    )
    declaration = write_artifact(source, "predecessor.json", b'{"test_loader":"validated-gate"}')
    inputs = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            name = path.relative_to(source).as_posix()
            _, _, digest = read_d1_artifact_bytes(source, name, label="fixture")
            inputs.append(ArtifactIdentity(name, ContentHash("sha256", digest)))
    start = _start(
        logical_run_id=auth.logical_run_id,
        execution_segment_id=execution_segment_id(auth.logical_run_id, 0),
    )
    request = RemoteLadderRequest(
        profile,
        auth.campaign_artifact,
        corpus,
        tuple(inputs),
        ActorIdentity("operator", "op-fixture", ActorRole.OPERATOR),
        auth.digest,
        start,
        "b" * 40,
        "pod123",
        "c" * 64,
        declaration,
    )
    objects = _ObjectStore()
    identity = publish_remote_request(objects, request, source)
    result = run_remote_worker(
        objects,
        identity,
        worker,
        observed_start=start,
        observed_code_commit=request.code_commit,
        factory=UnavailableResourceActivationFactory(),
        predecessor_loader=lambda _: captured[0],
    )
    receipt = json.loads(objects.values[result.key])
    assert receipt["request_sha256"] == request.digest
    assert receipt["result"]["run_class"] == "pilot"
    assert receipt["result"]["step"] == 2
    assert {"step-000002.pt", "manifest.json"} <= {item["path"] for item in receipt["artifacts"]}
    assert objects.publishes.index(f"{request.prefix}/ready.json") < objects.publishes.index(
        f"{request.prefix}/start.json"
    )
    assert objects.publishes[-1] == result.key
    checkpoint = objects.read(f"{request.prefix}/checkpoint-step-000001.json")
    assert checkpoint is not None
    recovered_checkpoint = tmp_path / "checkpoint-recovery"
    materialize_remote_output(objects, request, checkpoint.identity, recovered_checkpoint)
    assert (recovered_checkpoint / "step-000001.pt").is_file()
    assert not (recovered_checkpoint / "step-000002.pt").exists()
    recovered = tmp_path / "recovered"
    materialize_remote_output(objects, request, result, recovered)
    assert (recovered / "step-000002.pt").read_bytes() == (
        worker / "run" / "step-000002.pt"
    ).read_bytes()
    key = receipt["artifacts"][0]["key"]
    objects.values[key] = b"tampered"
    with pytest.raises(ValueError, match="identity"):
        materialize_remote_output(objects, request, result, tmp_path / "tampered")
