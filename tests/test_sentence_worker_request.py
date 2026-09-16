"""Worker input contracts do not grant spending or scientific authority."""

import json
from dataclasses import replace

import pytest
from tests.test_sentence_profile import profile

from entropic_substrate.analysis.predictive_probe import ProbeConfig
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.sentence_prediction.worker_request import (
    ProbeRequest,
    TransferLimits,
    parse_probe_request,
)

pytestmark = pytest.mark.unit


def request() -> ProbeRequest:
    template = profile()
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", template.source_digest)
    )
    return ProbeRequest(
        template,
        (source,),
        "1" * 64,
        TransferLimits(10_000_000, 50_000_000, 100),
        ProbeConfig(train_positions=8, eval_positions=8, updates=2, batch_size=4),
        code_commit=None,
        code_bundle=None,
        image=None,
        resume=None,
        pause=None,
    )


def test_full_request_roundtrip_and_non_authority() -> None:
    value = request()
    assert parse_probe_request(value.payload, max_bytes=1_000_000) == value
    assert value.prefix == f"h-probes/v1/{value.nonce}"
    assert value.manifest["scientific_authority"] == "none"
    assert value.manifest["spending_authority"] == "none"
    assert replace(value, nonce="2" * 64).digest != value.digest


@pytest.mark.parametrize("path", ["../escape", "/absolute", "run/unrequested.pt", "other/data"])
def test_fresh_input_paths_are_confined(path: str) -> None:
    value = request()
    with pytest.raises(ValueError):
        replace(
            value, inputs=(*value.inputs, ArtifactIdentity(path, ContentHash("sha256", "a" * 64)))
        )


def test_profile_source_binding_and_explicit_fields() -> None:
    value = request()
    with pytest.raises(ValueError, match="source"):
        replace(value, profile=replace(value.profile, source_digest="b" * 64))
    row = json.loads(value.payload)
    del row["probe"]["learning_rate"]
    with pytest.raises(ValueError):
        parse_probe_request(json.dumps(row).encode(), max_bytes=1_000_000)
    with pytest.raises(ValueError, match="byte"):
        parse_probe_request(value.payload, max_bytes=1)


def test_remote_and_scientific_scopes_require_distinct_admission() -> None:
    value = request()
    remote = replace(value.profile, run_class="remote_tiny", device="cuda", precision="bf16")
    with pytest.raises(ValueError, match="source proof"):
        replace(value, profile=remote)
    with pytest.raises(ValueError, match="scientific"):
        replace(value, profile=replace(remote, run_class="pilot", name="h0-fisher"))
    with pytest.raises(ValueError, match="successor"):
        replace(
            value,
            profile=replace(
                remote,
                stage="H1",
                name="h1-fisher-tiny",
                consumer="isotropic",
                optimization=replace(remote.optimization, future_weight=0),
            ),
        )


def test_remote_envelope_binds_expected_proof_without_verifying_execution() -> None:
    value = request()
    bundle = ArtifactIdentity("code.bundle", ContentHash("sha256", "b" * 64))
    remote = replace(
        value,
        profile=replace(value.profile, run_class="remote_tiny", device="cuda", precision="bf16"),
        inputs=(*value.inputs, bundle),
        code_bundle=bundle,
        code_commit="c" * 40,
        image="ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64,
    )
    assert parse_probe_request(remote.payload, max_bytes=1_000_000) == remote
    with pytest.raises(ValueError, match="image"):
        replace(remote, image="ghcr.io/luiz-frias/entropic-substrate-lm:latest")


def test_resume_and_pause_are_explicit_and_bounded() -> None:
    value = request()
    receipt = ArtifactIdentity("run/paused.json", ContentHash("sha256", "c" * 64))
    resumed = replace(
        value, inputs=(*value.inputs, receipt), resume=receipt, pause=("prediction", 2)
    )
    assert parse_probe_request(resumed.payload, max_bytes=1_000_000) == resumed
    with pytest.raises(ValueError, match="pause"):
        replace(value, pause=("prediction", 1))
    with pytest.raises(ValueError, match="resume"):
        replace(value, resume=receipt)
    with pytest.raises(ValueError):
        replace(value, inputs=(*value.inputs, value.inputs[0]))
