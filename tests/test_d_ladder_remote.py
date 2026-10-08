"""Hash-bound generic remote requests and existing immutable-object transport."""

import argparse
import json
from dataclasses import replace
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


def _published_request(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    request = request_fixture(source)
    objects = _ObjectStore()
    return request, objects, publish_remote_request(objects, request, source)


def _run_worker(request, objects, identity, root, **kwargs):
    return run_remote_worker(
        objects,
        identity,
        root,
        observed_start=request.start,
        observed_code_commit=request.code_commit,
        factory=UnavailableResourceActivationFactory(),
        **kwargs,
    )


def test_identity_mismatch_on_a_later_input_leaves_no_partial_workspace_and_retry_reaches_identity(
    tmp_path: Path,
) -> None:
    """FOR-1580: artifact 2 of 4 is tampered; artifact 1 is already on disk when it fails.

    Without cleanup the half-written root makes every retry die on
    ``FileExistsError`` before it can reach the identity check it needs to see.
    """
    request, objects, identity = _published_request(tmp_path)
    objects.values[f"{request.prefix}/inputs/{request.inputs[1].path}"] = b"changed"
    root = tmp_path / "worker"

    with pytest.raises(ValueError, match="remote input object identity mismatch"):
        _run_worker(request, objects, identity, root)
    assert not root.exists() or not any(root.iterdir())
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["error_type"] == "ValueError"

    with pytest.raises(ValueError, match="remote input object identity mismatch"):
        _run_worker(request, objects, identity, root)


def test_failure_after_full_materialization_also_empties_the_workspace(tmp_path: Path) -> None:
    request, objects, identity = _published_request(tmp_path)
    root = tmp_path / "worker"

    def _boom(_root: Path):
        assert (root / "profile.json").is_file()  # materialization really happened
        raise RuntimeError("predecessor unavailable")

    with pytest.raises(RuntimeError, match="predecessor unavailable"):
        _run_worker(request, objects, identity, root, predecessor_loader=_boom)
    assert not any(root.iterdir())
    assert f"{request.prefix}/failure.json" in objects.values


def test_failure_never_deletes_a_workspace_the_worker_did_not_start_empty(tmp_path: Path) -> None:
    """Ownership guard: a foreign non-empty root is refused and left exactly as found."""
    request, objects, identity = _published_request(tmp_path)
    root = tmp_path / "worker"
    root.mkdir()
    (root / "keep.txt").write_bytes(b"operator data")

    with pytest.raises(FileExistsError, match="empty real directory"):
        _run_worker(request, objects, identity, root)
    assert (root / "keep.txt").read_bytes() == b"operator data"
    assert f"{request.prefix}/failure.json" in objects.values


def test_pre_created_empty_workspace_is_cleared_and_retry_reaches_identity(tmp_path: Path) -> None:
    """An existing EMPTY root is owned: a failed attempt empties it again for the retry.

    Pins ``_is_unused_workspace`` for the present-but-empty case; treating only an
    absent root as owned would leave the half-written files and every retry would
    die on ``FileExistsError`` before reaching the identity check.
    """
    request, objects, identity = _published_request(tmp_path)
    objects.values[f"{request.prefix}/inputs/{request.inputs[1].path}"] = b"changed"
    root = tmp_path / "worker"
    root.mkdir()

    with pytest.raises(ValueError, match="remote input object identity mismatch"):
        _run_worker(request, objects, identity, root)
    assert root.is_dir()
    assert not any(root.iterdir())
    with pytest.raises(ValueError, match="remote input object identity mismatch"):
        _run_worker(request, objects, identity, root)


def test_workspace_is_already_empty_when_the_failure_receipt_is_published(tmp_path: Path) -> None:
    """The receipt is the retry signal, so the root is cleared BEFORE it appears."""
    request, objects, identity = _published_request(tmp_path)
    objects.values[f"{request.prefix}/inputs/{request.inputs[1].path}"] = b"changed"
    root = tmp_path / "worker"
    at_publish: list[list[str]] = []
    real_publish = objects.publish

    def _spy(key: str, payload: bytes, *, expected_sha256: str):
        if key.endswith("/failure.json"):
            at_publish.append(sorted(p.name for p in root.iterdir()))
        return real_publish(key, payload, expected_sha256=expected_sha256)

    objects.publish = _spy  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="remote input object identity mismatch"):
        _run_worker(request, objects, identity, root)
    assert at_publish == [[]]


def test_failed_cleanup_still_publishes_the_failure_receipt_and_surfaces_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.historical.d_ladder import d_ladder_remote

    request, objects, identity = _published_request(tmp_path)
    objects.values[f"{request.prefix}/inputs/{request.inputs[1].path}"] = b"changed"

    def _cannot_clear(_root: Path) -> None:
        raise OSError("cleanup refused")

    monkeypatch.setattr(d_ladder_remote, "_clear_workspace", _cannot_clear)
    with pytest.raises(OSError, match="cleanup refused") as caught:
        _run_worker(request, objects, identity, tmp_path / "worker")
    assert isinstance(caught.value.__context__, ValueError)
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


# ------------------------------------------------ lowercase SHA boundaries (P14-SHA-SEQ)
# Four BOUNDARY rows of docs/plans/measurements/p14-sha-callers.txt moved from the
# any-case kernel predicate to strict_json's lowercase one. Each refusal keeps the
# exception type and message the any-case predicate used for a malformed digest.
_MIXED_CASE_DIGEST = "aB" * 32


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("attempt_sha256", "remote attempt"),
        ("authorization_digest", "remote expected authorization"),
    ],
)
def test_remote_request_refuses_a_mixed_case_digest_with_the_unchanged_message(
    tmp_path: Path, field: str, label: str
) -> None:
    request = request_fixture(tmp_path)
    with pytest.raises(ValueError) as refused:
        replace(request, **{field: _MIXED_CASE_DIGEST})
    assert type(refused.value) is ValueError
    assert str(refused.value) == f"{label} must be a 64-character hexadecimal SHA-256 digest"
    assert replace(request, **{field: "ab" * 32}).digest  # lowercase still constructs


@pytest.mark.parametrize(
    ("attempt", "image", "label"),
    [
        (_MIXED_CASE_DIGEST, "sha256:" + "d" * 64, "worker attempt"),
        ("c" * 64, "sha256:" + _MIXED_CASE_DIGEST, "image digest"),
    ],
)
def test_serve_refuses_a_mixed_case_digest_with_the_unchanged_message(
    attempt: str, image: str, label: str
) -> None:
    import entropic_substrate.historical.d_ladder.d_ladder_worker_command as command

    arguments = argparse.Namespace(attempt_sha256=attempt)
    environ = {
        "RUNPOD_POD_ID": "pod123",
        "D_LADDER_CODE_COMMIT": "b" * 40,
        "D_LADDER_IMAGE_DIGEST": image,
    }
    objects = _ObjectStore()
    with pytest.raises(ValueError) as refused:
        command._serve(arguments, environ, objects)  # pyright: ignore[reportPrivateUsage]
    assert type(refused.value) is ValueError
    assert str(refused.value) == f"{label} must be a 64-character hexadecimal SHA-256 digest"
    assert not objects.publishes  # refusal precedes any publication


def test_unreadable_workspace_still_publishes_the_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The very first workspace read is inside the guarded region, so it is receipted."""
    from entropic_substrate.historical.d_ladder import d_ladder_remote

    request, objects, identity = _published_request(tmp_path)

    def _unreadable(_root: Path) -> bool:
        raise OSError("workspace unreadable")

    monkeypatch.setattr(d_ladder_remote, "_is_unused_workspace", _unreadable)
    with pytest.raises(OSError, match="workspace unreadable"):
        _run_worker(request, objects, identity, tmp_path / "worker")
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["error_type"] == "OSError"


def test_unknown_workspace_ownership_never_clears_foreign_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ruling 20 safe default: when ownership cannot be read, nothing is cleared.

    ``owns_workspace`` starts False; a failing ``_is_unused_workspace`` must leave a
    non-empty foreign root untouched while still publishing the typed receipt.
    """
    from entropic_substrate.historical.d_ladder import d_ladder_remote

    request, objects, identity = _published_request(tmp_path)
    root = tmp_path / "worker"
    root.mkdir()
    (root / "keep.txt").write_bytes(b"operator data")

    def _unreadable(_root: Path) -> bool:
        raise OSError("workspace unreadable")

    monkeypatch.setattr(d_ladder_remote, "_is_unused_workspace", _unreadable)
    with pytest.raises(OSError, match="workspace unreadable"):
        _run_worker(request, objects, identity, root)
    assert (root / "keep.txt").read_bytes() == b"operator data"
    failure = json.loads(objects.values[f"{request.prefix}/failure.json"])
    assert failure["error_type"] == "OSError"


def test_clear_workspace_removes_subdirs_and_does_not_follow_symlink_children(
    tmp_path: Path,
) -> None:
    """Directory arm and symlink arm of the cleanup, pinned directly."""
    from entropic_substrate.historical.d_ladder import d_ladder_remote

    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "c.txt").write_bytes(b"x")
    (root / "d.txt").write_bytes(b"y")
    outside.mkdir()
    (outside / "precious.txt").write_bytes(b"z")
    (root / "link").symlink_to(outside, target_is_directory=True)

    d_ladder_remote._clear_workspace(root)  # pyright: ignore[reportPrivateUsage]

    assert root.is_dir()
    assert not any(root.iterdir())
    assert (outside / "precious.txt").read_bytes() == b"z"


def test_failure_after_a_subdirectory_was_materialized_still_empties_root(
    tmp_path: Path,
) -> None:
    """Run-time shape: the executor leaves ``root/run/...`` behind when it fails mid-run."""
    request, objects, identity = _published_request(tmp_path)
    root = tmp_path / "worker"

    def _boom(_root: Path):
        (root / "run" / "checkpoints").mkdir(parents=True)
        (root / "run" / "checkpoints" / "step-000001.pt").write_bytes(b"partial")
        raise RuntimeError("mid-run failure")

    with pytest.raises(RuntimeError, match="mid-run failure"):
        _run_worker(request, objects, identity, root, predecessor_loader=_boom)
    assert root.is_dir()
    assert not any(root.iterdir())
