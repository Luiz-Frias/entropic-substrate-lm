"""Byte transfer ordering is tested separately from native model admission."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_request import request

from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest
from entropic_substrate.ladder.run.worker_transport import (
    materialize_probe_request,
    publish_probe_request,
)

pytestmark = pytest.mark.unit


def inputs(root: Path) -> ProbeRequest:
    (root / "source").mkdir()
    data = b"byte-transfer fixture; not a native corpus"
    publish_d1_atomic_bytes(root, "source/prepared-corpus.json", data, label="test input")
    digest = sha256(data).hexdigest()
    value = request()
    return replace(
        value,
        profile=replace(value.profile, source_digest=digest),
        inputs=(ArtifactIdentity("source/prepared-corpus.json", ContentHash("sha256", digest)),),
    )


def test_inputs_precede_request_marker_and_materialize_exactly(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    marker = publish_probe_request(store, value, tmp_path)
    assert store.publishes[-1] == f"{value.prefix}/request.json"
    target = tmp_path / "worker"
    target.mkdir()
    observed = materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)
    assert observed == value
    assert (target / "source/prepared-corpus.json").read_bytes() == (
        tmp_path / "source/prepared-corpus.json"
    ).read_bytes()
    assert (target / "request.json").read_bytes() == value.payload
    with pytest.raises(ValueError, match="empty"):
        materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)


def test_tampering_cannot_publish_a_materialization_marker(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    marker = publish_probe_request(store, value, tmp_path)
    store.values[f"{value.prefix}/inputs/source/prepared-corpus.json"] = b"different"
    target = tmp_path / "worker"
    target.mkdir()
    with pytest.raises(ValueError, match="identity"):
        materialize_probe_request(store, marker, target, max_request_bytes=1_000_000)
    assert not (target / "request.json").exists()


def test_budget_failure_precedes_all_remote_writes(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    value = replace(value, limits=replace(value.limits, max_total_bytes=1))
    with pytest.raises(ValueError, match="budget"):
        publish_probe_request(store, value, tmp_path)
    assert store.publishes == []


def test_local_input_hash_is_checked_before_publication(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    (tmp_path / "source/prepared-corpus.json").write_bytes(b"replaced")
    store = _ObjectStore()
    with pytest.raises(ValueError, match="identity"):
        publish_probe_request(store, value, tmp_path)
    assert store.publishes == []


def test_missing_readback_never_publishes_the_request_marker(tmp_path: Path) -> None:
    class DroppingStore(_ObjectStore):
        def publish(
            self, key: str, payload: bytes, *, expected_sha256: str
        ) -> D1TrainerMailboxObjectIdentity:
            result = super().publish(key, payload, expected_sha256=expected_sha256)
            del self.values[key]
            return result

    value = inputs(tmp_path)
    store = DroppingStore()
    with pytest.raises(ValueError, match="readback"):
        publish_probe_request(store, value, tmp_path)
    assert f"{value.prefix}/request.json" not in store.publishes


def test_request_namespace_cannot_be_relabelled(tmp_path: Path) -> None:
    value = inputs(tmp_path)
    store = _ObjectStore()
    wrong = store.publish("other/request.json", value.payload, expected_sha256=value.digest)
    target = tmp_path / "worker"
    target.mkdir()
    with pytest.raises(ValueError, match="prefix"):
        materialize_probe_request(store, wrong, target, max_request_bytes=1_000_000)
    assert not list(target.iterdir())
