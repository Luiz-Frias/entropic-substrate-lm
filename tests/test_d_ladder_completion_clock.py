"""Remote completion carries a measured worker end clock, never arrival time."""

import hashlib
from datetime import timedelta

import pytest

from entropic_substrate.d_ladder_remote import materialize_remote_output
from entropic_substrate.experiments.d1_primitives import canonical_json

from .test_d1_trainer_object_transport import _ObjectStore
from .test_d_ladder_remote_request import request_fixture

pytestmark = pytest.mark.unit


def receipt(root, *, version=2, ended_at=None):
    request = request_fixture(root)
    objects = _ObjectStore()
    payload = b"measured output"
    output_key = f"{request.prefix}/outputs/record.json"
    objects.values[output_key] = payload
    row = {
        "schema": f"d-ladder-remote-complete-v{version}",
        "request_sha256": request.digest,
        "authorization_digest": request.authorization_digest,
        "logical_run_id": request.start.logical_run_id,
        "execution_segment_id": request.start.execution_segment_id,
        "result": {},
        "artifacts": [
            {
                "path": "record.json",
                "key": output_key,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
        "meaning": "executor-returned-and-output-bytes-exported",
    }
    if version == 2:
        row["trainer_ended_at"] = (
            ended_at
            if ended_at is not None
            else (request.start.trainer_started_at + timedelta(seconds=1)).isoformat()
        )
    key = f"{request.prefix}/complete.json"
    objects.values[key] = canonical_json(row).encode()
    return objects, request, objects.read(key).identity, row


@pytest.mark.parametrize("version", [1, 2])
def test_explicit_completion_versions_preserve_available_clock(tmp_path, version):
    objects, request, identity, expected = receipt(tmp_path, version=version)
    observed = materialize_remote_output(objects, request, identity, tmp_path / "output")
    assert observed == expected
    assert ("trainer_ended_at" in observed) is (version == 2)


@pytest.mark.parametrize("value", [True, "2026-08-01T00:00:00", "1900-01-01T00:00:00+00:00"])
def test_invalid_worker_end_clock_is_refused_before_materialization(tmp_path, value):
    objects, request, identity, _ = receipt(tmp_path, ended_at=value)
    with pytest.raises(ValueError, match="end"):
        materialize_remote_output(objects, request, identity, tmp_path / "output")
    assert not (tmp_path / "output").exists()
