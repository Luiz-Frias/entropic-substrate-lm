"""Remote request codecs are independently testable before worker integration."""

from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.d_ladder_remote_request import (
    RemoteLadderRequest,
    parse_remote_request,
    serialize_remote_request,
)
from entropic_substrate.evidence.lifecycle import ActorIdentity, ActorRole

from .test_ladder_profile import write_artifact
from .test_resource_provenance import _start

pytestmark = pytest.mark.unit


def request_fixture(root: Path) -> RemoteLadderRequest:
    inputs = tuple(
        write_artifact(root, name, b"{}")
        for name in ("profile.json", "campaign.json", "enwik8", "predecessor.json")
    )
    return RemoteLadderRequest(
        profile=inputs[0],
        campaign=inputs[1],
        source=inputs[2],
        inputs=inputs,
        operator=ActorIdentity("operator", "session-1", ActorRole.OPERATOR),
        authorization_digest="a" * 64,
        start=_start(),
        code_commit="b" * 40,
        pod_id="pod123",
        attempt_sha256="c" * 64,
        predecessor_declaration=inputs[3],
    )


def test_remote_request_roundtrip_does_not_construct_authority(tmp_path: Path) -> None:
    request = request_fixture(tmp_path)
    assert parse_remote_request(serialize_remote_request(request)) == request
    assert request.manifest["schema"] == "d-ladder-remote-request-v1"
    assert request.start.image_digest in serialize_remote_request(request).decode()


@pytest.mark.parametrize("change", ["missing", "duplicate", "path", "short_commit"])
def test_remote_request_rejects_unbound_inputs_and_ambiguous_identity(
    tmp_path: Path,
    change: str,
) -> None:
    request = request_fixture(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        if change == "missing":
            replace(request, inputs=request.inputs[:2])
        elif change == "duplicate":
            replace(request, inputs=(*request.inputs, request.inputs[0]))
        elif change == "path":
            replace(request, pod_id="../escape")
        else:
            replace(request, code_commit="b" * 7)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    data = serialize_remote_request(request_fixture(tmp_path))
    with pytest.raises(ValueError, match="duplicate"):
        parse_remote_request(b'{"schema":"duplicate",' + data[1:])
