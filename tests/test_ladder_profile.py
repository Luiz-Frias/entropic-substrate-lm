"""Complete profile codecs preserve mechanism and predecessor identities."""

import hashlib
import json
from pathlib import Path

import pytest

from entropic_substrate.historical.d_ladder.ladder_profile import (
    load_rung_profile,
    rung_profile_mapping,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.config import configs_dir
from entropic_substrate.ladder.gate.ladder_campaign import artifact_mapping
from entropic_substrate.ladder.register.d1_config import load_d1_profile
from entropic_substrate.ladder.register.ladder import CoarseSpec, Rung, RungSpec, Treatment

pytestmark = pytest.mark.unit


def write_artifact(root: Path, name: str, payload: bytes) -> ArtifactIdentity:
    (root / name).write_bytes(payload)
    return ArtifactIdentity(name, ContentHash("sha256", hashlib.sha256(payload).hexdigest()))


def profile_chain(root: Path) -> tuple[ArtifactIdentity, RungSpec]:
    content = (configs_dir() / "d1-both.yaml").read_bytes()
    base = write_artifact(root, "base.yaml", content)
    spec = RungSpec(load_d1_profile(root / "base.yaml").draft, Rung.D2, Treatment.S, 7)
    previous = None
    for rung in Rung:
        if rung is not Rung.D2:
            spec = RungSpec(
                spec.base,
                rung,
                Treatment.S,
                7,
                horizons=(4,),
                coarse=None if rung is Rung.D3 else CoarseSpec("fixed"),
                permutation=tuple(reversed(range(256))) if rung is Rung.D5 else None,
                predecessor=spec,
            )
        row = rung_profile_mapping(spec, base_profile=base, base_scale="full", predecessor=previous)
        previous = write_artifact(root, f"{rung.value}.json", json.dumps(row).encode())
    assert previous is not None
    return previous, spec


def test_complete_d2_through_d5_profile_roundtrip(tmp_path: Path) -> None:
    artifact, expected = profile_chain(tmp_path)
    observed = load_rung_profile(tmp_path, artifact)
    assert observed == expected
    assert observed.digest == expected.digest


def test_changed_bound_base_cannot_be_loaded(tmp_path: Path) -> None:
    artifact, _ = profile_chain(tmp_path)
    with (tmp_path / "base.yaml").open("ab") as handle:
        handle.write(b"\n# changed\n")
    with pytest.raises(ValueError, match="digest"):
        load_rung_profile(tmp_path, artifact)


@pytest.mark.parametrize("change", ["unknown", "missing", "spec_digest", "predecessor", "boolean"])
def test_profile_rejects_drift_and_implicit_defaults(tmp_path: Path, change: str) -> None:
    artifact, _ = profile_chain(tmp_path)
    row = json.loads((tmp_path / artifact.path).read_bytes())
    if change == "unknown":
        row["future_private_default"] = True
    elif change == "missing":
        del row["solver"]["relative_tolerance"]
    elif change == "spec_digest":
        row["spec_digest"] = "0" * 64
    elif change == "predecessor":
        row["predecessor"] = artifact_mapping(artifact)
    else:
        row["solver"]["damping"] = True
    broken = write_artifact(tmp_path, "broken.json", json.dumps(row).encode())
    with pytest.raises((TypeError, ValueError)):
        load_rung_profile(tmp_path, broken)
