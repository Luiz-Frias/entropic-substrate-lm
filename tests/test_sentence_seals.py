"""A valid journal and checkpoint must also belong to the same measured lineage."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.seals import read_seal, resume_seal, seal_step

pytestmark = pytest.mark.unit
LIMIT = 10_000_000


def test_sealed_chain_replays_warmup_transition_and_resume(tmp_path: Path) -> None:
    run = engine(tmp_path)
    warmup = seal_step(
        tmp_path, "first", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    run.begin_prediction()
    prediction = seal_step(
        tmp_path, "first", run, (run.advance(), run.advance()), parent=warmup, max_bytes=LIMIT
    )
    seal = read_seal(
        tmp_path,
        prediction,
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    assert seal.phase == "prediction" and seal.step == 2
    fresh = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    resume_seal(tmp_path, prediction, fresh, max_bytes=LIMIT)
    tail = seal_step(
        tmp_path,
        "resumed",
        fresh,
        (fresh.advance(), fresh.advance()),
        parent=prediction,
        max_bytes=LIMIT,
    )
    final = read_seal(
        tmp_path, tail, run.profile, run.source_root, max_bytes=LIMIT, max_source_bytes=1_000_000
    )
    assert final.step == 4 and final.chain_length == 3


def test_initial_or_phase_transition_gaps_are_refused(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    with pytest.raises(ValueError, match="initial"):
        seal_step(tmp_path, "gap", run, (run.advance(),), parent=None, max_bytes=LIMIT)
    assert not list(tmp_path.glob("gap*"))
    run.begin_prediction()
    with pytest.raises(ValueError, match="initial"):
        seal_step(tmp_path, "skip", run, (run.advance(),), parent=None, max_bytes=LIMIT)


def test_swapped_individually_valid_checkpoint_fails_measurement_binding(tmp_path: Path) -> None:
    run = engine(tmp_path)
    first = seal_step(
        tmp_path, "a", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    other = SentenceExecution(run.profile, run.source_root, max_source_bytes=1_000_000)
    torch.randn(11)
    second = seal_step(
        tmp_path, "b", other, (other.advance(), other.advance()), parent=None, max_bytes=LIMIT
    )
    payload = read_json(tmp_path, first, max_bytes=LIMIT)
    payload["checkpoint"] = read_json(tmp_path, second, max_bytes=LIMIT)["checkpoint"]
    raw = json.dumps(payload).encode()
    (tmp_path / "swapped.json").write_bytes(raw)
    forged = ArtifactIdentity("swapped.json", ContentHash("sha256", sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match="measurement"):
        read_seal(
            tmp_path,
            forged,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_seal_prefix_and_artifact_tampering_are_not_silently_repaired(tmp_path: Path) -> None:
    run = engine(tmp_path)
    rows = (run.advance(), run.advance())
    with pytest.raises(ValueError, match="prefix"):
        seal_step(tmp_path, "../escape", run, rows, parent=None, max_bytes=LIMIT)
    seal = seal_step(tmp_path, "one", run, rows, parent=None, max_bytes=LIMIT)
    payload = read_json(tmp_path, seal, max_bytes=LIMIT)
    checkpoint = tmp_path / payload["checkpoint"]["path"]
    checkpoint.write_bytes(b"not the captured checkpoint")
    with pytest.raises(ValueError, match="hash"):
        read_seal(
            tmp_path,
            seal,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_optimizer_drift_cannot_keep_the_old_profile_label(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.optimizer.defaults["lr"] *= 2
    for group in run.optimizer.param_groups:
        group["lr"] *= 2
    rows = (run.advance(), run.advance())
    with pytest.raises(ValueError, match="construction"):
        artifact = seal_step(tmp_path, "drift", run, rows, parent=None, max_bytes=LIMIT)
        read_seal(
            tmp_path,
            artifact,
            run.profile,
            run.source_root,
            max_bytes=LIMIT,
            max_source_bytes=1_000_000,
        )


def test_prediction_cannot_skip_unsealed_warmup_work(tmp_path: Path) -> None:
    run = engine(tmp_path)
    parent = seal_step(tmp_path, "partial", run, (run.advance(),), parent=None, max_bytes=LIMIT)
    run.advance()
    run.begin_prediction()
    with pytest.raises(ValueError, match="complete warm-up"):
        seal_step(tmp_path, "skip", run, (run.advance(),), parent=parent, max_bytes=LIMIT)


def test_seal_inspection_preserves_training_rng(tmp_path: Path) -> None:
    run = engine(tmp_path)
    artifact = seal_step(
        tmp_path, "one", run, (run.advance(), run.advance()), parent=None, max_bytes=LIMIT
    )
    before = torch.get_rng_state()
    read_seal(
        tmp_path,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)
