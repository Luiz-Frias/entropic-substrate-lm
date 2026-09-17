"""Journal custody replays sources and rejects plausible but mismatched measurements."""

import copy
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.sentence_prediction.journal import read_journal, write_journal

pytestmark = pytest.mark.unit


def test_journal_roundtrip_is_source_bound_and_does_not_change_model_rng(tmp_path: Path) -> None:
    run = engine(tmp_path)
    rows = (run.advance(), run.advance())
    rng = torch.get_rng_state()
    artifact = write_journal(tmp_path, "warmup.json", run, rows, parent=None, max_bytes=1_000_000)
    result = read_journal(
        tmp_path,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=1_000_000,
        max_source_bytes=1_000_000,
    )
    assert result["start_step"] == 1 and result["end_step"] == 2
    assert result["phase"] == "codec_warmup" and result["parent"] is None
    assert result["run_class"] == "local_tiny"
    assert result["scientific_authority"] == "none"
    assert result["rows"][0]["source_batch"] == rows[0]["source_batch"]
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


@pytest.mark.parametrize(
    "field",
    [
        "source_batch",
        "source_examples",
        "source_bytes",
        "objective",
        "aggregate_norm",
        "gradients",
        "step",
        "status",
    ],
)
def test_forged_training_rows_are_refused_before_publication(tmp_path: Path, field: str) -> None:
    run = engine(tmp_path)
    row = copy.deepcopy(run.advance())
    if field == "source_batch":
        row[field] = "wrong"
    elif field == "source_examples":
        row[field][0]["target"][0] += 1
    elif field == "source_bytes":
        row[field] += 1
    elif field in ("objective", "aggregate_norm"):
        row[field] += 1.0
    elif field == "gradients":
        row[field]["encoder"]["status"] = "measured_zero"
    elif field == "step":
        row[field] = True
    else:
        row[field] = "started"
    with pytest.raises(ValueError):
        write_journal(tmp_path, "forged.json", run, (row,), parent=None, max_bytes=1_000_000)
    assert not (tmp_path / "forged.json").exists()


def test_prediction_journal_keeps_absent_encoder_and_active_future_loss(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    rows = (run.advance(), run.advance())
    artifact = write_journal(
        tmp_path, "prediction.json", run, rows, parent=None, max_bytes=1_000_000
    )
    payload = read_journal(
        tmp_path,
        artifact,
        run.profile,
        run.source_root,
        max_bytes=1_000_000,
        max_source_bytes=1_000_000,
    )
    assert payload["rows"][0]["gradients"]["encoder"]["norm"] is None
    assert payload["rows"][0]["future_nll_per_dimension"] is not None


def test_nonfinite_json_and_rehashed_source_forgery_are_rejected(tmp_path: Path) -> None:
    import json

    from entropic_substrate.evidence.lifecycle import ArtifactIdentity

    run = engine(tmp_path)
    artifact = write_journal(
        tmp_path, "valid.json", run, (run.advance(),), parent=None, max_bytes=1_000_000
    )
    payload = json.loads(Path(artifact.path).read_bytes())
    payload["rows"][0]["wall_seconds"] = float("nan")
    raw = json.dumps(payload).encode()
    (tmp_path / "nonfinite.json").write_bytes(raw)
    forged = ArtifactIdentity("nonfinite.json", ContentHash("sha256", sha256(raw).hexdigest()))
    with pytest.raises(ValueError, match="nonfinite"):
        read_journal(
            tmp_path,
            forged,
            run.profile,
            run.source_root,
            max_bytes=1_000_000,
            max_source_bytes=1_000_000,
        )


def test_duplicate_json_keys_and_incomplete_step_prefix_are_refused(tmp_path: Path) -> None:
    from entropic_substrate.evidence.lifecycle import ArtifactIdentity

    run = engine(tmp_path)
    rows = (run.advance(), run.advance())
    with pytest.raises(ValueError, match="consecutive"):
        write_journal(
            tmp_path, "missing.json", run, (rows[0], rows[0]), parent=None, max_bytes=1_000_000
        )
    raw = b'{"schema":"first","schema":"second"}'
    (tmp_path / "duplicate.json").write_bytes(raw)
    with pytest.raises(ValueError, match="duplicate"):
        read_journal(
            tmp_path,
            ArtifactIdentity("duplicate.json", ContentHash("sha256", sha256(raw).hexdigest())),
            run.profile,
            run.source_root,
            max_bytes=1_000_000,
            max_source_bytes=1_000_000,
        )


def test_swapped_in_memory_corpus_cannot_override_the_bound_capture(tmp_path: Path) -> None:
    from tests.test_sentence_sampling import corpus

    from entropic_substrate.sentence_prediction.sampling import SentenceSampler

    run = engine(tmp_path)
    run.corpus = corpus()
    run.sampler = SentenceSampler(run.corpus, run.sampler.spec, run.codec, run.device)
    row = run.advance()
    with pytest.raises(ValueError, match="source coordinates"):
        write_journal(tmp_path, "swapped.json", run, (row,), parent=None, max_bytes=1_000_000)
