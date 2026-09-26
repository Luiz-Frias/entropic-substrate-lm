"""One numerical loop for phase transitions, evaluation isolation, and resume."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_sentence_profile import profile
from tests.test_sentence_sampling import corpus

from entropic_substrate.core.sentence.corpus import SplitPolicy, prepare_documents
from entropic_substrate.ladder.capture.capture import write_prepared_corpus
from entropic_substrate.ladder.run.checkpoint import save_checkpoint
from entropic_substrate.ladder.run.execution import SentenceExecution

pytestmark = pytest.mark.unit


def engine(root: Path) -> SentenceExecution:
    source = root / "source"
    source.mkdir()
    original = corpus()
    prepared = prepare_documents(
        original.candidates,
        replace(original.spec, splits=SplitPolicy(validation_buckets=2000, test_buckets=1000)),
    )
    artifact = write_prepared_corpus(source, prepared, max_bytes=1_000_000)
    return SentenceExecution(
        replace(profile(), source_digest=artifact.content_hash.digest),
        source,
        max_source_bytes=1_000_000,
    )


def test_real_loop_reaches_both_phases_and_records_actual_source_exposure(tmp_path: Path) -> None:
    run = engine(tmp_path)
    with pytest.raises(ValueError, match="warm-up"):
        run.begin_prediction()
    first = run.advance()
    assert first["phase"] == "codec_warmup" and first["step"] == 1
    assert first["source_batch"] and first["source_examples"]
    assert first["source_bytes"] > 0 and first["wall_seconds"] >= 0
    assert first["cuda_seconds"] is None
    run.advance()
    with pytest.raises(ValueError, match="complete"):
        run.advance()
    run.begin_prediction()
    assert run.step == 0 and run.sampler.cursor == 0
    assert all(not p.requires_grad for p in run.model.codec.encoder.parameters())
    assert not run.optimizer.state
    for _ in range(run.profile.schedule.steps):
        row = run.advance()
    assert row["step"] == 4 and row["run_identity"] == run.profile.identity
    assert row["gradients"]["encoder"]["status"] == "structurally_absent_frozen_encoder"


def test_evaluation_preserves_training_rng_and_weights(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    run.advance()
    rng = torch.get_rng_state()
    before = [p.detach().clone() for p in run.model.parameters()]
    evaluation = run.evaluate()
    assert evaluation["meaning"] == "importance_weighted_log_marginal_estimate"
    assert evaluation["step"] == 1 and evaluation["examples"]
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert run.model.training
    for actual, expected in zip(run.model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fresh_engine_resume_matches_uninterrupted_next_step(tmp_path: Path) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    run.advance()
    checkpoint = save_checkpoint(
        tmp_path,
        "step.pt",
        run.model,
        run.optimizer,
        run.sampler,
        phase=run.phase,
        step=run.step,
        run_identity=run.profile.identity,
        max_bytes=10_000_000,
    )
    expected = run.advance()
    restored = SentenceExecution(run.profile, tmp_path / "source", max_source_bytes=1_000_000)
    restored.resume(
        tmp_path, "step.pt", checkpoint.content_hash, phase="prediction", max_bytes=10_000_000
    )
    actual = restored.advance()
    for field in (
        "step",
        "source_batch",
        "source_examples",
        "objective",
        "gradients",
        "decoder_ce",
        "future_nll_per_dimension",
    ):
        assert actual[field] == expected[field]
    with pytest.raises(ValueError, match="fresh"):
        restored.resume(
            tmp_path, "step.pt", checkpoint.content_hash, phase="prediction", max_bytes=10_000_000
        )


def test_source_mismatch_fails_before_model_initialization(tmp_path: Path) -> None:
    run = engine(tmp_path)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="digest"):
        SentenceExecution(profile(), tmp_path / "source", max_source_bytes=1_000_000)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert run.step == 0


def test_inventories_measure_all_deployed_and_phase_active_parameters(tmp_path: Path) -> None:
    run = engine(tmp_path)
    inventory = run.inventory()
    assert inventory["total_deployed"] == sum(p.numel() for p in run.model.parameters())
    assert inventory["optimizer_active"] == sum(
        p.numel() for g in run.optimizer.param_groups for p in g["params"]
    )
    assert sum(inventory["components"].values()) == inventory["total_deployed"]


def test_empty_validation_split_is_rejected_without_training_fallback(tmp_path: Path) -> None:
    artifact = write_prepared_corpus(tmp_path, corpus(), max_bytes=1_000_000)
    rng = torch.get_rng_state()
    with pytest.raises(ValueError, match="empty"):
        SentenceExecution(
            replace(profile(), source_digest=artifact.content_hash.digest),
            tmp_path,
            max_source_bytes=1_000_000,
        )
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_postcommit_resource_failure_is_not_a_committed_evidence_row(tmp_path: Path) -> None:
    run = engine(tmp_path)

    class BrokenTimer:
        def start(self, phase: object) -> object:
            return object()

        def stop(self, ticket: object, phase: object) -> float:
            raise RuntimeError("measured timer failure")

    run.timer = BrokenTimer()
    with pytest.raises(RuntimeError, match="timer failure"):
        run.advance()
    assert run.last_attempt["status"] == "execution_defect"
    assert run.last_attempt["resource_error"] == "measured timer failure"
    run.timer = None
    with pytest.raises(ValueError, match="recover"):
        run.advance()


def test_zero_direction_gap_is_retained_without_infrastructure_reclassification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    run = engine(tmp_path)

    def collapsed(*args: object) -> None:
        raise UndefinedDirection(2)

    monkeypatch.setattr(run.model.codec, "reconstruct", collapsed)
    with pytest.raises(UndefinedDirection):
        run.advance()
    assert run.last_attempt["status"] == "scientific_gap"
    assert run.last_attempt["affected_positions"] == 2
    with pytest.raises(ValueError, match="recover"):
        run.advance()


def test_rejected_resume_cannot_train_a_frozen_unwarmed_model(tmp_path: Path) -> None:
    from entropic_substrate.kernel.lifecycle import ContentHash

    run = engine(tmp_path)
    with pytest.raises(ValueError):
        run.resume(
            tmp_path,
            "absent.pt",
            ContentHash("sha256", run.profile.identity),
            phase="prediction",
            max_bytes=1000,
        )
    with pytest.raises(ValueError, match="recover"):
        run.advance()


def test_engine_resume_retains_checkpointed_generation_feedback(tmp_path: Path) -> None:
    from entropic_substrate.core.sentence.rollout import advance_rollout, begin_rollout

    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    run.model.eval()
    state = advance_rollout(run.model, begin_rollout(run.model, (b"A.",), run.codec), run.codec)
    checkpoint = save_checkpoint(
        tmp_path,
        "feedback.pt",
        run.model,
        run.optimizer,
        run.sampler,
        phase=run.phase,
        step=0,
        run_identity=run.profile.identity,
        rollout=state,
        max_bytes=10_000_000,
    )
    fresh = SentenceExecution(run.profile, tmp_path / "source", max_source_bytes=1_000_000)
    fresh.resume(
        tmp_path, "feedback.pt", checkpoint.content_hash, phase="prediction", max_bytes=10_000_000
    )
    assert fresh.rollout is not None and fresh.rollout.observations == state.observations
