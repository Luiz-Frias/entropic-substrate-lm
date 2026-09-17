"""Hash-bound recovery reproduces training and generation without partial mutation."""

from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.test_sentence_model import tiny_model
from tests.test_sentence_sampling import sampler

from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.sentence_prediction.checkpoint import load_checkpoint, save_checkpoint
from entropic_substrate.sentence_prediction.codec import CodecSpec
from entropic_substrate.sentence_prediction.model import SentencePredictor
from entropic_substrate.sentence_prediction.objective import prediction_loss
from entropic_substrate.sentence_prediction.records import content_identity
from entropic_substrate.sentence_prediction.rollout import advance_rollout, begin_rollout
from entropic_substrate.sentence_prediction.sampling import SentenceSampler
from entropic_substrate.sentence_prediction.training import (
    SentenceStep,
    apply_sentence_step,
    build_sentence_optimizer,
)

pytestmark = pytest.mark.unit
RUN_ID = content_identity(("excluded-checkpoint-fixture",))
LIMIT = 10_000_000


def setup_run() -> tuple[SentencePredictor, torch.optim.AdamW, SentenceSampler]:
    model = tiny_model()
    source = sampler()
    # Sampling fixture has spans longer than the width-8 model's development cap.
    model = SentencePredictor(
        CodecSpec(8, 1, 8, 1, 4, 64), model.spine.spec, model.codec.law_spec, consumer="isotropic"
    )
    model.codec.freeze_encoder()
    optimizer = build_sentence_optimizer(model, "prediction", learning_rate=1e-3, weight_decay=0.01)
    return model, optimizer, source


def step(
    model: SentencePredictor, optimizer: torch.optim.Optimizer, source: SentenceSampler
) -> tuple[str, SentenceStep]:
    batch = source.next_batch()
    result = apply_sentence_step(
        model,
        optimizer,
        "prediction",
        lambda: prediction_loss(model, batch.history, batch.target, future_weight=0.3).total,
        clip_norm=1.0,
    )
    return batch.identity, result


def test_disk_resume_reproduces_next_training_transaction(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    step(model, optimizer, source)
    artifact = save_checkpoint(
        tmp_path,
        "step-1.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=1,
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    expected_batch, expected_result = step(model, optimizer, source)
    restored = load_checkpoint(
        tmp_path,
        "step-1.pt",
        artifact.content_hash,
        model,
        optimizer,
        source,
        phase="prediction",
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    assert restored.model is not model and restored.optimizer is not optimizer
    assert restored.step == restored.sampler.cursor == 1
    actual_batch, actual_result = step(restored.model, restored.optimizer, restored.sampler)
    assert actual_batch == expected_batch
    assert actual_result == expected_result
    for left, right in zip(model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_disk_resume_preserves_pending_rollout_and_shadow(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    model.eval()
    state = advance_rollout(model, begin_rollout(model, (b"A.",), source.codec), source.codec)
    artifact = save_checkpoint(
        tmp_path,
        "rollout.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=0,
        run_identity=RUN_ID,
        rollout=state,
        max_bytes=LIMIT,
    )
    expected = advance_rollout(model, state, source.codec)
    restored = load_checkpoint(
        tmp_path,
        "rollout.pt",
        artifact.content_hash,
        model,
        optimizer,
        source,
        phase="prediction",
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    assert restored.rollout is not None and not restored.model.training
    actual = advance_rollout(restored.model, restored.rollout, source.codec)
    assert actual.observations == expected.observations
    assert actual.status == expected.status
    torch.testing.assert_close(actual.context, expected.context, atol=0, rtol=0)


def test_hash_and_identity_rejections_leave_running_state_unchanged(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    artifact = save_checkpoint(
        tmp_path,
        "step-0.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=0,
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    weights = [p.detach().clone() for p in model.parameters()]
    rng = torch.get_rng_state()
    for digest, run in (
        (ContentHash("sha256", content_identity(("wrong",))), RUN_ID),
        (artifact.content_hash, content_identity(("other-run",))),
    ):
        with pytest.raises(ValueError):
            load_checkpoint(
                tmp_path,
                "step-0.pt",
                digest,
                model,
                optimizer,
                source,
                phase="prediction",
                run_identity=run,
                max_bytes=LIMIT,
            )
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for p, original in zip(model.parameters(), weights, strict=True):
            torch.testing.assert_close(p, original, atol=0, rtol=0)


def test_save_refuses_uncommitted_cursor_and_occupied_or_small_outputs(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    with pytest.raises(ValueError, match="cursor"):
        save_checkpoint(
            tmp_path,
            "wrong.pt",
            model,
            optimizer,
            source,
            phase="prediction",
            step=1,
            run_identity=RUN_ID,
            max_bytes=LIMIT,
        )


@pytest.mark.parametrize(
    "mutation", ["precision", "parameter_order", "version", "reader", "legacy"]
)
def test_rehashed_incompatible_payload_is_rejected(tmp_path: Path, mutation: str) -> None:
    model, optimizer, source = setup_run()
    step(model, optimizer, source)
    save_checkpoint(
        tmp_path,
        "good.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=1,
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    payload = torch.load(tmp_path / "good.pt", weights_only=True)
    if mutation == "precision":
        payload["optimizer"]["state"][0]["exp_avg"] = payload["optimizer"]["state"][0][
            "exp_avg"
        ].double()
    elif mutation == "parameter_order":
        group = payload["optimizer"]["param_groups"][0]["params"]
        states = payload["optimizer"]["state"]
        first, second = next(
            (i, j)
            for i in range(len(group))
            for j in range(i + 1, len(group))
            if states[group[i]]["exp_avg"].shape == states[group[j]]["exp_avg"].shape
        )
        group[first], group[second] = group[second], group[first]
    elif mutation == "legacy":
        payload["schema"] = "h-training-checkpoint-v1"
        payload.pop("measurement")
    else:
        payload["schema" if mutation == "version" else "reader"] = "unknown"
    torch.save(payload, tmp_path / "forged.pt")
    digest = ContentHash("sha256", sha256((tmp_path / "forged.pt").read_bytes()).hexdigest())
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="original reader" if mutation == "legacy" else None):
        load_checkpoint(
            tmp_path,
            "forged.pt",
            digest,
            model,
            optimizer,
            source,
            phase="prediction",
            run_identity=RUN_ID,
            max_bytes=LIMIT,
        )
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def test_small_and_occupied_outputs_preserve_existing_artifacts(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    with pytest.raises(ValueError, match="limit"):
        save_checkpoint(
            tmp_path,
            "small.pt",
            model,
            optimizer,
            source,
            phase="prediction",
            step=0,
            run_identity=RUN_ID,
            max_bytes=10,
        )
    assert not list(tmp_path.iterdir())
    save_checkpoint(
        tmp_path,
        "once.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=0,
        run_identity=RUN_ID,
        max_bytes=LIMIT,
    )
    with pytest.raises(ValueError, match="occupied"):
        save_checkpoint(
            tmp_path,
            "once.pt",
            model,
            optimizer,
            source,
            phase="prediction",
            step=0,
            run_identity=RUN_ID,
            max_bytes=LIMIT,
        )


def test_checkpoint_requires_the_same_external_measurement_identity(tmp_path: Path) -> None:
    model, optimizer, source = setup_run()
    measurement = ContentHash("sha256", content_identity(("actual-step-journal",)))
    checkpoint = save_checkpoint(
        tmp_path,
        "bound.pt",
        model,
        optimizer,
        source,
        phase="prediction",
        step=0,
        run_identity=RUN_ID,
        measurement=measurement,
        max_bytes=LIMIT,
    )
    with pytest.raises(ValueError, match="measurement"):
        load_checkpoint(
            tmp_path,
            "bound.pt",
            checkpoint.content_hash,
            model,
            optimizer,
            source,
            phase="prediction",
            run_identity=RUN_ID,
            max_bytes=LIMIT,
        )
    restored = load_checkpoint(
        tmp_path,
        "bound.pt",
        checkpoint.content_hash,
        model,
        optimizer,
        source,
        phase="prediction",
        run_identity=RUN_ID,
        expected_measurement=measurement,
        max_bytes=LIMIT,
    )
    assert restored.step == 0
