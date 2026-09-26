"""Document-bound source order is independent of latent RNG and byte recoding."""

from dataclasses import replace

import pytest
import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    PreparedCorpus,
    prepare_documents,
)
from entropic_substrate.core.sentence.records import SourceDocument
from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy

pytestmark = pytest.mark.unit


def corpus() -> PreparedCorpus:
    return prepare_documents(
        [
            SourceDocument(
                f"fixture-{i}", f"Item {i} is here. This is next. Then it ends.", "tiny-v1"
            )
            for i in range(30)
        ],
        PreparationSpec(30, 10000, "tiny-v1", segmentation=SegmentationPolicy(64)),
    )


def sampler(codec: SentenceByteCodec | None = None) -> SentenceSampler:
    return SentenceSampler(
        corpus(), SamplingSpec("train", 4, 2, 42), codec or SentenceByteCodec(), torch.device("cpu")
    )


def test_batch_spans_are_exact_causal_and_document_local() -> None:
    source = sampler()
    for _ in range(12):
        batch = source.next_batch()
        assert batch.history.shape[0] == batch.target.shape[0] == 4
        for index, example in enumerate(batch.examples):
            target = example.target
            assert all(
                span.document.identity == target.document.identity for span in example.history
            )
            assert all(span.end <= target.start for span in example.history)
            assert all(span.index < target.index for span in example.history)
            assert len(example.history) == batch.history.shape[1] <= 2
            labels = batch.target[index][batch.target[index] != 259].tolist()
            assert bytes(labels[:-1]) == target.source
            assert labels[-1] == example.terminal
            assert example.terminal == (257 if target.end == len(target.document.source) else 256)
            for position, span in enumerate(example.history):
                observed = batch.history[index, position]
                values = observed[observed != 259].tolist()
                assert bytes(values[:-1]) == span.source and values[-1] == 256
        assert batch.ordinal == source.cursor - 1


def test_model_rng_consumption_cannot_change_source_order() -> None:
    first, second = sampler(), sampler()
    for _ in range(10):
        a = first.next_batch()
        torch.randn(791)
        b = second.next_batch()
        assert a.identity == b.identity
        torch.testing.assert_close(a.history, b.history, atol=0, rtol=0)
        torch.testing.assert_close(a.target, b.target, atol=0, rtol=0)


def test_recoding_changes_observations_not_paired_source_selection() -> None:
    identity = sampler()
    codec = SentenceByteCodec(tuple(reversed(range(256))))
    recoded = sampler(codec)
    assert identity.identity != recoded.identity
    for _ in range(5):
        a, b = identity.next_batch(), recoded.next_batch()
        assert a.source_identity == b.source_identity
        assert a.examples == b.examples
        mask = a.target < 256
        torch.testing.assert_close(b.target[mask], 255 - a.target[mask])
        torch.testing.assert_close(b.target[~mask], a.target[~mask])


def test_resume_uses_bound_cursor_and_reproduces_next_batch() -> None:
    original = sampler()
    for _ in range(7):
        original.next_batch()
    restored = sampler()
    restored.restore(original.snapshot())
    assert original.next_batch().identity == restored.next_batch().identity
    invalid = original.snapshot() | {"identity": "wrong"}
    with pytest.raises(ValueError, match="identity"):
        restored.restore(invalid)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="cursor"):
            restored.restore(original.snapshot() | {"cursor": value})


def test_heldout_split_never_enters_training_selection() -> None:
    source = corpus()
    train_ids = {r.segmentation.document.identity for r in source.records if r.split == "train"}
    training = SentenceSampler(
        source, SamplingSpec("train", 4, 2, 42), SentenceByteCodec(), torch.device("cpu")
    )
    for _ in range(10):
        assert all(e.target.document.identity in train_ids for e in training.next_batch().examples)


def test_empty_split_and_forged_preparation_are_not_model_ready() -> None:
    prepared = corpus()
    with pytest.raises(ValueError, match="replay"):
        SentenceSampler(
            replace(prepared, records=()),
            SamplingSpec("train", 1, 2, 42),
            SentenceByteCodec(),
            torch.device("cpu"),
        )
    empty = prepare_documents([], PreparationSpec(1, 100, "tiny-v1"))
    with pytest.raises(ValueError, match="empty"):
        SentenceSampler(
            empty, SamplingSpec("train", 1, 2, 42), SentenceByteCodec(), torch.device("cpu")
        )


@pytest.mark.parametrize(
    "changes", [{"split": "other"}, {"batch_size": 0}, {"max_history": -1}, {"seed": True}]
)
def test_invalid_sampling_spec_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(SamplingSpec("train", 4, 2, 42), **changes)
