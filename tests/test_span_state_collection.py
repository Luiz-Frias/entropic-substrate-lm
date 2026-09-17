"""Bounded source selection and causal sentence-state collection for frozen probes."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tests.test_sentence_profile import profile

from entropic_substrate.sentence_prediction.capture import write_prepared_corpus
from entropic_substrate.sentence_prediction.corpus import (
    PreparationSpec,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.sentence_prediction.execution import SentenceExecution
from entropic_substrate.sentence_prediction.records import SourceDocument
from entropic_substrate.sentence_prediction.segmentation import SegmentationPolicy
from entropic_substrate.sentence_prediction.span_states import collect_span_states

pytestmark = pytest.mark.unit


def model(root: Path) -> SentenceExecution:
    source = root / "source"
    source.mkdir()
    docs = [
        SourceDocument(
            f"span-{i}",
            f"Record {i} stayed here. " + "Another complete sentence follows. " * 7,
            "span-fixture-v1",
        )
        for i in range(30)
    ]
    prepared = prepare_documents(
        docs,
        PreparationSpec(
            30,
            100000,
            "span-fixture-v1",
            SegmentationPolicy(64),
            SplitPolicy(validation_buckets=2000, test_buckets=1000),
        ),
    )
    artifact = write_prepared_corpus(source, prepared, max_bytes=1_000_000)
    run = SentenceExecution(
        replace(profile(), source_digest=artifact.content_hash.digest),
        source,
        max_source_bytes=1_000_000,
    )
    run.advance()
    run.advance()
    run.begin_prediction()
    return run


def test_collector_retains_all_lags_and_exact_causal_coordinates(tmp_path: Path) -> None:
    run = model(tmp_path)
    before_rng = torch.get_rng_state()
    result = collect_span_states(
        run, "checkpoint-fixture", split="train", count=8, lags=(2, 4, 16, 64), seed=42
    )
    assert result.before.shape == result.after.shape == (8, 8)
    assert not result.before.requires_grad and not result.after.requires_grad
    assert {position.lag for position in result.positions} == {2, 4, 16, 64}
    docs = {r.segmentation.document.identity: r.segmentation.document for r in run.corpus.records}
    for index, position in enumerate(result.positions):
        assert position.window_start <= position.span_start < position.span_end
        assert position.target_position == position.span_end - 1 + position.lag
        assert (
            result.labels[index].item()
            == docs[position.document_id].source[position.target_position]
        )
        assert (
            result.before_horizons[index] - result.after_horizons[index]
            == position.span_end - position.span_start
        )
    torch.testing.assert_close(torch.get_rng_state(), before_rng, atol=0, rtol=0)
    assert run.model.training


def test_collection_is_deterministic_and_split_bound(tmp_path: Path) -> None:
    run = model(tmp_path)
    first = collect_span_states(
        run, "checkpoint", split="validation", count=8, lags=(2, 4, 16, 64), seed=42
    )
    torch.randn(99)
    second = collect_span_states(
        run, "checkpoint", split="validation", count=8, lags=(2, 4, 16, 64), seed=42
    )
    assert first.positions == second.positions
    torch.testing.assert_close(first.before, second.before, atol=0, rtol=0)
    valid = {
        r.segmentation.document.identity for r in run.corpus.records if r.split == "validation"
    }
    assert all(position.document_id in valid for position in first.positions)


def test_collector_states_match_direct_causal_prefixes(tmp_path: Path) -> None:
    run = model(tmp_path)
    result = collect_span_states(
        run, "checkpoint", split="train", count=4, lags=(2, 4, 16, 64), seed=42
    )
    records = {r.segmentation.document.identity: r.segmentation for r in run.corpus.records}
    run.model.eval()
    with torch.no_grad():
        for index, position in enumerate(result.positions):
            spans = records[position.document_id].spans
            prefix = [s for s in spans if position.window_start <= s.start < position.span_start]
            width = max((len(s.source) + 1 for s in prefix), default=1)
            tokens = torch.full((1, len(prefix), width), 259, dtype=torch.long)
            for j, span in enumerate(prefix):
                encoded = [*(run.codec.permutation[b] for b in span.source), 256]
                tokens[0, j, : len(encoded)] = torch.tensor(encoded)
            expected = run.model.predict(tokens).states[0, -1]
            torch.testing.assert_close(result.before[index], expected, atol=1e-5, rtol=1e-5)


def test_short_or_invalid_probe_population_never_silently_drops_lags(tmp_path: Path) -> None:
    run = model(tmp_path)
    with pytest.raises(ValueError, match="positions"):
        collect_span_states(
            run, "checkpoint", split="validation", count=100000, lags=(2, 4, 16, 64), seed=42
        )
    with pytest.raises(ValueError):
        collect_span_states(run, "checkpoint", split="train", count=1, lags=(2, 4, 16, 64), seed=42)
