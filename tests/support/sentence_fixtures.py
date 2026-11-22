"""Tiny sentence profile, synthetic corpus and execution engine shared by the sentence tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from entropic_substrate.core.latent.dynamics import SolverConfig
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.corpus import (
    PreparationSpec,
    PreparedCorpus,
    SplitPolicy,
    prepare_documents,
)
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.profile import HProfile, OptimizationSpec, RunSchedule
from entropic_substrate.core.sentence.records import SourceDocument, content_identity
from entropic_substrate.core.sentence.segmentation import SegmentationPolicy
from entropic_substrate.core.sentence.spine import SpineSpec
from entropic_substrate.ladder.capture.capture import write_prepared_corpus
from entropic_substrate.ladder.run.execution import SentenceExecution


def profile() -> HProfile:
    return HProfile(
        name="h0-fisher-tiny",
        stage="H0",
        run_class="local_tiny",
        source_digest=content_identity(("source-fixture",)),
        seed=42,
        source_seed=42,
        device="cpu",
        precision="fp32",
        codec=CodecSpec(8, 1, 8, 1, 4, 64),
        spine=SpineSpec(8, 2, 4, 4, "magnitude"),
        law=LawSpec(8, "magnitude"),
        consumer="fisher",
        solver=SolverConfig(),
        optimization=OptimizationSpec(0.001, 0.01, (0.9, 0.95), 1e-8, 1.0, 0.3),
        schedule=RunSchedule(2, 4, 2, 2, 2, 1, (2, 4), 1, 2, 2, diagnostic_samples=8),
        permutation=tuple(range(256)),
    )


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
