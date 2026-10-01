"""Independent admission of importance scores; no reconstruction/likelihood conflation."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler
from entropic_substrate.core.sentence.scoring import importance_statistics
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.run.execution import batch_coordinates
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.seals import SealMemo, read_seal


def _real(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError("evaluation number must be finite and not boolean")
    return float(cast(float, value))


def _matrix(value: object, samples: int, batch: int, *, zero_density: bool) -> torch.Tensor:
    if type(value) is not list or len(value) != samples:
        raise ValueError("evaluation sample count differs from its envelope")
    rows: list[list[float]] = []
    for row in value:
        if type(row) is not list or len(row) != batch:
            raise ValueError("evaluation batch count differs from its envelope")
        rows.append([-math.inf if zero_density and item == "-inf" else _real(item) for item in row])
    return torch.tensor(rows, dtype=torch.float64)


def read_evaluation(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
    memo: SealMemo | None = None,
) -> dict[str, object]:
    """Recompute weights/ESS/MCSE and source positions from immutable artifacts.

    ``memo``, when given, is passed straight through to :func:`read_seal`'s
    lookup of the evaluation's own checkpoint seal (design-gate ruling DG-B):
    the already-verified parent seal may be reused in-process, but a fresh
    process (``memo=None``) still walks the full chain as the admission
    predicate.
    """
    row = recovery_record(
        read_json(root, artifact, max_bytes=max_bytes),
        "schema meaning run_identity step examples wall_seconds cuda_seconds checkpoint_seal",
    )
    if (
        row["schema"] != "h-predictive-evaluation-v1"
        or row["meaning"] != "importance_weighted_log_marginal_estimate"
        or row["run_identity"] != profile.identity
    ):
        raise ValueError("evaluation construction or scoring meaning mismatch")
    seal = read_seal(
        root,
        artifact_from_mapping(row["checkpoint_seal"]),
        profile,
        source_root,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        memo=memo,
    )
    if seal.phase != "prediction" or type(row["step"]) is not int or row["step"] != seal.step:
        raise ValueError("evaluation is not bound to its exact prediction checkpoint")
    if _real(row["wall_seconds"]) < 0:
        raise ValueError("evaluation wall duration must be nonnegative")
    if profile.device == "cpu":
        if row["cuda_seconds"] is not None:
            raise ValueError("CPU evaluation cannot claim CUDA time")
    elif _real(row["cuda_seconds"]) < 0:
        raise ValueError("evaluation CUDA duration must be nonnegative")
    examples = row["examples"]
    if type(examples) is not list or len(examples) != profile.schedule.evaluation_batches:
        raise ValueError("evaluation does not cover its declared batch population")
    corpus = read_prepared_corpus(
        source_root, ContentHash("sha256", profile.source_digest), max_bytes=max_source_bytes
    )
    sampler = SentenceSampler(
        corpus,
        SamplingSpec(
            "validation",
            profile.schedule.batch_size,
            profile.spine.context_sentences,
            profile.source_seed,
        ),
        SentenceByteCodec(profile.permutation),
        torch.device("cpu"),
    )
    for value in examples:
        recorded = recovery_record(
            value,
            "source_batch source_examples source_bytes boundary_symbols by_sample_count log_weights decoder_log_prob prior_log_prob proposal_log_prob",
        )
        source = sampler.next_batch()
        if recorded["source_batch"] != source.identity or json.dumps(
            recorded["source_examples"], sort_keys=True
        ) != json.dumps(batch_coordinates(source), sort_keys=True):
            raise ValueError("evaluation held-out source positions differ from replay")
        for name, expected in (
            ("source_bytes", [len(e.target.source) for e in source.examples]),
            ("boundary_symbols", [1] * len(source.examples)),
        ):
            if json.dumps(recorded[name]) != json.dumps(expected):
                raise ValueError("evaluation byte or boundary counts differ from source")
        count, size = profile.schedule.importance_samples[-1], profile.schedule.batch_size
        weights = _matrix(recorded["log_weights"], count, size, zero_density=True)
        decoder = _matrix(recorded["decoder_log_prob"], count, size, zero_density=False)
        prior = _matrix(recorded["prior_log_prob"], count, size, zero_density=True)
        proposal = _matrix(recorded["proposal_log_prob"], count, size, zero_density=False)
        if bool((decoder > 0).any()) or not torch.allclose(
            weights, decoder + prior - proposal, rtol=1e-12, atol=1e-10
        ):
            raise ValueError("evaluation importance terms do not reconcile")
        summaries = recovery_record(
            recorded["by_sample_count"],
            " ".join(str(k) for k in profile.schedule.importance_samples),
        )
        for count in profile.schedule.importance_samples:
            values = summaries[str(count)]
            if type(values) is not list or len(values) != size:
                raise ValueError("evaluation summary population differs from samples")
            for actual, expected in zip(
                values, importance_statistics(weights[:count]), strict=True
            ):
                fields = recovery_record(
                    actual, "log_estimate effective_sample_size likelihood_relative_mcse gap"
                )
                for name, expected_value in asdict(expected).items():
                    actual_value = fields[name]
                    if expected_value is None or type(expected_value) is str:
                        if actual_value != expected_value:
                            raise ValueError("evaluation gap differs from measured weight mass")
                    elif not math.isclose(
                        _real(actual_value), expected_value, rel_tol=1e-10, abs_tol=1e-10
                    ):
                        raise ValueError("evaluation summary differs from raw importance samples")
    return row
