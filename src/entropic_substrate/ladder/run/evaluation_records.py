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
from entropic_substrate.core.sentence.scoring import COSINE_SLACK, importance_statistics
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.run.execution import batch_coordinates
from entropic_substrate.ladder.run.journal import read_json
from entropic_substrate.ladder.run.seals import CorpusMemo, SealMemo, read_seal

_V1 = "h-predictive-evaluation-v1"
_V2 = "h-predictive-evaluation-v2"
_ENVELOPE_V1 = "schema meaning run_identity step examples wall_seconds cuda_seconds checkpoint_seal"
# v2 adds ONE envelope key: the latent-usage operand the writer's injected implementation declares
_ENVELOPE_V2 = _ENVELOPE_V1 + " latent_operand"
_EXAMPLE_V1 = "source_batch source_examples source_bytes boundary_symbols by_sample_count log_weights decoder_log_prob prior_log_prob proposal_log_prob"
# v2 adds three keys; q stays at its v1 keys, so every reader of q resolves it unchanged
_EXAMPLE_V2 = _EXAMPLE_V1 + " proposals diagnostics latent_usage"
_PRIOR_BLOCK = "proposal by_sample_count log_weights decoder_log_prob prior_log_prob"
_MIXTURE_BLOCK = "proposal allocation component by_sample_count log_weights decoder_log_prob q_log_prob prior_log_prob"
_MIXTURE_ALLOCATION = "bernoulli_half"
_DIAGNOSTICS = "prior_log_radius_mean q_log_radius_mean prior_q_cosine log_weight_sd"


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


def _check_summaries(raw: object, weights: torch.Tensor, profile: HProfile) -> None:
    """Recompute ``importance_statistics`` for every k and require the recorded rows to match."""
    size = profile.schedule.batch_size
    summaries = recovery_record(raw, " ".join(str(k) for k in profile.schedule.evaluated_samples))
    for count in profile.schedule.evaluated_samples:
        values = summaries[str(count)]
        if type(values) is not list or len(values) != size:
            raise ValueError("evaluation summary population differs from samples")
        for actual, expected in zip(values, importance_statistics(weights[:count]), strict=True):
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


def _reals(value: object, size: int, *, what: str) -> list[float]:
    if type(value) is not list or len(value) != size:
        raise ValueError(f"evaluation {what} must list one number per row")
    return [_real(item) for item in value]


def _check_v2_example(
    recorded: dict[str, object], q_weights: torch.Tensor, profile: HProfile
) -> None:
    """Re-derive every v2 addition from its raw recorded terms; fail LOUD on any mismatch.

    Nothing derived is trusted from the file: the mixture density is recomputed from the two
    recorded component columns, every weight is reconciled with its terms, every k-prefix
    summary and log-weight sd is recomputed, and each non-derivable diagnostic is type-, range-
    and mode-checked. The admission of ``-inf`` per column is exactly what the math admits.
    """
    count, size = profile.schedule.evaluated_samples[-1], profile.schedule.batch_size
    blocks = recovery_record(recorded["proposals"], "prior mixture")

    prior_block = recovery_record(blocks["prior"], _PRIOR_BLOCK)
    if prior_block["proposal"] != "prior":
        raise ValueError("evaluation prior block is not named for the prior proposal")
    prior_decoder = _matrix(prior_block["decoder_log_prob"], count, size, zero_density=False)
    prior_density = _matrix(prior_block["prior_log_prob"], count, size, zero_density=False)
    prior_weights = _matrix(prior_block["log_weights"], count, size, zero_density=False)
    if bool((prior_decoder > 0).any()) or not torch.allclose(
        prior_weights, prior_decoder + prior_density - prior_density, rtol=1e-12, atol=1e-10
    ):
        raise ValueError("evaluation prior importance terms do not reconcile")
    _check_summaries(prior_block["by_sample_count"], prior_weights, profile)

    mixture_block = recovery_record(blocks["mixture"], _MIXTURE_BLOCK)
    if mixture_block["proposal"] != "mixture" or mixture_block["allocation"] != _MIXTURE_ALLOCATION:
        raise ValueError("evaluation mixture block is not the named half-and-half mixture")
    component = mixture_block["component"]
    if (
        type(component) is not list
        or len(component) != count
        or any(
            type(row) is not list
            or len(row) != size
            or any(type(c) is not int or c not in (0, 1) for c in row)
            for row in component
        )
    ):
        raise ValueError("evaluation mixture component assignment is malformed")
    from_prior = torch.tensor(component, dtype=torch.bool)
    mix_decoder = _matrix(mixture_block["decoder_log_prob"], count, size, zero_density=False)
    mix_q = _matrix(mixture_block["q_log_prob"], count, size, zero_density=True)
    mix_prior = _matrix(mixture_block["prior_log_prob"], count, size, zero_density=True)
    mix_weights = _matrix(mixture_block["log_weights"], count, size, zero_density=True)
    producing = torch.where(from_prior, mix_prior, mix_q)
    density = torch.logsumexp(torch.stack([mix_q, mix_prior]), dim=0) - math.log(2)
    if (
        not bool(torch.isfinite(producing).all())
        or not bool(torch.isfinite(density).all())
        or bool((mix_decoder > 0).any())
        or not torch.allclose(
            mix_weights, mix_decoder + mix_prior - density, rtol=1e-12, atol=1e-10
        )
    ):
        raise ValueError("evaluation mixture importance terms do not reconcile")
    _check_summaries(mixture_block["by_sample_count"], mix_weights, profile)

    diagnostics = recovery_record(recorded["diagnostics"], _DIAGNOSTICS)
    for name in ("prior_log_radius_mean", "q_log_radius_mean"):
        if profile.law.mode == "magnitude":
            _reals(diagnostics[name], size, what=name)
        elif diagnostics[name] is not None:
            raise ValueError(f"evaluation {name} must be absent in a sphere-law cell")
    cosines = _reals(diagnostics["prior_q_cosine"], size, what="prior_q_cosine")
    if any(abs(c) > COSINE_SLACK for c in cosines):
        raise ValueError("evaluation cosine of two unit directions is outside [-1, 1]")
    spreads = recovery_record(diagnostics["log_weight_sd"], "q prior mixture")
    for name, weights in (("q", q_weights), ("prior", prior_weights), ("mixture", mix_weights)):
        by_k = recovery_record(
            spreads[name], " ".join(str(k) for k in profile.schedule.evaluated_samples)
        )
        for k in profile.schedule.evaluated_samples:
            entries = by_k[str(k)]
            if type(entries) is not list or len(entries) != size:
                raise ValueError("evaluation log-weight sd must list one entry per row")
            for entry, column in zip(entries, weights[:k].unbind(dim=1), strict=True):
                if not bool(torch.isfinite(column).all()):
                    if entry is not None:
                        raise ValueError("evaluation log-weight sd exists for a zero-density row")
                elif entry is None or not math.isclose(
                    _real(entry), float(column.std(unbiased=True)), rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError("evaluation log-weight sd differs from raw importance samples")

    usage = recovery_record(recorded["latent_usage"], "q_decoder_log_prob null_decoder_log_prob")
    for name in ("q_decoder_log_prob", "null_decoder_log_prob"):
        if any(v > 0 for v in _reals(usage[name], size, what=name)):
            raise ValueError("evaluation latent-usage log-probability must not be positive")


def read_evaluation(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_bytes: int,
    max_source_bytes: int,
    memo: SealMemo | None = None,
    corpora: CorpusMemo | None = None,
) -> dict[str, object]:
    """Recompute weights/ESS/MCSE and source positions from immutable artifacts.

    ``corpora`` (FOR-1693, DH-41) is the process's :class:`CorpusMemo` of ``source_root``: the
    memo ARGUMENT only. With it the prepared corpus is replayed once instead of once per call, and
    the sampler below receives the proof instead of replaying a second time.

    ``memo``, when given, serves two reuses (design-gate ruling DG-B). RETRACTION
    (house rule 4, FOR-1670): this docstring used to say the memo is "passed
    straight through to :func:`read_seal`" and that only the seal is reused; the
    evaluation record itself was re-read and fully replayed on every call. Now a
    record whose content digest the memo already remembers is returned as is
    (the remembered object, read-only for callers) with NO filesystem access and
    no seal, corpus or sampler work, and ``memo.evaluations_memo_hits`` advances.
    Otherwise the full verification below runs; only after it returns does
    ``memo.evaluations_read`` advance and the row get remembered, so a refused
    record is neither counted nor remembered. TRUST BOUNDARY: a hit does not
    detect a file rewritten after admission under the same identity; a fresh
    process (``memo=None``) or a fresh memo still reads and replays everything as
    the admission predicate.
    """
    if memo is not None:
        remembered = memo.get_evaluation(artifact)
        if remembered is not None:
            memo.evaluations_memo_hits += 1
            return remembered
    document = read_json(root, artifact, max_bytes=max_bytes)
    v2 = type(document) is dict and document.get("schema") == _V2
    row = recovery_record(document, _ENVELOPE_V2 if v2 else _ENVELOPE_V1)
    if (
        row["schema"] not in (_V1, _V2)
        or row["meaning"] != "importance_weighted_log_marginal_estimate"
        or row["run_identity"] != profile.identity
    ):
        raise ValueError("evaluation construction or scoring meaning mismatch")
    if v2 and row["latent_operand"] != "posterior_mean":
        raise ValueError(
            f"evaluation latent operand {row['latent_operand']!r} is not readable: only "
            "'posterior_mean' re-derives from recorded rows ('draw_average' is FOR-1721's door)"
        )
    seal = read_seal(
        root,
        artifact_from_mapping(row["checkpoint_seal"]),
        profile,
        source_root,
        max_bytes=max_bytes,
        max_source_bytes=max_source_bytes,
        memo=memo,
        corpora=corpora,
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
    expected = ContentHash("sha256", profile.source_digest)
    corpus = (
        read_prepared_corpus(source_root, expected, max_bytes=max_source_bytes)
        if corpora is None
        else corpora.read(source_root, expected, max_bytes=max_source_bytes)
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
        recorded = recovery_record(value, _EXAMPLE_V2 if v2 else _EXAMPLE_V1)
        source = sampler.next_batch()
        if recorded["source_batch"] != source.identity or json.dumps(
            recorded["source_examples"], sort_keys=True
        ) != json.dumps(batch_coordinates(source), sort_keys=True):
            raise ValueError("evaluation held-out source positions differ from replay")
        expected_bytes = [len(e.target.source) for e in source.examples]
        for name, expected in (
            ("source_bytes", expected_bytes),
            ("boundary_symbols", [1] * len(source.examples)),
        ):
            if json.dumps(recorded[name]) != json.dumps(expected):
                raise ValueError("evaluation byte or boundary counts differ from source")
        count, size = profile.schedule.evaluated_samples[-1], profile.schedule.batch_size
        weights = _matrix(recorded["log_weights"], count, size, zero_density=True)
        decoder = _matrix(recorded["decoder_log_prob"], count, size, zero_density=False)
        prior = _matrix(recorded["prior_log_prob"], count, size, zero_density=True)
        proposal = _matrix(recorded["proposal_log_prob"], count, size, zero_density=False)
        if bool((decoder > 0).any()) or not torch.allclose(
            weights, decoder + prior - proposal, rtol=1e-12, atol=1e-10
        ):
            raise ValueError("evaluation importance terms do not reconcile")
        _check_summaries(recorded["by_sample_count"], weights, profile)
        if v2:
            _check_v2_example(recorded, weights, profile)
    if memo is not None:
        memo.evaluations_read += 1
        memo.remember_evaluation(artifact, row)
    return row
