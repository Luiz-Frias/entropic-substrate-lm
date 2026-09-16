"""Bounded stratified selection of causal sentence-event states and future bytes."""

from dataclasses import dataclass
from random import Random

import torch

from entropic_substrate.evidence.lifecycle import ContentHash
from entropic_substrate.sentence_prediction.capture import read_prepared_corpus
from entropic_substrate.sentence_prediction.corpus import CorpusSplit
from entropic_substrate.sentence_prediction.execution import SentenceExecution


@dataclass(frozen=True, slots=True)
class SpanProbePosition:
    document_id: str
    window_start: int
    event_index: int
    span_start: int
    span_end: int
    target_position: int
    lag: int


@dataclass(frozen=True)
class SpanStateSet:
    checkpoint_id: str
    codec_id: str
    source_digest: str
    split: CorpusSplit
    positions: tuple[SpanProbePosition, ...]
    before: torch.Tensor
    after: torch.Tensor
    before_horizons: torch.Tensor
    after_horizons: torch.Tensor
    labels: torch.Tensor
    selection_seed: int
    selection_policy: str = "stratified-reservoir-by-lag-v1"


@dataclass(frozen=True, slots=True)
class _Candidate:
    document: int
    event: int
    window: int
    lag: int


@torch.no_grad()
def collect_span_states(
    execution: SentenceExecution,
    checkpoint_id: str,
    *,
    split: CorpusSplit,
    count: int,
    lags: tuple[int, ...],
    seed: int,
) -> SpanStateSet:
    """Use only complete past/current spans; labels never enter the model forward.

    Equal lag strata use independent bounded reservoirs, then deterministic
    document/event ordering. Window resets are explicit; no state update spans
    two windows or documents. The caller must bind checkpoint_id to loaded
    checkpoint bytes before treating this collection as captured evidence.
    """
    if type(checkpoint_id) is not str or not checkpoint_id.strip():
        raise ValueError("span collection requires a checkpoint identity")
    if (
        split not in ("train", "validation", "test")
        or type(count) is not int
        or type(seed) is not int
        or not 0 <= seed < 2**63
    ):
        raise ValueError("invalid span selection split, positions or seed")
    if (
        type(lags) is not tuple
        or not lags
        or any(type(lag) is not int or lag < 2 for lag in lags)
        or tuple(sorted(set(lags))) != lags
        or count < len(lags)
    ):
        raise ValueError("span positions must cover all distinct future lags")
    if execution.phase != "prediction" or any(
        p.requires_grad for p in execution.model.codec.encoder.parameters()
    ):
        raise ValueError("span collection requires the frozen prediction construction")
    corpus = read_prepared_corpus(
        execution.source_root,
        ContentHash("sha256", execution.profile.source_digest),
        max_bytes=execution.max_source_bytes,
    )
    quotas = [count // len(lags) + int(index < count % len(lags)) for index in range(len(lags))]
    reservoirs: list[list[_Candidate]] = [[] for _ in lags]
    # Reproducible statistical reservoirs, not security-sensitive randomness.
    generators = [Random(seed + index) for index in range(len(lags))]  # nosec B311
    seen = [0] * len(lags)
    window_size = execution.profile.spine.context_sentences
    for document_index, record in enumerate(corpus.records):
        if record.split != split:
            continue
        for event_index, span in enumerate(record.segmentation.spans):
            for index, lag in enumerate(lags):
                if span.end - 1 + lag >= len(span.document.source):
                    continue
                seen[index] += 1
                candidate = _Candidate(
                    document_index, event_index, event_index // window_size * window_size, lag
                )
                if len(reservoirs[index]) < quotas[index]:
                    reservoirs[index].append(candidate)
                else:
                    slot = generators[index].randrange(seen[index])
                    if slot < quotas[index]:
                        reservoirs[index][slot] = candidate
    if any(len(rows) != quota for rows, quota in zip(reservoirs, quotas, strict=True)):
        raise ValueError(
            f"insufficient causal span positions by lag: {dict(zip(lags, seen, strict=True))}"
        )
    selected = sorted(
        (item for rows in reservoirs for item in rows), key=lambda c: (c.document, c.event, c.lag)
    )
    width = execution.profile.spine.width
    before, after = torch.empty(count, width), torch.empty(count, width)
    positions: list[SpanProbePosition] = []
    labels, before_horizons, after_horizons = [], [], []
    groups: dict[tuple[int, int], list[tuple[int, _Candidate]]] = {}
    for index, candidate in enumerate(selected):
        groups.setdefault((candidate.document, candidate.window), []).append((index, candidate))
        spans = corpus.records[candidate.document].segmentation.spans
        span = spans[candidate.event]
        target = span.end - 1 + candidate.lag
        positions.append(
            SpanProbePosition(
                span.document.identity,
                spans[candidate.window].start,
                candidate.event,
                span.start,
                span.end,
                target,
                candidate.lag,
            )
        )
        labels.append(span.document.source[target])
        before_horizons.append(target - span.start + 1)
        after_horizons.append(candidate.lag)
    training = execution.model.training
    try:
        execution.model.eval()
        with torch.random.fork_rng(devices=[0] if execution.device.type == "cuda" else []):
            for (document, start), rows in groups.items():
                end = max(candidate.event for _, candidate in rows) + 1
                spans = corpus.records[document].segmentation.spans[start:end]
                length = max(len(span.source) + 1 for span in spans)
                tokens = torch.full(
                    (1, len(spans), length),
                    execution.codec.pad,
                    dtype=torch.long,
                    device=execution.device,
                )
                for index, span in enumerate(spans):
                    values = [
                        *(execution.codec.permutation[b] for b in span.source),
                        execution.codec.sentence_end,
                    ]
                    tokens[0, index, : len(values)] = torch.tensor(values, device=execution.device)
                with torch.autocast(
                    execution.device.type,
                    dtype=torch.bfloat16,
                    enabled=execution.profile.precision == "bf16",
                ):
                    states = execution.model.predict(tokens).states[0].detach().cpu().float()
                for destination, candidate in rows:
                    local = candidate.event - start
                    before[destination], after[destination] = states[local], states[local + 1]
    finally:
        execution.model.train(training)
    return SpanStateSet(
        checkpoint_id,
        execution.codec.identity,
        execution.profile.source_digest,
        split,
        tuple(positions),
        before,
        after,
        torch.tensor(before_horizons),
        torch.tensor(after_horizons),
        torch.tensor(labels),
        seed,
    )
