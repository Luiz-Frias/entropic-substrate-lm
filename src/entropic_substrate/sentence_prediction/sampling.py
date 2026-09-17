"""Counter-seeded sentence batches with exact document/source-span custody."""

from dataclasses import dataclass

import torch

from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.corpus import (
    CorpusSplit,
    PreparedCorpus,
    prepare_documents,
)
from entropic_substrate.sentence_prediction.records import SentenceSpan, content_identity


@dataclass(frozen=True, slots=True)
class SamplingSpec:
    split: CorpusSplit
    batch_size: int
    max_history: int
    seed: int

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("unknown sentence source split")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch size must be positive")
        if type(self.max_history) is not int or not 0 <= self.max_history <= 64:
            raise ValueError("history must be an integer between 0 and 64")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("source seed must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SentenceExample:
    history: tuple[SentenceSpan, ...]
    target: SentenceSpan
    terminal: int

    @property
    def identity(self) -> str:
        return content_identity(
            (
                self.target.document.identity,
                str(self.target.start),
                str(self.target.end),
                str(self.target.index),
                str(self.terminal),
                str(tuple((s.start, s.end) for s in self.history)),
            )
        )


@dataclass(frozen=True)
class SentenceBatch:
    history: torch.Tensor
    target: torch.Tensor
    examples: tuple[SentenceExample, ...]
    ordinal: int
    source_identity: str
    identity: str


class SentenceSampler:
    """Uniform target-sentence marginals, batching equal history lengths.

    Choose a history-length bucket with probability proportional to its target
    count, then draw the whole batch uniformly from that bucket with replacement.
    Rows are not independent model replicates. Context consists of at most the
    most recent declared number of complete spans, never cross-document bytes.
    Source draws use a local CPU generator seeded from configuration and cursor;
    model/global RNG consumption and byte permutations cannot change pairing.
    """

    def __init__(
        self,
        corpus: PreparedCorpus,
        spec: SamplingSpec,
        codec: SentenceByteCodec,
        device: torch.device,
    ) -> None:
        if type(corpus) is not PreparedCorpus or type(spec) is not SamplingSpec:
            raise TypeError("sentence sampler requires prepared source and explicit sampling spec")
        if prepare_documents(corpus.candidates, corpus.spec) != corpus:
            raise ValueError("prepared source fails independent replay")
        examples: list[SentenceExample] = []
        for record in corpus.records:
            if record.split != spec.split:
                continue
            spans = record.segmentation.spans
            for index, span in enumerate(spans):
                terminal = codec.document_end if index == len(spans) - 1 else codec.sentence_end
                examples.append(
                    SentenceExample(spans[max(0, index - spec.max_history) : index], span, terminal)
                )
        if not examples:
            raise ValueError("selected sentence source split is empty")
        self.examples = tuple(examples)
        self.buckets = {
            length: tuple(e for e in self.examples if len(e.history) == length)
            for length in {len(e.history) for e in self.examples}
        }
        self.spec, self.codec, self.device = spec, codec, device
        self.order_identity = content_identity(
            (
                "h-uniform-target-length-buckets-v1",
                corpus.identity,
                spec.split,
                str(spec.batch_size),
                str(spec.max_history),
                str(spec.seed),
            )
        )
        self.identity = content_identity((self.order_identity, codec.identity))
        self.cursor = 0

    def snapshot(self) -> dict[str, object]:
        """Checkpoint cursor, separately from latent/device RNG streams."""
        return {"schema": "h-source-cursor-v1", "identity": self.identity, "cursor": self.cursor}

    def restore(self, value: object) -> None:
        if type(value) is not dict or set(value) != {"schema", "identity", "cursor"}:
            raise ValueError("invalid source cursor fields")
        if value["schema"] != "h-source-cursor-v1" or value["identity"] != self.identity:
            raise ValueError("source cursor schema or identity mismatch")
        cursor = value["cursor"]
        if type(cursor) is not int or cursor < 0:
            raise ValueError("source cursor must be a nonnegative integer")
        self.cursor = cursor

    def next_batch(self) -> SentenceBatch:
        seed = int(content_identity((self.order_identity, str(self.cursor)))[:16], 16) % (2**63)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        selector = int(torch.randint(len(self.examples), (1,), generator=generator).item())
        bucket = self.buckets[len(self.examples[selector].history)]
        indices = torch.randint(len(bucket), (self.spec.batch_size,), generator=generator).tolist()
        examples = tuple(bucket[index] for index in indices)
        history_length = len(examples[0].history)
        history_width = max((len(s.source) + 1 for e in examples for s in e.history), default=1)
        target_width = max(len(e.target.source) + 1 for e in examples)
        history = torch.full(
            (self.spec.batch_size, history_length, history_width), self.codec.pad, dtype=torch.int64
        )
        target = torch.full((self.spec.batch_size, target_width), self.codec.pad, dtype=torch.int64)
        for row, example in enumerate(examples):
            for column, span in enumerate(example.history):
                values = [
                    *(self.codec.permutation[b] for b in span.source),
                    self.codec.sentence_end,
                ]
                history[row, column, : len(values)] = torch.tensor(values)
            values = [*(self.codec.permutation[b] for b in example.target.source), example.terminal]
            target[row, : len(values)] = torch.tensor(values)
        source_identity = content_identity(
            (self.order_identity, str(self.cursor), *(e.identity for e in examples))
        )
        result = SentenceBatch(
            history.to(self.device),
            target.to(self.device),
            examples,
            self.cursor,
            source_identity,
            content_identity((source_identity, self.codec.identity)),
        )
        self.cursor += 1
        return result
