"""Bounded, content-deduplicated document preparation before sentence training."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from typing import Literal

from entropic_substrate.sentence_prediction.records import SourceDocument, content_identity
from entropic_substrate.sentence_prediction.segmentation import (
    OversizeDocument,
    SegmentationPolicy,
    SentenceSegmentation,
    segment_document,
)

type CorpusSplit = Literal["train", "validation", "test"]
type RejectionReason = Literal["duplicate_content", "oversize_sentence", "source_budget"]
type StopReason = Literal["source_exhausted", "candidate_limit", "source_budget"]


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Content-hash partitions: aliases cannot acquire different split labels."""

    seed: int = 42
    validation_buckets: int = 500
    test_buckets: int = 500

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("split seed must be a nonnegative integer")
        if any(type(n) is not int or n < 1 for n in (self.validation_buckets, self.test_buckets)):
            raise ValueError("held-out split buckets must be positive integers")
        if self.validation_buckets + self.test_buckets >= 10000:
            raise ValueError("split policy must leave a nonempty training partition")

    @property
    def identity(self) -> str:
        return content_identity(
            (
                "sentence-split-v1",
                str(self.seed),
                str(self.validation_buckets),
                str(self.test_buckets),
            )
        )

    def assign(self, document: SourceDocument) -> CorpusSplit:
        """Assign before segmentation, without examining model outcomes."""
        if type(document) is not SourceDocument:
            raise TypeError("split assignment requires a source document")
        bucket = (
            int(
                content_identity(("sentence-split-v1", str(self.seed), document.content_digest))[
                    :16
                ],
                16,
            )
            % 10000
        )
        if bucket < self.validation_buckets:
            return "validation"
        if bucket < self.validation_buckets + self.test_buckets:
            return "test"
        return "train"


@dataclass(frozen=True, slots=True)
class PreparationSpec:
    """Limits bound candidates examined and bytes admitted, not network traffic."""

    candidate_limit: int
    admitted_byte_limit: int
    revision: str
    segmentation: SegmentationPolicy = field(default_factory=SegmentationPolicy)
    splits: SplitPolicy = field(default_factory=SplitPolicy)
    dataset_name: str | None = None

    def __post_init__(self) -> None:
        if any(
            type(n) is not int or n < 1 for n in (self.candidate_limit, self.admitted_byte_limit)
        ):
            raise ValueError("preparation limits must be positive integers")
        if type(self.revision) is not str or not self.revision.strip():
            raise ValueError("source revision must be explicit and nonblank")
        if self.dataset_name is not None and (
            type(self.dataset_name) is not str
            or not self.dataset_name.strip()
            or self.dataset_name != self.dataset_name.strip()
        ):
            raise ValueError("declared dataset name must be a canonical nonblank string")
        if (
            type(self.segmentation) is not SegmentationPolicy
            or type(self.splits) is not SplitPolicy
        ):
            raise TypeError("preparation requires declared segmentation and split policies")

    @property
    def identity(self) -> str:
        return content_identity(
            (
                "sentence-preparation-v2",
                str(self.candidate_limit),
                str(self.admitted_byte_limit),
                self.revision,
                self.segmentation.identity,
                self.splits.identity,
                *(
                    ("dataset-declared", self.dataset_name)
                    if self.dataset_name is not None
                    else ("dataset-not-declared",)
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class PreparedRecord:
    segmentation: SentenceSegmentation
    split: CorpusSplit


@dataclass(frozen=True, slots=True)
class DocumentRejection:
    """Retain the candidate for independent replay of the admission decision."""

    document: SourceDocument
    reason: RejectionReason
    sentence_index: int | None = None

    @property
    def source_id(self) -> str:
        return self.document.source_id


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """Preparation output only; this value grants no training/scientific authority.

    A persisted reader must replay preparation from the bound candidates, never
    trust independently supplied split, rejection, or count fields as evidence.
    """

    spec: PreparationSpec
    candidates: tuple[SourceDocument, ...]
    records: tuple[PreparedRecord, ...]
    rejections: tuple[DocumentRejection, ...]
    stopped_by: StopReason

    @property
    def examined_documents(self) -> int:
        return len(self.candidates)

    @property
    def admitted_bytes(self) -> int:
        return sum(len(row.segmentation.document.source) for row in self.records)

    @property
    def status(self) -> Literal["prepared", "empty"]:
        return "prepared" if self.records else "empty"

    @property
    def identity(self) -> str:
        return content_identity(
            (
                "prepared-sentence-corpus-v1",
                self.spec.identity,
                *(doc.identity for doc in self.candidates),
            )
        )


def prepare_documents(documents: Iterable[SourceDocument], spec: PreparationSpec) -> PreparedCorpus:
    """Take a bounded prefix, deduplicate, split, then segment complete documents.

    Stop at the first novel document exceeding remaining admitted-byte budget;
    do not search subsequent rows for shorter examples. Retain rejected source
    candidates and distinguish an empty preparation from model-ready data.
    """
    if type(spec) is not PreparationSpec:
        raise TypeError("document preparation requires an explicit spec")
    candidates: list[SourceDocument] = []
    records: list[PreparedRecord] = []
    rejections: list[DocumentRejection] = []
    source_ids: dict[str, str] = {}
    contents: set[str] = set()
    admitted = 0
    stopped: StopReason = "source_exhausted"
    for document in islice(documents, spec.candidate_limit):
        if type(document) is not SourceDocument:
            raise TypeError("candidate is not a SourceDocument")
        if document.revision != spec.revision:
            raise ValueError("candidate source revision differs from preparation spec")
        previous = source_ids.get(document.source_id)
        if previous is not None and previous != document.content_digest:
            raise ValueError("source ID was reused with different content")
        source_ids[document.source_id] = document.content_digest
        candidates.append(document)
        split = spec.splits.assign(document)
        if document.content_digest in contents:
            rejections.append(DocumentRejection(document, "duplicate_content"))
            continue
        contents.add(document.content_digest)
        if admitted + len(document.source) > spec.admitted_byte_limit:
            rejections.append(DocumentRejection(document, "source_budget"))
            stopped = "source_budget"
            break
        segmented = segment_document(document, spec.segmentation)
        if isinstance(segmented, OversizeDocument):
            rejections.append(
                DocumentRejection(document, "oversize_sentence", segmented.span.index)
            )
            continue
        records.append(PreparedRecord(segmented, split))
        admitted += len(document.source)
    else:
        if len(candidates) == spec.candidate_limit:
            stopped = "candidate_limit"
    return PreparedCorpus(spec, tuple(candidates), tuple(records), tuple(rejections), stopped)
