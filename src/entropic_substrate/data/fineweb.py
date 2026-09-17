"""FineWeb-Edu streaming loader (HuggingFace ``datasets``, streaming=True).

Following the holobyte reference: pull documents from the ``sample-10BT``
config, join them with ``<|endoftext|>`` and treat the result as a utf-8 byte
stream. Streaming avoids materialising the 10BT shard on disk, which matters
on ephemeral GPU boxes (RunPod / Vast).
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from contextlib import closing
from itertools import islice
from typing import cast

from entropic_substrate.sentence_prediction.records import SourceDocument

__all__ = [
    "FINEWEB_EOT",
    "FINEWEB_REPO",
    "FINEWEB_REVISION",
    "fineweb_text",
    "stream_fineweb_bytes",
    "stream_fineweb_documents",
]

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"  # pragma: allowlist secret
FINEWEB_EOT = "<|endoftext|>"


def stream_fineweb_documents(
    *, max_documents: int, name: str = "sample-10BT"
) -> Generator[SourceDocument, None, None]:
    """Bounded, provenance-preserving H-program input, without EOT text injection.

    The old flattened stream below remains unchanged. Missing provider IDs or
    non-text rows are defects, not fabricated IDs or silently coerced content.
    Callers own this generator and must close it when stopping early. Its scope
    closes the underlying dataset iterator; a bounded islice does not do that.
    """
    if type(max_documents) is not int or max_documents < 1:
        raise ValueError("document limit must be a positive integer")
    if type(name) is not str or not name.strip():
        raise ValueError("FineWeb dataset config must be nonblank")
    from datasets import load_dataset

    dataset = load_dataset(  # nosec B615 -- existing immutable revision is forwarded unchanged
        FINEWEB_REPO, name=name, split="train", streaming=True, revision=FINEWEB_REVISION
    )
    iterator = iter(dataset)
    if not isinstance(cast(object, iterator), Generator):
        raise TypeError("FineWeb streaming requires a closeable dataset generator")
    with closing(iterator):
        for item in islice(iterator, max_documents):
            if not isinstance(item, dict):
                raise ValueError("FineWeb document row must be a mapping")
            source_id, text = item.get("id"), item.get("text")
            if type(source_id) is not str or not source_id.strip() or type(text) is not str:
                raise ValueError("FineWeb document requires its original ID and text")
            yield SourceDocument(f"{FINEWEB_REPO}/{name}/{source_id}", text, FINEWEB_REVISION)


def _iter_documents(name: str, split: str) -> Iterator[str]:
    from datasets import load_dataset  # imported lazily: heavy + optional at test time

    dataset = load_dataset(  # nosec B615 -- immutable revision constant is forwarded below
        FINEWEB_REPO,
        name=name,
        split=split,
        streaming=True,
        revision=FINEWEB_REVISION,
    )
    for item in dataset:
        text = item.get("text") if isinstance(item, dict) else None
        if text:
            yield str(text)


def fineweb_text(
    num_chars: int = 20_000_000,
    *,
    name: str = "sample-10BT",
    split: str = "train",
) -> str:
    """Concatenate streamed documents until ``num_chars`` characters are held."""
    parts: list[str] = []
    total = 0
    for text in _iter_documents(name, split):
        chunk = text + FINEWEB_EOT
        parts.append(chunk)
        total += len(chunk)
        if total >= num_chars:
            break
    return "".join(parts)[:num_chars]


def stream_fineweb_bytes(
    max_bytes: int | None = None,
    *,
    name: str = "sample-10BT",
    split: str = "train",
) -> Iterator[bytes]:
    """Yield utf-8 encoded document chunks, capped at ``max_bytes`` total.

    Useful for :func:`entropic_substrate.init.fisher.count_transitions`, which
    consumes an iterable of byte chunks without buffering the whole corpus.
    """
    emitted = 0
    for text in _iter_documents(name, split):
        blob = (text + FINEWEB_EOT).encode("utf-8")
        if max_bytes is not None:
            remaining = max_bytes - emitted
            if remaining <= 0:
                return
            blob = blob[:remaining]
        emitted += len(blob)
        yield blob
        if max_bytes is not None and emitted >= max_bytes:
            return
