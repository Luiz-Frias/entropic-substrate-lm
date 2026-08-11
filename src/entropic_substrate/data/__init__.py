"""Datasets, tokenisation and batch sampling."""

from __future__ import annotations

from entropic_substrate.data.batching import (
    Corpus,
    CorpusMeta,
    SequenceSampler,
    TokenSplit,
)
from entropic_substrate.data.build import build_corpus, corpus_byte_stream
from entropic_substrate.data.enwik8 import (
    ENWIK8_URL,
    download_enwik8,
    load_enwik8_bytes,
)
from entropic_substrate.data.fineweb import fineweb_text, stream_fineweb_bytes
from entropic_substrate.data.synthetic import synthetic_text

__all__ = [
    "ENWIK8_URL",
    "Corpus",
    "CorpusMeta",
    "SequenceSampler",
    "TokenSplit",
    "build_corpus",
    "corpus_byte_stream",
    "download_enwik8",
    "fineweb_text",
    "load_enwik8_bytes",
    "stream_fineweb_bytes",
    "synthetic_text",
]
