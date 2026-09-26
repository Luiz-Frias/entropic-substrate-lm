"""Turn a :class:`~entropic_substrate.config.DataConfig` into a :class:`Corpus`.

Key detail for the E0 comparison: ``bytes_per_token`` is **measured** on the
held-out split (total utf-8 bytes / number of tokens), never assumed to be 4.0.
That ratio is what converts a BPE cross-entropy in nats/token into nats/byte,
the only metric on which a byte model and a BPE model can be compared fairly.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from entropic_substrate.config import DataConfig
from entropic_substrate.data.batching import Corpus, CorpusMeta, TokenSplit
from entropic_substrate.data.enwik8 import load_enwik8_bytes
from entropic_substrate.data.fineweb import fineweb_text, stream_fineweb_bytes
from entropic_substrate.data.synthetic import synthetic_text

__all__ = ["build_corpus", "corpus_byte_stream", "gpt2_encoding"]


def gpt2_encoding():
    """Return the GPT-2 tiktoken encoding (lazy import; cached by tiktoken)."""
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def _raw_bytes(cfg: DataConfig) -> bytes:
    if cfg.source == "enwik8":
        arr = load_enwik8_bytes(cfg.path)
        return bytes(np.asarray(arr, dtype=np.uint8).tobytes())
    if cfg.source == "fineweb":
        return fineweb_text(cfg.fineweb_chars).encode("utf-8")
    if cfg.source == "synthetic":
        return synthetic_text(cfg.synthetic_bytes).encode("utf-8")
    msg = f"unknown data source {cfg.source!r}"
    raise ValueError(msg)


def corpus_byte_stream(cfg: DataConfig, max_bytes: int | None = None) -> Iterator[bytes]:
    """Yield byte chunks for corpus statistics (Fisher init) without full load."""
    if cfg.source == "fineweb":
        yield from stream_fineweb_bytes(max_bytes)
        return
    blob = _raw_bytes(cfg)
    if max_bytes is not None:
        blob = blob[:max_bytes]
    chunk = 1 << 20
    for start in range(0, len(blob), chunk):
        yield blob[start : start + chunk]


def build_corpus(cfg: DataConfig, tokenizer: str) -> Corpus:
    """Materialise train/val token streams for ``tokenizer`` in {byte, gpt2}."""
    blob = _raw_bytes(cfg)
    if len(blob) < 1024:
        msg = f"corpus too small ({len(blob)} bytes)"
        raise ValueError(msg)

    split_at = int(len(blob) * (1.0 - cfg.val_fraction))
    train_bytes, val_bytes = blob[:split_at], blob[split_at:]

    if tokenizer == "byte":
        train_tokens = np.frombuffer(train_bytes, dtype=np.uint8).astype(np.int32)
        val_tokens = np.frombuffer(val_bytes, dtype=np.uint8).astype(np.int32)
        vocab_size = 256
        bytes_per_token = 1.0
    elif tokenizer == "gpt2":
        enc = gpt2_encoding()
        train_ids = enc.encode_ordinary(train_bytes.decode("utf-8", errors="replace"))
        val_text = val_bytes.decode("utf-8", errors="replace")
        val_ids = enc.encode_ordinary(val_text)
        train_tokens = np.asarray(train_ids, dtype=np.int32)
        val_tokens = np.asarray(val_ids, dtype=np.int32)
        vocab_size = enc.n_vocab
        if len(val_ids) == 0:
            msg = "validation split produced zero GPT-2 tokens"
            raise ValueError(msg)
        # MEASURED, not assumed: nats/byte = nats/token / bytes_per_token.
        bytes_per_token = len(val_text.encode("utf-8")) / len(val_ids)
    else:
        msg = f"unknown tokenizer {tokenizer!r}"
        raise ValueError(msg)

    meta = CorpusMeta(
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        bytes_per_token=bytes_per_token,
        train_tokens=int(train_tokens.shape[0]),
        val_tokens=int(val_tokens.shape[0]),
        val_bytes=len(val_bytes),
        source=cfg.source,
    )
    return Corpus(
        train=TokenSplit(np.ascontiguousarray(train_tokens)),
        val=TokenSplit(np.ascontiguousarray(val_tokens)),
        meta=meta,
    )
