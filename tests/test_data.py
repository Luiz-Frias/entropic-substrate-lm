"""Data loading, splitting and batch-sampling tests (no network access)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from entropic_substrate.config import DataConfig
from entropic_substrate.data import enwik8, fineweb
from entropic_substrate.data.batching import SequenceSampler, TokenSplit
from entropic_substrate.data.build import build_corpus, corpus_byte_stream
from entropic_substrate.data.enwik8 import load_enwik8_bytes
from entropic_substrate.data.synthetic import synthetic_text

pytestmark = pytest.mark.unit


def test_synthetic_text_is_deterministic_and_sized() -> None:
    a = synthetic_text(5_000, seed=3)
    assert len(a) == 5_000
    assert a == synthetic_text(5_000, seed=3)
    assert a != synthetic_text(5_000, seed=4)


def test_build_corpus_byte_split_and_metadata() -> None:
    cfg = DataConfig(source="synthetic", synthetic_bytes=20_000, val_fraction=0.1)
    corpus = build_corpus(cfg, "byte")
    assert corpus.meta.tokenizer == "byte"
    assert corpus.meta.vocab_size == 256
    assert corpus.meta.bytes_per_token == 1.0
    assert corpus.meta.val_tokens == corpus.meta.val_bytes
    assert len(corpus.train) + len(corpus.val) == 20_000
    assert corpus.train.tokens.max() < 256


def test_build_corpus_measures_bytes_per_token_for_gpt2() -> None:
    cfg = DataConfig(source="synthetic", synthetic_bytes=40_000, val_fraction=0.1)
    corpus = build_corpus(cfg, "gpt2")
    meta = corpus.meta
    assert meta.tokenizer == "gpt2"
    assert meta.vocab_size == 50257
    # Measured, not assumed: exactly val_bytes / val_tokens, and > 1 byte/token.
    assert meta.bytes_per_token == pytest.approx(meta.val_bytes / meta.val_tokens, rel=1e-9)
    assert 1.0 < meta.bytes_per_token < 20.0
    assert meta.bytes_per_token != 4.0


def test_build_corpus_rejects_unknown_tokenizer_and_source() -> None:
    cfg = DataConfig(source="synthetic", synthetic_bytes=5_000)
    with pytest.raises(ValueError, match="unknown tokenizer"):
        build_corpus(cfg, "runes")
    with pytest.raises(ValueError, match="unknown data source"):
        build_corpus(DataConfig(source="mars"), "byte")  # type: ignore[arg-type]


def test_corpus_byte_stream_chunks_cover_the_prefix() -> None:
    cfg = DataConfig(source="synthetic", synthetic_bytes=30_000)
    joined = b"".join(corpus_byte_stream(cfg, max_bytes=1_500))
    assert len(joined) == 1_500
    assert joined == synthetic_text(30_000).encode("utf-8")[:1_500]


def test_load_enwik8_bytes_reports_a_helpful_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="download_enwik8"):
        load_enwik8_bytes(tmp_path / "nope")


def test_load_enwik8_bytes_reads_and_truncates(tmp_path) -> None:
    path = tmp_path / "enwik8"
    path.write_bytes(bytes(range(256)))
    assert len(load_enwik8_bytes(path)) == 256
    assert len(load_enwik8_bytes(path, max_bytes=10)) == 10


def test_enwik8_download_source_is_https_and_content_pinned(tmp_path) -> None:
    assert enwik8.ENWIK8_URL == "https://mattmahoney.net/dc/enwik8.zip"

    archive = tmp_path / "enwik8.zip"
    archive.write_bytes(b"tampered archive")
    with pytest.raises(RuntimeError, match="archive SHA-256"):
        enwik8.download_enwik8(tmp_path)


def test_fineweb_stream_pins_the_dataset_revision(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_dataset(repo: str, **kwargs):
        captured.update(repo=repo, **kwargs)
        return [{"text": "measured"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    assert next(fineweb.stream_fineweb_bytes()) == b"measured<|endoftext|>"
    assert captured["revision"] == fineweb.FINEWEB_REVISION


def _split(n: int = 1_000) -> TokenSplit:
    return TokenSplit(np.arange(n, dtype=np.int32) % 251)


def test_sampler_shapes_and_next_token_alignment() -> None:
    sampler = SequenceSampler(_split(), batch_size=3, seq_len=16, seed=0)
    x, y = sampler.batch()
    assert x.shape == (3, 16) and y.shape == (3, 16)
    assert x.dtype == torch.int64
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_sampler_rejects_short_splits() -> None:
    with pytest.raises(ValueError, match="at least"):
        SequenceSampler(_split(10), batch_size=2, seq_len=32)


def test_train_sampler_is_seed_reproducible() -> None:
    a = SequenceSampler(_split(), 4, 16, train=True, seed=7).batch()[0]
    b = SequenceSampler(_split(), 4, 16, train=True, seed=7).batch()[0]
    c = SequenceSampler(_split(), 4, 16, train=True, seed=8).batch()[0]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_eval_sampler_is_deterministic_after_reset() -> None:
    sampler = SequenceSampler(_split(), 2, 16, train=False, seed=0)
    first = [x.clone() for x, _ in sampler.iter_batches(3)]
    sampler.reset_eval()
    second = [x.clone() for x, _ in sampler.iter_batches(3)]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_sampler_state_roundtrip_restores_the_stream() -> None:
    sampler = SequenceSampler(_split(), 2, 16, train=True, seed=0)
    sampler.batch()
    state = sampler.state_dict()
    expected = sampler.batch()[0]

    restored = SequenceSampler(_split(), 2, 16, train=True, seed=99)
    restored.load_state_dict(state)
    assert torch.equal(restored.batch()[0], expected)


# ------------------------------------------------------- data-order digest
# Pairing evidence for the E1 preregistration (docs/e1-preregistration.md):
# "same seed ⇒ same data order" must be RECORDED per run, not asserted about
# the code. The sampler attests its first 32 train batches' source offsets.


def test_train_sampler_data_order_digest_is_deterministic_and_stable() -> None:
    a = SequenceSampler(_split(), 4, 16, train=True, seed=7)
    b = SequenceSampler(_split(), 4, 16, train=True, seed=7)
    for _ in range(32):
        a.batch()
        b.batch()
    assert a.data_order_digest() == b.data_order_digest()
    digest, count = a.data_order_digest()
    assert count == 32
    a.batch()  # batch 33 must NOT change the attestation window
    assert a.data_order_digest() == (digest, 32)

    c = SequenceSampler(_split(), 4, 16, train=True, seed=8)
    for _ in range(32):
        c.batch()
    assert c.data_order_digest()[0] != digest


def test_train_sampler_digest_reports_partial_batch_counts() -> None:
    """A run shorter than the window attests what it actually consumed."""
    sampler = SequenceSampler(_split(), 4, 16, train=True, seed=7)
    assert sampler.data_order_digest() is None
    for _ in range(5):
        sampler.batch()
    digest, count = sampler.data_order_digest()
    assert count == 5
    assert len(digest) == 64  # sha256 hex


def test_eval_sampler_never_attests_a_data_order() -> None:
    """The eval grid is deterministic by construction — attesting it would
    let an eval digest masquerade as train-pairing evidence."""
    sampler = SequenceSampler(_split(), 4, 16, train=False)
    sampler.batch()
    assert sampler.data_order_digest() is None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs an Apple-silicon mps device"
)
def test_batch_reaches_mps_without_cuda_pinning() -> None:
    """pin_memory() is a CUDA page-locked host-buffer optimization; applied
    on the way to any other accelerator it is invalid. MEASURED 2026-08-20
    (torch 2.13, mps): the unguarded ``pin_memory().to("mps")`` path raised
    ``RuntimeError: Attempted to set the storage of a tensor on device "cpu"
    to a storage on different device "mps:0"`` — killing the field-pretrain
    pilot at its first batch. CI has no mps runner, so this guards the local
    venue only; the cpu path keeps its own coverage above and the cuda path
    is exercised on pods."""
    sampler = SequenceSampler(_split(), batch_size=2, seq_len=16, seed=0, device="mps")
    x, y = sampler.batch()
    assert x.device.type == "mps"
    assert y.device.type == "mps"
