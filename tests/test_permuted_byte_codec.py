"""D5 exact byte relabeling controls at the observation/readout boundary."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from entropic_substrate.core.observation.contracts import ObservationCodecPort
from entropic_substrate.core.observation.permuted_byte_codec import PermutedByteObservationCodec

pytestmark = pytest.mark.unit


def test_codec_is_reversible_stable_and_preserves_source_length() -> None:
    codec = PermutedByteObservationCodec.from_seed(42)
    source = bytes(range(256))
    assert isinstance(codec, ObservationCodecPort)
    assert codec.decode(codec.encode(source)) == source
    assert codec.encode(source).numel() == len(source)
    assert codec.encode(b"").numel() == 0
    assert codec.decode(codec.encode(b"")) == b""
    assert codec.mapping != tuple(range(256))
    assert codec == PermutedByteObservationCodec.from_seed(42)
    assert (
        codec.construction_digest != PermutedByteObservationCodec.from_seed(43).construction_digest
    )
    assert codec.identity.cardinality == 256


def test_mapping_and_digest_are_both_validated() -> None:
    with pytest.raises(ValueError, match="bijection"):
        PermutedByteObservationCodec(mapping=(0,) * 256)
    with pytest.raises(ValueError, match="bijection"):
        PermutedByteObservationCodec(mapping=tuple(range(255)))
    with pytest.raises(ValueError, match="bijection"):
        PermutedByteObservationCodec(mapping=(True, *range(1, 256)))
    with pytest.raises(ValueError, match="seed"):
        PermutedByteObservationCodec.from_seed(-1)
    with pytest.raises(ValueError, match="digest"):
        PermutedByteObservationCodec.from_mapping(
            mapping=tuple(range(256)), expected_digest="0" * 64
        )
    codec = PermutedByteObservationCodec.from_seed(3)
    restored = PermutedByteObservationCodec.from_mapping(
        mapping=codec.mapping, expected_digest=codec.construction_digest
    )
    assert restored == codec


def test_coupled_initialization_preserves_hidden_states_and_canonical_readout() -> None:
    torch.manual_seed(9)
    codec = PermutedByteObservationCodec.from_seed(42)
    embedding = nn.Embedding(256, 8)
    middle = nn.Sequential(nn.Linear(8, 8), nn.SiLU())
    readout = nn.Linear(8, 256)
    source = b"same future outcome\x00\xff"
    canonical_ids = torch.tensor(list(source))
    canonical_hidden = middle(embedding(canonical_ids))
    mapped_embedding = codec.permute_boundary_rows(embedding.weight)
    mapped_readout = codec.permute_boundary_rows(readout.weight)
    mapped_bias = codec.permute_boundary_rows(readout.bias)
    mapped_hidden = middle(mapped_embedding[codec.encode(source)])
    torch.testing.assert_close(mapped_hidden, canonical_hidden, atol=0.0, rtol=0.0)
    mapped_logits = torch.nn.functional.linear(mapped_hidden, mapped_readout, mapped_bias)
    canonical_logits = codec.canonical_logits(mapped_logits, codec_identity=codec.identity)
    torch.testing.assert_close(canonical_logits, readout(canonical_hidden), atol=0.0, rtol=0.0)
    canonical_logits.square().mean().backward()
    assert embedding.weight.grad is not None
    assert readout.weight.grad is not None
    assert readout.bias.grad is not None


def test_codec_refuses_wrong_ids_dimensions_and_readout_provenance() -> None:
    codec = PermutedByteObservationCodec.from_seed(42)
    with pytest.raises(ValueError, match="range"):
        codec.decode(torch.tensor([256]))
    with pytest.raises(TypeError, match="int64"):
        codec.decode(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="rank-1"):
        codec.decode(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="256"):
        codec.permute_boundary_rows(torch.randn(255, 8))
    with pytest.raises(ValueError, match="256"):
        codec.canonical_logits(torch.zeros(2, 255), codec_identity=codec.identity)
    with pytest.raises(ValueError, match="identity"):
        codec.canonical_logits(
            torch.zeros(2, 256), codec_identity=PermutedByteObservationCodec.from_seed(43).identity
        )
