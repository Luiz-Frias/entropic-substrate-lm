"""D1 observation-boundary contracts.

These tests pin the first predictive-state seam: alphabet cardinality belongs
to an external codec and language readout, never to the latent state width.
"""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.core.observation.byte_codec import ByteObservationCodec
from entropic_substrate.core.observation.contracts import (
    CodecIdentity,
    EncodedObservations,
    ObservationAlignment,
    SourceSpan,
)
from entropic_substrate.core.observation.modules import LanguageReadout, LearnedObservationAdapter

pytestmark = pytest.mark.unit


def _codec_identity(cardinality: int) -> CodecIdentity:
    return CodecIdentity(
        name=f"fixture-codec-{cardinality}",
        construction_digest="a" * 64,
        cardinality=cardinality,
    )


def test_identity_byte_codec_round_trips_all_boundary_byte_ids() -> None:
    codec = ByteObservationCodec()
    source = bytes([0, 1, 127, 254, 255])

    ids = codec.encode(source)

    assert ids.dtype == torch.int64
    assert ids.tolist() == [0, 1, 127, 254, 255]
    assert codec.decode(ids) == source


def test_byte_codec_construction_digest_is_stable_and_configuration_specific() -> None:
    default_a = ByteObservationCodec()
    default_b = ByteObservationCodec()
    revised = ByteObservationCodec(construction_label="identity-byte-v2")

    assert default_a.construction_digest == default_b.construction_digest
    assert default_a.codec_identity.construction_digest == default_a.construction_digest
    assert default_a.codec_identity.cardinality == 256
    assert default_a.construction_digest != revised.construction_digest


def test_source_spans_encode_exact_predictive_alignment() -> None:
    alignment = ObservationAlignment.predictive(start=11, sequence_length=4, horizon=3)

    assert alignment.online_observations == SourceSpan(11, 15)
    assert alignment.language_targets == SourceSpan(12, 16)
    assert alignment.target_encoder_context == SourceSpan(11, 18)
    assert alignment.future_observations == SourceSpan(15, 18)
    assert alignment.horizon == 3


@pytest.mark.parametrize("vocab_size", [7, 256, 509])
def test_adapter_and_readout_are_untied_and_latent_width_is_vocab_independent(
    vocab_size: int,
) -> None:
    latent_width = 13
    adapter = LearnedObservationAdapter(vocab_size=vocab_size, latent_width=latent_width)
    readout = LanguageReadout(latent_width=latent_width, vocab_size=vocab_size)
    ids = torch.tensor([[0, vocab_size - 1]], dtype=torch.int64)

    encoded = adapter(ids, _codec_identity(vocab_size))
    logits = readout(encoded.values)

    assert encoded.values.shape == (1, 2, latent_width)
    assert logits.shape == (1, 2, vocab_size)
    assert adapter.embedding.weight.data_ptr() != readout.projection.weight.data_ptr()
    assert adapter.latent_width == latent_width
    assert readout.latent_width == latent_width


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (torch.zeros(2, 3, dtype=torch.float32), "rank-3"),
        (torch.zeros(2, 3, 4, dtype=torch.int64), "floating"),
        (torch.zeros(0, 3, 4, dtype=torch.float32), "positive"),
    ],
)
def test_encoded_observations_reject_invalid_tensor_values(
    values: torch.Tensor, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        EncodedObservations(values=values, codec_identity=ByteObservationCodec().codec_identity)


def test_encoded_observations_reject_nonidentity_codec_provenance() -> None:
    with pytest.raises(TypeError, match="CodecIdentity"):
        EncodedObservations(
            values=torch.zeros(1, 1, 1, dtype=torch.float32),
            codec_identity="not-a-codec-identity",  # type: ignore[arg-type]
        )


def test_observation_adapter_refuses_codec_cardinality_mismatch() -> None:
    adapter = LearnedObservationAdapter(vocab_size=7, latent_width=3)
    ids = torch.tensor([[0, 6]], dtype=torch.int64)

    with pytest.raises(ValueError, match="cardinality"):
        adapter(ids, _codec_identity(8))


@pytest.mark.parametrize("shape", [(0, 2, 3), (1, 0, 3)])
def test_language_readout_rejects_empty_batch_or_time_axis(shape: tuple[int, int, int]) -> None:
    readout = LanguageReadout(latent_width=3, vocab_size=7)
    with pytest.raises(ValueError, match="positive"):
        readout(torch.zeros(shape, dtype=torch.float32))


def test_value_contracts_fail_loudly_on_invalid_spans_and_byte_ids() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SourceSpan(4, 4)
    with pytest.raises(ValueError, match="horizon"):
        ObservationAlignment.predictive(start=0, sequence_length=4, horizon=0)
    with pytest.raises(ValueError, match="0, 255"):
        ByteObservationCodec().decode(torch.tensor([256], dtype=torch.int64))
