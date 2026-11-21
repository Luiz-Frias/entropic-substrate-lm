"""H-program source and boundary contracts, independent of model training."""

from dataclasses import replace

import pytest

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.records import SentenceSpan, SourceDocument

pytestmark = pytest.mark.unit


def document(text: str = "é. Next.\n") -> SourceDocument:
    return SourceDocument(source_id="document-1", text=text, revision="fixture-v1")


def test_document_content_identity_is_separate_from_source_identity() -> None:
    original = document()
    alias = replace(original, source_id="document-2")
    assert original.source == "é. Next.\n".encode()
    assert original.content_digest == alias.content_digest
    assert original.identity != alias.identity
    assert original.identity != replace(original, revision="fixture-v2").identity
    assert original.content_digest != document("different").content_digest


@pytest.mark.parametrize(
    "field,value",
    [("source_id", ""), ("revision", " "), ("text", ""), ("text", "\n "), ("text", 3)],
)
def test_document_refuses_missing_or_invalid_source(field: str, value: object) -> None:
    values = {"source_id": "doc", "text": "content", "revision": "v1"}
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        SourceDocument(**values)


def test_document_refuses_non_utf8_surrogate() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        document("\ud800")


def test_sentence_span_uses_bytes_and_preserves_whitespace() -> None:
    source = document()
    first = SentenceSpan(source, 0, 4, 0)
    second = SentenceSpan(source, 4, len(source.source), 1)
    assert first.source == "é. ".encode()
    assert second.source == b"Next.\n"
    assert first.source + second.source == source.source
    assert first.document.identity == second.document.identity


@pytest.mark.parametrize(
    "start,end,index",
    [(-1, 2, 0), (0, 0, 0), (0, 99, 0), (1, 3, 0), (0, 1, 0), (0, 3, -1), (False, 3, 0)],
)
def test_sentence_span_refuses_invalid_or_partial_utf8_offsets(
    start: int, end: int, index: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        SentenceSpan(document(), start, end, index)


def test_boundary_codec_roundtrips_every_byte_without_control_collision() -> None:
    codec = SentenceByteCodec()
    sentences = (bytes(range(256)), b"\nNext.\n")
    encoded = codec.encode_document(sentences)
    assert encoded[:256] == tuple(range(256))
    assert encoded[256] == codec.sentence_end
    assert encoded[-1] == codec.document_end
    assert codec.decode_document(encoded) == sentences
    assert codec.output_size == 258
    assert codec.input_size == 260
    assert codec.bos not in encoded and codec.pad not in encoded


def test_bijective_relabeling_leaves_control_symbols_and_canonical_bytes_intact() -> None:
    identity = SentenceByteCodec()
    recoded = SentenceByteCodec(tuple(reversed(range(256))))
    source = (bytes(range(256)), b"hello")
    tokens = recoded.encode_document(source)
    assert tokens[0] == 255
    assert recoded.decode_document(tokens) == source
    assert tokens.count(recoded.sentence_end) == 1
    assert tokens[-1] == identity.document_end
    assert recoded.identity != identity.identity
    assert recoded.canonical_byte(255) == 0
    with pytest.raises(ValueError, match="canonical byte"):
        recoded.canonical_byte(recoded.sentence_end)


@pytest.mark.parametrize("mapping", [(0,) * 256, tuple(range(255)), (*range(255), True)])
def test_boundary_codec_refuses_non_bijection(mapping: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="bijection"):
        SentenceByteCodec(mapping)


@pytest.mark.parametrize("sentences", [(), (b"",), ("not bytes",), [b"not immutable"]])
def test_boundary_encoder_refuses_empty_or_invalid_sentences(sentences: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        SentenceByteCodec().encode_document(sentences)


@pytest.mark.parametrize(
    "tokens",
    [
        (),
        (65,),
        (256, 257),
        (65, 256),
        (65, 257, 66),
        (65, 257, 66, 257),
        (65, 256, 257),
        (65, 258, 257),
        (65, 259, 257),
        (-1, 257),
        (True, 257),
        [65, 257],
    ],
)
def test_boundary_decoder_refuses_malformed_envelopes(tokens: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        SentenceByteCodec().decode_document(tokens)
