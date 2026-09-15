"""A learned sentence codec is not a probability-table state replacement."""

from dataclasses import replace

import pytest
import torch

from entropic_substrate.sentence_prediction.codec import CodecSpec, NeuralSentenceCodec
from entropic_substrate.sentence_prediction.law import LawSpec

pytestmark = pytest.mark.unit


def codec(mode: str = "magnitude") -> NeuralSentenceCodec:
    torch.manual_seed(42)
    spec = CodecSpec(
        width=16,
        encoder_layers=1,
        decoder_width=16,
        decoder_layers=1,
        head_width=4,
        max_sentence_bytes=16,
    )
    return NeuralSentenceCodec(spec, LawSpec(16, mode))


def test_encoder_returns_one_raw_vector_without_final_normalization() -> None:
    model = codec().eval()
    tokens = torch.tensor([[65, 46, 256], [66, 46, 257]])
    before = model.encoder(tokens)
    with torch.no_grad():
        model.encoder.projection.weight.mul_(3)
    after = model.encoder(tokens)
    assert before.shape == (2, 16)
    torch.testing.assert_close(after, 3 * before)


def test_encoder_is_padding_invariant_and_order_sensitive() -> None:
    model = codec().eval()
    original = torch.tensor([[65, 66, 256]])
    padded = torch.tensor([[65, 66, 256, 259, 259]])
    torch.testing.assert_close(model.encoder(original), model.encoder(padded), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(model.encoder(original), model.encoder(torch.tensor([[66, 65, 256]])))


def test_local_decoder_is_causal_and_uses_latent_magnitude() -> None:
    model = codec().eval()
    latent = torch.randn(1, 16)
    prefix = torch.tensor([[258, 65, 66, 67]])
    altered = prefix.clone()
    altered[:, 3] = 90
    baseline = model.decoder(latent, prefix)
    torch.testing.assert_close(
        baseline[:, :3], model.decoder(latent, altered)[:, :3], atol=1e-6, rtol=1e-6
    )
    assert baseline.shape == (1, 4, 258)
    assert not torch.allclose(baseline, model.decoder(3 * latent, prefix))


@pytest.mark.parametrize("mode", ["sphere", "magnitude"])
def test_reconstruction_path_trains_encoder_and_decoder(mode: str) -> None:
    model = codec(mode)
    complete = torch.tensor([[65, 46, 256]])
    prefix = torch.tensor([[258, 65, 46]])
    logits = model.reconstruct(complete, prefix)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), complete.flatten())
    loss.backward()
    assert any(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.encoder.parameters()
    )
    assert any(
        p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.decoder.parameters()
    )


def test_freezing_encoder_does_not_freeze_the_decoder() -> None:
    model = codec()
    model.freeze_encoder()
    model.train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())
    model.reconstruct(torch.tensor([[65, 256]]), torch.tensor([[258, 65]])).sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([[65, 66]]),
        torch.tensor([[256]]),
        torch.tensor([[65, 256, 66]]),
        torch.tensor([[65, 259, 256]]),
        torch.tensor([[258, 256]]),
        torch.tensor([[65.0, 256.0]]),
    ],
)
def test_encoder_refuses_unclosed_or_malformed_sentences(tokens: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        codec().encoder(tokens)


@pytest.mark.parametrize(
    "tokens", [torch.tensor([[65, 66]]), torch.tensor([[258, 256]]), torch.tensor([[258, 259, 65]])]
)
def test_decoder_refuses_invalid_local_prefixes(tokens: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        codec().decoder(torch.ones(1, 16), tokens)


@pytest.mark.parametrize(
    "changes",
    [
        {"width": 0},
        {"encoder_layers": True},
        {"head_width": 3},
        {"width": 14},
        {"decoder_width": 14},
        {"max_sentence_bytes": 0},
    ],
)
def test_invalid_codec_configuration_is_not_defaulted(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(codec().spec, **changes)


def test_mismatched_law_width_is_refused() -> None:
    with pytest.raises(ValueError, match="width"):
        NeuralSentenceCodec(codec().spec, LawSpec(8, "sphere"))


def test_encoder_layers_are_independently_initialized() -> None:
    original = codec()
    model = NeuralSentenceCodec(replace(original.spec, encoder_layers=2), original.law_spec)
    weights = [
        parameter
        for name, parameter in model.encoder.named_parameters()
        if "linear1.weight" in name
    ]
    assert len(weights) == 2
    assert not torch.equal(weights[0], weights[1])


def test_codec_state_roundtrip_preserves_encoder_freezing() -> None:
    original = codec()
    original.freeze_encoder()
    restored = codec()
    restored.load_state_dict(original.state_dict())
    restored.train()
    assert not restored.encoder.training
    assert all(not p.requires_grad for p in restored.encoder.parameters())
    assert all(p.requires_grad for p in restored.decoder.parameters())


def test_non_tensor_batches_are_rejected_explicitly() -> None:
    with pytest.raises(TypeError, match="tensor"):
        codec().encoder([[65, 256]])


def test_codec_normalization_owns_its_fp32_island(monkeypatch: pytest.MonkeyPatch) -> None:
    original = torch.nn.functional.layer_norm
    seen: list[tuple[torch.dtype, bool]] = []

    def observe(values: torch.Tensor, *args, **kwargs):
        seen.append((values.dtype, torch.is_autocast_enabled("cpu")))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "layer_norm", observe)
    model = codec()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        value = model.encoder.blocks[0].norm1(torch.ones(2, 16, dtype=torch.bfloat16))
    assert seen == [(torch.float32, False)]
    assert value.dtype == torch.float32


@pytest.mark.parametrize(
    "state", [{}, {"encoder_frozen": True, "extra": 1}, {"encoder_frozen": 1}, None]
)
def test_invalid_freeze_payload_is_not_defaulted(state: object) -> None:
    with pytest.raises(ValueError):
        codec().set_extra_state(state)
