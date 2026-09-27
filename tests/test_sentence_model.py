"""Complete history-to-predicted-latent-to-local-readout causal composition."""

import pytest
import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec as Bytes
from entropic_substrate.core.sentence.codec import CodecSpec
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.spine import SpineSpec

pytestmark = pytest.mark.unit


def tiny_model(
    interface: str = "magnitude", carrier: str = "magnitude", consumer: str = "isotropic"
) -> SentencePredictor:
    torch.manual_seed(42)
    return SentencePredictor(
        CodecSpec(
            width=8,
            encoder_layers=1,
            decoder_width=8,
            decoder_layers=1,
            head_width=4,
            max_sentence_bytes=16,
        ),
        SpineSpec(width=8, layers=2, head_width=4, context_sentences=4, carrier=carrier),
        LawSpec(8, interface),
        consumer=consumer,
    )


def history() -> torch.Tensor:
    return torch.tensor([[[65, 46, Bytes.sentence_end], [66, 46, Bytes.sentence_end]]])


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
def test_generator_composes_each_independent_preservation_construction(
    interface: str, carrier: str
) -> None:
    model = tiny_model(interface, carrier)
    result = model(history(), torch.tensor([[Bytes.bos, 67]]))
    assert result.logits.shape == (1, 2, 258)
    assert result.sample.shape == (1, 8)
    assert result.prediction.states.shape == (1, 3, 8)
    assert result.prediction.sentences.shape == (1, 2, 8)
    assert result.prediction.diagnostics is not None
    norms = result.prediction.sentences.norm(dim=-1)
    if interface == "sphere":
        torch.testing.assert_close(norms, torch.ones_like(norms))
    else:
        assert not torch.allclose(norms, torch.ones_like(norms))


def test_future_history_suffix_cannot_change_prior_spine_states() -> None:
    model = tiny_model().eval()
    before = model.predict(history())
    mutated = history()
    mutated[:, 1, 0] = 88
    after = model.predict(mutated)
    torch.testing.assert_close(before.states[:, :2], after.states[:, :2])
    assert not torch.allclose(before.states[:, 2:], after.states[:, 2:])


def test_local_byte_prefix_cannot_change_macro_state_or_latent_sample() -> None:
    model = tiny_model().eval()
    rng = torch.get_rng_state()
    before = model(history(), torch.tensor([[Bytes.bos, 67]]))
    torch.set_rng_state(rng)
    after = model(history(), torch.tensor([[Bytes.bos, 99]]))
    torch.testing.assert_close(before.prediction.states, after.prediction.states, atol=0, rtol=0)
    torch.testing.assert_close(before.sample, after.sample, atol=0, rtol=0)
    assert not torch.allclose(before.logits[:, 1], after.logits[:, 1])


@pytest.mark.parametrize("consumer", ["isotropic", "fisher"])
def test_decoder_loss_trains_spine_and_law_without_future_latent_supervision(consumer: str) -> None:
    model = tiny_model(consumer=consumer)
    model.codec.freeze_encoder()
    result = model(history(), torch.tensor([[Bytes.bos, 67]]))
    torch.nn.functional.cross_entropy(
        result.logits.flatten(0, 1), torch.tensor([67, 46])
    ).backward()
    assert all(p.grad is None for p in model.codec.encoder.parameters())
    for component in (model.codec.decoder, model.spine, model.head):
        assert all(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in component.parameters()
        )
        assert sum(float(p.grad.abs().sum()) for p in component.parameters()) > 0


def test_no_future_or_raw_history_argument_exists_at_the_decoder_boundary() -> None:
    model = tiny_model()
    observed: list[tuple[torch.Size, torch.Size]] = []

    def record(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        assert len(inputs) == 2
        observed.append((inputs[0].shape, inputs[1].shape))

    with model.codec.decoder.register_forward_pre_hook(record):
        model(history(), torch.tensor([[Bytes.bos]]))
    assert observed == [(torch.Size([1, 8]), torch.Size([1, 1]))]


def test_empty_history_uses_bos_and_does_not_encode_a_fake_sentence() -> None:
    model = tiny_model()
    result = model(torch.empty(2, 0, 1, dtype=torch.int64), torch.full((2, 1), Bytes.bos))
    assert result.prediction.sentences.shape == (2, 0, 8)
    assert result.prediction.states.shape == (2, 1, 8)


def test_ended_document_and_context_overflow_are_not_silently_reinterpreted() -> None:
    model = tiny_model()
    ended = history()
    ended[:, -1, -1] = Bytes.document_end
    with pytest.raises(ValueError, match="document-end"):
        model.predict(ended)
    with pytest.raises(ValueError, match="context"):
        model.predict(history().repeat(1, 3, 1))


def test_nonfinite_and_malformed_history_are_refused() -> None:
    model = tiny_model()
    model.predict(history())
    with pytest.raises(ValueError, match="terminal boundary"):
        model.predict(torch.tensor([[[65, 46]]]))
    with pytest.raises(ValueError, match="control symbol"):
        model.predict(torch.tensor([[[Bytes.bos, 46, Bytes.sentence_end]]]))
    with pytest.raises(ValueError, match="padding"):
        model.predict(torch.tensor([[[65, Bytes.pad, Bytes.sentence_end]]]))
    with pytest.raises(ValueError, match="history"):
        model.predict(torch.ones(1, 3, dtype=torch.int64))
    with pytest.raises(TypeError, match="int64"):
        model.predict(history().float())


def test_inventories_count_encoder_spine_decoder_and_head_exactly_once() -> None:
    model = tiny_model()
    parameter_ids = [
        id(p)
        for component in (model.codec, model.spine, model.head)
        for p in component.parameters()
    ]
    assert len(parameter_ids) == len(set(parameter_ids)) == len(list(model.parameters()))
