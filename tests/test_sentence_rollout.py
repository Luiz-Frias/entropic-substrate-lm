"""Bytewise generation exposes failures and uses emitted, not imagined, feedback."""

from dataclasses import replace

import pytest
import torch
from tests.test_sentence_model import tiny_model

from entropic_substrate.sentence_prediction.boundary import SentenceByteCodec
from entropic_substrate.sentence_prediction.rollout import advance_rollout, begin_rollout

pytestmark = pytest.mark.unit


def scripted_tokens(monkeypatch: pytest.MonkeyPatch, values: list[int]) -> None:
    sequence = iter(values)

    def sample(probabilities: torch.Tensor, num_samples: int) -> torch.Tensor:
        assert probabilities.shape == (258,) and num_samples == 1
        return torch.tensor([next(sequence)], device=probabilities.device)

    monkeypatch.setattr(torch, "multinomial", sample)


def test_multiple_sentences_feedback_and_shadow_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model().eval()
    codec = SentenceByteCodec()
    scripted_tokens(monkeypatch, [66, 46, 256, 67, 46, 257])
    state = begin_rollout(model, (b"A.",), codec)
    for _ in range(6):
        state = advance_rollout(model, state, codec)
    assert state.status == "document_end"
    assert state.history == (b"A.", b"B.", b"C.")
    assert [item.terminal for item in state.emissions] == [256, 257]
    assert state.pending == () and state.sample is None
    assert state.context.shape == (1, 3, 8)
    for index, item in enumerate(state.emissions, 1):
        complete = torch.tensor([[*item.source, item.terminal]])
        expected = model.codec.encoder(complete)
        torch.testing.assert_close(item.operative, expected)
        torch.testing.assert_close(state.context[:, index], item.operative)
        assert not torch.allclose(item.shadow, item.operative)
        assert not item.shadow.requires_grad and not item.operative.requires_grad


def test_mutating_archived_shadow_cannot_change_next_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, [65, 256])
    state = begin_rollout(model, (), codec)
    state = advance_rollout(model, advance_rollout(model, state, codec), codec)
    changed = replace(state, emissions=(replace(state.emissions[0], shadow=torch.zeros(1, 8)),))
    rng = torch.get_rng_state()
    scripted_tokens(monkeypatch, [66, 66])
    first = advance_rollout(model, state, codec)
    torch.set_rng_state(rng)
    second = advance_rollout(model, changed, codec)
    torch.testing.assert_close(first.sample, second.sample, atol=0, rtol=0)


def test_bytewise_rng_replay_preserves_pending_sample_and_prefix() -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    state = begin_rollout(model, (b"A.",), codec)
    state = advance_rollout(model, state, codec)
    assert state.status == "active" and len(state.pending) == 1
    rng = torch.get_rng_state()
    first = advance_rollout(model, state, codec)
    torch.set_rng_state(rng)
    second = advance_rollout(model, state, codec)
    assert first.pending == second.pending and first.status == second.status
    torch.testing.assert_close(first.sample, second.sample, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("tokens", "status"),
    [([256], "empty_sentence"), ([255, 256], "invalid_utf8"), ([65] * 17, "length_limit")],
)
def test_invalid_generation_is_retained_not_repaired(
    tokens: list[int], status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, tokens)
    state = begin_rollout(model, (), codec)
    for _ in tokens:
        state = advance_rollout(model, state, codec)
    assert state.status == status
    assert state.history == () and state.emissions == ()
    assert state.sample is not None
    assert len(state.log_scores) == len(tokens)
    assert state.observations == tuple(tokens)
    with pytest.raises(ValueError, match="terminal"):
        advance_rollout(model, state, codec)


def test_inverse_codec_precedes_utf8_validation_and_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model().eval()
    codec = SentenceByteCodec(tuple(reversed(range(256))))
    scripted_tokens(monkeypatch, [codec.permutation[65], 256])
    state = begin_rollout(model, (), codec)
    state = advance_rollout(model, advance_rollout(model, state, codec), codec)
    assert state.history == (b"A",)
    assert state.emissions[0].source == b"A"
    with pytest.raises(ValueError, match="codec"):
        advance_rollout(model, state, SentenceByteCodec())


def test_prompt_limits_utf8_and_explicit_boundaries_are_checked() -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    for prompt in ((b"",), (b"\xff",), (b"A" * 17,), (b"A",) * 5):
        with pytest.raises(ValueError):
            begin_rollout(model, prompt, codec)
    with pytest.raises(TypeError, match="tuple"):
        begin_rollout(model, [b"A."], codec)


def test_context_limit_is_reported_without_sliding_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, [65, 256])
    state = begin_rollout(model, (b"A.",) * 4, codec)
    state = advance_rollout(model, advance_rollout(model, state, codec), codec)
    assert state.status == "context_limit"
    assert len(state.history) == 5
    assert len(state.emissions) == 1


def test_encoded_feedback_uses_the_same_predictor_path_as_observations() -> None:
    model = tiny_model().eval()
    source = torch.tensor([[[65, 46, 256]]])
    expected = model.predict(source)
    actual = model.predict_latents(expected.sentences)
    torch.testing.assert_close(actual.states, expected.states, atol=0, rtol=0)
    torch.testing.assert_close(actual.law.direction, expected.law.direction, atol=0, rtol=0)


@pytest.mark.parametrize("interface", ["sphere", "magnitude"])
@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
def test_rollout_preserves_axes_with_fisher_under_autocast(
    interface: str, carrier: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, codec = tiny_model(interface, carrier, "fisher").eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, [65, 256])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        state = begin_rollout(model, (b"B.",), codec)
        state = advance_rollout(model, advance_rollout(model, state, codec), codec)
    assert state.status == "active"
    assert state.context.dtype == torch.float32
    assert state.emissions[0].shadow.dtype == torch.float32


def test_spherical_feedback_cannot_accept_an_unnormalized_carrier() -> None:
    model = tiny_model("sphere").eval()
    with pytest.raises(ValueError, match="unit operative"):
        model.predict_latents(torch.ones(1, 2, 8))
