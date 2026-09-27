"""Safe primitive recovery payloads preserve sampled and operative rollout state."""

from dataclasses import replace
from io import BytesIO

import pytest
import torch
from tests.test_sentence_model import tiny_model
from tests.test_sentence_rollout import scripted_tokens

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.recovery_state import (
    capture_rng,
    restore_rng,
    rollout_from_payload,
    rollout_to_payload,
)
from entropic_substrate.core.sentence.rollout import advance_rollout, begin_rollout

pytestmark = pytest.mark.unit


def test_rng_payload_replays_torch_sampling_without_sharing_mutable_storage() -> None:
    torch.manual_seed(12)
    saved = capture_rng()
    expected = torch.randn(15)
    torch.randn(31)
    restore_rng(saved)
    torch.testing.assert_close(torch.randn(15), expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="RNG"):
        restore_rng(saved | {"schema": "unknown"})
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="RNG"):
        restore_rng(saved | {"cpu": torch.zeros(2, dtype=torch.uint8)})
    torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)


def test_pending_rollout_roundtrip_preserves_next_observation() -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    state = advance_rollout(model, begin_rollout(model, (b"A.",), codec), codec)
    saved = rollout_to_payload(state, model, codec)
    rng = capture_rng()
    expected = advance_rollout(model, state, codec)
    restored = rollout_from_payload(saved, model, codec)
    assert restored is not None and restored.sample is not None
    assert restored.context.data_ptr() != state.context.data_ptr()
    restore_rng(rng)
    actual = advance_rollout(model, restored, codec)
    assert actual.observations == expected.observations
    torch.testing.assert_close(actual.sample, expected.sample, atol=0, rtol=0)


def test_completed_and_failed_rollout_payloads_retain_their_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, [65, 256, 255, 257])
    state = begin_rollout(model, (), codec)
    for _ in range(4):
        state = advance_rollout(model, state, codec)
    assert state.status == "invalid_utf8"
    restored = rollout_from_payload(rollout_to_payload(state, model, codec), model, codec)
    assert restored.status == state.status
    assert restored.pending == state.pending and restored.observations == state.observations
    torch.testing.assert_close(restored.emissions[0].operative, state.emissions[0].operative)
    torch.testing.assert_close(restored.emissions[0].shadow, state.emissions[0].shadow)
    assert rollout_from_payload(None, model, codec) is None


@pytest.mark.parametrize("mutation", ["codec", "history", "trace", "scores", "nonfinite", "sample"])
def test_forged_feedback_payload_is_rejected(mutation: str) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    state = advance_rollout(model, begin_rollout(model, (b"A.",), codec), codec)
    payload = rollout_to_payload(state, model, codec)
    assert payload is not None
    if mutation == "codec":
        payload["codec_identity"] = "wrong"
    elif mutation == "history":
        payload["history"] = ()
    elif mutation == "trace":
        payload["observations"] = (256,)
    elif mutation == "scores":
        payload["log_scores"] = (float("nan"),)
    elif mutation == "nonfinite":
        payload["context"].fill_(float("nan"))
    else:
        payload["sample"] = None
    with pytest.raises(ValueError):
        rollout_from_payload(payload, model, codec)


def test_completed_feedback_cannot_be_replaced_by_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, [65, 256])
    state = begin_rollout(model, (), codec)
    state = advance_rollout(model, advance_rollout(model, state, codec), codec)
    wrong = replace(state, context=state.emissions[0].shadow.unsqueeze(1))
    with pytest.raises(ValueError, match="operative"):
        rollout_to_payload(wrong, model, codec)


@pytest.mark.parametrize(
    ("tokens", "prompt", "status"),
    [
        ([65, 257], (), "document_end"),
        ([65, 256], (b"A.",) * 4, "context_limit"),
        ([256], (), "empty_sentence"),
        ([65] * 17, (), "length_limit"),
        ([65, 256], (), "active"),
    ],
)
def test_safe_torch_payload_roundtrip_covers_terminal_and_ready_states(
    tokens: list[int], prompt: tuple[bytes, ...], status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, codec = tiny_model().eval(), SentenceByteCodec()
    scripted_tokens(monkeypatch, tokens)
    state = begin_rollout(model, prompt, codec)
    for _ in tokens:
        state = advance_rollout(model, state, codec)
    stream = BytesIO()
    torch.save(rollout_to_payload(state, model, codec), stream)
    stream.seek(0)
    restored = rollout_from_payload(torch.load(stream, weights_only=True), model, codec)
    assert restored is not None and restored.status == status
    assert restored.observations == state.observations
    torch.testing.assert_close(restored.context, state.context, atol=0, rtol=0)
