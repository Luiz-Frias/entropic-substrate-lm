"""D1 horizon-aware predictive batching contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from entropic_substrate.core.corpus.batching import PredictiveBatch, SequenceSampler, TokenSplit
from entropic_substrate.core.observation.contracts import ObservationAlignment

pytestmark = pytest.mark.unit


def _numbered_split(n: int = 64) -> TokenSplit:
    return TokenSplit(np.arange(n, dtype=np.int32))


def _predictive_batch(
    *,
    online_shape: tuple[int, int] = (1, 2),
    target_context: torch.Tensor | None = None,
    source_starts: torch.Tensor | None = None,
) -> PredictiveBatch:
    batch_size, sequence_length = online_shape
    online = torch.zeros(online_shape, dtype=torch.int64)
    language = torch.zeros(online_shape, dtype=torch.int64)
    context = (
        target_context
        if target_context is not None
        else torch.zeros((batch_size, sequence_length + 1), dtype=torch.int64)
    )
    starts = (
        source_starts if source_starts is not None else torch.arange(batch_size, dtype=torch.int64)
    )
    return PredictiveBatch(
        online_observations=online,
        language_targets=language,
        target_encoder_context=context,
        source_starts=starts,
        alignments=tuple(
            ObservationAlignment.predictive(start=start, sequence_length=sequence_length, horizon=1)
            for start in range(batch_size)
        ),
        horizon=1,
    )


def test_predictive_batch_horizon_one_uses_one_contiguous_span_per_example() -> None:
    sampler = SequenceSampler(_numbered_split(), batch_size=2, seq_len=4, train=False)

    batch = sampler.predictive_batch(horizon=1)

    assert batch.source_starts.tolist() == [0, 29]
    torch.testing.assert_close(batch.online_observations[0], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(batch.language_targets[0], torch.tensor([1, 2, 3, 4]))
    torch.testing.assert_close(batch.target_encoder_context[0], torch.tensor([0, 1, 2, 3, 4]))
    assert batch.alignments[0].online_observations.start == 0
    assert batch.alignments[1].online_observations.start == 29


def test_predictive_batch_horizon_more_than_one_has_exact_offsets_and_spans() -> None:
    sampler = SequenceSampler(_numbered_split(), batch_size=2, seq_len=4, train=False)

    batch = sampler.predictive_batch(horizon=3)

    assert batch.source_starts.tolist() == [0, 28]
    torch.testing.assert_close(batch.online_observations[1], torch.tensor([28, 29, 30, 31]))
    torch.testing.assert_close(batch.language_targets[1], torch.tensor([29, 30, 31, 32]))
    torch.testing.assert_close(
        batch.target_encoder_context[1], torch.tensor([28, 29, 30, 31, 32, 33, 34])
    )
    assert batch.alignments[1].target_encoder_context.stop == 35
    assert batch.alignments[1].future_observations.start == 32


def test_future_context_is_not_an_alias_of_the_online_slice() -> None:
    sampler = SequenceSampler(_numbered_split(), batch_size=1, seq_len=4, train=False)
    batch = sampler.predictive_batch(horizon=3)
    before = batch.online_observations.clone()

    batch.target_encoder_context[0, -1] = -999

    assert torch.equal(batch.online_observations, before)


@pytest.mark.parametrize("online_shape", [(0, 2), (1, 0)])
def test_predictive_batch_rejects_empty_model_axes(online_shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="positive"):
        _predictive_batch(online_shape=online_shape)


def test_predictive_batch_requires_one_device_for_all_model_tensors() -> None:
    with pytest.raises(ValueError, match="same device"):
        _predictive_batch(target_context=torch.empty((1, 3), dtype=torch.int64, device="meta"))


def test_predictive_batch_requires_cpu_source_metadata() -> None:
    with pytest.raises(ValueError, match="CPU"):
        _predictive_batch(source_starts=torch.empty((1,), dtype=torch.int64, device="meta"))


@pytest.mark.parametrize("horizon", [0, -1])
def test_predictive_batch_rejects_nonpositive_horizons(horizon: int) -> None:
    sampler = SequenceSampler(_numbered_split(), batch_size=1, seq_len=4)
    with pytest.raises(ValueError, match="horizon"):
        sampler.predictive_batch(horizon=horizon)


def test_predictive_batch_rejects_a_horizon_that_does_not_fit_the_split() -> None:
    sampler = SequenceSampler(_numbered_split(10), batch_size=1, seq_len=8)
    with pytest.raises(ValueError, match="need at least"):
        sampler.predictive_batch(horizon=3)


def test_legacy_batch_outputs_and_data_order_digest_are_bitwise_unchanged() -> None:
    sampler = SequenceSampler(_numbered_split(), batch_size=3, seq_len=5, train=True, seed=19)
    expected_x = torch.tensor([[46, 47, 48, 49, 50], [7, 8, 9, 10, 11], [2, 3, 4, 5, 6]])
    expected_y = torch.tensor([[47, 48, 49, 50, 51], [8, 9, 10, 11, 12], [3, 4, 5, 6, 7]])

    first_x, first_y = sampler.batch()
    assert torch.equal(first_x, expected_x)
    assert torch.equal(first_y, expected_y)
    for _ in range(31):
        sampler.batch()

    assert (
        sampler.data_order_digest()
        == (
            "984e5c552c61f1e6ae14a6f2fa76ecc28968af2c3365f79041e5f46a0861aee3",  # pragma: allowlist secret
            32,
        )
    )


def test_horizon_one_preserves_the_legacy_seeded_offset_stream() -> None:
    legacy = SequenceSampler(_numbered_split(), batch_size=3, seq_len=5, train=True, seed=23)
    predictive = SequenceSampler(_numbered_split(), batch_size=3, seq_len=5, train=True, seed=23)

    legacy_x, legacy_y = legacy.batch()
    predictive_batch = predictive.predictive_batch(horizon=1)

    assert torch.equal(predictive_batch.online_observations, legacy_x)
    assert torch.equal(predictive_batch.language_targets, legacy_y)
    assert predictive.data_order_digest() == legacy.data_order_digest()


def _legacy_offset_oracle(
    *, train: bool, seed: int, n: int, seq_len: int, batch_size: int, draws: int
) -> tuple[list[list[int]], torch.Tensor]:
    """Offsets and final generator state of the deleted ``SequenceSampler._starts``.

    Copied from the parent tip (68332cc) before its deletion: one
    ``torch.randint(0, n - seq_len, (batch_size,))`` per train draw, and the
    strided modular grid with a cursor per eval draw.
    """
    max_start = n - seq_len - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out: list[list[int]] = []
    for cursor in range(draws):
        if train:
            out.append(torch.randint(0, max_start + 1, (batch_size,), generator=generator).tolist())
        else:
            stride = max(max_start // max(batch_size, 1), 1)
            base = (cursor * batch_size * stride) % (max_start + 1)
            out.append(((base + np.arange(batch_size) * stride) % (max_start + 1)).tolist())
    return out, generator.get_state()


@pytest.mark.parametrize("train", [True, False])
def test_batch_draws_the_legacy_offsets_and_consumes_the_legacy_rng(train: bool) -> None:
    """``batch()`` and ``predictive_batch(1)`` both draw the parent tip's offset stream.

    The generator-state comparison pins the draw ORDER and COUNT: a second
    ``torch.randint`` per batch leaves different state even when the first
    draw happens to match.
    """
    n, seq_len, batch_size, draws = 64, 5, 3, 6
    expected, expected_state = _legacy_offset_oracle(
        train=train, seed=29, n=n, seq_len=seq_len, batch_size=batch_size, draws=draws
    )
    legacy = SequenceSampler(
        _numbered_split(n), batch_size=batch_size, seq_len=seq_len, train=train, seed=29
    )
    predictive = SequenceSampler(
        _numbered_split(n), batch_size=batch_size, seq_len=seq_len, train=train, seed=29
    )
    for offsets in expected:
        x, _y = legacy.batch()
        assert x[:, 0].tolist() == offsets
        assert predictive.predictive_batch(horizon=1).source_starts.tolist() == offsets
    assert torch.equal(legacy.generator.get_state(), expected_state)
    assert torch.equal(predictive.generator.get_state(), expected_state)
    assert legacy.data_order_digest() == predictive.data_order_digest()


def test_batch_places_tensors_on_a_non_cpu_device() -> None:
    """The non-CUDA accelerator branch (``meta`` stands in for mps) moves both tensors."""
    sampler = SequenceSampler(_numbered_split(), batch_size=2, seq_len=4, device="meta")
    x, y = sampler.batch()
    assert x.device.type == "meta"
    assert y.device.type == "meta"
    batch = sampler.predictive_batch(horizon=2)
    assert batch.online_observations.device.type == "meta"
    assert batch.language_targets.device.type == "meta"
    assert batch.target_encoder_context.device.type == "meta"
    assert batch.source_starts.device.type == "cpu"


class _RecordingTensor:
    """Stand-in that records the placement calls a CUDA device would receive."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def pin_memory(self) -> _RecordingTensor:
        self.calls.append("pin_memory")
        return self

    def to(self, device: torch.device, *, non_blocking: bool = False) -> _RecordingTensor:
        self.calls.append(f"to({device.type}, non_blocking={non_blocking})")
        return self


def test_device_placement_helper_pins_then_copies_only_for_cuda() -> None:
    """MEASURED 2026-08-20 (comment in the helper): mps must never see ``pin_memory``."""
    from entropic_substrate.core.corpus.batching import _to_device

    cuda = _RecordingTensor()
    assert _to_device(cuda, torch.device("cuda")) is cuda  # type: ignore[arg-type]
    assert cuda.calls == ["pin_memory", "to(cuda, non_blocking=True)"]

    mps = _RecordingTensor()
    assert _to_device(mps, torch.device("mps")) is mps  # type: ignore[arg-type]
    assert mps.calls == ["to(mps, non_blocking=False)"]

    cpu = torch.zeros(2, dtype=torch.int64)
    assert _to_device(cpu, torch.device("cpu")) is cpu
