"""D4 causal chunk construction and honest execution-count contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from entropic_substrate.core.latent.coarse_graining import (
    CausalCoarseGrainer,
    CoarseGrainingResult,
)

pytestmark = pytest.mark.unit
MODES = ("identity", "fixed", "learned", "always", "never", "random", "fixed_duration")


@pytest.mark.parametrize("mode", MODES)
def test_prefix_is_unchanged_by_future_and_prefix_truncation(mode: str) -> None:
    torch.manual_seed(7)
    module = CausalCoarseGrainer(width=3, mode=mode)
    states = torch.randn(2, 19, 3)
    changed = states.clone()
    changed[:, 9:] += 100
    whole, suffix, prefix = module(states), module(changed), module(states[:, :9])
    torch.testing.assert_close(whole.states[:, :9], suffix.states[:, :9], atol=0, rtol=0)
    torch.testing.assert_close(whole.states[:, :9], prefix.states, atol=0, rtol=0)
    assert torch.equal(whole.diagnostics.boundaries[:, :9], prefix.diagnostics.boundaries)
    assert not torch.equal(whole.states[:, 9:], suffix.states[:, 9:])


def test_fixed_eight_closes_at_seventh_index_without_flushing_final_partial_chunk() -> None:
    module = CausalCoarseGrainer(width=2, mode="fixed")
    states = torch.arange(38, dtype=torch.float32).reshape(1, 19, 2)
    result = module(states)
    assert result.diagnostics.boundaries.nonzero().tolist() == [[0, 7], [0, 15]]
    assert result.diagnostics.durations[0, [7, 15]].tolist() == [8, 8]
    assert result.diagnostics.segment_counts.tolist() == [2]
    assert result.diagnostics.pending_lengths.tolist() == [3]
    torch.testing.assert_close(result.states[:, :7], states[:, :7])
    assert result.diagnostics.macro_updates == 2
    assert result.diagnostics.macro_candidate_evaluations == 19
    assert result.diagnostics.boundary_entropy.item() == 0.0


def test_closing_observation_influences_only_its_own_and_later_outputs() -> None:
    module = CausalCoarseGrainer(width=2, mode="fixed")
    states = torch.randn(1, 16, 2, requires_grad=True)
    result = module(states)
    earlier = torch.autograd.grad(result.states[:, 6].sum(), states, retain_graph=True)[0]
    closing = torch.autograd.grad(result.states[:, 7].sum(), states)[0]
    assert earlier[:, 7:].count_nonzero() == 0
    assert closing[:, :7].abs().sum() > 0
    assert closing[:, 8:].count_nonzero() == 0


def test_macro_consumes_completed_segment_mean() -> None:
    module = CausalCoarseGrainer(width=1, mode="fixed", window=2)
    with torch.no_grad():
        module.macro.weight.copy_(torch.tensor([[1.0, 0.0]]))
        module.macro.bias.zero_()
        module.readout.weight.fill_(1.0)
    states = torch.tensor([[[1.0], [3.0], [5.0]]])
    result = module(states)
    expected = torch.tensor(
        [[[1.0], [3.0 + torch.tanh(torch.tensor(2.0))], [5.0 + torch.tanh(torch.tensor(2.0))]]]
    )
    torch.testing.assert_close(result.states, expected)


def test_all_modes_use_identical_parameter_shapes_and_identity_equals_always() -> None:
    reference = CausalCoarseGrainer(width=5, mode="identity")
    shapes = {name: value.shape for name, value in reference.state_dict().items()}
    for mode in (*MODES, "whitespace", "punctuation"):
        module = CausalCoarseGrainer(width=5, mode=mode)
        assert {name: value.shape for name, value in module.state_dict().items()} == shapes
    always = CausalCoarseGrainer(width=5, mode="always")
    always.load_state_dict(reference.state_dict())
    states = torch.randn(2, 4, 5)
    torch.testing.assert_close(reference(states).states, always(states).states)


def test_learned_boundary_has_nonzero_task_and_rate_gradients() -> None:
    module = CausalCoarseGrainer(width=3, mode="learned")
    states = torch.randn(2, 12, 3)
    result = module(states)
    task_grad = torch.autograd.grad(
        result.states.square().sum(), module.router.weight, retain_graph=True
    )[0]
    rate_grad = torch.autograd.grad(result.rate_penalty, module.router.bias)[0]
    assert task_grad.abs().sum() > 0
    assert rate_grad.abs().sum() > 0
    assert not result.diagnostics.boundaries.requires_grad
    assert result.diagnostics.probabilities.requires_grad


@pytest.mark.parametrize("mode, expected", [("whitespace", [1, 3]), ("punctuation", [2, 4])])
def test_codec_heuristics_are_explicit_causal_controls(mode: str, expected: list[int]) -> None:
    module = CausalCoarseGrainer(width=3, mode=mode)
    ids = torch.tensor([[65, 32, 46, 10, 33]])
    states = torch.randn(1, 5, 3)
    result = module(states, byte_ids=ids)
    assert result.diagnostics.boundaries[0].nonzero().flatten().tolist() == expected
    prefix = module(states[:, :3], byte_ids=ids[:, :3])
    torch.testing.assert_close(result.states[:, :3], prefix.states, atol=0, rtol=0)
    with pytest.raises(ValueError, match="byte_ids"):
        module(states)


def test_learned_never_consumes_byte_identity() -> None:
    with pytest.raises(ValueError, match="heuristic"):
        CausalCoarseGrainer(width=2, mode="learned")(
            torch.zeros(1, 2, 2), byte_ids=torch.zeros(1, 2, dtype=torch.long)
        )


def test_empty_short_and_collapsed_states_are_valid_without_forced_boundary() -> None:
    module = CausalCoarseGrainer(width=2, mode="fixed")
    empty = module(torch.zeros(1, 0, 2))
    assert empty.states.shape == (1, 0, 2)
    assert empty.rate_penalty.item() == 0
    assert empty.diagnostics.macro_updates == 0
    short = module(torch.zeros(1, 3, 2))
    assert short.diagnostics.segment_counts.tolist() == [0]
    assert short.diagnostics.pending_lengths.tolist() == [3]
    assert torch.isfinite(short.states).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"width": 2, "mode": "bad"},
        {"width": 2, "window": 0},
        {"width": 2, "smoothing": 1.0},
    ],
)
def test_invalid_configuration_fails(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CausalCoarseGrainer(**kwargs)  # type: ignore[arg-type]


def test_nonfinite_or_invalid_state_shapes_fail() -> None:
    module = CausalCoarseGrainer(width=2)
    for states in (
        torch.zeros(2, 2),
        torch.zeros(1, 2, 3),
        torch.full((1, 2, 2), float("nan")),
        torch.zeros(0, 2, 2),
    ):
        with pytest.raises(ValueError):
            module(states)
    with pytest.raises(TypeError):
        module(torch.zeros(1, 2, 2, dtype=torch.long))


def test_prefix_guard_detects_future_leaking_mutant(monkeypatch: pytest.MonkeyPatch) -> None:
    original = CausalCoarseGrainer.forward

    def leak(
        self: CausalCoarseGrainer, values: torch.Tensor, *, byte_ids: torch.Tensor | None = None
    ) -> CoarseGrainingResult:
        result = original(self, values, byte_ids=byte_ids)
        return replace(result, states=result.states + values.mean(dim=1, keepdim=True))

    monkeypatch.setattr(CausalCoarseGrainer, "forward", leak)
    with pytest.raises(AssertionError):
        test_prefix_is_unchanged_by_future_and_prefix_truncation("fixed")
