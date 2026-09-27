"""Sentence-clock carrier modes share sublayers, not a hidden final retraction."""

from dataclasses import replace

import pytest
import torch

from entropic_substrate.core.sentence.spine import SentenceSpine, SpineSpec

pytestmark = pytest.mark.unit


def spine(carrier: str = "sphere") -> SentenceSpine:
    torch.manual_seed(42)
    return SentenceSpine(
        8, SpineSpec(width=16, layers=2, head_width=4, context_sentences=8, carrier=carrier)
    )


def test_spine_advances_on_sentences_and_has_a_bos_state() -> None:
    model = spine()
    result = model(torch.randn(2, 3, 8))
    assert result.shape == (2, 4, 16)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(2, 4), atol=1e-6, rtol=1e-6)
    assert model(torch.empty(2, 0, 8)).shape == (2, 1, 16)


@pytest.mark.parametrize("carrier", ["sphere", "magnitude"])
def test_future_sentences_cannot_change_prior_states(carrier: str) -> None:
    model = spine(carrier).eval()
    original = torch.randn(1, 4, 8)
    changed = original.clone()
    changed[:, 2:] += 10
    before, after = model(original), model(changed)
    torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(before[:, 3:], after[:, 3:])


def test_magnitude_carrier_is_not_overwritten_by_final_normalization() -> None:
    model = spine("magnitude")
    with torch.no_grad():
        for block in model.blocks:
            block.attn_alpha.zero_()
            block.mlp_alpha.zero_()
    source = torch.randn(1, 2, 8)
    expected = model.initial_states(source)
    actual = model(source)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.allclose(actual.norm(dim=-1), torch.ones(1, 3))


def test_carrier_variants_share_parameter_shapes_and_initialization() -> None:
    unit, raw = spine("sphere"), spine("magnitude")
    assert unit.state_dict().keys() == raw.state_dict().keys()
    for name, value in unit.state_dict().items():
        torch.testing.assert_close(value, raw.state_dict()[name], atol=0, rtol=0)


def test_intervention_is_only_at_the_final_block_and_isotropic_is_exact() -> None:
    model = spine()
    values = torch.randn(1, 3, 8)
    calls: list[tuple[torch.Size, torch.Size]] = []

    def identity(current: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        calls.append((current.shape, proposed.shape))
        return proposed

    expected = model(values)
    torch.testing.assert_close(model(values, consumer=identity), expected, atol=0, rtol=0)
    assert calls == [(torch.Size([1, 4, 16]), torch.Size([1, 4, 16]))]


def test_weights_are_normalized_without_changing_carrier_policy() -> None:
    model = spine("magnitude")
    model.normalize_weights()
    torch.testing.assert_close(
        model.projection.weight.norm(dim=-1), torch.ones(16), atol=1e-6, rtol=1e-6
    )
    assert model.spec.carrier == "magnitude"


def test_geometry_views_promote_under_autocast() -> None:
    model = spine()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(torch.ones(1, 2, 8))
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(1, 3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "changes",
    [
        {"carrier": "unknown"},
        {"layers": 0},
        {"head_width": 3},
        {"alpha_init": float("nan")},
        {"context_sentences": True},
    ],
)
def test_invalid_spine_spec_is_refused(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(spine().spec, **changes)


def test_context_limit_and_bad_consumer_are_not_silently_repaired() -> None:
    model = spine()
    with pytest.raises(ValueError, match="context"):
        model(torch.ones(1, 9, 8))
    with pytest.raises(ValueError, match="consumer"):
        model(torch.ones(1, 2, 8), consumer=lambda current, proposed: proposed[:, :1])


def test_unit_consumer_cannot_return_nonunit_states() -> None:
    with pytest.raises(ValueError, match="unit"):
        spine()(torch.ones(1, 2, 8), consumer=lambda current, proposed: 2 * proposed)


def test_spine_rejects_non_tensor_input_explicitly() -> None:
    with pytest.raises(TypeError, match="tensor"):
        spine().initial_states([1, 2])
