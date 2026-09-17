"""Stable views must not overwrite the magnitude-preserving latent carrier."""

import pytest
import torch

from entropic_substrate.predictive_state.metric import FisherNumericalError
from entropic_substrate.sentence_prediction.numerics import (
    UndefinedDirection,
    apply_interface,
    checked_coordinates,
    coordinate_map,
)

pytestmark = pytest.mark.unit


def test_magnitude_interface_preserves_the_actual_carrier() -> None:
    original = torch.tensor([[1e-30, 2e-30], [1e30, 2e30]])
    saved = original.clone()
    result = apply_interface(original, "magnitude")
    assert result is original
    assert torch.equal(original, saved)


def test_direction_is_stable_without_small_component_floors() -> None:
    values = torch.tensor([[1e-30, 2e-30], [1e30, 2e30]])
    result = checked_coordinates(values)
    torch.testing.assert_close(result.direction[0], result.direction[1])
    torch.testing.assert_close(result.direction.norm(dim=-1), torch.ones(2))
    assert bool(torch.isfinite(result.log_radius).all())
    small = apply_interface(torch.tensor([[1.0, 1e-14]]), "sphere")
    assert float(small[0, 1]) == pytest.approx(1e-14, rel=1e-6, abs=0)


def test_direction_and_magnitude_are_separate_controls() -> None:
    values = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    coordinates = checked_coordinates(values)
    torch.testing.assert_close(coordinates.direction[0], coordinates.direction[1])
    torch.testing.assert_close(coordinates.log_radius.exp(), torch.tensor([5.0, 10.0]))
    assert not torch.equal(apply_interface(values, "magnitude")[0], coordinates.direction[0])


def test_fp32_island_under_bfloat16_autocast() -> None:
    values = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        coordinates = checked_coordinates(values)
        magnitude = apply_interface(values, "magnitude")
    assert coordinates.direction.dtype == coordinates.log_radius.dtype == torch.float32
    assert magnitude.dtype == torch.float32
    torch.testing.assert_close(magnitude, values.float())


def test_zero_direction_is_explicit_not_a_fabricated_point() -> None:
    values = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    with pytest.raises(UndefinedDirection) as failure:
        checked_coordinates(values)
    assert failure.value.code == "undefined_zero_norm"
    assert failure.value.positions == 2
    result = checked_coordinates(values, allow_zero=True)
    assert result.zero_mask.tolist() == [True, False, True]
    assert torch.equal(result.direction[0], torch.zeros(2))
    assert bool(torch.isneginf(result.log_radius[0]))
    assert apply_interface(values, "magnitude") is values


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_state_is_a_numerical_defect(value: float) -> None:
    with pytest.raises(FisherNumericalError, match="nonfinite"):
        checked_coordinates(torch.tensor([[1.0, value]]))


@pytest.mark.parametrize(
    "values",
    [torch.ones(2, dtype=torch.int64), torch.tensor(1.0), torch.ones(0, 3), torch.ones(2, 1)],
)
def test_invalid_state_shape_and_dtype_are_refused(values: torch.Tensor) -> None:
    with pytest.raises(FisherNumericalError):
        checked_coordinates(values)


def test_functional_coordinate_map_supports_vmap_and_derivatives() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 4.0], [2.0, -3.0, 1.0]], dtype=torch.float64, requires_grad=True
    )
    result = torch.vmap(coordinate_map)(values)
    torch.testing.assert_close(result.direction, checked_coordinates(values).direction)
    assert result.direction.dtype == torch.float64

    def mapped(x: torch.Tensor) -> torch.Tensor:
        coordinates = coordinate_map(x)
        return torch.cat((coordinates.direction, coordinates.log_radius.unsqueeze(-1)), dim=-1)

    assert torch.autograd.gradcheck(mapped, (values,))
    assert torch.autograd.gradgradcheck(mapped, (values,))


def test_unknown_interface_is_not_a_spherical_default() -> None:
    with pytest.raises(ValueError, match="interface"):
        apply_interface(torch.ones(3), "unknown")


def test_non_tensor_input_is_not_coerced() -> None:
    with pytest.raises(TypeError, match="tensor"):
        checked_coordinates([1.0, 2.0])


def test_zero_policy_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="boolean"):
        checked_coordinates(torch.ones(3), allow_zero="yes")
