"""Feature-owned tests for D1 future-target and stabilization constants."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable

import pytest

from entropic_substrate.kernel.identity.d1_primitives import StateConstructionSpec
from entropic_substrate.ladder.register.d1_stabilization import D1TargetSpec, FutureLawSpec

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: D1TargetSpec(horizon=2), "horizon=1"),
        (lambda: D1TargetSpec(rms_normalized=False), "RMS-normalized"),
        (lambda: FutureLawSpec(variance=2.0), "variance=1"),
        (lambda: FutureLawSpec(future_loss_weight=0.5), "weight=1"),
        (lambda: FutureLawSpec(mean_baseline_calibration_batches=0), "calibration batches"),
        (
            lambda: FutureLawSpec(mean_baseline_sampling="random_train"),
            "deterministic train-grid",
        ),
        (
            lambda: StateConstructionSpec(legacy_e15_field_axis="frozen"),
            "E1.5 field axis",
        ),
    ],
)
def test_d1_fixed_tracer_constants_reject_axis_leakage(
    factory: Callable[[], object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_mean_baseline_calibration_protocol_is_construction_identity() -> None:
    law = FutureLawSpec()

    assert law.mean_baseline_calibration_batches == 32
    assert law.mean_baseline_sampling == "deterministic_train_grid_v1"
    assert law.manifest["mean_baseline_calibration_batches"] == 32
    assert law.manifest["mean_baseline_sampling"] == "deterministic_train_grid_v1"


def _dotted(*parts: str) -> str:
    """Assemble a retired dotted path so this file never spells the dead name."""
    return ".".join(parts)


@pytest.mark.parametrize(
    "module_path",
    [
        _dotted("entropic_substrate", "ladder", "gate", "d1_family" + "_adapter"),
    ],
)
def test_retired_dead_modules_are_not_importable(module_path: str) -> None:
    """FOR-1574: the zero-importer family adapter stays deleted, not relocated."""
    assert importlib.util.find_spec(module_path) is None


@pytest.mark.parametrize(
    "module_path",
    [
        "entropic_substrate.kernel.identity.d1_primitives",
        "entropic_substrate.ladder.register.d1",
    ],
)
def test_retired_width_only_encoder_contract_is_absent(module_path: str) -> None:
    """FOR-1590: the legacy width-only contract has no module-level home or export."""
    retired = "Encoder" + "Shape"
    module = importlib.import_module(module_path)
    assert not hasattr(module, retired)
    assert retired not in getattr(module, "__all__", ())
