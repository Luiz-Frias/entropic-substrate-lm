"""Feature-owned tests for D1 future-target and stabilization constants."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from entropic_substrate.experiments.d1_primitives import EncoderShape, StateConstructionSpec
from entropic_substrate.experiments.d1_stabilization import D1TargetSpec, FutureLawSpec

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: EncoderShape(online_width=32, target_width=64), "equal widths"),
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
