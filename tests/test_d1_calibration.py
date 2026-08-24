"""One-shot, source-attested D1 mean-baseline calibration."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from d1_runtime_fixtures import runtime_system
from entropic_substrate.d1_calibration import D1BaselineCalibrator
from entropic_substrate.data.batching import SequenceSampler, TokenSplit
from entropic_substrate.experiments.d1_primitives import D1Arm, canonical_digest

pytestmark = pytest.mark.unit


def _split() -> TokenSplit:
    return TokenSplit(np.arange(1024, dtype=np.int32) % 11)


def test_calibration_fits_once_from_a_dedicated_deterministic_training_grid() -> None:
    system = runtime_system(D1Arm.BOTH)
    protocol = system.draft.execution_protocol
    training_sampler = SequenceSampler(
        _split(),
        batch_size=protocol.batching.batch_size,
        seq_len=protocol.batching.sequence_length,
        train=True,
        seed=7,
    )
    sampler_before = training_sampler.state_dict()

    attestation = D1BaselineCalibrator().calibrate(
        system=system,
        split=_split(),
        successful_step=0,
    )

    assert system.mean_displacement_baseline.is_fitted
    assert attestation.spec_digest == canonical_digest(protocol.baseline.manifest)
    assert attestation.num_batches == 32
    assert attestation.online_positions == 32 * 16 * 8
    assert attestation.source_span_positions == 32 * 16 * 9
    assert len(attestation.source_order_digest) == 64
    assert len(attestation.mean_parameter_digest) == 64
    assert attestation.manifest == {
        "codec_digest": attestation.codec_digest,
        "initialization_seed": 7,
        "mean_parameter_digest": attestation.mean_parameter_digest,
        "num_batches": 32,
        "online_positions": 32 * 16 * 8,
        "protocol_digest": attestation.protocol_digest,
        "schema_version": "d1-baseline-calibration-v1",
        "source_order_digest": attestation.source_order_digest,
        "source_span_positions": 32 * 16 * 9,
        "spec_digest": attestation.spec_digest,
        "split_digest": attestation.split_digest,
        "successful_step_at_fit": 0,
    }
    sampler_after = training_sampler.state_dict()
    assert torch.equal(sampler_before["generator"], sampler_after["generator"])
    assert sampler_before["eval_cursor"] == sampler_after["eval_cursor"]


def test_calibration_is_repeatable_for_the_same_seed_and_source() -> None:
    first = runtime_system(D1Arm.SG, seed=19)
    second = runtime_system(D1Arm.SG, seed=19)

    first_attestation = D1BaselineCalibrator().calibrate(
        system=first, split=_split(), successful_step=0
    )
    second_attestation = D1BaselineCalibrator().calibrate(
        system=second, split=_split(), successful_step=0
    )

    assert first_attestation == second_attestation
    torch.testing.assert_close(
        first.mean_displacement_baseline.mean,
        second.mean_displacement_baseline.mean,
        atol=0.0,
        rtol=0.0,
    )


def test_calibration_attestation_changes_with_the_paired_initialization_seed() -> None:
    first = runtime_system(D1Arm.NONE, seed=1)
    second = runtime_system(D1Arm.NONE, seed=2)

    first_attestation = D1BaselineCalibrator().calibrate(
        system=first, split=_split(), successful_step=0
    )
    second_attestation = D1BaselineCalibrator().calibrate(
        system=second, split=_split(), successful_step=0
    )

    assert first_attestation.source_order_digest == second_attestation.source_order_digest
    assert first_attestation.mean_parameter_digest != second_attestation.mean_parameter_digest


def test_calibration_refuses_post_optimization_or_refitting() -> None:
    system = runtime_system(D1Arm.AC)
    calibrator = D1BaselineCalibrator()
    with pytest.raises(ValueError, match="step zero"):
        calibrator.calibrate(system=system, split=_split(), successful_step=1)

    calibrator.calibrate(system=system, split=_split(), successful_step=0)
    with pytest.raises(RuntimeError, match="already fitted"):
        calibrator.calibrate(system=system, split=_split(), successful_step=0)


def test_calibration_refuses_the_wrong_split_or_runtime_type() -> None:
    system = runtime_system(D1Arm.NONE)
    short = TokenSplit(np.arange(4, dtype=np.int32))
    with pytest.raises(ValueError, match="need at least"):
        D1BaselineCalibrator().calibrate(system=system, split=short, successful_step=0)
    with pytest.raises(TypeError, match="D1PredictiveSystem"):
        D1BaselineCalibrator().calibrate(  # type: ignore[arg-type]
            system=object(), split=_split(), successful_step=0
        )
