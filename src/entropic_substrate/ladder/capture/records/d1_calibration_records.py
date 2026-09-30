"""Record half of the D1 calibration split (torch half:
``__legacy/d1_calibration.py``).

RED-first split (ruling R-parallel; contract C17-p5): the immutable
attestation type lives here, dataclass + ``require_sha256_hex_any_case`` only, no torch/
numpy import. The torch half imports and RE-EXPORTS this name (never
redefines it) for the out-of-envelope importers still on the old dotted
name; p8/p9 rewrite those, never this phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from entropic_substrate.kernel.identity.d1_primitives import require_sha256_hex_any_case

__all__ = ["D1BaselineCalibrationAttestation"]


@dataclass(frozen=True, slots=True)
class D1BaselineCalibrationAttestation:
    """Identity and accounted work of one successful calibration transition."""

    protocol_digest: str
    spec_digest: str
    split_digest: str
    source_order_digest: str
    mean_parameter_digest: str
    codec_digest: str
    initialization_seed: int
    num_batches: int
    online_positions: int
    source_span_positions: int
    successful_step_at_fit: int = 0
    schema_version: str = "d1-baseline-calibration-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_digest", self.protocol_digest),
            ("spec_digest", self.spec_digest),
            ("split_digest", self.split_digest),
            ("source_order_digest", self.source_order_digest),
            ("mean_parameter_digest", self.mean_parameter_digest),
            ("codec_digest", self.codec_digest),
        ):
            require_sha256_hex_any_case(value, label=label)
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            msg = "calibration initialization_seed must be nonnegative"
            raise ValueError(msg)
        for label, value in (
            ("num_batches", self.num_batches),
            ("online_positions", self.online_positions),
            ("source_span_positions", self.source_span_positions),
        ):
            if type(value) is not int or value < 1:
                msg = f"calibration {label} must be a positive integer"
                raise ValueError(msg)
        if self.successful_step_at_fit != 0:
            msg = "D1 calibration attestation must be fitted at step zero"
            raise ValueError(msg)
        if self.schema_version != "d1-baseline-calibration-v1":
            msg = "unsupported D1 baseline calibration schema"
            raise ValueError(msg)

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "codec_digest": self.codec_digest,
            "initialization_seed": self.initialization_seed,
            "mean_parameter_digest": self.mean_parameter_digest,
            "num_batches": self.num_batches,
            "online_positions": self.online_positions,
            "protocol_digest": self.protocol_digest,
            "schema_version": self.schema_version,
            "source_order_digest": self.source_order_digest,
            "source_span_positions": self.source_span_positions,
            "spec_digest": self.spec_digest,
            "split_digest": self.split_digest,
            "successful_step_at_fit": self.successful_step_at_fit,
        }
