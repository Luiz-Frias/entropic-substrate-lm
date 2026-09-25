"""Scientific run requests bind mechanisms to admitted design and source bytes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from entropic_substrate.core.corpus.batching import TokenSplit
from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.ladder.gate.ladder_campaign import RunAuthorization
from entropic_substrate.ladder.register.d1_execution import D1ExecutionProtocol
from entropic_substrate.ladder.register.ladder import RungSpec


@dataclass(frozen=True, slots=True)
class ScientificRequest:
    """Only an admitted design cell can select pilot or confirmatory execution."""

    spec: RungSpec
    authorization: RunAuthorization
    source_root: Path
    source_relative_path: str

    def __post_init__(self) -> None:
        if type(self.spec) is not RungSpec or type(self.authorization) is not RunAuthorization:
            raise TypeError("scientific execution requires RungSpec and RunAuthorization")
        auth = self.authorization
        if auth.spec_digest != self.spec.digest or auth.seed != self.spec.seed:
            raise ValueError("authorization does not bind this seeded construction")
        if auth.construction_digest != self.spec.scientific_digest:
            raise ValueError("authorization statistical arm differs from scientific construction")
        if auth.rung != self.spec.rung.value:
            raise ValueError("authorization rung differs from construction")
        if auth.rung == "d4":
            if auth.d4_phase not in {"fixed", "learned"}:
                raise ValueError("D4 requires its explicit scientific phase")
            if (
                self.spec.coarse is not None
                and self.spec.coarse.mode == "learned"
                and auth.d4_phase != "learned"
            ):
                raise ValueError("learned D4 routing requires the filed fixed-phase successor gate")
        elif auth.d4_phase is not None:
            raise ValueError("D4 phase cannot be attached to another rung")
        if auth.execution_protocol_digest != self.protocol.protocol_digest:
            raise ValueError("authorization execution protocol differs from construction")
        if auth.run_class not in {"pilot", "confirmatory"}:
            raise ValueError("scientific requests require pilot or confirmatory classification")
        split = auth.source_split
        if set(split) != {
            "train_start",
            "train_end",
            "eval_start",
            "eval_end",
            "source_total_bytes",
        }:
            raise ValueError("source split must declare every interval and total length")
        if any(type(value) is not int for value in split.values()):
            raise TypeError("source split coordinates must be integers")
        if not (
            0
            <= split["train_start"]
            < split["train_end"]
            <= split["eval_start"]
            < split["eval_end"]
            <= split["source_total_bytes"]
        ):
            raise ValueError(
                "source split must contain ordered disjoint train/evaluation intervals"
            )
        if canonical_digest(split) != auth.split_digest:
            raise ValueError("source split digest mismatch")
        needed = self.protocol.batching.sequence_length + max(self.spec.horizons)
        if (
            min(split["train_end"] - split["train_start"], split["eval_end"] - split["eval_start"])
            < needed
        ):
            raise ValueError("source split is shorter than the registered predictive span")

    @property
    def protocol(self) -> D1ExecutionProtocol:
        return self.spec.base.execution_protocol

    @property
    def max_steps(self) -> int:
        return self.protocol.optimization.max_steps

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "d-ladder-scientific-run-v1",
            "run_class": self.authorization.run_class,
            "authorization": self.authorization.manifest,
            "construction": self.spec.manifest,
            "source_sha256": self.authorization.source_sha256,
            "source_split": dict(self.authorization.source_split),
            "execution": self.protocol.manifest,
            "device": self.authorization.device,
            "reader": self.authorization.reader_identity,
        }

    def load_splits(self) -> tuple[TokenSplit, TokenSplit]:
        """Read and hash the same immutable byte snapshot used by the samplers."""
        split = self.authorization.source_split
        _, content, digest = read_d1_artifact_bytes(
            self.source_root,
            self.source_relative_path,
            label="registered corpus",
            max_bytes=split["source_total_bytes"],
        )
        if (
            digest != self.authorization.source_sha256
            or len(content) != split["source_total_bytes"]
        ):
            raise ValueError("registered corpus content digest or length mismatch")
        values = np.frombuffer(content, dtype=np.uint8)
        return (
            TokenSplit(values[split["train_start"] : split["train_end"]]),
            TokenSplit(values[split["eval_start"] : split["eval_end"]]),
        )
