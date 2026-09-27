"""Bounded selection over explicitly supplied D3 parameter-matching candidates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

from entropic_substrate.ladder.register.ladder import MatchingLane, Rung, RungSpec
from entropic_substrate.ladder.run.construction.ladder_runtime import LadderPredictiveSystem


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    total: int
    trainable: int
    frozen: int


def measured_inventory(runtime: LadderPredictiveSystem) -> ParameterInventory:
    """Read counts from instantiated unique parameter tensors, including teachers."""
    trainable = sum(p.numel() for p in runtime.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in runtime.parameters() if not p.requires_grad)
    return ParameterInventory(trainable + frozen, trainable, frozen)


@dataclass(frozen=True, slots=True)
class CandidateMeasurement:
    construction_identity: str
    inventory: ParameterInventory
    relative_gap: float


@dataclass(frozen=True, slots=True)
class MatchedRuntime:
    runtime: LadderPredictiveSystem
    inventory: ParameterInventory
    relative_gap: float
    candidates: tuple[CandidateMeasurement, ...]


@dataclass(frozen=True, slots=True)
class NoAdmissibleMatch:
    """A measured gap, never permission to widen the registered tolerance."""

    reference: ParameterInventory
    candidates: tuple[CandidateMeasurement, ...]
    reason: str = "no_candidate_within_parameter_budget"


def select_total_match(
    reference: LadderPredictiveSystem, candidates: Iterable[RungSpec]
) -> MatchedRuntime | NoAdmissibleMatch:
    """Choose the largest measured total within the reference and declared gap.

    No shape is invented and no runtime is launched. The caller supplies legal
    candidate shapes and the reference instance, already construction-bound.
    Equal totals break ties by construction digest, independently of loss.
    """
    anchor = measured_inventory(reference)
    roster = tuple(candidates)
    if not roster:
        raise ValueError("parameter matching requires explicit candidate constructions")
    first = roster[0]
    measurements: list[CandidateMeasurement] = []
    selected: tuple[LadderPredictiveSystem, CandidateMeasurement] | None = None
    for spec in roster:
        declared = spec.parameter_reference
        if spec.rung is not Rung.D3 or spec.matching_lane is not MatchingLane.MATCHED_TOTAL:
            raise ValueError("parameter matching requires D3 matched-total candidates")
        if (
            declared is None
            or declared.construction_identity != reference.construction_identity
            or declared.total_parameters != anchor.total
        ):
            raise ValueError("candidate anchor disagrees with measured reference")
        if replace(spec, base=first.base) != first:
            raise ValueError("matching candidates may vary only encoder shape")
        runtime = LadderPredictiveSystem(spec)
        inventory = measured_inventory(runtime)
        gap = (anchor.total - inventory.total) / anchor.total
        measured = CandidateMeasurement(spec.digest, inventory, gap)
        measurements.append(measured)
        if 0 <= gap <= declared.max_relative_gap and (
            selected is None
            or (inventory.total, spec.digest)
            > (selected[1].inventory.total, selected[1].construction_identity)
        ):
            selected = runtime, measured
    if selected is None:
        return NoAdmissibleMatch(anchor, tuple(measurements))
    return MatchedRuntime(
        selected[0], selected[1].inventory, selected[1].relative_gap, tuple(measurements)
    )
