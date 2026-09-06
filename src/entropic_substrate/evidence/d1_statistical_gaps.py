"""Versioned witnesses for mathematically undefined D1 cosine estimands."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from entropic_substrate.d1_evaluation_values import D1CosineGap, parse_d1_cosine


@dataclass(frozen=True, slots=True)
class D1EndpointGap:
    evaluations: tuple[tuple[int, int, D1CosineGap], ...]

    def __post_init__(self) -> None:
        if type(self.evaluations) is not tuple or not self.evaluations:
            raise ValueError("undefined endpoint requires its evaluation witnesses")
        cursors = []
        for step, index, gap in self.evaluations:
            if (
                type(step) is not int
                or type(index) is not int
                or step < 0
                or index < 0
                or type(gap) is not D1CosineGap
            ):
                raise ValueError("invalid undefined endpoint witness")
            cursors.append((step, index))
        if cursors != sorted(set(cursors)):
            raise ValueError("undefined endpoint witnesses must be unique and ordered")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "d1-endpoint-gap-v1",
            "reason": "undefined_zero_norm",
            "evaluations": [
                {"step": step, "eval_index": index, "gap": gap.manifest}
                for step, index, gap in self.evaluations
            ],
        }


@dataclass(frozen=True, slots=True)
class D1GapCell:
    arm: str
    seed: int
    endpoint: D1EndpointGap

    def __post_init__(self) -> None:
        if (
            self.arm not in {"none", "sg", "ac", "both", "A", "B", "C", "D"}
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.endpoint) is not D1EndpointGap
        ):
            raise ValueError("invalid undefined-estimate cell")


@dataclass(frozen=True, slots=True)
class D1EstimateGap:
    cells: tuple[D1GapCell, ...]

    def __post_init__(self) -> None:
        if (
            type(self.cells) is not tuple
            or not self.cells
            or any(type(cell) is not D1GapCell for cell in self.cells)
        ):
            raise ValueError("undefined estimate requires its typed cell witnesses")
        if len({(cell.arm, cell.seed) for cell in self.cells}) != len(self.cells):
            raise ValueError("undefined estimate repeats a cell witness")

    def manifest(self, aliases: Mapping[str, str] | None = None) -> dict[str, object]:
        key = "arm_id" if aliases is None else "arm_alias"
        return {
            "schema_version": "d1-estimate-gap-v1",
            "reason": "undefined_zero_norm",
            "cells": [
                {
                    key: cell.arm if aliases is None else aliases[cell.arm],
                    "seed": cell.seed,
                    "endpoint": cell.endpoint.manifest,
                }
                for cell in self.cells
            ],
        }


def _object(value: object, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError("undefined statistic requires exact versioned fields")
    return cast(dict[str, object], value)


def parse_endpoint_gap(value: object) -> D1EndpointGap:
    row = _object(value, {"schema_version", "reason", "evaluations"})
    if (
        row["schema_version"] != "d1-endpoint-gap-v1"
        or row["reason"] != "undefined_zero_norm"
        or type(row["evaluations"]) is not list
    ):
        raise ValueError("unknown undefined endpoint schema/reason")
    points = []
    for raw in row["evaluations"]:
        point = _object(raw, {"step", "eval_index", "gap"})
        gap = parse_d1_cosine(point["gap"])
        if (
            type(gap) is not D1CosineGap
            or type(point["step"]) is not int
            or type(point["eval_index"]) is not int
        ):
            raise ValueError("undefined endpoint requires exact cosine-gap observations")
        points.append((point["step"], point["eval_index"], gap))
    return D1EndpointGap(tuple(points))


def parse_estimate_gap(value: object, *, aliases: bool = False) -> D1EstimateGap:
    row = _object(value, {"schema_version", "reason", "cells"})
    if (
        row["schema_version"] != "d1-estimate-gap-v1"
        or row["reason"] != "undefined_zero_norm"
        or type(row["cells"]) is not list
    ):
        raise ValueError("unknown undefined estimate schema/reason")
    key = "arm_alias" if aliases else "arm_id"
    allowed = {"A", "B", "C", "D"} if aliases else {"none", "sg", "ac", "both"}
    cells = []
    for raw in row["cells"]:
        cell = _object(raw, {key, "seed", "endpoint"})
        if type(cell[key]) is not str or cell[key] not in allowed or type(cell["seed"]) is not int:
            raise ValueError("undefined estimate cell identity differs from schema")
        cells.append(
            D1GapCell(cast(str, cell[key]), cell["seed"], parse_endpoint_gap(cell["endpoint"]))
        )
    return D1EstimateGap(tuple(cells))
