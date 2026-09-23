"""Successful-commit EMA decay schedules for D1 target updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["CosineEmaDecaySchedule", "EmaDecaySchedule", "FixedEmaDecaySchedule"]


def _validate_decay(name: str, value: float) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value < 1.0:
        msg = f"{name} must be a finite float in [0, 1), got {value!r}"
        raise ValueError(msg)
    return value


def _validate_successful_commit(step: int, *, upper_bound: int | None = None) -> int:
    if type(step) is not int or step < 1:
        msg = f"successful commit step {step!r} is outside the schedule domain [1, inf)"
        raise ValueError(msg)
    if upper_bound is not None and step > upper_bound:
        msg = f"successful commit step {step} is outside the schedule domain [1, {upper_bound}]"
        raise ValueError(msg)
    return step


@runtime_checkable
class EmaDecaySchedule(Protocol):
    """Serializable decay policy indexed only by successful optimizer commits."""

    def decay_for_successful_commit(self, step: int) -> float: ...

    def state_dict(self) -> dict[str, int | float | str]: ...


@dataclass(frozen=True, slots=True)
class FixedEmaDecaySchedule:
    """Compatibility schedule with one constant teacher decay."""

    tau: float

    def __post_init__(self) -> None:
        _validate_decay("tau", self.tau)

    def decay_for_successful_commit(self, step: int) -> float:
        """Return the same validated decay at every successful commit."""
        _validate_successful_commit(step)
        return self.tau

    def state_dict(self) -> dict[str, int | float | str]:
        """Return the complete fixed schedule identity."""
        return {"kind": "fixed", "tau": self.tau}


@dataclass(frozen=True, slots=True)
class CosineEmaDecaySchedule:
    """Monotone cosine decay between first and final successful commits."""

    start_decay: float
    final_decay: float
    total_successful_commits: int

    def __post_init__(self) -> None:
        _validate_decay("start_decay", self.start_decay)
        _validate_decay("final_decay", self.final_decay)
        if self.final_decay < self.start_decay:
            msg = "cosine EMA final_decay must be at least start_decay so the schedule is monotonic"
            raise ValueError(msg)
        if type(self.total_successful_commits) is not int or self.total_successful_commits < 1:
            msg = f"total_successful_commits must be a positive integer, got {self.total_successful_commits!r}"
            raise ValueError(msg)
        if self.total_successful_commits == 1 and self.start_decay != self.final_decay:
            msg = "a one-commit cosine EMA schedule requires identical start and final decay"
            raise ValueError(msg)

    def decay_for_successful_commit(self, step: int) -> float:
        """Return the decay at ``step`` in the registered successful-commit domain."""
        _validate_successful_commit(step, upper_bound=self.total_successful_commits)
        if self.total_successful_commits == 1:
            return self.start_decay
        progress = (step - 1) / (self.total_successful_commits - 1)
        return (
            self.start_decay
            + (self.final_decay - self.start_decay) * (1.0 - math.cos(math.pi * progress)) / 2.0
        )

    def state_dict(self) -> dict[str, int | float | str]:
        """Return all schedule constants required to recreate this curve."""
        return {
            "kind": "cosine",
            "start_decay": self.start_decay,
            "final_decay": self.final_decay,
            "total_successful_commits": self.total_successful_commits,
        }


def schedule_from_state(state: object) -> EmaDecaySchedule:
    """Decode one complete, validated persisted schedule identity."""
    if not isinstance(state, dict):
        msg = "EMA schedule state must be a dictionary"
        raise TypeError(msg)
    kind = state.get("kind")
    if kind == "fixed":
        tau = state.get("tau")
        if not isinstance(tau, float):
            msg = f"fixed EMA schedule has invalid tau {tau!r}"
            raise ValueError(msg)
        return FixedEmaDecaySchedule(tau=tau)
    if kind == "cosine":
        start_decay = state.get("start_decay")
        final_decay = state.get("final_decay")
        total_successful_commits = state.get("total_successful_commits")
        if not isinstance(start_decay, float) or not isinstance(final_decay, float):
            msg = "cosine EMA schedule has invalid decay endpoints"
            raise ValueError(msg)
        if type(total_successful_commits) is not int:
            msg = "cosine EMA schedule has invalid total_successful_commits"
            raise ValueError(msg)
        return CosineEmaDecaySchedule(
            start_decay=start_decay,
            final_decay=final_decay,
            total_successful_commits=total_successful_commits,
        )
    msg = f"unknown EMA decay schedule kind {kind!r}"
    raise ValueError(msg)
