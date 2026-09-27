"""Target-update policies and post-commit teacher synchronization for D1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from entropic_substrate.core.latent.ema_schedules import (
    CosineEmaDecaySchedule,
    EmaDecaySchedule,
    FixedEmaDecaySchedule,
    schedule_from_state,
)
from entropic_substrate.core.latent.target_values import FutureTarget, StabilizedTargets

__all__ = [
    "OnlineTargetUpdate",
    "OptimizationStepCommitted",
    "StopGradientEmaUpdate",
    "TargetUpdatePolicy",
]


@dataclass(frozen=True, slots=True)
class OptimizationStepCommitted:
    """Value event emitted only after a successful online optimizer update."""

    step: int

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 1:
            msg = f"committed optimizer step must be a positive integer, got {self.step!r}"
            raise ValueError(msg)


@runtime_checkable
class TargetUpdatePolicy(Protocol):
    """Strategy boundary between target construction and online optimization."""

    def configure_target_encoder(self, target: nn.Module) -> None: ...

    def run_target_branch(
        self, target_forward: Callable[[], FutureTarget]
    ) -> StabilizedTargets: ...

    def stabilize(self, target: FutureTarget) -> StabilizedTargets: ...

    def after_optimization_commit(
        self, online: nn.Module, target: nn.Module, event: OptimizationStepCommitted
    ) -> None: ...


class OnlineTargetUpdate:
    """Control strategy: target encoder trains normally and targets are attached."""

    def configure_target_encoder(self, target: nn.Module) -> None:
        target.requires_grad_(True)

    def run_target_branch(self, target_forward: Callable[[], FutureTarget]) -> StabilizedTargets:
        """Run the complete trainable target branch under normal autograd."""
        return self.stabilize(target_forward())

    def stabilize(self, target: FutureTarget) -> StabilizedTargets:
        if type(target) is not FutureTarget:
            msg = "target-update policy requires a FutureTarget from its target branch"
            raise TypeError(msg)
        return StabilizedTargets(
            values=target.values, codec_identity=target.codec_identity, detached=False
        )

    def after_optimization_commit(
        self, online: nn.Module, target: nn.Module, event: OptimizationStepCommitted
    ) -> None:
        """The online-update control has no teacher-copy transition."""
        del online, target, event

    def state_dict(self) -> dict[str, int | str]:
        """Persist the identity of this stateless control policy."""
        return {"kind": "online", "version": 1}


def _parameter_map(module: nn.Module) -> dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def _validated_parameter_maps(
    online: nn.Module, target: nn.Module
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    online_map = _parameter_map(online)
    target_map = _parameter_map(target)
    if online_map.keys() != target_map.keys():
        msg = "online and target parameter-name maps differ"
        raise ValueError(msg)
    for name, online_parameter in online_map.items():
        target_parameter = target_map[name]
        if online_parameter.shape != target_parameter.shape:
            msg = f"online and target parameter shape mismatch for {name!r}"
            raise ValueError(msg)
        if online_parameter.dtype != target_parameter.dtype:
            msg = f"online and target parameter dtype mismatch for {name!r}"
            raise ValueError(msg)
        if online_parameter.device != target_parameter.device:
            msg = f"online and target parameter device mismatch for {name!r}"
            raise ValueError(msg)
        if online_parameter.data_ptr() == target_parameter.data_ptr():
            msg = f"online and target parameter storage is shared for {name!r}"
            raise ValueError(msg)
    return online_map, target_map


class StopGradientEmaUpdate:
    """Detached target policy with post-commit EMA teacher synchronization."""

    _VERSION = 2

    def __init__(self, *, tau: float | None = None, schedule: object | None = None) -> None:
        if tau is not None and schedule is not None:
            msg = "provide either fixed tau or an EMA decay schedule, not both"
            raise ValueError(msg)
        if tau is not None:
            self.schedule: EmaDecaySchedule = FixedEmaDecaySchedule(tau=tau)
        elif schedule is not None:
            if not isinstance(schedule, EmaDecaySchedule):
                msg = "schedule must implement EmaDecaySchedule"
                raise TypeError(msg)
            self.schedule = schedule
        else:
            msg = "stop-gradient EMA requires fixed tau or an EMA decay schedule"
            raise ValueError(msg)
        self.update_cursor = 0

    @property
    def tau(self) -> float:
        """Fixed-decay compatibility accessor; cosine schedules have no one tau."""
        if not isinstance(self.schedule, FixedEmaDecaySchedule):
            msg = "cosine EMA schedules do not expose one fixed tau"
            raise AttributeError(msg)
        return self.schedule.tau

    def configure_target_encoder(self, target: nn.Module) -> None:
        """Freeze every teacher-branch parameter before an optimizer can own it."""
        target.requires_grad_(False)

    def run_target_branch(self, target_forward: Callable[[], FutureTarget]) -> StabilizedTargets:
        """Run the entire teacher branch under no-grad before stabilization."""
        with torch.no_grad():
            target = target_forward()
        return self.stabilize(target)

    def stabilize(self, target: FutureTarget) -> StabilizedTargets:
        """Detach the target tensor at the strategy boundary, not at scoring."""
        if type(target) is not FutureTarget:
            msg = "target-update policy requires a FutureTarget from its target branch"
            raise TypeError(msg)
        return StabilizedTargets(
            values=target.values.detach(), codec_identity=target.codec_identity, detached=True
        )

    def validate_optimizer_excludes_target(
        self, optimizer: torch.optim.Optimizer, target: nn.Module
    ) -> None:
        """Fail loudly if a frozen teacher parameter enters online optimization."""
        target_ids = {id(parameter) for parameter in target.parameters()}
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
            if isinstance(parameter, nn.Parameter)
        }
        if target_ids & optimizer_ids:
            msg = "target encoder parameters must be absent from the online optimizer"
            raise ValueError(msg)

    def after_optimization_commit(
        self, online: nn.Module, target: nn.Module, event: OptimizationStepCommitted
    ) -> None:
        """EMA-copy teacher weights after, and only after, a committed step."""
        expected_step = self.update_cursor + 1
        if event.step != expected_step:
            msg = (
                "committed target EMA steps must be monotonic and contiguous: "
                f"expected {expected_step}, got {event.step}"
            )
            raise ValueError(msg)
        decay = self.schedule.decay_for_successful_commit(event.step)
        online_map, target_map = _validated_parameter_maps(online, target)
        with torch.no_grad():
            for name, target_parameter in target_map.items():
                online_parameter = online_map[name]
                target_parameter.copy_(target_parameter * decay + online_parameter * (1.0 - decay))
        self.update_cursor = event.step

    def state_dict(self) -> dict[str, object]:
        """Small checkpoint payload for the strategy's identity and cursor."""
        return {
            "kind": "stop_gradient_ema",
            "version": self._VERSION,
            "schedule": self.schedule.state_dict(),
            "update_cursor": self.update_cursor,
        }

    def load_state_dict(self, state: object) -> None:
        """Restore only a complete, compatible strategy state."""
        if not isinstance(state, dict):
            msg = "target-update state must be a dictionary"
            raise TypeError(msg)
        if state.get("kind") != "stop_gradient_ema":
            msg = "target-update state is not a compatible stop-gradient EMA payload"
            raise ValueError(msg)
        version = state.get("version")
        cursor = state.get("update_cursor")
        if type(cursor) is not int or cursor < 0:
            msg = f"target-update state has invalid update_cursor {cursor!r}"
            raise ValueError(msg)
        if version == 1:
            tau = state.get("tau")
            if not isinstance(tau, float):
                msg = f"legacy target-update state has invalid tau {tau!r}"
                raise ValueError(msg)
            schedule: EmaDecaySchedule = FixedEmaDecaySchedule(tau=tau)
        elif version == self._VERSION:
            schedule = schedule_from_state(state.get("schedule"))
        else:
            msg = f"target-update state has unsupported version {version!r}"
            raise ValueError(msg)
        if (
            isinstance(schedule, CosineEmaDecaySchedule)
            and cursor > schedule.total_successful_commits
        ):
            msg = "target-update cursor exceeds the persisted cosine schedule domain"
            raise ValueError(msg)
        self.schedule = schedule
        self.update_cursor = cursor
