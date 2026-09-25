"""Reuse production resource activation and cumulative accounting across rungs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Self, cast

from entropic_substrate.kernel.evidence_class.resource_provenance import RunStartProvenance
from entropic_substrate.ladder.capture.resource.resource_ledger import ResourceLedger
from entropic_substrate.ladder.capture.resource.resource_ledger_codec import (
    resource_ledger_state_from_mapping,
    resource_ledger_state_to_mapping,
)
from entropic_substrate.ladder.capture.resource.resource_meter_models import ResourcePhase
from entropic_substrate.ladder.capture.resource.resource_meter_runtime import SystemMonotonicClock
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    resource_snapshot_to_mapping,
)
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ResourceActivation,
    D1ResourceActivationFactory,
)


class LadderResources:
    """One physical segment; the factory and provenance must explicitly agree."""

    def __init__(self, start: RunStartProvenance, factory: D1ResourceActivationFactory) -> None:
        if type(start) is not RunStartProvenance or not isinstance(
            cast(object, factory), D1ResourceActivationFactory
        ):
            raise TypeError("resources require production provenance and activation factory")
        if start.resource_activation != factory.registration:
            raise ValueError("resource activation registration differs from runtime provenance")
        self.start, self.factory = start, factory
        self._activation: D1ResourceActivation | None = None
        self._ledger: ResourceLedger | None = None
        self._clock = SystemMonotonicClock()
        self._entered = False

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("resource segment is already active")
        self._entered = True
        return self

    def __exit__(self, *args: object) -> None:
        try:
            if self._activation is not None:
                self._activation.__exit__(*args)
        finally:
            self._entered = False
            self._activation = None
            self._ledger = None

    def activate(self, saved_state: object | None) -> None:
        """Restore both meter and ledger, or begin the admitted fresh segment."""
        if not self._entered or self._activation is not None:
            raise RuntimeError("resource segment must be active and not initialized")
        state = None if saved_state is None else resource_ledger_state_from_mapping(saved_state)
        self._activation = (
            self.factory.fresh(self.start)
            if state is None
            else self.factory.resume(self.start, state)
        )
        self._activation.__enter__()
        origin = self._clock.monotonic_ns()
        self._ledger = (
            ResourceLedger(
                self.start, monotonic_origin_ns=origin, meter_strategy=self._activation.meter
            )
            if state is None
            else ResourceLedger.restore(
                self.start, state, monotonic_origin_ns=origin, meter_strategy=self._activation.meter
            )
        )

    def _active(self) -> tuple[D1ResourceActivation, ResourceLedger]:
        if self._activation is None or self._ledger is None or not self._entered:
            raise RuntimeError("resource segment is not active")
        return self._activation, self._ledger

    def measure(self, phase: ResourcePhase) -> AbstractContextManager[None]:
        activation, _ = self._active()
        return activation.meter.measure(phase)

    def baseline(self, target: int) -> None:
        activation, ledger = self._active()
        activation.meter.record_positions(
            ResourcePhase.TRAIN, online_positions=0, target_positions=target
        )
        ledger.record_baseline(target_positions=target)

    def train(self, *, online: int, target: int, committed: bool) -> None:
        activation, ledger = self._active()
        activation.meter.record_positions(
            ResourcePhase.TRAIN, online_positions=online, target_positions=target
        )
        ledger.record_train_attempt(
            online_positions=online,
            target_positions=target,
            quality_source_byte_exposures=online,
            committed=committed,
        )

    def evaluate(self, *, online: int, target: int, step: int) -> dict[str, object]:
        activation, ledger = self._active()
        activation.meter.record_positions(
            ResourcePhase.EVAL, online_positions=online, target_positions=target
        )
        snapshot = ledger.record_eval(
            online_positions=online,
            target_positions=target,
            training_step=step,
            observed_at=datetime.now(UTC),
            monotonic_ns=self._clock.monotonic_ns(),
        )
        return resource_snapshot_to_mapping(snapshot)

    def state(self) -> dict[str, object]:
        _, ledger = self._active()
        ledger.record_meter_observation(monotonic_ns=self._clock.monotonic_ns())
        return resource_ledger_state_to_mapping(ledger.current_state())
