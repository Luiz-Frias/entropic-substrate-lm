"""Immutable metadata and metric-extraction contracts for tracking mirrors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

type TrackingValue = str | int | float | bool


def _metadata(
    value: Mapping[str, TrackingValue] | None,
    *,
    label: str,
) -> Mapping[str, TrackingValue]:
    if value is None:
        return MappingProxyType({})
    copied: dict[str, TrackingValue] = {}
    for key, item in value.items():
        if type(key) is not str or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        if type(item) not in (str, int, float, bool):
            raise TypeError(f"{label}[{key!r}] must be an MLflow scalar")
        copied[key] = item
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class TrackingRunDescriptor:
    """Complete immutable metadata needed to compose one tracking mirror."""

    run_name: str
    tiny: bool
    params: Mapping[str, TrackingValue]
    tags: Mapping[str, TrackingValue]
    config_hash: str | None = None
    dataset_pointer_path: Path | None = None
    dataset_split_spec: Mapping[str, TrackingValue] | None = None

    def __post_init__(self) -> None:
        if type(self.run_name) is not str or not self.run_name.strip():
            raise ValueError("tracking run_name must be a non-empty string")
        if type(self.tiny) is not bool:
            raise TypeError("tracking tiny must be a bool")
        if self.config_hash is not None and (
            type(self.config_hash) is not str or not self.config_hash.strip()
        ):
            raise ValueError("tracking config_hash must be None or a non-empty string")
        if self.dataset_pointer_path is not None:
            object.__setattr__(self, "dataset_pointer_path", Path(self.dataset_pointer_path))
        object.__setattr__(self, "params", _metadata(self.params, label="tracking params"))
        object.__setattr__(self, "tags", _metadata(self.tags, label="tracking tags"))
        split = self.dataset_split_spec
        object.__setattr__(
            self,
            "dataset_split_spec",
            None if split is None else _metadata(split, label="dataset_split_spec"),
        )


@dataclass(frozen=True, slots=True)
class ExtractedMetrics:
    """One event's immutable scalar projection into an operational mirror."""

    step: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))


@runtime_checkable
class MetricExtractor(Protocol):
    """Strategy translating an authoritative event into optional MLflow metrics."""

    def extract(self, event: object) -> ExtractedMetrics | None: ...


class LegacyMetricExtractor:
    """Exact pre-D1 eval allowlist and kappa projection."""

    _SCALARS = (
        "loss",
        "nats_per_byte",
        "bits_per_byte",
        "perplexity",
        "gram_frob_sq",
        "gram_penalty",
        "field_ce",
        "field_nats_per_byte",
        "field_brier",
        "field_ece",
        "field_entropy",
        "field_top1",
    )

    def extract(self, event: object) -> ExtractedMetrics | None:
        if not isinstance(event, Mapping) or event.get("event") != "eval":
            return None
        step = int(event.get("step", 0))
        metrics = {key: float(event[key]) for key in self._SCALARS if key in event}
        kappa = event.get("kappa")
        if kappa:
            metrics.update(_kappa_metrics(kappa))
        return ExtractedMetrics(step=step, metrics=metrics)


def _kappa_metrics(value: Any) -> dict[str, float]:
    """Shape-agnostic legacy kappa metrics retained byte-for-byte in naming."""
    metrics: dict[str, float] = {}
    all_values: list[float] = []
    spreads: list[float] = []
    for layer, row in enumerate(value):
        values = [float(item) for item in row]
        for head, item in enumerate(values):
            metrics[f"kappa/L{layer}/h{head}"] = item
        if values:
            spread = max(values) - min(values)
            spreads.append(spread)
            metrics[f"kappa/L{layer}/spread"] = spread
        all_values.extend(values)
    if spreads:
        metrics["kappa/spread_max"] = max(spreads)
    if all_values:
        metrics["kappa/mean_abs"] = sum(abs(item) for item in all_values) / len(all_values)
    return metrics
