"""Bounded frozen-probe artifacts with independently replayed diagnostic scores.

These report-only bundles bind selected tensors, not their model provenance.
The execution adapter must supply the checkpoint, codec and source identities
from its verified training lineage; callers cannot obtain scientific authority here.
"""

import math
from collections.abc import Buffer, Callable
from dataclasses import asdict, dataclass, fields, replace
from inspect import signature
from io import BytesIO
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import torch

from entropic_substrate.analysis.predictive_probe import ProbeConfig
from entropic_substrate.d1_artifact_snapshot import publish_d1_atomic_bytes, read_d1_artifact_bytes
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.sentence_prediction.journal import artifact_reference
from entropic_substrate.sentence_prediction.recovery_state import recovery_record
from entropic_substrate.sentence_prediction.span_probe import (
    FrozenSpanProbe,
    SpanProbeManifest,
    SpanProbeReport,
    fit_span_probe,
    score_span_probe,
    validate_span_states,
)
from entropic_substrate.sentence_prediction.span_states import SpanProbePosition, SpanStateSet


@dataclass(frozen=True)
class SpanProbeBundle:
    train: SpanStateSet
    heldout: SpanStateSet
    probes: tuple[FrozenSpanProbe, ...]
    reports: tuple[SpanProbeReport, ...]


def _limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("probe artifact byte limit must be positive")


def _duration(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("probe resource duration must be numeric and not boolean")
    number = float(cast(float, value))
    if not math.isfinite(number) or number < 0:
        raise ValueError("probe resource duration must be finite and nonnegative")
    return number


def _construct[T](factory: Callable[..., T], raw: object) -> T:
    return factory(**recovery_record(raw, " ".join(signature(factory).parameters)))


class _LimitedBuffer(BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        _limit(limit)
        self.limit = limit
        self.overflowed = False

    def write(self, data: Buffer, /) -> int:
        if self.tell() + memoryview(data).nbytes > self.limit:
            self.overflowed = True
            raise ValueError("probe serialization exceeds byte limit")
        return super().write(data)


def _serialize(value: object, limit: int) -> bytes:
    with _LimitedBuffer(limit) as stream:
        try:
            torch.save(value, stream)
        except Exception as error:
            # The Torch ZIP finalizer can replace the original write exception.
            if stream.overflowed:
                raise ValueError("probe serialization exceeds byte limit") from error
            raise
        return stream.getvalue()


def capture_span_probes(
    root: Path,
    name: str,
    train: SpanStateSet,
    heldout: SpanStateSet,
    config: ProbeConfig,
    *,
    max_bytes: int,
) -> ArtifactIdentity:
    """Freeze both independently fitted controls before scoring; publish bytes last."""
    _limit(max_bytes)
    validate_span_states(train)
    validate_span_states(heldout)
    if (train.checkpoint_id, train.codec_id, train.source_digest) != (
        heldout.checkpoint_id,
        heldout.codec_id,
        heldout.source_digest,
    ):
        raise ValueError("probe capture binding mismatch")
    # Minimum tensor payload from the actual selected arrays and fixed head shapes.
    # Metadata/ZIP overhead is separately bounded by the serialization stream.
    tensor_bytes = sum(
        t.numel() * t.element_size()
        for states in (train, heldout)
        for t in (
            states.before,
            states.after,
            states.before_horizons,
            states.after_horizons,
            states.labels,
        )
    )
    head_bytes = 2 * 256 * (train.before.shape[1] + 2) * torch.float32.itemsize
    if tensor_bytes + head_bytes > max_bytes:
        raise ValueError("probe tensor payload exceeds byte limit before fitting")
    probes = tuple(fit_span_probe(train, config, control=c) for c in ("observed", "shuffled"))
    reports = tuple(score_span_probe(probe, heldout) for probe in probes)
    payload = _serialize(
        {
            "schema": "h-span-probes-v1",
            "scientific_authority": "none",
            "meaning": "realized_predictive_score_differences",
            "train": asdict(train),
            "heldout": asdict(heldout),
            "probes": tuple(
                {"manifest": asdict(p.manifest), "weights": p.state_dict()} for p in probes
            ),
            "reports": tuple(asdict(report) for report in reports),
        },
        max_bytes,
    )
    return publish_d1_atomic_bytes(root, name, payload, label="frozen span probes")


def _positions(raw: object) -> tuple[SpanProbePosition, ...]:
    if type(raw) is not tuple:
        raise ValueError("probe positions must be an explicit tuple")
    return tuple(_construct(SpanProbePosition, row) for row in raw)


def _states(raw: object) -> SpanStateSet:
    row = recovery_record(raw, " ".join(f.name for f in fields(SpanStateSet))).copy()
    row["positions"] = _positions(row["positions"])
    for name in ("before", "after", "before_horizons", "after_horizons", "labels"):
        if not isinstance(row[name], torch.Tensor):
            raise ValueError("probe state artifact requires tensors")
    states = _construct(SpanStateSet, row)
    validate_span_states(states)
    return states


def _probe(raw: object, train: SpanStateSet) -> FrozenSpanProbe:
    row = recovery_record(raw, "manifest weights")
    manifest = recovery_record(
        row["manifest"], " ".join(f.name for f in fields(SpanProbeManifest))
    ).copy()
    config = recovery_record(manifest["config"], " ".join(f.name for f in fields(ProbeConfig)))
    manifest["config"] = _construct(ProbeConfig, config)
    manifest["training_positions"] = _positions(manifest["training_positions"])
    _duration(manifest["fit_seconds"])
    weights = recovery_record(row["weights"], "weight bias")
    weight, bias = weights["weight"], weights["bias"]
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("frozen probe weights must be tensors")
    probe = FrozenSpanProbe(_construct(SpanProbeManifest, manifest), weight, bias)
    probe.verify_training(train)
    return probe


def reconcile_span_evidence(expected: object, saved: object) -> None:
    """Replay tolerance covers FP32 CPU arithmetic, not changed labels or policies."""
    if type(expected) is float:
        if (
            type(saved) not in (int, float)
            or not math.isfinite(cast(float, saved))
            or not math.isclose(expected, cast(float, saved), abs_tol=1e-5, rel_tol=1e-6)
        ):
            raise ValueError("probe numerical report differs from replay")
    elif type(expected) is dict:
        if type(saved) is not dict or expected.keys() != saved.keys():
            raise ValueError("probe report fields differ from replay")
        for key in expected:
            reconcile_span_evidence(expected[key], saved[key])
    elif isinstance(expected, (tuple, list)):
        if (
            not isinstance(saved, (tuple, list))
            or type(saved) is not type(expected)
            or len(expected) != len(saved)
        ):
            raise ValueError("probe report population differs from replay")
        for left, right in zip(expected, saved, strict=True):
            reconcile_span_evidence(left, right)
    elif type(expected) is not type(saved) or expected != saved:
        raise ValueError("probe report identity or coordinate differs from replay")


def read_span_probes(
    root: Path,
    artifact: ArtifactIdentity,
    *,
    expected_checkpoint: str,
    expected_codec: str,
    expected_source: str,
    max_bytes: int,
) -> SpanProbeBundle:
    """Verify frozen weights and recompute every held-out score and summary."""
    _limit(max_bytes)
    ref = artifact_reference(root, artifact)
    _, data, digest = read_d1_artifact_bytes(
        root, ref["path"], label="span probes", max_bytes=max_bytes
    )
    if digest != ref["sha256"]:
        raise ValueError("probe artifact content hash mismatch")
    with ZipFile(BytesIO(data)) as archive:
        if sum(info.file_size for info in archive.infolist()) > max_bytes:
            raise ValueError("expanded probe artifact exceeds byte limit")
    raw = recovery_record(
        torch.load(BytesIO(data), map_location="cpu", weights_only=True),
        "schema scientific_authority meaning train heldout probes reports",
    )
    if (
        raw["schema"] != "h-span-probes-v1"
        or raw["scientific_authority"] != "none"
        or raw["meaning"] != "realized_predictive_score_differences"
    ):
        raise ValueError("unknown probe evidence version, authority, or meaning")
    train, heldout = _states(raw["train"]), _states(raw["heldout"])
    for states in (train, heldout):
        if (states.checkpoint_id, states.codec_id, states.source_digest) != (
            expected_checkpoint,
            expected_codec,
            expected_source,
        ):
            raise ValueError("probe artifact lineage binding mismatch")
    probe_rows, report_rows = raw["probes"], raw["reports"]
    if (
        type(probe_rows) is not tuple
        or len(probe_rows) != 2
        or type(report_rows) is not tuple
        or len(report_rows) != 2
    ):
        raise ValueError("probe artifact must retain both fitted controls")
    probes = tuple(_probe(value, train) for value in probe_rows)
    if tuple(p.manifest.control for p in probes) != ("observed", "shuffled"):
        raise ValueError("probe control roster mismatch")
    if probes[0].manifest.config != probes[1].manifest.config:
        raise ValueError("probe controls used different fitting envelopes")
    reports = []
    for probe, saved in zip(probes, report_rows, strict=True):
        report_row = recovery_record(saved, " ".join(f.name for f in fields(SpanProbeReport)))
        measured = score_span_probe(probe, heldout)
        report = replace(measured, score_seconds=_duration(report_row["score_seconds"]))
        reconcile_span_evidence(asdict(report), saved)
        reports.append(report)
    return SpanProbeBundle(train, heldout, probes, tuple(reports))
