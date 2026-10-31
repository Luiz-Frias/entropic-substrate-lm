"""The H1 observation adapter: captured H runs to paired pilot contrasts.

``paired_contrasts`` refuses H designs by design (``ladder_registration.py``) and
``_d1_runs`` refuses H telemetry; this module is the explicit H adapter they name. It
turns four-cell blocks of captured H runs (one :class:`~h_preservation.H1PreservationBlock`
per seed) into :class:`~seed_sizing.PilotContrast` values that ``plan_seed_count`` and
``dual_confidence_intervals`` consume UNCHANGED. It never sizes, never decides and holds
no margin: direction, margins and multiplicity belong to the sealed hypotheses.

Registration values live here only as documented parameter defaults (skeleton Q4 and
DH-32); each is a keyword the caller can override at invocation, so a second estimator's
keys are selectable without editing the adapter:

* ``endpoint_key`` = :data:`ENDPOINT_METRIC_KEY`, the per-canonical-byte log-marginal at k = 64;
* ``gate_key`` = :data:`GATE_METRIC_KEY`, the per-row MEDIAN of ESS/k at k = 64 (never the
  mean ``eval/ess_k64`` the runtime extractor emits);
* ``window`` = :data:`ENDPOINT_WINDOW`, the endpoint is the mean of that many last evaluations;
* ``ess_threshold`` = :data:`ESS_THRESHOLD`, compared as ``value >= threshold``.

Both metric values are read ONLY from ``HRun.elm`` (``h_elm_projection.project_elm``): an
absent key is absence, never ``0.0`` or ``nan``, and no other k is ever substituted.

Exposure has two units, kept apart on purpose (operator ruling Q-EXP (a), 2026-10-28):

* ``StatisticalEvaluation.source_bytes`` is the cumulative canonical prediction-phase byte
  count, :func:`h1_exposure_bytes` over the lineage's step rows up to and including the
  evaluation. It is re-read from EVERY segment's ``metrics.jsonl`` (``HRun`` carries no byte
  counts, and a stitched resumed run's ``metrics_path`` is the final pod's alone), and the
  re-read committed prediction rows must equal ``HRun.training``'s prediction entries one to
  one or the call raises.
* ``PairedSeed.left_exposure``/``right_exposure`` is the committed prediction optimizer step
  count at the last evaluation of the endpoint window, equal across blocks by the shared
  schedule. Canonical bytes and committed step counts are checked equal across the cells of a
  block at every evaluation the cells share (a mismatch before the window raises too); the
  window bytes are reported as a diagnostic and never enter ``PairedSeed`` or ``PilotContrast``.

Failure semantics. Defects (identity or class mismatch, unequal exposure inside a block,
a window schedule that differs across blocks, a duplicate seed, a conflicting construction
digest, observations handed to one ``pilot_inputs`` call that were projected under different
``(endpoint_key, gate_key, window)``) RAISE: the closed :class:`H1ExclusionReason` vocabulary
has no member for them and they are execution defects, not exclusions. Only the enumerated conditions below map to a
reason, and any reason in any cell drops the WHOLE block from all three contrasts, reported
and never silent:

* ``READER_REFUSAL``: a ``TelemetryGap`` raised by the H reader seam (``read_h_rows``);
* ``MISSING_ENDPOINT``: fewer than ``window`` evaluations, ``endpoint_key`` absent at any
  window evaluation (the window never shrinks and absent evaluations are never skipped), or a
  ``zero_importance_weight_mass`` gap row at the selected k (the key's ``_k<n>`` suffix) at any
  window evaluation: the reduction would average the finite rows only, so it is unavailable;
* ``NONFINITE_ENDPOINT``: the window reduction is not finite;
* ``INADMISSIBLE_ESS``: ``gate_key`` is absent or below the threshold at ANY evaluation of the
  endpoint window, each compared on its own (never averaged across evaluations);
* ``INDEPENDENT_WARMUP``: a caller-supplied cell label (the detector is not this module's).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.analyze.h_aggregate import HRun, read_h_rows
from entropic_substrate.ladder.analyze.h_resumed_run import HResumedRun
from entropic_substrate.ladder.analyze.ladder_registration import (
    StatisticalEvaluation,
    StatisticalRun,
)
from entropic_substrate.ladder.analyze.seed_sizing import PairedSeed, PilotContrast
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.register.h_preservation import (
    H1ExclusionReason,
    H1PreservationBlock,
    h1_construction_digest,
    h1_exposure_bytes,
)

__all__ = [
    "ENDPOINT_METRIC_KEY",
    "ENDPOINT_WINDOW",
    "ESS_THRESHOLD",
    "GATE_METRIC_KEY",
    "H1BlockCapture",
    "H1BlockExclusion",
    "H1BlockExposure",
    "H1BlockObservation",
    "H1CellProjection",
    "H1EvaluationReading",
    "H1ExclusionFinding",
    "H1PilotInputs",
    "observe_h1_blocks",
    "pilot_inputs",
    "pilot_inputs_by_threshold",
    "project_h1_cell",
]

# Registration values (skeleton Q4, DH-32), documented parameter defaults only.
ENDPOINT_METRIC_KEY = "eval/log_marginal_per_byte_k64"
GATE_METRIC_KEY = "eval/ess_ratio_median_k64"
ENDPOINT_WINDOW = 5
ESS_THRESHOLD = 0.25

_STEP_SCHEMA = "h-training-step-v2"
_PREDICTION = "prediction"
_COMMITTED = "committed"
_GATE_SCOPE = "every evaluation of the endpoint window, each on its own"


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonblank string")
    return value


def _window(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("endpoint window must be an integer >= 1")
    return value


def _threshold(value: object) -> float:
    if type(value) is not int and type(value) is not float:
        raise ValueError("ESS threshold must be a number in (0, 1]")
    if not 0 < value <= 1:  # a ratio ESS/k lies in (0, 1]
        raise ValueError("ESS threshold must be a number in (0, 1]")
    return float(value)


@dataclass(frozen=True, slots=True)
class H1EvaluationReading:
    """One evaluation of one cell, on the global step axis.

    ``committed_steps`` and ``source_bytes`` are over the SAME inclusive lineage prefix
    (committed prediction rows with global step <= ``global_step``). A metric absent from
    the captured projection is ``None``.
    """

    global_step: int
    committed_steps: int
    source_bytes: int
    endpoint_value: float | None
    gate_value: float | None


@dataclass(frozen=True, slots=True)
class H1CellProjection:
    """One cell (or one diagnostic arm): reader binding, exposure, window and endpoint.

    ``window_readings`` is empty when fewer than ``window`` evaluations exist; no endpoint
    and no gate verdict is ever computed over fewer. ``endpoint`` is ``None`` exactly when
    ``endpoint_reason`` is set, and the other way around.
    """

    profile_identity: str
    run_class: str
    warmup_steps: int
    segment_metrics: tuple[Path, ...]
    endpoint_key: str
    gate_key: str
    window: int
    readings: tuple[H1EvaluationReading, ...]
    window_readings: tuple[H1EvaluationReading, ...]
    endpoint: float | None
    endpoint_reason: H1ExclusionReason | None
    endpoint_detail: str


@dataclass(frozen=True, slots=True)
class H1ExclusionFinding:
    """One typed reason, in one cell, with its text. ``cell`` is the arm label."""

    reason: H1ExclusionReason
    cell: str
    detail: str

    def __post_init__(self) -> None:
        # StrEnum members equal their bare strings, so value equality would admit an
        # unregistered string
        if type(self.reason) is not H1ExclusionReason:
            raise TypeError("exclusion reason must be an H1ExclusionReason member")


@dataclass(frozen=True, slots=True)
class H1BlockExposure:
    """Per-block exposure diagnostic: equal across the four cells by construction here."""

    window_global_steps: tuple[int, ...]
    window_committed_steps: tuple[int, ...]
    window_source_bytes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class H1BlockCapture:
    """One seed block: its validated four profiles and the captured run of each cell.

    ``cells`` is keyed by the profile name (the cell label, which is also the run's arm) and
    must equal the block's name set. A resumed lineage is passed as an :class:`HResumedRun`.
    ``independent_warmup`` holds the labels the caller's detector flagged; this module only maps
    them to an exclusion.
    """

    block: H1PreservationBlock
    cells: Mapping[str, HRun | HResumedRun]
    independent_warmup: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class H1BlockObservation:
    """Threshold-independent observation of one block. ``cells[i]`` is ``None`` for a
    refused cell and is aligned with ``block.profiles``."""

    seed: int
    pairing_id: str
    block: H1PreservationBlock
    cells: tuple[H1CellProjection | None, ...]
    findings: tuple[H1ExclusionFinding, ...]
    construction_digests: tuple[str, ...]
    exposure: H1BlockExposure | None


@dataclass(frozen=True, slots=True)
class H1BlockExclusion:
    seed: int
    pairing_id: str
    findings: tuple[H1ExclusionFinding, ...]

    @property
    def reasons(self) -> tuple[H1ExclusionReason, ...]:
        return tuple(dict.fromkeys(finding.reason for finding in self.findings))

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(finding.cell for finding in self.findings))


@dataclass(frozen=True, slots=True)
class H1PilotInputs:
    """The admitted population at one threshold, ready for ``plan_seed_count``.

    ``contrasts`` is empty when no block is admitted (``plan_seed_count`` then refuses the
    empty family loudly); ``exclusions`` lists every dropped block, in seed order.
    """

    ess_threshold: float
    admitted_seeds: tuple[int, ...]
    exclusions: tuple[H1BlockExclusion, ...]
    runs: tuple[StatisticalRun, ...]
    contrasts: tuple[PilotContrast, ...]


# ------------------------------------------------------------------------ cell projection
def _lineage_rows(segments: Sequence[HRun], warmup_steps: int) -> list[tuple[int, dict[str, Any]]]:
    """``(global step, row)`` of every step row of every segment, in lineage order."""
    rows: list[tuple[int, dict[str, Any]]] = []
    for segment in segments:
        for row in read_h_rows(segment.metrics_path):
            if row.get("schema") != _STEP_SCHEMA:
                continue
            step = row.get("step")
            if type(step) is not int or step < 1:
                raise ValueError(f"{segment.metrics_path}: step row without a positive int step")
            offset = warmup_steps if row.get("phase") == _PREDICTION else 0
            rows.append((step + offset, row))
    return rows


def _committed_prediction(row: Mapping[str, Any]) -> bool:
    return row.get("phase") == _PREDICTION and row.get("status") == _COMMITTED


def _reading_value(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"projected metric {key!r} is not a number: {value!r}")
    return float(value)


def _key_k(key: str) -> int:
    """The k named by a metric key's ``_k<n>`` suffix (the registered keys all carry one)."""
    head, separator, digits = key.rpartition("_k")
    if not separator or not head or not digits.isdigit() or digits[0] == "0":
        raise ValueError(f"metric key {key!r} names no k; the gap check needs the selected k")
    return int(digits)


def _endpoint(
    readings: tuple[H1EvaluationReading, ...],
    window: int,
    gaps: Sequence[tuple[int, int, int, int]] = (),
) -> tuple[tuple[H1EvaluationReading, ...], float | None, H1ExclusionReason | None, str]:
    """``gaps`` holds ``(global step, k, gap rows, rows)`` for each evaluation with a gap row."""
    if len(readings) < window:
        return (
            (),
            None,
            H1ExclusionReason.MISSING_ENDPOINT,
            f"{len(readings)} evaluation(s), fewer than the {window}-evaluation endpoint window",
        )
    tail = readings[-window:]
    absent = [r.global_step for r in tail if r.endpoint_value is None]
    if absent:
        return (
            tail,
            None,
            H1ExclusionReason.MISSING_ENDPOINT,
            f"endpoint metric absent at window evaluation(s) {absent}",
        )
    window_steps = {r.global_step for r in tail}
    in_window = [gap for gap in gaps if gap[0] in window_steps]
    if in_window:
        described = "; ".join(
            f"step {step} k={k}: {gap_rows} of {rows} rows" for step, k, gap_rows, rows in in_window
        )
        return (
            tail,
            None,
            H1ExclusionReason.MISSING_ENDPOINT,
            f"gap rows at window evaluation(s) ({described}); a zero_importance_weight_mass "
            "row is not a finite measurement, so a mean over the finite rows is not the endpoint",
        )
    values = [r.endpoint_value for r in tail if r.endpoint_value is not None]
    try:
        mean = math.fsum(values) / window
    except (OverflowError, ValueError):  # finite overflow, or inf + -inf
        mean = math.nan
    if not math.isfinite(mean):
        return (
            tail,
            None,
            H1ExclusionReason.NONFINITE_ENDPOINT,
            f"the mean of the last {window} evaluations is not finite",
        )
    return tail, mean, None, ""


def project_h1_cell(
    source: HRun | HResumedRun,
    *,
    endpoint_key: str = ENDPOINT_METRIC_KEY,
    gate_key: str = GATE_METRIC_KEY,
    window: int = ENDPOINT_WINDOW,
) -> H1CellProjection:
    """Project one captured cell (or diagnostic arm) onto its global-axis readings.

    The same code serves the block path and the per-cell diagnostic (a k <= 8 capture is
    projected by selecting its keys here). Raises ``TelemetryGap`` when the H reader refuses
    a segment's file, and ``ValueError`` when the re-read committed prediction rows do not
    equal ``HRun.training``'s prediction entries (a stitched run's lone final pod, say).
    """
    _text(endpoint_key, "endpoint key")
    _text(gate_key, "gate key")
    _window(window)
    if type(source) is HResumedRun:
        run, segments = source.run, source.segments
    elif type(source) is HRun:
        run, segments = source, (source,)
    else:
        raise TypeError("a cell is a captured HRun or HResumedRun")
    rows = _lineage_rows(segments, run.warmup_steps)
    # the full-lineage pass: an unrecognized phase/status or a bad source_bytes raises here
    # (measurement path, fail LOUD), before any prefix is formed
    h1_exposure_bytes(row for _, row in rows)
    re_read = [step for step, row in rows if _committed_prediction(row)]
    held = [step for step, phase, _ in run.training if phase == _PREDICTION]
    if re_read != held:
        raise ValueError(
            f"{[str(s.metrics_path) for s in segments]}: the committed prediction rows re-read "
            f"from the segment files ({len(re_read)} rows) differ from the run's training "
            f"entries ({len(held)} rows); a stitched resumed run needs every segment"
        )
    readings: list[H1EvaluationReading] = []
    for global_step, elm in run.elm:
        prefix = [(step, row) for step, row in rows if step <= global_step]
        metrics: Mapping[str, Any] = elm["metrics"]
        readings.append(
            H1EvaluationReading(
                global_step,
                sum(1 for _, row in prefix if _committed_prediction(row)),
                h1_exposure_bytes(row for _, row in prefix),
                _reading_value(metrics, endpoint_key),
                _reading_value(metrics, gate_key),
            )
        )
    all_readings = tuple(readings)
    # the selected k of each key: a gap row at one of them makes the endpoint or the gate a
    # mean over finite rows only, so it is recorded per evaluation and gated inside the window
    selected = sorted({_key_k(endpoint_key), _key_k(gate_key)})
    gaps: list[tuple[int, int, int, int]] = []
    for global_step, elm in run.elm:
        samples: Mapping[str, Mapping[str, Any]] = elm["samples"]
        for k in selected:
            sample = samples.get(str(k))
            if sample is not None and sample["gap_rows"] > 0:
                gaps.append((global_step, k, sample["gap_rows"], sample["rows"]))
    tail, endpoint, reason, detail = _endpoint(all_readings, window, gaps)
    return H1CellProjection(
        profile_identity=run.profile_identity,
        run_class=_captured_class(run),
        warmup_steps=run.warmup_steps,
        segment_metrics=tuple(segment.metrics_path for segment in segments),
        endpoint_key=endpoint_key,
        gate_key=gate_key,
        window=window,
        readings=all_readings,
        window_readings=tail,
        endpoint=endpoint,
        endpoint_reason=reason,
        endpoint_detail=detail,
    )


def _captured_class(run: HRun) -> str:
    value = run.run_start.get("run_class")
    if type(value) is not str or not value:
        raise ValueError(f"{run.metrics_path}: run_start carries no run_class")
    return value


# ---------------------------------------------------------------------- block observation
def _bind(label: str, profile: HProfile, source: HRun | HResumedRun) -> None:
    run = source.run if type(source) is HResumedRun else source
    if type(run) is not HRun:
        raise TypeError("a cell is a captured HRun or HResumedRun")
    if run.profile_identity != profile.identity:
        raise ValueError(
            f"cell {label!r}: captured profile identity {run.profile_identity} differs from "
            f"the block profile's {profile.identity}"
        )
    if _captured_class(run) != profile.run_class:
        raise ValueError(
            f"cell {label!r}: captured run_class {_captured_class(run)!r} differs from the "
            f"block profile's {profile.run_class!r}"
        )


def _within_block_exposure(
    seed: int, cells: Sequence[H1CellProjection], labels: Sequence[str]
) -> H1BlockExposure | None:
    """The within-block equality checks; a mismatch is an execution defect and RAISES.

    Every evaluation that a cell shares with the LONGEST cell must agree on the global step,
    the committed step count and the canonical bytes, so a pre-window divergence cannot be
    admitted just because it re-converges before the window. A cell that stops early is a
    prefix and is left to the window rule (MISSING_ENDPOINT). With the longest cell as the
    reference every other cell is compared over its own whole length, so any two cells agree
    on every evaluation they share (both are prefixes of the same sequence). A first-cell
    reference left a divergence between two later cells unseen whenever cell 0 stopped
    before it: CodeRabbit comment 4428407643 reported that block as MISSING_ENDPOINT
    where this docstring says it raises. Returns the window exposure, or ``None`` when a
    cell forms no window.
    """
    reference = max(cells, key=lambda cell: len(cell.readings)).readings
    for label, cell in zip(labels, cells, strict=True):
        shared = len(cell.readings)
        if [r.global_step for r in cell.readings[:shared]] != [
            r.global_step for r in reference[:shared]
        ]:
            raise ValueError(f"seed {seed}: cell {label!r} has different evaluation steps")
        if [r.committed_steps for r in cell.readings[:shared]] != [
            r.committed_steps for r in reference[:shared]
        ]:
            raise ValueError(f"seed {seed}: cell {label!r} has different committed step counts")
        if [r.source_bytes for r in cell.readings[:shared]] != [
            r.source_bytes for r in reference[:shared]
        ]:
            raise ValueError(f"seed {seed}: cell {label!r} has different canonical exposure bytes")
    if not all(cell.window_readings for cell in cells):
        return None
    first = cells[0].window_readings
    steps = tuple(r.global_step for r in first)
    for label, cell in zip(labels[1:], cells[1:], strict=True):
        if tuple(r.global_step for r in cell.window_readings) != steps:
            raise ValueError(f"seed {seed}: cell {label!r} has different window evaluation steps")
    return H1BlockExposure(
        steps,
        tuple(r.committed_steps for r in first),
        tuple(r.source_bytes for r in first),
    )


def _observe_block(
    capture: H1BlockCapture, endpoint_key: str, gate_key: str, window: int
) -> H1BlockObservation:
    if type(capture) is not H1BlockCapture:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("blocks are H1BlockCapture values")
    block = capture.block
    if type(block) is not H1PreservationBlock:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("a capture needs an exact H1PreservationBlock")
    profiles = block.profiles
    labels = tuple(profile.name for profile in profiles)
    if set(capture.cells) != set(labels) or len(capture.cells) != len(labels):
        raise ValueError(f"cells {sorted(capture.cells)} must equal the block's arms {labels}")
    flagged = tuple(capture.independent_warmup)
    if any(label not in labels for label in flagged):
        raise ValueError("an independent-warmup flag names a cell outside the block")
    seed = profiles[0].seed  # H1PreservationBlock refuses a block whose members differ in seed
    findings: list[H1ExclusionFinding] = []
    cells: list[H1CellProjection | None] = []
    for label, profile in zip(labels, profiles, strict=True):
        source = capture.cells[label]
        _bind(label, profile, source)
        try:
            cell = project_h1_cell(
                source, endpoint_key=endpoint_key, gate_key=gate_key, window=window
            )
        except TelemetryGap as gap:  # the H reader seam refused a file of this cell
            cells.append(None)
            findings.append(H1ExclusionFinding(H1ExclusionReason.READER_REFUSAL, label, str(gap)))
            continue
        cells.append(cell)
        if cell.endpoint_reason is not None:
            findings.append(H1ExclusionFinding(cell.endpoint_reason, label, cell.endpoint_detail))
    findings.extend(
        H1ExclusionFinding(
            H1ExclusionReason.INDEPENDENT_WARMUP, label, "flagged by the caller's detector"
        )
        for label in labels
        if label in flagged
    )
    exposure: H1BlockExposure | None = None
    projected = [(label, cell) for label, cell in zip(labels, cells, strict=True) if cell]
    if len(projected) > 1:
        exposure = _within_block_exposure(
            seed, [cell for _, cell in projected], [label for label, _ in projected]
        )
    return H1BlockObservation(
        seed=seed,
        pairing_id=block.identity,
        block=block,
        cells=tuple(cells),
        findings=tuple(findings),
        construction_digests=tuple(h1_construction_digest(profile) for profile in profiles),
        exposure=exposure,
    )


def _require_one_family(observed: Sequence[H1BlockObservation]) -> None:
    """The cross-block checks of one design family; every violation RAISES.

    Run by ``observe_h1_blocks`` over the blocks of one call and again by ``pilot_inputs`` over
    whatever observations it is handed, so observations pooled from several calls meet the same
    refusals as one call over all of them (FOR-1711, review comment 4427838782, PR #416: seeds
    101-104 and 105-108 with ``encoder_layers=2`` observed in two calls and concatenated were
    admitted by ``pilot_inputs`` with 0 exclusions and sized by ``plan_seed_count`` at
    ``required_seeds`` 159, while the same eight blocks in one call raised). ``observed`` is in
    seed order, which fixes the arm a digest refusal names. Distinct seeds, one construction
    digest per arm label and per (sentence, spine) axis, and an identical window schedule
    (committed step counts) across every block that forms a window.
    """
    if len({item.seed for item in observed}) != len(observed):
        raise ValueError("duplicate seed block")
    by_label: dict[str, str] = {}
    by_axis: dict[tuple[str, str], str] = {}
    for item in observed:
        for profile, digest in zip(item.block.profiles, item.construction_digests, strict=True):
            axis = (profile.law.mode, profile.spine.carrier)
            if by_label.setdefault(profile.name, digest) != digest or (
                by_axis.setdefault(axis, digest) != digest
            ):
                raise ValueError(
                    f"construction digest of arm {profile.name!r} differs across seed blocks"
                )
    schedules = {item.exposure.window_committed_steps for item in observed if item.exposure}
    if len(schedules) > 1:
        raise ValueError(
            f"blocks differ in the committed step counts at the endpoint window: {schedules}"
        )


def observe_h1_blocks(
    captures: Sequence[H1BlockCapture],
    *,
    endpoint_key: str = ENDPOINT_METRIC_KEY,
    gate_key: str = GATE_METRIC_KEY,
    window: int = ENDPOINT_WINDOW,
) -> tuple[H1BlockObservation, ...]:
    """Observe every block once, independent of any ESS threshold; ordered by seed.

    Cross-block checks RAISE (``_require_one_family``): distinct seeds, one construction
    digest per arm label and per (sentence, spine) axis, and an identical window schedule
    (committed step counts) across every block that forms a window. ``pilot_inputs`` runs the
    same checks again, so concatenating the results of several calls cannot bypass them.
    """
    if not captures:
        raise ValueError("at least one block is required")
    observed = sorted(
        (_observe_block(capture, endpoint_key, gate_key, window) for capture in captures),
        key=lambda item: item.seed,
    )
    _require_one_family(observed)
    return tuple(observed)


# ------------------------------------------------------------------------- pilot contrasts
def _gate_findings(item: H1BlockObservation, threshold: float) -> list[H1ExclusionFinding]:
    out: list[H1ExclusionFinding] = []
    for label, cell in zip((p.name for p in item.block.profiles), item.cells, strict=True):
        if cell is None or not cell.window_readings:
            continue  # no window formed: no gate verdict is computed over fewer evaluations
        failing = [
            (r.global_step, r.gate_value)
            for r in cell.window_readings
            if r.gate_value is None or not r.gate_value >= threshold
        ]
        if failing:
            out.append(
                H1ExclusionFinding(
                    H1ExclusionReason.INADMISSIBLE_ESS,
                    label,
                    f"{cell.gate_key} below {threshold} or absent at (global step, value) "
                    f"{failing}; gate scope: {_GATE_SCOPE}",
                )
            )
    return out


def _admitted_cells(item: H1BlockObservation) -> tuple[H1CellProjection, ...]:
    """The four cells of an ADMITTED block; one with a finding never reaches here."""
    cells = tuple(cell for cell in item.cells if cell is not None)
    if len(cells) != len(item.cells) or any(cell.endpoint is None for cell in cells):
        raise ValueError(
            f"seed {item.seed}: an admitted block holds a refused or endpoint-less cell"
        )
    return cells


def _runs(item: H1BlockObservation, family_digest: str) -> tuple[StatisticalRun, ...]:
    runs: list[StatisticalRun] = []
    for profile, cell, digest in zip(
        item.block.profiles, _admitted_cells(item), item.construction_digests, strict=True
    ):
        runs.append(
            StatisticalRun(
                arm=profile.name,
                seed=profile.seed,
                run_class=cell.run_class,
                family_digest=family_digest,
                construction_digest=digest,
                evals=tuple(
                    StatisticalEvaluation(
                        r.global_step,
                        r.source_bytes,
                        ()
                        if r.endpoint_value is None
                        else ((cell.endpoint_key, r.endpoint_value),),
                    )
                    for r in cell.readings
                ),
            )
        )
    return tuple(runs)


def pilot_inputs(
    observations: Sequence[H1BlockObservation],
    *,
    family_digest: str,
    units: str,
    expected_exposure: int,
    ess_threshold: float = ESS_THRESHOLD,
) -> H1PilotInputs:
    """Gate, exclude whole blocks, and build the three deciding ``PilotContrast`` values.

    ``family_digest`` (also the contrasts' regime id), ``units`` and ``expected_exposure``
    (the declared committed prediction steps at the endpoint) are bound to the sealed design
    by the caller and never derived from the observations they are checked against. Contrast
    weights come from ``H1PreservationBlock.factorial_terms`` unchanged; each pair is
    ``(seed, fsum(weight x endpoint), 0.0)`` in the ``paired_contrasts`` convention, with the
    committed step count at the last window evaluation as both exposures. Observations are
    ordered by seed here, so ``admitted_seeds``, ``exclusions``, ``runs`` and the pairs of every
    contrast are in seed order whatever order the caller passes.

    Raises ``ValueError``, in this order, on: observations projected under different
    ``(endpoint_key, gate_key, window)`` (one call is one projection regime, never a mixture;
    first, because window schedules are comparable only within one regime); then the family
    checks of ``_require_one_family`` over the observations as handed in, however many
    ``observe_h1_blocks`` calls produced them: a duplicate seed, a construction digest that
    differs across blocks per arm label or per (sentence, spine) axis, and a window schedule
    (committed step counts) that differs across blocks; then admitted blocks that carry
    different captured run classes.
    """
    _text(family_digest, "family digest")
    _text(units, "units")
    threshold = _threshold(ess_threshold)
    ordered = sorted(observations, key=lambda observed: observed.seed)
    regimes = {
        (cell.endpoint_key, cell.gate_key, cell.window)
        for item in ordered
        for cell in item.cells
        if cell
    }
    if len(regimes) > 1:
        raise ValueError(
            "observations carry different projection regimes (endpoint_key, gate_key, window): "
            f"{sorted(regimes)}"
        )
    # after the regime check: a window schedule is only comparable within one regime
    _require_one_family(ordered)
    admitted: list[H1BlockObservation] = []
    exclusions: list[H1BlockExclusion] = []
    for item in ordered:
        findings = (*item.findings, *_gate_findings(item, threshold))
        if findings:
            exclusions.append(H1BlockExclusion(item.seed, item.pairing_id, findings))
        else:
            admitted.append(item)
    run_classes = {cell.run_class for item in admitted for cell in item.cells if cell}
    if len(run_classes) > 1:
        raise ValueError(f"admitted blocks carry different captured run classes {run_classes}")
    runs = tuple(run for item in admitted for run in _runs(item, family_digest))
    contrasts: list[PilotContrast] = []
    if admitted:
        for name in admitted[0].block.factorial_terms:
            pairs: list[PairedSeed] = []
            for item in admitted:
                endpoint: dict[str, float] = {}
                for profile, cell in zip(item.block.profiles, _admitted_cells(item), strict=True):
                    if cell.endpoint is None:  # unreachable: _admitted_cells refuses it
                        raise ValueError(f"seed {item.seed}: {profile.name!r} has no endpoint")
                    endpoint[profile.name] = cell.endpoint
                if item.exposure is None:
                    raise ValueError(f"seed {item.seed}: an admitted block has no window exposure")
                steps = item.exposure.window_committed_steps[-1]
                value = math.fsum(
                    weight * endpoint[arm] for arm, weight in item.block.factorial_terms[name]
                )
                pairs.append(PairedSeed(item.seed, value, 0.0, steps, steps))
            contrasts.append(
                PilotContrast(
                    name,
                    units,
                    next(iter(run_classes)),
                    family_digest,
                    expected_exposure,
                    tuple(pairs),
                )
            )
    return H1PilotInputs(
        ess_threshold=threshold,
        admitted_seeds=tuple(item.seed for item in admitted),
        exclusions=tuple(exclusions),
        runs=runs,
        contrasts=tuple(contrasts),
    )


def pilot_inputs_by_threshold(
    observations: Sequence[H1BlockObservation],
    thresholds: Sequence[float],
    *,
    family_digest: str,
    units: str,
    expected_exposure: int,
) -> tuple[H1PilotInputs, ...]:
    """One gate-and-exclusion definition mapped over a caller-supplied threshold list.

    The result at threshold ``t`` IS ``pilot_inputs(..., ess_threshold=t)``: there is no
    second code path for the sensitivity readings.
    """
    values = tuple(thresholds)
    if not values or len(set(values)) != len(values):
        raise ValueError("thresholds must be a nonempty list without duplicates")
    return tuple(
        pilot_inputs(
            observations,
            family_digest=family_digest,
            units=units,
            expected_exposure=expected_exposure,
            ess_threshold=value,
        )
        for value in values
    )
