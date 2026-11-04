"""H analysis aggregate: paired arms over captured H runs, on one global step axis.

Reads the two files an H worker leaves behind -- ``metrics.jsonl`` and
``worker-log.jsonl`` -- and nothing else: no network, no MLflow, a pure function of
the files. Admissibility goes through the existing H reader
(:func:`~entropic_substrate.ladder.capture.reader.h_telemetry.final_h_run_end` and
:func:`~entropic_substrate.ladder.capture.reader.h_telemetry.require_json_object_lines`),
reused and never forked; an inadmissible or self-contradicting file raises
:class:`~entropic_substrate.ladder.capture.reader.telemetry.TelemetryGap` naming the
file, and no empty or partial aggregate is ever returned (measurement path, fail LOUD).

MEASURED at 501c0447 on the HEpP7g capture (four files, two runs), each of which this
module pins rather than assumes:

* The captured rows carry no carrier, magnitude, sphere or ``config_hash`` marker.
  A run's arm identity is ``run_start.profile_identity`` and every step and
  evaluation row repeats it as ``run_identity``. Arm LABELS therefore come only from
  the caller (``identity=label`` pairs), never from a file name and never inferred.
* The file-name prefix is the profile identity, not the request digest; a worker log
  is bound to its metrics file by ``worker.start.request_sha256 ==
  run_start.request_sha256``, never by name.
* Step and evaluation rows carry a PHASE-LOCAL ``step`` (warm-up ``1..W``, then
  prediction ``1..N``); the evaluation rows are phase-local too. The global axis is
  rebuilt from the worker log: ``W = checkpoint.seal.global_step - step`` over the
  prediction seals, which must be constant and must equal both the count and the
  maximum step of the committed ``codec_warmup`` rows. Evaluation rows map in stream
  order, one to one, onto ``evaluation.start`` events. The ``annotate`` rows'
  ``checkpoint/<8-digit>/...`` keys are ALREADY global and are never offset.
* ``future_nll_per_dimension`` is null on every ``codec_warmup`` row (the future law
  is structurally absent there) and negative on most prediction rows; both are
  reported as captured, never clamped, abs-ed or filtered.

Every scalar the aggregate reports for a training or evaluation row equals
:class:`~entropic_substrate.ladder.run.h_metric_extractor.HTrainingStepExtractor`'s
projection of the same row (one definition, imported). The projection is not an
admission check: it omits null estimates by design, because a zero-importance-weight
gap entry carries none. This module therefore validates every row's shape itself,
against the engine's admission predicate, before it projects, and refuses rather
than drops.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, NoReturn

from entropic_substrate.kernel.codec.strict_json import strict_loads
from entropic_substrate.ladder.analyze.h_elm_projection import (
    DEFAULT_DENOMINATOR,
    HEADLINE_SAMPLES,
    ZERO_MASS_GAP,
    project_elm,
    require_importance_entry,
)
from entropic_substrate.ladder.capture.reader.h_telemetry import (
    final_h_run_end,
    is_scientific_stop,
    require_json_object_lines,
)
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor

__all__ = [
    "AGGREGATE_SCHEMA",
    "CodecHandoffEvidence",
    "HRun",
    "aggregate_h_runs",
    "load_h_run",
    "parse_arm_labels",
    "read_h_rows",
    "run_summary",
]

AGGREGATE_SCHEMA = "h-aggregate-v1"

_LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"
_STEP_SCHEMA = "h-training-step-v2"
_EVALUATION_SCHEMA = "h-predictive-evaluation-v1"
_COMMITTED = "committed"
_WARMUP = "codec_warmup"
_PREDICTION = "prediction"
_COMPLETE_RESULT = "probe_workload_complete"
# The scalar names of the step schema; pinned equal to the extractor's projection keys
# by tests/test_h_aggregate.py (a schema fact, not a second formula).
_STEP_SCALARS = (
    "objective",
    "aggregate_norm",
    "decoder_ce",
    "future_nll_per_dimension",
    "wall_seconds",
)
_FUTURE_NLL = "future_nll_per_dimension"

_IDENTITY = re.compile(r"[0-9a-f]{64}")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_PUBLICATION_KEY = re.compile(r"checkpoint/([0-9]{8})/publication_seconds")


@dataclass(frozen=True, slots=True)
class CodecHandoffEvidence:
    """What a verified codec hand-off tells the reader about one recipient (FOR-1692, D12).

    The fields are the ones ``ladder.run.codec_handoff.VerifiedHandoff`` carries, by the same
    names; tests/test_h_codec_handoff.py pins the two field sets equal. The type is declared
    HERE and not imported from the run side because this reader never imports the hand-off
    member (the analysis reader must not pull in the training engine). The caller converts
    ``admit_codec_pair``'s result; this module only checks the recipient's telemetry against it.
    """

    recipient_identity: str
    donor_identity: str
    warmup_steps: int
    record_digest: str
    donor_seal_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.recipient_identity,
            self.donor_identity,
            self.record_digest,
            self.donor_seal_digest,
        ):
            if type(value) is not str or _IDENTITY.fullmatch(value) is None:
                raise ValueError("hand-off evidence identities are 64 lowercase hex digests")
        if type(self.warmup_steps) is not int or self.warmup_steps < 1:
            raise ValueError("hand-off evidence needs a positive inherited warm-up length")


@dataclass(frozen=True, slots=True)
class HRun:
    """One admitted H run: its lineage, its global-axis series and its worker timings."""

    metrics_path: Path
    worker_log_path: Path
    profile_identity: str
    request_sha256: str
    run_start: dict[str, Any]
    run_end: dict[str, Any]
    runtime: dict[str, Any]
    warmup_steps: int
    # (global step, phase, projected scalars); a null future_nll is absent from the dict
    training: tuple[tuple[int, str, dict[str, float]], ...]
    uncommitted_step_rows: int
    # (global step, projected eval/* metrics), in stream order
    evaluations: tuple[tuple[int, dict[str, float]], ...]
    # global checkpoint step -> publication seconds (annotate rows)
    publication_seconds: dict[int, float]
    seal_steps: tuple[int, ...]
    mlflow_run_id: str | None
    stages: tuple[dict[str, Any], ...]
    admissions: tuple[dict[str, Any], ...]
    # (global step, E-LM projection of the same evaluation row), in stream order; one
    # entry per ``evaluations`` entry (h_elm_projection.project_elm, default denominator)
    elm: tuple[tuple[int, dict[str, Any]], ...]
    # set only for a codec-hand-off recipient admitted through ``load_h_run(handoff=...)``
    handoff: CodecHandoffEvidence | None = None


def _gap(message: str) -> NoReturn:
    raise TelemetryGap(message)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Strictly decode an already admitted JSON-lines file (duplicate keys and
    non-finite constants refused through the codec's own fail channel)."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue

            def fail(message: str, *, _number: int = number) -> NoReturn:
                _gap(f"{path}: line {_number}: {message}")

            value = strict_loads(raw, fail=fail)
            if not isinstance(value, dict):
                _gap(f"{path}: line {number} is not a JSON object")
            rows.append(value)  # pyright: ignore[reportUnknownArgumentType]
    return rows


def read_h_rows(path: str | Path) -> list[dict[str, Any]]:
    """Every row of one captured H JSON-lines file, through the existing reader.

    Well-formedness is :func:`require_json_object_lines`'s; the decode is this module's
    strict one (duplicate keys and non-finite constants refused). Raises
    :class:`TelemetryGap` naming the file; never returns a partial list.
    """
    path = Path(path)
    require_json_object_lines(path)
    return _read_rows(path)


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))  # pyright: ignore[reportArgumentType]
    except OverflowError:  # a JSON integer beyond float range is not a finite measurement
        return False


def _require_identity(value: object, *, where: str, path: Path) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        _gap(f"{path}: {where} must be 64 lowercase hex characters, got {value!r}")
    return value


def _scientifically_stopped_warmup(rows: Sequence[dict[str, Any]]) -> int | None:
    """The warm-up length ``W`` of a log that stopped before its warm-up-final seal, else ``None``.

    DH-46 (FOR-1693). ``W`` is ``step_to`` of EXACTLY ONE ``phase.start`` event whose phase is
    ``codec_warmup`` (the runner logs it BEFORE the first advance, so even a donor that
    committed no row carries it): an ``int`` >= 1. The log must also hold no prediction
    ``checkpoint.seal`` (the caller guarantees it), no ``codec_warmup`` ``checkpoint.seal`` at
    step ``W`` and no ``evaluation.start``. Anything else returns ``None`` and the caller
    raises its existing message.
    """
    starts = [r for r in rows if r.get("event") == "phase.start" and r.get("phase") == _WARMUP]
    if len(starts) != 1:
        return None
    declared = starts[0].get("step_to")
    if type(declared) is not int or declared < 1:
        return None
    for row in rows:
        if row.get("event") == "evaluation.start":
            return None
        if (
            row.get("event") == "checkpoint.seal"
            and row.get("phase") == _WARMUP
            and row.get("step") == declared
        ):
            return None
    return declared


def _holds_prediction_work(row: Mapping[str, Any]) -> bool:
    """A prediction-phase step row of any status, or an evaluation row (DH-46)."""
    schema = row.get("schema")
    return schema == _EVALUATION_SCHEMA or (
        schema == _STEP_SCHEMA and row.get("phase") == _PREDICTION
    )


def _worker_log_facts(
    path: Path, request_sha256: str, declared_warmup: int | None, *, scientific_stop: bool = False
) -> tuple[
    int,
    list[int],
    list[int],
    str | None,
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    bool,
]:
    """Bind the worker log to the run and read the global-axis facts out of it.

    Returns ``(warmup_steps, seal_steps, evaluation_steps, mlflow_run_id, stages,
    admissions, stopped_in_warmup)``. ``declared_warmup`` is a resumed lineage's warm-up
    length: this log's prediction seals, when it has any, must agree with it, and a log with
    none (a pod stopped inside the warm-up) is admitted at the declared length.

    AMENDMENT (house rule 4, FOR-1693, DH-46, 2026-11-01): ``scientific_stop`` says the run's
    ``run_end`` records a scientific stop. With it, no ``declared_warmup`` and no prediction
    seal, the warm-up length is the one ``phase.start`` event's ``step_to``
    (:func:`_scientifically_stopped_warmup`) and the last element is ``True``. Without those
    conditions the length is derived from prediction seals alone, exactly as the text above
    says, and a log with none raises the unchanged message.
    """
    rows = _read_rows(path)
    starts = [r for r in rows if r.get("event") == "worker.start"]
    if len(starts) != 1:
        _gap(f"{path}: exactly one worker.start event is required, found {len(starts)}")
    if starts[0].get("request_sha256") != request_sha256:
        _gap(
            f"{path}: worker.start.request_sha256 {starts[0].get('request_sha256')!r} does not "
            f"equal the metrics run_start.request_sha256 {request_sha256!r} -- this worker log "
            "belongs to a different run"
        )
    runs = [r for r in rows if r.get("event") == "mlflow.run"]
    if len(runs) > 1:
        _gap(f"{path}: at most one mlflow.run event is allowed, found {len(runs)}")
    mlflow_run_id: str | None = None
    if runs:
        run_id = runs[0].get("run_id")
        if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
            _gap(f"{path}: mlflow.run.run_id must be 32 lowercase hex characters, got {run_id!r}")
        mlflow_run_id = run_id

    seals = [r for r in rows if r.get("event") == "checkpoint.seal"]
    seal_steps: list[int] = []
    offsets: set[int] = set()
    for seal in seals:
        global_step, step = seal.get("global_step"), seal.get("step")
        if type(global_step) is not int or type(step) is not int:
            _gap(f"{path}: checkpoint.seal without integer step/global_step: {seal!r}")
        seal_steps.append(global_step)
        if seal.get("phase") == _PREDICTION:
            offsets.add(global_step - step)
    if declared_warmup is not None:
        if offsets - {declared_warmup}:
            _gap(
                f"{path}: prediction checkpoint.seal events put the warm-up at "
                f"{sorted(offsets)} steps, but the resumed lineage declares {declared_warmup}"
            )
        offsets = {declared_warmup}
    stopped_in_warmup = False
    if scientific_stop and declared_warmup is None and not offsets:
        stopped_length = _scientifically_stopped_warmup(rows)
        if stopped_length is not None:
            offsets = {stopped_length}
            stopped_in_warmup = True
    if len(offsets) != 1:
        _gap(
            f"{path}: the warm-up length cannot be derived: prediction checkpoint.seal events "
            f"must exist and agree on global_step - step, found offsets {sorted(offsets)}"
        )
    warmup_steps = offsets.pop()
    if warmup_steps < 0:
        _gap(f"{path}: derived warm-up length {warmup_steps} is negative")

    evaluation_steps: list[int] = []
    for event in rows:
        if event.get("event") == "evaluation.start":
            global_step = event.get("global_step")
            if type(global_step) is not int:
                _gap(f"{path}: evaluation.start without an integer global_step: {event!r}")
            evaluation_steps.append(global_step)

    # the emitter writes stage and caller as strings, seconds as a float and the three
    # admission counts as ints on every ``.end`` row; any other shape is a corrupt log,
    # refused here so ``eval h`` never prints a final leg that silently lacks a row
    stages: list[dict[str, Any]] = []
    admissions: list[dict[str, Any]] = []
    for row in rows:
        event = row.get("event")
        if event == "stage.end":
            stage = row.get("stage")
            if not isinstance(stage, str) or not _finite_number(row.get("seconds")):
                _gap(f"{path}: stage.end without a string stage and finite seconds: {row!r}")
            stages.append({"seconds": float(row["seconds"]), "stage": stage})
        elif event == "admission.end":
            caller = row.get("caller")
            counts = {k: row.get(k) for k in ("evaluations", "journals_read", "seals_walked")}
            if (
                not isinstance(caller, str)
                or not _finite_number(row.get("seconds"))
                or any(type(v) is not int for v in counts.values())
            ):
                _gap(
                    f"{path}: admission.end without a string caller, finite seconds and "
                    f"integer evaluations/journals_read/seals_walked counts: {row!r}"
                )
            admissions.append({"caller": caller, "seconds": float(row["seconds"]), **counts})
    return (
        warmup_steps,
        seal_steps,
        evaluation_steps,
        mlflow_run_id,
        tuple(stages),
        tuple(admissions),
        stopped_in_warmup,
    )


def _check_step_scalars(row: dict[str, Any], phase: str, path: Path) -> None:
    for key in _STEP_SCALARS:
        value = row.get(key)
        if value is None:
            if key == _FUTURE_NLL and phase == _WARMUP:
                continue  # structurally absent in the codec warm-up: admissible only here
            _gap(f"{path}: step {row.get('step')} ({phase}) has a null {key}")
        if not _finite_number(value):
            _gap(f"{path}: step {row.get('step')} ({phase}) has a non-finite {key}: {value!r}")


# the one gap the reader admits; the predicate lives in h_elm_projection, this name is
# pinned by tests/test_h_final_probe_checks.py against that module's mirror
_ZERO_MASS_GAP = ZERO_MASS_GAP


def _check_evaluation_numbers(row: dict[str, Any], path: Path, warmup_steps: int) -> None:
    step = row.get("step")
    # the row's step is phase-local; refusals name the global step the metrics land on
    where = (
        f"evaluation at global step {step + warmup_steps}"
        if type(step) is int
        else f"evaluation at step {step!r}"
    )
    examples = row.get("examples")
    if not isinstance(examples, list) or not examples:
        _gap(f"{path}: {where} carries no examples")
    for example in examples:  # pyright: ignore[reportUnknownVariableType]
        by_count = example.get("by_sample_count") if isinstance(example, dict) else None
        if not isinstance(by_count, dict) or not by_count:
            _gap(f"{path}: {where} has an example without by_sample_count")
        for entries in by_count.values():  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(entries, list) or not entries:
                _gap(f"{path}: {where} has an empty sample-count entry")
            for entry in entries:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(entry, dict):
                    _gap(f"{path}: {where} has a non-object entry")
                require_importance_entry(entry, f"{path}: {where}")


def load_h_run(
    metrics_path: str | Path,
    worker_log_path: str | Path,
    *,
    warmup_steps: int | None = None,
    handoff: CodecHandoffEvidence | None = None,
) -> HRun:
    """Admit one captured H run and rebuild its global step axis.

    Raises :class:`TelemetryGap` naming the file for every inadmissible or
    self-contradicting input; never returns a partial run.

    ``warmup_steps`` is given ONLY for one pod of a resumed lineage (by
    :func:`~entropic_substrate.ladder.analyze.h_resumed_run.load_h_resumed_run`): the
    warm-up length is then the lineage's, and the full-warm-up coverage check moves to
    the caller, which checks it over the union of the pods (a resumed pod that starts
    in prediction commits no warm-up row, and that is expected, not a gap). Every other
    admission check is unchanged. The default ``None`` is the single-pod reader.

    ``handoff`` is given ONLY for a codec-hand-off recipient (FOR-1692, D12): a cell whose
    codec was materialized from its donor's warm-up-final seal commits NO ``codec_warmup``
    row of its own, and that is expected, not a gap. It is admitted only when it has exactly
    zero committed warm-up rows, the warm-up length the worker log derives from the prediction
    seals equals the evidence's, and ``run_start.profile_identity`` is the evidence's
    recipient. It is a distinct typed argument and never combines with ``warmup_steps``: that
    override skips the per-file coverage check by design and must not admit a truncated stream.
    Without either, a stream with no warm-up rows still raises, exactly as before.

    AMENDMENT (house rule 4, FOR-1693, DH-46, 2026-11-01): the default path (neither
    argument) also admits a scientifically stopped warm-up-only stream. The admission is keyed
    on the recorded stop, never on a row count: the ``run_end`` carries ``status == failed``
    and ``result_status == scientific_gap``; the worker log holds no prediction
    ``checkpoint.seal``, no ``codec_warmup`` seal at the warm-up's last step and no
    ``evaluation.start``; the warm-up length ``W`` is ``step_to`` of the one ``codec_warmup``
    ``phase.start`` event; and the metrics hold no prediction step row, no evaluation row and
    committed warm-up rows exactly ``1..k`` with ``k < W`` (zero rows allowed). Such a run
    loads with its warm-up rows only and no evaluations. Every other stream with no prediction
    seal raises the unchanged messages, and the ``warmup_steps=`` and ``handoff=`` paths never
    combine with this branch. A stopped stream that HAS its warm-up final seal stays refused.

    RETRACTION (house rule 4, 2026-11-03, PR #430 review): the sentence above said the
    ``warmup_steps=`` and ``handoff=`` paths were "byte-unchanged" by DH-46. They were not:
    :func:`final_h_run_end` waives its committed-row check for every caller once the run_end
    records a scientific stop, so a stopped pod with ZERO committed rows and one prediction
    seal loaded under EITHER override with an empty training axis (measured on both paths).
    Without a prediction seal the ``handoff=`` path refused, but with "the warm-up length
    cannot be derived", not "the run never trained". Both overrides now refuse a pod with no
    committed row with the reader's never-trained text; the seal-less hand-off case keeps its
    earlier derivation refusal.
    """
    if handoff is not None and type(handoff) is not CodecHandoffEvidence:
        raise TypeError("handoff evidence must be a CodecHandoffEvidence")
    if handoff is not None and warmup_steps is not None:
        raise ValueError("hand-off evidence and a warmup_steps override are mutually exclusive")
    metrics = Path(metrics_path)
    worker_log = Path(worker_log_path)
    run_end = final_h_run_end(metrics)
    require_json_object_lines(worker_log)
    rows = _read_rows(metrics)

    start = next(
        r for r in rows if r.get("schema") == _LIFECYCLE_SCHEMA and r.get("event") == "run_start"
    )
    identity = _require_identity(
        start.get("profile_identity"), where="run_start.profile_identity", path=metrics
    )
    runtime: object = start.get("runtime")
    request_sha256 = _require_identity(
        start.get("request_sha256"), where="run_start.request_sha256", path=metrics
    )
    for row in rows:
        if "run_identity" in row and row["run_identity"] != identity:
            _gap(
                f"{metrics}: a row carries run_identity {row['run_identity']!r}, but "
                f"run_start.profile_identity is {identity!r}"
            )

    declared_warmup = warmup_steps
    (
        warmup_steps,
        seal_steps,
        evaluation_starts,
        mlflow_run_id,
        stages,
        admissions,
        stopped_in_warmup,
    ) = _worker_log_facts(
        worker_log,
        request_sha256,
        declared_warmup,
        scientific_stop=handoff is None
        and is_scientific_stop(run_end)
        and not any(_holds_prediction_work(row) for row in rows),
    )

    extractor = HTrainingStepExtractor(warmup_steps=warmup_steps)
    training: list[tuple[int, str, dict[str, float]]] = []
    evaluations: list[tuple[int, dict[str, float]]] = []
    elm: list[tuple[int, dict[str, Any]]] = []
    publication: dict[int, float] = {}
    uncommitted = 0
    warmup_rows = 0
    warmup_max = 0
    previous = 0
    for row in rows:
        schema = row.get("schema")
        if schema == _STEP_SCHEMA:
            if row.get("run_identity") != identity:
                _gap(f"{metrics}: step row {row.get('step')} carries no run_identity of this run")
            if row.get("status") != _COMMITTED:
                uncommitted += 1
                continue
            phase = row.get("phase")
            if phase not in (_WARMUP, _PREDICTION):
                _gap(f"{metrics}: committed row {row.get('step')} has no usable phase: {phase!r}")
            _check_step_scalars(row, phase, metrics)
            projected = extractor.extract(row)
            if projected is None:
                _gap(f"{metrics}: committed row {row.get('step')} ({phase}) has no projection")
            step = projected.step
            if step <= previous:
                _gap(
                    f"{metrics}: the reconstructed global step axis is not strictly increasing "
                    f"at {phase} step {row.get('step')} (global {step} after {previous})"
                )
            previous = step
            if phase == _WARMUP:
                warmup_rows += 1
                warmup_max = max(warmup_max, step)
            scalars = {k: v for k, v in projected.metrics.items() if k != "phase_index"}
            training.append((step, phase, scalars))
        elif schema == _EVALUATION_SCHEMA:
            if row.get("run_identity") != identity:
                _gap(f"{metrics}: evaluation row carries no run_identity of this run")
            _check_evaluation_numbers(row, metrics, warmup_steps)
            projected = extractor.extract(row)
            if projected is None or not projected.metrics:
                _gap(f"{metrics}: evaluation at step {row.get('step')} has no projection")
            evaluations.append((projected.step, dict(projected.metrics)))
            where = f"{metrics}: evaluation at global step {projected.step}"
            elm.append((projected.step, project_elm(row, where=where)))
        elif row.get("event") == "annotate":
            for key, value in row.items():
                match = _PUBLICATION_KEY.fullmatch(key)
                if match is None:
                    continue
                step = int(match.group(1))
                if step in publication:
                    _gap(f"{metrics}: duplicate publication timing for checkpoint {step}")
                try:
                    seconds = float(value) if isinstance(value, str) else math.nan
                except ValueError:
                    seconds = math.nan
                if not math.isfinite(seconds):
                    _gap(
                        f"{metrics}: checkpoint {step} publication_seconds {value!r} is not a "
                        "finite number string"
                    )
                publication[step] = seconds

    if handoff is not None:
        if identity != handoff.recipient_identity:
            _gap(
                f"{metrics}: run_start.profile_identity {identity!r} is not the hand-off "
                f"recipient {handoff.recipient_identity!r}"
            )
        if warmup_steps != handoff.warmup_steps:
            _gap(
                f"{metrics}: the worker log puts the warm-up at {warmup_steps} steps, but the "
                f"hand-off evidence inherited {handoff.warmup_steps}"
            )
        if warmup_rows != 0:
            _gap(
                f"{metrics}: a hand-off recipient commits no codec_warmup row, but "
                f"{warmup_rows} are present"
            )
    elif stopped_in_warmup:
        # DH-46: the committed warm-up rows are exactly steps 1..k with k < W (consecutive
        # because the axis is strictly increasing from zero), never a complete warm-up.
        if not warmup_rows == warmup_max < warmup_steps:
            _gap(
                f"{metrics}: the worker log puts the warm-up at {warmup_steps} steps and the stop "
                f"inside it, but the committed codec_warmup rows number {warmup_rows} and end at "
                f"{warmup_max}"
            )
    elif declared_warmup is not None:
        if warmup_max > warmup_steps:
            _gap(
                f"{metrics}: a committed codec_warmup row ends at {warmup_max}, beyond the "
                f"resumed lineage's warm-up of {warmup_steps} steps"
            )
    elif warmup_rows != warmup_steps or warmup_max != warmup_steps:
        _gap(
            f"{metrics}: the worker log puts the warm-up at {warmup_steps} steps, but the "
            f"committed codec_warmup rows number {warmup_rows} and end at {warmup_max}"
        )
    if (handoff is not None or declared_warmup is not None) and not training:
        # the scientific-stop waiver in final_h_run_end lets a zero-row stream reach here; an
        # override derives the warm-up from a prediction seal and must not admit it empty
        _gap(
            f"{metrics}: an admissible H run holds at least one committed {_STEP_SCHEMA} "
            "row, found none — the run never trained"
        )
    evaluation_steps = [step for step, _ in evaluations]
    if any(later <= earlier for earlier, later in pairwise(evaluation_steps)):
        _gap(
            f"{metrics}: evaluation rows at global steps {evaluation_steps} are not strictly "
            "increasing; a repeated global step would be silently collapsed"
        )
    if [step for step, _ in evaluations] != evaluation_starts:
        _gap(
            f"{metrics}: evaluation rows at global steps {[s for s, _ in evaluations]} do not "
            f"match the evaluation.start events {evaluation_starts} of {worker_log} one to one"
        )
    unsealed = sorted(set(publication) - set(seal_steps))
    if unsealed:
        _gap(
            f"{metrics}: publication timing for checkpoints {unsealed} has no matching "
            "checkpoint.seal in the worker log"
        )
    return HRun(
        metrics_path=metrics,
        worker_log_path=worker_log,
        profile_identity=identity,
        request_sha256=request_sha256,
        run_start={k: v for k, v in start.items() if k != "runtime"},
        run_end=dict(run_end),
        runtime=dict(runtime) if isinstance(runtime, dict) else {},
        warmup_steps=warmup_steps,
        training=tuple(training),
        uncommitted_step_rows=uncommitted,
        evaluations=tuple(evaluations),
        publication_seconds=publication,
        seal_steps=tuple(seal_steps),
        mlflow_run_id=mlflow_run_id,
        stages=stages,
        admissions=admissions,
        elm=tuple(elm),
        handoff=handoff,
    )


def parse_arm_labels(pairs: Sequence[str]) -> dict[str, str]:
    """``identity=label`` strings to ``{identity: label}``; any malformed, duplicate or
    ambiguous pair raises :class:`ValueError` naming it."""
    labels: dict[str, str] = {}
    used: dict[str, str] = {}
    for pair in pairs:
        identity, separator, label = pair.partition("=")
        if not separator or not _IDENTITY.fullmatch(identity):
            raise ValueError(
                f"--arm {pair!r}: expected <64 lowercase hex profile identity>=<label>"
            )
        if not _LABEL.fullmatch(label):
            raise ValueError(
                f"--arm {identity}: label {label!r} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
            )
        if identity in labels:
            raise ValueError(f"--arm {identity}: identity given more than once")
        if label in used:
            raise ValueError(
                f"--arm {identity}: label {label!r} is already the label of {used[label]}"
            )
        labels[identity] = label
        used[label] = identity
    return labels


def _bind_labels(runs: Sequence[HRun], labels: Mapping[str, str]) -> dict[str, HRun]:
    by_identity: dict[str, HRun] = {}
    for run in runs:
        if run.profile_identity in by_identity:
            raise ValueError(f"run {run.profile_identity} was given more than once")
        by_identity[run.profile_identity] = run
    unknown = sorted(set(labels) - set(by_identity))
    if unknown:
        raise ValueError(f"--arm names an identity no input run carries: {unknown[0]}")
    unlabelled = sorted(set(by_identity) - set(labels))
    if unlabelled:
        raise ValueError(f"input run {unlabelled[0]} has no --arm label")
    return {labels[identity]: run for identity, run in by_identity.items()}


def _nll_summary(run: HRun) -> dict[str, Any]:
    values = [s[_FUTURE_NLL] for _, _, s in run.training if _FUTURE_NLL in s]
    return {
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "negative_rows": sum(1 for v in values if v < 0),
        "null_rows": sum(1 for _, _, s in run.training if _FUTURE_NLL not in s),
        "rows": len(values),
    }


def run_summary(run: HRun) -> dict[str, Any]:
    """One run's lineage, status and axis summary.

    Every value is read from the files. ``completed`` is derived, not copied: the worker
    writes status ``ok`` for every non-failure terminal (complete, paused, stopped), so it
    is ``True`` only when the result status is also ``probe_workload_complete``.
    """
    steps = [step for step, _, _ in run.training]
    first_prediction = next((s for s, phase, _ in run.training if phase == _PREDICTION), None)
    denominators = sorted({projection["denominator"] for _, projection in run.elm})
    return {
        "checkpoint_steps": sorted(run.publication_seconds),
        "committed_step_rows": len(run.training),
        "completed": run.run_end.get("status") == "ok"
        and run.run_end.get("result_status") == _COMPLETE_RESULT,
        "elm": {
            "denominator": denominators[0] if len(denominators) == 1 else denominators,
            "headline_samples": HEADLINE_SAMPLES,
            "headline_steps": [step for step, p in run.elm if p["headline"]["present"]],
            "samples": sorted({int(k) for _, p in run.elm for k in p["samples"]}),
        },
        "evaluation_steps": [step for step, _ in run.evaluations],
        "first_prediction_step": first_prediction,
        "future_nll_per_dimension": _nll_summary(run),
        "lineage": {
            "code_commit": run.run_start.get("code_commit"),
            "image": run.run_start.get("image"),
            "run_class": run.run_start.get("run_class"),
        },
        "mlflow_run_id": run.mlflow_run_id,
        "profile_identity": run.profile_identity,
        "request_sha256": run.request_sha256,
        "run_end": {
            "result_status": run.run_end.get("result_status"),
            "status": run.run_end.get("status"),
            "successful_optimizer_steps": run.run_end.get("successful_optimizer_steps"),
            "wall_seconds": run.run_end.get("wall_seconds"),
        },
        "step_range": [steps[0], steps[-1]] if steps else None,
        "uncommitted_step_rows": run.uncommitted_step_rows,
        "warmup_steps": run.warmup_steps,
    }


def _paired(
    series: Mapping[str, Mapping[int, object]],
) -> tuple[list[int], dict[str, list[int]]]:
    """Steps present on every arm, and per arm the steps it holds that others lack."""
    common = sorted(set.intersection(*(set(s) for s in series.values())))
    unpaired = {label: sorted(set(s) - set(common)) for label, s in series.items()}
    return common, unpaired


def _metric_order(name: str) -> tuple[str, int]:
    """``eval/<quantity>_k<k>`` keys ordered by quantity, then numerically by k."""
    stem, _, k = name.rpartition("_k")
    return (stem, int(k)) if k.isdigit() else (name, 0)


def _one_denominator(elm: Mapping[str, Mapping[int, Mapping[str, Any]]]) -> str:
    found = {p["denominator"] for e in elm.values() for p in e.values()}
    if len(found) > 1:
        raise ValueError(f"arms were projected with different E-LM denominators {sorted(found)}")
    return found.pop() if found else DEFAULT_DENOMINATOR


def aggregate_h_runs(runs: Sequence[HRun], labels: Mapping[str, str]) -> dict[str, Any]:
    """Pair the labelled arms by global step.

    ``labels`` maps each run's full profile identity to its arm label; an unknown
    identity, an unlabelled run or a duplicate raises :class:`ValueError`. Steps held
    by only some arms are listed under ``unpaired``, never silently truncated.
    """
    if not runs:
        raise ValueError("aggregate needs at least one captured run")
    arms = _bind_labels(runs, labels)
    order = sorted(arms)

    train: dict[str, dict[int, dict[str, float]]] = {
        label: {step: scalars for step, _, scalars in arm.training} for label, arm in arms.items()
    }
    train_steps, train_unpaired = _paired(dict(train.items()))
    metrics = sorted(
        {m for arm in arms.values() for _, _, s in arm.training for m in s} | {_FUTURE_NLL}
    )
    training_doc: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        training_doc[metric] = [
            {"step": step, "values": {label: train[label][step].get(metric) for label in order}}
            for step in train_steps
        ]
    # one step map per arm; the global axis is strictly increasing (load_h_run refuses
    # otherwise), so a step maps to exactly one phase and the lookup is O(1)
    phases: dict[str, dict[int, str]] = {
        label: {step: phase for step, phase, _ in arm.training} for label, arm in arms.items()
    }
    phase_index = [
        {"step": step, "values": {label: phases[label][step] for label in order}}
        for step in train_steps
    ]

    evals: dict[str, dict[int, dict[str, float]]] = {
        label: dict(arm.evaluations) for label, arm in arms.items()
    }
    eval_steps, eval_unpaired = _paired(dict(evals.items()))
    eval_metrics = sorted({m for e in evals.values() for s in e.values() for m in s})
    evaluation_doc = {
        metric: [
            {"step": step, "values": {label: evals[label][step].get(metric) for label in order}}
            for step in eval_steps
        ]
        for metric in eval_metrics
    }

    elm: dict[str, dict[int, dict[str, Any]]] = {
        label: dict(arm.elm) for label, arm in arms.items()
    }
    elm_metrics = sorted(
        {m for e in elm.values() for p in e.values() for m in p["metrics"]},
        key=_metric_order,
    )
    evaluation_elm = {
        "denominator": _one_denominator(elm),
        "headline": {
            label: [
                {"step": step, **elm[label][step]["headline"]}
                for step in eval_steps
                if elm[label][step]["headline"]["present"]
            ]
            for label in order
        },
        "headline_samples": HEADLINE_SAMPLES,
        "metrics": {
            metric: [
                {
                    "step": step,
                    "values": {label: elm[label][step]["metrics"].get(metric) for label in order},
                }
                for step in eval_steps
            ]
            for metric in elm_metrics
        },
    }

    pub: dict[str, dict[int, float]] = {
        label: dict(arm.publication_seconds) for label, arm in arms.items()
    }
    pub_steps, pub_unpaired = _paired(dict(pub.items()))

    return {
        "arms": {label: run_summary(arms[label]) for label in order},
        "evaluation": {
            "elm": evaluation_elm,
            "metrics": evaluation_doc,
            "paired_steps": eval_steps,
            "unpaired": eval_unpaired,
        },
        "phase_index": phase_index,
        "publication_seconds": {
            "paired_steps": pub_steps,
            "unpaired": pub_unpaired,
            "values": [
                {"step": step, "values": {label: pub[label][step] for label in order}}
                for step in pub_steps
            ],
        },
        "schema": AGGREGATE_SCHEMA,
        "training": {
            "metrics": training_doc,
            "paired_steps": train_steps,
            "unpaired": train_unpaired,
        },
    }
