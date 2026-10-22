"""H0 final-probe check suite: one machine verdict per H0 exit (K1-K7).

Each check is a pure function of evidence a probe leaves behind (receipts, controller
records, captured ``metrics.jsonl`` / ``worker-log.jsonl``, a generation receipt) and
returns a :class:`CheckResult` whose verdict is ``PASS``, ``FAIL`` or ``UNPROVEN``. Only
``PASS`` passes; ``UNPROVEN`` means the evidence the check needs is structurally absent
from this image (ruling D1: stochastic rollout), never "probably fine".

Admission is always the existing reader's (``read_training_progress``, ``load_h_run``,
``final_h_run_end``, ``project_worker_stage_record``); a reader's refusal of bad evidence
(``ValueError`` / ``TelemetryGap``) is a FAIL carrying the reader's own message. Anything
else propagates: a defect in this module or a reader is not a verdict. Nothing here
prints; ``scripts/h0_final_probe_checks.py`` is the runnable entry (ruling D6).

The criteria, ruled 2026-10-22 (brief ``h0-final-probe-checks.md``):

* K1 chain: from the completion receipt, every ``parent_receipt`` hop admits through
  ``read_training_progress``; ``successful_optimizer_steps`` strictly decreases along the
  chain; the declared stopped receipt (path AND sha256) is in it and is ``paused``; the
  final receipt is ``training_complete`` at ``warmup_steps + steps``.
* K1 request: the resumed request binds the collected ``resume_artifact`` through the
  existing ``resume`` field, with the same profile and a fresh nonce (D4).
* K1 seam: the stopped pod and its continuation admit as ONE run on a continuous global
  step axis (:func:`~entropic_substrate.ladder.analyze.h_resumed_run.load_h_resumed_run`, D3).
* K2: one ``terminal_collected`` controller record per declared arm, workload and
  termination verified, empty ``error_types``, native completion receipt verified, and
  distinct carriers across arms.
* K3: every captured ``metrics.jsonl`` passes ``final_h_run_end`` and its
  ``run_start.request_sha256`` is a declared request, each exactly once.
* K4 (D5): every admission site the log declares has exactly one ``admission.end`` that
  projects through the closed worker-stage vocabulary; every ``memo=shared, root=run`` row
  reads 0 evaluations from disk (``evaluations_memo_hits`` are reported, never counted);
  fresh walks are reported, unbounded.
* K5 (D1): the boundary arm checks the rollout's terminal outcome against its
  observations (semantics of ``core.sentence.rollout``: 256 ends a sentence, 257 the
  document); the stochasticity arm is PROVEN only when a recorded sampling law carries
  ``continuations > 1``, else UNPROVEN.
* K6 (D2): every evaluation example's ``by_sample_count`` key set EQUALS the profile's
  ``evaluated_samples``; every finite entry's effective sample size lies in ``[1, k]``
  (an addition to the brief's key-set criterion: true by construction for normalized
  importance weights, so a violation is a defect).
* K7: every arm admits through the analysis surface (single pod or resumed lineage),
  reports, and aggregates with the other arms.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.analyze.h_aggregate import (
    HRun,
    aggregate_h_runs,
    load_h_run,
    read_h_rows,
)
from entropic_substrate.ladder.analyze.h_report import h_report_from_run
from entropic_substrate.ladder.analyze.h_resumed_run import load_h_resumed_run
from entropic_substrate.ladder.capture.reader.h_telemetry import final_h_run_end
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
from entropic_substrate.ladder.run.journal import artifact_reference
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import SealMemo
from entropic_substrate.ladder.run.worker_log_events import (
    ADMISSION_CALLERS,
    ADMISSION_END,
    STAGE_END,
    project_worker_stage_record,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest

__all__ = [
    "CheckResult",
    "Verdict",
    "check_analysis_surface",
    "check_capture",
    "check_completion",
    "check_generation",
    "check_importance_curve",
    "check_read_floor",
    "check_resume_chain",
    "check_resume_request",
    "check_resume_seam",
    "evaluated_samples_from_manifest",
]

type Verdict = Literal["PASS", "FAIL", "UNPROVEN"]

_COLLECTION_SCHEMA = "h-probe-terminal-collection-v1"
_RESUMABLE_CAPTURES = ("pause_bytes_verified", "stop_bytes_verified")
_CONTROLLER_SCHEMA = "h-probe-controller-result-v1"
_LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"
_EVALUATION_SCHEMA = "h-predictive-evaluation-v1"
# The generation receipt schemas this check reads. The DG-A "receipt v3" ticket (ruling
# D1) adds its schema id here together with its sampling-law fields.
_GENERATION_SCHEMAS = ("h-generation-receipt-v2",)
_SENTENCE_END, _DOCUMENT_END = 256, 257  # core/sentence/boundary.py
# Terminal rollout outcomes (core/sentence/rollout.py RolloutStatus minus "active", plus
# generation.py's "sentence_budget"), grouped by what their last observation must be.
_ENDS_ON_EMISSION = ("sentence_budget", "document_end", "context_limit")
_ENDS_ON_REFUSED_BOUNDARY = ("empty_sentence", "invalid_utf8")
_ENDS_ON_BYTE = ("length_limit",)
# Provisional sampling-law field names (ruling D1; owned by the receipt-v3 ticket).
_SAMPLING_FIELDS = ("sampler", "temperature", "continuations")
_SAMPLING_SEED_FIELDS = ("seed", "rng_digest")
_ESS_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check's verdict, the reasons behind a non-PASS, and the facts it read."""

    check: str
    verdict: Verdict
    reasons: tuple[str, ...]
    facts: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "facts": self.facts,
            "reasons": list(self.reasons),
            "verdict": self.verdict,
        }


def _result(
    check: str,
    failures: Sequence[str],
    facts: dict[str, Any],
    unproven: Sequence[str] = (),
) -> CheckResult:
    if failures:
        return CheckResult(check, "FAIL", tuple(failures), facts)
    if unproven:
        return CheckResult(check, "UNPROVEN", tuple(unproven), facts)
    return CheckResult(check, "PASS", (), facts)


def _finite(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))  # pyright: ignore[reportArgumentType]
    except OverflowError:  # a JSON integer beyond float range is not a finite number
        return False


# ------------------------------------------------------------------ K1
def check_resume_chain(
    run_root: Path,
    final_receipt: ArtifactIdentity,
    stopped_receipt: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> CheckResult:
    """K1 chain: the completion receipt reaches the declared stopped receipt."""
    failures: list[str] = []
    stopped = artifact_reference(run_root, stopped_receipt)
    memo = SealMemo()
    steps: list[int] = []
    statuses: list[object] = []
    stopped_hop: int | None = None
    seen: set[tuple[str, str]] = set()
    current: ArtifactIdentity | None = final_receipt
    while current is not None:
        hop = len(steps)
        reference = artifact_reference(run_root, current)
        key = (reference["path"], reference["sha256"])
        if key in seen:
            failures.append(f"hop {hop}: the chain revisits {reference['path']}")
            break
        seen.add(key)
        try:
            row = read_training_progress(
                run_root,
                current,
                profile,
                source_root,
                max_source_bytes=max_source_bytes,
                max_artifact_bytes=max_artifact_bytes,
                memo=memo,
            )
        except ValueError as error:
            failures.append(f"hop {hop} ({reference['path']}): {error}")
            break
        if reference == stopped and stopped_hop is None:
            stopped_hop = hop
        steps.append(int(row["successful_optimizer_steps"]))  # pyright: ignore[reportArgumentType]
        statuses.append(row["status"])
        parent = row["parent_receipt"]
        current = None if parent is None else artifact_from_mapping(parent)
    if any(later >= earlier for earlier, later in pairwise(steps)):
        failures.append(
            f"successful_optimizer_steps along the chain are not strictly decreasing: {steps}"
        )
    if stopped_hop is None:
        failures.append(
            f"the chain never reaches the declared stopped receipt {stopped['path']} "
            f"(sha256 {stopped['sha256']})"
        )
    elif statuses[stopped_hop] != "paused":
        failures.append(f"the declared stopped receipt has status {statuses[stopped_hop]!r}")
    budget = profile.schedule.warmup_steps + profile.schedule.steps
    if statuses and (statuses[0] != "training_complete" or steps[0] != budget):
        failures.append(
            f"the final receipt is {statuses[0]!r} at {steps[0]} optimizer steps, not "
            f"training_complete at {budget}"
        )
    facts = {
        "chain_optimizer_steps": steps,
        "chain_statuses": statuses,
        "hops": len(steps),
        "stopped_hop": stopped_hop,
    }
    return _result("k1_chain", failures, facts)


def check_resume_request(
    collection: Mapping[str, object], previous: ProbeRequest, resumed: ProbeRequest
) -> CheckResult:
    """K1 request: the resumed request continues the collected stop through ``resume``."""
    failures: list[str] = []
    if (
        collection.get("schema") != _COLLECTION_SCHEMA
        or collection.get("request_sha256") != previous.digest
    ):
        failures.append(
            f"the collection (request {collection.get('request_sha256')!r}) belongs to "
            f"another request than the stopped one ({previous.digest})"
        )
    if collection.get("capture_status") not in _RESUMABLE_CAPTURES:
        failures.append(
            f"capture_status {collection.get('capture_status')!r} is not a stop or pause"
        )
    if resumed.resume is None:
        failures.append("the resumed request carries no resume artifact")
    else:
        try:
            declared = artifact_from_mapping(collection.get("resume_artifact"))
        except ValueError as error:
            failures.append(f"the collection's resume_artifact is malformed: {error}")
        else:
            if resumed.resume != declared:
                failures.append(
                    f"the request's resume differs from the collected resume_artifact: "
                    f"{resumed.resume} != {declared}"
                )
    if resumed.nonce == previous.nonce:
        failures.append("the resumed request reuses the stopped request's nonce")
    if resumed.profile != previous.profile:
        failures.append("the resumed request runs another profile")
    facts = {
        "previous_request_sha256": previous.digest,
        "resumed_request_sha256": resumed.digest,
        "resume": None if resumed.resume is None else resumed.resume.path,
    }
    return _result("k1_request", failures, facts)


def check_resume_seam(segments: Sequence[tuple[Path, Path]]) -> CheckResult:
    """K1 seam: the pods admit as one run on a continuous global step axis."""
    try:
        resumed = load_h_resumed_run(segments)
    except TelemetryGap as gap:
        return _result("k1_seam", [str(gap)], {"segments": len(segments)})
    pods = [[pod.training[0][0], pod.training[-1][0]] for pod in resumed.segments]
    facts = {
        "profile_identity": resumed.run.profile_identity,
        "request_sha256": [pod.request_sha256 for pod in resumed.segments],
        "segments": pods,
        "warmup_steps": resumed.run.warmup_steps,
    }
    return _result("k1_seam", [], facts)


# ------------------------------------------------------------------ K2
def _record_failures(record: Mapping[str, object], arm: str) -> list[str]:
    failures: list[str] = []
    if record.get("schema") != _CONTROLLER_SCHEMA:
        failures.append(f"arm {arm}: schema {record.get('schema')!r} is not {_CONTROLLER_SCHEMA}")
    if record.get("status") != "terminal_collected":
        failures.append(f"arm {arm}: status {record.get('status')!r}, not 'terminal_collected'")
    for flag in ("workload_verified", "termination_verified"):
        if record.get(flag) is not True:
            failures.append(f"arm {arm}: {flag} is {record.get(flag)!r}")
    if record.get("error_types") != []:
        failures.append(f"arm {arm}: error_types {record.get('error_types')!r} is not empty")
    collection = record.get("collection")
    capture = collection.get("capture_status") if isinstance(collection, Mapping) else None
    if capture != "native_completion_receipt_verified":
        failures.append(f"arm {arm}: collection capture_status {capture!r} is not a completion")
    return failures


def check_completion(
    records: Sequence[Mapping[str, object]], requests: Sequence[ProbeRequest]
) -> CheckResult:
    """K2: every declared arm completed and was collected, with distinct carriers."""
    failures: list[str] = []
    carriers = [request.profile.spine.carrier for request in requests]
    if len(set(carriers)) != len(carriers):
        failures.append(f"the arms do not run distinct carriers: {carriers}")
    declared = {request.digest for request in requests}
    if len(declared) != len(requests):
        failures.append("two arms share one request digest")
    by_digest: dict[object, list[Mapping[str, object]]] = {}
    for record in records:
        by_digest.setdefault(record.get("request_sha256"), []).append(record)
    for digest in sorted(str(d) for d in set(by_digest) - declared):
        failures.append(f"a controller record names an undeclared request {digest}")
    for request, carrier in zip(requests, carriers, strict=True):
        arm = f"{carrier} ({request.digest})"
        found = by_digest.get(request.digest, [])
        if len(found) != 1:
            failures.append(
                f"arm {arm}: no controller record"
                if not found
                else f"arm {arm}: {len(found)} controller records, expected one"
            )
            continue
        failures.extend(_record_failures(found[0], arm))
    facts = {"arms": len(requests), "carriers": carriers, "records": len(records)}
    return _result("k2", failures, facts)


# ------------------------------------------------------------------ K3
def check_capture(metrics: Sequence[Path], request_digests: Sequence[str]) -> CheckResult:
    """K3: each captured metrics file is admissible and bound to a declared request."""
    failures: list[str] = []
    observed: list[str] = []
    files: list[dict[str, Any]] = []
    for path in metrics:
        try:
            run_end = final_h_run_end(path)
            start = next(
                row
                for row in read_h_rows(path)
                if row.get("schema") == _LIFECYCLE_SCHEMA and row.get("event") == "run_start"
            )
        except TelemetryGap as gap:
            failures.append(str(gap))
            continue
        digest = start.get("request_sha256")
        if digest not in request_digests:
            failures.append(
                f"{path}: run_start.request_sha256 {digest!r} is not a declared request"
            )
        observed.append(str(digest))
        files.append(
            {
                "path": str(path),
                "request_sha256": digest,
                "result_status": run_end.get("result_status"),
                "status": run_end.get("status"),
            }
        )
    for digest, count in sorted(Counter(observed).items()):
        if count > 1:
            failures.append(f"request {digest} has {count} captured metrics files")
    for digest in sorted(set(request_digests) - set(observed)):
        failures.append(f"declared request {digest} has no admissible capture")
    return _result("k3", failures, {"files": files})


# ------------------------------------------------------------------ K4
def _read_floor_of(path: Path, failures: list[str]) -> dict[str, Any]:
    rows = read_h_rows(path)
    admitted: list[dict[str, Any]] = []
    for row in rows:
        if row.get("event") != ADMISSION_END:
            continue
        projected = project_worker_stage_record(row)
        if projected is None:
            failures.append(
                f"{path}: admission.end {row.get('caller')!r} does not project through the "
                f"closed worker-stage vocabulary (caller, root, memo, counts, seconds): {row!r}"
            )
            continue
        admitted.append({**projected, "evaluations_memo_hits": row.get("evaluations_memo_hits")})
    stages = {row.get("stage") for row in rows if row.get("event") == STAGE_END}
    declared = {"completion"} | {stage for stage in stages if stage in ADMISSION_CALLERS}
    counts = Counter(row["caller"] for row in admitted)
    for caller in sorted(declared):
        if counts[caller] != 1:
            failures.append(
                f"{path}: declared admission site {caller} has {counts[caller]} admission.end "
                "rows, expected exactly one"
            )
    shared = [row for row in admitted if row["memo"] == "shared" and row["root"] == "run"]
    disk = [row for row in shared if row["evaluations"] != 0]
    for row in disk:
        failures.append(
            f"{path}: shared-memo stage {row['caller']} read {row['evaluations']} evaluations "
            "from disk (root=run; the floor is 0)"
        )
    keys = ("caller", "root", "evaluations", "seals_walked", "journals_read", "seconds")
    return {
        "fresh_walks": [{k: row[k] for k in keys} for row in admitted if row["memo"] == "fresh"],
        "path": str(path),
        "shared_disk_reads": [
            {"caller": row["caller"], "evaluations": row["evaluations"]} for row in disk
        ],
        "shared_memo_hits": {row["caller"]: row["evaluations_memo_hits"] for row in shared},
    }


def check_read_floor(worker_logs: Sequence[Path]) -> CheckResult:
    """K4 (D5): shared-memo final-leg admissions read nothing from disk."""
    failures: list[str] = []
    logs: list[dict[str, Any]] = []
    for path in worker_logs:
        try:
            logs.append(_read_floor_of(path, failures))
        except TelemetryGap as gap:
            failures.append(str(gap))
    return _result("k4", failures, {"logs": logs})


# ------------------------------------------------------------------ K5
def _boundary_failures(summary: Mapping[str, object]) -> list[str]:
    outcome = summary.get("outcome")
    sentences, observations = summary.get("sentences"), summary.get("observations")
    pending = summary.get("pending")
    if not isinstance(sentences, list) or not isinstance(observations, list):
        return ["summary carries no sentences/observations lists"]
    if any(type(value) is not int for value in observations):  # pyright: ignore[reportUnknownVariableType]
        return ["summary observations are not all integer tokens"]
    if outcome not in (*_ENDS_ON_EMISSION, *_ENDS_ON_REFUSED_BOUNDARY, *_ENDS_ON_BYTE):
        return [f"outcome {outcome!r} is not a terminal rollout outcome"]
    failures: list[str] = []
    boundaries = sum(1 for value in observations if value in (_SENTENCE_END, _DOCUMENT_END))  # pyright: ignore[reportUnknownVariableType]
    expected = len(sentences) + (1 if outcome in _ENDS_ON_REFUSED_BOUNDARY else 0)  # pyright: ignore[reportUnknownArgumentType]
    if boundaries != expected:
        failures.append(
            f"{boundaries} boundary tokens (256/257) for {len(sentences)} emitted "  # pyright: ignore[reportUnknownArgumentType]
            f"sentences under outcome {outcome!r}; expected {expected}"
        )
    last = observations[-1] if observations else None  # pyright: ignore[reportUnknownVariableType]
    on_boundary = last in (_SENTENCE_END, _DOCUMENT_END)
    if outcome in _ENDS_ON_BYTE and on_boundary:
        failures.append(f"outcome {outcome!r} ends on boundary token {last!r}")
    if outcome not in _ENDS_ON_BYTE and not on_boundary:
        failures.append(f"outcome {outcome!r} but the last observation {last!r} is no boundary")
    if outcome == "document_end" and last != _DOCUMENT_END:
        failures.append(f"outcome 'document_end' but the last observation is {last!r}")
    if outcome in _ENDS_ON_EMISSION and pending:
        failures.append(f"outcome {outcome!r} leaves pending bytes after its boundary")
    if outcome == "sentence_budget" and not sentences:
        failures.append("outcome 'sentence_budget' with no emitted sentence")
    return failures


def _continuations(receipt: Mapping[str, object]) -> tuple[int, bool]:
    """``(continuations, sampling law recorded)``; absent law means one continuation."""
    sampling = receipt.get("sampling")
    if not isinstance(sampling, Mapping):
        return 1, False
    recorded = (
        isinstance(sampling.get("sampler"), str)
        and _finite(sampling.get("temperature"))
        and any(sampling.get(key) is not None for key in _SAMPLING_SEED_FIELDS)
        and type(sampling.get("continuations")) is int
    )
    if not recorded:
        return 1, False
    return int(sampling["continuations"]), True  # pyright: ignore[reportArgumentType]


def check_generation(receipts: Sequence[Mapping[str, object]]) -> CheckResult:
    """K5 (D1): boundary-terminated outputs; stochasticity PROVEN only with N > 1."""
    failures: list[str] = []
    unproven: list[str] = []
    facts: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        where = f"receipt {index}"
        own: list[str] = []
        if receipt.get("schema") not in _GENERATION_SCHEMAS:
            own.append(f"{where}: schema {receipt.get('schema')!r} is not a generation receipt")
        if receipt.get("status") != "generation_complete":
            own.append(f"{where}: status {receipt.get('status')!r} is not generation_complete")
        summary = receipt.get("summary")
        if isinstance(summary, Mapping):
            own.extend(f"{where}: {failure}" for failure in _boundary_failures(summary))  # pyright: ignore[reportUnknownArgumentType]
        else:
            own.append(f"{where}: no summary")
        continuations, recorded = _continuations(receipt)
        if not (recorded and continuations > 1):
            unproven.append(
                f"{where}: stochastic rollout UNPROVEN: continuations={continuations}, "
                f"sampling law recorded={recorded} (fields {_SAMPLING_FIELDS} plus one of "
                f"{_SAMPLING_SEED_FIELDS}; ruling D1)"
            )
        failures.extend(own)
        facts.append(
            {
                "boundary": "FAIL" if own else "PASS",
                "continuations": continuations,
                "outcome": summary.get("outcome") if isinstance(summary, Mapping) else None,  # pyright: ignore[reportUnknownMemberType]
                "sampling_law_recorded": recorded,
            }
        )
    if not receipts:
        failures.append("no generation receipt was given")
    return _result("k5", failures, {"receipts": facts}, unproven)


# ------------------------------------------------------------------ K6
def evaluated_samples_from_manifest(manifest: object) -> tuple[int, ...]:
    """The k-set a run's evaluations must carry: its profile's curve plus diagnostic k."""
    return HProfile.from_manifest(manifest).schedule.evaluated_samples


def _example_failures(example: object, expected: set[str], where: str) -> list[str]:
    counts = example.get("by_sample_count") if isinstance(example, dict) else None  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(counts, dict):
        return [f"{where}: no by_sample_count"]
    failures: list[str] = []
    if set(counts) != expected:  # pyright: ignore[reportUnknownArgumentType]
        failures.append(
            f"{where}: k-set {sorted(counts)} differs from the profile's {sorted(expected)}"  # pyright: ignore[reportUnknownArgumentType]
        )
    for key, entries in counts.items():  # pyright: ignore[reportUnknownVariableType]
        if not str(key).isdigit() or not isinstance(entries, list):  # pyright: ignore[reportUnknownArgumentType]
            failures.append(f"{where}: malformed sample-count entry {key!r}")
            continue
        k = int(str(key))  # pyright: ignore[reportUnknownArgumentType]
        for entry in entries:  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(entry, dict):
                failures.append(f"{where}: k={k}: a non-object entry")
                continue
            if entry.get("gap") is not None:  # pyright: ignore[reportUnknownMemberType]
                continue  # a zero-importance-weight row carries no estimate
            ess = entry.get("effective_sample_size")  # pyright: ignore[reportUnknownMemberType]
            low, high = 1.0 - _ESS_TOLERANCE, k * (1.0 + _ESS_TOLERANCE)
            if not _finite(ess) or not low <= float(ess) <= high:  # pyright: ignore[reportArgumentType]
                failures.append(f"{where}: k={k}: effective_sample_size {ess!r} outside [1, {k}]")
    return failures


def check_importance_curve(
    metrics: Sequence[Path], evaluated_samples: Sequence[int]
) -> CheckResult:
    """K6 (D2): every evaluation row carries exactly the profile's k-set, ESS within k."""
    expected = {str(k) for k in evaluated_samples}
    failures: list[str] = []
    rows_seen = 0
    for path in metrics:
        try:
            rows = read_h_rows(path)
        except TelemetryGap as gap:
            failures.append(str(gap))
            continue
        evaluations = [row for row in rows if row.get("schema") == _EVALUATION_SCHEMA]
        if not evaluations:
            failures.append(f"{path}: no {_EVALUATION_SCHEMA} rows")
        for row in evaluations:
            rows_seen += 1
            examples = row.get("examples")
            if not isinstance(examples, list) or not examples:
                failures.append(f"{path}: evaluation step {row.get('step')} has no examples")
                continue
            for index, example in enumerate(examples):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                where = f"{path}: evaluation step {row.get('step')} example {index}"
                failures.extend(_example_failures(example, expected, where))
    facts = {"evaluated_samples": sorted(int(k) for k in expected), "rows": rows_seen}
    return _result("k6", failures, facts)


# ------------------------------------------------------------------ K7
def _arm_run(segments: Sequence[tuple[Path, Path]]) -> HRun:
    if len(segments) == 1:
        return load_h_run(*segments[0])
    return load_h_resumed_run(segments).run


def check_analysis_surface(arms: Mapping[str, Sequence[tuple[Path, Path]]]) -> CheckResult:
    """K7: every arm admits, reports and aggregates through the analysis surface."""
    failures: list[str] = []
    runs: dict[str, HRun] = {}
    for label, segments in arms.items():
        if not segments:
            failures.append(f"arm {label}: no captured run")
            continue
        try:
            run = _arm_run(segments)
            h_report_from_run(run)
        except TelemetryGap as gap:
            failures.append(f"arm {label}: {gap}")
            continue
        runs[label] = run
    if failures or not runs:
        return _result("k7", failures or ["no arm was given"], {"arms": sorted(arms)})
    labels = {run.profile_identity: label for label, run in runs.items()}
    try:
        document = aggregate_h_runs(list(runs.values()), labels)
    except ValueError as error:
        return _result("k7", [f"aggregate: {error}"], {"arms": sorted(arms)})
    facts = {
        "arms": sorted(arms),
        "completed": {label: document["arms"][label]["completed"] for label in sorted(runs)},
        "paired_evaluation_steps": document["evaluation"]["paired_steps"],
        "paired_training_steps": len(document["training"]["paired_steps"]),
    }
    return _result("k7", [], facts)
