"""Namespace-bound D/H statistical declarations and captured D1 pilot preparation.

This module produces proposals, not experiment authorization. A D2-D5 capture
adapter must exist before those rungs can supply pilot variance; local smoke
captures are never scientific inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from entropic_substrate.core.corpus.batching import DATA_ORDER_DIGEST_BATCHES
from entropic_substrate.kernel.identity.d1_primitives import require_sha256
from entropic_substrate.ladder.analyze.seed_sizing import (
    ContrastHypothesis,
    DualConfidenceReport,
    PairedSeed,
    PilotContrast,
    SeedPlan,
    dual_confidence_intervals,
    plan_seed_count,
    validate_hypothesis,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1RunEvidence

if TYPE_CHECKING:
    from entropic_substrate.__legacy.d1_excluded_capture import D1ExcludedCaptureReceipt
    from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture


@dataclass(frozen=True, slots=True)
class StatisticalContrast:
    hypothesis: ContrastHypothesis
    terms: tuple[tuple[str, float], ...]
    metric: str
    eval_steps: tuple[int, ...]
    source_bytes: int


@dataclass(frozen=True, slots=True)
class StatisticalDesign:
    source_sha256: str
    run_class: str
    family_digest: str
    construction_digests: tuple[tuple[str, str], ...]
    seeds: tuple[int, ...]
    registered_design_digest: str | None
    contrasts: tuple[StatisticalContrast, ...]
    namespace: Literal["D", "H"] = "D"


def _object(value: object, fields: str | None = None) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError("statistical specification requires a JSON object")
    result = cast(dict[str, object], value)
    if fields is not None and set(result) != set(fields.split()):
        raise ValueError("statistical specification keys must be exact; no defaults")
    return result


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate statistical specification key: {key}")
        result[key] = value
    return result


def _text(value: object) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError("expected nonempty trimmed text")
    return value


def _number(value: object) -> float:
    if type(value) not in (int, float):
        raise ValueError("expected finite number, not boolean")
    result = float(cast(float, value))
    if not math.isfinite(result):
        raise ValueError("expected finite number")
    return result


def _integer(value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"expected integer >= {minimum}")
    return value


def _sequence(value: object) -> list[object]:
    if type(value) is not list or not value:
        raise ValueError("expected nonempty JSON array")
    return cast(list[object], value)


def _digest(value: object) -> str:
    result = _text(value)
    require_sha256(result, label="statistical identity")
    return result


def _contrast(value: object, arms: set[str]) -> StatisticalContrast:
    row = _object(
        value,
        "contrast_id terms metric units eval_steps source_bytes kind expected_effect lower_bound upper_bound",
    )
    terms = tuple((arm, _number(weight)) for arm, weight in _object(row["terms"]).items())
    if (
        len(terms) < 2
        or any(arm not in arms or weight == 0 for arm, weight in terms)
        or not math.isclose(math.fsum(weight for _, weight in terms), 0, abs_tol=1e-12)
    ):
        raise ValueError("contrast requires nonzero, zero-sum weights over declared arms")
    steps = tuple(_integer(step) for step in _sequence(row["eval_steps"]))
    if tuple(sorted(set(steps))) != steps:
        raise ValueError("eval steps must be unique and increasing")
    kind = _text(row["kind"])
    if kind not in {"superiority", "noninferiority", "equivalence"}:
        raise ValueError("unknown hypothesis kind")
    hypothesis = ContrastHypothesis(
        _text(row["contrast_id"]),
        _text(row["units"]),
        cast(Literal["superiority", "noninferiority", "equivalence"], kind),
        _number(row["expected_effect"]),
        _number(row["lower_bound"]),
        None if row["upper_bound"] is None else _number(row["upper_bound"]),
    )
    validate_hypothesis(hypothesis)
    return StatisticalContrast(
        hypothesis, terms, _text(row["metric"]), steps, _integer(row["source_bytes"], 1)
    )


def read_statistical_design(payload: bytes, *, expected_sha256: str) -> StatisticalDesign:
    """Read D declarations only; preserve the historical namespace contract."""
    return _read_statistical_design(
        payload, expected_sha256=expected_sha256, schema="d-ladder-statistical-design-v1"
    )


def read_h_statistical_design(payload: bytes, *, expected_sha256: str) -> StatisticalDesign:
    """Read H declarations without defaults, evidence admission or launch authority.

    Rung adapters still bind constructions, admissible metrics, captured outcomes
    and source exposure. This reader neither rewrites D bytes nor sizes from smoke.
    """
    return _read_statistical_design(
        payload, expected_sha256=expected_sha256, schema="h-ladder-statistical-design-v1"
    )


def _read_statistical_design(
    payload: bytes,
    *,
    expected_sha256: str,
    schema: Literal["d-ladder-statistical-design-v1", "h-ladder-statistical-design-v1"],
) -> StatisticalDesign:
    """Shared arithmetic contract; public readers choose one exact namespace."""
    if type(payload) is not bytes:
        raise TypeError("statistical specification must be bytes")
    _digest(expected_sha256)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("statistical specification digest mismatch")
    value = _object(
        json.loads(payload, object_pairs_hook=_unique_object),
        "schema run_class family_digest construction_digests seeds contrasts registered_design_digest",
    )
    if value["schema"] != schema:
        raise ValueError("unsupported statistical specification schema")
    if value["run_class"] not in ("pilot", "confirmatory"):
        raise ValueError("statistical inputs must be pilot or confirmatory, never smoke")
    registered = value["registered_design_digest"]
    if value["run_class"] == "confirmatory":
        registered = _digest(registered)
    elif registered is not None:
        raise ValueError("pilot specification must not claim confirmatory registration")
    seeds = tuple(_integer(seed) for seed in _sequence(value["seeds"]))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique paired seeds required")
    constructions = tuple(
        (_text(arm), _digest(digest))
        for arm, digest in _object(value["construction_digests"]).items()
    )
    contrasts = tuple(
        _contrast(item, {arm for arm, _ in constructions}) for item in _sequence(value["contrasts"])
    )
    if len({item.hypothesis.contrast_id for item in contrasts}) != len(contrasts):
        raise ValueError("duplicate deciding contrast")
    if {arm for item in contrasts for arm, _ in item.terms} != {arm for arm, _ in constructions}:
        raise ValueError("declared arms must equal the complete contrast family")
    return StatisticalDesign(
        expected_sha256,
        _text(value["run_class"]),
        _digest(value["family_digest"]),
        constructions,
        seeds,
        registered,
        contrasts,
        namespace="D" if schema == "d-ladder-statistical-design-v1" else "H",
    )


@dataclass(frozen=True, slots=True)
class RegistrationProposal:
    """A derived proposal never grants launch, successor or publishing authority."""

    design_sha256: str
    capture_digests: tuple[str, ...]
    sizing: SeedPlan
    next_step: Literal["preregistration_preparation", "operator_cap_review_required"]
    schema: str = "d-ladder-seed-proposal-v1"


@dataclass(frozen=True, slots=True)
class StatisticalEvaluation:
    step: int
    source_bytes: int
    metrics: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class StatisticalRun:
    """Normalized admitted observations; this value itself grants no authority."""

    arm: str
    seed: int
    run_class: str
    family_digest: str
    construction_digest: str
    evals: tuple[StatisticalEvaluation, ...]
    data_order: tuple[tuple[int, str, int], ...] | None = None


def _validate_paired_order(runs: tuple[StatisticalRun, ...]) -> None:
    if all(run.data_order is None for run in runs):
        return  # Historical D1 readers do not claim the new attestation format.
    by_seed: dict[int, tuple[tuple[int, str, int], ...]] = {}
    for run in runs:
        series = run.data_order
        if type(series) is not tuple:
            raise ValueError("data-order evidence must cover the entire cohort")
        for row in series:
            if type(row) is not tuple or len(row) != 3:
                raise ValueError("data-order row requires step, SHA-256, batch count")
            step, digest, batches = row
            _integer(step, 1)
            _digest(digest)
            _integer(batches, 1)
            if batches != min(step, DATA_ORDER_DIGEST_BATCHES):
                raise ValueError("data-order batch count differs from the measured audit prefix")
        expected = tuple(item.step for item in run.evals if item.step > 0)
        if tuple(row[0] for row in series) != expected or expected != tuple(sorted(set(expected))):
            raise ValueError("data-order attestations must match every positive eval step")
        frozen = {digest for _, digest, count in series if count == DATA_ORDER_DIGEST_BATCHES}
        if len(frozen) > 1:
            raise ValueError("data-order digest changed after the audit prefix was complete")
        if run.seed in by_seed and by_seed[run.seed] != series:
            raise ValueError("data-order differs across arms of the same paired seed")
        by_seed[run.seed] = series


def paired_contrasts(
    runs: tuple[StatisticalRun, ...], design: StatisticalDesign
) -> tuple[PilotContrast, ...]:
    """Apply D source-order and endpoint rules after its adapter admits bytes."""
    if design.namespace != "D":
        raise ValueError("H designs require an explicit H observation/source-order adapter")
    constructions = dict(design.construction_digests)
    cells: dict[tuple[str, int], StatisticalRun] = {}
    for run in runs:
        arm, seed = _text(run.arm), _integer(run.seed)
        for item in run.evals:
            _integer(item.step)
            _integer(item.source_bytes)
        if (
            run.run_class != design.run_class
            or run.family_digest != design.family_digest
            or constructions.get(arm) != run.construction_digest
        ):
            raise ValueError("captured run differs from prespecified class/family/construction")
        if (arm, seed) in cells:
            raise ValueError("duplicate captured arm/seed")
        cells[arm, seed] = run
    expected = {(arm, seed) for arm in constructions for seed in design.seeds}
    if set(cells) != expected:
        raise ValueError("capture must equal complete prespecified arm/seed roster")
    _validate_paired_order(runs)
    result: list[PilotContrast] = []
    for contrast in design.contrasts:
        pairs: list[PairedSeed] = []
        for seed in design.seeds:
            terms: list[float] = []
            for arm, coefficient in contrast.terms:
                evals = {item.step: item for item in cells[arm, seed].evals}
                if len(evals) != len(cells[arm, seed].evals):
                    raise ValueError("duplicate captured eval step")
                if not set(contrast.eval_steps) <= evals.keys():
                    raise ValueError("captured endpoint lacks prespecified eval steps")
                final = evals[contrast.eval_steps[-1]]
                if final.source_bytes != contrast.source_bytes:
                    raise ValueError("captured source exposure differs from prespecified endpoint")
                values: list[float] = []
                for step in contrast.eval_steps:
                    metrics = dict(evals[step].metrics)
                    if len(metrics) != len(evals[step].metrics) or contrast.metric not in metrics:
                        raise ValueError("duplicate or absent captured metric")
                    values.append(_number(metrics[contrast.metric]))
                endpoint = math.fsum(values) / len(values)
                terms.append(coefficient * endpoint)
            # Weighted factorial contrasts remain one observation per seed.
            pairs.append(
                PairedSeed(
                    seed, math.fsum(terms), 0.0, contrast.source_bytes, contrast.source_bytes
                )
            )
        result.append(
            PilotContrast(
                contrast.hypothesis.contrast_id,
                contrast.hypothesis.units,
                design.run_class,
                design.family_digest,
                contrast.source_bytes,
                tuple(pairs),
            )
        )
    return tuple(result)


def _d1_runs(
    runs: tuple[D1RunEvidence, ...], design: StatisticalDesign
) -> tuple[StatisticalRun, ...]:
    if design.namespace != "D":
        raise ValueError("H designs require an H source adapter, not D1 telemetry")
    metrics = tuple(sorted({item.metric for item in design.contrasts}))
    return tuple(
        StatisticalRun(
            _text(run.plan_manifest["arm"]),
            _integer(run.plan_manifest["seed"]),
            run.identity.run_class.value,
            run.identity.family_digest,
            run.identity.construction_digest,
            tuple(
                StatisticalEvaluation(
                    item.step,
                    item.resource_work.committed_quality_source_bytes,
                    tuple(
                        (metric, item.quality.scalar_metric(metric))
                        for metric in metrics
                        if any(
                            rule.metric == metric and item.step in rule.eval_steps
                            for rule in design.contrasts
                        )
                    ),
                )
                for item in run.evals
            ),
        )
        for run in runs
    )


def prepare_registration(
    receipts: tuple[D1ExcludedCaptureReceipt, ...],
    rule_bytes: bytes,
    expected_rule_sha256: str,
    *,
    max_seeds: int,
) -> RegistrationProposal:
    """Size from finalized excluded D1 captures and exact predeclared rule bytes.

    This first adapter admits FULL D1 pilots only. D2-D5 smoke construction
    coverage is not a source of pilot variance and receives no fallback path.
    """
    from entropic_substrate.__legacy.d1_excluded_capture import D1ExcludedCaptureReceipt
    from entropic_substrate.kernel.identity.d1_primitives import RunClass
    from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence_bytes
    from entropic_substrate.ladder.register.d1_config import D1LaunchScale

    design = read_statistical_design(rule_bytes, expected_sha256=expected_rule_sha256)
    if design.run_class != "pilot" or type(receipts) is not tuple or not receipts:
        raise ValueError("seed preparation requires finalized pilot captures")
    runs: list[D1RunEvidence] = []
    digests: list[str] = []
    for receipt in receipts:
        if type(receipt) is not D1ExcludedCaptureReceipt:
            raise TypeError("pilot input must be a finalized D1ExcludedCaptureReceipt")
        if receipt.plan.profile.scale is not D1LaunchScale.FULL:
            raise ValueError("pilot variance requires FULL scale, not tiny or smoke")
        payload = receipt.read_journal_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt.journal_sha256:
            raise ValueError("captured journal digest mismatch")
        runs.append(
            read_d1_run_evidence_bytes(
                payload,
                journal_path=receipt.journal.path,
                expected_run_class=RunClass.PILOT,
                expected_family_digest=design.family_digest,
                expected_plan_digest=receipt.plan.plan_digest,
                expected_logical_run_id=receipt.plan.logical_run_id,
            )
        )
        digests.append(_digest(receipt.capture_digest))
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate pilot capture")
    inputs = paired_contrasts(_d1_runs(tuple(runs), design), design)
    sizing = plan_seed_count(
        inputs,
        tuple(item.hypothesis for item in design.contrasts),
        max_seeds=max_seeds,
        expected_pilot_blocks=len(design.seeds),
    )
    return RegistrationProposal(
        design.source_sha256,
        tuple(sorted(digests)),
        sizing,
        "preregistration_preparation"
        if sizing.required_seeds is not None
        else "operator_cap_review_required",
    )


def report_registered_contrasts(
    capture: D1AdmittedAnalysisCapture, root: Path, rule_bytes: bytes, expected_rule_sha256: str
) -> tuple[tuple[str, DualConfidenceReport], ...]:
    """Reverify captured confirmation bytes and report one prespecified population.

    This initial adapter requires complete seeds; it refuses an exclusion case
    rather than quietly changing the declared population or multiplicity.
    """
    from entropic_substrate.kernel.identity.d1_primitives import RunClass, canonical_digest
    from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture
    from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
        D1ArtifactRole,
        verify_d1_artifact,
    )
    from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence_bytes

    design = read_statistical_design(rule_bytes, expected_sha256=expected_rule_sha256)
    if type(capture) is not D1AdmittedAnalysisCapture or design.run_class != "confirmatory":
        raise TypeError("report requires admitted confirmation capture and confirmation rules")
    if capture.design_digest != design.registered_design_digest or capture.exclusions:
        raise ValueError("registration mismatch or unsupported excluded population")
    runs: list[D1RunEvidence] = []
    for artifact in capture.input_artifacts:
        verify_d1_artifact(root, artifact)
        if artifact.role is D1ArtifactRole.JOURNAL:
            path = root / artifact.normalized_path
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != artifact.sha256:
                raise ValueError("confirmation journal changed during verification")
            runs.append(
                read_d1_run_evidence_bytes(
                    payload,
                    journal_path=path,
                    expected_run_class=RunClass.CONFIRMATORY,
                    expected_family_digest=design.family_digest,
                )
            )
    seal = canonical_digest(
        {
            "capture": capture.capture_digest,
            "rules": design.source_sha256,
            "sources": [item.manifest for item in capture.input_artifacts],
        }
    )
    return tuple(
        (
            contrast.contrast_id,
            dual_confidence_intervals(
                contrast.pairs, family_size=len(design.contrasts), sealed_data_id=seal
            ),
        )
        for contrast in paired_contrasts(_d1_runs(tuple(runs), design), design)
    )
