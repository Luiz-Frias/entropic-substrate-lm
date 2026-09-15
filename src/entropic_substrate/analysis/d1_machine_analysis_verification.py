"""Final D1 machine-output binding to authoritative capture and diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_capture import (
    D1AppliedExclusion,
    derive_d1_analysis_exclusions,
)
from entropic_substrate.analysis.d1_diagnostics import (
    D1DiagnosticPolicy,
    D1TargetOperator,
    parse_d1_diagnostic_policy,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1ExternalAnalysisInputControl,
)
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.analysis.d1_machine_analysis_binding import (
    D1ContractBoundMachineAnalysis,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

__all__ = [
    "D1MachineAnalysisVerificationError",
    "D1VerifiedMachineAnalysis",
    "verify_d1_machine_analysis",
]

_TOKEN = object()


class D1MachineAnalysisVerificationError(ValueError):
    """Contract-bound output differs from authoritative capture or diagnostics."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1MachineAnalysisVerificationError(message)
    raise D1MachineAnalysisVerificationError(message) from error


def _mapping(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"machine analysis {label} must be an object")
    return cast(dict[str, object], value)


def _rows(value: object, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"machine analysis {label} must be an array")
    return cast(list[object], value)


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _exclusion_rows(exclusions: tuple[D1AppliedExclusion, ...]) -> list[dict[str, str]]:
    return [
        {
            "evidence_digest": exclusion.evidence_digest,
            "exclusion_alias": f"X{index:02d}",
            "reason_id": exclusion.reason_id,
        }
        for index, exclusion in enumerate(exclusions, start=1)
    ]


def _survivors(
    binding: D1ContractBoundMachineAnalysis,
    aggregate: D1HeterogeneousCaptureAggregate,
    exclusions: tuple[D1AppliedExclusion, ...],
) -> tuple[int, ...]:
    if aggregate.manifest_digest != binding.expected.aggregate.manifest_digest:
        _fail("authoritative aggregate differs from bound aggregate declaration")
    excluded_pairs = {item.pairing_id for item in exclusions}
    excluded_seeds = {
        item.seed
        for item in aggregate.roster.expected_confirmatory
        if item.pairing_id in excluded_pairs
    }
    registered = binding.expected.design.registered_seeds
    included = tuple(seed for seed in registered if seed not in excluded_seeds)
    excluded = tuple(seed for seed in registered if seed in excluded_seeds)
    root = binding.analysis.manifest
    if root["included_seeds"] != list(included) or root["excluded_seeds"] != list(excluded):
        _fail("machine analysis seed partition differs from authoritative exclusions")
    if _canonical(root["exclusions"]) != _canonical(_exclusion_rows(exclusions)):
        _fail("machine analysis exclusion rows differ from authoritative defect evidence")
    return included


def _crosses(policy: D1DiagnosticPolicy, value: float) -> bool:
    if policy.target_operator is D1TargetOperator.AT_MOST:
        return value <= policy.target_threshold
    return value >= policy.target_threshold


def _target(
    diagnostic: dict[str, object],
    policy: D1DiagnosticPolicy,
    trajectory: list[object],
) -> None:
    target = _mapping(diagnostic["quality_target"], "quality target")
    points = tuple(_mapping(value, "trajectory point") for value in trajectory)
    declared = (target["metric"], target["operator"], target["threshold"])
    registered = (policy.target_metric, policy.target_operator.value, policy.target_threshold)
    if _canonical(declared) != _canonical(registered):
        _fail("quality target policy differs from registered diagnostic policy")
    if target["reached"] is True:
        metric_value = float(cast(int | float, target["metric_value"]))
        if not _crosses(policy, metric_value):
            _fail("quality target reached value contradicts registered operator and threshold")
        cursor = (target["eval_index"], target["step"])
        matches = [point for point in points if (point["eval_index"], point["step"]) == cursor]
        if len(matches) != 1:
            _fail("quality target cursor is absent from the reported trajectory")
        if policy.target_metric == policy.aulc_metric and _canonical(
            matches[0]["value"]
        ) != _canonical(target["metric_value"]):
            _fail("quality target metric differs from the same-cursor trajectory value")
        if float(cast(int | float, target["trainer_wall_seconds"])) < 0.0:
            _fail("quality target trainer wall time must be nonnegative")
    elif policy.target_metric == policy.aulc_metric and any(
        _crosses(policy, float(cast(int | float, point["value"]))) for point in points
    ):
        _fail("quality target unreached state contradicts the reported trajectory")


def _diagnostics(
    binding: D1ContractBoundMachineAnalysis,
    included: tuple[int, ...],
    policy: D1DiagnosticPolicy,
) -> None:
    rows = _rows(binding.analysis.manifest["diagnostics"], "diagnostics")
    aliases = binding.expected.alias_application.aliases
    expected_grid = [(aliases.for_arm(arm), seed) for seed in included for arm in D1Arm]
    observed_grid: list[tuple[object, object]] = []
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"diagnostic[{index}]")
        observed_grid.append((row["arm_alias"], row["seed"]))
        if (
            row["aulc_metric"] != policy.aulc_metric
            or row["aulc_normalization"] != policy.aulc_normalization
            or row["aulc_deciding"] is not False
        ):
            _fail("machine analysis diagnostic differs from registered AULC policy")
        trajectory = _rows(row["trajectory"], f"diagnostic[{index}].trajectory")
        _target(row, policy, trajectory)
    if observed_grid != expected_grid:
        _fail("machine analysis diagnostic grid differs from authoritative survivors")


def _verify(
    binding: D1ContractBoundMachineAnalysis,
    aggregate: D1HeterogeneousCaptureAggregate,
    policy: D1DiagnosticPolicy,
) -> None:
    exclusions = derive_d1_analysis_exclusions(
        aggregate=aggregate,
        decision_contract=binding.expected.decision_contract,
    )
    included = _survivors(binding, aggregate, exclusions)
    _diagnostics(binding, included, policy)


@dataclass(frozen=True, slots=True)
class D1VerifiedMachineAnalysis:
    """Final output typestate bound to registered controls and capture authority."""

    binding: D1ContractBoundMachineAnalysis
    aggregate: D1HeterogeneousCaptureAggregate = field(repr=False)
    canonical_bytes: bytes = field(repr=False)
    analysis: D1AliasedMachineAnalysis
    expected: D1ExternalAnalysisInputControl = field(repr=False)
    input_manifest_sha256: str
    analysis_digest: str
    diagnostic_policy: D1DiagnosticPolicy
    _factory_token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _TOKEN:
            raise TypeError(
                "D1VerifiedMachineAnalysis is factory-only; use verify_d1_machine_analysis"
            )
        if type(self.binding) is not D1ContractBoundMachineAnalysis:
            raise TypeError("verified machine analysis requires contract-bound analysis")
        if type(self.aggregate) is not D1HeterogeneousCaptureAggregate:
            raise TypeError("verified machine analysis requires authoritative aggregate")
        if type(self.expected) is not D1ExternalAnalysisInputControl:
            raise TypeError("verified machine analysis input must remain factory-verified")
        if self.analysis is not self.binding.analysis or self.expected is not self.binding.expected:
            raise ValueError("verified declaration or input differs from contract binding")
        if self.canonical_bytes != self.binding.canonical_bytes:
            raise ValueError("verified canonical bytes differ from contract binding")
        if self.input_manifest_sha256 != self.expected.input_manifest_sha256:
            raise ValueError("verified machine analysis differs from bound input digest")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.analysis_digest:
            raise ValueError("verified machine analysis digest differs from canonical bytes")
        registered = parse_d1_diagnostic_policy(self.expected.decision_contract)
        if self.diagnostic_policy != registered:
            raise ValueError("verified diagnostic policy differs from bound input")
        _verify(self.binding, self.aggregate, self.diagnostic_policy)

    @property
    def manifest(self) -> dict[str, object]:
        return self.analysis.manifest


def verify_d1_machine_analysis(
    *,
    binding: D1ContractBoundMachineAnalysis,
    aggregate: D1HeterogeneousCaptureAggregate,
) -> D1VerifiedMachineAnalysis:
    """Complete verification without consulting live services or recomputing measurements."""
    if type(binding) is not D1ContractBoundMachineAnalysis:
        raise TypeError("machine analysis verification requires contract-bound analysis")
    if type(aggregate) is not D1HeterogeneousCaptureAggregate:
        raise TypeError("machine analysis verification requires authoritative aggregate")
    try:
        policy = parse_d1_diagnostic_policy(binding.expected.decision_contract)
    except (TypeError, ValueError) as error:
        _fail("bound input lacks a valid registered diagnostic policy", error)
    _verify(binding, aggregate, policy)
    return D1VerifiedMachineAnalysis(
        binding=binding,
        aggregate=aggregate,
        canonical_bytes=binding.canonical_bytes,
        analysis=binding.analysis,
        expected=binding.expected,
        input_manifest_sha256=binding.input_manifest_sha256,
        analysis_digest=binding.analysis_digest,
        diagnostic_policy=policy,
        _factory_token=_TOKEN,
    )
