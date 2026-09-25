"""Verified blind-review and filing boundaries for captured D2-D5 analyses."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from entropic_substrate.kernel.artifacts import (
    artifact_from_mapping,
    artifact_mapping,
    object_fields,
    publish_d1_atomic_bytes,
    read_bound_artifact,
    read_json_artifact,
)
from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    FiledVerdictEvidence,
    LifecycleEvent,
    LifecycleEventKind,
    LifecycleRecord,
    LifecycleState,
    SuccessorAuthorization,
    transition,
)
from entropic_substrate.ladder.analyze.ladder_analysis import (
    LadderAnalysisInputs,
    analyze_ladder_sources,
    read_ladder_run,
)
from entropic_substrate.ladder.analyze.ladder_registration import (
    paired_contrasts,
)
from entropic_substrate.ladder.analyze.seed_sizing import ContrastHypothesis
from entropic_substrate.ladder.gate.ladder_campaign import (
    _TOKEN,  # pyright: ignore[reportPrivateUsage] -- paired checked factory boundary
    FiledLadderGate,
    actor_from_mapping,
    read_campaign_design,
    validate_ladder_phase_transition,
)


def prepare_ladder_blind_packet(
    root: Path, inputs: LadderAnalysisInputs, analysis: ArtifactIdentity
) -> dict[str, object]:
    """Recompute the analysis, then remove arm/run identifiers from review data."""
    report = read_json_artifact(root, analysis)
    analyst = actor_from_mapping(report.get("analyst"))
    if report != analyze_ladder_sources(root, inputs, analyst=analyst):
        raise ValueError("analysis artifact differs from authoritative recomputation")
    _, design = read_campaign_design(root, inputs.design)
    runs = tuple(
        read_ladder_run(root, source, inputs.reader, expected_class="confirmatory")
        for source in inputs.sources
    )
    paired = paired_contrasts(runs, design)
    aliases = {
        name: f"C{index}" for index, name in enumerate(sorted(item.contrast_id for item in paired))
    }
    rules = {item.hypothesis.contrast_id: item.hypothesis for item in design.contrasts}
    reports = cast(dict[str, object], report["contrasts"])
    return {
        "schema": "d-ladder-blind-packet-v1",
        "analysis_sha256": analysis.content_hash.digest,
        "capture_digest": report["capture_digest"],
        "statistics_sha256": design.source_sha256,
        "contrasts": [
            {
                "alias": aliases[item.contrast_id],
                "units": item.units,
                "kind": rules[item.contrast_id].kind,
                "lower_bound": rules[item.contrast_id].lower_bound,
                "upper_bound": rules[item.contrast_id].upper_bound,
                "paired_values": [
                    {
                        "seed": pair.seed,
                        "effect": pair.left - pair.right,
                        "source_bytes": pair.left_exposure,
                    }
                    for pair in item.pairs
                ],
                "report": reports[item.contrast_id],
            }
            for item in paired
        ],
    }


def decide_ladder_outcome(
    rung: str,
    hypotheses: tuple[ContrastHypothesis, ...],
    intervals: Mapping[str, tuple[float, float]],
    strategies: Mapping[str, tuple[str, ...]],
) -> tuple[
    SuccessorAuthorization,
    Literal["successor", "operator_decision", "complete", "valid_negative"],
    tuple[str, ...],
]:
    """Derive strategy conjunctions from the primary intervals only."""
    rules = {item.contrast_id: item for item in hypotheses}
    if (
        rung not in {"d2", "d3", "d4", "d5"}
        or not rules
        or set(intervals) != set(rules)
        or not strategies
    ):
        raise ValueError("outcome requires a complete registered rung/contrast/strategy family")
    status: dict[str, str] = {}
    for name, (lower, upper) in intervals.items():
        rule = rules[name]
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("primary interval must be finite and ordered")
        passes = lower > rule.lower_bound and (rule.upper_bound is None or upper < rule.upper_bound)
        fails = upper < rule.lower_bound or (
            rule.upper_bound is not None and lower > rule.upper_bound
        )
        status[name] = "pass" if passes else "fail" if fails else "inconclusive"
    eligible, rejected = [], []
    for name, required in strategies.items():
        if not required or not set(required) <= rules.keys():
            raise ValueError("strategy references absent deciding contrasts")
        if all(status[key] == "pass" for key in required):
            eligible.append(name)
        rejected.append(any(status[key] == "fail" for key in required))
    if eligible:
        return (
            SuccessorAuthorization.COMPLETE if rung == "d5" else SuccessorAuthorization.AUTHORIZED,
            "complete" if rung == "d5" else "successor",
            tuple(sorted(eligible)),
        )
    if all(rejected):
        return SuccessorAuthorization.VALID_NEGATIVE, "valid_negative", ()
    return SuccessorAuthorization.INCONCLUSIVE, "operator_decision", ()


def _registered_eligibility(
    root: Path, inputs: LadderAnalysisInputs, report: dict[str, object]
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    str,
    str | None,
    SuccessorAuthorization,
    Literal["successor", "operator_decision", "complete", "valid_negative"],
]:
    row, design = read_campaign_design(root, inputs.design)
    prereg = read_json_artifact(root, artifact_from_mapping(row["preregistration"]))
    policy = object_fields(prereg, "schema successor_rung successor_phase strategies")
    if policy["schema"] != "d-ladder-successor-policy-v1":
        raise ValueError("successor requires explicit preregistered policy, not an outcome summary")
    rules = {item.hypothesis.contrast_id: item.hypothesis for item in design.contrasts}
    reports = cast(dict[str, dict[str, object]], report["contrasts"])
    strategies = policy["strategies"]
    if type(strategies) is not dict or not strategies:
        raise ValueError("successor policy requires explicit nonempty strategies")
    requirements: dict[str, tuple[str, ...]] = {}
    arms_by_strategy: dict[str, set[str]] = {}
    for name, raw in strategies.items():
        clause = object_fields(raw, "arms requires")
        if (
            type(name) is not str
            or not name
            or type(clause["arms"]) is not list
            or type(clause["requires"]) is not list
            or not clause["arms"]
            or not clause["requires"]
        ):
            raise ValueError("successor strategy requires explicit arms and deciding contrasts")
        required = set(clause["requires"])
        candidate_arms = set(clause["arms"])
        if (
            not required <= rules.keys()
            or not candidate_arms <= dict(design.construction_digests).keys()
        ):
            raise ValueError("successor policy references unknown arms or contrasts")
        requirements[name] = tuple(sorted(required))
        arms_by_strategy[name] = candidate_arms
    intervals = {
        name: (
            cast(dict[str, float], value["primary"])["lower"],
            cast(dict[str, float], value["primary"])["upper"],
        )
        for name, value in reports.items()
    }
    posture, next_action, eligible = decide_ladder_outcome(
        cast(str, row["rung"]), tuple(rules.values()), intervals, requirements
    )
    arms = tuple(sorted({arm for strategy in eligible for arm in arms_by_strategy[strategy]}))
    return (
        eligible,
        arms,
        cast(str, policy["successor_rung"]),
        cast(str | None, policy["successor_phase"]),
        posture,
        next_action,
    )


@dataclass(frozen=True, slots=True)
class FiledLadderResult:
    record: LifecycleRecord
    machine: ArtifactIdentity
    gate: FiledLadderGate | None
    next_action: Literal["successor", "operator_decision", "complete", "valid_negative"]


def file_ladder_verdict(
    root: Path,
    inputs: LadderAnalysisInputs,
    *,
    analysis: ArtifactIdentity,
    packet: ArtifactIdentity,
    review: ArtifactIdentity,
    narrative: ArtifactIdentity,
    record: LifecycleRecord,
    operator: ActorIdentity,
    machine_path: str,
    existing_machine: ArtifactIdentity | None = None,
) -> FiledLadderResult:
    """File recomputed results after independently recorded agreement; derive gate.

    Review execution and DVC push are external adapters. Passing a hand-written
    outcome string cannot substitute for their linked artifacts or the data.
    """
    campaign, _ = read_campaign_design(root, inputs.design)
    if campaign["rung"] == "d1":
        raise ValueError("D1 filing must use its existing full semantic graph adapter")
    if operator != actor_from_mapping(campaign["operator"]):
        raise ValueError("filing operator differs from prepared operator")
    if record.rung.lower() != campaign["rung"] or record.state is not LifecycleState.BLIND_REVIEWED:
        raise ValueError("filing requires the same rung's blind-reviewed lifecycle")
    if not any(
        event.kind is LifecycleEventKind.CAPTURE_DVC and event.artifacts == inputs.sources
        for event in record.events
    ):
        raise ValueError("captured source roster does not belong to this lifecycle")
    for kind in (LifecycleEventKind.PREREGISTER, LifecycleEventKind.START_CONFIRMATORY):
        if not any(
            event.kind is kind and event.actor == operator and inputs.design in event.artifacts
            for event in record.events
        ):
            raise ValueError("prepared design/operator does not belong to this lifecycle")
    expected_packet = prepare_ladder_blind_packet(root, inputs, analysis)
    if read_json_artifact(root, packet) != expected_packet:
        raise ValueError("blind packet differs from recomputed capture analysis")
    report = read_json_artifact(root, analysis)
    reviewer = object_fields(
        read_json_artifact(root, review),
        "schema actor packet analysis_sha256 agrees open_major_findings",
    )
    actor = actor_from_mapping(reviewer["actor"])
    analyst = actor_from_mapping(report["analyst"])
    builder = actor_from_mapping(campaign["implementation"])
    if (
        reviewer["schema"] != "d-ladder-review-v1"
        or reviewer["agrees"] is not True
        or reviewer["open_major_findings"] != []
    ):
        raise ValueError("review has disagreement, open findings or unknown schema")
    if (
        artifact_from_mapping(reviewer["packet"]) != packet
        or reviewer["analysis_sha256"] != analysis.content_hash.digest
    ):
        raise ValueError("review does not bind this packet and analysis")
    if actor.role is not ActorRole.REVIEW or any(
        actor.actor_id == prior.actor_id or actor.session_id == prior.session_id
        for prior in (analyst, builder)
    ):
        raise ValueError("review requires an independent actor and session")
    if not any(
        event.kind is LifecycleEventKind.ANALYZE
        and event.actor == analyst
        and analysis in event.artifacts
        for event in record.events
    ):
        raise ValueError("analysis receipt does not belong to the lifecycle")
    latest = record.events[-1]
    if (
        latest.kind is not LifecycleEventKind.BLIND_REVIEW
        or latest.actor != actor
        or latest.sealed_packet != packet
        or review not in latest.artifacts
    ):
        raise ValueError("review receipt does not belong to the lifecycle")
    narrative_bytes = read_bound_artifact(root, narrative)
    if f"Analysis-SHA256: {analysis.content_hash.digest}".encode() not in narrative_bytes:
        raise ValueError("narrative must identify its analysis artifact")
    strategies, arms, successor, successor_phase, posture, next_action = _registered_eligibility(
        root, inputs, report
    )
    if posture is not SuccessorAuthorization.AUTHORIZED:
        strategies, arms = (), ()
    machine_data = {
        "schema": "d-ladder-filed-verdict-v1",
        "rung": campaign["rung"],
        "d4_phase": campaign["d4_phase"],
        "design": artifact_mapping(inputs.design),
        "capture_digest": report["capture_digest"],
        "analysis": artifact_mapping(analysis),
        "packet": artifact_mapping(packet),
        "review": artifact_mapping(review),
        "narrative": artifact_mapping(narrative),
        "authorization": posture.value,
        "eligible_strategies": strategies,
        "eligible_arms": arms,
        "successor_rung": successor if posture is SuccessorAuthorization.AUTHORIZED else None,
        "successor_phase": successor_phase
        if posture is SuccessorAuthorization.AUTHORIZED
        else None,
        "next_action": next_action,
    }
    # Validate successor shape before publishing any immutable verdict bytes.
    if posture is SuccessorAuthorization.AUTHORIZED:
        validate_ladder_phase_transition(
            cast(str, campaign["rung"]),
            cast(str | None, campaign["d4_phase"]),
            successor,
            successor_phase,
        )
    payload = json.dumps(machine_data, sort_keys=True, allow_nan=False).encode()
    if existing_machine is None:
        machine = publish_d1_atomic_bytes(root, machine_path, payload, label="ladder verdict")
    else:
        if read_json_artifact(root, existing_machine) != json.loads(payload):
            raise ValueError("filed machine verdict differs from recomputed scientific outcome")
        machine = existing_machine
    evidence = FiledVerdictEvidence(narrative, machine, packet, (), posture, arms, strategies)
    filed = transition(
        record,
        LifecycleEvent(
            f"file-{machine.content_hash.digest}",
            LifecycleEventKind.FILE,
            operator,
            (narrative, machine, packet, review, analysis),
            filed_verdict=evidence,
        ),
    )
    gate = None
    if posture is SuccessorAuthorization.AUTHORIZED:
        filed = transition(
            filed,
            LifecycleEvent(
                f"successor-{machine.content_hash.digest}",
                LifecycleEventKind.AUTHORIZE_SUCCESSOR,
                operator,
                (machine,),
            ),
        )
        gate = FiledLadderGate(
            cast(str, campaign["rung"]),
            filed,
            evidence,
            successor_rung=successor,
            d4_phase=cast(str | None, campaign["d4_phase"]),
            successor_phase=successor_phase,
            _token=_TOKEN,
        )
    return FiledLadderResult(
        filed,
        machine,
        gate,
        next_action,
    )
