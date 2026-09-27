"""Typed collapse evidence traverses existing external-analysis and filing ports.

The external worker/reader/reviewer/Git transports are the existing explicit
test doubles. No statistic is replaced after analysis or after review.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.kernel.identity.d1_primitives import thaw_for_digest
from entropic_substrate.kernel.verdict.d1_evaluation_values import D1ChannelMoments, D1CosineGap
from entropic_substrate.ladder.file.d1_filing import validate_d1_filing_manifest
from entropic_substrate.ladder.file.d1_filing_application import (
    D1LocalFilingArtifactStore,
    prepare_d1_filing_bundle,
)
from entropic_substrate.ladder.file.d1_filing_graph import verify_d1_filing
from entropic_substrate.ladder.file.d1_filing_receipts import D1FilingRevisions

from . import test_d1_external_worker as worker_fixture
from .test_d1_filing_application import _filing_git, _paths
from .test_d1_real_arm_verdict_assembly import _lineage
from .test_d1_registered_design_hydration import _RAW_PREREGISTRATION

pytestmark = pytest.mark.unit


def test_collapsed_endpoint_survives_fresh_analysis_review_and_verified_filing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_case, original_decision = worker_fixture._analysis_case, worker_fixture._decision

    def collapsed_case(root):
        capture_root, design, base, aggregate, runs = original_case(root)
        for name, run in runs.items():
            evals = []
            for point in run.evals:
                width = point.quality.state_moments.width
                zeros = (0.0,) * width
                quality = replace(
                    point.quality,
                    state_moments=D1ChannelMoments(zeros, zeros, zeros, 0.0, 0, 0.0),
                    mean_pairwise_cosine_distance=D1CosineGap(
                        point.quality.online_positions, point.quality.online_positions
                    ),
                )
                evals.append(replace(point, quality=quality))
            runs[name] = replace(run, evals=tuple(evals))
        return capture_root, design, base, aggregate, runs

    def cosine_decision(base):
        decision = original_decision(base)
        constants = thaw_for_digest(decision.registered_constants)
        by_arm = constants["analysis_policy"]["clauses"]["future_grounding"]["conditions_by_arm"]
        for arm in by_arm:
            by_arm[arm] = [
                {
                    "source": "REGISTERED_CONTRAST_INTERVAL",
                    "operator": "AT_LEAST",
                    "threshold": 0.0,
                    "contrast_id": decision.contrast_ids[0],
                    "metric": None,
                    "estimator": None,
                    "smoothing_window": None,
                }
            ]
        return replace(
            decision,
            registered_constants=constants,
            contrast_rules={
                name: replace(
                    rule, endpoint=replace(rule.endpoint, metric="mean_pairwise_cosine_distance")
                )
                for name, rule in decision.contrast_rules.items()
            },
        )

    monkeypatch.setattr(worker_fixture, "_analysis_case", collapsed_case)
    monkeypatch.setattr(worker_fixture, "_decision", cosine_decision)
    lineage = _lineage(tmp_path / "analysis", monkeypatch)
    manifest = lineage.verified.manifest
    assert manifest["schema_version"] == "d1-aliased-machine-analysis-v3"
    assert manifest["included_seeds"] == [7] and manifest["excluded_seeds"] == []
    assert all(
        row["estimate"] is None and row["paired_estimates"] == [None]
        for row in manifest["contrasts"]
    )
    assert all(
        arm["noncollapse"]["status"] == "FAIL"
        and arm["future_grounding"]["status"] == "INCONCLUSIVE"
        for arm in manifest["clauses_by_alias"].values()
    )
    prepared = prepare_d1_filing_bundle(
        verified=lineage.verified,
        analysis_receipt=lineage.analysis_receipt,
        execution=lineage.execution,
        reveal=lineage.reveal,
        adjudication=lineage.adjudication,
        verdict="VALID_NEGATIVE",
        narrative_body=b"# D1 collapse fixture\n",
        preregistration_bytes=_RAW_PREREGISTRATION,
        paths=_paths(),
    )
    root = tmp_path / "filing-root"
    (root / "filing").mkdir(parents=True)
    store = D1LocalFilingArtifactStore(root)
    store.publish(prepared, manifest_path="filing/manifest.json")
    revisions = D1FilingRevisions("b" * 40, "c" * 40, "d" * 40, "e" * 40, "1" * 40, "2" * 40)
    detached = validate_d1_filing_manifest(
        prepared.verdict.manifest,
        json.loads(prepared.manifest_bytes),
        decision_contract=prepared.verdict.verified.expected.decision_contract,
        artifact_root=root,
    )
    filed = verify_d1_filing(
        detached,
        prepared.verdict.verified.aggregate,
        revisions,
        store,
        _filing_git(prepared, revisions),
    )
    assert filed.verdict.schema_version == "d1-verdict/v3"
    assert filed.verdict.verdict == "VALID_NEGATIVE" and filed.verdict.exclusions == ()
    assert all(row.unavailable is not None for row in filed.verdict.contrasts)
