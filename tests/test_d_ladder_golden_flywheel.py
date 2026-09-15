"""Local scientific plumbing evidence, not real confirmatory research results.

The golden runs use an intentionally broad, predeclared toy acceptance margin.
Only transport/Git/review services are doubles; every analyzed metric comes
from the actual tiny trainer whose captured checkpoint binds those bytes.
"""

import json
from dataclasses import replace

import pytest

from entropic_substrate.analysis.ladder_analysis import analyze_ladder_sources
from entropic_substrate.analysis.ladder_filing import decide_ladder_outcome
from entropic_substrate.analysis.seed_sizing import ContrastHypothesis
from entropic_substrate.evidence.ladder_campaign import authorize_ladder_run
from entropic_substrate.evidence.lifecycle import SuccessorAuthorization

from .d_ladder_golden_fixtures import (
    ANALYST,
    OPERATOR,
    build_campaign,
    execute_campaign,
    file_campaign,
    predecessor_gate,
)

pytestmark = pytest.mark.unit


def test_actual_d2_through_d5_share_one_producer_capture_analysis_filing_lineage(
    tmp_path, monkeypatch
):
    predecessor = predecessor_gate(tmp_path, monkeypatch)
    profiles = {
        "d2": ("d2-l", "d2-s", "d2-g"),
        "d3": ("d3-h1", "d3-h4"),
        "d4": ("d4-identity", "d4-fixed"),
        "d5": ("d5-identity", "d5-permuted"),
    }
    for rung, experiments in profiles.items():
        prepared = build_campaign(tmp_path, rung, experiments, predecessor, real_git=rung == "d2")
        captured = execute_campaign(prepared)
        filed, report, packet = file_campaign(captured)
        assert len(report["population"]) == len(experiments) * 2
        assert {row["seed"] for row in report["population"]} == {7, 8}
        assert all(row["run_class"] == "confirmatory" for row in report["population"])
        assert packet["capture_digest"] == report["capture_digest"]
        assert captured.dvc_events == ["add", "push", "verify"]
        assert report["contrasts"]["toy-control"]["secondary"]["deciding"] is False
        assert not any(name in str(packet["contrasts"]) for name in experiments)
        if rung == "d2":
            event = json.loads((tmp_path / "d2-runs/d2-g-7/step-000002.json").read_bytes())
            assert event["metrics"]["batches"]
            for batch in event["metrics"]["batches"]:
                assert "fisher_iterations" in batch
                assert "fisher_metric_action_norm" in batch
        if rung == "d5":
            assert filed.next_action == "complete"
            assert filed.gate is None
        else:
            assert filed.next_action == "successor"
            assert filed.gate is not None
            filed.gate.verify_files(tmp_path)
            predecessor = filed.gate


def test_same_lineage_refuses_missing_capture_freshness_and_major_review(tmp_path, monkeypatch):
    prepared = build_campaign(
        tmp_path, "d2", ("d2-s", "d2-g"), predecessor_gate(tmp_path, monkeypatch)
    )
    captured = execute_campaign(prepared)
    with pytest.raises(ValueError, match="fresh"):
        analyze_ladder_sources(
            tmp_path, captured.inputs, analyst=replace(ANALYST, session_id="builder")
        )
    with pytest.raises(ValueError, match="DVC"):
        analyze_ladder_sources(
            tmp_path, replace(captured.inputs, dvc_handoff=None), analyst=ANALYST
        )
    with pytest.raises(ValueError, match="open findings"):
        file_campaign(captured, open_major_findings=["fixture reviewer requires correction"])
    assert not (tmp_path / "d2-verdict.json").exists()
    checkpoint = next((tmp_path / "d2-runs").rglob("*.pt"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises((RuntimeError, ValueError), match=r"identity changed|digest"):
        analyze_ladder_sources(tmp_path, captured.inputs, analyst=ANALYST)


def test_excluded_pilot_capture_cannot_enter_confirmation_analysis(tmp_path, monkeypatch):
    prepared = build_campaign(
        tmp_path, "d2", ("d2-s", "d2-g"), predecessor_gate(tmp_path, monkeypatch), run_class="pilot"
    )
    captured = execute_campaign(prepared)
    with pytest.raises(ValueError, match="pilot"):
        analyze_ladder_sources(tmp_path, captured.inputs, analyst=ANALYST)


def test_registration_uses_actual_git_blob_and_ancestry_adapter(tmp_path, monkeypatch):
    prepared = build_campaign(
        tmp_path, "d2", ("d2-s", "d2-g"), predecessor_gate(tmp_path, monkeypatch), real_git=True
    )
    auth = authorize_ladder_run(
        tmp_path,
        prepared.design,
        "d2-s-7",
        operator=OPERATOR,
        predecessor=prepared.predecessor,
        registration_git=prepared.git,
    )
    assert auth.run_class == "confirmatory"
    assert auth.seed == 7


@pytest.mark.parametrize(
    "rung,interval,expected",
    [
        ("d2", (0.02, 0.03), "successor"),
        ("d2", (-0.03, -0.02), "valid_negative"),
        ("d2", (-0.01, 0.01), "operator_decision"),
        ("d5", (0.02, 0.03), "complete"),
    ],
)
def test_synthetic_policy_cases_are_separate_from_actual_golden_measurements(
    rung, interval, expected
):
    rule = ContrastHypothesis("claim", "nats/byte", "superiority", 0.01, 0, None)
    posture, action, _ = decide_ladder_outcome(
        rung, (rule,), {"claim": interval}, {"route": ("claim",)}
    )
    assert action == expected
    assert (posture is SuccessorAuthorization.AUTHORIZED) == (expected == "successor")
