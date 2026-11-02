"""FOR-1694 (R2): a resumed codec-hand-off recipient lineage, admitted only with hand-off evidence.

The recipient's codec ARRIVED from the donor's warm-up-final seal, so its telemetry holds no
``codec_warmup`` row on any pod and starts at ``warmup_steps + 1``. Zero warm-up rows on their own
are exactly the forgery the hand-off tooth targets: only typed evidence admits them.
"""

from pathlib import Path
from typing import Any

import pytest
from tests.test_h_aggregate import IDENTITY_A, IDENTITY_B, REQUEST_A, REQUEST_B
from tests.test_h_final_probe_checks import _segment  # pyright: ignore[reportPrivateUsage]

from entropic_substrate.ladder.analyze import h_resumed_run
from entropic_substrate.ladder.analyze.h_aggregate import CodecHandoffEvidence, load_h_run
from entropic_substrate.ladder.analyze.h_resumed_run import load_h_resumed_run
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

pytestmark = pytest.mark.unit


def evidence(
    recipient: str = IDENTITY_A, warmup: int = 2, donor: str = IDENTITY_B
) -> CodecHandoffEvidence:
    return CodecHandoffEvidence(recipient, donor, warmup, "c" * 64, "d" * 64)


def recipient_lineage(
    directory: Path,
    *,
    a_warmup: tuple[int, ...] = (),
    b_prediction: tuple[int, ...] = (3, 4),
    a_status: str = "stopped_by_operator",
) -> list[tuple[Path, Path]]:
    """A recipient paused after its prediction step 2 (global 4) and resumed by a second pod."""
    first = _segment(
        directory,
        "a",
        identity=IDENTITY_A,
        request=REQUEST_A,
        warmup=a_warmup,
        prediction=(1, 2),
        result_status=a_status,
    )
    second = _segment(
        directory,
        "b",
        identity=IDENTITY_A,
        request=REQUEST_B,
        warmup=(),
        prediction=b_prediction,
        result_status="probe_workload_complete",
    )
    return [first, second]


def test_a_paused_and_resumed_recipient_lineage_loads_with_handoff_evidence(
    tmp_path: Path,
) -> None:
    proof = evidence()
    stitched = load_h_resumed_run(recipient_lineage(tmp_path), handoff=proof)
    run = stitched.run
    assert [step for step, _, _ in run.training] == [3, 4, 5, 6]  # starts at warmup + 1
    assert all(phase == "prediction" for _, phase, _ in run.training)  # zero warm-up rows
    assert run.warmup_steps == 2 and run.handoff == proof
    assert [s.request_sha256 for s in stitched.segments] == [REQUEST_A, REQUEST_B]
    assert [step for step, _ in run.evaluations] == [4, 6]


def test_the_same_lineage_without_handoff_evidence_stays_refused_with_the_existing_gap_text(
    tmp_path: Path,
) -> None:
    """The tooth: absence of warm-up rows is not proof of a hand-off."""
    with pytest.raises(TelemetryGap, match="global step axis is not continuous"):
        load_h_resumed_run(recipient_lineage(tmp_path))


def test_handoff_evidence_never_rides_with_a_warmup_override_in_one_load_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    real = load_h_run

    def spy(metrics: Any, log: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real(metrics, log, **kwargs)

    monkeypatch.setattr(h_resumed_run, "load_h_run", spy)
    load_h_resumed_run(recipient_lineage(tmp_path), handoff=evidence())
    assert len(calls) == 2
    assert not any("handoff" in kw and "warmup_steps" in kw for kw in calls)
    assert set(calls[0]) == {"handoff"} and set(calls[1]) == {"warmup_steps"}


@pytest.mark.parametrize(
    ("proof", "message"),
    [
        (evidence(recipient=IDENTITY_B), "is not the hand-off recipient"),
        (evidence(warmup=3), "hand-off evidence inherited 3"),
    ],
)
def test_evidence_for_another_recipient_or_warmup_is_refused(
    tmp_path: Path, proof: CodecHandoffEvidence, message: str
) -> None:
    with pytest.raises(TelemetryGap, match=message):
        load_h_resumed_run(recipient_lineage(tmp_path), handoff=proof)


def test_evidence_is_typed_and_a_lineage_with_warmup_rows_cannot_claim_a_handoff(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="CodecHandoffEvidence"):
        load_h_resumed_run(
            recipient_lineage(tmp_path / "typed"),
            handoff={"recipient_identity": IDENTITY_A},  # type: ignore[arg-type]
        )
    # a pod that DID commit warm-up rows trained its own codec: it is not a hand-off recipient
    with pytest.raises(TelemetryGap, match="commits no codec_warmup row"):
        load_h_resumed_run(
            recipient_lineage(tmp_path / "trained", a_warmup=(1, 2)), handoff=evidence()
        )


def test_a_gap_across_the_seam_is_still_refused_with_evidence(tmp_path: Path) -> None:
    with pytest.raises(TelemetryGap, match="not continuous across the resume seam"):
        load_h_resumed_run(recipient_lineage(tmp_path, b_prediction=(4,)), handoff=evidence())


def test_a_lineage_that_starts_at_step_one_is_unchanged_by_the_new_argument(tmp_path: Path) -> None:
    """The ordinary resumed lineage (warm-up trained on pod A) loads without evidence exactly as
    before, and is refused WITH evidence: its first pod commits warm-up rows."""
    first = _segment(
        tmp_path,
        "a",
        identity=IDENTITY_A,
        request=REQUEST_A,
        warmup=(1, 2),
        prediction=(1, 2),
        result_status="stopped_by_operator",
    )
    second = _segment(
        tmp_path,
        "b",
        identity=IDENTITY_A,
        request=REQUEST_B,
        warmup=(),
        prediction=(3, 4),
        result_status="probe_workload_complete",
    )
    plain = load_h_resumed_run([first, second])
    assert [step for step, _, _ in plain.run.training] == [1, 2, 3, 4, 5, 6]
    assert plain.run.handoff is None
    with pytest.raises(TelemetryGap):
        load_h_resumed_run([first, second], handoff=evidence())
