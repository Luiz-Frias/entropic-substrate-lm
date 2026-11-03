"""FOR-1694 (h1-l4c): the pair's remote path that needs the checkpoint-reader members.

Capture by reference (``worker_snapshot``, ``object_worker``), per-cell resume (``worker_request``,
``worker_execution``, ``worker_transport``) and the pair publish verb (``worker_command``). These
tests ride the DG-A patch with the member edits: at the lane tip, which cannot carry a member edit
(the pre-commit digest test refuses it), they do not apply.

Every PUT and read count below is measured at the object PORT (the fake store's ``publish`` and
``read`` calls), never read back from the worker's own publication block, which is computed from the
ledger and so cannot be evidence about the client.
"""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_codec_handoff import Pair, identity_of, make_pair, warmup_final
from tests.test_h_pair_remote import CountingStore, collect, pair_world
from tests.test_h_pair_request import marker_store, pair_request, run_command

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.h_probe_resume import (
    PAIR_RESUME_SOURCES,
    build_pair_resume_request,
)
from entropic_substrate.ladder.run.object_worker import encode_stop_marker
from entropic_substrate.ladder.run.worker_request import (
    PairCell,
    PairProbeRequest,
    parse_probe_request,
)
from entropic_substrate.ladder.run.worker_snapshot import materialize_worker_snapshot
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit

MAX = 10_000_000
# The digest of ``pair_request(make_pair(...))`` taken at the UNMODIFIED parent tip b429635e
# (2026-11-02), before per-cell resume was admitted: admitting it changed no byte of a v3 payload.
PARENT_TIP_PAIR_DIGEST = (
    "59cb1ab7fc8d2fee4494a4a87c2d4a8d83343af8b20f8b889597310657e4fd3d"  # pragma: allowlist secret
)


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return make_pair(tmp_path)


def counted_world(tmp_path: Path, pair: Pair) -> tuple[PairProbeRequest, CountingStore, Any]:
    request = pair_request(pair)
    objects = CountingStore()
    marker = publish_probe_request(objects, request, tmp_path)
    objects.publishes.clear()
    objects.reads.clear()
    return request, objects, marker


# ------------------------------------------------------------------ the request: per-cell resume
def test_a_v3_pair_payload_keeps_its_bytes_and_digest_now_that_resume_is_admitted(
    pair: Pair,
) -> None:
    request = pair_request(pair)
    assert request.digest == PARENT_TIP_PAIR_DIGEST
    assert parse_probe_request(request.payload, max_bytes=1_000_000) == request
    assert [cell["resume"] for cell in request.manifest["cells"]] == [None, None]  # type: ignore[index]


def evidence_root(
    tmp_path: Path, objects: Any, request: PairProbeRequest, event: str, name: str
) -> Path:
    found = read_probe_event(objects, request, event, expected_pod_id=None, max_bytes=MAX)
    assert found is not None and found.snapshot is not None
    destination = tmp_path / name
    destination.mkdir()
    materialize_worker_snapshot(objects, request, found.snapshot, destination, max_marker_bytes=MAX)
    return destination


def completed_world(tmp_path: Path, pair: Pair) -> tuple[PairProbeRequest, _ObjectStore]:
    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    return request, objects


def run_resumed(
    tmp_path: Path, pair: Pair, resumed: PairProbeRequest, evidence: Path, name: str
) -> tuple[dict[str, Any], Path, Path]:
    """Publish ``resumed`` from a local input root (the pair's source plus the collected run
    roots) to a FRESH store and run it as a new pod; returns result v2, the workspace, the logs."""
    inputs = tmp_path / f"inputs-{name}"
    inputs.mkdir()
    shutil.copytree(pair.source, inputs / "source")
    shutil.copytree(evidence / "run", inputs / "run")
    store = _ObjectStore()
    marker = publish_probe_request(store, resumed, inputs)
    status, error, workspace, observability = run_command(tmp_path, store, marker, name=name)
    assert status == 0, error
    return json.loads((workspace / "worker/result.json").read_text()), workspace, observability


def test_the_three_death_cases_build_the_right_request_and_the_resumed_pair_completes(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects = completed_world(tmp_path, pair)
    cases = {
        # (1) the donor died during warm-up: it resumes from its last receipt
        "donor-warmup": ("checkpoint-donor-00000002", "donor", "codec_warmup-00000002", None),
        # (2) after the donor's training, before the recipient's first prediction seal: the donor
        # carries its final receipt and the recipient is RE-MATERIALIZED (no recipient resume, and
        # no hand-off receipt is named anywhere)
        "after-donor": ("checkpoint-donor-00000006", "donor", "prediction-00000004", None),
        # (3) the recipient trained past its hand-off: it resumes from its OWN receipt, the donor
        # root rides along as inputs
        "recipient": (
            "checkpoint-recipient-00000004",
            "donor",
            "prediction-00000004",
            "prediction-00000002",
        ),
    }
    for label, (event, _donor, donor_stage, recipient_stage) in cases.items():
        evidence = evidence_root(tmp_path, objects, request, event, f"ev-{label}")
        resumed = build_pair_resume_request(request, evidence, nonce="2" * 64)
        donor_cell, recipient_cell = resumed.cells
        assert donor_cell.resume is not None
        assert donor_cell.resume.path.startswith("run/donor/")
        assert donor_cell.resume.path.endswith(f"-{donor_stage}-progress.json"), label
        if recipient_stage is None:
            assert recipient_cell.resume is None, label
            assert not any(a.path.startswith("run/recipient/") for a in resumed.inputs)
        else:
            assert recipient_cell.resume is not None
            assert recipient_cell.resume.path.endswith(f"-{recipient_stage}-progress.json")
            assert not recipient_cell.resume.path.split("/")[-1].startswith("handoff-")
            assert any(a.path.startswith("run/recipient/") for a in resumed.inputs)
        assert any(a.path.startswith("run/donor/") for a in resumed.inputs)
        # the resumed pair re-derives its pair id: same block, same authority, same two profiles
        assert resumed.binding == request.binding and resumed.authority == request.authority
        assert resumed.run_identity == request.run_identity and resumed.nonce == "2" * 64
        assert resumed.digest != request.digest  # a fresh nonce names a fresh run
        assert parse_probe_request(resumed.payload, max_bytes=1_000_000) == resumed
        # and a fresh pod runs it to completion
        result, _, _ = run_resumed(tmp_path, pair, resumed, evidence, f"pod-{label}")
        assert result["status"] == "probe_workload_complete"
        assert [(c["outcome"], c["successful_optimizer_steps"]) for c in result["cells"]] == [
            ("complete", 6),
            ("complete", 6),
        ], label


def test_a_recipient_with_only_its_handoff_receipt_is_re_materialized_never_resumed_from_it(
    tmp_path: Path, pair: Pair
) -> None:
    """R3: the evidence of a recipient that died before its first prediction seal holds only the
    hand-off receipt (``handoff-codec_warmup-*``); the request built from it names no recipient
    resume, and no request may name that receipt as one."""
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-r3")
    # keep ONLY the hand-off receipt of the recipient's root, as the dead pod would have left it
    for path in (evidence / "run" / "recipient").glob("*-prediction-*"):
        path.unlink()
    assert sorted(p.name for p in (evidence / "run" / "recipient").glob("*progress.json")) == [
        "handoff-codec_warmup-00000002-progress.json"
    ]
    resumed = build_pair_resume_request(request, evidence, nonce="3" * 64)
    assert resumed.cells[1].resume is None
    # nothing in the built request names the hand-off receipt, as a resume or as an input
    assert not any(
        a.path.endswith("handoff-codec_warmup-00000002-progress.json") for a in resumed.inputs
    )
    # a hand-built request that DOES name the hand-off receipt is refused at the gate, by name
    receipt = evidence / "run" / "recipient" / "handoff-codec_warmup-00000002-progress.json"
    named = ArtifactIdentity(
        "run/recipient/handoff-codec_warmup-00000002-progress.json",
        ContentHash("sha256", hashlib.sha256(receipt.read_bytes()).hexdigest()),
    )
    from dataclasses import replace

    with pytest.raises(ValueError, match="never names its hand-off receipt"):
        replace(
            resumed,
            cells=(resumed.cells[0], PairCell(request.recipient, named, None)),
            inputs=(*resumed.inputs, named),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "run_inputs_without_a_resume",
        "resume_not_an_input",
        "resume_under_another_cell",
        "resume_not_a_receipt_name",
        "resume_of_a_cell_with_foreign_root_file",
        "recipient_resume_without_donor_root",
    ],
)
def test_the_request_gate_refuses_every_resume_that_is_not_its_cells_own_receipt(
    tmp_path: Path, pair: Pair, mutation: str
) -> None:
    from dataclasses import replace

    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-gate")
    good = build_pair_resume_request(request, evidence, nonce="2" * 64)
    donor_resume, recipient_resume = good.cells[0].resume, good.cells[1].resume
    assert donor_resume is not None and recipient_resume is not None
    kept = tuple(a for a in good.inputs if not a.path.startswith("run/"))
    donor_files = tuple(a for a in good.inputs if a.path.startswith("run/donor/"))
    recipient_files = tuple(a for a in good.inputs if a.path.startswith("run/recipient/"))
    some = ArtifactIdentity("run/donor/forged.json", ContentHash("sha256", "a" * 64))
    if mutation == "run_inputs_without_a_resume":
        bad: dict[str, Any] = {
            "cells": (PairCell(request.donor, None, None), PairCell(request.recipient, None, None)),
            "inputs": (*kept, *donor_files),
        }
        match = "outside the declared input namespace"
    elif mutation == "resume_not_an_input":
        bad = {"inputs": (*kept, *donor_files)}  # the recipient resume is no input
        match = "must bind its own training receipt in run/<cell>/"
    elif mutation == "resume_under_another_cell":
        wrong = ArtifactIdentity(
            donor_resume.path.replace("run/donor/", "run/recipient/"), donor_resume.content_hash
        )
        bad = {
            "cells": (PairCell(request.donor, wrong, None), good.cells[1]),
            "inputs": (*good.inputs, wrong),
        }
        match = "must bind its own training receipt in run/<cell>/"
    elif mutation == "resume_not_a_receipt_name":
        journal = next(a for a in donor_files if a.path.endswith("-journal.json"))
        bad = {"cells": (PairCell(request.donor, journal, None), good.cells[1])}
        match = "must bind its own training receipt in run/<cell>/"
    elif mutation == "resume_of_a_cell_with_foreign_root_file":
        # the donor does not resume, yet ships its run/donor/ files: a hidden input
        bad = {
            "cells": (PairCell(request.donor, None, None), PairCell(request.recipient, None, None)),
            "inputs": (*kept, some),
        }
        match = "outside the declared input namespace"
    else:
        bad = {
            "cells": (PairCell(request.donor, None, None), good.cells[1]),
            "inputs": (*kept, *recipient_files),
        }
        match = "requires the donor's root among its inputs"
    with pytest.raises(ValueError, match=match):
        replace(good, **bad)


# ---------------------------------------------------------------------- the builder's refusals
def test_a_resume_is_built_only_from_a_pause_a_stop_or_the_dh45_failure(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_h_pair_request import Fault

    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    assert PAIR_RESUME_SOURCES == (
        "pause_bytes_verified",
        "stop_bytes_verified",
        "dh45_post_training_failure",
    )
    # a stopped pair IS a source (and resumes, in the capstone below); a COMPLETE pair, an
    # excluded pair and a reader_refusal failure are recorded outcomes and never are
    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    report = collect(tmp_path, objects, request, "c-complete")
    assert report is not None
    evidence = tmp_path / "c-complete"
    with pytest.raises(ValueError, match="not a resumable source"):
        build_pair_resume_request(request, evidence, nonce="2" * 64, collection=report)
    excluded_world = tmp_path / "excluded"
    excluded_world.mkdir()
    ex_pair = make_pair(excluded_world)
    ex_request, ex_objects, ex_marker = pair_world(excluded_world, ex_pair)
    with monkeypatch.context() as patch:
        Fault(patch, "prediction", 1, UndefinedDirection(2))
        status, error, _, _ = run_command(excluded_world, ex_objects, ex_marker)
    assert status == 0, error
    ex_report = collect(excluded_world, ex_objects, ex_request, "c-excluded")
    assert ex_report is not None and ex_report["terminal_name"] == "excluded"
    with pytest.raises(ValueError, match="A recorded exclusion is never overwritten"):
        build_pair_resume_request(
            ex_request, excluded_world / "c-excluded", nonce="2" * 64, collection=ex_report
        )
    refused_world = tmp_path / "refused"
    refused_world.mkdir()
    rf_pair = make_pair(refused_world)
    rf_request, rf_objects, rf_marker = pair_world(refused_world, rf_pair)
    with monkeypatch.context() as patch:
        Fault(patch, "warmup", 1, RuntimeError("injected reader refusal"))
        status, _, _, _ = run_command(refused_world, rf_objects, rf_marker)
    assert status == 2
    rf_report = collect(refused_world, rf_objects, rf_request, "c-refused")
    assert rf_report is not None and rf_report["verdict"]["reason"] == "reader_refusal"
    with pytest.raises(ValueError, match="not a resumable source"):
        build_pair_resume_request(
            rf_request, refused_world / "c-refused", nonce="2" * 64, collection=rf_report
        )


@pytest.mark.parametrize("outcome", ["scientific_gap", "diagnostic_gap"])
def test_a_post_training_gap_failure_is_registered_negative_evidence_never_a_resume_source(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Only an ``execution_defect`` failure is a DH-45 source: a scientific or diagnostic gap is a
    recorded outcome that a resumed lineage would overwrite. The collection is the REAL one of a
    pair whose second generation call raised (reported_outcome scientific_gap, no verdict); the
    diagnostic arm names the same collection with the other gap, as the producer would."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run import worker_execution

    request, objects, marker = pair_world(tmp_path, pair)
    real = worker_execution.run_probe_generation
    calls: list[str] = []

    def generation(*args: Any, **kwargs: Any) -> Any:
        calls.append("generation")
        if len(calls) == 2:
            raise UndefinedDirection(3)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_execution, "run_probe_generation", generation)
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2
    report = collect(tmp_path, objects, request, "c-gap")
    assert report is not None and report["terminal_name"] == "failure"
    assert report["capture_status"] == "failure_bytes_verified" and report["verdict"] is None
    assert report["reported_outcome"] == "scientific_gap"
    named = {**report, "reported_outcome": outcome}
    with pytest.raises(ValueError, match="not a resumable source"):
        build_pair_resume_request(request, tmp_path / "c-gap", nonce="2" * 64, collection=named)
    # the control: the same collection as an execution defect is the DH-45 source and builds
    control = {**report, "reported_outcome": "execution_defect"}
    resumed = build_pair_resume_request(
        request, tmp_path / "c-gap", nonce="2" * 64, collection=control
    )
    assert resumed.nonce == "2" * 64 and resumed.run_identity == request.run_identity


def test_the_builder_refuses_a_v1_collection_a_reused_nonce_and_evidence_of_nothing(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-donor-00000004", "ev-refuse")
    with pytest.raises(
        ValueError, match="minimum supported version is h-probe-terminal-collection-v2"
    ):
        build_pair_resume_request(
            request,
            evidence,
            nonce="2" * 64,
            collection={"schema": "h-probe-terminal-collection-v1"},
        )
    with pytest.raises(ValueError, match="fresh nonce"):
        build_pair_resume_request(request, evidence, nonce=request.nonce)
    empty = tmp_path / "empty"
    (empty / "run" / "donor").mkdir(parents=True)
    with pytest.raises(ValueError, match="holds no donor receipt"):
        build_pair_resume_request(request, empty, nonce="2" * 64)


# --------------------------------------------- the capstone: stop, resume, stitch with the proof
def test_a_stopped_pair_resumes_and_the_host_verifies_the_handoff_its_resumed_lineage_needs(
    tmp_path: Path, pair: Pair
) -> None:
    from dataclasses import asdict

    from tests.test_h_host_pair_verifier import SOURCE_LIMIT  # noqa: F401

    from entropic_substrate.ladder.analyze.h_aggregate import CodecHandoffEvidence
    from entropic_substrate.ladder.analyze.h_resumed_run import load_h_resumed_run
    from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap
    from entropic_substrate.ladder.run.host_pair_verifier import admit_codec_pair_on_host

    request = pair_request(pair)
    objects = marker_store(request, encode_stop_marker(request, ("prediction", 2, "recipient")))
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, workspace_a, observability_a = run_command(tmp_path, objects, marker)
    assert status == 0, error
    report = collect(tmp_path, objects, request, "collected-stop")
    assert report is not None and report["capture_status"] == "stop_bytes_verified"
    evidence = tmp_path / "collected-stop"
    resumed = build_pair_resume_request(request, evidence, nonce="2" * 64, collection=report)
    result, _, observability_b = run_resumed(tmp_path, pair, resumed, evidence, "pod-b")
    assert result["status"] == "probe_workload_complete"

    # R2: the recipient's two pods stitch into ONE lineage that starts at warm-up + 1 with zero
    # warm-up rows, but only with the hand-off evidence the HOST verifier admits from pod A's bytes
    seals = sorted((workspace_a / "run" / "recipient").glob("*-prediction-*-seal.json"))
    verified = admit_codec_pair_on_host(
        workspace_a / "run" / "donor",
        warmup_final(workspace_a / "run" / "donor"),
        pair.donor,
        workspace_a / "run" / "recipient",
        identity_of(seals[-1]),
        pair.recipient,
        workspace_a / "source",
        max_source_bytes=1_000_000,
        max_artifact_bytes=MAX,
    )
    proof = CodecHandoffEvidence(**asdict(verified))
    assert (proof.recipient_identity, proof.donor_identity, proof.warmup_steps) == (
        pair.recipient.identity,
        pair.donor.identity,
        2,
    )
    # pod B's recipient lineage starts after the seam pod A stopped at: global step 5, 6
    recipient_b = result["cells"][1]
    assert recipient_b["successful_optimizer_steps"] == 6
    for pod in (observability_a, observability_b):
        assert (pod / "recipient" / "metrics.jsonl").is_file()
        assert (pod / "recipient" / "worker-log.jsonl").is_file()
    # The REAL files cannot be stitched here: this tip's analysis loader refuses a pair cell's
    # null future term (future_weight 0), which is FOR-1716 and out of this slice. The stitch
    # reader's R2 contract is pinned on the synthetic lineage of tests/test_h_resumed_run_handoff.py
    # with the SAME evidence type this verifier produces (one definition of its fields).
    with pytest.raises(TelemetryGap, match="future_nll_per_dimension"):
        load_h_resumed_run(
            [
                (
                    observability_a / "recipient" / "metrics.jsonl",
                    observability_a / "recipient" / "worker-log.jsonl",
                ),
                (
                    observability_b / "recipient" / "metrics.jsonl",
                    observability_b / "recipient" / "worker-log.jsonl",
                ),
            ],
            handoff=proof,
        )


# ------------------------------------------------------------------------ capture by reference
def test_the_completion_snapshot_publishes_capture_copies_as_references_with_no_second_put(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, marker = counted_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    snapshot_key = f"{request.prefix}/snapshots/complete.json"
    manifest = json.loads(objects.values[snapshot_key])
    assert manifest["schema"] == "h-worker-snapshot-v2"
    files = {item["path"]: item for item in manifest["files"]}
    references = {path: item for path, item in files.items() if "reference" in item}
    assert references and all(path.startswith("capture/") for path in references)
    for path, item in references.items():
        twin = files[item["reference"]]
        cell, name = path.split("/")[1], path.split("/")[-1]
        assert item["reference"] == f"run/{cell}/{name}" and "reference" not in twin
        assert (item["sha256"], item["size_bytes"], item["key"]) == (
            twin["sha256"],
            twin["size_bytes"],
            twin["key"],
        )
    # the capture files with no run twin are real objects (the manifest and the source copy)
    real_capture = sorted(p for p in files if p.startswith("capture/") and p not in references)
    assert real_capture == [
        "capture/donor/capture.json",
        "capture/donor/source/prepared-corpus.json",
        "capture/recipient/capture.json",
        "capture/recipient/source/prepared-corpus.json",
    ]
    # AT THE PORT: no object key under capture/ was published for a reference, and no key twice
    published_capture = [k for k in objects.publishes if "/outputs/capture/" in k]
    assert sorted(published_capture) == sorted(
        f"{request.prefix}/outputs/{p}" for p in real_capture
    )
    assert len(published_capture) == 4
    run_published = [k for k in objects.publishes if "/outputs/run/" in k]
    assert len(set(run_published)) == len(run_published)  # each run object exactly once
    # artifact_bytes counts a reference exactly as v1 counted the copy: no budget loosens
    assert manifest["artifact_count"] == len(files) == 88
    assert manifest["artifact_bytes"] == sum(i["size_bytes"] for i in files.values())
    print(
        f"COUNTS-AFTER publishes={len(objects.publishes)} capture_outputs={len(published_capture)} "
        f"reads={len(objects.reads)}"
    )


def test_the_host_resolves_a_reference_to_the_same_bytes_with_one_read_per_object(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, marker = counted_world(tmp_path, pair)
    status, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    found = read_probe_event(objects, request, "complete", expected_pod_id=None, max_bytes=MAX)
    assert found is not None and found.snapshot is not None
    destination = tmp_path / "host"
    destination.mkdir()
    objects.reads.clear()
    row = materialize_worker_snapshot(
        objects, request, found.snapshot, destination, max_marker_bytes=MAX
    )
    references = [i for i in row["files"] if "reference" in i]
    assert references
    for item in references:
        assert (destination / item["path"]).read_bytes() == (
            destination / item["reference"]
        ).read_bytes()
        assert (destination / item["path"]).read_bytes() == (workspace / item["path"]).read_bytes()
    # ONE read for the marker and ONE per distinct referenced object: the reference never costs a read
    keys = [i["key"] for i in row["files"]]
    assert sorted(objects.reads) == sorted([found.snapshot.key, *set(keys)])
    assert len(references) > 0 and len(set(keys)) == len(keys) - len(references)


def forged_marker(
    objects: _ObjectStore, request: PairProbeRequest, change: Any
) -> tuple[ArtifactIdentity, Any]:
    key = f"{request.prefix}/snapshots/complete.json"
    manifest = json.loads(objects.values[key])
    change(manifest)
    payload = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[key] = payload
    from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
        D1TrainerMailboxObjectIdentity,
    )

    return key, D1TrainerMailboxObjectIdentity(
        key, len(payload), hashlib.sha256(payload).hexdigest()
    )  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("unequal_bytes", "reference does not name an equal run entry"),
        ("chained", "reference names another cell or file"),
        ("other_cell", "reference names another cell or file"),
        ("other_file", "reference names another cell or file"),
        ("wrong_key", "entry budget or identity mismatch"),
        ("dangling", "reference does not name an equal run entry"),
        ("v1_with_reference", "recovery"),
    ],
)
def test_the_host_refuses_a_reference_that_is_not_an_equal_run_twin_of_its_own_cell(
    tmp_path: Path, pair: Pair, kind: str, message: str
) -> None:
    request, objects, marker = counted_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error

    def tamper(manifest: dict[str, Any]) -> None:
        files = manifest["files"]
        capture = next(
            f for f in files if "reference" in f and f["path"].startswith("capture/donor/")
        )
        run = next(f for f in files if f["path"] == capture["reference"])
        if kind == "unequal_bytes":
            capture["sha256"] = "0" * 64
        elif kind == "chained":
            other = next(
                f
                for f in files
                if "reference" in f and f is not capture and f["path"].startswith("capture/donor/")
            )
            capture["reference"] = other["path"]
            capture["key"] = other["key"]
            capture["sha256"], capture["size_bytes"] = other["sha256"], other["size_bytes"]
        elif kind == "other_cell":
            capture["reference"] = capture["reference"].replace("run/donor/", "run/recipient/")
        elif kind == "other_file":
            capture["reference"] = capture["reference"].replace("run/donor/", "run/donor/x-")
        elif kind == "wrong_key":
            capture["key"] = f"{request.prefix}/outputs/{capture['path']}"  # not the twin's key
        elif kind == "dangling":
            files.remove(run)
            manifest["artifact_count"] = len(files)
            manifest["artifact_bytes"] = sum(f["size_bytes"] for f in files)
        else:
            manifest["schema"] = "h-worker-snapshot-v1"

    _, identity = forged_marker(objects, request, tamper)
    destination = tmp_path / "host-forged"
    destination.mkdir()
    with pytest.raises(ValueError, match=message):
        materialize_worker_snapshot(objects, request, identity, destination, max_marker_bytes=MAX)


def test_a_v2_marker_on_a_single_profile_request_is_refused_by_name_and_v1_stays_for_singles(
    tmp_path: Path,
) -> None:
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run import object_worker

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "executed"
    worker.mkdir()
    object_worker.run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    key = f"{request.prefix}/snapshots/complete.json"
    manifest = json.loads(objects.values[key])
    assert manifest["schema"] == "h-worker-snapshot-v1"  # a single profile still writes v1
    assert all("reference" not in f for f in manifest["files"])
    manifest["schema"] = "h-worker-snapshot-v2"
    payload = json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    objects.values[key] = payload
    from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
        D1TrainerMailboxObjectIdentity,
    )

    identity = D1TrainerMailboxObjectIdentity(
        key, len(payload), hashlib.sha256(payload).hexdigest()
    )
    destination = tmp_path / "host-single"
    destination.mkdir()
    with pytest.raises(ValueError, match="a single-profile request admits h-worker-snapshot-v1"):
        materialize_worker_snapshot(objects, request, identity, destination, max_marker_bytes=MAX)


def test_a_pairs_failure_snapshot_republishes_nothing_the_run_already_published(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured defect (2026-10-28): the failure snapshot published with no ledger and
    re-PUT 62 files for 1,382.7 s. A pair's failure after both cells trained must publish only the
    objects that are NEW since the last snapshot, and never one the run already verified."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run import worker_execution

    request, objects, marker = counted_world(tmp_path, pair)
    real = worker_execution.run_probe_generation
    calls: list[str] = []

    def generation(*args: Any, **kwargs: Any) -> Any:
        calls.append("generation")
        if len(calls) == 2:  # the recipient's: both cells' training is already published
            raise UndefinedDirection(3)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_execution, "run_probe_generation", generation)
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2
    counts: dict[str, int] = {}
    for key in objects.publishes:
        counts[key] = counts.get(key, 0) + 1
    repeated = {k: n for k, n in counts.items() if n > 1 and "/outputs/" in k}
    assert repeated == {}, repeated  # no output object is PUT twice, the failure snapshot included
    marker_json = json.loads(objects.values[f"{request.prefix}/snapshots/failure.json"])
    assert marker_json["schema"] == "h-worker-snapshot-v2"
    # and it still materializes on a host
    found = read_probe_event(objects, request, "failure", expected_pod_id=None, max_bytes=MAX)
    assert found is not None and found.snapshot is not None
    destination = tmp_path / "host-failure"
    destination.mkdir()
    materialize_worker_snapshot(objects, request, found.snapshot, destination, max_marker_bytes=MAX)


def test_the_pair_publication_block_keeps_its_exact_seven_fields(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, marker = counted_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    event = json.loads(objects.values[f"{request.prefix}/complete.json"])
    assert set(event["publication"]) == {
        "l1_seconds",
        "l2_seconds",
        "l3_seconds",
        "objects_put",
        "objects_skipped",
        "verification_reads",
        "retries",
    }


def test_a_resumed_pair_keeps_one_progress_ladder_per_cell_above_each_cells_own_floor(
    tmp_path: Path, pair: Pair
) -> None:
    """The controller's advisory tick for a RESUMED pair: each cell's ladder starts after that
    cell's own resume step. Unreachable at the lane tip (the request parser refuses a pair
    resume there), so it is pinned here with the member patch. Control first: the unresumed pair."""
    from entropic_substrate.ladder.run.worker.h_probe_controller import _CheckpointProgress

    request, objects = completed_world(tmp_path, pair)

    def ladders(candidate: PairProbeRequest) -> list[tuple[int, ...]]:
        return [
            ladder.steps
            for ladder in _CheckpointProgress(candidate)._build()  # pyright: ignore[reportPrivateUsage]
        ]

    assert ladders(request) == [(2, 4, 6), (4, 6)]  # control: donor warm-up seal, recipient floor
    expected = {
        "checkpoint-donor-00000002": [(4, 6), (4, 6)],  # donor died in warm-up
        "checkpoint-donor-00000006": [(), (4, 6)],  # after the donor trained: recipient fresh
        "checkpoint-recipient-00000004": [(), (6,)],  # the recipient's own receipt is its floor
    }
    for event, steps in expected.items():
        evidence = evidence_root(tmp_path, objects, request, event, f"ev-{event}")
        resumed = build_pair_resume_request(request, evidence, nonce="3" * 64)
        assert ladders(resumed) == steps, event


# --------------------------------------------- the reference producer and a referenced object
@pytest.mark.parametrize("edit", ["appended", "same_size_one_byte"])
def test_a_capture_copy_whose_bytes_differ_from_its_run_twin_is_published_not_referenced(
    tmp_path: Path, pair: Pair, edit: str
) -> None:
    """The producer's sha256 equality has a tooth: a same-named capture and run file whose bytes
    DIFFER is a real object, never a reference to the run twin. ``appended`` changes the size and
    the digest together; ``same_size_one_byte`` keeps the size, so only the digest separates them.
    Size equality is not a separate tooth: equal SHA-256 digests imply equal bytes, hence size."""
    from entropic_substrate.ladder.run.worker_snapshot import publish_worker_snapshot

    request, objects, marker = counted_world(tmp_path, pair)
    status, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    manifest = json.loads(objects.values[f"{request.prefix}/snapshots/complete.json"])
    reference = next(f for f in manifest["files"] if "reference" in f)
    capture_path, run_path = reference["path"], reference["reference"]
    original = (workspace / capture_path).read_bytes()
    if edit == "appended":
        changed = original + b" "
    else:
        changed = original[:-1] + bytes([original[-1] ^ 1])
        assert len(changed) == len(original)
    (workspace / capture_path).write_bytes(changed)
    # control: the completed pair above holds this very file as a reference; the bytes now differ
    publish_worker_snapshot(
        objects, request, workspace, (run_path, capture_path), name="forged-copy"
    )
    forged = {
        f["path"]: f
        for f in json.loads(objects.values[f"{request.prefix}/snapshots/forged-copy.json"])["files"]
    }
    assert "reference" not in forged[capture_path]
    assert forged[capture_path]["key"] == f"{request.prefix}/outputs/{capture_path}"
    assert objects.values[f"{request.prefix}/outputs/{capture_path}"] == changed
    assert forged[capture_path]["sha256"] != forged[run_path]["sha256"]


def test_only_a_flat_capture_file_has_a_run_twin_to_reference() -> None:
    """``reference_path`` maps exactly ``capture/<cell>/<file>`` to ``run/<cell>/<file>``. Deeper
    capture paths, verification paths and run paths have no twin and never reference."""
    from entropic_substrate.ladder.run.worker_snapshot import reference_path

    assert reference_path("capture/recipient/progress.json") == "run/recipient/progress.json"
    assert reference_path("capture/recipient/source/prepared-corpus.json") is None
    assert reference_path("capture/recipient/a/b.json") is None
    assert reference_path("verification/recipient/progress.json") is None
    assert reference_path("run/recipient/progress.json") is None


def test_the_host_refuses_a_referenced_object_whose_stored_bytes_were_corrupted(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, marker = counted_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    found = read_probe_event(objects, request, "complete", expected_pod_id=None, max_bytes=MAX)
    assert found is not None and found.snapshot is not None
    manifest = json.loads(objects.values[found.snapshot.key])
    twin = next(f for f in manifest["files"] if "reference" in f)["reference"]
    key = next(f["key"] for f in manifest["files"] if f["path"] == twin)
    stored = objects.values[key]
    destination = tmp_path / "host-control"
    destination.mkdir()
    materialize_worker_snapshot(  # control: the intact store materializes
        objects, request, found.snapshot, destination, max_marker_bytes=MAX
    )
    objects.values[key] = bytes([stored[0] ^ 1]) + stored[1:]  # same size, one flipped bit
    corrupted = tmp_path / "host-corrupted"
    corrupted.mkdir()
    with pytest.raises(ValueError):
        materialize_worker_snapshot(
            objects, request, found.snapshot, corrupted, max_marker_bytes=MAX
        )


# ------------------------------------------------- the builder reads the evidence it is given
def forge_newest_receipt(evidence: Path, token: str, name: str, mutate: Any) -> Path:
    """A copy of ``evidence`` whose newest ``token`` receipt is ``mutate(row)``."""
    forged = evidence.parent / name
    shutil.copytree(evidence, forged)
    root = forged / "run" / token
    rows = {path: json.loads(path.read_text()) for path in sorted(root.glob("*-progress.json"))}
    newest = max(rows, key=lambda path: rows[path]["successful_optimizer_steps"])
    newest.write_text(json.dumps(mutate(rows[newest]), sort_keys=True, separators=(",", ":")))
    return forged


def test_a_resumed_pair_can_itself_be_resumed_and_ships_no_stale_run_inputs(
    tmp_path: Path, pair: Pair
) -> None:
    """Second-generation resume: the next request keeps the non-``run/`` inputs of the previous one
    and REPLACES its ``run/`` roots with the new evidence's, so no path is named twice and no root
    of the earlier generation rides along."""
    request, objects = completed_world(tmp_path, pair)
    first_evidence = evidence_root(
        tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-gen1"
    )
    first = build_pair_resume_request(request, first_evidence, nonce="2" * 64)
    assert any(a.path.startswith("run/recipient/") for a in first.inputs)
    second_evidence = evidence_root(
        tmp_path, objects, request, "checkpoint-donor-00000002", "ev-gen2"
    )
    second = build_pair_resume_request(first, second_evidence, nonce="3" * 64)
    paths = [a.path for a in second.inputs]
    assert len(paths) == len(set(paths))
    # the recipient root of generation one is gone; the donor root is generation two's own
    assert not any(path.startswith("run/recipient/") for path in paths)
    assert {a for a in second.inputs if a.path.startswith("run/donor/")} == {
        a for a in second.inputs if a.path.startswith("run/")
    }
    assert [a for a in second.inputs if not a.path.startswith("run/")] == [
        a for a in first.inputs if not a.path.startswith("run/")
    ]
    assert second.nonce == "3" * 64 and second.cells[1].resume is None
    assert parse_probe_request(second.payload, max_bytes=1_000_000) == second


def test_a_donor_resume_names_only_a_warmup_or_prediction_receipt(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-donor-00000006", "ev-phase")
    build_pair_resume_request(request, evidence, nonce="2" * 64)  # control: the honest evidence
    forged = forge_newest_receipt(evidence, "donor", "ev-phase-bad", lambda r: {**r, "phase": "x"})
    with pytest.raises(ValueError, match="names a warm-up or prediction receipt"):
        build_pair_resume_request(request, forged, nonce="2" * 64)


def test_a_recipient_receipt_beside_an_unfinished_donor_is_refused_by_name(
    tmp_path: Path, pair: Pair
) -> None:
    """A recipient cannot have trained past its hand-off while the donor's training was still
    running: such evidence is contradictory, so the builder names it instead of picking a side."""
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-un")
    build_pair_resume_request(request, evidence, nonce="2" * 64)  # control: donor complete
    forged = forge_newest_receipt(
        evidence, "donor", "ev-un-bad", lambda r: {**r, "status": "paused"}
    )
    assert any((forged / "run" / "recipient").glob("*-progress.json"))
    with pytest.raises(ValueError, match="donor's training is not complete"):
        build_pair_resume_request(request, forged, nonce="2" * 64)


@pytest.mark.parametrize("cell", ["donor", "recipient"])
@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda row: {**row, "pair": "f" * 64}, "is not this pair's own"),
        (lambda row: {**row, "block": "f" * 64}, "is not this pair's own"),
        (lambda row: {**row, "cell": "foreign"}, "is not this pair's own"),
        (lambda row: {**row, "run_identity": "f" * 64}, "is not this pair's own"),
        (lambda row: {**row, "schema": "h-probe-training-progress-v9"}, "is not this pair's own"),
        # a retired v1 receipt is refused by NAME, with the minimum version
        (
            lambda row: {**row, "schema": "h-probe-training-progress-v1"},
            "minimum supported version is h-probe-training-progress-v2",
        ),
    ],
    ids=["pair", "block", "cell", "run_identity", "schema_v9", "schema_v1"],
)
def test_the_builder_refuses_a_receipt_that_is_not_this_pairs_own_cell(
    tmp_path: Path, pair: Pair, cell: str, mutate: Any, match: str
) -> None:
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-own")
    forged = forge_newest_receipt(evidence, cell, "ev-own-bad", mutate)
    with pytest.raises(ValueError, match=match):
        build_pair_resume_request(request, forged, nonce="2" * 64)


@pytest.mark.parametrize(
    "mutate",
    [
        # each conjunct of "this is the recipient's hand-off receipt" is separate evidence: a
        # receipt missing any ONE of them is the recipient's own training and is resumed from
        lambda row: {**row, "own_optimizer_steps": 0},
        lambda row: {**row, "phase": "codec_warmup"},
        lambda row: {**row, "handoff": None, "own_optimizer_steps": 0, "phase": "codec_warmup"},
    ],
    ids=["has_own_steps_zero_only", "is_warmup_only", "has_no_handoff_evidence"],
)
def test_a_recipient_receipt_is_a_handoff_only_when_all_three_marks_agree(
    tmp_path: Path, pair: Pair, mutate: Any
) -> None:
    request, objects = completed_world(tmp_path, pair)
    evidence = evidence_root(tmp_path, objects, request, "checkpoint-recipient-00000004", "ev-mark")
    forged = forge_newest_receipt(evidence, "recipient", "ev-mark-bad", mutate)
    resumed = build_pair_resume_request(request, forged, nonce="2" * 64)
    assert resumed.cells[1].resume is not None
    assert resumed.cells[1].resume.path.endswith("-prediction-00000002-progress.json")
