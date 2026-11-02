"""FOR-1693 (h1-l4b): scientific pair admission, end to end on CPU.

The tracer reuses L4a's pair fixture (two ``local_tiny`` H1 profiles that share a codec by
construction: a magnitude-spine donor and a sphere-spine recipient, warm-up 2, 4 steps). The
scientific binding (one sealed authority for both cells) cannot execute on a CPU: a pilot
profile requires CUDA, and no test patches admission to pretend otherwise. It is proven where
the real predicates live: the request parse and digest, the admission tables, and the receipt
binding.

Scoped 2026-11-02 (house rule 4): this module pins the smoke arm of the receipt binding only (an
authority digest on a smoke receipt is refused). The scientific arm, the digest the binding
carries, and the threading of the request's one authority into the pair path are pinned in
``tests/test_h_pair_request_pins.py`` (a pure-function pin, a pilot-profile reader pin with only
the seal walk stood in, and a structural pin on the call shapes); the executed run under an
authority belongs to the live pod wave, since a CPU cannot construct a scientific profile.
"""

import json
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_codec_handoff import Pair, make_pair
from tests.test_h_probe_launch import pilot_request
from tests.test_run_authority import authority as sealed_authority

from entropic_substrate.core.probe.predictive_probe import ProbeConfig
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.worker_request import (
    REQUEST_SCHEMA_V1,
    REQUEST_SCHEMA_V2,
    REQUEST_SCHEMA_V3,
    BlockBinding,
    PairCell,
    PairProbeRequest,
    ProbeRequest,
    TransferLimits,
    cell_token,
    pair_identity,
    parse_probe_request,
)

pytestmark = pytest.mark.unit

BLOCK = "b" * 64
LIMITS = TransferLimits(10_000_000, 400_000_000, 500)
# lags (2, 4): the fixture corpus (30 three-sentence documents) has no lag-64 span population
PROBE = ProbeConfig(lags=(2, 4), train_positions=8, eval_positions=8, updates=2, batch_size=4)


def binding_for(pair: Pair) -> BlockBinding:
    return BlockBinding(
        BLOCK,
        pair_identity(pair.donor, pair.recipient),
        pair.donor.source_digest,
        pair.donor.seed,
    )


def pair_request(pair: Pair, **changes: Any) -> PairProbeRequest:
    source = ArtifactIdentity(
        "source/prepared-corpus.json", ContentHash("sha256", pair.donor.source_digest)
    )
    fields: dict[str, Any] = {
        "cells": (PairCell(pair.donor, None, None), PairCell(pair.recipient, None, None)),
        "inputs": (source,),
        "nonce": "1" * 64,
        "limits": LIMITS,
        "probe": PROBE,
        "code_commit": None,
        "code_bundle": None,
        "image": None,
        "binding": binding_for(pair),
    }
    fields.update(changes)
    return PairProbeRequest(**fields)


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return make_pair(tmp_path)


def reencode(row: dict[str, object]) -> bytes:
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


# ------------------------------------------------------------------------ the tracer
def run_command(
    tmp_path: Path, objects: _ObjectStore, marker: Any, name: str = "command-worker"
) -> tuple[int, str, Path, Path]:
    from entropic_substrate.ladder.run.worker_command import run_worker_command

    workspace = tmp_path / name
    workspace.mkdir()
    output, error = StringIO(), StringIO()
    status = run_worker_command(
        [
            "run",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "10000000",
            "--max-objects",
            "500",
            "--workspace-root",
            str(workspace),
            "--request-key",
            marker.key,
            "--request-sha256",
            marker.sha256,
            "--request-size",
            str(marker.size_bytes),
            "--local-tiny",
        ],
        environ={},
        objects=objects,
        stdout=output,
        stderr=error,
    )
    return status, error.getvalue(), workspace, tmp_path / f"{name}.observability"


def test_a_pair_request_runs_both_cells_under_one_authority_and_the_recipient_cites_the_handoff(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.journal import read_json
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request = pair_request(pair)
    # a v3 pair request round-trips, and its digest binds the cells and the block binding
    assert parse_probe_request(request.payload, max_bytes=1_000_000) == request
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error

    # one MLflow-shaped run, metrics file and worker log per cell, each with its own lifecycle
    for token in ("donor", "recipient"):
        rows = [
            json.loads(line)
            for line in (observability / token / "metrics.jsonl").read_text().splitlines()
        ]
        lifecycle = [r for r in rows if r.get("schema") == "h-run-lifecycle-v1"]
        assert [r["event"] for r in lifecycle] == ["run_start", "run_end"]
        assert lifecycle[0]["request_sha256"] == request.digest
        assert lifecycle[1]["status"] == "ok"
        log = [
            json.loads(line)
            for line in (observability / token / "worker-log.jsonl").read_text().splitlines()
        ]
        assert [r["event"] for r in log].count("worker.start") == 1

    # the pair's result is result v2 with two cell entries and a complete verdict
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["schema"] == "h-probe-worker-result-v2"
    assert result["verdict"] == {"verdict": "complete", "reason": None, "cell": None}
    assert [e["cell"] for e in result["cells"]] == ["donor", "recipient"]
    assert all(e["outcome"] == "complete" for e in result["cells"])

    # two receipts; the recipient's cites the hand-off record and splits inherited from own
    donor_entry, recipient_entry = result["cells"]
    donor = read_json(
        workspace / "run" / "donor",
        ArtifactIdentity(
            donor_entry["training"]["path"].rsplit("/", 1)[1],
            ContentHash("sha256", donor_entry["training"]["sha256"]),
        ),
        max_bytes=10_000_000,
    )
    receipt = read_json(
        workspace / "run" / "recipient",
        ArtifactIdentity(
            recipient_entry["training"]["path"].rsplit("/", 1)[1],
            ContentHash("sha256", recipient_entry["training"]["sha256"]),
        ),
        max_bytes=10_000_000,
    )
    assert donor["schema"] == receipt["schema"] == "h-probe-training-progress-v2"
    assert (donor["inherited_warmup_steps"], donor["handoff"]) == (0, None)
    assert receipt["inherited_warmup_steps"] == pair.recipient.schedule.warmup_steps == 2
    assert receipt["own_optimizer_steps"] == pair.recipient.schedule.steps == 4
    assert receipt["successful_optimizer_steps"] == 6
    assert (receipt["block"], receipt["pair"], receipt["cell"]) == (
        BLOCK,
        request.binding.pair_id,
        "recipient",
    )
    record = read_json(
        workspace / "run" / "recipient",
        ArtifactIdentity(
            receipt["handoff"]["path"], ContentHash("sha256", receipt["handoff"]["sha256"])
        ),
        max_bytes=10_000_000,
    )
    assert record["schema"] == "h-codec-handoff-v1"
    assert record["donor_identity"] == pair.donor.identity


# -------------------------------------------------------------- the request: wire and digest
# MEASURED 2026-11-01 by hashing the same fixtures through the module at the parent tip
# (2994331f) and through this one: the v1 and v2 bytes did not move.
V1_SMOKE_DIGEST = (
    "32b9db1fb8698d57a2cf84c19576cb92eb3ef900be98eb6e6c8d8487616d9349"  # pragma: allowlist secret
)
V1_REMOTE_DIGEST = (
    "9aa6b3a3d362817e54bb72aa69dfc91d4cfe552b57f0dd1429d4469b9b8b470d"  # pragma: allowlist secret
)
V2_PILOT_DIGEST = (
    "e19b34514b019f019b99863351e54cd5ecf1aaa801eba864677da310ff170999"  # pragma: allowlist secret
)


def test_v1_and_v2_keep_their_bytes_and_read_as_the_same_class_never_equal_to_a_v3(
    pair: Pair,
) -> None:
    from tests.test_h_probe_launch import remote_request
    from tests.test_sentence_worker_request import request as smoke_request

    smoke, remote, pilot = smoke_request(), remote_request(), pilot_request()
    assert (smoke.digest, remote.digest, pilot.digest) == (
        V1_SMOKE_DIGEST,
        V1_REMOTE_DIGEST,
        V2_PILOT_DIGEST,
    )
    assert [r.manifest["schema"] for r in (smoke, remote, pilot)] == [
        REQUEST_SCHEMA_V1,
        REQUEST_SCHEMA_V1,
        REQUEST_SCHEMA_V2,
    ]
    for value in (smoke, remote, pilot):
        parsed = parse_probe_request(value.payload, max_bytes=1_000_000)
        assert type(parsed) is ProbeRequest and parsed == value
    v3 = pair_request(pair)
    assert v3.manifest["schema"] == REQUEST_SCHEMA_V3
    assert type(parse_probe_request(v3.payload, max_bytes=1_000_000)) is PairProbeRequest
    assert all(v3 != other and other != v3 for other in (smoke, remote, pilot))
    assert v3.digest not in (V1_SMOKE_DIGEST, V1_REMOTE_DIGEST, V2_PILOT_DIGEST)


def test_the_pair_digest_binds_both_cells_the_pauses_the_block_and_the_nonce(pair: Pair) -> None:
    base = pair_request(pair)
    renamed = replace(pair.recipient, name="h1-pair-ms-other-tiny")
    variants = [
        pair_request(pair, nonce="2" * 64),
        pair_request(pair, binding=replace(base.binding, block_id="c" * 64)),
        pair_request(
            pair,
            cells=(
                PairCell(pair.donor, None, None),
                PairCell(pair.recipient, None, ("prediction", 2)),
            ),
        ),
        pair_request(
            pair,
            cells=(
                PairCell(pair.donor, None, ("codec_warmup", 2)),
                PairCell(pair.recipient, None, None),
            ),
        ),
        pair_request(
            pair,
            cells=(PairCell(pair.donor, None, None), PairCell(renamed, None, None)),
            binding=replace(base.binding, pair_id=pair_identity(pair.donor, renamed)),
        ),
    ]
    assert len({base.digest, *(v.digest for v in variants)}) == len(variants) + 1
    assert base.prefix == f"h-probes/v1/{base.nonce}"
    assert base.manifest["scientific_authority"] == base.manifest["spending_authority"] == "none"
    assert "authority" not in base.manifest


def pilot_pair(**authority_changes: object) -> PairProbeRequest:
    """A scientific pair: two pilot cells and ONE sealed authority (remote proof from the fixture)."""
    template = pilot_request(**authority_changes)
    donor = template.profile
    recipient = replace(
        donor,
        name="h1-sphere-pilot",
        spine=replace(donor.spine, carrier="sphere"),
    )
    donor = replace(donor, name="h1-magnitude-pilot")
    return PairProbeRequest(
        cells=(PairCell(donor, None, None), PairCell(recipient, None, None)),
        inputs=template.inputs,
        nonce=template.nonce,
        limits=template.limits,
        probe=template.probe,
        code_commit=template.code_commit,
        code_bundle=template.code_bundle,
        image=template.image,
        binding=BlockBinding(
            BLOCK, pair_identity(donor, recipient), donor.source_digest, donor.seed
        ),
        authority=template.authority,
    )


def test_one_sealed_authority_covers_the_pair_and_binds_the_digest() -> None:
    value = pilot_pair()
    assert value.authority is not None
    assert value.manifest["scientific_authority"] == value.authority.digest
    assert value.manifest["spending_authority"] == value.authority.digest
    assert value.manifest["authority"] == value.authority.manifest
    assert parse_probe_request(value.payload, max_bytes=1_000_000) == value
    other = pilot_pair(spend_cap=replace(sealed_authority().spend_cap, usd=41.0))
    assert other.digest != value.digest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("authority_tamper", "run authority digest does not match its content"),
        ("authority_other_stage", "run authority names another stage"),
        ("mandate_none", "unknown worker request scope or authority"),
        ("mandate_other", "unknown worker request scope or authority"),
        ("no_authority", "scientific run requires a sealed run authority"),
        ("smoke_with_authority", "smoke run classes carry no run authority"),
        ("one_cell", "pair request requires exactly two cells"),
        ("swapped_cells", "pair cells are donor first"),
        ("pair_id", "pair identifier is not the identity of its two cells"),
        ("block_id_format", "block and pair identifiers are 64 lowercase hex digests"),
        ("binding_seed", "pair block binding differs from its cells' source or seed"),
        ("resume", "pair resume is not admitted before per-cell resume lands"),
    ],
)
def test_the_pair_reader_refuses_every_mandate_and_binding_out_of_family(
    pair: Pair, mutation: str, message: str
) -> None:
    sealed = sealed_authority()
    row = pilot_pair().manifest
    if mutation == "authority_tamper":
        authority = dict(sealed.manifest)
        authority["gpu_class"] = "NVIDIA H100 80GB HBM3"
        row["authority"] = authority
    elif mutation == "authority_other_stage":
        other = sealed_authority(stage="H2")
        row["authority"] = other.manifest
        row["scientific_authority"] = row["spending_authority"] = other.digest
    elif mutation == "mandate_none":
        row["spending_authority"] = "none"
    elif mutation == "mandate_other":
        row["scientific_authority"] = sealed_authority(stage="H2").digest
    elif mutation == "no_authority":
        del row["authority"]
        row["scientific_authority"] = row["spending_authority"] = "none"
    elif mutation == "smoke_with_authority":
        row = pair_request(pair).manifest
        row["authority"] = sealed.manifest
        row["scientific_authority"] = row["spending_authority"] = sealed.digest
    else:
        row = pair_request(pair).manifest
        cells = row["cells"]
        assert type(cells) is list
        if mutation == "one_cell":
            row["cells"] = cells[:1]
        elif mutation == "swapped_cells":
            row["cells"] = [cells[1], cells[0]]
        elif mutation == "pair_id":
            block = dict(row["block"])  # type: ignore[arg-type]
            block["pair_id"] = "d" * 64
            row["block"] = block
        elif mutation == "block_id_format":
            block = dict(row["block"])  # type: ignore[arg-type]
            block["block_id"] = "BLOCK"
            row["block"] = block
        elif mutation == "binding_seed":
            block = dict(row["block"])  # type: ignore[arg-type]
            block["seed"] = pair.donor.seed + 1
            row["block"] = block
        else:
            cells[1] = dict(cells[1]) | {
                "resume": {"path": "run/recipient/x.json", "sha256": "e" * 64}
            }
    with pytest.raises(ValueError, match=message):
        parse_probe_request(reencode(row), max_bytes=1_000_000)


def test_the_pair_wire_has_exactly_its_own_key_set_and_canonical_bytes(pair: Pair) -> None:
    value = pair_request(pair)
    row = value.manifest
    extra = dict(row) | {"resume": None}
    with pytest.raises(ValueError, match="fields differ"):
        parse_probe_request(reencode(extra), max_bytes=1_000_000)
    with pytest.raises(ValueError, match="canonical"):
        parse_probe_request(json.dumps(row).encode(), max_bytes=1_000_000)
    duplicate = b'{"nonce":"' + value.nonce.encode() + b'",' + value.payload[1:]
    with pytest.raises(ValueError, match="duplicate"):
        parse_probe_request(duplicate, max_bytes=1_000_000)


# --------------------------------------------------------- the cell token and the ID grammar
def test_the_cell_token_is_one_function_of_the_profile_never_its_identity(pair: Pair) -> None:
    assert (cell_token(pair.donor), cell_token(pair.recipient)) == ("donor", "recipient")
    assert pair.donor.identity not in (cell_token(pair.donor), cell_token(pair.recipient))
    request = pair_request(pair)
    assert request.tokens == ("donor", "recipient")
    assert max(len(f"checkpoint-{t}-00000000") for t in request.tokens) <= 64
    assert request.run_identity == request.binding.pair_id
    assert request.run_identity not in (pair.donor.identity, pair.recipient.identity)
    assert pair_identity(pair.donor, pair.recipient) != pair_identity(pair.recipient, pair.donor)


def test_the_asymmetric_donor_first_predicate_equals_the_blocks_codec_sharing_groups() -> None:
    """Parity pin (P5-03): the member restates the predicate and never imports the block; here
    the real H1 block's sharing groups are checked to pair a magnitude donor with a sphere
    recipient in BOTH groups."""
    from tests.test_h_preservation import profiles

    from entropic_substrate.ladder.register.h_preservation import H1PreservationBlock

    value = H1PreservationBlock(profiles())
    by_name = {profile.name: profile for profile in value.profiles}
    for donor_name, recipient_name in value.codec_sharing_groups:
        donor, recipient = by_name[donor_name], by_name[recipient_name]
        assert (donor.spine.carrier, recipient.spine.carrier) == ("magnitude", "sphere")
        assert donor.law.mode == recipient.law.mode
        assert (cell_token(donor), cell_token(recipient)) == ("donor", "recipient")


# ------------------------------------------------------------------- progress v2: the receipt
SOURCE_LIMIT = 1_000_000
LIMIT = 10_000_000
PAIR_ID_FIXTURE = "a" * 64


def trained_cell(pair: Pair, subject: Any, root: Path, scope: Any = None, **options: Any) -> Any:
    from entropic_substrate.ladder.run.runner import run_probe_training

    return run_probe_training(
        subject,
        pair.source,
        root,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
        scope=scope,
        **options,
    )


def read_receipt(
    pair: Pair, subject: Any, root: Path, artifact: ArtifactIdentity, **kw: Any
) -> Any:
    from entropic_substrate.ladder.run.runner import read_training_progress

    return read_training_progress(
        root,
        artifact,
        subject,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
        **kw,
    )


def forged(root: Path, receipt: ArtifactIdentity, name: str, **changes: Any) -> ArtifactIdentity:
    from entropic_substrate.ladder.run.journal import read_json, write_json

    row = read_json(root, receipt, max_bytes=LIMIT) | changes
    return write_json(root, name, row, max_bytes=LIMIT)


def test_a_single_smoke_cell_writes_progress_v2_with_a_null_binding(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.journal import read_json

    root = tmp_path / "donor"
    root.mkdir()
    done = trained_cell(pair, pair.donor, root)
    row = read_json(root, done.receipt, max_bytes=LIMIT)
    assert row["schema"] == "h-probe-training-progress-v2"
    assert [row[k] for k in ("block", "pair", "cell", "authority_digest", "handoff")] == [None] * 5
    assert (row["inherited_warmup_steps"], row["own_optimizer_steps"]) == (0, 6)
    assert row["successful_optimizer_steps"] == 6
    assert row["own_optimizer_steps"] + row["inherited_warmup_steps"] == 6
    assert read_receipt(pair, pair.donor, root, done.receipt)["status"] == "training_complete"


def test_a_pair_cell_receipt_binds_block_pair_and_cell_and_a_binding_mismatch_is_refused(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.journal import read_json
    from entropic_substrate.ladder.run.runner import CellScope, ReceiptBinding

    root = tmp_path / "donor"
    root.mkdir()
    scope = CellScope(BLOCK, PAIR_ID_FIXTURE, "donor")
    done = trained_cell(pair, pair.donor, root, scope)
    row = read_json(root, done.receipt, max_bytes=LIMIT)
    assert (row["block"], row["pair"], row["cell"]) == (BLOCK, PAIR_ID_FIXTURE, "donor")
    assert row["authority_digest"] is None  # a smoke run class: null exactly for smoke
    expected = ReceiptBinding(BLOCK, PAIR_ID_FIXTURE, "donor", None)
    assert read_receipt(pair, pair.donor, root, done.receipt, expected=expected)
    for other in (
        ReceiptBinding(),
        ReceiptBinding(BLOCK, PAIR_ID_FIXTURE, "recipient", None),
        ReceiptBinding("c" * 64, PAIR_ID_FIXTURE, "donor", None),
        ReceiptBinding(BLOCK, PAIR_ID_FIXTURE, "donor", "f" * 64),
    ):
        with pytest.raises(ValueError, match="binding differs from the expected binding"):
            read_receipt(pair, pair.donor, root, done.receipt, expected=other)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"authority_digest": "f" * 64}, "authority digest disagrees with its run class"),
        ({"block": BLOCK}, "block, pair and cell together or none"),
        ({"cell": "Bad Cell", "block": BLOCK, "pair": PAIR_ID_FIXTURE}, "safe cell"),
        ({"inherited_warmup_steps": 2, "own_optimizer_steps": 4}, "inherited and own steps differ"),
        ({"own_optimizer_steps": 5}, "inherited and own steps differ"),
        ({"handoff": {"path": "x.json", "sha256": "a" * 64}}, "cites no hand-off record"),
    ],
)
def test_a_forged_receipt_binding_is_refused_by_the_reader(
    tmp_path: Path, pair: Pair, changes: dict[str, Any], message: str
) -> None:
    root = tmp_path / "donor"
    root.mkdir()
    done = trained_cell(pair, pair.donor, root)
    bad = forged(root, done.receipt, "forged-progress.json", **changes)
    with pytest.raises(ValueError, match=message):
        read_receipt(pair, pair.donor, root, bad)


def test_a_v1_receipt_cannot_pass_the_v2_reader_or_the_capture_closure(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.journal import read_json, write_json
    from entropic_substrate.ladder.run.trace_capture import (
        _collect,  # pyright: ignore[reportPrivateUsage]
    )

    root = tmp_path / "donor"
    root.mkdir()
    done = trained_cell(pair, pair.donor, root)
    row = read_json(root, done.receipt, max_bytes=LIMIT)
    v1 = {k: v for k, v in row.items() if k not in _V2_ONLY} | {
        "schema": "h-probe-training-progress-v1"
    }
    old = write_json(root, "old-progress.json", v1, max_bytes=LIMIT)
    with pytest.raises(
        ValueError,
        match=r"h-probe-training-progress-v1.*minimum supported version is h-probe-training-progress-v2",
    ):
        read_receipt(pair, pair.donor, root, old)
    with pytest.raises(
        ValueError, match=r"minimum supported version is h-probe-training-progress-v2"
    ):
        _collect(root, (old,), pair.donor, max_bytes=LIMIT, max_artifacts=100)


_V2_ONLY = (
    "own_optimizer_steps",
    "inherited_warmup_steps",
    "block",
    "pair",
    "cell",
    "authority_digest",
    "handoff",
)


def test_the_handoff_receipt_is_own_zero_inherited_w_and_the_reader_derives_it_from_the_chain(
    tmp_path: Path, pair: Pair
) -> None:
    from tests.test_h_codec_handoff import materialize

    from entropic_substrate.ladder.run.journal import read_json
    from entropic_substrate.ladder.run.runner import CellScope, ReceiptBinding

    donor_root, recipient_root = tmp_path / "donor", tmp_path / "recipient"
    donor_root.mkdir()
    recipient_root.mkdir()
    trained_cell(pair, pair.donor, donor_root)
    handed = materialize(pair, donor_root, recipient_root)
    row = read_json(recipient_root, handed.receipt, max_bytes=LIMIT)
    assert (row["own_optimizer_steps"], row["inherited_warmup_steps"]) == (0, 2)
    assert row["successful_optimizer_steps"] == 2
    assert row["handoff"] == {
        "path": handed.record.path,
        "sha256": handed.record.content_hash.digest,
    }
    assert (
        read_receipt(pair, pair.recipient, recipient_root, handed.receipt)["status"] == "checkpoint"
    )
    # the reader derives inherited steps from the verified chain ROOT, never from the claim
    lying = forged(
        recipient_root,
        handed.receipt,
        "lying-progress.json",
        inherited_warmup_steps=0,
        own_optimizer_steps=2,
    )
    with pytest.raises(ValueError, match="inherited and own steps differ"):
        read_receipt(pair, pair.recipient, recipient_root, lying)
    wrong_record = forged(
        recipient_root,
        handed.receipt,
        "wrong-record-progress.json",
        handoff={"path": handed.record.path, "sha256": "9" * 64},
    )
    with pytest.raises(ValueError, match="hand-off record is not its chain root's record"):
        read_receipt(pair, pair.recipient, recipient_root, wrong_record)
    unbound = forged(recipient_root, handed.receipt, "unbound-progress.json", handoff=None)
    with pytest.raises(ValueError):
        read_receipt(pair, pair.recipient, recipient_root, unbound)
    # the recipient continues from it under the SAME binding, citing the SAME record
    resumed = trained_cell(
        pair,
        pair.recipient,
        recipient_root,
        None,
        resume=handed.receipt,
    )
    final = read_json(recipient_root, resumed.receipt, max_bytes=LIMIT)
    assert final["handoff"] == row["handoff"]
    assert (final["own_optimizer_steps"], final["inherited_warmup_steps"]) == (4, 2)
    assert final["successful_optimizer_steps"] == 6
    _ = (CellScope, ReceiptBinding)


def test_a_resume_must_carry_the_identical_binding_on_every_hop(tmp_path: Path, pair: Pair) -> None:
    from entropic_substrate.ladder.run.runner import CellScope

    root = tmp_path / "donor"
    root.mkdir()
    scope = CellScope(BLOCK, PAIR_ID_FIXTURE, "donor")
    paused = trained_cell(
        pair,
        pair.donor,
        root,
        scope,
        stop_after_seal=lambda phase, step: (phase, step) == ("prediction", 2),
    )
    assert paused.status == "paused"
    other = CellScope(BLOCK, PAIR_ID_FIXTURE, "recipient")
    with pytest.raises(ValueError, match="binding differs from the expected binding"):
        trained_cell(pair, pair.donor, root, other, resume=paused.receipt)
    with pytest.raises(ValueError, match="binding differs from the expected binding"):
        trained_cell(pair, pair.donor, root, None, resume=paused.receipt)
    done = trained_cell(pair, pair.donor, root, scope, resume=paused.receipt)
    assert done.status == "training_complete"
    # a child receipt whose parent hop carries another binding is refused
    child = forged(root, done.receipt, "child-progress.json", pair="d" * 64)
    with pytest.raises(ValueError, match="binding"):
        read_receipt(pair, pair.donor, root, child)


# ======================================================================== the failure rows
# Faults are injected only at the numerical primitive BELOW the runner (the codec
# reconstruction of a warm-up step, the prediction objective), or as a real corrupt seal on
# disk; never in the pair loop (P5-08).
class Fault:
    """Raise ``error`` on the n-th call of a numerical primitive (counted from the run's start)."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, where: str, nth: int, error: Exception
    ) -> None:
        self.calls = 0
        self.nth, self.error = nth, error
        if where == "warmup":
            from entropic_substrate.core.sentence.codec import NeuralSentenceCodec

            real = NeuralSentenceCodec.reconstruct

            def reconstruct(codec: Any, *args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == self.nth:
                    raise self.error
                return real(codec, *args, **kwargs)

            monkeypatch.setattr(NeuralSentenceCodec, "reconstruct", reconstruct)
        else:
            from entropic_substrate.ladder.run import execution

            real_loss = execution.prediction_loss

            def prediction_loss(*args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                if self.calls == self.nth:
                    raise self.error
                return real_loss(*args, **kwargs)

            monkeypatch.setattr(execution, "prediction_loss", prediction_loss)


def pair_world(tmp_path: Path, pair: Pair) -> tuple[PairProbeRequest, _ObjectStore, Any]:
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request = pair_request(pair)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, tmp_path)
    return request, objects, marker


def lifecycle_of(observability: Path, token: str) -> list[dict[str, Any]]:
    path = observability / token / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r.get("schema") == "h-run-lifecycle-v1"]


def terminal_names(objects: _ObjectStore, request: PairProbeRequest) -> set[str]:
    prefix = f"{request.prefix}/"
    return {
        key[len(prefix) :].removesuffix(".json")
        for key in objects.values
        if key.startswith(prefix) and "/" not in key[len(prefix) :]
    } & {"complete", "paused", "stopped", "failure", "excluded"}


def test_a_donor_with_a_nonfinite_endpoint_still_hands_off_and_the_recipient_trains(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    request, objects, marker = pair_world(tmp_path, pair)
    # the donor's FIRST prediction step diverges, after its warm-up-final seal was admitted
    Fault(monkeypatch, "prediction", 1, UndefinedDirection(2))
    status, error, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error  # a training-phase scientific outcome returns normally
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["status"] == "pair_excluded"
    assert result["verdict"] == {
        "verdict": "excluded",
        "reason": "nonfinite_endpoint",
        "cell": "donor",
    }
    donor, recipient = result["cells"]
    assert (donor["outcome"], recipient["outcome"]) == ("scientific_gap", "complete")
    assert donor["failure"] is not None and recipient["capture"] is not None
    assert terminal_names(objects, request) == {"excluded"}
    donor_end = lifecycle_of(observability, "donor")[-1]
    assert (donor_end["status"], donor_end["result_status"]) == ("failed", "scientific_gap")
    assert lifecycle_of(observability, "recipient")[-1]["status"] == "ok"


@pytest.mark.parametrize("failing_step", [1, 2])
def test_a_donor_whose_warmup_diverged_before_its_final_seal_stops_the_pair_without_a_handoff(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch, failing_step: int
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.analyze.h1_observations import project_h1_cell
    from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
    from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", failing_step, UndefinedDirection(2))
    status, error, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    recorded = H1ExclusionReason(result["verdict"]["reason"])
    assert type(recorded) is H1ExclusionReason and recorded == H1ExclusionReason.MISSING_ENDPOINT
    assert result["verdict"]["cell"] == "donor" and result["status"] == "pair_excluded"
    assert [c["outcome"] for c in result["cells"]] == ["scientific_gap", "not_run"]
    assert not (workspace / "run" / "recipient").exists()  # never materialized, never spent
    assert not (observability / "recipient").exists()  # no sink, no MLflow run, no log
    assert terminal_names(objects, request) == {"excluded"}
    # DH-42's pin: the worker's recorded member equals the member L3's adapter derives from the
    # donor's OWN files, on the default path with no override
    cell = project_h1_cell(
        load_h_run(
            observability / "donor" / "metrics.jsonl", observability / "donor" / "worker-log.jsonl"
        )
    )
    assert type(cell.endpoint_reason) is H1ExclusionReason
    assert cell.endpoint_reason == recorded


def failure_terminal(objects: _ObjectStore, request: PairProbeRequest) -> dict[str, Any]:
    return json.loads(objects.values[f"{request.prefix}/failure.json"])


def failure_snapshot_files(objects: _ObjectStore, request: PairProbeRequest) -> dict[str, bytes]:
    """path -> bytes of every file the failure snapshot carries, read back through the port."""
    marker = json.loads(objects.values[f"{request.prefix}/snapshots/failure.json"])
    return {item["path"]: objects.values[item["key"]] for item in marker["files"]}


# The schema ids a pair's failure snapshot may hold: the parent tip's known set, plus the two of
# the slice's five new names that ride a snapshot (progress v2 and result v2). A sixth name, a
# pair-failure artifact, would be a new door nobody ruled (DH-47).
PARENT_TIP_SCHEMAS = frozenset(
    {
        "h-run-profile-v3",
        "h-checkpoint-seal-v1",
        "h-codec-handoff-seal-v1",
        "h-codec-handoff-v1",
        "h-journal-shard-v1",
        "h-journal-shard-v2",
        "h-predictive-evaluation-v1",
        "h-generation-receipt-v3",
        "h-probe-training-failure-v2",
        "h-generation-failure-v1",
        "h-span-diagnostics-v1",
        "h-span-diagnostics-failure-v1",
        "h-worker-start-v1",
        "h-worker-failure-v1",
        "h-model-trace-capture-v2",
        "h-native-verification-v1",
        "h-probe-worker-result-v1",
    }
)
NEW_SCHEMAS = frozenset({"h-probe-training-progress-v2", "h-probe-worker-result-v2"})


def schemas_in(files: dict[str, bytes]) -> set[str]:
    found: set[str] = set()
    for path, payload in files.items():
        if path.endswith(".json"):
            row = json.loads(payload)
            if isinstance(row, dict) and "schema" in row:
                found.add(row["schema"])
    return found


def test_an_execution_class_training_stop_of_the_donor_carries_result_v2_in_the_failure_snapshot(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, RuntimeError("injected reader refusal"))
    status, _, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 2  # the ORIGINAL exception propagated, not a wrapper
    failure = failure_terminal(objects, request)
    assert failure["status"] == "execution_defect" and failure["error_type"] == "RuntimeError"
    assert failure["run_identity"] == pair.donor.identity  # the failing cell's, not the pair's
    files = failure_snapshot_files(objects, request)
    record = json.loads(files["worker/failure.json"])
    assert record["schema"] == "h-worker-failure-v1" and record["status"] == "execution_defect"
    assert record["error_type"] == "RuntimeError" and "pair" not in record  # v1 key set intact
    result = json.loads(files["worker/result.json"])
    assert result["schema"] == "h-probe-worker-result-v2"
    assert result["status"] == "execution_defect"
    assert result["verdict"] == {"verdict": "excluded", "reason": "reader_refusal", "cell": "donor"}
    assert [c["outcome"] for c in result["cells"]] == ["execution_defect", "not_run"]
    # the failure snapshot MATERIALIZES on a host (the pair's cell paths pass the namespace
    # predicate on the receiving side too) and carries result v2 beside the v1 failure record
    from entropic_substrate.ladder.run.worker_snapshot import materialize_worker_snapshot

    host = tmp_path / "host"
    host.mkdir()
    snapshot_key = f"{request.prefix}/snapshots/failure.json"
    materialize_worker_snapshot(
        objects,
        request,
        objects._identity(snapshot_key, objects.values[snapshot_key]),  # pyright: ignore[reportPrivateUsage]
        host,
        max_marker_bytes=1_000_000,
    )
    assert (host / "worker" / "result.json").is_file() and (
        host / "worker" / "failure.json"
    ).is_file()
    # no sixth schema name: every schema in the snapshot is known or one of the new names
    found = schemas_in(files)
    assert found - PARENT_TIP_SCHEMAS <= NEW_SCHEMAS, found - PARENT_TIP_SCHEMAS - NEW_SCHEMAS
    assert "h-probe-worker-result-v2" in found
    assert terminal_names(objects, request) == {"failure"}
    assert not (workspace / "run" / "recipient").exists()
    end = lifecycle_of(observability, "donor")[-1]
    assert (end["status"], end["result_status"]) == ("failed", "execution_defect")


class RemovingSinks:
    """A per-cell sink provider that deletes the donor's warm-up-final checkpoint ON DISK after
    the donor's run ended: a real missing seal between the donor's workload and the hand-off.

    (Corrupting the bytes instead would make the failure snapshot collide with the already
    published copy of the same key, which is a different, true refusal.)"""

    def __init__(self, observability: Path, workspace: Path) -> None:
        from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink

        self._sink = JsonlSink
        self.observability, self.workspace = observability, workspace
        self.begun: list[str] = []
        self.finished: list[tuple[str, str]] = []

    def begin(self, cell: str, profile: Any) -> Any:
        self.begun.append(cell)
        return self._sink(self.observability / cell / "metrics.jsonl")

    def finish(self, cell: str, status: str) -> None:
        self.finished.append((cell, status))
        if cell == "donor":
            (target,) = (self.workspace / "run" / "donor").glob("*-codec_warmup-00000002.pt")
            target.unlink()


def test_a_handoff_failure_is_the_recipients_reader_refusal_and_leaves_no_recipient_run(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.object_worker import run_object_worker

    request, objects, marker = pair_world(tmp_path, pair)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sinks = RemovingSinks(tmp_path / "obs", workspace)
    with pytest.raises(ValueError):
        run_object_worker(objects, marker, workspace, max_request_bytes=1_000_000, cell_sinks=sinks)
    assert sinks.begun == ["donor"]  # the recipient has no sink, no MLflow run, no metrics, no log
    assert sinks.finished == [("donor", "FINISHED")]
    assert not (tmp_path / "obs" / "recipient").exists()
    failure = failure_terminal(objects, request)
    assert failure["status"] == "execution_defect"
    assert (
        failure["run_identity"] == pair.recipient.identity
    )  # a hand-off failure is the recipient's
    result = json.loads(failure_snapshot_files(objects, request)["worker/result.json"])
    assert result["verdict"] == {
        "verdict": "excluded",
        "reason": "reader_refusal",
        "cell": "recipient",
    }
    assert [c["outcome"] for c in result["cells"]] == ["complete", "execution_defect"]
    assert terminal_names(objects, request) == {"failure"}


def test_a_recipient_generation_gap_publishes_the_v1_failure_marker_and_no_result_v2(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run import worker_execution

    request, objects, marker = pair_world(tmp_path, pair)
    real = worker_execution.run_probe_generation
    calls: list[str] = []

    def generation(*args: Any, **kwargs: Any) -> Any:
        calls.append("generation")
        if len(calls) == 2:  # the recipient's: the donor's workload already completed
            raise UndefinedDirection(3)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_execution, "run_probe_generation", generation)
    status, _, _, observability = run_command(tmp_path, objects, marker)
    assert status == 2
    failure = failure_terminal(objects, request)
    assert failure["status"] == "scientific_gap" and failure["error_type"] == "UndefinedDirection"
    assert failure["run_identity"] == pair.recipient.identity
    files = failure_snapshot_files(objects, request)
    assert "worker/result.json" not in files  # no result v2, no relabelling, no exclusion member
    assert schemas_in(files) - PARENT_TIP_SCHEMAS <= {"h-probe-training-progress-v2"}
    assert terminal_names(objects, request) == {"failure"}
    assert lifecycle_of(observability, "donor")[-1]["status"] == "ok"
    end = lifecycle_of(observability, "recipient")[-1]
    assert (end["status"], end["result_status"]) == ("failed", "scientific_gap")


def test_a_donor_span_diagnostics_gap_stops_the_pair_before_the_recipient_is_materialized(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import worker_execution
    from entropic_substrate.ladder.run.span_states import SpanPopulationGap

    request, objects, marker = pair_world(tmp_path, pair)

    def diagnostics(*args: Any, **kwargs: Any) -> Any:
        raise SpanPopulationGap({2: 1}, {2: 0})

    monkeypatch.setattr(worker_execution, "run_span_diagnostics", diagnostics)
    status, _, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 2
    failure = failure_terminal(objects, request)
    assert failure["status"] == "diagnostic_gap" and failure["error_type"] == "SpanPopulationGap"
    assert failure["run_identity"] == pair.donor.identity
    assert not (workspace / "run" / "recipient").exists()
    assert not (observability / "recipient").exists()
    assert "worker/result.json" not in failure_snapshot_files(objects, request)
    end = lifecycle_of(observability, "donor")[-1]
    assert (end["status"], end["result_status"]) == ("failed", "diagnostic_gap")


# ===================================================================== the stop marker (v2)
def marker_store(request: PairProbeRequest, payload: bytes) -> Any:
    from tests.test_h_probe_operator_stop import _MarkerStore

    return _MarkerStore(
        control_key=f"{request.prefix}/control/stop.json", control_payloads=(payload,)
    )


def test_a_v2_marker_naming_the_recipient_at_a_prediction_seal_stops_the_recipient_only(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.object_worker import (
        STOP_MARKER_SCHEMA_V2,
        encode_stop_marker,
    )
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request = pair_request(pair)
    payload = encode_stop_marker(request, ("prediction", 2, "recipient"))
    row = json.loads(payload)
    assert row["schema"] == STOP_MARKER_SCHEMA_V2 == "h-probe-stop-marker-v2"
    assert row["target"] == {"phase": "prediction", "step": 2, "cell": "recipient"}
    objects = marker_store(request, payload)
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["status"] == "stopped_by_operator"
    donor, recipient = result["cells"]
    # the donor reached the same coordinate (prediction, 2) and ran to completion: the marker
    # names the recipient, so it stopped the recipient only
    assert (donor["outcome"], recipient["outcome"]) == ("complete", "stopped_by_operator")
    assert recipient["successful_optimizer_steps"] == 4 and recipient["capture"] is None
    assert result["verdict"] == {"verdict": "none", "reason": None, "cell": None}
    assert terminal_names(objects, request) == {"stopped"}
    assert lifecycle_of(observability, "recipient")[-1]["result_status"] == "stopped_by_operator"


def test_a_v2_marker_naming_the_donor_stops_the_pair_with_no_handoff(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.object_worker import encode_stop_marker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request = pair_request(pair)
    objects = marker_store(request, encode_stop_marker(request, ("prediction", 2, "donor")))
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["status"] == "stopped_by_operator"
    assert [c["outcome"] for c in result["cells"]] == ["stopped_by_operator", "not_run"]
    assert not (workspace / "run" / "recipient").exists()
    assert not (observability / "recipient").exists()


def test_a_declared_pause_on_the_recipient_pauses_the_pair_after_the_donor_completed(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request = pair_request(
        pair,
        cells=(PairCell(pair.donor, None, None), PairCell(pair.recipient, None, ("prediction", 2))),
    )
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["status"] == "paused" and result["verdict"]["verdict"] == "none"
    assert [c["outcome"] for c in result["cells"]] == ["complete", "paused"]
    assert result["cells"][1]["successful_optimizer_steps"] == 4
    assert terminal_names(objects, request) == {"paused"}


def test_a_v1_marker_on_a_pair_and_a_v2_marker_on_a_single_profile_are_refused_by_name(
    tmp_path: Path, pair: Pair
) -> None:
    from tests.test_sentence_worker_request import request as single_request

    from entropic_substrate.ladder.run.object_worker import (
        classify_stop_marker,
        encode_stop_marker,
        stop_marker_hits,
    )

    pair_value, single = pair_request(pair), single_request()
    v1 = encode_stop_marker(single, ("prediction", 2))
    v2 = encode_stop_marker(pair_value, ("prediction", 2, "recipient"))
    with pytest.raises(ValueError, match=r"minimum supported version is h-probe-stop-marker-v2"):
        classify_stop_marker(v1, pair_value)
    with pytest.raises(ValueError, match=r"names a cell.*admits h-probe-stop-marker-v1"):
        classify_stop_marker(v2, single)
    # a v1 marker of the very same request digest cannot be forged onto the pair either
    forged = json.dumps(
        {"schema": "h-probe-stop-marker-v1", "request_sha256": pair_value.digest},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(ValueError, match="minimum supported version is h-probe-stop-marker-v2"):
        stop_marker_hits(forged, pair_value, ("prediction", 2), "donor")
    # and a v1 marker published for a pair stops the worker as a refusal, not as a stop
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    objects = marker_store(pair_value, forged)
    marker = publish_probe_request(objects, pair_value, tmp_path)
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2
    record = json.loads(failure_snapshot_files(objects, pair_value)["worker/failure.json"])
    assert record["status"] == "execution_defect"
    assert "minimum supported version is h-probe-stop-marker-v2" in record["detail"]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        (None, "names its cell"),
        (("prediction", 2), "must be .non-empty phase, positive step, cell."),
        (("prediction", 2, "nobody"), "must be .non-empty phase, positive step, cell."),
        (("prediction", 3, "donor"), "outside the request schedule"),
        (("codec_warmup", 3, "recipient"), "outside the request schedule"),
        (("prediction", 0, "donor"), "must be .non-empty phase, positive step"),
    ],
)
def test_a_pair_marker_names_its_cell_and_is_validated_against_that_cells_schedule(
    pair: Pair, target: Any, message: str
) -> None:
    from entropic_substrate.ladder.run.object_worker import encode_stop_marker

    with pytest.raises(ValueError, match=message):
        encode_stop_marker(pair_request(pair), target)


def test_the_stop_verb_refuses_a_pair_without_a_cell_and_a_cell_for_a_single_request_before_publishing(
    tmp_path: Path, pair: Pair
) -> None:
    from tests.test_sentence_worker_request import request as single_request

    from entropic_substrate.ladder.run.worker.h_probe_stop import request_stop

    objects = _ObjectStore()
    ticks = iter(range(1_000_000))  # the clock advances: a guard that is dropped must time out
    kwargs: dict[str, Any] = {
        "wait_seconds": 1.0,
        "poll_seconds": 0.1,
        "monotonic": lambda: float(next(ticks)),
        "sleep": lambda _: None,
    }
    # RETRACTED 2026-11-02 (FOR-1694): this refused EVERY pair ("does not carry a pair request").
    # A pair is stopped now, but only at a named cell and seal; without one it stays refused,
    # by name, before anything is published.
    with pytest.raises(ValueError, match="a pair stop names its cell: pass --at-cell"):
        request_stop(objects, pair_request(pair), target=("prediction", 2), **kwargs)
    with pytest.raises(ValueError, match="a cell is given for a single-profile request"):
        request_stop(objects, single_request(), target=("prediction", 2), cell="donor", **kwargs)
    assert objects.values == {} and objects.publishes == []  # irreversibility last: nothing written


# ============================================================= result v2 and completion v2
def terminal_identity(objects: _ObjectStore, request: PairProbeRequest, name: str) -> Any:
    key = f"{request.prefix}/{name}.json"
    return objects._identity(key, objects.values[key])  # pyright: ignore[reportPrivateUsage]


def test_the_pair_result_is_v2_the_host_completion_is_v2_and_a_v1_result_is_refused_by_name(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.worker_completion import receive_worker_completion
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, objects, marker = pair_world(tmp_path, pair)
    status, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    host = tmp_path / "host"
    host.mkdir()
    completion = receive_worker_completion(
        objects,
        request,
        terminal_identity(objects, request, "complete"),
        host,
        expected_pod_id=None,
        max_marker_bytes=1_000_000,
    )
    assert completion["schema"] == "h-host-probe-completion-v2"
    assert completion["status"] == "probe_completion_verified"
    assert completion["verdict"] == {"verdict": "complete", "reason": None, "cell": None}
    assert completion["run_identity"] == request.binding.pair_id
    assert [c["cell"] for c in completion["cells"]] == ["donor", "recipient"]  # type: ignore[union-attr]
    assert all(
        c["outcome"] == "complete"  # type: ignore[index]
        and c["successful_optimizer_steps"] == 6  # type: ignore[index]
        and c["capture"]["path"].startswith(f"capture/{c['cell']}/")  # type: ignore[index]
        for c in completion["cells"]  # type: ignore[union-attr]
    )
    # Q11: the publication COUNTS ride the completion, and only the counts
    assert set(completion["publication"]) == {  # type: ignore[arg-type]
        "objects_put",
        "objects_skipped",
        "verification_reads",
        "retries",
    }
    # a pair result presented as v1 is refused by name, with the minimum supported version
    result = json.loads((workspace / "worker/result.json").read_text())
    as_v1 = result | {"schema": "h-probe-worker-result-v1"}
    with pytest.raises(
        ValueError,
        match=r"h-probe-worker-result-v1.*minimum supported version is h-probe-worker-result-v2",
    ):
        parse_pair_result(as_v1, request)
    assert parse_pair_result(result, request)["status"] == "probe_workload_complete"


def test_a_scientific_exclusion_is_received_under_its_own_terminal_with_the_verdict(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run.worker_completion import receive_worker_completion

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, UndefinedDirection(2))
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0
    host = tmp_path / "host"
    host.mkdir()
    completion = receive_worker_completion(
        objects,
        request,
        terminal_identity(objects, request, "excluded"),
        host,
        expected_pod_id=None,
        max_marker_bytes=1_000_000,
    )
    assert completion["verdict"] == {
        "verdict": "excluded",
        "reason": "missing_endpoint",
        "cell": "donor",
    }
    assert [c["outcome"] for c in completion["cells"]] == ["scientific_gap", "not_run"]  # type: ignore[union-attr]
    # the 'complete' terminal never exists for an excluded pair, and 'failure' is never admitted
    with pytest.raises(ValueError, match="marker identity mismatch"):
        receive_worker_completion(
            objects,
            request,
            objects._identity(f"{request.prefix}/failure.json", b"{}"),  # pyright: ignore[reportPrivateUsage]
            tmp_path / "other",
            expected_pod_id=None,
            max_marker_bytes=1_000_000,
        )


def test_a_single_profile_request_still_produces_v1_bytes_and_refuses_a_result_v2_by_name(
    tmp_path: Path,
) -> None:
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run.object_worker import (
        _admit_result,  # pyright: ignore[reportPrivateUsage]
        run_object_worker,
    )
    from entropic_substrate.ladder.run.worker_completion import receive_worker_completion
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    terminal = run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    result = json.loads((worker / "worker/result.json").read_text())
    assert result["schema"] == "h-probe-worker-result-v1"
    assert set(result) == {
        "schema", "request_sha256", "run_identity", "scientific_authority", "status", "runtime",
        "successful_optimizer_steps", "training", "generation", "diagnostics", "capture",
        "verification", "wall_seconds",
    }  # fmt: skip
    assert result["training"]["path"].count("/") == 1  # flat run/<name>: the v1 namespace
    host = tmp_path / "host"
    host.mkdir()
    completion = receive_worker_completion(
        objects, request, terminal, host, expected_pod_id=None, max_marker_bytes=1_000_000
    )
    assert completion["schema"] == "h-host-probe-completion-v1"
    with pytest.raises(
        ValueError, match=r"h-probe-worker-result-v2.*admits h-probe-worker-result-v1"
    ):
        _admit_result(worker, request, result | {"schema": "h-probe-worker-result-v2"})
    assert terminal.key.endswith("/complete.json")


# ================================================== per-cell MLflow runs, admission, refusals
def test_each_cell_gets_its_own_named_run_with_block_pair_cell_and_authority_tags(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
    from entropic_substrate.ladder.run import worker_command

    request, objects, marker = pair_world(tmp_path, pair)
    descriptors: list[Any] = []
    paths: list[tuple[Path, Path]] = []

    def build(descriptor: Any, *, results_path: Path, log_path: Path, **kwargs: Any) -> Any:
        descriptors.append(descriptor)
        paths.append((Path(results_path), Path(log_path)))
        return JsonlSink(results_path)

    monkeypatch.setattr(worker_command, "build_sink_for_run", build)
    status, error, _, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    assert [d.run_name for d in descriptors] == [
        f"h-worker-{pair.donor.identity}",
        f"h-worker-{pair.recipient.identity}",
    ]
    for descriptor, cell in zip(descriptors, ("donor", "recipient"), strict=True):
        assert descriptor.tiny is True  # tiny stays DERIVED from the (absent) authority
        assert descriptor.tags["schema"] == "h-probe-worker-result-v2"
        assert (descriptor.tags["block"], descriptor.tags["pair"], descriptor.tags["cell"]) == (
            BLOCK,
            request.binding.pair_id,
            cell,
        )
        assert descriptor.tags["request_sha256"] == request.digest
        assert "run_authority" not in descriptor.tags  # a smoke pair carries no authority
    # one metrics file and one worker log per cell, under the cell's own directory
    assert paths == [
        (observability / token / "metrics.jsonl", observability / token / "worker-log.jsonl")
        for token in ("donor", "recipient")
    ]
    # each per-cell log carries exactly one worker.start with the request digest and ONLY its
    # own cell's checkpoint.seal and evaluation.start events
    counts = {}
    for token in ("donor", "recipient"):
        log = [
            json.loads(line)
            for line in paths[("donor", "recipient").index(token)][1].read_text().splitlines()
        ]
        starts = [r for r in log if r["event"] == "worker.start"]
        assert len(starts) == 1 and starts[0]["request_sha256"] == request.digest
        # each cell's log closes with its OWN worker.end, after that cell's run finished
        ends = [r for r in log if r["event"] == "worker.end"]
        assert len(ends) == 1 and ends[0]["status"] == "ok"
        assert [r["event"] for r in log].index("worker.end") > [r["event"] for r in log].index(
            "sink.finish.end"
        )
        counts[token] = (
            [(r["phase"], r["step"]) for r in log if r["event"] == "checkpoint.seal"],
            sum(r["event"] == "evaluation.start" for r in log),
        )
    assert counts["donor"] == ([("codec_warmup", 2), ("prediction", 2), ("prediction", 4)], 2)
    assert counts["recipient"] == (
        [("prediction", 2), ("prediction", 4)],
        2,
    )  # no warm-up of its own


def test_the_scientific_pair_names_its_authority_as_param_and_tag_and_tiny_is_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authority half of D11 where it can be read without a pod: the per-cell provider builds
    each cell's descriptor from the request alone. A sealed authority makes ``tiny`` False and
    rides every cell run as the ``run_authority`` PARAM (as for a single cell) and TAG."""
    from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
    from entropic_substrate.ladder.run import worker_command

    value = pilot_pair()
    assert value.authority is not None
    seen: list[Any] = []

    def build(descriptor: Any, *, results_path: Path, **_: Any) -> Any:
        seen.append(descriptor)
        return JsonlSink(results_path)

    monkeypatch.setattr(worker_command, "build_sink_for_run", build)
    sinks = worker_command._PairCellSinks(value, tmp_path / "obs")  # pyright: ignore[reportPrivateUsage]
    sinks.begin("donor", value.donor)
    sinks.begin("recipient", value.recipient)
    assert [d.run_name for d in seen] == [
        f"h-worker-{value.donor.identity}",
        f"h-worker-{value.recipient.identity}",
    ]
    for descriptor, cell in zip(seen, value.tokens, strict=True):
        assert descriptor.tiny is False
        assert descriptor.params["run_authority"] == value.authority.digest
        assert descriptor.tags["run_authority"] == value.authority.digest
        assert descriptor.params["git_sha"] == value.code_commit
        assert descriptor.params["image"] == value.image
        assert (descriptor.tags["block"], descriptor.tags["cell"]) == (BLOCK, cell)


def test_a_scientific_pair_is_admitted_only_on_the_remote_surface_with_its_proof() -> None:
    """The pilot class is never admitted on the local surface: a pilot pair with no immutable
    image and source proof (what a local worker would carry) is refused."""
    value = pilot_pair()
    with pytest.raises(ValueError, match="remote worker requires immutable image and source proof"):
        replace(
            value,
            code_commit=None,
            code_bundle=None,
            image=None,
            inputs=tuple(i for i in value.inputs if i.path != "code.bundle"),
        )


def test_pair_outputs_are_admitted_only_under_a_declared_cell_token(pair: Pair) -> None:
    from tests.test_sentence_worker_request import request as single_request

    from entropic_substrate.ladder.run.worker_snapshot import (
        _path,  # pyright: ignore[reportPrivateUsage]
    )

    request, single = pair_request(pair), single_request()
    for name in (
        "run/donor/x.json",
        "run/recipient/x.pt",
        "capture/donor/capture.json",
        "verification/recipient/native.json",
        "capture/donor/source/prepared-corpus.json",
        "worker/result.json",
    ):
        assert _path(name, request) == name
    for name in (
        "run/x.json",  # the flat v1 form is outside a pair's namespace
        "run/stranger/x.json",  # an undeclared token
        "capture/stranger/source/prepared-corpus.json",
        "worker/donor/x.json",
        "run/donor/sub/x.json",
        "capture/source/prepared-corpus.json",
    ):
        with pytest.raises(ValueError, match="outside its owned namespace"):
            _path(name, request)
    # v1 and v2 keep the flat two-part namespace byte for byte
    assert _path("run/x.json", single) == "run/x.json"
    assert (
        _path("capture/source/prepared-corpus.json", single)
        == "capture/source/prepared-corpus.json"
    )
    with pytest.raises(ValueError, match="outside its owned namespace"):
        _path("run/donor/x.json", single)
    from entropic_substrate.ladder.run.worker_request import worker_artifact, worker_cell_artifact

    ref = {"path": "run/donor/x.json", "sha256": "a" * 64}
    assert worker_cell_artifact(ref, "run", "donor").path == "x.json"
    for bad in (("run", "recipient"), ("capture", "donor")):
        with pytest.raises(ValueError, match="another cell or section"):
            worker_cell_artifact(ref, *bad)
    with pytest.raises(ValueError, match="another section"):
        worker_artifact(ref, "run")  # the v1/v2 two-part rule and its message are unchanged


def test_a_pair_is_refused_by_name_at_the_publish_verb_and_by_the_launch_class_gate(
    tmp_path: Path, pair: Pair
) -> None:
    from tests.test_h_probe_launch import launch

    from entropic_substrate.ladder.run.h_probe_launch import parse_probe_launch
    from entropic_substrate.ladder.run.worker_command import run_worker_command

    request = pair_request(pair)
    (tmp_path / "pair-request.json").write_bytes(request.payload)
    output, error = StringIO(), StringIO()
    status = run_worker_command(
        [
            "publish",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "10000000",
            "--max-objects",
            "500",
            "--artifact-root",
            str(tmp_path),
            "--request-file",
            "pair-request.json",
        ],
        environ={},
        objects=_ObjectStore(),
        stdout=output,
        stderr=error,
    )
    assert status == 2 and output.getvalue() == ""
    assert json.loads(error.getvalue())["stage"] == "request_publish"
    # the launch (RETRACTED 2026-11-02, FOR-1694: a pair launch is carried now, so the old
    # "does not carry a pair request" refusal is gone): this local_tiny pair is still no launch,
    # because the class gate admits only remote tiny and scientific classes at the launch, and a
    # v5 payload that embeds a pair under a manifest that says it carries none is a mismatch
    with pytest.raises(ValueError, match="restricted to remote H0 tiny probes"):
        replace(launch(), request=request)
    row = json.loads(launch().payload)
    row["request"] = request.manifest
    with pytest.raises(ValueError, match="scope or authority mismatch"):
        parse_probe_launch(reencode(row), max_bytes=1_000_000)


def test_the_resume_builder_and_the_collector_refuse_v1_receipts_and_pair_results_by_name(
    tmp_path: Path,
) -> None:
    from tests.test_sentence_worker_request import request as single_request

    from entropic_substrate.ladder.run.h_probe_collection import (
        _single_result,  # pyright: ignore[reportPrivateUsage]
    )
    from entropic_substrate.ladder.run.h_probe_resume import build_resume_request
    from entropic_substrate.ladder.run.journal import write_json

    previous = single_request()
    run_root = tmp_path / "run"
    run_root.mkdir()
    body = {
        "schema": "h-probe-training-progress-v1",
        "run_identity": previous.profile.identity,
        "status": "paused",
    }
    receipt = write_json(run_root, "receipt.json", body, max_bytes=LIMIT)
    collection = {
        "schema": "h-probe-terminal-collection-v1",
        "request_sha256": previous.digest,
        "capture_status": "pause_bytes_verified",
        "resume_artifact": {"path": "run/receipt.json", "sha256": receipt.content_hash.digest},
    }
    with pytest.raises(
        ValueError, match="minimum supported version is h-probe-training-progress-v2"
    ):
        build_resume_request(previous, collection, run_root, nonce="2" * 64)
    with pytest.raises(ValueError, match=r"names a pair's cells.*admits h-probe-worker-result-v1"):
        _single_result({"schema": "h-probe-worker-result-v2"})


# ================================================================= the corpus memo (DH-41)
def corpus_world(tmp_path: Path) -> tuple[Path, Any]:
    """A prepared corpus on disk and its digest (the L4a pair fixture's own source)."""
    pair = make_pair(tmp_path)
    return pair.source, ContentHash("sha256", pair.donor.source_digest)


def test_the_memo_replays_a_corpus_once_per_digest_and_serves_the_proof_to_every_sampler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entropic_substrate.core.sentence.sampling as sampling
    from entropic_substrate.core.sentence.boundary import SentenceByteCodec
    from entropic_substrate.core.sentence.sampling import (
        SamplingSpec,
        SentenceSampler,
        VerifiedCorpus,
    )
    from entropic_substrate.ladder.run import seals

    source, digest = corpus_world(tmp_path)
    reads: list[Path] = []
    real = seals.read_prepared_corpus
    monkeypatch.setattr(
        seals,
        "read_prepared_corpus",
        lambda root, expected, **kw: reads.append(root) or real(root, expected, **kw),
    )
    memo = seals.CorpusMemo()
    first = memo.read(source, digest, max_bytes=SOURCE_LIMIT)
    second = memo.read(source, digest, max_bytes=SOURCE_LIMIT)
    assert type(first) is VerifiedCorpus and first is second
    assert (len(reads), memo.read_replays, memo.proof_replays, memo.hits) == (1, 1, 1, 1)
    # a sampler handed the proof replays nothing; one handed the bare corpus still does
    replays: list[int] = []
    real_prepare = sampling.prepare_documents
    monkeypatch.setattr(
        sampling, "prepare_documents", lambda *a, **k: replays.append(1) or real_prepare(*a, **k)
    )
    spec = SamplingSpec("train", 4, 0, 42)
    SentenceSampler(first, spec, SentenceByteCodec(), __import__("torch").device("cpu"))
    assert replays == []
    SentenceSampler(first.corpus, spec, SentenceByteCodec(), __import__("torch").device("cpu"))
    assert replays == [1]


def test_a_memo_hit_keeps_every_per_call_predicate_and_the_memo_serves_exactly_one_root(
    tmp_path: Path,
) -> None:
    from entropic_substrate.kernel.artifacts import D1ArtifactSnapshotError
    from entropic_substrate.ladder.run.seals import CorpusMemo

    source, digest = corpus_world(tmp_path)
    memo = CorpusMemo()
    memo.read(source, digest, max_bytes=SOURCE_LIMIT)
    with pytest.raises(ValueError, match="expected SHA-256 digest"):
        memo.read(source, ContentHash("md5", "0" * 32), max_bytes=SOURCE_LIMIT)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive byte limit"):
            memo.read(source, digest, max_bytes=bad)  # type: ignore[arg-type]
    with pytest.raises(D1ArtifactSnapshotError, match="exceeds its byte limit"):
        memo.read(source, digest, max_bytes=10)  # a hit enforces the SAME limit the miss path does
    other = tmp_path / "elsewhere"
    other.mkdir()
    with pytest.raises(ValueError, match="exactly one source root"):
        memo.read(other, digest, max_bytes=SOURCE_LIMIT)
    # a fresh memo (a fresh process) has nothing and still verifies against the file
    fresh = CorpusMemo()
    with pytest.raises(ValueError, match="does not match expected artifact"):
        fresh.read(source, ContentHash("sha256", "0" * 64), max_bytes=SOURCE_LIMIT)
    assert (fresh.read_replays, fresh.hits) == (0, 0)


def test_the_memo_is_keyed_by_the_prepared_digest_never_by_the_path(tmp_path: Path) -> None:
    """The same path rewritten under another digest must not be served the old corpus."""
    from tests.test_sentence_sampling import corpus

    from entropic_substrate.core.sentence.corpus import SplitPolicy, prepare_documents
    from entropic_substrate.ladder.capture.capture import write_prepared_corpus
    from entropic_substrate.ladder.run.seals import CorpusMemo

    source, digest = corpus_world(tmp_path)
    memo = CorpusMemo()
    original = memo.read(source, digest, max_bytes=SOURCE_LIMIT)
    base = corpus()
    smaller = prepare_documents(
        base.candidates[:20],
        replace(
            base.spec,
            candidate_limit=20,
            splits=SplitPolicy(validation_buckets=2000, test_buckets=1000),
        ),
    )
    (source / "prepared-corpus.json").unlink()
    rewritten = write_prepared_corpus(source, smaller, max_bytes=SOURCE_LIMIT)
    assert rewritten.content_hash.digest != digest.digest
    other = memo.read(source, rewritten.content_hash, max_bytes=SOURCE_LIMIT)
    assert other.corpus == smaller and other.corpus != original.corpus
    assert (
        memo.read(source, digest, max_bytes=SOURCE_LIMIT).corpus == original.corpus
    )  # still keyed


def test_a_verified_corpus_cannot_be_minted_from_a_forged_preparation() -> None:
    from tests.test_sentence_sampling import corpus

    from entropic_substrate.core.sentence.sampling import VerifiedCorpus

    good = corpus()
    assert VerifiedCorpus(good).corpus == good
    forged = replace(good, records=())
    with pytest.raises(ValueError, match="prepared source fails independent replay"):
        VerifiedCorpus(forged)
    with pytest.raises(TypeError, match="prepared source"):
        VerifiedCorpus(object())  # type: ignore[arg-type]


def test_one_worker_run_replays_each_source_root_once_and_no_sampler_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DH-41's measured acceptance, pinned: after one single-cell worker run, ``read_prepared_corpus``
    ran ONCE for the workspace source root and ONCE for the capture root (it ran 64 and 20 times at
    043bee0c) and no ``SentenceSampler`` re-prepared a corpus (42 times). Counted at EVERY
    importing site, with a passing control so 'zero' cannot mean 'counter not wired'."""
    import importlib
    import sys

    from tests.test_sentence_worker_execution import prepared

    import entropic_substrate.core.sentence.sampling as sampling
    from entropic_substrate.ladder.run.object_worker import run_object_worker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    sites = (
        "journal",
        "seals",
        "execution",
        "evaluation_records",
        "span_states",
        "span_diagnostics",
        "worker_execution",
    )
    roots: list[tuple[str, str]] = []
    for site in sites:
        module = importlib.import_module(f"entropic_substrate.ladder.run.{site}")
        inner = module.read_prepared_corpus

        def counted(
            root: Path, expected: Any, _site: str = site, _inner: Any = inner, **kw: Any
        ) -> Any:
            roots.append((_site, "/".join(Path(root).resolve().parts[-2:])))
            return _inner(root, expected, **kw)

        monkeypatch.setattr(module, "read_prepared_corpus", counted)
    replays: list[str] = []
    real_prepare = sampling.prepare_documents

    def counted_prepare(*args: Any, **kwargs: Any) -> Any:
        if sys._getframe(1).f_code.co_name == "__init__":  # a SentenceSampler construction
            replays.append("sampler")
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(sampling, "prepare_documents", counted_prepare)
    # positive control for the frame filter: a sampler built from a BARE corpus does replay
    import torch
    from tests.test_sentence_sampling import corpus as bare_corpus

    from entropic_substrate.core.sentence.boundary import SentenceByteCodec
    from entropic_substrate.core.sentence.sampling import SamplingSpec, SentenceSampler

    SentenceSampler(
        bare_corpus(), SamplingSpec("train", 4, 2, 42), SentenceByteCodec(), torch.device("cpu")
    )
    assert replays == ["sampler"]
    replays.clear()
    request, inputs = prepared(tmp_path)
    roots.clear()  # the fixture's own engine construction is not the run under test
    replays.clear()
    control = importlib.import_module("entropic_substrate.ladder.run.journal")
    control.read_prepared_corpus(
        inputs / "source",
        ContentHash("sha256", request.profile.source_digest),
        max_bytes=10_000_000,
    )
    assert roots == [("journal", "worker/source")]  # the control: the counter IS wired
    roots.clear()
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    worker = tmp_path / "object-worker"
    worker.mkdir()
    run_object_worker(objects, marker, worker, max_request_bytes=1_000_000)
    by_root = {r: sum(1 for _, x in roots if x == r) for r in {x for _, x in roots}}
    assert by_root == {"object-worker/source": 1, "capture/source": 1}, roots
    assert replays == []


# =============================================== recorded members, event identities, analysis
def test_the_recorded_exclusion_members_are_register_members_with_pinned_values() -> None:
    """P5-08: a recorded reason is an ``H1ExclusionReason`` member (never a free string) and the
    values the worker writes into result v2 are pinned here, so a rename in the register layer
    fails this test instead of silently changing a wire value."""
    from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
    from entropic_substrate.ladder.run.worker_execution import verdict_record

    pinned = {
        "NONFINITE_ENDPOINT": "nonfinite_endpoint",
        "MISSING_ENDPOINT": "missing_endpoint",
        "READER_REFUSAL": "reader_refusal",
    }
    assert {name: H1ExclusionReason[name].value for name in pinned} == pinned
    for name, value in pinned.items():
        record = verdict_record("excluded", H1ExclusionReason[name], "donor")
        assert record == {"verdict": "excluded", "reason": value, "cell": "donor"}
    with pytest.raises(TypeError, match="H1ExclusionReason member"):
        verdict_record("excluded", "missing_endpoint", "donor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="names its reason and its cell"):
        verdict_record("excluded", None, None)
    with pytest.raises(ValueError, match="names its reason and its cell"):
        verdict_record("complete", H1ExclusionReason.MISSING_ENDPOINT, "donor")


def test_pair_level_events_carry_the_pair_id_and_cell_events_their_cells_identity(
    tmp_path: Path, pair: Pair
) -> None:
    request, objects, marker = pair_world(tmp_path, pair)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    events = {
        key.removeprefix(f"{request.prefix}/").removesuffix(".json"): json.loads(value)
        for key, value in objects.values.items()
        if key.startswith(f"{request.prefix}/")
        and "/" not in key.removeprefix(f"{request.prefix}/")
        and key.endswith(".json")
        and key != f"{request.prefix}/request.json"
    }
    pair_id = request.binding.pair_id
    assert pair_id not in (pair.donor.identity, pair.recipient.identity)
    assert (
        events["started"]["run_identity"] == pair_id
        and events["complete"]["run_identity"] == pair_id
    )
    checkpoints = {name: e for name, e in events.items() if name.startswith("checkpoint-")}
    # the cells seal at the same global steps; the cell token keeps their names apart
    assert sorted(checkpoints) == [
        "checkpoint-donor-00000002",
        "checkpoint-donor-00000004",
        "checkpoint-donor-00000006",
        "checkpoint-recipient-00000004",
        "checkpoint-recipient-00000006",
    ]
    for name, event in checkpoints.items():
        expected = pair.donor.identity if "-donor-" in name else pair.recipient.identity
        assert event["run_identity"] == expected, name
        assert event["scientific_authority"] == "none" and event["schema"] == "h-worker-event-v1"
    # 'started' is published once per invocation
    assert objects.publishes.count(f"{request.prefix}/started.json") == 1


def test_a_completed_h1_cells_files_are_measured_against_the_analysis_reader(
    tmp_path: Path, pair: Pair
) -> None:
    """P5-15: ``load_h_run`` is run on each completed cell's (metrics, log) and the outcome is
    REPORTED, not assumed. MEASURED 2026-11-01 on the pair tracer: it REFUSES both cells with a
    typed ``TelemetryGap`` at the first prediction row, because an H1 cell trains with
    ``future_weight = 0`` and its prediction rows carry a null ``future_nll_per_dimension``, which
    the reader admits only in the codec warm-up. That is a latent risk for the analysis lane (not
    a crash, so not this slice's to fix); the pin below makes a change to either side loud."""
    from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
    from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

    _, objects, marker = pair_world(tmp_path, pair)
    status, error, _, observability = run_command(tmp_path, objects, marker)
    assert status == 0, error
    for token in ("donor", "recipient"):
        with pytest.raises(
            TelemetryGap, match=r"step 1 \(prediction\) has a null future_nll_per_dimension"
        ):
            load_h_run(
                observability / token / "metrics.jsonl", observability / token / "worker-log.jsonl"
            )


def test_the_nonfinite_donors_files_on_the_default_path_are_measured_not_claimed(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5-11: NO parity is claimed for NONFINITE_ENDPOINT. The donor that diverged AFTER its
    warm-up-final seal has that seal in its log, so the widened loader (keyed on 'before the
    warm-up-final seal') stays out and the unchanged refusal fires. MEASURED here: the adapter
    derives READER_REFUSAL (a TelemetryGap) while the worker recorded NONFINITE_ENDPOINT; the two
    differ and this test pins that measurement so a change to either is noticed."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
    from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

    _, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "prediction", 1, UndefinedDirection(2))
    status, _, workspace, observability = run_command(tmp_path, objects, marker)
    assert status == 0
    recorded = json.loads((workspace / "worker/result.json").read_text())["verdict"]["reason"]
    assert recorded == "nonfinite_endpoint"
    with pytest.raises(TelemetryGap, match="warm-up length cannot be derived"):
        load_h_run(
            observability / "donor" / "metrics.jsonl", observability / "donor" / "worker-log.jsonl"
        )


def test_the_new_terminal_and_the_closed_status_sets_are_pinned() -> None:
    """P5-17/P5-16: the exclusion terminal's name and status join their closed sets, and the sets
    are pinned EXACTLY so a sixth name cannot arrive by drift."""
    from entropic_substrate.ladder.run import object_worker
    from entropic_substrate.ladder.run.worker_request import CELL_OUTCOMES, PAIR_STATUSES

    assert object_worker._TERMINAL_MARKERS == {  # pyright: ignore[reportPrivateUsage]
        "complete",
        "paused",
        "stopped",
        "failure",
        "excluded",
    }
    assert PAIR_STATUSES == (
        "probe_workload_complete",
        "paused",
        "stopped_by_operator",
        "pair_excluded",
        "execution_defect",
    )
    assert CELL_OUTCOMES == (
        "complete",
        "paused",
        "stopped_by_operator",
        "scientific_gap",
        "execution_defect",
        "not_run",
    )


# ============================================ the result v2 integrity layers, pinned by forgery
@pytest.mark.parametrize(
    ("status", "verdict"),
    [
        pytest.param(
            "paused",
            {"verdict": "complete", "reason": None, "cell": None},
            id="paused-claims-complete",
        ),
        pytest.param(
            "probe_workload_complete",
            {"verdict": "none", "reason": None, "cell": None},
            id="complete-claims-none",
        ),
        pytest.param(
            "pair_excluded",
            {"verdict": "complete", "reason": None, "cell": None},
            id="excluded-claims-complete",
        ),
    ],
)
def test_a_result_v2_whose_status_and_verdict_disagree_is_refused_by_the_structure_check(
    tmp_path: Path, pair: Pair, status: str, verdict: dict[str, object]
) -> None:
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, objects, marker = pair_world(tmp_path, pair)
    code, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert code == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert parse_pair_result(result, request)["status"] == "probe_workload_complete"
    forged = result | {"status": status, "verdict": verdict}
    with pytest.raises(ValueError, match="status, verdict and cell outcomes disagree"):
        parse_pair_result(forged, request)


def test_a_forged_excluded_reason_passes_the_structure_check_and_the_derivation_refuses_it(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run.object_worker import _admit_pair_result
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, objects, marker = pair_world(tmp_path, pair)
    # no warm-up-final seal exists (the donor diverged at warm-up step 1): the donor's own
    # evidence derives MISSING_ENDPOINT, so the honest result is admitted by the derivation
    Fault(monkeypatch, "warmup", 1, UndefinedDirection(2))
    code, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert code == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["verdict"]["reason"] == "missing_endpoint"
    assert _admit_pair_result(workspace, request, result, {}, None)["verdict"] == result["verdict"]
    # the reason is swapped for a member the structure admits (same cell, same outcome) and the
    # evidence does not derive: only the derivation layer can refuse it
    forged = result | {"verdict": result["verdict"] | {"reason": "nonfinite_endpoint"}}
    assert parse_pair_result(forged, request)["status"] == "pair_excluded"
    with pytest.raises(ValueError, match="verdict differs from the verdict its cells derive"):
        _admit_pair_result(workspace, request, forged, {}, None)


# ============================================ execution-defect stop evidence, pinned per source
def test_an_execution_defect_is_confirmed_only_by_its_own_on_disk_evidence(
    tmp_path: Path, pair: Pair
) -> None:
    from entropic_substrate.ladder.run.object_worker import _execution_stop_confirmed
    from entropic_substrate.ladder.run.runner import PROGRESS_SCHEMA_V2

    request = pair_request(pair)
    donor, recipient = request.cells[0].profile, request.cells[1].profile
    donor_root, recipient_root = tmp_path / "donor", tmp_path / "recipient"
    donor_root.mkdir()
    recipient_root.mkdir()
    # source 1, donor: no warm-up-final seal on disk is the donor's execution evidence
    assert _execution_stop_confirmed(donor_root, "donor", donor, None) is True
    seal = {
        "schema": "h-checkpoint-seal-v1",
        "phase": "codec_warmup",
        "step": donor.schedule.warmup_steps,
    }
    (donor_root / "seal.json").write_text(json.dumps(seal))
    assert _execution_stop_confirmed(donor_root, "donor", donor, None) is False  # refused
    # source 2, recipient: a recipient root holding no progress receipt is the evidence
    assert _execution_stop_confirmed(recipient_root, "recipient", recipient, None) is True
    (recipient_root / "progress.json").write_text(json.dumps({"schema": PROGRESS_SCHEMA_V2}))
    assert _execution_stop_confirmed(recipient_root, "recipient", recipient, None) is False
    # source 3, the cell's own training failure record: its status decides
    assert (
        _execution_stop_confirmed(donor_root, "donor", donor, {"status": "scientific_gap"}) is False
    )
    assert (
        _execution_stop_confirmed(donor_root, "donor", donor, {"status": "execution_defect"})
        is True
    )


# ============================================ host completion v2: each re-verification refusal
def _bump_donor_steps(row: dict[str, Any]) -> None:
    row["cells"][0]["successful_optimizer_steps"] += 1


def _rebind_donor_training(row: dict[str, Any]) -> None:
    row["cells"][0]["training"] = row["cells"][0]["training"] | {"sha256": "0" * 64}


def _pause_the_recipient(row: dict[str, Any]) -> None:
    row["cells"][1]["outcome"] = "paused"


def _relabel_the_status(row: dict[str, Any]) -> None:
    row["status"] = "paused"


@pytest.mark.parametrize(
    ("forge", "message"),
    [
        pytest.param(
            _bump_donor_steps,
            "does not establish the complete requested workload",
            id="C1-complete-cell-steps",
        ),
        pytest.param(
            _rebind_donor_training,
            "substituted a different captured lineage",
            id="C2-capture-manifest-training-bind",
        ),
        pytest.param(
            _pause_the_recipient,
            "a complete pair completes both cells",
            id="C3-complete-pair-completes-both",
        ),
        pytest.param(
            _relabel_the_status,
            "worker result status differs from its terminal event",
            id="C4-status-matches-the-terminal-event",
        ),
    ],
)
def test_the_host_completion_refuses_a_forged_cell_that_it_re_verifies_itself(
    tmp_path: Path,
    pair: Pair,
    monkeypatch: pytest.MonkeyPatch,
    forge: Any,
    message: str,
) -> None:
    from entropic_substrate.ladder.run import worker_completion
    from entropic_substrate.ladder.run.worker_completion import receive_worker_completion

    request, objects, marker = pair_world(tmp_path, pair)
    code, error, _, _ = run_command(tmp_path, objects, marker)
    assert code == 0, error
    real_parse = worker_completion.parse_pair_result
    # control: the honest result is admitted by the host completion
    (tmp_path / "host-honest").mkdir()
    honest = receive_worker_completion(
        objects,
        request,
        terminal_identity(objects, request, "complete"),
        tmp_path / "host-honest",
        expected_pod_id=None,
        max_marker_bytes=1_000_000,
    )
    assert honest["verdict"] == {"verdict": "complete", "reason": None, "cell": None}

    # the structure check admits the forged row (it is a consistent-looking v2 result), so only
    # the host's own re-verification of the cell can refuse it
    def parsed(value: object, req: PairProbeRequest) -> dict[str, object]:
        row = real_parse(value, req)
        forge(row)
        return row

    monkeypatch.setattr(worker_completion, "parse_pair_result", parsed)
    host = tmp_path / "host-forged"
    host.mkdir()
    with pytest.raises(ValueError, match=message):
        receive_worker_completion(
            objects,
            request,
            terminal_identity(objects, request, "complete"),
            host,
            expected_pod_id=None,
            max_marker_bytes=1_000_000,
        )


def _count_pair_replays(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[tuple[str, bool], int], dict[bool, int]]:
    """Run one pair request through the command and count both replay sites, split by whether a
    ``codec_handoff`` frame is on the stack: ``({(root, via_handoff): n}, {via_handoff: n})``."""
    import importlib
    import sys

    import entropic_substrate.core.sentence.sampling as sampling
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    def via_handoff() -> bool:
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_globals.get("__name__", "").endswith("codec_handoff"):
                return True
            frame = frame.f_back  # type: ignore[assignment]
        return False

    reads: dict[tuple[str, bool], int] = {}
    sites = (
        "journal",
        "seals",
        "execution",
        "evaluation_records",
        "span_states",
        "span_diagnostics",
        "worker_execution",
    )
    for site in sites:
        module = importlib.import_module(f"entropic_substrate.ladder.run.{site}")
        inner = module.read_prepared_corpus

        def counted(root: Path, expected: Any, _inner: Any = inner, **kw: Any) -> Any:
            key = ("/".join(Path(root).resolve().parts[-2:]), via_handoff())
            reads[key] = reads.get(key, 0) + 1
            return _inner(root, expected, **kw)

        monkeypatch.setattr(module, "read_prepared_corpus", counted)
    preps: dict[bool, int] = {}
    real_prepare = sampling.prepare_documents

    def counted_prepare(*args: Any, **kwargs: Any) -> Any:
        if sys._getframe(1).f_code.co_name == "__init__":  # a SentenceSampler construction
            key = via_handoff()
            preps[key] = preps.get(key, 0) + 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(sampling, "prepare_documents", counted_prepare)
    request = pair_request(pair)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, tmp_path)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    return reads, preps


def test_one_pair_run_replays_each_source_root_once_outside_the_handoff_and_no_sampler_does(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DH-41 on the PAIR path (the single-cell tracer above does not cover ``_execute_pair``).
    MEASURED at 53e1fc1f: outside ``codec_handoff`` frames ``read_prepared_corpus`` ran ONCE for
    each of the three roots (workspace source, donor capture, recipient capture) and no
    ``SentenceSampler`` re-prepared a corpus. Inside ``codec_handoff`` frames it ran 10 times on the
    workspace source root and 14 samplers re-prepared: that is the filed 'codec_handoff replay
    residual' (a reader-member file the lane may not commit), pinned here only as a CEILING so it
    cannot grow, never as a target. Dropping the pair's corpus memo (``source_corpora = None``) is
    measured to move the totals to 139 reads and 81 re-preparations."""
    reads, preps = _count_pair_replays(tmp_path, pair, monkeypatch)
    outside = {root: n for (root, via), n in reads.items() if not via}
    assert outside == {
        "command-worker/source": 1,
        "donor/source": 1,
        "recipient/source": 1,
    }, reads
    assert preps.get(False, 0) == 0, preps
    # the residual ceiling (measured 10 and 14): a regression grows it, a fix lowers it
    assert sum(n for (_, via), n in reads.items() if via) <= 10, reads
    assert preps.get(True, 0) <= 14, preps


# ============================== L3 fixes: corners the mutation round showed unpinned (FOR-1693)
@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(
            lambda p: replace(p.recipient, schedule=replace(p.recipient.schedule, batch_size=3)),
            "donor and recipient manifests differ beyond their name and spine carrier",
            id="recipient-manifest-differs",
        ),
        pytest.param(
            lambda p: replace(p.donor, name="h1-pair-twin-tiny"),
            "a pair needs two different spine carriers",
            id="same-carrier-twin",
        ),
    ],
)
def test_the_request_refuses_a_pair_that_is_not_codec_sharing_before_any_spend(
    pair: Pair, change: Any, message: str
) -> None:
    """The request-time pair predicate is the ONLY check before hardware: without it a
    mismatched pair is admitted, the donor trains to completion and only the hand-off refuses.
    MEASURED at 19e58ad6 with the ``require_pair`` call replaced by ``pass``: the pair whose
    cells differ in ``schedule.batch_size`` is admitted."""
    from entropic_substrate.ladder.run.codec_handoff import CodecHandoffDefect

    assert (
        pair_request(pair).cells[1].profile == pair.recipient
    )  # control: the honest pair is admitted
    other = change(pair)
    cells = (PairCell(pair.donor, None, None), PairCell(other, None, None))
    with pytest.raises(CodecHandoffDefect, match=message):
        pair_request(pair, cells=cells)


def test_a_retired_v1_parent_receipt_is_refused_by_name_at_the_parent_hop(
    tmp_path: Path, pair: Pair
) -> None:
    """The second line of the paused-v1-lineage cutover defence: for a single smoke cell every
    binding field is ``None`` on both sides of the hop comparison, so only the by-name refusal
    stops a v1 parent that carries the same lineage fields."""
    from entropic_substrate.ladder.run.journal import read_json, write_json

    root = tmp_path / "donor"
    root.mkdir()
    paused = trained_cell(
        pair,
        pair.donor,
        root,
        stop_after_seal=lambda phase, step: (phase, step) == ("prediction", 2),
    )
    assert paused.status == "paused"
    done = trained_cell(pair, pair.donor, root, resume=paused.receipt)
    assert done.status == "training_complete"
    parent = read_json(root, paused.receipt, max_bytes=LIMIT)
    v1_parent = write_json(
        root,
        "v1-parent-progress.json",
        {k: v for k, v in parent.items() if k not in _V2_ONLY}
        | {"schema": "h-probe-training-progress-v1"},
        max_bytes=LIMIT,
    )
    # control: the chain as written reads
    read_receipt(pair, pair.donor, root, done.receipt)
    child = forged(
        root,
        done.receipt,
        "v1-parent-child.json",
        parent_receipt={
            "path": v1_parent.path,
            "sha256": v1_parent.content_hash.digest,
        },
    )
    with pytest.raises(
        ValueError,
        match=r"h-probe-training-progress-v1.*minimum supported version is h-probe-training-progress-v2",
    ):
        read_receipt(pair, pair.donor, root, child)


def test_only_the_runner_assigns_the_progress_v2_schema_name() -> None:
    """``runner.write_progress_receipt`` is the ONE writer of an ``h-probe-training-progress-v2``
    receipt. Structural pin: across ``src`` no dict literal and no keyword argument other than
    the one inside that function sets ``schema`` to the progress-v2 name (by constant or by
    literal), so a second inline writer goes RED instead of drifting from the reader's shape."""
    import ast

    import entropic_substrate
    from entropic_substrate.ladder.run.runner import PROGRESS_SCHEMA_V2

    def is_name(node: ast.AST | None) -> bool:
        return (isinstance(node, ast.Name) and node.id == "PROGRESS_SCHEMA_V2") or (
            isinstance(node, ast.Constant) and node.value == PROGRESS_SCHEMA_V2
        )

    sites: list[str] = []
    root = Path(entropic_substrate.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners: dict[int, str] = {}
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(function):
                    owners[id(child)] = function.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                hit = any(
                    isinstance(k, ast.Constant) and k.value == "schema" and is_name(v)
                    for k, v in zip(node.keys, node.values, strict=True)
                )
            elif isinstance(node, ast.Call):
                hit = any(k.arg == "schema" and is_name(k.value) for k in node.keywords)
            else:
                continue
            if hit:
                sites.append(f"{path.relative_to(root)}:{owners.get(id(node), '<module>')}")
    assert sites == ["ladder/run/runner.py:write_progress_receipt"], sites


# ---- DH-15: the two failure classes meet in one result, and the execution class wins
def _fail_donor_scientifically_then_recipient_by_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    # the second Fault wraps the first: the donor's first prediction step (call 1) reaches the
    # inner one and diverges; the recipient's first prediction step (call 2) is the outer one's
    Fault(monkeypatch, "prediction", 1, UndefinedDirection(2))
    Fault(monkeypatch, "prediction", 2, RuntimeError("injected recipient defect"))


def test_a_donor_scientific_gap_then_a_recipient_execution_defect_is_the_execution_class(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only pair where DH-15's two classes share one result: the donor stops scientifically
    (NONFINITE_ENDPOINT, the hand-off proceeds) and the recipient then stops as an execution
    defect. The execution defect decides the verdict (READER_REFUSAL naming the recipient), and
    the failure arm reads the result back through the derivation, which fails closed if the
    precedence is swapped (the swapped derivation names the donor's NONFINITE_ENDPOINT)."""
    request, objects, marker = pair_world(tmp_path, pair)
    _fail_donor_scientifically_then_recipient_by_execution(monkeypatch)
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2  # the recipient's ORIGINAL exception propagated
    failure = failure_terminal(objects, request)
    assert failure["status"] == "execution_defect" and failure["error_type"] == "RuntimeError"
    assert failure["evidence_delivery"] == "captured"  # the read-back held, the snapshot went out
    result = json.loads(failure_snapshot_files(objects, request)["worker/result.json"])
    assert result["status"] == "execution_defect"
    assert result["verdict"] == {
        "verdict": "excluded",
        "reason": "reader_refusal",
        "cell": "recipient",
    }
    assert [c["outcome"] for c in result["cells"]] == ["scientific_gap", "execution_defect"]


# ---- result v2: the host-side reader couples status, reason and the named cell's outcome
def _scientific_and_execution_results(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> tuple[PairProbeRequest, dict[str, Any], dict[str, Any]]:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    request, objects, marker = pair_world(tmp_path, pair)
    with monkeypatch.context() as scientific:
        Fault(scientific, "prediction", 1, UndefinedDirection(2))
        code, error, workspace, _ = run_command(tmp_path, objects, marker, "sci-worker")
        assert code == 0, error
        excluded = json.loads((workspace / "worker/result.json").read_text())
    _, objects, marker = pair_world(tmp_path, pair)
    with monkeypatch.context() as execution:
        Fault(execution, "warmup", 1, RuntimeError("injected reader refusal"))
        code, _, _, _ = run_command(tmp_path, objects, marker, "exe-worker")
        assert code == 2
        defect = json.loads(failure_snapshot_files(objects, request)["worker/result.json"])
    return request, excluded, defect


def test_the_pair_reader_refuses_every_status_reason_and_named_cell_class_mix(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each forgery changes exactly ONE coupling of an otherwise honest result, so each row is
    caught by exactly one clause of the structure check (not by the worker's disk derivation,
    which the host-side ``receive_worker_completion`` path never runs):
    status pair_excluded needs a SCIENTIFIC reason naming a scientific_gap cell; status
    execution_defect needs reader_refusal naming an execution_defect cell."""
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, excluded, defect = _scientific_and_execution_results(tmp_path, pair, monkeypatch)
    assert excluded["status"] == "pair_excluded" and defect["status"] == "execution_defect"
    assert parse_pair_result(excluded, request) == excluded  # controls: both honest rows read
    assert parse_pair_result(defect, request) == defect
    forgeries = {
        "A18-excluded-with-reader-refusal": excluded
        | {"verdict": excluded["verdict"] | {"reason": "reader_refusal"}},
        "A18b-defect-with-missing-endpoint": defect
        | {"verdict": defect["verdict"] | {"reason": "missing_endpoint"}},
        "A18c-excluded-names-a-complete-cell": excluded
        | {"verdict": excluded["verdict"] | {"cell": "recipient"}},
        "A18d-defect-names-a-not-run-cell": defect
        | {"verdict": defect["verdict"] | {"cell": "recipient"}},
    }
    for name, forged_row in forgeries.items():
        with pytest.raises(ValueError, match="status, verdict and cell outcomes disagree"):
            parse_pair_result(forged_row, request)
        assert name  # the id of the clause this row pins


# ---- the failure arm reads an execution stop's result v2 back before publishing its snapshot
def test_the_failure_arm_refuses_a_bad_result_v2_before_it_publishes_the_snapshot(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import worker_execution

    real = worker_execution.pair_result

    def misnamed(*args: Any, **kwargs: Any) -> dict[str, object]:
        row = real(*args, **kwargs)
        # the verdict names the recipient, whose cell never ran: not a result the evidence derives
        return row | {"verdict": dict(row["verdict"]) | {"cell": "recipient"}}  # type: ignore[arg-type]

    monkeypatch.setattr(worker_execution, "pair_result", misnamed)
    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, RuntimeError("injected reader refusal"))
    status, _, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2  # the ORIGINAL exception still propagates
    failure = failure_terminal(objects, request)
    assert failure["status"] == "execution_defect" and failure["error_type"] == "RuntimeError"
    assert failure["evidence_delivery"] == "failed"
    assert failure["delivery_error_type"] == "ValueError"
    assert failure["snapshot"] is None
    assert f"{request.prefix}/snapshots/failure.json" not in objects.values  # nothing published


@pytest.mark.parametrize("free_reason", ["not a member of the closed set", ""])
def test_an_excluded_verdict_whose_reason_is_a_free_string_is_refused_by_the_enum_member_guard(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch, free_reason: str
) -> None:
    """Sacred invariant (DH-45/DH-46/DH-47): the recorded exclusion reason is an H1ExclusionReason
    member, never a free string. The membership guard in parse_pair_result must name the enum, so
    removing it changes the refusal (a status/verdict message alone would also refuse, but not
    by naming the closed set)."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, UndefinedDirection(2))
    code, error, workspace, _ = run_command(tmp_path, objects, marker)
    assert code == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    assert result["verdict"]["reason"] == "missing_endpoint"
    forged = result | {"verdict": result["verdict"] | {"reason": free_reason}}
    with pytest.raises(ValueError, match="H1ExclusionReason"):
        parse_pair_result(forged, request)
