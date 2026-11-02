"""FOR-1693 (h1-l4b), fixer round: the pair predicates the mutation round left unguarded.

``tests/test_h_pair_request.py`` pins the pair end to end. Its mutation round found predicates
that a flip left GREEN (a guard whose happy path passes does not show the guard worked). Each
test below names the predicate it pins and goes RED when that predicate is removed or inverted.

Two limits are stated, not hidden (house rule 4):

- A scientific pair cannot EXECUTE on a CPU: a pilot profile requires CUDA/bf16 at construction
  and no test patches admission to pretend otherwise. The receipt-layer half of "both cells under
  one authority" is therefore pinned where it is reachable: the pure binding function, the reader
  arms (with only the seal-chain walk stood in, never admission), and a structural pin that every
  site of the pair path passes the request's one authority. The structural pin reads source text,
  so it proves the call shape and not the run; the run itself belongs to the live pod wave.
- ``h_probe_launch.parse_probe_launch``'s own pair refusal is shadowed by ``H0ProbeLaunch``'s
  constructor, which raises the same message and always runs: the two guards are indistinguishable
  by any input, so no separate pin exists for the parse-side line (defence in depth).
"""

import ast
import copy
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_codec_handoff import Pair, make_pair
from tests.test_h_pair_request import (
    BLOCK,
    LIMIT,
    PAIR_ID_FIXTURE,
    SOURCE_LIMIT,
    Fault,
    failure_snapshot_files,
    failure_terminal,
    pair_request,
    pair_world,
    run_command,
    terminal_identity,
)
from tests.test_h_probe_launch import pilot_request
from tests.test_run_authority import authority as sealed_authority

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash

pytestmark = pytest.mark.unit


@pytest.fixture
def pair(tmp_path: Path) -> Pair:
    return make_pair(tmp_path)


def refused(case: str, message: str, call: Any) -> None:
    """Run ``call``; it must raise ``ValueError`` matching ``message`` (the case names the row)."""
    import re

    try:
        call()
    except ValueError as error:
        assert re.search(message, str(error)), f"{case}: refused with {error!s}"
        return
    pytest.fail(f"{case}: not refused")


def refusal_of(call: Any) -> str:
    try:
        call()
    except ValueError as error:
        return str(error)
    return ""


# ===================================================================== two real pair runs
@dataclass(frozen=True)
class World:
    pair: Pair
    request: Any
    objects: _ObjectStore
    workspace: Path
    result: dict[str, Any]


def run_world(root: Path, fault: Any = None) -> World:
    pair = make_pair(root)
    request, objects, marker = pair_world(root, pair)
    with pytest.MonkeyPatch.context() as patch:
        if fault is not None:
            fault(patch)
        status, error, workspace, _ = run_command(root, objects, marker)
    assert status == 0, error
    result = json.loads((workspace / "worker/result.json").read_text())
    return World(pair, request, objects, workspace, result)


@pytest.fixture(scope="module")
def honest(tmp_path_factory: pytest.TempPathFactory) -> World:
    """A pair that ran to completion: the evidence the forgeries below start from."""
    return run_world(tmp_path_factory.mktemp("honest"))


@pytest.fixture(scope="module")
def gapped(tmp_path_factory: pytest.TempPathFactory) -> World:
    """A pair whose donor diverged at warm-up step 1: a scientific gap with a failure record."""
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    return run_world(
        tmp_path_factory.mktemp("gapped"),
        lambda patch: Fault(patch, "warmup", 1, UndefinedDirection(2)),
    )


# ============================================== the receipt binding and the authority (lead)
def test_a_receipt_binding_carries_the_cell_scope_and_the_sealed_authority_digest() -> None:
    from entropic_substrate.ladder.run.runner import CellScope, ReceiptBinding

    authority = sealed_authority()
    scope = CellScope(BLOCK, PAIR_ID_FIXTURE, "donor")
    assert ReceiptBinding.of(scope, authority) == ReceiptBinding(
        BLOCK, PAIR_ID_FIXTURE, "donor", authority.digest
    )
    assert ReceiptBinding.of(scope, None) == ReceiptBinding(BLOCK, PAIR_ID_FIXTURE, "donor", None)
    assert ReceiptBinding.of(None, authority) == ReceiptBinding(None, None, None, authority.digest)
    assert ReceiptBinding.of(None, None) == ReceiptBinding()
    assert ReceiptBinding.of(scope, authority) != ReceiptBinding.of(scope, None)


def test_a_scientific_receipt_must_name_its_authority_and_a_smoke_receipt_must_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader's two arms of ``(digest is None) != (run class is scientific)``. The smoke arm
    is pinned in ``test_h_pair_request``; the scientific arm needs a pilot profile. Only the
    seal-chain walk is stood in (its verdict is not what is under test); the class admission,
    the receipt writer and the reader are the real ones."""
    from entropic_substrate.ladder.run import runner
    from entropic_substrate.ladder.run.journal import write_json
    from entropic_substrate.ladder.run.runner import (
        CellScope,
        ReceiptBinding,
        read_training_progress,
        write_progress_receipt,
    )

    profile = pilot_request().profile
    authority = sealed_authority()
    root = tmp_path / "pilot"
    root.mkdir()
    profile_artifact = write_json(root, "profile.json", profile.manifest, max_bytes=LIMIT)
    seal = write_json(root, "seal.json", {"stand_in": "a checkpoint seal"}, max_bytes=LIMIT)
    stand_in = SimpleNamespace(
        phase="codec_warmup", step=1, root_kind="fresh", root_journal="0" * 64, seal_hashes=set()
    )
    monkeypatch.setattr(runner, "read_seal", lambda *args, **kwargs: stand_in)

    def receipt(name: str, binding: ReceiptBinding) -> ArtifactIdentity:
        return write_progress_receipt(
            root,
            name,
            profile,
            profile_artifact,
            status="checkpoint",
            phase="codec_warmup",
            step=1,
            last_seal=seal,
            evaluations=[],
            parent_receipt=None,
            inventory={},
            max_bytes=LIMIT,
            binding=binding,
        )

    def read(artifact: ArtifactIdentity, **kwargs: Any) -> dict[str, object]:
        return read_training_progress(
            root,
            artifact,
            profile,
            tmp_path,
            max_source_bytes=SOURCE_LIMIT,
            max_artifact_bytes=LIMIT,
            **kwargs,
        )

    bound = ReceiptBinding.of(CellScope(BLOCK, PAIR_ID_FIXTURE, "donor"), authority)
    sealed = receipt("progress-sealed.json", bound)
    row = read(sealed, expected=bound)  # control: the scientific receipt with its digest is read
    assert row["authority_digest"] == authority.digest and row["run_class"] == "pilot"
    with pytest.raises(ValueError, match="binding differs from the expected binding"):
        read(sealed, expected=replace(bound, authority_digest="f" * 64))
    bare = receipt("progress-bare.json", replace(bound, authority_digest=None))
    with pytest.raises(ValueError, match="authority digest disagrees with its run class"):
        read(bare)


def test_every_training_and_receipt_site_of_the_pair_path_passes_the_requests_one_authority() -> (
    None
):
    """Structural pin (precedent: ``test_only_the_runner_assigns_the_progress_v2_schema_name``):
    a scientific pair cannot run on a CPU, so the threading of ``request.authority`` into
    ``run_probe_training`` and ``ReceiptBinding.of`` is pinned on the call shapes."""
    import entropic_substrate

    root = Path(entropic_substrate.__file__).parent / "ladder" / "run"

    def is_request_authority(node: ast.AST | None) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "authority"
            and isinstance(node.value, ast.Name)
            and node.value.id == "request"
        )

    for name in ("worker_execution.py", "object_worker.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        binders = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "of"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ReceiptBinding"
        ]
        assert binders, f"{name} builds no ReceiptBinding"
        for site in binders:
            assert len(site.args) == 2 and is_request_authority(site.args[1]), (
                f"{name}:{site.lineno} binds a receipt without the request's authority"
            )
    tree = ast.parse((root / "worker_execution.py").read_text(encoding="utf-8"))
    pair_function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_pair"
    )
    trainers = [
        (node, scope)
        for scope, subtree in (("file", tree), ("pair", pair_function))
        for node in ast.walk(subtree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_probe_training"
    ]
    assert any(scope == "pair" for _, scope in trainers), "the pair path runs no training"
    for site, scope in trainers:
        keywords = {k.arg: k.value for k in site.keywords}
        assert is_request_authority(keywords.get("authority")), (
            f"worker_execution:{site.lineno} trains without the request's authority"
        )
        if scope == "pair":  # the single-profile site (outside _execute_pair) has no cell scope
            assert "scope" in keywords, f"worker_execution:{site.lineno} trains without its scope"


@pytest.mark.parametrize(
    ("block", "pair_id", "cell"),
    [
        pytest.param("g" * 64, PAIR_ID_FIXTURE, "donor", id="block-not-hex"),
        pytest.param("b" * 63, PAIR_ID_FIXTURE, "donor", id="block-short"),
        pytest.param("B" * 64, PAIR_ID_FIXTURE, "donor", id="block-uppercase"),
        pytest.param(BLOCK, "g" * 64, "donor", id="pair-not-hex"),
        pytest.param(BLOCK, "a" * 65, "donor", id="pair-long"),
        pytest.param(BLOCK, "A" * 64, "donor", id="pair-uppercase"),
        pytest.param(BLOCK, PAIR_ID_FIXTURE, "Donor", id="cell-uppercase"),
    ],
)
def test_a_cell_scope_names_a_64_hex_block_a_64_hex_pair_and_a_safe_cell(
    block: str, pair_id: str, cell: str
) -> None:
    from entropic_substrate.ladder.run.runner import CellScope

    assert CellScope(BLOCK, PAIR_ID_FIXTURE, "donor").cell == "donor"  # control
    with pytest.raises(ValueError, match="64-hex block, a 64-hex pair and a safe cell"):
        CellScope(block, pair_id, cell)


# ============================================================ the pair request's own guards
def test_the_pair_request_checks_each_cells_pause_and_its_input_namespace(pair: Pair) -> None:
    from entropic_substrate.ladder.run.worker_request import PairCell

    pair_request(pair)  # control
    # 3 is not a seal of the 4-step schedule (seals at 2 and 4): the pair CALLS the pause check
    for index in (0, 1):
        subjects = [pair.donor, pair.recipient]
        cells = [PairCell(subjects[0], None, None), PairCell(subjects[1], None, None)]
        cells[index] = PairCell(subjects[index], None, ("prediction", 3))
        refused(
            f"pause-on-cell-{index}",
            "worker pause must select a declared seal boundary",
            lambda cells=cells: pair_request(pair, cells=(cells[0], cells[1])),
        )
    request = pair_request(pair)
    stranger = ArtifactIdentity("other/x.json", ContentHash("sha256", "c" * 64))
    refused(
        "input-outside-namespace",
        "worker input is outside the declared input namespace",
        lambda: pair_request(pair, inputs=(*request.inputs, stranger)),
    )


# ====================================================== the result v2 structure (host boundary)
def test_the_pair_result_structure_check_pins_identity_entries_references_and_the_status_set(
    honest: World,
) -> None:
    from entropic_substrate.ladder.run.worker_request import parse_pair_result

    request, result = honest.request, honest.result
    assert parse_pair_result(result, request) == result  # control: the honest row reads
    donor_run = result["cells"][0]["training"]
    donor_name = donor_run["path"].rsplit("/", 1)[1]
    donor_capture = result["cells"][0]["capture"]

    def entry(index: int, **fields: Any) -> dict[str, Any]:
        cells = copy.deepcopy(result["cells"])
        cells[index] = cells[index] | fields
        return result | {"cells": cells}

    def bare(index: int, outcome: str, steps: int = 0, **keep: Any) -> dict[str, Any]:
        """One cell entry stripped to ``outcome`` (every reference None, then ``keep``)."""
        blank = dict.fromkeys(("training", "failure", "generation", "diagnostics", "capture")) | {
            "verification": None
        }
        return entry(index, outcome=outcome, successful_optimizer_steps=steps, **(blank | keep))

    identity = "pair worker result identity mismatch"
    cell_entry = "pair worker result cell entry identity or outcome mismatch"
    references = "pair worker result cell references differ from its outcome"
    foreign = "worker cell artifact belongs to another cell or section"
    # a defect-shaped result whose ONLY fault is its status (P12): the else-arm would admit it
    defect = (
        bare(0, "execution_defect")
        | {"cells": [bare(0, "execution_defect")["cells"][0], bare(1, "not_run")["cells"][1]]}
        | {"verdict": {"verdict": "excluded", "reason": "reader_refusal", "cell": "donor"}}
        | {"status": "execution_defect"}
    )
    assert parse_pair_result(defect, request)["status"] == "execution_defect"  # control
    forgeries: dict[str, tuple[dict[str, Any], str]] = {
        "request-digest": (result | {"request_sha256": "0" * 64}, identity),
        "run-identity": (result | {"run_identity": "0" * 64}, identity),
        "authority": (result | {"scientific_authority": "x"}, identity),
        "schema": (result | {"schema": "h-probe-worker-result-v0"}, identity),
        "duration-negative": (result | {"wall_seconds": -1.0}, "duration is invalid"),
        "duration-bool": (result | {"wall_seconds": True}, "duration is invalid"),
        "duration-infinite": (result | {"wall_seconds": float("inf")}, "duration is invalid"),
        "status-outside-the-closed-set": (
            defect | {"status": "bogus"},
            "status is outside the closed set",
        ),
        "one-entry": (result | {"cells": result["cells"][:1]}, "exactly two cell entries"),
        "entry-names-the-other-cell": (entry(0, cell="recipient"), cell_entry),
        "entry-carries-the-other-identity": (
            entry(0, run_identity=result["cells"][1]["run_identity"]),
            cell_entry,
        ),
        "entry-outcome-unknown": (entry(0, outcome="bogus"), cell_entry),
        "entry-steps-negative": (entry(0, successful_optimizer_steps=-1), cell_entry),
        "entry-steps-bool": (entry(0, successful_optimizer_steps=True), cell_entry),
        "training-under-the-other-cell": (
            entry(
                0, training={"path": f"run/recipient/{donor_name}", "sha256": donor_run["sha256"]}
            ),
            foreign,
        ),
        "capture-under-the-run-section": (entry(0, capture=donor_run), foreign),
        "complete-without-generation": (entry(0, generation=None), references),
        "complete-without-capture": (entry(0, capture=None), references),
        "complete-with-a-failure": (entry(0, failure=donor_run), references),
        "paused-carries-the-later-stages": (entry(0, outcome="paused"), references),
        "paused-without-training": (bare(0, "paused"), references),
        "gap-without-its-failure": (bare(0, "scientific_gap"), references),
        "gap-carries-a-capture": (
            bare(0, "scientific_gap", failure=donor_run, capture=donor_capture),
            references,
        ),
        "defect-carries-a-capture": (
            bare(0, "execution_defect", capture=donor_capture),
            references,
        ),
        "not-run-with-steps": (bare(0, "not_run", steps=1), references),
        "not-run-with-training": (bare(0, "not_run", training=donor_run), references),
        "verdict-kind-outside-the-set": (
            result | {"verdict": {"verdict": "bogus", "reason": None, "cell": None}},
            "verdict is outside the closed set",
        ),
        "complete-verdict-with-a-reason": (
            result
            | {"verdict": {"verdict": "complete", "reason": "missing_endpoint", "cell": None}},
            "names no reason and no cell",
        ),
    }
    for case, (forged, message) in forgeries.items():
        refused(case, message, lambda forged=forged: parse_pair_result(forged, request))
    # the controls of the reference table: a clean entry of each outcome is NOT refused for its
    # references (the later status/verdict coupling may still refuse the row as a whole)
    for case, clean in {
        "gap": bare(0, "scientific_gap", failure=donor_run),
        "defect": bare(0, "execution_defect"),
        "not-run": bare(0, "not_run"),
        "paused": bare(0, "paused", training=donor_run),
    }.items():
        assert references not in refusal_of(
            lambda clean=clean: parse_pair_result(clean, request)
        ), case


# ======================================================== the disk derivation (worker side)
def with_failure(world: World, root: Path, **changes: Any) -> tuple[Path, dict[str, Any]]:
    """A copy of ``world``'s workspace plus its result with the donor's failure record replaced
    by a sibling record carrying ``changes`` (content-addressed, so the reference moves too)."""
    from entropic_substrate.ladder.run.journal import read_json, write_json

    workspace = root / "copy"
    shutil.copytree(world.workspace, workspace)
    donor_root = workspace / "run" / "donor"
    reference = world.result["cells"][0]["failure"]
    name = reference["path"].rsplit("/", 1)[1]
    record = read_json(
        donor_root,
        ArtifactIdentity(name, ContentHash("sha256", reference["sha256"])),
        max_bytes=LIMIT,
    )
    written = write_json(donor_root, "forged-failure.json", record | changes, max_bytes=LIMIT)
    moved = {"path": "run/donor/forged-failure.json", "sha256": written.content_hash.digest}
    cells = copy.deepcopy(world.result["cells"])
    cells[0]["failure"] = moved
    return workspace, world.result | {"cells": cells}


def test_the_disk_derivation_pins_each_cells_steps_failure_record_and_capture_lineage(
    honest: World, gapped: World, tmp_path: Path
) -> None:
    from entropic_substrate.ladder.run.object_worker import _admit_pair_result

    # ---- a completed pair: the step count and the capture lineage of each reference
    workspace = tmp_path / "honest-copy"
    shutil.copytree(honest.workspace, workspace)
    request, result = honest.request, honest.result
    assert _admit_pair_result(workspace, request, result, {}, None) == result  # control

    def cell(index: int, **fields: Any) -> dict[str, Any]:
        cells = copy.deepcopy(result["cells"])
        cells[index] = cells[index] | fields
        return result | {"cells": cells}

    donor = result["cells"][0]
    receipt_name = donor["training"]["path"].rsplit("/", 1)[1]
    shutil.copyfile(
        workspace / "run" / "donor" / receipt_name, workspace / "run" / "donor" / "copy.json"
    )
    copied = {"path": "run/donor/copy.json", "sha256": donor["training"]["sha256"]}
    lineage = "worker result is not the verified capture lineage"
    for case, forged, message in (
        (
            "steps-differ-from-the-admitted-training",
            cell(0, successful_optimizer_steps=donor["successful_optimizer_steps"] + 1),
            "worker result step count differs from admitted training",
        ),
        ("generation-is-another-file", cell(0, generation=donor["training"]), lineage),
        ("diagnostics-are-another-file", cell(0, diagnostics=donor["training"]), lineage),
        ("training-is-an-identical-copy-under-another-name", cell(0, training=copied), lineage),
    ):
        refused(case, message, lambda f=forged: _admit_pair_result(workspace, request, f, {}, None))

    # ---- a scientific gap: the failure record's identity, schema and status
    gap = gapped.result
    assert gap["cells"][0]["outcome"] == "scientific_gap"
    root = tmp_path / "gap"
    root.mkdir()
    base_workspace, base_result = with_failure(gapped, root)  # an unchanged copy: the control
    assert _admit_pair_result(base_workspace, gapped.request, base_result, {}, None) == base_result
    for case, changes, message in (
        (
            "failure-of-another-run",
            {"run_identity": "0" * 64},
            "worker result failure record belongs to another run",
        ),
        (
            "failure-of-another-schema",
            {"schema": "h-probe-training-failure-v1"},
            "worker result failure record belongs to another run",
        ),
        (
            "scientific-outcome-with-an-execution-record",
            {"status": "execution_defect"},
            "a scientific cell outcome needs its scientific failure record",
        ),
    ):
        scratch = tmp_path / f"gap-{case}"
        scratch.mkdir()
        forged_workspace, forged_result = with_failure(gapped, scratch, **changes)
        refused(
            case,
            message,
            lambda w=forged_workspace, r=forged_result: _admit_pair_result(
                w, gapped.request, r, {}, None
            ),
        )


# ====================================================== host completion v2: the event and cells
def test_the_host_completion_binds_the_terminal_event_and_re_verifies_each_cell_itself(
    honest: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entropic_substrate.ladder.run import worker_completion
    from entropic_substrate.ladder.run.worker_completion import receive_worker_completion

    request, objects = honest.request, honest.objects
    key = f"{request.prefix}/complete.json"

    def canonical(row: dict[str, Any]) -> bytes:
        return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()

    def receive(store: _ObjectStore, name: str) -> dict[str, object]:
        host = tmp_path / name
        host.mkdir()
        return receive_worker_completion(
            store,
            request,
            terminal_identity(store, request, "complete"),
            host,
            expected_pod_id=None,
            max_marker_bytes=1_000_000,
        )

    event = json.loads(objects.values[key])
    assert canonical(event) == objects.values[key]  # control: the honest event is canonical
    # ---- the event is bound to this request, this pair, no authority, and canonical bytes
    forgeries = {
        "event-of-another-request": canonical(event | {"request_sha256": "0" * 64}),
        "event-of-another-pair": canonical(event | {"run_identity": "0" * 64}),
        "event-claims-an-authority": canonical(event | {"scientific_authority": "x"}),
        "event-status-differs-from-its-terminal": canonical(event | {"status": "paused"}),
        "event-bytes-are-not-canonical": json.dumps(event, sort_keys=True, indent=1).encode(),
    }
    for case, payload in forgeries.items():
        store = _ObjectStore(values=dict(objects.values))
        store.values[key] = payload
        refused(
            case,
            "worker completion identity, authority or encoding mismatch",
            lambda store=store, case=case: receive(store, f"host-{case}"),
        )
    # ---- each completed cell is re-verified by the host against ITS OWN capture/verification
    seen: list[tuple[Path, Path, ContentHash]] = []
    real = worker_completion.read_native_verification

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append((args[0], args[2], kwargs["expected_request"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_completion, "read_native_verification", spy)
    completion = receive(objects, "host-honest")
    assert completion["status"] == "probe_completion_verified"
    host = tmp_path / "host-honest"
    digest = ContentHash("sha256", request.digest)
    assert seen == [
        (host / "verification" / token, host / "capture" / token, digest)
        for token in ("donor", "recipient")
    ]

    def native_refusal(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("native verification refused")

    monkeypatch.setattr(worker_completion, "read_native_verification", native_refusal)
    refused(
        "native-verification-refuses",
        "native verification refused",
        lambda: receive(objects, "host-refused"),
    )


# ============================================ classify(): the on-disk record meets the error
def tampering_with_failure_records(patch: pytest.MonkeyPatch, rewrite: Any) -> None:
    """Intercept the runner's write of its failure record only: ``rewrite`` returns the row to
    write, or ``None`` to write nothing (disk evidence that disagrees with the raised error)."""
    from entropic_substrate.ladder.run import runner

    real = runner.write_json

    def write(*args: Any, **kwargs: Any) -> Any:
        row = args[2] if len(args) > 2 else None
        if isinstance(row, dict) and row.get("schema") == "h-probe-training-failure-v2":
            changed = rewrite(row)
            if changed is None:
                return None
            args = (*args[:2], changed, *args[3:])
        return real(*args, **kwargs)

    patch.setattr(runner, "write_json", write)


@pytest.mark.parametrize(
    ("rewrite", "message"),
    [
        pytest.param(
            lambda row: row | {"status": "execution_defect"},
            "the on-disk training failure disagrees with the raised error",
            id="record-status-disagrees-with-the-raised-class",
        ),
        pytest.param(
            lambda row: None,
            "a scientific stop left no training failure record",
            id="scientific-stop-left-no-record",
        ),
    ],
)
def test_a_scientific_stop_whose_disk_evidence_disagrees_fails_hard_not_silently(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch, rewrite: Any, message: str
) -> None:
    from entropic_substrate.core.sentence.numerics import UndefinedDirection

    request, objects, marker = pair_world(tmp_path, pair)
    Fault(monkeypatch, "warmup", 1, UndefinedDirection(2))
    tampering_with_failure_records(monkeypatch, rewrite)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 2, error  # no pair verdict was written for evidence that contradicts itself
    failure = failure_terminal(objects, request)
    assert failure["status"] == "execution_defect" and failure["error_type"] == "ValueError"
    detail = json.loads(failure_snapshot_files(objects, request)["worker/failure.json"])["detail"]
    assert detail == message
    assert f"{request.prefix}/excluded.json" not in objects.values


# ============================================================ wiring: sinks and their lifecycle
def test_a_pair_is_handed_cell_sinks_and_a_single_profile_request_is_not(
    tmp_path: Path, pair: Pair
) -> None:
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
    from entropic_substrate.ladder.run.object_worker import run_object_worker
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    request, objects, marker = pair_world(tmp_path, pair)
    workspace = tmp_path / "pair-ws"
    workspace.mkdir()
    refused(
        "pair-handed-one-sink",
        "a pair is observed per cell",
        lambda: run_object_worker(
            objects,
            marker,
            workspace,
            max_request_bytes=1_000_000,
            sink=JsonlSink(tmp_path / "one-sink.jsonl"),
        ),
    )
    assert request.prefix in marker.key and not any(workspace.iterdir())  # refused before any work
    single_root = tmp_path / "single"
    single_root.mkdir()
    single, inputs = prepared(single_root)
    store = _ObjectStore()
    single_marker = publish_probe_request(store, single, inputs)
    single_workspace = tmp_path / "single-ws"
    single_workspace.mkdir()
    refused(
        "single-handed-cell-sinks",
        "a single-profile request has no cells to give sinks to",
        lambda: run_object_worker(
            store,
            single_marker,
            single_workspace,
            max_request_bytes=1_000_000,
            cell_sinks=cast(Any, object()),
        ),
    )
    assert not any(single_workspace.iterdir())


def test_a_pair_cell_run_left_open_by_an_interrupt_is_finished_failed(
    tmp_path: Path, pair: Pair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``worker_command`` finishes every cell run its provider opened and the pair loop never
    closed (an exit that bypassed it): ``KeyboardInterrupt`` is not an ``Exception``."""
    from entropic_substrate.ladder.run import worker_command

    request, objects, marker = pair_world(tmp_path, pair)
    finishes: list[str] = []
    real_build = worker_command.build_sink_for_run

    class Recording:
        def __init__(self, sink: Any) -> None:
            self._sink = sink

        def __getattr__(self, name: str) -> Any:
            return getattr(self._sink, name)

        def finish(self, *, status: str) -> None:
            finishes.append(status)
            self._sink.finish(status=status)

    def spy(*args: Any, **kwargs: Any) -> Any:
        return Recording(real_build(*args, **kwargs))

    def interrupt(*args: Any, **kwargs: Any) -> Any:
        kwargs["cell_sinks"].begin("donor", request.donor)  # the donor's run is now open
        raise KeyboardInterrupt

    monkeypatch.setattr(worker_command, "build_sink_for_run", spy)
    monkeypatch.setattr(worker_command, "run_object_worker", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_command(tmp_path, objects, marker)
    assert finishes == ["FAILED"]


def test_a_single_profile_run_keeps_the_v1_tag_schema_and_carries_no_pair_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_sentence_worker_execution import prepared

    from entropic_substrate.ladder.run import worker_command
    from entropic_substrate.ladder.run.worker_transport import publish_probe_request

    descriptors: list[Any] = []
    real_build = worker_command.build_sink_for_run

    def spy(descriptor: Any, **kwargs: Any) -> Any:
        descriptors.append(descriptor)
        return real_build(descriptor, **kwargs)

    monkeypatch.setattr(worker_command, "build_sink_for_run", spy)
    single, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, single, inputs)
    status, error, _, _ = run_command(tmp_path, objects, marker)
    assert status == 0, error
    (descriptor,) = descriptors
    assert descriptor.tags["schema"] == "h-probe-worker-result-v1"
    assert not {"block", "pair", "cell"} & set(descriptor.tags)
    assert descriptor.run_name == f"h-worker-{single.profile.identity}"


# ============================================================================ the stop marker
def test_a_pair_stop_marker_must_be_canonical_and_another_requests_marker_is_ignored(
    pair: Pair,
) -> None:
    from entropic_substrate.ladder.run.object_worker import (
        classify_stop_marker,
        encode_stop_marker,
    )

    request = pair_request(pair)
    payload = encode_stop_marker(request, ("prediction", 2, "recipient"))
    assert classify_stop_marker(payload, request) is True  # control
    spaced = json.dumps(json.loads(payload), sort_keys=True, indent=1).encode()
    assert json.loads(spaced) == json.loads(payload) and spaced != payload
    refused(
        "non-canonical-pair-marker",
        "stop marker is not canonical JSON",
        lambda: classify_stop_marker(spaced, request),
    )
    other = pair_request(pair, nonce="2" * 64)
    foreign = encode_stop_marker(other, ("prediction", 2, "recipient"))
    assert classify_stop_marker(foreign, request) is False  # another request's: ignored


def test_the_h0_checks_script_refuses_a_pair_request_file_by_name(
    tmp_path: Path, pair: Pair
) -> None:
    """``scripts/h0_final_probe_checks.py`` reads single-profile arms; handing it a pair's request
    file is a named refusal at the read, never an attribute error further on."""
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "h0_final_probe_checks.py"
    spec = importlib.util.spec_from_file_location("h0_final_probe_checks_pair_pin", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "pair-request.json"
    path.write_bytes(pair_request(pair).payload)
    refused(
        "pair-request-file",
        "h0_final_probe_checks does not carry a pair request",
        lambda: module._request(path),  # pyright: ignore[reportPrivateUsage]
    )
