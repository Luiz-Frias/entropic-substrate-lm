"""FOR-1694 (h1-l4c): the identity bindings of a PAIR's events, each pinned by a forged event.

``read_probe_event`` binds a pair event to its pair: a checkpoint to its CELL (name and profile
identity), a terminal to the pair id, a failure to the pair or to the failing cell. Each forgery
below changes exactly one of those bindings in an event the real worker published, so a reader
that stopped checking it would admit the forgery. The store is a copy: the honest event is the
control in every test.

One binding has no independent tooth, by construction rather than by omission: the checkpoint
count's upper bound is read from the CELL's own schedule, and ``require_pair`` refuses a pair
whose manifests differ beyond their name and spine carrier, so both cells share one schedule and
taking it from the donor instead of the cell changes no admitted event.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_codec_handoff import Pair, make_pair
from tests.test_h_pair_request import Fault, pair_world, run_command

from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.worker_request import PairProbeRequest

pytestmark = pytest.mark.unit

MAX = 10_000_000
World = tuple[PairProbeRequest, dict[str, bytes]]


def _run(base: Path, fault: bool) -> World:
    pair: Pair = make_pair(base)
    request, objects, marker = pair_world(base, pair)
    if fault:
        with pytest.MonkeyPatch.context() as patch:
            Fault(patch, "warmup", 1, RuntimeError("injected reader refusal"))
            status, _, _, _ = run_command(base, objects, marker)
        assert status == 2
    else:
        status, error, _, _ = run_command(base, objects, marker)
        assert status == 0, error
    return request, dict(objects.values)


@pytest.fixture(scope="module")
def completed(tmp_path_factory: pytest.TempPathFactory) -> World:
    return _run(tmp_path_factory.mktemp("pair-events-complete"), fault=False)


@pytest.fixture(scope="module")
def failed(tmp_path_factory: pytest.TempPathFactory) -> World:
    return _run(tmp_path_factory.mktemp("pair-events-failure"), fault=True)


def _encode(row: dict[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _read(world: World, name: str, mutate: Any = None, *, key: str | None = None) -> Any:
    """Read event ``name`` from a COPY of the store; ``mutate`` forges its row first, and ``key``
    stores the (forged) row under another event's name."""
    request, values = world
    store = _ObjectStore(values=dict(values))
    source = f"{request.prefix}/{key or name}.json"
    row = json.loads(store.values[source])
    if mutate is not None:
        row = mutate(row)
    store.values[f"{request.prefix}/{name}.json"] = _encode(row)
    return read_probe_event(store, request, name, expected_pod_id=None, max_bytes=MAX)


def test_the_honest_pair_events_are_read(completed: World, failed: World) -> None:
    for name in ("checkpoint-donor-00000002", "checkpoint-recipient-00000006", "complete"):
        assert _read(completed, name) is not None
    assert _read(failed, "failure") is not None


def test_a_checkpoint_is_bound_to_its_cells_profile_identity(completed: World) -> None:
    request, _ = completed
    forged = _read  # one cell's event dressed as the other cell's
    with pytest.raises(ValueError, match="identity, authority or encoding mismatch"):
        forged(
            completed,
            "checkpoint-donor-00000002",
            lambda row: {**row, "run_identity": request.recipient.identity},
        )
    with pytest.raises(ValueError, match="identity, authority or encoding mismatch"):
        forged(
            completed,
            "checkpoint-recipient-00000006",
            lambda row: {**row, "run_identity": request.run_identity},
        )


def test_a_checkpoint_naming_a_cell_outside_its_pair_is_refused_before_any_read(
    completed: World,
) -> None:
    request, values = completed
    with pytest.raises(ValueError, match="names a cell outside its pair"):
        read_probe_event(
            _ObjectStore(values=dict(values)),
            request,
            "checkpoint-observer-00000002",
            expected_pod_id=None,
            max_bytes=MAX,
        )


def test_a_checkpoint_count_must_agree_with_its_event_name_and_its_cells_schedule(
    completed: World,
) -> None:
    with pytest.raises(ValueError, match="checkpoint count differs"):
        _read(
            completed,
            "checkpoint-donor-00000002",
            lambda row: {**row, "successful_optimizer_steps": 3},
        )
    # the schedule bound: a seal past the cell's total (6) and a zero are not checkpoints
    for step, name in ((7, "checkpoint-donor-00000007"), (0, "checkpoint-donor-00000000")):
        with pytest.raises(ValueError, match="checkpoint count differs"):
            _read(
                completed,
                name,
                lambda row, step=step: {**row, "successful_optimizer_steps": step},
                key="checkpoint-donor-00000002",
            )
    # the control at the bound: the cell's own last seal is admitted
    assert _read(completed, "checkpoint-donor-00000006") is not None


def test_a_terminal_event_is_bound_to_the_pair_id_and_not_to_a_member_cell(
    completed: World,
) -> None:
    request, _ = completed
    for identity in (request.donor.identity, request.recipient.identity, "f" * 64):
        with pytest.raises(ValueError, match="identity, authority or encoding mismatch"):
            _read(completed, "complete", lambda row, i=identity: {**row, "run_identity": i})


def test_a_failure_is_bound_to_the_pair_or_to_one_of_its_cells_and_to_nothing_else(
    failed: World,
) -> None:
    request, _ = failed
    for identity in (request.run_identity, request.donor.identity, request.recipient.identity):
        assert _read(failed, "failure", lambda row, i=identity: {**row, "run_identity": i})
    with pytest.raises(ValueError, match="identity, authority or encoding mismatch"):
        _read(failed, "failure", lambda row: {**row, "run_identity": "f" * 64})
