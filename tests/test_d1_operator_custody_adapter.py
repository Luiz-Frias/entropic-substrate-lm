"""Production operator custody publishes every controller boundary durably."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from entropic_substrate.d1_operator_custody_adapter import D1OperatorCustodyAdapter
from entropic_substrate.d1_operator_ports import D1RunPodAuthReceipt

from .test_d1_finalized_cell_custody_values import _attest, _finalized_case

pytestmark = pytest.mark.unit


def _adapter(tmp_path: Path) -> tuple[D1OperatorCustodyAdapter, Path, Path]:
    root = tmp_path / "operator"
    finalized = tmp_path / "finalized"
    root.mkdir()
    finalized.mkdir()
    return D1OperatorCustodyAdapter(root=root, finalized_cell_root=finalized), root, finalized


def test_adapter_publishes_every_operator_boundary_and_finalized_discovery(
    tmp_path: Path,
) -> None:
    case = _finalized_case(tmp_path / "case")
    record = _attest(case)
    adapter, root, finalized = _adapter(tmp_path)

    artifacts = (
        adapter.persist_precreate(case.intent, case.ledger, D1RunPodAuthReceipt("account-1")),
        adapter.persist_attempt_ledger(case.ledger),
        adapter.persist_run_start(case.starts[0]),
        adapter.persist_finalization(case.finalization),
        adapter.persist_problem(case.intent.plan.logical_run_id, "token=secret-value"),
    )
    finalized_artifact = adapter.persist_finalized_cell(record)

    assert len({item.content_hash.digest for item in artifacts}) == len(artifacts)
    for artifact in artifacts:
        payload = Path(artifact.path).read_bytes()
        assert artifact.path.startswith(str(root))
        assert hashlib.sha256(payload).hexdigest() == artifact.content_hash.digest
        json.loads(payload)
    problem = Path(artifacts[-1].path).read_text(encoding="utf-8")
    assert "secret-value" not in problem
    assert hashlib.sha256(b"token=secret-value").hexdigest() in problem
    assert finalized_artifact.path.startswith(str(finalized))
    discovered = adapter.discover_finalized_cells()
    assert discovered == ((record, finalized_artifact),)


def test_adapter_is_content_idempotent_and_close_is_terminal(tmp_path: Path) -> None:
    case = _finalized_case(tmp_path / "case")
    adapter, _root, _finalized = _adapter(tmp_path)

    first = adapter.persist_attempt_ledger(case.ledger)
    second = adapter.persist_attempt_ledger(case.ledger)

    assert first == second
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.persist_attempt_ledger(case.ledger)


def test_adapter_refuses_wrong_nominal_values_before_publication(tmp_path: Path) -> None:
    adapter, root, _finalized = _adapter(tmp_path)

    with pytest.raises(TypeError):
        adapter.persist_attempt_ledger(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        adapter.persist_finalized_cell(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        adapter.persist_problem("", "failure")
    assert list(root.iterdir()) == []
