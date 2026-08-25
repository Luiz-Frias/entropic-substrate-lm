"""Real-arm verdict authority survives construction and later revalidation."""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from entropic_substrate.analysis.d1_real_arm_verdict import (
    D1AssembledRealArmVerdict,
    D1VerdictDecision,
    assemble_d1_real_arm_verdict,
)

from .test_d1_real_arm_verdict_assembly import _inputs

pytestmark = pytest.mark.unit


class _AttestationProxy:
    def __init__(self, value: object) -> None:
        self._value = value

    def require_factory_attested(self) -> None:
        pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value, name)


def _assembled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    verified, execution, reveal, adjudication = _inputs(tmp_path, monkeypatch)
    assembled = assemble_d1_real_arm_verdict(
        verified=verified,
        execution=execution,
        reveal=reveal,
        adjudication=adjudication,
        decision=D1VerdictDecision("SUPPORTED"),
    )
    return assembled, (verified, execution, reveal, adjudication)


@pytest.mark.parametrize("field", ("execution", "reveal", "adjudication"))
def test_post_mint_attestation_proxy_substitution_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    assembled, _ = _assembled(tmp_path, monkeypatch)
    object.__setattr__(assembled, field, _AttestationProxy(getattr(assembled, field)))

    with pytest.raises(TypeError, match="exact factory-attested"):
        assembled.require_factory_attested()


def test_assembled_verdict_is_noncopyable_and_detects_byte_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled, _ = _assembled(tmp_path, monkeypatch)

    with pytest.raises(TypeError, match="factory-attested"):
        D1AssembledRealArmVerdict()
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.copy(assembled)
    with pytest.raises(TypeError, match="cannot be copied"):
        copy.deepcopy(assembled)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(assembled)
    with pytest.raises(TypeError, match="factory-attested"):
        replace(assembled, canonical_bytes=b"{}")
    object.__setattr__(assembled, "canonical_bytes", b"{}")
    with pytest.raises(TypeError, match="evidence changed"):
        assembled.require_factory_attested()


@pytest.mark.parametrize("index", (1, 2, 3))
def test_pre_mint_attestation_proxies_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index: int,
) -> None:
    assembled, values = _assembled(tmp_path, monkeypatch)
    mutable = list(values)
    mutable[index] = _AttestationProxy(mutable[index])

    with pytest.raises(TypeError, match="exact attested evidence"):
        assemble_d1_real_arm_verdict(
            verified=mutable[0],  # type: ignore[arg-type]
            execution=mutable[1],  # type: ignore[arg-type]
            reveal=mutable[2],  # type: ignore[arg-type]
            adjudication=mutable[3],  # type: ignore[arg-type]
            decision=D1VerdictDecision(assembled.verdict.verdict),
        )
