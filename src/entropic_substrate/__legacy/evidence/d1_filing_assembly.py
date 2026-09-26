"""Byte-owned assembly of the D1 narrative and exact filing manifest."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1FilingManifest,
)

__all__ = [
    "D1FilingArtifactMaterial",
    "assemble_d1_filing_manifest",
    "assemble_d1_narrative",
]

_NARRATIVE_MARKER_PREFIX = b"<!-- d1-machine-verdict-sha256:"
_ROLES: frozenset[str] = frozenset(REQUIRED_D1_FILING_ARTIFACTS)
_PATH_ERROR = "D1 filing material path must be nonempty and trimmed"
_MARKER_ERROR = "D1 narrative body contains a machine-verdict marker"
_MATERIAL_TUPLE_ERROR = "D1 filing assembly requires a typed tuple"
_ROLE_SET_ERROR = "D1 filing assembly requires exactly thirteen roles"
_NARRATIVE_BINDING_ERROR = "D1 narrative must bind machine verdict"


def _machine_marker(machine_verdict: bytes) -> bytes:
    return (
        _NARRATIVE_MARKER_PREFIX
        + hashlib.sha256(machine_verdict).hexdigest().encode("ascii")
        + b" -->"
    )


@dataclass(frozen=True, slots=True)
class D1FilingArtifactMaterial:
    """One named filed payload before its content identities are measured."""

    role: str
    path: str
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.role) is not str:
            raise TypeError("D1 filing material role must be a string")
        if self.role not in _ROLES:
            raise ValueError("D1 filing material role is not registered")
        if type(self.path) is not str:
            raise TypeError("D1 filing material path must be a string")
        if not self.path or self.path.strip() != self.path:
            raise ValueError(_PATH_ERROR)
        if type(self.payload) is not bytes:
            raise TypeError("D1 filing material payload must be exact bytes")
        if not self.payload:
            raise ValueError("D1 filing material payload must be nonempty")


def assemble_d1_narrative(*, body: bytes, machine_verdict: bytes) -> bytes:
    """Append one canonical machine-verdict digest marker to Markdown."""
    if type(body) is not bytes:
        raise TypeError("D1 narrative body must be exact bytes")
    if not body:
        raise ValueError("D1 narrative body must be nonempty")
    if type(machine_verdict) is not bytes:
        raise TypeError("D1 narrative machine verdict must be exact bytes")
    if not machine_verdict:
        raise ValueError("D1 narrative machine verdict must be nonempty")
    try:
        body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("D1 narrative body must be UTF-8") from error
    if _NARRATIVE_MARKER_PREFIX in body:
        raise ValueError(_MARKER_ERROR)
    marker = _machine_marker(machine_verdict)
    return body.rstrip(b"\n") + b"\n\n" + marker + b"\n"


def _all_materials(values: tuple[object, ...]) -> bool:
    return all(type(value) is D1FilingArtifactMaterial for value in values)


def assemble_d1_filing_manifest(
    materials: tuple[D1FilingArtifactMaterial, ...],
) -> D1FilingManifest:
    """Measure exact bytes and construct the thirteen-role manifest."""
    if type(materials) is not tuple:
        raise TypeError(_MATERIAL_TUPLE_ERROR)
    if not _all_materials(materials):
        raise TypeError(_MATERIAL_TUPLE_ERROR)
    roles = tuple(material.role for material in materials)
    if len(set(roles)) != len(roles):
        raise ValueError("D1 filing assembly contains a duplicate role")
    exact_count = len(materials) == len(REQUIRED_D1_FILING_ARTIFACTS)
    exact_roles = frozenset(roles) == _ROLES
    if not exact_count or not exact_roles:
        raise ValueError(_ROLE_SET_ERROR)
    by_role = {material.role: material for material in materials}
    narrative = by_role["narrative"].payload
    expected_marker = _machine_marker(by_role["machine_verdict"].payload)
    one_marker = narrative.count(_NARRATIVE_MARKER_PREFIX) == 1
    correct_marker = narrative.count(expected_marker) == 1
    canonical_suffix = b"\n\n" + expected_marker + b"\n"
    terminal_marker = narrative.endswith(canonical_suffix)
    if not (one_marker and correct_marker and terminal_marker):
        raise ValueError(_NARRATIVE_BINDING_ERROR)
    artifacts = {
        role: {
            "md5": hashlib.md5(
                by_role[role].payload,
                usedforsecurity=False,
            ).hexdigest(),
            "path": by_role[role].path,
            "sha256": hashlib.sha256(by_role[role].payload).hexdigest(),
            "size": len(by_role[role].payload),
        }
        for role in REQUIRED_D1_FILING_ARTIFACTS
    }
    return D1FilingManifest.from_mapping(
        {
            "artifacts": artifacts,
            "rung": "d1",
            "schema_version": "d1-filing-manifest/v2",
        }
    )
