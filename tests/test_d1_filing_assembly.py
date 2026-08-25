"""Production D1 narrative and exact fourteen-role filing assembly."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import cast

import pytest

from entropic_substrate.evidence.d1_filing import (
    REQUIRED_D1_FILING_ARTIFACTS,
    D1FilingValidationError,
)
from entropic_substrate.evidence.d1_filing_assembly import (
    D1FilingArtifactMaterial,
    assemble_d1_filing_manifest,
    assemble_d1_narrative,
)

pytestmark = pytest.mark.unit
_MaterialTuple = tuple[D1FilingArtifactMaterial, ...]


def _materials() -> tuple[D1FilingArtifactMaterial, ...]:
    machine = b'{"rung":"d1","schema_version":"d1-verdict/v2"}'
    narrative = assemble_d1_narrative(
        body=b"# D1 stabilized latent verdict\n\nRegistered result.\n",
        machine_verdict=machine,
    )
    return tuple(
        D1FilingArtifactMaterial(
            role=role,
            path=f"filing/{role}.bin",
            payload=(
                narrative
                if role == "narrative"
                else machine
                if role == "machine_verdict"
                else f"{role}\n".encode()
            ),
        )
        for role in REQUIRED_D1_FILING_ARTIFACTS
    )


def test_narrative_contains_exactly_one_machine_verdict_marker() -> None:
    machine = b'{"schema_version":"d1-verdict/v2"}'

    narrative = assemble_d1_narrative(
        body=b"# D1 verdict\n\nMeasured result.\n",
        machine_verdict=machine,
    )

    machine_digest = hashlib.sha256(machine).hexdigest()
    marker = f"<!-- d1-machine-verdict-sha256:{machine_digest} -->".encode()
    assert narrative.endswith(marker + b"\n")
    assert narrative.count(marker) == 1


@pytest.mark.parametrize(
    ("body", "machine", "message"),
    [
        (b"", b"{}", "body"),
        (b"# verdict\n", b"", "machine verdict"),
        (b"\xff", b"{}", "UTF-8"),
        (
            b"# verdict\n<!-- d1-machine-verdict-sha256:already -->\n",
            b"{}",
            "marker",
        ),
    ],
)
def test_narrative_refuses_ambiguous_or_nontext_inputs(
    body: bytes, machine: bytes, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        assemble_d1_narrative(body=body, machine_verdict=machine)


def test_manifest_assembles_exact_roles_and_measured_byte_identities() -> None:
    materials = _materials()

    manifest = assemble_d1_filing_manifest(materials)

    assert tuple(manifest.artifacts) == REQUIRED_D1_FILING_ARTIFACTS
    for material in materials:
        artifact = manifest.artifacts[material.role]
        expected_sha256 = hashlib.sha256(material.payload).hexdigest()
        assert artifact.path == material.path
        assert artifact.size == len(material.payload)
        assert artifact.sha256.digest == expected_sha256
        expected_md5 = hashlib.md5(
            material.payload,
            usedforsecurity=False,
        ).hexdigest()
        assert artifact.md5.digest == expected_md5


def test_manifest_refuses_narrative_to_machine_substitution() -> None:
    materials = _materials()
    replacement = replace(
        materials[1],
        payload=b'{"rung":"different","schema_version":"d1-verdict/v2"}',
    )
    substituted = (materials[0], replacement, *materials[2:])

    with pytest.raises(ValueError, match=r"narrative.*machine verdict"):
        assemble_d1_filing_manifest(substituted)


def test_manifest_refuses_matching_marker_inside_markdown_code_block() -> None:
    materials = _materials()
    machine_digest = hashlib.sha256(materials[1].payload).hexdigest()
    marker = f"<!-- d1-machine-verdict-sha256:{machine_digest} -->".encode()
    code_block = b"# D1 verdict\n\n```text\n" + marker + b"\n```\n"
    embedded = (replace(materials[0], payload=code_block), *materials[1:])

    with pytest.raises(ValueError, match=r"narrative.*machine verdict"):
        assemble_d1_filing_manifest(embedded)


def test_manifest_refuses_missing_duplicate_and_unsafe_materials() -> None:
    materials = _materials()
    mutable_materials: object = list(materials)
    with pytest.raises(TypeError, match="typed tuple"):
        assemble_d1_filing_manifest(cast(_MaterialTuple, mutable_materials))
    with pytest.raises(TypeError, match="typed tuple"):
        assemble_d1_filing_manifest(
            cast(
                tuple[D1FilingArtifactMaterial, ...],
                (*materials[:-1], object()),
            )
        )
    with pytest.raises(ValueError, match="exactly fourteen"):
        assemble_d1_filing_manifest(materials[:-1])
    with pytest.raises(ValueError, match="duplicate role"):
        assemble_d1_filing_manifest((*materials, materials[0]))
    with pytest.raises(D1FilingValidationError, match="unsafe_path"):
        assemble_d1_filing_manifest(
            (
                replace(materials[0], path="../outside.bin"),
                *materials[1:],
            )
        )
    with pytest.raises(D1FilingValidationError, match="duplicate_path"):
        assemble_d1_filing_manifest(
            (
                materials[0],
                replace(materials[1], path=materials[0].path),
                *materials[2:],
            )
        )


def test_material_requires_nominal_role_path_and_bytes() -> None:
    invalid_role: object = 1
    role = cast(str, invalid_role)
    with pytest.raises(TypeError, match="role must be a string"):
        D1FilingArtifactMaterial(role, "filing/value.bin", b"value")
    with pytest.raises(ValueError, match="role"):
        D1FilingArtifactMaterial("not-a-role", "filing/value.bin", b"value")
    invalid_path: object = 1
    path = cast(str, invalid_path)
    with pytest.raises(TypeError, match="path must be a string"):
        D1FilingArtifactMaterial("narrative", path, b"value")
    with pytest.raises(ValueError, match="path"):
        D1FilingArtifactMaterial("narrative", "", b"value")
    with pytest.raises(ValueError, match="trimmed"):
        D1FilingArtifactMaterial("narrative", " filing/value.bin", b"value")
    invalid_payload = cast(bytes, "value")
    valid_path = "filing/value.bin"
    with pytest.raises(TypeError, match="payload"):
        D1FilingArtifactMaterial("narrative", valid_path, invalid_payload)
    with pytest.raises(ValueError, match="payload must be nonempty"):
        D1FilingArtifactMaterial("narrative", "filing/value.bin", b"")


@pytest.mark.parametrize(
    ("body", "machine"),
    [("# verdict", b"{}"), (b"# verdict", {})],
)
def test_narrative_requires_exact_byte_inputs(
    body: object,
    machine: object,
) -> None:
    with pytest.raises(TypeError, match="exact bytes"):
        assemble_d1_narrative(
            body=cast(bytes, body),
            machine_verdict=cast(bytes, machine),
        )
