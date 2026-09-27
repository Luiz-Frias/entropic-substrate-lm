"""Strict structural decoder for canonical aliased D1 machine analysis output."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.ladder.analyze.d1_machine_analysis_codec import (
    D1MachineAnalysisCodecError,
    decode_d1_machine_analysis_declaration,
)

from .test_d1_aliased_analysis import _packet

pytestmark = pytest.mark.unit


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_generated_machine_analysis_decodes_with_full_structural_shape() -> None:
    packet = _packet()
    payload = packet.canonical_json.encode("utf-8")

    declaration = decode_d1_machine_analysis_declaration(payload)

    assert declaration == packet
    assert declaration.analysis_digest == packet.analysis_digest
    assert declaration.manifest["schema_version"] == "d1-aliased-machine-analysis-v2"


@pytest.mark.parametrize(
    "payload",
    [
        b"{}",
        b'{"schema_version":"d1-aliased-machine-analysis-v2","schema_version":"x"}',
        b'{"included_seeds":[NaN]}',
        b" {}",
        b"[]",
    ],
)
def test_decoder_rejects_empty_duplicate_nonfinite_noncanonical_or_nonobject_output(
    payload: bytes,
) -> None:
    with pytest.raises(D1MachineAnalysisCodecError):
        decode_d1_machine_analysis_declaration(payload)


def test_top_and_nested_field_sets_are_exact() -> None:
    manifest = _packet().manifest
    manifest["invented"] = True
    with pytest.raises(D1MachineAnalysisCodecError, match="fields"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))

    manifest = _packet().manifest
    manifest["contrasts"][0]["invented"] = True  # type: ignore[index]
    with pytest.raises(D1MachineAnalysisCodecError, match="fields"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("arm_alias", "sg"),
        ("logical_run_id", "R001"),
        ("construction_digest", "0" * 64),
        ("path", "results/captured/run.jsonl"),
        ("nonce", "0" * 64),
        ("private_reveal", "secret"),
    ],
)
def test_recursive_blindness_scan_rejects_true_labels_and_construction_fields(
    key: str, value: str
) -> None:
    manifest = _packet().manifest
    manifest["diagnostics"][0][key] = value  # type: ignore[index]
    with pytest.raises(D1MachineAnalysisCodecError, match="blind"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))


def test_seed_partition_endpoint_and_diagnostic_grid_must_be_complete() -> None:
    manifest = _packet().manifest
    manifest["excluded_seeds"] = [manifest["included_seeds"][0]]  # type: ignore[index]
    with pytest.raises(D1MachineAnalysisCodecError, match="seed"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))

    manifest = _packet().manifest
    manifest["endpoint"]["values_by_alias"].pop("A")  # type: ignore[index,union-attr]
    with pytest.raises(D1MachineAnalysisCodecError, match="endpoint"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))

    manifest = _packet().manifest
    manifest["diagnostics"].pop()  # type: ignore[union-attr]
    with pytest.raises(D1MachineAnalysisCodecError, match="diagnostic"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update({"contrast_alias": "C99"}),
        lambda row: row.update({"n": True}),
        lambda row: row.update({"df": 99}),
        lambda row: row["paired_estimates"].pop(),
        lambda row: row["margins"].update({"practical": True}),
        lambda row: row["terms"].append({"arm_alias": "A", "coefficient": 0.0}),
    ],
)
def test_contrast_statistics_alias_order_and_terms_are_structurally_consistent(
    mutation: object,
) -> None:
    manifest = _packet().manifest
    mutation(manifest["contrasts"][0])  # type: ignore[index,operator]
    with pytest.raises(D1MachineAnalysisCodecError, match="contrast"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))


def test_effect_clause_and_quality_target_shapes_are_enforced() -> None:
    manifest = _packet().manifest
    manifest["factorial_effects"][0]["deciding"] = True  # type: ignore[index]
    with pytest.raises(D1MachineAnalysisCodecError, match="effect"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))

    manifest = _packet().manifest
    manifest["clauses_by_alias"]["B"].pop("language")  # type: ignore[index,union-attr]
    with pytest.raises(D1MachineAnalysisCodecError, match="clause"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))

    manifest = _packet().manifest
    manifest["diagnostics"][0]["quality_target"]["reached"] = True  # type: ignore[index]
    with pytest.raises(D1MachineAnalysisCodecError, match="quality target"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metric", ""),
        ("operator", "SIDEWAYS"),
        ("operator", []),
        ("operator", {}),
        ("threshold", "0.79"),
    ],
)
def test_quality_target_policy_fields_are_strict(field: str, value: object) -> None:
    manifest = _packet().manifest
    manifest["diagnostics"][0]["quality_target"][field] = value  # type: ignore[index]

    with pytest.raises(D1MachineAnalysisCodecError, match="quality target"):
        decode_d1_machine_analysis_declaration(_bytes(manifest))
