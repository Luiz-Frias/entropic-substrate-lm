"""Strict source-manifest declaration for the authoritative D1 reader code set."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.analysis.d1_external_analysis_control import (
    D1_COMPLETE_READER_IDENTITY,
    D1_READER_SOURCE_MODULES,
    D1_TERMINAL_DEFECT_READER_IDENTITY,
    D1ExternalAnalysisControlError,
    derive_d1_analysis_code_digest,
    parse_d1_reader_source_manifest,
)

pytestmark = pytest.mark.unit


def _modules() -> list[dict[str, object]]:
    return [
        {
            "identity": identity,
            "path": identity.replace(".", "/") + ".py",
            "sha256": f"{index + 1:064x}",
        }
        for index, identity in enumerate(D1_READER_SOURCE_MODULES)
    ]


def _manifest() -> dict[str, object]:
    modules = _modules()
    return {
        "analysis_code_digest": derive_d1_analysis_code_digest(modules),
        "complete_reader": D1_COMPLETE_READER_IDENTITY,
        "modules": modules,
        "schema_version": "d1-reader-source-manifest-v1",
        "terminal_defect_reader": D1_TERMINAL_DEFECT_READER_IDENTITY,
    }


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_exact_ordered_reader_source_set_and_derived_digest_round_trip() -> None:
    payload = _bytes(_manifest())

    control = parse_d1_reader_source_manifest(payload)

    assert control.canonical_bytes == payload
    assert tuple(item.identity for item in control.modules) == D1_READER_SOURCE_MODULES
    assert len({item.path for item in control.modules}) == len(control.modules)
    assert control.complete_reader == D1_COMPLETE_READER_IDENTITY
    assert control.terminal_defect_reader == D1_TERMINAL_DEFECT_READER_IDENTITY
    assert control.analysis_code_digest == derive_d1_analysis_code_digest(_modules())
    assert {
        "entropic_substrate.telemetry",
        "entropic_substrate.evidence.d1_telemetry_query",
        "entropic_substrate.evidence.d1_terminal_defect_query",
    } <= set(D1_READER_SOURCE_MODULES)


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered", "duplicate"])
def test_reader_module_identity_set_and_order_are_not_caller_selectable(mutation: str) -> None:
    manifest = _manifest()
    modules = manifest["modules"]
    assert isinstance(modules, list)
    if mutation == "missing":
        modules.pop()
    elif mutation == "extra":
        modules.append(
            {
                "identity": "entropic_substrate.invented",
                "path": "entropic_substrate/invented.py",
                "sha256": "f" * 64,
            }
        )
    elif mutation == "reordered":
        modules[0], modules[1] = modules[1], modules[0]
    else:
        modules[1] = dict(modules[0])
    manifest["analysis_code_digest"] = derive_d1_analysis_code_digest(modules)

    with pytest.raises(D1ExternalAnalysisControlError, match="module identities"):
        parse_d1_reader_source_manifest(_bytes(manifest))


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/absolute.py", "path"),
        ("../escape.py", "path"),
        ("entropic_substrate\\telemetry.py", "path"),
        ("entropic_substrate/other.py", "path"),
    ],
)
def test_reader_paths_are_normalized_unique_and_canonical_for_identity(
    path: str, message: str
) -> None:
    manifest = _manifest()
    modules = manifest["modules"]
    assert isinstance(modules, list)
    modules[0]["path"] = path
    manifest["analysis_code_digest"] = derive_d1_analysis_code_digest(modules)

    with pytest.raises(D1ExternalAnalysisControlError, match=message):
        parse_d1_reader_source_manifest(_bytes(manifest))


@pytest.mark.parametrize("sha256", ["a" * 63, "A" * 64, "g" * 64, True])
def test_each_reader_source_hash_is_exact_lowercase_sha256(sha256: object) -> None:
    manifest = _manifest()
    modules = manifest["modules"]
    assert isinstance(modules, list)
    modules[0]["sha256"] = sha256

    with pytest.raises(D1ExternalAnalysisControlError, match="sha256"):
        parse_d1_reader_source_manifest(_bytes(manifest))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("complete_reader", "entropic_substrate.telemetry.other"),
        ("terminal_defect_reader", "entropic_substrate.telemetry.other"),
        ("analysis_code_digest", "0" * 64),
    ],
)
def test_reader_callable_or_derived_analysis_digest_drift_is_rejected(
    field: str, replacement: str
) -> None:
    manifest = _manifest()
    manifest[field] = replacement

    with pytest.raises(D1ExternalAnalysisControlError):
        parse_d1_reader_source_manifest(_bytes(manifest))


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":"d1-reader-source-manifest-v1","schema_version":"x"}',
        b'{"analysis_code_digest":NaN}',
        b" {}",
        b"{}\n",
        b"[]",
    ],
)
def test_reader_manifest_rejects_duplicate_nonfinite_noncanonical_or_nonobject_bytes(
    payload: bytes,
) -> None:
    with pytest.raises(D1ExternalAnalysisControlError):
        parse_d1_reader_source_manifest(payload)
