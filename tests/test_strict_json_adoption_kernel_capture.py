"""p14-j-kernel-capture: kernel and ladder/capture codecs adopt kernel.codec.strict_json.

Four groups, one per ticket facet (FOR-1572, FOR-1577, FOR-1613):

1. adoption -- an AST walk finds no local ``_pairs``-style strict-JSON hook
   definition left under ``kernel`` or ``ladder/capture`` (the identity-bearing
   ``ladder/capture/capture.py`` is exempt, ruling 5).
2. codec boundary -- every codec raises its OWN exception type for duplicate
   keys, non-finite constants, overflow floats and malformed JSON, because
   ``fail=`` is wired to the codec's own error channel.
3. exact-key sites -- the kernel verdict parsers and the capture telemetry
   payload helpers keep their error codes while delegating to ``exact_keys``
   and ``require_lowercase_sha256``.
4. SHA boundary -- every repointed digest site refuses a mixed-case digest
   with that site's own exception type.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.ladder.capture.reader.d1_journal_snapshot as snapshot_module
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import (
    ResourceProvenanceCodecGap,
    parse_run_start_line,
)
from entropic_substrate.kernel.verdict.d1_evaluation_values import parse_d1_cosine
from entropic_substrate.kernel.verdict.d1_parsing import object_exact, object_optional
from entropic_substrate.kernel.verdict.d1_statistical_gaps import parse_endpoint_gap
from entropic_substrate.kernel.verdict.d1_types import D1VerdictValidationError
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_codec import (
    D1CaptureAggregateHydrationError,
    hydrate_d1_capture_aggregate,
    parse_d1_capture_aggregate_declaration,
)
from entropic_substrate.ladder.capture.custody.d1_capture_declaration_codec import (
    D1CaptureDeclarationCodecError,
    parse_artifact_identity,
)
from entropic_substrate.ladder.capture.custody.d1_capture_preparation_custody import (
    D1CampaignCaptureAuthority,
    D1CapturePreparationCustodyError,
    capture_preparation_custody_from_preparation,
    parse_d1_capture_preparation_custody,
)
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import (
    D1LifecycleValueError,
    parse_lifecycle_event_line,
)
from entropic_substrate.ladder.capture.records.d1_calibration_records import (
    D1BaselineCalibrationAttestation,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalError,
    parse_event_line,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_payloads import (
    exact_fields,
    parse_baseline_event,
    parse_run_start_event,
    require_sha256,
)
from entropic_substrate.ladder.capture.records.d1_telemetry_values import D1TelemetryGap
from entropic_substrate.ladder.capture.resource.resource_ledger_codec import (
    ResourceLedgerCodecGap,
    parse_resource_ledger_state_line,
)
from entropic_substrate.ladder.capture.resource.resource_work_codec import (
    ResourceWorkCodecGap,
    parse_resource_snapshot_line,
)

from .d1_telemetry_fixtures import replace_payload, valid_events, write_events
from .test_d1_capture_preparation_custody import _authority, _case

pytestmark = pytest.mark.unit

_SRC = Path(__file__).resolve().parents[1] / "src" / "entropic_substrate"
_SCANNED = (_SRC / "kernel", _SRC / "ladder" / "capture")
_EXEMPT = frozenset({"ladder/capture/capture.py"})
_FORBIDDEN = frozenset({"_pairs", "_strict_pairs", "_nonfinite", "_reject_constant"})

_DIGEST = "1" * 64
_MIXED = "A" + "1" * 63
_SHA_MESSAGE = "must be a 64-character hexadecimal SHA-256 digest"


# --------------------------------------------------------------------------
# 1. adoption: no local strict-JSON hook definitions


def _forbidden_definitions(source: str) -> list[str]:
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _FORBIDDEN
    ]


def test_the_ast_walk_detects_a_local_hook_definition_at_any_depth() -> None:
    nested = "def outer():\n    def _reject_constant(value):\n        return 0\n"
    assert _forbidden_definitions(nested) == ["_reject_constant"]
    assert _forbidden_definitions("def parse():\n    return 1\n") == []


def test_kernel_and_capture_define_no_local_strict_json_hooks() -> None:
    files = [path for root in _SCANNED for path in sorted(root.rglob("*.py"))]
    assert len(files) > 50, "scan is misrooted: positive control on the file count failed"
    assert _SRC / "ladder" / "capture" / "capture.py" in files, (
        "exempt file must be scanned-then-skipped"
    )
    found = {
        rel: names
        for path in files
        if (rel := path.relative_to(_SRC).as_posix()) not in _EXEMPT
        if (names := _forbidden_definitions(path.read_text(encoding="utf-8")))
    }
    assert found == {}


# --------------------------------------------------------------------------
# 2. codec boundary: strict_loads(fail=<codec channel>) keeps each exception type

_DUPLICATE = '{"event":"x","event":"y"}'
_NAN = '{"event":NaN}'
_NEG_INFINITY = '{"event":-Infinity}'
_OVERFLOW = '{"event":1e999}'

_TEXT_CODECS: list[tuple[str, Callable[[str], object], type[Exception]]] = [
    ("provenance", parse_run_start_line, ResourceProvenanceCodecGap),
    ("ledger", parse_resource_ledger_state_line, ResourceLedgerCodecGap),
    ("work", parse_resource_snapshot_line, ResourceWorkCodecGap),
    ("aggregate", parse_d1_capture_aggregate_declaration, D1CaptureAggregateHydrationError),
    ("declaration", parse_artifact_identity, D1CaptureDeclarationCodecError),
    ("preparation", parse_d1_capture_preparation_custody, D1CapturePreparationCustodyError),
]


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        (_DUPLICATE, "duplicate key: 'event'"),
        (_NAN, "non-finite constant: 'NaN'"),
        (_NEG_INFINITY, "non-finite constant: '-Infinity'"),
        (_OVERFLOW, "non-finite float literal: '1e999'"),
    ],
)
@pytest.mark.parametrize(
    ("parse", "exception"),
    [pytest.param(parse, exc, id=name) for name, parse, exc in _TEXT_CODECS],
)
def test_text_codecs_refuse_strict_json_violations_with_their_own_error(
    parse: Callable[[str], object],
    exception: type[Exception],
    payload: str,
    needle: str,
) -> None:
    with pytest.raises(exception) as caught:
        parse(payload)
    assert needle in str(caught.value)


@pytest.mark.parametrize(
    ("line", "code", "needle"),
    [
        (b'{"event_seq":0,"event_seq":0}\n', "duplicate_field", "duplicate key: 'event_seq'"),
        (b'{"event_seq":NaN}\n', "nonfinite_json", "non-finite constant: 'NaN'"),
        (b'{"event_seq":Infinity}\n', "nonfinite_json", "non-finite constant: 'Infinity'"),
        (b'{"event_seq":1e999}\n', "nonfinite_json", "non-finite float literal: '1e999'"),
    ],
)
@pytest.mark.parametrize(
    ("parse", "exception"),
    [
        pytest.param(parse_lifecycle_event_line, D1LifecycleValueError, id="lifecycle"),
        pytest.param(parse_event_line, D1JournalError, id="journal"),
    ],
)
def test_line_codecs_keep_their_error_code_for_strict_json_violations(
    parse: Callable[[bytes], object],
    exception: type[Exception],
    line: bytes,
    code: str,
    needle: str,
) -> None:
    with pytest.raises(exception) as caught:
        parse(line)
    assert getattr(caught.value, "code") == code  # noqa: B009
    assert needle in str(caught.value)


@pytest.mark.parametrize(
    ("parse", "exception"),
    [pytest.param(parse, exc, id=name) for name, parse, exc in _TEXT_CODECS],
)
def test_decoder_level_errors_are_wrapped_at_each_text_codec_boundary(
    parse: Callable[[str], object], exception: type[Exception]
) -> None:
    with pytest.raises(exception, match="invalid JSON") as caught:
        parse("{")
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


# --------------------------------------------------------------------------
# 3. exact-key sites keep their error codes

_FIELDS = frozenset({"a", "b"})


@pytest.mark.parametrize(
    ("value", "code", "needle"),
    [
        ([], "invalid_type", "must be an object"),
        ({1: "x"}, "invalid_field_name", "must be strings"),
        ({"a": 1}, "missing_field", "missing required field(s): b"),
        ({"a": 1, "b": 2, "c": 3, "d": 4}, "unknown_field", "unknown field(s): c, d"),
    ],
)
def test_object_exact_refuses_with_its_verdict_codes(value: object, code: str, needle: str) -> None:
    with pytest.raises(D1VerdictValidationError) as caught:
        object_exact(value, path="x", fields=_FIELDS)
    assert caught.value.code == code
    assert needle in str(caught.value)


def test_object_exact_and_optional_accept_exactly_their_declared_keys() -> None:
    assert object_exact({"a": 1, "b": 2}, path="x", fields=_FIELDS) == {"a": 1, "b": 2}
    optional = object_optional(
        {"a": 1, "c": 3}, path="x", required=frozenset({"a"}), optional=frozenset({"c"})
    )
    assert optional == {"a": 1, "c": 3}
    with pytest.raises(D1VerdictValidationError) as caught:
        object_optional(
            {"a": 1, "z": 3}, path="x", required=frozenset({"a"}), optional=frozenset({"c"})
        )
    assert caught.value.code == "unknown_field"


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": "d1-endpoint-gap-v1"},
        {"schema_version": "d1-endpoint-gap-v1", "reason": "undefined_zero_norm", "extra": 1},
        [],
        {1: 2},
    ],
)
def test_statistical_gap_objects_refuse_inexact_fields(value: object) -> None:
    with pytest.raises(ValueError, match="undefined statistic requires exact versioned fields"):
        parse_endpoint_gap(value)


@pytest.mark.parametrize(
    "value",
    [
        {"schema_version": "d1-cosine-gap-v1", "reason": "undefined_zero_norm"},
        {
            "schema_version": "d1-cosine-gap-v1",
            "reason": "undefined_zero_norm",
            "affected_positions": 1,
            "total_positions": 2,
            "extra": 0,
        },
    ],
)
def test_cosine_gap_refuses_missing_and_extra_fields(value: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="D1 cosine gap requires exact versioned fields"):
        parse_d1_cosine(value)


def test_telemetry_exact_fields_and_sha_keep_their_gap_codes() -> None:
    with pytest.raises(D1TelemetryGap) as missing:
        exact_fields({"a": 1}, {"a", "b"}, path="p")
    assert missing.value.code == "field_set"
    with pytest.raises(D1TelemetryGap) as extra:
        exact_fields({"a": 1, "b": 2, "c": 3}, {"a", "b"}, path="p")
    assert extra.value.code == "field_set"
    exact_fields({"a": 1, "b": 2}, frozenset({"a", "b"}), path="p")

    for bad in (_MIXED, "1" * 63, "1" * 65, "g" * 64, "", 12, None):
        with pytest.raises(D1TelemetryGap) as caught:
            require_sha256(bad, path="p.digest")
        assert caught.value.code == "invalid_sha256"
        assert caught.value.path == "p.digest"
        assert "must be 64 lowercase hexadecimal characters" in str(caught.value)
    assert require_sha256(_DIGEST, path="p.digest") == _DIGEST


def test_run_start_refuses_a_mixed_case_plan_digest_with_a_telemetry_gap() -> None:
    events = valid_events()
    payload = dict(events[0].manifest["payload"])  # type: ignore[arg-type]
    payload["plan_digest"] = _MIXED
    with pytest.raises(D1TelemetryGap) as caught:
        parse_run_start_event(replace_payload(events[0], payload))
    assert caught.value.code == "invalid_sha256"
    parse_baseline_event(events[1])


# --------------------------------------------------------------------------
# 4. SHA boundary: a mixed-case digest raises the site's own exception


def test_expected_aggregate_digest_refuses_mixed_case_with_the_hydration_error(
    tmp_path: Path,
) -> None:
    for bad in (_MIXED, "1" * 63, "g" * 64):
        with pytest.raises(
            D1CaptureAggregateHydrationError,
            match="expected aggregate manifest digest must be lowercase SHA-256",
        ):
            hydrate_d1_capture_aggregate("{}", expected_manifest_digest=bad, capture_root=tmp_path)


def test_preparation_record_refuses_a_mixed_case_aggregate_digest(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path)
    record = capture_preparation_custody_from_preparation(preparation, authority=_authority())
    upper = record.aggregate_sha256.upper()
    assert upper != record.aggregate_sha256, "fixture digest has no letters to upcase"
    with pytest.raises(
        D1CapturePreparationCustodyError,
        match="aggregate is invalid: capture aggregate sha256 " + _SHA_MESSAGE,
    ):
        replace(record, aggregate_sha256=upper)


@pytest.mark.parametrize(
    "field",
    [
        "campaign_digest",
        "registered_design_digest",
        "matrix_digest",
        "attempt_ledger_digest",
        "lifecycle_prefix",
    ],
)
def test_campaign_authority_refuses_a_mixed_case_digest(field: str) -> None:
    with pytest.raises(ValueError, match=f"campaign capture {field} {_SHA_MESSAGE}"):
        replace(_authority(), **{field: _MIXED})
    assert (
        D1CampaignCaptureAuthority(_DIGEST, _DIGEST, _DIGEST, _DIGEST, _DIGEST, 1).campaign_digest
        == _DIGEST
    )


@pytest.mark.parametrize(
    "field",
    [
        "protocol_digest",
        "spec_digest",
        "split_digest",
        "source_order_digest",
        "mean_parameter_digest",
        "codec_digest",
    ],
)
def test_calibration_attestation_refuses_a_mixed_case_digest(field: str) -> None:
    fields: dict[str, object] = {
        "protocol_digest": _DIGEST,
        "spec_digest": _DIGEST,
        "split_digest": _DIGEST,
        "source_order_digest": _DIGEST,
        "mean_parameter_digest": _DIGEST,
        "codec_digest": _DIGEST,
        "initialization_seed": 7,
        "num_batches": 1,
        "online_positions": 4,
        "source_span_positions": 8,
    }
    D1BaselineCalibrationAttestation(**fields)  # type: ignore[arg-type]
    fields[field] = _MIXED
    with pytest.raises(ValueError, match=f"{field} {_SHA_MESSAGE}"):
        D1BaselineCalibrationAttestation(**fields)  # type: ignore[arg-type]


def test_journal_snapshot_validation_refuses_a_mixed_case_sha256(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    target = root / "runs" / "run.jsonl"
    target.parent.mkdir(parents=True)
    write_events(target, valid_events())
    snapshot = snapshot_module.read_d1_journal_snapshot(
        root, "runs/run.jsonl", max_bytes=target.stat().st_size
    )
    validate = snapshot_module._validate_snapshot  # pyright: ignore[reportPrivateUsage]
    validate(snapshot)
    object.__setattr__(snapshot, "sha256", snapshot.sha256.upper())
    assert snapshot.sha256 != snapshot.sha256.lower(), "fixture digest has no letters to upcase"
    with pytest.raises(ValueError, match=f"D1 journal snapshot sha256 {_SHA_MESSAGE}"):
        validate(snapshot)
