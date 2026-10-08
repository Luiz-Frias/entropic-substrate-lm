"""p14-j-analyze-review-file: ladder/{analyze,review,file} adopt kernel.codec.strict_json.

Groups, one per ticket facet (FOR-1572, FOR-1577, FOR-1604, FOR-1610, FOR-1613):

1. adoption -- an AST walk finds no local strict-JSON hook definition left under
   the three packages, and none of the seven adopting files passes an
   ``object_pairs_hook`` to a decoder any more.
2. decoder boundary -- every adopted decoder raises its OWN exception type for
   duplicate keys, non-finite constants and overflow floats, because ``fail=``
   is wired to the decoder's own error channel.
3. exact-key sites -- the ``_object`` / ``_mapping`` helpers keep their error
   types and messages while delegating to ``exact_keys`` / ``require_mapping``.
4. SHA boundary -- every digest site refuses a mixed-case digest with that
   site's own exception type and its existing message (ruling 12: lowercase
   strict at every boundary, never ``require_sha256_hex_any_case``).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.kernel.verdict.d1_types import D1_OPAQUE_ARM_ALIAS_IDS
from entropic_substrate.ladder.analyze import (
    d1_analysis_receipt_codec,
    d1_clause_policy,
    d1_diagnostics,
    d1_inspect_request,
    d1_machine_analysis_codec,
    ladder_registration,
)
from entropic_substrate.ladder.analyze.d1_analysis_receipt_codec import (
    parse_d1_analysis_execution_receipt,
)
from entropic_substrate.ladder.analyze.d1_clause_policy import D1ClausePolicyError
from entropic_substrate.ladder.analyze.d1_diagnostics import D1DiagnosticPolicyError
from entropic_substrate.ladder.analyze.d1_inspect_codec import (
    D1InspectCodecError,
    parse_d1_inspect_result,
)
from entropic_substrate.ladder.analyze.d1_inspect_values import (
    D1InspectCheckpoint,
    D1InspectEventRef,
    D1InspectJournal,
    D1InspectRun,
)
from entropic_substrate.ladder.analyze.d1_machine_analysis import D1AliasError, D1ArmAliasMap
from entropic_substrate.ladder.analyze.d1_machine_analysis_codec import (
    D1MachineAnalysisCodecError,
    decode_d1_machine_analysis_declaration,
)
from entropic_substrate.ladder.analyze.ladder_analysis import parse_scientific_data_order
from entropic_substrate.ladder.capture.records.d1_journal_values import JournalEventKind
from entropic_substrate.ladder.file import d1_filing, d1_filing_receipts
from entropic_substrate.ladder.file.d1_filing import D1FilingValidationError
from entropic_substrate.ladder.review import (
    d1_execution_receipt,
    d1_external_analysis_codec,
    d1_external_analysis_control,
    d1_external_review_adapter,
    d1_machine_analysis_binding,
    d1_machine_analysis_verification,
    d1_review_adjudication,
    d1_review_execution,
)
from entropic_substrate.ladder.review.d1_external_analysis_codec import (
    decode_d1_external_analysis_worker_response_v2,
)
from entropic_substrate.ladder.review.d1_external_analysis_control import (
    D1ExternalAnalysisControlError,
    parse_d1_alias_commitment_control,
)
from entropic_substrate.ladder.review.d1_machine_analysis_binding import (
    D1MachineAnalysisBindingError,
)
from entropic_substrate.ladder.review.d1_machine_analysis_verification import (
    D1MachineAnalysisVerificationError,
)
from entropic_substrate.ladder.review.d1_review_adjudication import (
    parse_d1_review_execution_receipt,
)

pytestmark = pytest.mark.unit

_DIGEST = "1" * 64
_MIXED = "A" + "1" * 63
_ANY_CASE_MESSAGE = "must be a 64-character hexadecimal SHA-256 digest"

_SRC = Path(__file__).resolve().parents[1] / "src" / "entropic_substrate"
_SCANNED = tuple(_SRC / "ladder" / package for package in ("analyze", "review", "file"))
_FORBIDDEN = frozenset({"_pairs", "_strict_pairs", "_nonfinite", "_reject_constant"})
_ADOPTERS = (
    "ladder/review/d1_external_analysis_control.py",
    "ladder/review/d1_external_analysis_codec.py",
    "ladder/review/d1_external_review_adapter.py",
    "ladder/review/d1_review_adjudication.py",
    "ladder/analyze/d1_analysis_receipt_codec.py",
    "ladder/analyze/d1_inspect_codec.py",
    "ladder/analyze/d1_machine_analysis_codec.py",
)


# --------------------------------------------------------------------------
# 1. adoption: no local strict-JSON hook definitions


def _forbidden_definitions(source: str) -> list[str]:
    return [
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _FORBIDDEN
    ]


def _hook_keywords(source: str) -> list[str]:
    return [
        keyword.arg
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg in {"object_pairs_hook", "parse_constant"}
    ]


def test_the_ast_walk_detects_a_local_hook_definition_at_any_depth() -> None:
    nested = "def outer():\n    def _strict_pairs(pairs):\n        return {}\n"
    assert _forbidden_definitions(nested) == ["_strict_pairs"]
    assert _forbidden_definitions("def parse():\n    return 1\n") == []
    assert _hook_keywords("loads(raw, object_pairs_hook=f, parse_constant=g)\n") == [
        "object_pairs_hook",
        "parse_constant",
    ]
    assert _hook_keywords("loads(raw)\n") == []


def test_analyze_review_file_define_no_local_strict_json_hooks() -> None:
    files = [path for root in _SCANNED for path in sorted(root.rglob("*.py"))]
    assert len(files) > 40, "scan is misrooted: positive control on the file count failed"
    for adopter in _ADOPTERS:
        assert _SRC / adopter in files, f"{adopter} must be scanned"
    found = {
        path.relative_to(_SRC).as_posix(): names
        for path in files
        if (names := _forbidden_definitions(path.read_text(encoding="utf-8")))
    }
    assert found == {}


@pytest.mark.parametrize("adopter", _ADOPTERS)
def test_adopters_pass_no_decoder_hooks_of_their_own(adopter: str) -> None:
    assert _hook_keywords((_SRC / adopter).read_text(encoding="utf-8")) == []


# --------------------------------------------------------------------------
# 2. decoder boundary: strict_loads(fail=<decoder channel>) keeps each exception type

_DUPLICATE = b'{"a":1,"a":2}'
_NAN = b'{"a":NaN}'
_NEG_INFINITY = b'{"a":-Infinity}'
_OVERFLOW = b'{"a":1e999}'

_DECODERS: list[tuple[str, Callable[[bytes], object], type[Exception]]] = [
    ("receipt", parse_d1_analysis_execution_receipt, ValueError),
    ("external-codec", decode_d1_external_analysis_worker_response_v2, ValueError),
    ("control", parse_d1_alias_commitment_control, D1ExternalAnalysisControlError),
    ("adjudication", parse_d1_review_execution_receipt, ValueError),
    ("adapter", d1_external_review_adapter._parse_result, ValueError),
    ("inspect", parse_d1_inspect_result, D1InspectCodecError),
    ("machine-analysis", decode_d1_machine_analysis_declaration, D1MachineAnalysisCodecError),
]


@pytest.mark.parametrize(("name", "decode", "error"), _DECODERS, ids=[d[0] for d in _DECODERS])
@pytest.mark.parametrize("payload", [_DUPLICATE, _NAN, _NEG_INFINITY, _OVERFLOW])
def test_decoders_refuse_strict_json_violations_with_their_own_error(
    name: str, decode: Callable[[bytes], object], error: type[Exception], payload: bytes
) -> None:
    with pytest.raises(ValueError) as raised:
        decode(payload)
    assert type(raised.value) is error, name


@pytest.mark.parametrize(
    ("payload", "needle"),
    [(_DUPLICATE, "duplicate"), (_NAN, "non-finite"), (_OVERFLOW, "non-finite")],
)
@pytest.mark.parametrize(
    "decode",
    [
        parse_d1_inspect_result,
        decode_d1_machine_analysis_declaration,
        parse_d1_alias_commitment_control,
        decode_d1_external_analysis_worker_response_v2,
        parse_d1_analysis_execution_receipt,
        parse_d1_review_execution_receipt,
    ],
)
def test_decode_level_messages_come_from_the_door(
    decode: Callable[[bytes], object], payload: bytes, needle: str
) -> None:
    with pytest.raises(ValueError, match=needle):
        decode(payload)


@pytest.mark.parametrize("payload", [_DUPLICATE, _NAN, _NEG_INFINITY, _OVERFLOW])
def test_adapter_decode_reports_the_strict_json_refusal(payload: bytes) -> None:
    """The adapter swallows the door's reason, so the refusal is pinned by its own message."""
    with pytest.raises(ValueError, match="worker stdout must be strict UTF-8 JSON"):
        d1_external_review_adapter._parse_result(payload)


def test_adapter_and_adjudication_decoders_refuse_inexact_fields_with_their_message() -> None:
    with pytest.raises(ValueError, match="worker result fields must match the protocol exactly"):
        d1_external_review_adapter._parse_result(b'{"a":1}')
    with pytest.raises(ValueError, match="fields must match the v2 schema exactly"):
        d1_review_adjudication._document(b'{"a":1}')


def test_review_receipt_decode_now_refuses_nonfinite_constants() -> None:
    """The adjudication receipt decoder carried no parse_constant hook before the door."""
    with pytest.raises(ValueError, match="non-finite constant: 'NaN'"):
        parse_d1_review_execution_receipt(b'{"round":NaN}')


# --------------------------------------------------------------------------
# 3. exact-key sites: each keeps its own error type and messages


def test_diagnostic_and_clause_objects_keep_their_codes() -> None:
    for module, error in (
        (d1_diagnostics, D1DiagnosticPolicyError),
        (d1_clause_policy, D1ClausePolicyError),
    ):
        assert module._object({"a": 1}, path="p", fields=frozenset({"a"})) == {"a": 1}
        with pytest.raises(error, match="must contain exactly") as keys:
            module._object({"a": 1}, path="p", fields=frozenset({"a", "b"}))
        assert keys.value.code == "fields"  # type: ignore[attr-defined]
        with pytest.raises(error, match="must contain exactly"):
            module._object({"a": 1, "z": 2}, path="p", fields=frozenset({"a"}))
        with pytest.raises(error, match="object with string keys") as kind:
            module._object([("a", 1)], path="p", fields=frozenset({"a"}))
        assert kind.value.code == "type"  # type: ignore[attr-defined]
        with pytest.raises(error, match="object with string keys"):
            module._object({1: 1}, path="p", fields=frozenset({1}))  # type: ignore[arg-type]


def test_registration_object_keeps_its_messages() -> None:
    assert ladder_registration._object({"a": 1, "b": 2}, "a b") == {"a": 1, "b": 2}
    assert ladder_registration._object({"anything": 1}) == {"anything": 1}
    with pytest.raises(ValueError, match="keys must be exact; no defaults"):
        ladder_registration._object({"a": 1}, "a b")
    with pytest.raises(ValueError, match="keys must be exact; no defaults"):
        ladder_registration._object({"a": 1, "b": 2, "c": 3}, "a b")
    with pytest.raises(ValueError, match="requires a JSON object"):
        ladder_registration._object([1])
    with pytest.raises(ValueError, match="requires a JSON object"):
        ladder_registration._object({1: 1})


def test_filing_object_reports_sorted_missing_unknown_and_type_errors() -> None:
    keys = frozenset({"a", "b", "c"})
    assert d1_filing._object({"a": 1, "b": 2, "c": 3}, path="p", keys=keys) == {
        "a": 1,
        "b": 2,
        "c": 3,
    }
    with pytest.raises(D1FilingValidationError, match=r"missing required field\(s\): b, c") as miss:
        d1_filing._object({"a": 1}, path="p", keys=keys)
    assert miss.value.code == "missing_field"
    with pytest.raises(D1FilingValidationError, match=r"unknown field\(s\): y, z") as extra:
        d1_filing._object({"a": 1, "b": 2, "c": 3, "z": 4, "y": 5}, path="p", keys=keys)
    assert extra.value.code == "unknown_field"
    with pytest.raises(D1FilingValidationError, match="must be an object") as kind:
        d1_filing._object([], path="p", keys=keys)
    assert kind.value.code == "invalid_type"
    with pytest.raises(D1FilingValidationError, match="field names must be strings") as name:
        d1_filing._object({1: 1}, path="p", keys=keys)
    assert name.value.code == "invalid_field_name"


@pytest.mark.parametrize(
    ("module", "error"),
    [
        (d1_machine_analysis_binding, D1MachineAnalysisBindingError),
        (d1_machine_analysis_verification, D1MachineAnalysisVerificationError),
    ],
)
def test_machine_analysis_mapping_keeps_its_own_exception_type(
    module: object, error: type[Exception]
) -> None:
    mapping = module._mapping  # type: ignore[attr-defined]
    assert mapping({"a": 1}, "x") == {"a": 1}
    for bad in ([], "text", None, {1: 1}):
        with pytest.raises(ValueError, match="machine analysis x must be an object") as raised:
            mapping(bad, "x")
        assert type(raised.value) is error


# --------------------------------------------------------------------------
# 4. SHA boundary: a mixed-case digest is refused by every site's own channel


def test_inspect_request_digest_refuses_mixed_case_with_its_message() -> None:
    d1_inspect_request._optional_digest(None, label="x")
    d1_inspect_request._optional_digest(_DIGEST, label="x")
    with pytest.raises(ValueError, match=f"D1 inspect x {_ANY_CASE_MESSAGE}"):
        d1_inspect_request._optional_digest(_MIXED, label="x")


def test_inspect_journal_checkpoint_and_event_refuse_a_mixed_case_digest() -> None:
    D1InspectJournal("p", "r", 1, _DIGEST)
    D1InspectCheckpoint(0, _DIGEST)
    D1InspectEventRef(JournalEventKind.RUN_START, 0, 0, _DIGEST)
    with pytest.raises(ValueError, match=f"D1 inspect journal sha256 {_ANY_CASE_MESSAGE}"):
        D1InspectJournal("p", "r", 1, _MIXED)
    with pytest.raises(ValueError, match=f"D1 inspect checkpoint sha256 {_ANY_CASE_MESSAGE}"):
        D1InspectCheckpoint(0, _MIXED)
    with pytest.raises(ValueError, match=f"D1 inspect event sha256 {_ANY_CASE_MESSAGE}"):
        D1InspectEventRef(JournalEventKind.RUN_START, 0, 0, _MIXED)


@pytest.mark.parametrize(
    "field",
    [
        "execution_digest",
        "aggregation_digest",
        "construction_digest",
        "family_digest",
        "plan_digest",
        "authority_digest",
    ],
)
def test_inspect_run_refuses_a_mixed_case_digest_in_every_field(field: str) -> None:
    fields = {
        "execution_digest": _DIGEST,
        "aggregation_digest": _DIGEST,
        "construction_digest": _DIGEST,
        "family_digest": _DIGEST,
        "plan_digest": _DIGEST,
        "authority_digest": _DIGEST,
    }
    D1InspectRun("run", RunClass.CONFIRMATORY, **fields)
    with pytest.raises(ValueError, match=f"D1 inspect {field} {_ANY_CASE_MESSAGE}"):
        D1InspectRun("run", RunClass.CONFIRMATORY, **{**fields, field: _MIXED})


def test_alias_map_refuses_a_mixed_case_commitment_with_its_alias_error() -> None:
    aliases = dict(zip(D1Arm, D1_OPAQUE_ARM_ALIAS_IDS, strict=False))
    D1ArmAliasMap(aliases, _DIGEST)
    with pytest.raises(D1AliasError, match="SHA-256 sealed commitment"):
        D1ArmAliasMap(aliases, _MIXED)


def test_scientific_data_order_refuses_a_mixed_case_digest() -> None:
    assert parse_scientific_data_order([_DIGEST, 1], 1) == (_DIGEST, 1)
    with pytest.raises(ValueError, match=f"scientific data order {_ANY_CASE_MESSAGE}"):
        parse_scientific_data_order([_MIXED, 1], 1)


def test_statistical_identity_refuses_a_mixed_case_digest() -> None:
    assert ladder_registration._digest(_DIGEST) == _DIGEST
    with pytest.raises(ValueError, match=f"statistical identity {_ANY_CASE_MESSAGE}"):
        ladder_registration._digest(_MIXED)


@pytest.mark.parametrize(
    ("check", "message"),
    [
        (d1_analysis_receipt_codec._require_sha256, "must be a lowercase SHA-256 digest"),
        (d1_external_analysis_codec._sha256, "must be a lowercase SHA-256 digest"),
        (d1_execution_receipt._sha256, "must be a lowercase SHA-256 digest"),
    ],
    ids=["analysis-receipt-codec", "external-analysis-codec", "execution-receipt"],
)
def test_keyword_label_sha_sites_refuse_mixed_case_with_a_value_error(
    check: Callable[..., object], message: str
) -> None:
    check(_DIGEST, label="x")
    with pytest.raises(ValueError, match=f"x {message}") as raised:
        check(_MIXED, label="x")
    assert type(raised.value) is ValueError


@pytest.mark.parametrize(
    "check",
    [
        d1_review_adjudication._sha,
        d1_external_review_adapter._digest,
        d1_review_execution._digest,
    ],
    ids=["adjudication", "adapter", "review-execution"],
)
def test_positional_label_sha_sites_refuse_mixed_case_with_a_value_error(
    check: Callable[..., object],
) -> None:
    check(_DIGEST, "x")
    with pytest.raises(ValueError, match="x must be lowercase SHA-256") as raised:
        check(_MIXED, "x")
    assert type(raised.value) is ValueError


def test_external_analysis_control_refuses_mixed_case_with_its_own_error() -> None:
    assert d1_external_analysis_control._sha256(_DIGEST, label="x") == _DIGEST
    with pytest.raises(D1ExternalAnalysisControlError, match="x must be lowercase SHA-256"):
        d1_external_analysis_control._sha256(_MIXED, label="x")


def test_machine_analysis_codec_refuses_mixed_case_with_its_own_error() -> None:
    assert d1_machine_analysis_codec._sha(_DIGEST, "x") == _DIGEST
    with pytest.raises(D1MachineAnalysisCodecError, match="x must be lowercase SHA-256"):
        d1_machine_analysis_codec._sha(_MIXED, "x")


def test_filing_receipt_sha_refuses_mixed_case_with_its_graph_error() -> None:
    assert d1_filing_receipts._sha256(_DIGEST, path="p") == _DIGEST
    with pytest.raises(d1_filing_receipts.D1FilingGraphError) as raised:
        d1_filing_receipts._sha256(_MIXED, path="p")
    assert raised.value.code == "invalid_sha256"
    assert raised.value.path == "p"
