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
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import pytest

from entropic_substrate.ladder.analyze import d1_clause_policy, d1_diagnostics, ladder_registration
from entropic_substrate.ladder.analyze.d1_analysis_receipt_codec import (
    parse_d1_analysis_execution_receipt,
)
from entropic_substrate.ladder.analyze.d1_clause_policy import D1ClausePolicyError
from entropic_substrate.ladder.analyze.d1_diagnostics import D1DiagnosticPolicyError
from entropic_substrate.ladder.analyze.d1_inspect_codec import (
    D1InspectCodecError,
    parse_d1_inspect_result,
)
from entropic_substrate.ladder.analyze.d1_machine_analysis_codec import (
    D1MachineAnalysisCodecError,
    decode_d1_machine_analysis_declaration,
)
from entropic_substrate.ladder.file import d1_filing
from entropic_substrate.ladder.file.d1_filing import D1FilingValidationError
from entropic_substrate.ladder.review import (
    d1_external_review_adapter,
    d1_machine_analysis_binding,
    d1_machine_analysis_verification,
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
        parse_d1_review_execution_receipt,
    ],
)
def test_decode_level_messages_come_from_the_door(
    decode: Callable[[bytes], object], payload: bytes, needle: str
) -> None:
    with pytest.raises(ValueError, match=needle):
        decode(payload)


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
