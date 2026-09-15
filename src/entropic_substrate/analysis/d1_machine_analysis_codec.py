"""Strict structural decoder for canonical aliased D1 machine analysis."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import NoReturn, cast

from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.evidence.d1_statistical_gaps import parse_endpoint_gap, parse_estimate_gap
from entropic_substrate.evidence.d1_types import D1_OPAQUE_ARM_ALIAS_IDS

__all__ = [
    "D1MachineAnalysisCodecError",
    "decode_d1_machine_analysis_declaration",
]

_SCHEMA = "d1-aliased-machine-analysis-v2"
_GAP_SCHEMA = "d1-aliased-machine-analysis-v3"
_ALIASES = frozenset(D1_OPAQUE_ARM_ALIAS_IDS)
_EVIDENCE = "DERIVED_FROM_MEASURED"
_STATUS = frozenset({"PASS", "FAIL", "INCONCLUSIVE"})
_TOP = frozenset(
    "alias_commitment bindings clause_precedence clauses_by_alias contrasts diagnostics "
    "endpoint excluded_seeds exclusions factorial_effects feasible_aliases included_seeds "
    "schema_version".split()
)
_CONTRAST = frozenset(
    "confidence_level contrast_alias df estimate evidence_class interval_lower interval_upper "
    "margins multiplicity n paired_estimates sample_sd standard_error t_critical terms".split()
)
_EFFECT = frozenset(
    {"deciding", "effect_alias", "estimate", "evidence_class", "n", "paired_estimates", "terms"}
)
_DIAGNOSTIC = frozenset(
    "arm_alias aulc aulc_deciding aulc_metric aulc_normalization evidence_class "
    "quality_target seed trajectory".split()
)
_PROHIBITED_KEYS = frozenset(
    "arm arm_id construction_digest file_path logical_run_id mapping nonce path private_reveal".split()
)
_TRUE_ARMS = frozenset({"none", "sg", "ac", "both"})
_CONDITION = frozenset(
    "contrast_alias estimator evidence_class metric observed_lower observed_point observed_upper "
    "operator smoothing_window source status threshold".split()
)
_QUALITY_TARGET: frozenset[str] = frozenset(
    "eval_index evidence_class metric metric_value model_tokens operator reached source_bytes "
    "step threshold trainer_wall_seconds".split()
)
_QUALITY_TARGET_RESULT = _QUALITY_TARGET - {
    "evidence_class",
    "metric",
    "operator",
    "reached",
    "threshold",
}


class D1MachineAnalysisCodecError(ValueError):
    """Machine analysis bytes are malformed, incomplete, or identity-revealing."""


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1MachineAnalysisCodecError(message)
    raise D1MachineAnalysisCodecError(message) from error


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate machine analysis key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite machine analysis constant {value!r}")


def _object(value: object, keys: frozenset[str], label: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{label} must be an object")
    item = cast(dict[str, object], value)
    if frozenset(item) != keys:
        _fail(f"{label} fields must be exact")
    return item


def _array(value: object, label: str) -> list[object]:
    if type(value) is not list:
        _fail(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be nonempty text")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        _fail(f"{label} must be finite numeric and nonboolean")
    result = float(cast(int | float, value))
    if not math.isfinite(result):
        _fail(f"{label} must be finite")
    return result


def _optional_number(value: object, label: str) -> float | None:
    return None if value is None else _number(value, label)


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be integer >= {minimum}")
    return value


def _sha(value: object, label: str) -> str:
    text = _text(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        _fail(f"{label} must be lowercase SHA-256")
    return text


def _alias(value: object, label: str) -> str:
    alias = _text(value, label)
    if alias not in _ALIASES:
        _fail(f"{label} must use opaque A/B/C/D alias")
    return alias


def _evidence(value: object, label: str) -> None:
    if value != _EVIDENCE:
        _fail(f"{label} evidence_class must be {_EVIDENCE}")


def _walk(value: object) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    strings: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            child_keys, child_strings = _walk(child)
            keys |= child_keys
            strings |= child_strings
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            child_keys, child_strings = _walk(child)
            keys |= child_keys
            strings |= child_strings
    elif isinstance(value, str):
        strings.add(value)
    return keys, strings


def _blind(value: dict[str, object]) -> None:
    keys, strings = _walk(value)
    if keys & _PROHIBITED_KEYS:
        _fail("machine analysis blind structure contains a construction-bearing field")
    tokens = {token.casefold() for text in strings for token in re.findall(r"[A-Za-z0-9_]+", text)}
    if tokens & _TRUE_ARMS:
        _fail("machine analysis blind structure contains a true arm label")


def _seed_list(value: object, label: str) -> tuple[int, ...]:
    result = tuple(
        _integer(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))
    )
    if len(set(result)) != len(result):
        _fail(f"{label} seed identities must be unique")
    return result


def _terms(value: object, label: str) -> None:
    rows = _array(value, label)
    aliases: list[str] = []
    if not rows:
        _fail(f"{label} must be nonempty")
    for index, raw in enumerate(rows):
        row = _object(raw, frozenset({"arm_alias", "coefficient"}), f"{label}[{index}]")
        aliases.append(_alias(row["arm_alias"], f"{label}[{index}].arm_alias"))
        if _number(row["coefficient"], f"{label}[{index}].coefficient") == 0.0:
            _fail(f"{label} coefficients must be nonzero")
    if len(set(aliases)) != len(aliases):
        _fail(f"{label} arm aliases must be unique")


def _unavailable(row: dict[str, object], seeds: tuple[int, ...]) -> None:
    missing = parse_estimate_gap(row["unavailable"], aliases=True)
    if not {cell.seed for cell in missing.cells} <= set(seeds):
        _fail("undefined estimate refers to an excluded or unknown seed")
    terms = _array(row["terms"], "undefined terms")
    arms = {cast(dict[str, object], term)["arm_alias"] for term in terms}
    if not {cell.arm for cell in missing.cells} <= arms:
        _fail("undefined estimate refers to an unused arm")
    if any(
        row.get(name) is not None
        for name in (
            "estimate",
            "sample_sd",
            "standard_error",
            "interval_lower",
            "interval_upper",
            "t_critical",
        )
    ):
        _fail("undefined estimate cannot contain an inferred aggregate statistic")
    pairs = _array(row["paired_estimates"], "undefined paired values")
    if len(pairs) != len(seeds):
        _fail("undefined estimate must retain the complete registered population")
    affected = {cell.seed for cell in missing.cells}
    for seed, value in zip(seeds, pairs, strict=True):
        if seed in affected:
            if value is not None:
                _fail("undefined paired value must be null")
        else:
            _number(value, "observed paired value")


def _contrast(raw: object, index: int, seeds: tuple[int, ...], allow_gaps: bool) -> None:
    missing = allow_gaps and isinstance(raw, dict) and "unavailable" in raw
    row = _object(raw, _CONTRAST | ({"unavailable"} if missing else set()), f"contrast[{index}]")
    n = len(seeds)
    if row["contrast_alias"] != f"C{index + 1:02d}":
        _fail("contrast aliases must be ordered C01..Cn")
    _evidence(row["evidence_class"], f"contrast[{index}]")
    if _integer(row["n"], f"contrast[{index}].n", positive=True) != n:
        _fail("contrast n must equal included seed count")
    df = row["df"]
    if not ((n == 1 and df is None) or (type(df) is int and df == n - 1)):
        _fail("contrast df must be n-1 or null for n=1")
    for field in ("confidence_level",) if missing else ("estimate", "confidence_level"):
        _number(row[field], f"contrast[{index}].{field}")
    interval = tuple(
        _optional_number(row[field], f"contrast[{index}].{field}")
        for field in (
            "sample_sd",
            "standard_error",
            "interval_lower",
            "interval_upper",
            "t_critical",
        )
    )
    if not missing and (
        (n == 1 and any(value is not None for value in interval))
        or (n > 1 and any(value is None for value in interval))
    ):
        _fail("contrast interval fields disagree with n")
    paired = _array(row["paired_estimates"], f"contrast[{index}].paired_estimates")
    if len(paired) != n:
        _fail("contrast paired estimate count differs from n")
    for pair_index, value in enumerate(paired):
        if not missing:
            _number(value, f"contrast[{index}].paired_estimates[{pair_index}]")
    margins = row["margins"]
    if type(margins) is not dict:
        _fail("contrast margins must be an object")
    for name, value in cast(dict[str, object], margins).items():
        if not name or _number(value, f"contrast margin {name}") < 0.0:
            _fail("contrast margins must be named and nonnegative")
    _text(row["multiplicity"], f"contrast[{index}].multiplicity")
    _terms(row["terms"], f"contrast[{index}].terms")
    if missing:
        _unavailable(row, seeds)


def _effect(raw: object, index: int, seeds: tuple[int, ...], allow_gaps: bool) -> None:
    missing = allow_gaps and isinstance(raw, dict) and "unavailable" in raw
    row = _object(raw, _EFFECT | ({"unavailable"} if missing else set()), f"effect[{index}]")
    n = len(seeds)
    if row["effect_alias"] != f"F{index + 1:02d}" or row["deciding"] is not False:
        _fail("effect aliases must be ordered and deciding must be false")
    _evidence(row["evidence_class"], f"effect[{index}]")
    if _integer(row["n"], f"effect[{index}].n", positive=True) != n:
        _fail("effect n must equal included seed count")
    if not missing:
        _number(row["estimate"], f"effect[{index}].estimate")
    paired = _array(row["paired_estimates"], f"effect[{index}].paired_estimates")
    if len(paired) != n:
        _fail("effect paired estimate count differs from n")
    for value in paired:
        if not missing:
            _number(value, f"effect[{index}].paired_estimate")
    _terms(row["terms"], f"effect[{index}].terms")
    if missing:
        _unavailable(row, seeds)


def _condition(raw: object, label: str) -> None:
    row = _object(raw, _CONDITION, label)
    _evidence(row["evidence_class"], label)
    if row["status"] not in _STATUS:
        _fail(f"{label} status is unsupported")
    if row["contrast_alias"] is not None and not re.fullmatch(
        r"C\d{2}", _text(row["contrast_alias"], f"{label}.contrast_alias")
    ):
        _fail(f"{label} contrast alias is invalid")
    for field in ("estimator", "metric"):
        if row[field] is not None:
            _text(row[field], f"{label}.{field}")
    if row["smoothing_window"] is not None:
        _integer(row["smoothing_window"], f"{label}.smoothing_window", positive=True)
    for field in ("observed_lower", "observed_point", "observed_upper"):
        _optional_number(row[field], f"{label}.{field}")
    _text(row["operator"], f"{label}.operator")
    _text(row["source"], f"{label}.source")
    _number(row["threshold"], f"{label}.threshold")


def _clauses(root: dict[str, object], precedence: tuple[str, ...]) -> set[str]:
    raw = root["clauses_by_alias"]
    if type(raw) is not dict or not raw:
        _fail("clauses_by_alias must be a nonempty object")
    aliases: set[str] = set()
    for alias, raw_clauses in cast(dict[str, object], raw).items():
        aliases.add(_alias(alias, "clause arm alias"))
        if type(raw_clauses) is not dict or frozenset(raw_clauses) != frozenset(precedence):
            _fail("clause keys must equal clause_precedence")
        for clause_id, raw_clause in cast(dict[str, object], raw_clauses).items():
            clause = _object(
                raw_clause,
                frozenset({"conditions", "evidence_class", "status"}),
                f"clause {clause_id}",
            )
            _evidence(clause["evidence_class"], f"clause {clause_id}")
            if clause["status"] not in _STATUS:
                _fail("clause status is unsupported")
            conditions = _array(clause["conditions"], f"clause {clause_id}.conditions")
            if not conditions:
                _fail("clause conditions must be nonempty")
            for index, condition in enumerate(conditions):
                _condition(condition, f"clause {clause_id}.condition[{index}]")
    return aliases


def _diagnostic(raw: object, index: int, included: set[int]) -> tuple[str, int]:
    row = _object(raw, _DIAGNOSTIC, f"diagnostic[{index}]")
    alias = _alias(row["arm_alias"], f"diagnostic[{index}].arm_alias")
    seed = _integer(row["seed"], f"diagnostic[{index}].seed")
    if seed not in included:
        _fail("diagnostic seed is not included")
    _evidence(row["evidence_class"], f"diagnostic[{index}]")
    _number(row["aulc"], f"diagnostic[{index}].aulc")
    if row["aulc_deciding"] is not False:
        _fail("diagnostic AULC must be report-only")
    _text(row["aulc_metric"], f"diagnostic[{index}].aulc_metric")
    _text(row["aulc_normalization"], f"diagnostic[{index}].aulc_normalization")
    trajectory = _array(row["trajectory"], f"diagnostic[{index}].trajectory")
    if not trajectory:
        _fail("diagnostic trajectory must be nonempty")
    previous = (-1, -1)
    for point_index, raw_point in enumerate(trajectory):
        point = _object(
            raw_point,
            frozenset({"eval_index", "step", "value"}),
            f"diagnostic[{index}].trajectory[{point_index}]",
        )
        cursor = (
            _integer(point["eval_index"], "trajectory eval_index"),
            _integer(point["step"], "trajectory step"),
        )
        if cursor[0] <= previous[0] or cursor[1] < previous[1]:
            _fail("diagnostic trajectory cursors must increase")
        previous = cursor
        _number(point["value"], "trajectory value")
    target = _object(
        row["quality_target"],
        _QUALITY_TARGET,
        f"diagnostic[{index}].quality target",
    )
    _evidence(target["evidence_class"], "quality target")
    _text(target["metric"], "quality target metric")
    operator = _text(target["operator"], "quality target operator")
    if operator not in {"AT_LEAST", "AT_MOST"}:
        _fail("quality target operator is unsupported")
    _number(target["threshold"], "quality target threshold")
    if type(target["reached"]) is not bool:
        _fail("quality target reached must be boolean")
    if target["reached"] is False and any(
        target[field] is not None for field in _QUALITY_TARGET_RESULT
    ):
        _fail("quality target unreached fields must be null")
    if target["reached"] is True:
        for field in ("eval_index", "model_tokens", "source_bytes", "step"):
            _integer(target[field], f"quality target {field}")
        for field in ("metric_value", "trainer_wall_seconds"):
            _number(target[field], f"quality target {field}")
    return alias, seed


def _validate_manifest(root: dict[str, object]) -> None:
    if frozenset(root) != _TOP or root.get("schema_version") not in {_SCHEMA, _GAP_SCHEMA}:
        _fail("machine analysis top-level fields or schema are invalid")
    _blind(root)
    _sha(root["alias_commitment"], "alias_commitment")
    bindings = _object(
        root["bindings"],
        frozenset({"capture_digest", "decision_contract_digest", "design_digest"}),
        "bindings",
    )
    for name, value in bindings.items():
        _sha(value, f"bindings.{name}")
    included = _seed_list(root["included_seeds"], "included_seeds")
    excluded = _seed_list(root["excluded_seeds"], "excluded_seeds")
    if set(included) & set(excluded):
        _fail("included and excluded seed identities must be disjoint")
    precedence = tuple(
        _text(item, "clause_precedence")
        for item in _array(root["clause_precedence"], "clause_precedence")
    )
    if not precedence or len(set(precedence)) != len(precedence):
        _fail("clause_precedence must be nonempty and unique")
    clause_aliases = _clauses(root, precedence)
    feasible = [
        _alias(item, "feasible alias")
        for item in _array(root["feasible_aliases"], "feasible_aliases")
    ]
    if len(set(feasible)) != len(feasible) or not set(feasible) <= clause_aliases:
        _fail("feasible aliases must be unique clause candidates")
    endpoint = root["endpoint"]
    allow_gaps = root["schema_version"] == _GAP_SCHEMA
    witnessed_gap = False
    contrasts = _array(root["contrasts"], "contrasts")
    effects = _array(root["factorial_effects"], "factorial_effects")
    diagnostics = _array(root["diagnostics"], "diagnostics")
    if endpoint is None:
        if included or contrasts or effects or diagnostics:
            _fail("null endpoint requires no included analysis rows")
    else:
        if not included:
            _fail("endpoint requires included seeds")
        endpoint_row = _object(
            endpoint,
            frozenset(
                {"estimator", "evidence_class", "metric", "smoothing_window", "values_by_alias"}
            ),
            "endpoint",
        )
        _evidence(endpoint_row["evidence_class"], "endpoint")
        _text(endpoint_row["estimator"], "endpoint.estimator")
        _text(endpoint_row["metric"], "endpoint.metric")
        _integer(endpoint_row["smoothing_window"], "endpoint.smoothing_window", positive=True)
        values = endpoint_row["values_by_alias"]
        if type(values) is not dict or frozenset(values) != _ALIASES:
            _fail("endpoint values must bind exact A/B/C/D aliases")
        for alias, raw_values in cast(dict[str, object], values).items():
            entries = _array(raw_values, f"endpoint.{alias}")
            if len(entries) != len(included):
                _fail("endpoint value count must equal included seeds")
            for value in entries:
                if allow_gaps and isinstance(value, dict):
                    missing = parse_endpoint_gap(value)
                    if endpoint_row["metric"] != "mean_pairwise_cosine_distance" or len(
                        missing.evaluations
                    ) > cast(int, endpoint_row["smoothing_window"]):
                        _fail("undefined endpoint does not match its registered cosine window")
                    witnessed_gap = True
                else:
                    _number(value, f"endpoint.{alias}")
        for index, contrast in enumerate(contrasts):
            _contrast(contrast, index, included, allow_gaps)
        for index, effect in enumerate(effects):
            _effect(effect, index, included, allow_gaps)
        observed = {
            _diagnostic(item, index, set(included)) for index, item in enumerate(diagnostics)
        }
        expected = {(alias, seed) for alias in _ALIASES for seed in included}
        if observed != expected or len(observed) != len(diagnostics):
            _fail("diagnostic alias/seed grid must equal every retained cell")
    if allow_gaps and not witnessed_gap:
        _fail("v3 machine analysis requires a witnessed undefined endpoint")
    exclusions = _array(root["exclusions"], "exclusions")
    for index, raw in enumerate(exclusions):
        row = _object(
            raw,
            frozenset({"evidence_digest", "exclusion_alias", "reason_id"}),
            f"exclusion[{index}]",
        )
        if row["exclusion_alias"] != f"X{index + 1:02d}":
            _fail("exclusion aliases must be ordered X01..Xn")
        _sha(row["evidence_digest"], f"exclusion[{index}].evidence_digest")
        _text(row["reason_id"], f"exclusion[{index}].reason_id")


def _decode(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("machine analysis payload must be bytes")
    try:
        value: object = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except D1MachineAnalysisCodecError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        _fail("machine analysis payload is invalid UTF-8 JSON", error)
    if type(value) is not dict:
        _fail("machine analysis payload must have an object root")
    root = cast(dict[str, object], value)
    canonical = json.dumps(
        root,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if canonical != payload:
        _fail("machine analysis payload must use canonical JSON bytes")
    _validate_manifest(root)
    return root


def decode_d1_machine_analysis_declaration(payload: bytes) -> D1AliasedMachineAnalysis:
    """Decode an untrusted declaration after exact structural validation."""
    _decode(payload)
    return D1AliasedMachineAnalysis(payload.decode("utf-8"))
