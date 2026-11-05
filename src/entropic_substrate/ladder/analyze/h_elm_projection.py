"""The H1 E-LM projection: per-canonical-byte log-marginal and ESS/k per sample count.

A pure function of ONE captured ``h-predictive-evaluation-v1`` or ``-v2`` event. The H1
preregistration skeleton (section 5, E-LM row; section 10.1, Q2/Q4/Q8) names the
deciding estimand as the importance-weighted log-marginal at k = 64 in nats per
canonical byte, and the Q4 admissibility reading as the per-row MEDIAN of ESS/k. The
runtime extractor projects ``eval/log_marginal_k{k}`` per SENTENCE and ``eval/ess_k{k}``
as a mean. Since L4d it also emits ``eval/log_marginal_per_byte_ratio_of_sums_k{k}``, the same
key name and the same formula as this module's ratio of sums; the two are pinned equal on one
record by ``test_the_q_ratio_of_sums_equals_the_extractors_on_the_same_record``. This module
is the analysis-side projection beside the extractor.

What the captured row counts, read from the code that writes it
(``core/sentence/scoring.py`` ``evaluate_proposals``, whose q rows come from ``_q_population``):

* ``log_estimate`` sums the decoder log-probability over every non-pad position of the
  complete target, so it INCLUDES the scored boundary symbol (``sentence_end`` 256 or
  ``document_end`` 257);
* ``source_bytes`` is ``(complete < 256).sum(-1)``: canonical bytes only, boundary,
  ``bos`` and ``pad`` excluded;
* ``boundary_symbols`` is 1 per row.

The default denominator is therefore ``source_bytes`` (the skeleton's "row's canonical
byte count", on the boundary-augmented task: the boundary symbol's nats are charged to
the canonical bytes). ``source_bytes_plus_boundary`` is the alternative reading, kept
as an explicit parameter; which one seals is an operator ruling, not this module's.

Fail LOUD (measurement path): every inadmissible input raises
:class:`~entropic_substrate.ladder.capture.reader.telemetry.TelemetryGap` naming where.
A ``zero_importance_weight_mass`` gap row is admissible (registered collapse is valid
negative evidence): it is counted per k and excluded from both means. A k whose rows
are all gaps reports ``None`` and omits its flat metric keys -- absent, never 0. A
non-gap row with a non-finite number is refused, so a "non-finite row" count would be
structurally zero and is not emitted. No threshold, margin or decision lives here.

FOR-1704 (L4d): the headline estimand is reported two ways beside each other. The MEAN OF RATIOS
averages each row's ``log_estimate / n`` and so lets a 2-byte sentence weigh as much as a
200-byte one (measured on FOR-1703: one 2-byte sentence supplied 92.6 percent of one reading).
The RATIO OF SUMS divides the summed ``log_estimate`` by the summed canonical bytes of the SAME
finite rows. A gap row contributes to neither sum. ``proposal`` selects which proposal's rows
are described (default q); a record that never evaluated the selected proposal (every v1 record)
raises rather than returning an empty projection, because an empty projection cannot be told
from registered collapse.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NoReturn

from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

__all__ = [
    "DEFAULT_DENOMINATOR",
    "DENOMINATORS",
    "HEADLINE_SAMPLES",
    "PROPOSALS",
    "ZERO_MASS_GAP",
    "project_elm",
    "require_importance_entry",
]

HEADLINE_SAMPLES = 64
DENOMINATORS = ("source_bytes", "source_bytes_plus_boundary")
# The closed proposal vocabulary of ``h-predictive-evaluation-v2`` (FOR-1704). q is the v1 proposal
# at its v1 keys; the others live under ``example["proposals"]``. Pinned equal to the writer's
# Literal by tests/test_h_elm_projection.py, never imported from a reader member.
PROPOSALS = ("q", "prior", "mixture")
DEFAULT_DENOMINATOR = DENOMINATORS[0]
ZERO_MASS_GAP = "zero_importance_weight_mass"
_ENTRY_FIELDS = ("log_estimate", "effective_sample_size", "likelihood_relative_mcse")
# ESS = 1 / sum(w_i^2) over k normalized weights lies in [1, k]; the slack absorbs only
# float rounding of the stored value, never a real excursion
_ESS_SLACK = 1e-9


def _gap(message: str) -> NoReturn:
    raise TelemetryGap(message)


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))  # pyright: ignore[reportArgumentType]
    except OverflowError:  # a JSON integer beyond float range is not a finite measurement
        return False


def require_importance_entry(entry: Mapping[str, Any], where: str) -> None:
    """The engine's admission predicate for one importance row.

    A finite row carries three finite numbers and no gap. A zero-importance-weight row
    carries a gap and three nulls. Every other combination is refused.
    """
    gap = entry.get("gap")
    values = {key: entry.get(key) for key in _ENTRY_FIELDS}
    if gap is None:
        for key, value in values.items():
            if not _finite_number(value):
                _gap(f"{where} has a non-finite {key}")
    elif gap == ZERO_MASS_GAP:
        present = [key for key, value in values.items() if value is not None]
        if present:
            _gap(f"{where} is a {ZERO_MASS_GAP} gap but carries {present}")
    else:
        _gap(f"{where} has an unknown gap {gap!r}")


def _counts(example: Mapping[str, Any], field: str, minimum: int, where: str) -> list[int]:
    value = example.get(field)
    if not isinstance(value, list) or not value:
        _gap(f"{where} has no {field} list: {value!r}")
    counts: list[int] = []
    for n in value:  # pyright: ignore[reportUnknownVariableType]
        if type(n) is not int or n < minimum:
            _gap(f"{where} has {field} {value!r}; every entry must be an integer >= {minimum}")
        counts.append(n)
    return counts


def _sample_count(key: object, where: str) -> int:
    if isinstance(key, str) and key.isdigit() and key == str(int(key)) and int(key) >= 2:
        return int(key)
    _gap(f"{where} has a sample count key {key!r} that is not an integer >= 2")


def _summary(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    finite = [v for v in values if v is not None]
    if not finite:
        return None, None
    return sum(finite) / len(finite), statistics.median(finite)


def _key(base: str, proposal: str, k: int) -> str:
    return f"eval/{base}_k{k}" if proposal == "q" else f"eval/{base}_{proposal}_k{k}"


def _proposal_rows(
    example: dict[str, Any], proposal: str, event: Mapping[str, Any], at: str
) -> Any:
    """The ``by_sample_count`` mapping of ONE proposal in one example (q: the v1 key)."""
    if proposal == "q":
        return example.get("by_sample_count")
    blocks = example.get("proposals")
    block = blocks.get(proposal) if isinstance(blocks, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(block, dict):
        _gap(
            f"{at} carries no {proposal!r} proposal (schema {event.get('schema')!r}): the "
            "selected proposal was not evaluated, which is not the same as every row being a gap"
        )
    return block.get("by_sample_count")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def project_elm(
    event: Mapping[str, Any],
    *,
    denominator: str = DEFAULT_DENOMINATOR,
    where: str = "evaluation",
    proposal: Literal["q", "prior", "mixture"] = "q",
) -> dict[str, Any]:
    """Project one evaluation event onto per-byte log-marginal and ESS/k, per k.

    Returns ``{"denominator", "headline", "metrics", "samples"}``. ``samples[str(k)]``
    holds the per-row values (``None`` on a gap row) in capture order, the row, finite
    and gap counts, the per-byte mean and the ESS/k median and mean; ``metrics`` holds
    the flat ``eval/*`` keys of the k with at least one finite row; ``headline`` is the
    k = 64 reading, ``present`` False (and both values ``None``) when the event did not
    evaluate k = 64.

    One call describes exactly one ``proposal``: its metrics and samples refer to the same rows.
    The q call returns every key it always did, plus the ratio-of-sums keys; a non-q call names
    its proposal in every flat key (``eval/<base>_<proposal>_k<k>``).
    """
    if denominator not in DENOMINATORS:
        raise ValueError(f"denominator {denominator!r} is not one of {DENOMINATORS}")
    if proposal not in PROPOSALS:
        raise ValueError(f"proposal {proposal!r} is not one of {PROPOSALS}")
    examples = event.get("examples")
    if not isinstance(examples, list) or not examples:
        _gap(f"{where} carries no examples")
    expected: list[int] | None = None
    per_byte: dict[int, list[float | None]] = {}
    ratios: dict[int, list[float | None]] = {}
    # ratio of sums: summed log_estimate and summed denominator bytes over the SAME finite rows
    numerator: dict[int, float] = {}
    sum_bytes: dict[int, int] = {}
    for index, example in enumerate(examples):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        at = f"{where} example {index}"
        if not isinstance(example, dict):
            _gap(f"{at} is not an object")
        source = _counts(example, "source_bytes", 1, at)  # pyright: ignore[reportUnknownArgumentType]
        boundary = _counts(example, "boundary_symbols", 0, at)  # pyright: ignore[reportUnknownArgumentType]
        if len(boundary) != len(source):
            _gap(f"{at} has {len(source)} source_bytes but {len(boundary)} boundary_symbols")
        bytes_ = (
            source
            if denominator == "source_bytes"
            else [s + b for s, b in zip(source, boundary, strict=True)]
        )
        by_count = _proposal_rows(example, proposal, event, at)  # pyright: ignore[reportUnknownArgumentType]
        if not isinstance(by_count, dict) or not by_count:
            _gap(f"{at} has no by_sample_count")
        counts = sorted(_sample_count(key, at) for key in by_count)  # pyright: ignore[reportUnknownVariableType]
        if expected is None:
            expected = counts
        elif counts != expected:
            _gap(f"{at} carries sample counts {counts}, but the first example carries {expected}")
        for k in counts:
            rows = by_count[str(k)]  # pyright: ignore[reportUnknownVariableType]
            if not isinstance(rows, list) or len(rows) != len(bytes_):  # pyright: ignore[reportUnknownArgumentType]
                _gap(f"{at} k={k} has rows {rows!r} that do not match {len(bytes_)} source_bytes")
            for row, n in zip(rows, bytes_, strict=True):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                if not isinstance(row, dict):
                    _gap(f"{at} k={k} has a non-object row")
                require_importance_entry(row, f"{at} k={k}")  # pyright: ignore[reportUnknownArgumentType]
                if row.get("gap") is not None:  # pyright: ignore[reportUnknownMemberType]
                    per_byte.setdefault(k, []).append(None)
                    ratios.setdefault(k, []).append(None)
                    continue
                ess = float(row["effective_sample_size"])  # pyright: ignore[reportUnknownArgumentType]
                if not 1 - _ESS_SLACK <= ess <= k * (1 + _ESS_SLACK):
                    _gap(f"{at} k={k} has effective_sample_size {ess} outside [1, {k}]")
                per_byte.setdefault(k, []).append(float(row["log_estimate"]) / n)  # pyright: ignore[reportUnknownArgumentType]
                numerator[k] = numerator.get(k, 0.0) + float(row["log_estimate"])  # pyright: ignore[reportUnknownArgumentType]
                sum_bytes[k] = sum_bytes.get(k, 0) + n
                ratios.setdefault(k, []).append(ess / k)
    if expected is None:  # unreachable: examples is non-empty and the first example sets it
        _gap(f"{where} carries no sample counts")

    samples: dict[str, dict[str, Any]] = {}
    metrics: dict[str, float] = {}
    for k in expected:
        mean_per_byte, _ = _summary(per_byte[k])
        mean_ratio, median_ratio = _summary(ratios[k])
        finite = sum(1 for v in per_byte[k] if v is not None)
        ratio_of_sums = numerator[k] / sum_bytes[k] if sum_bytes.get(k, 0) > 0 else None
        samples[str(k)] = {
            "ess_ratio": ratios[k],
            "ess_ratio_mean": mean_ratio,
            "ess_ratio_median": median_ratio,
            "finite_rows": finite,
            "gap_rows": len(per_byte[k]) - finite,
            "log_marginal_per_byte": per_byte[k],
            "mean_log_marginal_per_byte": mean_per_byte,
            "log_marginal_per_byte_ratio_of_sums": ratio_of_sums,
            "rows": len(per_byte[k]),
        }
        metrics[_key("finite_rows", proposal, k)] = float(finite)
        if mean_per_byte is not None and mean_ratio is not None and median_ratio is not None:
            metrics[_key("log_marginal_per_byte", proposal, k)] = mean_per_byte
            metrics[_key("ess_ratio_median", proposal, k)] = median_ratio
            metrics[_key("ess_ratio_mean", proposal, k)] = mean_ratio
        if ratio_of_sums is not None:  # absent, never 0, when no row of the proposal is finite
            metrics[_key("log_marginal_per_byte_ratio_of_sums", proposal, k)] = ratio_of_sums
    headline = samples.get(str(HEADLINE_SAMPLES))
    return {
        "denominator": denominator,
        "headline": {
            "ess_ratio_median": None if headline is None else headline["ess_ratio_median"],
            "log_marginal_per_byte": None
            if headline is None
            else headline["mean_log_marginal_per_byte"],
            "log_marginal_per_byte_ratio_of_sums": None
            if headline is None
            else headline["log_marginal_per_byte_ratio_of_sums"],
            "present": headline is not None,
            "samples": HEADLINE_SAMPLES,
        },
        "metrics": metrics,
        "proposal": proposal,
        "samples": samples,
    }
