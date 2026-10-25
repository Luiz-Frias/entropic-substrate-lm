"""Compact pre-update dynamics measurements; never semantic-information percentages."""

import math

import torch

from entropic_substrate.core.sentence.model import SentenceReadout
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record

# v2 (FOR-1673) labels a zero-sentence prediction entry with its absence reason. v1 rows
# (capture h0-plumbing-1k-v3.HEpP7g, both arms, measured: 250 warm-up and 18 zero-sentence
# prediction rows per arm, the latter carrying a bare null range) stay readable; the
# emitter writes v2 only and no capture is rewritten.
SCHEMA = "h-mechanism-evidence-v2"
LEGACY_SCHEMA = "h-mechanism-evidence-v1"
ABSENT_STATUS = "structurally_absent"
NO_SENTENCES = "no_sentence_states"


def _range(values: torch.Tensor) -> list[float]:
    values = values.detach().double()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("mechanism norms must be finite and nonnegative")
    return [float(values.min()), float(values.mean()), float(values.max())]


def _norms(values: torch.Tensor, *, absent_reason: str | None = None) -> dict[str, object]:
    if values.dtype != torch.float32:
        raise ValueError("mechanism state precision differs from FP32 carrier contract")
    # Accumulation is diagnostic-only FP64; it does not replace the carrier.
    norms = torch.linalg.vector_norm(values.detach().double(), dim=-1)
    entry: dict[str, object] = {
        "count": norms.numel(),
        "zero_count": int((norms == 0).sum()),
        "dtype": str(values.dtype),
        "range": _range(norms) if norms.numel() else None,
    }
    if absent_reason is not None and not norms.numel():
        entry |= {"status": ABSENT_STATUS, "reason": absent_reason}
    return entry


def collect_mechanism(
    readout: SentenceReadout | None, profile: HProfile, *, phase: str, zero_views: list[int]
) -> dict[str, object]:
    if phase == "codec_warmup":
        if readout is not None:
            raise ValueError("codec warm-up cannot produce a spine readout")
        return {"schema": SCHEMA, "status": ABSENT_STATUS, "reason": "codec_warmup"}
    if phase != "prediction" or readout is None:
        raise ValueError("prediction requires its active readout")
    prediction = readout.prediction
    diagnostics = prediction.diagnostics
    if diagnostics is None:
        raise ValueError("active spine consumer lacks solver diagnostics")
    shape = prediction.states.shape[:-1]
    iterations = diagnostics.iterations
    if iterations.shape != shape or iterations.dtype != torch.int64:
        raise ValueError("solver iteration evidence shape or dtype mismatch")
    values = {
        "residual": diagnostics.residual_norm,
        "tolerance": diagnostics.tolerance,
        "metric_action": diagnostics.metric_action_norm,
        "damping_action": diagnostics.damping_action_norm,
        "update": diagnostics.update_norm,
    }
    if any(
        v.shape != shape or v.dtype != torch.float32 or v.requires_grad for v in values.values()
    ):
        raise ValueError("solver diagnostics must be detached FP32 per-position measurements")
    if bool((diagnostics.tolerance <= 0).any()):
        raise ValueError("solver tolerance must be positive")
    distinct, counts = iterations.detach().flatten().cpu().unique(sorted=True, return_counts=True)
    solver = {
        "method": "conjugate_gradient" if profile.consumer == "fisher" else "analytical_isotropic",
        "positions": iterations.numel(),
        "damping": profile.solver.damping,
        "iterations": [[int(i), int(n)] for i, n in zip(distinct, counts, strict=True)],
        "max_residual_ratio": float(
            (diagnostics.residual_norm.double() / diagnostics.tolerance.double()).max()
        ),
        **{key: _range(value) for key, value in values.items()},
    }
    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "measured",
        "consumer": profile.consumer,
        "sentences": _norms(prediction.sentences, absent_reason=NO_SENTENCES),
        "states": _norms(prediction.states),
        "sample": _norms(readout.sample),
        "solver": solver,
        "zero_views": zero_views,
    }
    admit_mechanism(
        result,
        profile,
        phase="prediction",
        batch=readout.sample.shape[0],
        history=prediction.sentences.shape[1],
    )
    return result


def _number(value: object) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("mechanism measurement must be finite and nonnegative")
    return float(value)


def _read_range(value: object) -> tuple[float, float, float]:
    if type(value) is not list or len(value) != 3:
        raise ValueError("mechanism range requires min, mean and max")
    low, mean, high = (_number(v) for v in value)
    # FP64 mean([0.1] * 3) is 0.10000000000000002; retain the measured mean.
    if (
        low > high
        or (mean < low and not math.isclose(mean, low, rel_tol=1e-12, abs_tol=0))
        or (mean > high and not math.isclose(mean, high, rel_tol=1e-12, abs_tol=0))
    ):
        raise ValueError("mechanism range does not reconcile")
    return low, mean, high


def _read_norms(value: object, count: int, *, spherical: bool, labelled: bool = False) -> None:
    # Shape is chosen by schema label (labelled) plus count only; both key sets are exact.
    absent = labelled and count == 0
    row = recovery_record(
        value,
        "count zero_count dtype range status reason" if absent else "count zero_count dtype range",
    )
    zeros = row["zero_count"]
    if (
        type(row["count"]) is not int
        or row["count"] != count
        or row["dtype"] != "torch.float32"
        or type(zeros) is not int
        or not 0 <= zeros <= count
    ):
        raise ValueError("mechanism position count, zero count or dtype mismatch")
    if count == 0:
        if row["range"] is not None:
            raise ValueError("empty history must not invent norm measurements")
        if absent and (row["status"] != ABSENT_STATUS or row["reason"] != NO_SENTENCES):
            raise ValueError("empty history requires its structural absence label")
        return
    low, mean, high = _read_range(row["range"])
    if (
        (zeros == count and high != 0)
        or (zeros < count and mean == 0)
        or ((low == 0) != (zeros > 0))
    ):
        raise ValueError("mechanism zero counts and norms disagree")
    if spherical and (abs(low - 1) > 1e-5 or abs(high - 1) > 1e-5):
        raise ValueError("spherical mechanism norms violate the declared construction")


def admit_mechanism(
    value: object, profile: HProfile, *, phase: str, batch: int, history: int
) -> None:
    """Validate measured summaries against the declared construction and source grid."""
    if phase == "codec_warmup":
        if value not in (
            {"schema": SCHEMA, "status": ABSENT_STATUS, "reason": "codec_warmup"},
            {"schema": LEGACY_SCHEMA, "status": ABSENT_STATUS, "reason": "codec_warmup"},
        ):
            raise ValueError("codec warm-up cannot claim active spine evidence")
        return
    if phase != "prediction":
        raise ValueError("unknown mechanism phase")
    row = recovery_record(value, "schema status consumer sentences states sample solver zero_views")
    if (
        row["schema"] not in (SCHEMA, LEGACY_SCHEMA)
        or row["status"] != "measured"
        or row["consumer"] != profile.consumer
    ):
        raise ValueError("mechanism evidence construction mismatch")
    _read_norms(
        row["sentences"],
        batch * history,
        spherical=profile.law.mode == "sphere",
        labelled=row["schema"] == SCHEMA,
    )
    _read_norms(row["states"], batch * (history + 1), spherical=profile.spine.carrier == "sphere")
    _read_norms(row["sample"], batch, spherical=profile.law.mode == "sphere")
    zeros = row["zero_views"]
    if (
        type(zeros) is not list
        or len(zeros) != profile.spine.layers
        or any(type(n) is not int or n < 0 for n in zeros)
    ):
        raise ValueError("mechanism zero-view counts must cover each spine block")
    solver = recovery_record(
        row["solver"],
        "method positions damping iterations max_residual_ratio residual tolerance metric_action damping_action update",
    )
    count = batch * (history + 1)
    expected = "conjugate_gradient" if profile.consumer == "fisher" else "analytical_isotropic"
    if (
        solver["method"] != expected
        or type(solver["positions"]) is not int
        or solver["positions"] != count
        or _number(solver["damping"]) != profile.solver.damping
        or _number(solver["max_residual_ratio"]) > 1
    ):
        raise ValueError("solver evidence violates its construction or convergence contract")
    histogram = solver["iterations"]
    if type(histogram) is not list or not histogram:
        raise ValueError("solver requires an iteration histogram")
    total, previous = 0, -1
    for item in histogram:
        if (
            type(item) is not list
            or len(item) != 2
            or any(type(n) is not int for n in item)
            or not previous < item[0] <= profile.solver.max_iterations
            or item[1] < 1
        ):
            raise ValueError("solver iteration histogram is invalid")
        previous, total = item[0], total + item[1]
    if total != count:
        raise ValueError("solver iteration population mismatch")
    ranges = {
        name: _read_range(solver[name])
        for name in ("residual", "tolerance", "metric_action", "damping_action", "update")
    }
    residual_max = ranges["residual"][2]
    tolerance_min, _, tolerance_max = ranges["tolerance"]
    if tolerance_min <= 0 or residual_max > tolerance_max:
        raise ValueError("solver residual range violates its tolerance")
    ratio = _number(solver["max_residual_ratio"])
    lower, upper = residual_max / tolerance_max, residual_max / tolerance_min
    if (ratio < lower and not math.isclose(ratio, lower, rel_tol=1e-12, abs_tol=0)) or (
        ratio > upper and not math.isclose(ratio, upper, rel_tol=1e-12, abs_tol=0)
    ):
        raise ValueError("solver residual ratio does not reconcile with its ranges")
    if ranges["tolerance"][0] < profile.solver.absolute_tolerance:
        # The configured scalar is rounded once when the FP32 solve constructs it.
        if not math.isclose(
            ranges["tolerance"][0], profile.solver.absolute_tolerance, rel_tol=1e-6
        ):
            raise ValueError("solver tolerance is below its absolute threshold")
    for damped, update in zip(ranges["damping_action"], ranges["update"], strict=True):
        if not math.isclose(damped, profile.solver.damping * update, rel_tol=1e-5, abs_tol=1e-12):
            raise ValueError("solver damping action does not reconcile")
    if profile.consumer == "isotropic" and (
        histogram != [[0, count]]
        or ranges["residual"] != (0, 0, 0)
        or ranges["metric_action"] != ranges["update"]
    ):
        raise ValueError("isotropic analytical evidence is inconsistent")
