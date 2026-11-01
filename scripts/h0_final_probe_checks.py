"""Run the H0 final-probe check suite over one probe's collected evidence (ruling D6).

Usage::

    uv run python scripts/h0_final_probe_checks.py --plan PLAN.json

``PLAN.json`` names the evidence per check; only the checks it names run, in K order.
Each prints ONE JSON document (``sort_keys=True``) on its own stdout line. The exit code
is 0 only when every requested check is ``PASS``; ``FAIL`` and ``UNPROVEN`` exit 1. A
malformed plan (an unknown check, a missing field, an unreadable file) raises: that is a
defect in the operator's input, not a verdict.

Plan sections (paths as given, absolute or relative to the working directory)::

    "k1_chain":   {"run_root", "final_receipt": {path, sha256}, "stopped_receipt": {...},
                   "profile": manifest.json, "source_root",
                   "max_source_bytes", "max_artifact_bytes"}
    "k1_request": {"collection": collection.json, "previous_request": request.json,
                   "resumed_request": request.json}
    "k1_seam":    {"segments": [[metrics.jsonl, worker-log.jsonl], ...]}
    "k2":         {"records": [controller-result.json, ...], "requests": [request.json, ...]}
    "k3":         {"metrics": [metrics.jsonl, ...], "request_digests": [hex, ...]}
    "k4":         {"worker_logs": [worker-log.jsonl, ...]}
    "k5":         {"receipts": [generation.json, ...]}
    "k6":         {"metrics": [metrics.jsonl, ...], and one of
                   "evaluated_samples": [k, ...] or "profile": manifest.json}
    "k7":         {"arms": {label: [[metrics.jsonl, worker-log.jsonl], ...]}}
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.codec.strict_json import strict_loads
from entropic_substrate.ladder.analyze.h_final_probe_checks import (
    CheckResult,
    check_analysis_surface,
    check_capture,
    check_completion,
    check_generation,
    check_importance_curve,
    check_read_floor,
    check_resume_chain,
    check_resume_request,
    check_resume_seam,
    evaluated_samples_from_manifest,
)
from entropic_substrate.ladder.run.worker_request import (
    ProbeRequest,
    parse_probe_request,
    refuse_pair,
)

_MAX_REQUEST_BYTES = 1_000_000


def _json(path: object) -> dict[str, Any]:
    source = Path(str(path))

    def refuse(message: str) -> NoReturn:
        raise ValueError(f"{source}: {message}")

    value = strict_loads(source.read_bytes(), fail=refuse)
    if not isinstance(value, dict):
        raise ValueError(f"{source}: not a JSON object")
    return cast(dict[str, Any], value)


def _request(path: object) -> ProbeRequest:
    request = parse_probe_request(Path(str(path)).read_bytes(), max_bytes=_MAX_REQUEST_BYTES)
    if type(request) is not ProbeRequest:
        # FOR-1693: the H0 checks read single-profile arms; a pair's checks belong to FOR-1694
        refuse_pair("h0_final_probe_checks")
    return request


def _pairs(value: Sequence[Sequence[str]]) -> list[tuple[Path, Path]]:
    return [(Path(metrics), Path(log)) for metrics, log in value]


def _k1_chain(spec: Mapping[str, Any]) -> CheckResult:
    return check_resume_chain(
        Path(spec["run_root"]),
        artifact_from_mapping(spec["final_receipt"]),
        artifact_from_mapping(spec["stopped_receipt"]),
        HProfile.from_manifest(_json(spec["profile"])),
        Path(spec["source_root"]),
        max_source_bytes=int(spec["max_source_bytes"]),
        max_artifact_bytes=int(spec["max_artifact_bytes"]),
    )


def _k1_request(spec: Mapping[str, Any]) -> CheckResult:
    return check_resume_request(
        _json(spec["collection"]),
        _request(spec["previous_request"]),
        _request(spec["resumed_request"]),
    )


def _sample_counts(values: object) -> tuple[int, ...]:
    """Exact JSON integers only: ``int()`` would truncate 8.9 to 8 and read True as 1."""
    if not isinstance(values, list):
        raise ValueError(f"k6 evaluated_samples must be a list, got {values!r}")
    counts: list[int] = []
    for value in cast(list[object], values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"k6 evaluated_samples holds a non-integer count: {value!r}")
        counts.append(value)
    return tuple(counts)


def _k6(spec: Mapping[str, Any]) -> CheckResult:
    if ("evaluated_samples" in spec) == ("profile" in spec):
        raise ValueError("k6 needs exactly one of evaluated_samples or profile")
    samples = (
        _sample_counts(spec["evaluated_samples"])
        if "evaluated_samples" in spec
        else evaluated_samples_from_manifest(_json(spec["profile"]))
    )
    return check_importance_curve([Path(p) for p in spec["metrics"]], samples)


CHECKS: dict[str, Callable[[Mapping[str, Any]], CheckResult]] = {
    "k1_chain": _k1_chain,
    "k1_request": _k1_request,
    "k1_seam": lambda spec: check_resume_seam(_pairs(spec["segments"])),
    "k2": lambda spec: check_completion(
        [_json(p) for p in spec["records"]], [_request(p) for p in spec["requests"]]
    ),
    "k3": lambda spec: check_capture(
        [Path(p) for p in spec["metrics"]], [str(d) for d in spec["request_digests"]]
    ),
    "k4": lambda spec: check_read_floor([Path(p) for p in spec["worker_logs"]]),
    "k5": lambda spec: check_generation([_json(p) for p in spec["receipts"]]),
    "k6": _k6,
    "k7": lambda spec: check_analysis_surface(
        {label: _pairs(runs) for label, runs in spec["arms"].items()}
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the H0 final-probe check suite over one probe's evidence."
    )
    parser.add_argument("--plan", type=Path, required=True, help="the evidence plan JSON")
    args = parser.parse_args(argv)
    plan = _json(args.plan)
    unknown = sorted(set(plan) - set(CHECKS))
    if unknown or not plan:
        raise ValueError(f"plan names unknown checks {unknown} (known: {sorted(CHECKS)})")
    verdicts: list[str] = []
    for name, run in CHECKS.items():
        if name not in plan:
            continue
        result = run(plan[name])
        print(json.dumps(result.to_json(), sort_keys=True))
        verdicts.append(result.verdict)
    return 0 if all(verdict == "PASS" for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
