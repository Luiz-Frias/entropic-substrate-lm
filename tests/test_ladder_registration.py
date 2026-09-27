"""Registration preparation binds scientific arithmetic to captured bytes."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, canonical_digest
from entropic_substrate.ladder.analyze.ladder_registration import (
    _d1_runs,
    paired_contrasts,
    prepare_registration,
    read_statistical_design,
    report_registered_contrasts,
)
from entropic_substrate.ladder.capture.reader.telemetry import read_d1_run_evidence_bytes

from .d1_telemetry_fixtures import valid_events

pytestmark = pytest.mark.unit


def specification() -> dict[str, object]:
    return {
        "schema": "d-ladder-statistical-design-v1",
        "run_class": "pilot",
        "registered_design_digest": None,
        "family_digest": "4" * 64,
        "construction_digests": {"both": "3" * 64, "sg": "5" * 64},
        "seeds": [7, 8],
        "contrasts": [
            {
                "contrast_id": "both-vs-sg",
                "terms": {"both": 1.0, "sg": -1.0},
                "metric": "nats_per_byte",
                "units": "nats/byte",
                "eval_steps": [1, 2],
                "source_bytes": 8,
                "kind": "superiority",
                "expected_effect": 0.01,
                "lower_bound": 0.0,
                "upper_bound": None,
            }
        ],
    }


def sealed(value: dict[str, object]) -> tuple[bytes, str]:
    payload = json.dumps(value, allow_nan=False).encode()
    return payload, hashlib.sha256(payload).hexdigest()


def test_design_binds_exact_bytes_and_explicit_endpoint() -> None:
    payload, digest = sealed(specification())
    design = read_statistical_design(payload, expected_sha256=digest)
    assert design.source_sha256 == digest
    assert design.seeds == (7, 8)
    assert design.contrasts[0].eval_steps == (1, 2)
    assert design.contrasts[0].hypothesis.expected_effect == 0.01
    with pytest.raises(ValueError, match="digest"):
        read_statistical_design(payload + b" ", expected_sha256=digest)


@pytest.mark.parametrize("key", ["eval_steps", "units", "expected_effect", "lower_bound"])
def test_missing_scientific_choices_never_default(key: str) -> None:
    value = specification()
    del value["contrasts"][0][key]
    with pytest.raises(ValueError):
        read_statistical_design(sealed(value)[0], expected_sha256=sealed(value)[1])


@pytest.mark.parametrize(
    "key,value",
    [
        ("eval_steps", []),
        ("eval_steps", [2, 1]),
        ("eval_steps", [1, 1]),
        ("source_bytes", True),
        ("terms", {"both": 1, "sg": 1}),
        ("terms", {"both": 1, "unknown": -1}),
        ("expected_effect", True),
        ("kind", "equivalence"),
    ],
)
def test_invalid_endpoint_or_hypothesis_refused(key: str, value: object) -> None:
    data = specification()
    data["contrasts"][0][key] = value
    with pytest.raises(ValueError):
        read_statistical_design(sealed(data)[0], expected_sha256=sealed(data)[1])


def test_empty_family_duplicate_seed_and_smoke_refused() -> None:
    for change in ({"seeds": [7, 7]}, {"contrasts": []}, {"run_class": "smoke"}):
        value = specification() | change
        with pytest.raises(ValueError):
            read_statistical_design(sealed(value)[0], expected_sha256=sealed(value)[1])


def test_duplicate_json_keys_cannot_silently_replace_sealed_policy() -> None:
    payload = b'{"schema":"old","schema":"d-ladder-statistical-design-v1"}'
    with pytest.raises(ValueError, match="duplicate"):
        read_statistical_design(payload, expected_sha256=hashlib.sha256(payload).hexdigest())


def test_full_family_preserves_factorial_weights() -> None:
    value = specification()
    value["construction_digests"] = dict.fromkeys(("both", "sg", "none", "reg"), "3" * 64)
    interaction = deepcopy(value["contrasts"][0])
    interaction.update(contrast_id="interaction", terms={"both": 1, "sg": -1, "none": 1, "reg": -1})
    value["contrasts"].append(interaction)
    result = read_statistical_design(sealed(value)[0], expected_sha256=sealed(value)[1])
    assert len(result.contrasts) == 2
    assert len(result.contrasts[1].terms) == 4


def journal(arm: str, seed: int, run_class: str = "pilot") -> bytes:
    events = [deepcopy(event.manifest) for event in valid_events(arm=D1Arm(arm))]
    construction = "3" * 64 if arm == "both" else "5" * 64
    for event in events:
        event["run_class"] = run_class
        event["construction_digest"] = construction
    plan = events[0]["payload"]["plan"]
    plan.update(seed=seed, run_class=run_class, construction_digest=construction)
    events[0]["payload"]["plan_digest"] = canonical_digest(plan)
    return (
        "\n".join(
            json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for event in events
        )
        + "\n"
    ).encode()


def test_weighted_endpoints_come_through_authoritative_reader() -> None:
    design = read_statistical_design(
        sealed(specification())[0], expected_sha256=sealed(specification())[1]
    )
    runs = tuple(
        read_d1_run_evidence_bytes(journal(arm, seed), journal_path=f"{arm}-{seed}.jsonl")
        for arm in ("both", "sg")
        for seed in (7, 8)
    )
    contrasts = paired_contrasts(_d1_runs(runs, design), design)
    assert [pair.left for pair in contrasts[0].pairs] == [0.0, 0.0]
    assert [pair.left_exposure for pair in contrasts[0].pairs] == [8, 8]
    with pytest.raises(ValueError, match="complete"):
        paired_contrasts(_d1_runs(runs[:-1], design), design)
    changed = specification()
    changed["contrasts"][0]["source_bytes"] = 9
    with pytest.raises(ValueError, match="exposure"):
        paired_contrasts(
            _d1_runs(runs, design),
            read_statistical_design(sealed(changed)[0], expected_sha256=sealed(changed)[1]),
        )


def test_confirmatory_report_rechecks_bytes_and_keeps_one_population(tmp_path: Path) -> None:
    from entropic_substrate.ladder.analyze.d1_capture import D1AdmittedAnalysisCapture
    from entropic_substrate.ladder.capture.custody.d1_capture_artifacts import (
        D1ArtifactRole,
        identify_d1_artifact,
    )

    artifacts = []
    for arm in ("both", "sg"):
        for seed in (7, 8):
            path = tmp_path / f"{arm}-{seed}.jsonl"
            path.write_bytes(journal(arm, seed, "confirmatory"))
            artifacts.append(
                identify_d1_artifact(
                    tmp_path, path, role=D1ArtifactRole.JOURNAL, dvc_pointer_path=f"{path.name}.dvc"
                )
            )
    capture = D1AdmittedAnalysisCapture("d" * 64, "c" * 64, "a" * 64, (), (), (), tuple(artifacts))
    data = specification() | {"run_class": "confirmatory", "registered_design_digest": "d" * 64}
    payload, digest = sealed(data)
    reports = report_registered_contrasts(capture, tmp_path, payload, digest)
    assert reports[0][1].seed_ids == (7, 8)
    assert reports[0][1].df == 1
    assert reports[0][1].secondary.deciding is False
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="identity changed"):
        report_registered_contrasts(capture, tmp_path, payload, digest)


def test_real_excluded_capture_retains_tiny_variance_exclusion(tmp_path: Path) -> None:
    from .test_d1_excluded_capture import captured_pilot

    service, preparation, _ = captured_pilot(tmp_path)
    receipt = service.finalize(preparation)
    payload, digest = sealed(specification())
    with pytest.raises(ValueError, match="FULL"):
        prepare_registration((receipt,), payload, digest, max_seeds=100)
