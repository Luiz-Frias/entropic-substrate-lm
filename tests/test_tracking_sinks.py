"""Generic tracking-run composition without changing the legacy sink contract."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.ports.run_descriptor import LegacyMetricExtractor
from entropic_substrate.ladder.capture.reader.sinks import (
    CompositeSink,
    ExtractedMetrics,
    JsonlSink,
    MlflowSink,
    TrackingRunDescriptor,
    build_sink,
    build_sink_for_run,
    config_hash,
    git_sha,
)
from helpers import tiny_config
from test_sinks import FakeMlflowClient

pytestmark = pytest.mark.unit


def _descriptor(**overrides: object) -> TrackingRunDescriptor:
    values: dict[str, object] = {
        "run_name": "tracked-run",
        "tiny": False,
        "params": {"exp": "tracked-run", "seed": 7},
        "tags": {"exp": "tracked-run", "config_hash": "abc123"},
        "config_hash": "abc123",
        "dataset_pointer_path": None,
        "dataset_split_spec": None,
    }
    values.update(overrides)
    return TrackingRunDescriptor(**values)  # type: ignore[arg-type]


def test_tracking_descriptor_owns_immutable_copies_of_metadata() -> None:
    params: dict[str, object] = {"seed": 7}
    tags: dict[str, object] = {"arm": "both"}
    descriptor = _descriptor(params=params, tags=tags)

    params["seed"] = 99
    tags["arm"] = "none"
    assert descriptor.params == {"seed": 7}
    assert descriptor.tags == {"arm": "both"}
    with pytest.raises(TypeError):
        descriptor.params["later"] = "forbidden"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        descriptor.run_name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_name", " ", "run_name"),
        ("tiny", 1, "tiny"),
        ("params", {"": 1}, "params"),
        ("tags", {1: "value"}, "tags"),
        ("config_hash", "", "config_hash"),
    ],
)
def test_tracking_descriptor_rejects_ambiguous_metadata(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _descriptor(**{field: value})


def test_generic_builder_preserves_jsonl_authority_when_tracking_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    sink = build_sink_for_run(_descriptor(), results_path=tmp_path / "run.jsonl")
    assert type(sink) is JsonlSink


def test_generic_builder_threads_descriptor_metadata_to_mlflow(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    descriptor = _descriptor(
        params={"exp": "d1-both", "seed": 11, "plan": "p" * 64},
        tags={"exp": "d1-both", "family": "f" * 64},
    )
    sink = build_sink_for_run(
        descriptor,
        results_path=tmp_path / "run.jsonl",
        client=client,
    )

    assert type(sink) is CompositeSink
    mlflow = sink.best_effort[0]
    assert type(mlflow) is MlflowSink
    run = client._runs[mlflow.run_id]
    assert run.data.params == {"exp": "d1-both", "seed": "11", "plan": "p" * 64}
    assert run.data.tags["family"] == "f" * 64
    assert run.data.tags["mlflow.runName"] == descriptor.run_name


def test_legacy_builder_is_an_exact_metadata_adapter(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    cfg.name = "e1-fisher"
    cfg.train.seed = 17
    client = FakeMlflowClient()
    sink = build_sink(cfg, tiny=False, results_path=tmp_path / "legacy.jsonl", client=client)

    assert type(sink) is CompositeSink
    mlflow = sink.best_effort[0]
    assert type(mlflow) is MlflowSink
    run = client._runs[mlflow.run_id]
    expected_hash = config_hash(cfg)
    assert run.data.params == {
        "exp": "e1-fisher",
        "seed": "17",
        "config_hash": expected_hash,
        "git_sha": git_sha(),
    }
    assert {key: run.data.tags[key] for key in ("exp", "config_hash", "mlflow.runName")} == {
        "exp": "e1-fisher",
        "config_hash": expected_hash,
        "mlflow.runName": "e1-fisher",
    }


class _InjectedExtractor:
    def __init__(self) -> None:
        self.seen: list[object] = []

    def extract(self, event: object) -> ExtractedMetrics | None:
        self.seen.append(event)
        return ExtractedMetrics(step=23, metrics={"custom/measured/value": 1.2345678901234567})


def test_mlflow_sink_uses_an_injected_extractor_without_mutating_the_event() -> None:
    client = FakeMlflowClient()
    extractor = _InjectedExtractor()
    event = {"event": "not-a-legacy-eval", "nested": {"value": 1}}
    before = {"event": event["event"], "nested": dict(event["nested"])}  # type: ignore[arg-type]
    sink = MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="run",
        metric_extractor=extractor,
        client=client,
    )

    sink.log(event)
    sink.finish()

    assert extractor.seen == [event]
    assert event == before
    assert client._runs[sink.run_id].data.metrics == {"custom/measured/value": 1.2345678901234567}


def test_legacy_extractor_ignores_a_non_mapping_and_a_non_eval_record() -> None:
    """The runtime refusals survive the static narrowing (return None, never raise)."""
    extractor = LegacyMetricExtractor()
    assert extractor.extract("not-a-mapping") is None  # type: ignore[arg-type]
    assert extractor.extract({"event": "train", "loss": 1.0}) is None
    assert extractor.extract({"event": "eval", "step": 4, "loss": 1.5}) is not None


# ---------------------------------------------------- typing fixture (FOR-1569)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_D1_MARKER = "# D1-MUST-FAIL"
_H_EXTRACTOR_MARKER = "# H-MUST-FAIL"
_LEGACY_EXTRACTOR_MARKER = "# LEGACY-MUST-FAIL"
_D1_OBSERVER_MARKER = "# D1-OBSERVER-MUST-FAIL"
_TYPING_FIXTURE = """
from pathlib import Path

from entropic_substrate.ladder.capture.adapters.composite import build_sink_for_run
from entropic_substrate.ladder.capture.adapters.d1_observability import D1MlflowObserver
from entropic_substrate.ladder.capture.adapters.d1_tracking import D1MetricExtractor
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    LegacyMetricExtractor,
    TrackingRunDescriptor,
)
from entropic_substrate.ladder.run.h_metric_extractor import HTrainingStepExtractor

descriptor = TrackingRunDescriptor(run_name="r", tiny=True, params={}, tags={})
results = Path("results.jsonl")
h_extractor = HTrainingStepExtractor(warmup_steps=0)
legacy_extractor = LegacyMetricExtractor()
h_sink = build_sink_for_run(
    descriptor, results_path=results, metric_extractor=h_extractor
)
legacy_sink = build_sink_for_run(
    descriptor, results_path=results, metric_extractor=legacy_extractor
)
default_sink = build_sink_for_run(descriptor, results_path=results)
d1_extractor = D1MetricExtractor()
d1_sink = build_sink_for_run(
    descriptor, results_path=results, metric_extractor=d1_extractor  # D1-MUST-FAIL
)
# Direct-call negatives: H and Legacy extractors cannot accept non-mapping inputs
h_extract_bad = h_extractor.extract("x")  # H-MUST-FAIL
legacy_extract_bad = legacy_extractor.extract(1)  # LEGACY-MUST-FAIL
# D1MlflowObserver with incompatible extractor
d1_observer_bad = D1MlflowObserver(client=object(), run_id="r", extractor=h_extractor)  # D1-OBSERVER-MUST-FAIL
d1_observer_ok = D1MlflowObserver(client=object(), run_id="r", extractor=d1_extractor)

"""


def test_door_rejects_the_d1_extractor_and_admits_the_h_and_legacy_ones(
    tmp_path: Path,
) -> None:
    """The narrowed extractor signatures are type errors at the four wrong-input sites.

    FOR-1569: ``build_sink_for_run(metric_extractor=D1MetricExtractor())``, a direct
    ``extract`` call on the H and legacy extractors with a non-mapping, and a
    ``D1MlflowObserver`` built with the H extractor. The controls (H, legacy and
    default through the door, the D1 extractor through the observer) must stay clean.

    Run by the venv-pinned checker ``just typecheck`` uses, from the repo root so
    the repo's basedpyright config applies and ``entropic_substrate`` resolves
    (tests are excluded from the repo's own pass).
    """
    fixture = tmp_path / "typing_fixture.py"
    fixture.write_text(_TYPING_FIXTURE, encoding="utf-8")
    # Find all MUST-FAIL marker lines
    must_fail_lines = [
        number
        for number, line in enumerate(_TYPING_FIXTURE.splitlines(), start=1)
        if any(
            m in line
            for m in [
                _D1_MARKER,
                _H_EXTRACTOR_MARKER,
                _LEGACY_EXTRACTOR_MARKER,
                _D1_OBSERVER_MARKER,
            ]
        )
    ]
    result = subprocess.run(
        [sys.executable, "-m", "basedpyright", "--outputjson", str(fixture)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    report = json.loads(result.stdout)
    diagnostics = [
        (item.get("rule"), item["range"]["start"]["line"] + 1)
        for item in report["generalDiagnostics"]
    ]
    assert report["summary"]["filesAnalyzed"] == 1
    # Expect exactly 4 reportArgumentType diagnostics, one per MUST-FAIL marker
    assert len(diagnostics) == 4, (diagnostics, result.stderr)
    assert all(rule == "reportArgumentType" for rule, _ in diagnostics), (
        diagnostics,
        result.stderr,
    )
    assert sorted([line for _, line in diagnostics]) == sorted(must_fail_lines), (
        diagnostics,
        must_fail_lines,
        result.stderr,
    )
