"""Generic tracking-run composition without changing the legacy sink contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from entropic_substrate.sinks import (
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

    assert extractor.seen == [event]
    assert event == before
    assert client._runs[sink.run_id].data.metrics == {"custom/measured/value": 1.2345678901234567}
