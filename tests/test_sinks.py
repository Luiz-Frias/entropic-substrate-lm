"""Unit tests for :mod:`entropic_substrate.sinks` — the MetricSink port.

All tests here run against a FAKE MLflow client (``FakeMlflowClient``,
implementing just the slice of ``mlflow.tracking.MlflowClient`` the sink
uses) — no network, no real server. Live-stack behaviour (podman + Postgres +
real R2) is covered separately by the ``integration``-marked tests, gated
behind ``just container-test``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from mlflow.exceptions import MlflowException

from entropic_substrate.sinks import (
    MLFLOW_TINY_ENV,
    MLFLOW_TRACKING_ENV,
    CompositeSink,
    JsonlSink,
    MlflowSink,
    MlflowStartupError,
    build_sink,
    config_hash,
    git_sha,
    mlflow_enabled,
)
from helpers import tiny_config

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- fake client
@dataclass
class _FakeRunInfo:
    run_id: str


@dataclass
class _FakeRunData:
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeRun:
    info: _FakeRunInfo
    data: _FakeRunData = field(default_factory=_FakeRunData)


@dataclass
class _FakeExperiment:
    experiment_id: str


class FakeMlflowClient:
    """In-memory stand-in for ``mlflow.tracking.MlflowClient``.

    ``kill()``: flips the client into "every call from now on raises" —
    simulates "tracking killed mid-run" without racing the handful of calls
    ``MlflowSink.__init__`` itself makes during a successful startup.

    ``log_param`` enforces WRITE-ONCE, exactly like the real server: a
    param logged twice with the SAME value is a silent no-op, a param
    logged twice with a DIFFERENT value raises ``MlflowException`` — see
    ``sqlalchemy_store.py``'s ``log_param`` (the ``old_value != param.value``
    check after the unique-constraint ``IntegrityError``). A fake that
    modeled ``log_param`` as a plain dict overwrite (the pre-fix shape here)
    is MORE PERMISSIVE THAN REALITY: it cannot express the bug class where a
    call site treats a re-computable value as an immutable param (see
    ``entropic_substrate.mlflow_aggregate``'s ``n_seeds`` fix, and
    ``tests/test_mlflow_aggregate.py::test_growing_seed_group_updates_the_existing_parent``).
    """

    def __init__(self, *, fail_startup: bool = False) -> None:
        self._experiments: dict[str, str] = {}
        self._runs: dict[str, _FakeRun] = {}
        self._next_id = 0
        self.fail_startup = fail_startup
        self._killed = False
        self.artifacts_logged: list[tuple[str, str, str | None]] = []

    def kill(self) -> None:
        """Simulate the tracking server dying mid-run: every call now raises."""
        self._killed = True

    def _tick(self) -> None:
        if self._killed:
            msg = "simulated tracking server outage"
            raise ConnectionError(msg)

    def _mint(self, prefix: str) -> str:
        self._next_id += 1
        return f"{prefix}-{self._next_id}"

    def get_experiment_by_name(self, name: str) -> _FakeExperiment | None:
        if self.fail_startup:
            msg = "simulated unreachable tracking server"
            raise ConnectionError(msg)
        self._tick()
        if name not in self._experiments:
            return None
        return _FakeExperiment(experiment_id=self._experiments[name])

    def create_experiment(self, name: str) -> str:
        self._tick()
        exp_id = self._mint("exp")
        self._experiments[name] = exp_id
        return exp_id

    def create_run(self, experiment_id: str, tags: dict[str, str] | None = None) -> _FakeRun:
        self._tick()
        run_id = self._mint("run")
        run = _FakeRun(info=_FakeRunInfo(run_id=run_id), data=_FakeRunData(tags=dict(tags or {})))
        self._runs[run_id] = run
        return run

    def set_tag(self, run_id: str, key: str, value: Any) -> None:
        self._tick()
        self._runs[run_id].data.tags[str(key)] = str(value)

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self._tick()
        str_key, str_value = str(key), str(value)
        existing = self._runs[run_id].data.params.get(str_key)
        if existing is not None and existing != str_value:
            msg = (
                f"Changing param values is not allowed. Param with key='{str_key}' was "
                f"already logged with value='{existing}' for run ID='{run_id}'. Attempted "
                f"logging new value '{str_value}'."
            )
            raise MlflowException(msg)
        self._runs[run_id].data.params[str_key] = str_value

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self._tick()
        self._runs[run_id].data.metrics[key] = float(value)

    def log_artifact(self, run_id: str, path: str, artifact_path: str | None = None) -> None:
        self._tick()
        self.artifacts_logged.append((run_id, path, artifact_path))

    def set_terminated(self, run_id: str, status: str = "FINISHED") -> None:
        self._tick()
        self._runs[run_id].data.tags["mlflow.status"] = status

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str = "",
        max_results: int = 5000,
    ) -> list[_FakeRun]:
        self._tick()
        runs = list(self._runs.values())
        for key, value in _parse_tag_equals_filter(filter_string):
            runs = [r for r in runs if r.data.tags.get(key) == value]
        return runs[:max_results]


# MLflow's real filter_string grammar is a small SQL-like DSL; this fake only
# needs the ONE shape entropic_substrate.mlflow_aggregate actually emits:
# "tags.`key` = 'value'[ and tags.`key2` = 'value2']...". A test double that
# ignores filter_string entirely would silently return every run —
# search_runs(filter_string=...) exists precisely to narrow results, so a
# fake that skips it would let a real "find the existing parent" bug pass
# unnoticed in these tests.
_TAG_FILTER_RE = re.compile(r"tags\.`([^`]+)`\s*=\s*'([^']*)'")


def _parse_tag_equals_filter(filter_string: str) -> list[tuple[str, str]]:
    if not filter_string:
        return []
    return _TAG_FILTER_RE.findall(filter_string)


# ------------------------------------------------------------ FakeMlflowClient fidelity
def test_fake_log_param_is_a_silent_no_op_when_the_value_is_unchanged() -> None:
    """Real MLflow's write-once check compares OLD vs NEW value — re-logging
    the identical value must not raise (verified against the installed
    ``mlflow`` package's ``sqlalchemy_store.log_param``: the IntegrityError
    branch only re-raises when ``old_value != param.value``).
    """
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    run = client.create_run(exp_id)

    client.log_param(run.info.run_id, "n_seeds", 3)
    client.log_param(run.info.run_id, "n_seeds", 3)  # same value again — must not raise

    assert run.data.params["n_seeds"] == "3"


def test_fake_log_param_raises_mlflow_exception_when_the_value_changes() -> None:
    """The exact bug FIX 1 exists for: a param re-logged with a DIFFERENT
    value must raise, matching the real server, not silently overwrite.
    Reproduced against a real sqlite-backed ``mlflow.tracking.MlflowClient``
    in ``tests/test_mlflow_aggregate.py`` — this pins the fake to the same
    contract so unit tests can catch this bug class without a live server.
    """
    client = FakeMlflowClient()
    exp_id = client.create_experiment("exp")
    run = client.create_run(exp_id)

    client.log_param(run.info.run_id, "n_seeds", 3)
    with pytest.raises(MlflowException, match="Changing param values is not allowed"):
        client.log_param(run.info.run_id, "n_seeds", 4)


# ------------------------------------------------------------------- JsonlSink
def test_jsonl_sink_is_byte_identical_to_the_pre_port_serialisation(tmp_path: Path) -> None:
    """Golden-byte pin: old-style inline dump vs JsonlSink, same bytes."""
    record = {"run": "e3", "step": 5, "event": "eval", "loss": 1.5}

    legacy_path = tmp_path / "legacy.jsonl"
    with legacy_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    sink = JsonlSink(tmp_path / "ported.jsonl")
    sink.log(record)

    assert legacy_path.read_bytes() == sink.path.read_bytes()


def test_jsonl_sink_appends_multiple_records(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "out.jsonl")
    sink.log({"event": "run_start"})
    sink.log({"event": "eval", "loss": 1.0})
    lines = sink.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "run_start"
    assert json.loads(lines[1])["event"] == "eval"


def test_jsonl_sink_creates_parent_directories(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "nested" / "dir" / "out.jsonl")
    sink.log({"event": "run_start"})
    assert sink.path.exists()


def test_jsonl_sink_log_artifact_and_finish_are_no_ops(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "out.jsonl")
    sink.log_artifact(tmp_path / "ckpt.pt")  # must not raise
    sink.finish()  # must not raise


# ------------------------------------------------------------------- MlflowSink
def test_mlflow_sink_startup_probe_raises_typed_error_on_unreachable_server() -> None:
    client = FakeMlflowClient(fail_startup=True)
    with pytest.raises(MlflowStartupError):
        MlflowSink(
            tracking_uri="http://unreachable.invalid",
            experiment_name="exp",
            run_name="run",
            client=client,
        )


def test_mlflow_sink_construction_sets_params_and_tags() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="e1-fisher",
        params={"exp": "e1-fisher", "seed": 42, "config_hash": "abc123", "git_sha": "deadbeef"},
        tags={"exp": "e1-fisher", "config_hash": "abc123"},
        client=client,
    )
    run = client._runs[sink.run_id]
    assert run.data.params["seed"] == "42"
    assert run.data.params["config_hash"] == "abc123"
    assert run.data.tags["config_hash"] == "abc123"
    assert run.data.tags["mlflow.runName"] == "e1-fisher"


def test_mlflow_sink_logs_eval_scalar_metrics() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log(
        {
            "event": "eval",
            "step": 10,
            "loss": 1.23,
            "nats_per_byte": 0.5,
            "bits_per_byte": 0.72,
            "perplexity": 3.4,
        }
    )
    metrics = client._runs[sink.run_id].data.metrics
    assert metrics["loss"] == pytest.approx(1.23)
    assert metrics["nats_per_byte"] == pytest.approx(0.5)
    assert metrics["bits_per_byte"] == pytest.approx(0.72)
    assert metrics["perplexity"] == pytest.approx(3.4)
    assert sink.failed_writes == 0


def test_mlflow_sink_ignores_non_eval_events() -> None:
    """Scope: metrics land per eval, not per train step."""
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "train", "step": 1, "loss": 9.9})
    assert client._runs[sink.run_id].data.metrics == {}


def test_mlflow_sink_logs_108_shaped_per_head_kappa_plus_derived_series() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    n_layers, n_heads = 9, 12
    kappa = [[float(i * n_heads + j) / 10 for j in range(n_heads)] for i in range(n_layers)]
    sink.log({"event": "eval", "step": 5, "kappa": kappa})

    metrics = client._runs[sink.run_id].data.metrics
    head_keys = {k for k in metrics if k.startswith("kappa/L") and "/h" in k}
    assert len(head_keys) == n_layers * n_heads == 108
    for i in range(n_layers):
        for j in range(n_heads):
            assert metrics[f"kappa/L{i}/h{j}"] == pytest.approx(kappa[i][j])
        assert f"kappa/L{i}/spread" in metrics
    assert "kappa/spread_max" in metrics
    assert "kappa/mean_abs" in metrics
    assert metrics["kappa/spread_max"] == pytest.approx(max(kappa[-1]) - min(kappa[-1]))


def test_mlflow_sink_kappa_metrics_are_shape_agnostic_not_hardcoded_108() -> None:
    """Small (tiny-config) shapes must produce exactly that many keys, not 108."""
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    kappa = [[-1.0, 0.0, 1.0, 0.5], [2.0, -2.0, 0.0, 0.1]]  # 2 layers x 4 heads
    sink.log({"event": "eval", "step": 0, "kappa": kappa})
    metrics = client._runs[sink.run_id].data.metrics
    head_keys = {k for k in metrics if k.startswith("kappa/L") and "/h" in k}
    assert len(head_keys) == 8


def test_mlflow_sink_swallows_post_startup_failures_and_counts_them() -> None:
    """RUNTIME CONTINUE: a blip after startup never raises past the sink."""
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    client.kill()  # simulate tracking dying AFTER a successful startup
    sink.log({"event": "eval", "step": 1, "loss": 1.0})  # must not raise
    sink.log({"event": "eval", "step": 2, "loss": 1.0})  # must not raise
    sink.log_artifact(Path("/tmp/does-not-matter.pt"))  # must not raise
    assert sink.failed_writes == 3
    sink.finish()  # must not raise, even though set_terminated also fails
    assert sink.failed_writes == 4


def test_mlflow_sink_finish_is_idempotent() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.finish()
    sink.finish()  # second call must be a no-op, not a double-terminate error
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_mlflow_sink_reuses_existing_experiment() -> None:
    client = FakeMlflowClient()
    client.create_experiment("shared-exp")
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="shared-exp", run_name="run", client=client
    )
    assert len(client._experiments) == 1  # no duplicate experiment created
    assert sink.run_id in client._runs


def test_mlflow_sink_log_artifact_reads_sidecar_dvc_pointer(tmp_path: Path) -> None:
    ckpt = tmp_path / "e3-final.pt"
    ckpt.write_bytes(b"fake checkpoint bytes")
    (tmp_path / "e3-final.pt.dvc").write_text(
        "outs:\n- md5: abc123def456\n  size: 12\n  hash: md5\n  path: e3-final.pt\n  remote: r2-entropic-substrate-lm\n",
        encoding="utf-8",
    )
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log_artifact(ckpt, artifact_path="checkpoints")

    assert client.artifacts_logged == [(sink.run_id, str(ckpt), "checkpoints")]
    tags = client._runs[sink.run_id].data.tags
    assert tags["dvc/e3-final.pt_md5"] == "abc123def456"
    assert tags["dvc/e3-final.pt_remote"] == "r2-entropic-substrate-lm"


def test_mlflow_sink_log_artifact_without_dvc_pointer_skips_dvc_tags(tmp_path: Path) -> None:
    ckpt = tmp_path / "e3-final.pt"
    ckpt.write_bytes(b"x")
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log_artifact(ckpt)
    tags = client._runs[sink.run_id].data.tags
    assert not any(k.startswith("dvc/") for k in tags)


# --------------------------------------------------------------- CompositeSink
def test_composite_sink_fans_out_to_all_sinks(tmp_path: Path) -> None:
    jsonl = JsonlSink(tmp_path / "out.jsonl")
    client = FakeMlflowClient()
    mlflow_sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    composite = CompositeSink(jsonl, mlflow_sink)

    composite.log({"event": "eval", "step": 1, "loss": 1.0})
    composite.finish()

    assert json.loads(jsonl.path.read_text(encoding="utf-8").strip())["loss"] == 1.0
    assert client._runs[mlflow_sink.run_id].data.metrics["loss"] == pytest.approx(1.0)
    assert client._runs[mlflow_sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_composite_sink_authoritative_write_failure_propagates(tmp_path: Path) -> None:
    """JSONL is authoritative: a genuine write defect must NOT be swallowed."""

    class _BrokenSink:
        def log(self, record: Any) -> None:
            msg = "disk full"
            raise OSError(msg)

        def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
            return None

        def finish(self) -> None:
            return None

    composite = CompositeSink(_BrokenSink())
    with pytest.raises(OSError, match="disk full"):
        composite.log({"event": "eval"})


def test_composite_sink_authoritative_write_survives_even_if_best_effort_fails(
    tmp_path: Path,
) -> None:
    """A JSONL record must land even when the MLflow leg is down mid-run."""
    jsonl = JsonlSink(tmp_path / "out.jsonl")
    client = FakeMlflowClient()
    mlflow_sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    composite = CompositeSink(jsonl, mlflow_sink)
    client.kill()  # simulate tracking dying AFTER a successful startup

    for step in range(5):
        composite.log({"event": "eval", "step": step, "loss": float(step)})

    lines = jsonl.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5, "JSONL must be complete even though every MLflow write failed"
    assert mlflow_sink.failed_writes == 5


# ---------------------------------------------------------------- enablement
def test_mlflow_enabled_false_without_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MLFLOW_TRACKING_ENV, raising=False)
    assert mlflow_enabled(tiny=False) is False
    assert mlflow_enabled(tiny=True) is False


def test_mlflow_enabled_true_for_non_tiny_when_uri_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake")
    monkeypatch.delenv(MLFLOW_TINY_ENV, raising=False)
    assert mlflow_enabled(tiny=False) is True


def test_mlflow_enabled_false_for_tiny_without_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake")
    monkeypatch.delenv(MLFLOW_TINY_ENV, raising=False)
    assert mlflow_enabled(tiny=True) is False


def test_mlflow_enabled_true_for_tiny_with_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "http://fake")
    monkeypatch.setenv(MLFLOW_TINY_ENV, "1")
    assert mlflow_enabled(tiny=True) is True


def test_mlflow_enabled_blank_uri_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MLFLOW_TRACKING_ENV, "   ")
    assert mlflow_enabled(tiny=False) is False


# ------------------------------------------------------------------ build_sink
def test_build_sink_returns_bare_jsonl_sink_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(MLFLOW_TRACKING_ENV, raising=False)
    cfg = tiny_config(tmp_path)
    sink = build_sink(cfg, tiny=False, results_path=tmp_path / "out.jsonl")
    assert isinstance(sink, JsonlSink)


def test_build_sink_returns_composite_when_client_injected(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    client = FakeMlflowClient()
    sink = build_sink(cfg, tiny=False, results_path=tmp_path / "out.jsonl", client=client)
    assert isinstance(sink, CompositeSink)


def test_build_sink_startup_abort_propagates(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    client = FakeMlflowClient(fail_startup=True)
    with pytest.raises(MlflowStartupError):
        build_sink(cfg, tiny=False, results_path=tmp_path / "out.jsonl", client=client)


def test_build_sink_params_carry_exp_seed_hash_and_git_sha(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    cfg.name = "e1-fisher"
    cfg.train.seed = 7
    client = FakeMlflowClient()
    sink = build_sink(cfg, tiny=False, results_path=tmp_path / "out.jsonl", client=client)
    assert isinstance(sink, CompositeSink)
    mlflow_sink = sink.best_effort[0]
    run = client._runs[mlflow_sink.run_id]
    assert run.data.params["exp"] == "e1-fisher"
    assert run.data.params["seed"] == "7"
    assert run.data.params["config_hash"] == config_hash(cfg)
    assert run.data.params["git_sha"] == git_sha()
    assert run.data.tags["config_hash"] == config_hash(cfg)


# --------------------------------------------------------------- provenance
def test_config_hash_is_stable_across_identical_configs(tmp_path: Path) -> None:
    cfg_a = tiny_config(tmp_path)
    cfg_b = tiny_config(tmp_path)
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_is_seed_invariant_so_seed_sweeps_can_group(tmp_path: Path) -> None:
    """The seed-aggregation grouping key MUST NOT vary with the seed itself —
    a seed sweep is, by definition, N runs that differ ONLY by seed. Hashing
    the seed in would give every one of them a distinct hash: a group of 1
    that ``mlflow-aggregate --n-seeds 3`` could never complete (this is the
    exact bug the live-stack integration test caught: three "seed" runs each
    landed in their own singleton group and no parent was ever created).
    """
    cfg_a = tiny_config(tmp_path)
    cfg_b = tiny_config(tmp_path)
    cfg_b.train.seed = cfg_a.train.seed + 1
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_is_sensitive_to_non_seed_content(tmp_path: Path) -> None:
    cfg_a = tiny_config(tmp_path)
    cfg_b = tiny_config(tmp_path)
    cfg_b.model.d_model = cfg_a.model.d_model + 32
    assert config_hash(cfg_a) != config_hash(cfg_b)

    cfg_c = tiny_config(tmp_path)
    cfg_c.train.max_steps = cfg_a.train.max_steps + 1
    assert config_hash(cfg_a) != config_hash(cfg_c)


def test_config_hash_ignores_run_name_only_when_name_unchanged(tmp_path: Path) -> None:
    """Same knobs, different run name, MUST diverge (name IS part of Config)."""
    cfg_a = tiny_config(tmp_path)
    cfg_b = tiny_config(tmp_path)
    cfg_b.name = "different-name"
    assert config_hash(cfg_a) != config_hash(cfg_b)


def test_git_sha_returns_a_40_char_hex_string_in_this_checkout() -> None:
    sha = git_sha()
    assert sha != "unknown"
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_git_sha_falls_back_to_unknown_outside_a_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert git_sha() == "unknown"


# ------------------------------------------------------------- shape sanity
def test_kappa_metric_helper_handles_empty_rows_without_crashing() -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake", experiment_name="exp", run_name="run", client=client
    )
    sink.log({"event": "eval", "step": 0, "kappa": []})
    # No kappa keys, no crash — an empty kappa list is falsy and short-circuits.
    metrics = client._runs[sink.run_id].data.metrics
    assert not any(k.startswith("kappa/") for k in metrics)


# ------------------------------------------- cold server vs absent server
class ReadTimeoutError(Exception):
    """Stands in for urllib3's ReadTimeoutError.

    Named to match the real class EXACTLY: :func:`_is_cold_start` matches by
    type name to avoid importing urllib3, so a stub with a decorated name
    (``_ReadTimeoutError``) would not be recognised — as this test proved on
    its first run.
    """


class NewConnectionError(Exception):
    """Stands in for urllib3's connect-time failure: nothing is listening."""


def test_read_timeout_in_the_cause_chain_is_a_cold_start() -> None:
    """A wrapped read timeout means the server ANSWERED and was slow."""
    from entropic_substrate.sinks import _is_cold_start

    wrapped = RuntimeError("API request to /experiments/create failed")
    wrapped.__cause__ = ReadTimeoutError("Read timed out. (read timeout=10)")
    assert _is_cold_start(wrapped)


def test_connection_refused_is_not_a_cold_start() -> None:
    """Nothing listening is MISCONFIGURATION — it must abort, never retry.

    This is the startup-abort invariant: retrying a bad URI would burn the
    whole cold-start budget before failing, which is the opposite of failing
    fast before GPU hours are spent.
    """
    from entropic_substrate.sinks import _is_cold_start

    wrapped = RuntimeError("failed to connect")
    wrapped.__cause__ = NewConnectionError("[Errno 61] Connection refused")
    assert not _is_cold_start(wrapped)


def test_cold_start_detection_survives_a_broken_cause_chain() -> None:
    """Fallback path: a re-raise that dropped __cause__ still classifies."""
    from entropic_substrate.sinks import _is_cold_start

    assert _is_cold_start(RuntimeError("HTTPConnectionPool: Read timed out."))


def test_cold_start_walk_terminates_on_a_cyclic_cause_chain() -> None:
    """A self-referential chain must not hang the startup path."""
    from entropic_substrate.sinks import _is_cold_start

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert not _is_cold_start(a)


# ------------------------------------------------ results JSONL at finish()
# The sink owns the upload of the authoritative results JSONL as a run
# artifact (operator ruling 2026-08-17) — previously pod-side glue after
# training, which any new launch script could forget. The JSONL stays
# authoritative and local-first; the upload is derived, best-effort view.


def _results_backed_sink(tmp_path: Path, client: Any) -> tuple[MlflowSink, Path]:
    results = tmp_path / "results" / "run.jsonl"
    results.parent.mkdir(parents=True)
    results.write_text('{"step": 1}\n', encoding="utf-8")
    sink = MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="run",
        results_path=results,
        client=client,
    )
    return sink, results


def test_finish_uploads_results_jsonl_under_results_artifact_path(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    sink, results = _results_backed_sink(tmp_path, client)
    sink.finish()
    assert (sink.run_id, str(results), "results") in client.artifacts_logged
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_failed_results_upload_cannot_block_run_termination(tmp_path: Path) -> None:
    """The upload gets its OWN guard: a dead artifact store must not leave the
    run stuck in RUNNING — set_terminated still fires, the failure is counted.
    """

    class ArtifactStoreDown(FakeMlflowClient):
        def log_artifact(self, run_id: str, path: str, artifact_path: str | None = None) -> None:
            msg = "simulated artifact store outage"
            raise ConnectionError(msg)

    client = ArtifactStoreDown()
    sink, _ = _results_backed_sink(tmp_path, client)
    before = sink.failed_writes
    sink.finish()  # must not raise
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"
    assert sink.failed_writes == before + 1


def test_missing_results_file_at_finish_skips_upload_but_terminates(tmp_path: Path) -> None:
    client = FakeMlflowClient()
    sink = MlflowSink(
        tracking_uri="http://fake",
        experiment_name="exp",
        run_name="run",
        results_path=tmp_path / "never-written.jsonl",
        client=client,
    )
    sink.finish()
    assert client.artifacts_logged == []
    assert client._runs[sink.run_id].data.tags["mlflow.status"] == "FINISHED"


def test_build_sink_threads_results_path_into_the_mlflow_sink(tmp_path: Path) -> None:
    """The wiring, not just the unit: a composite built by build_sink must end
    a run with the results JSONL attached — the pod needs no glue script.
    """
    cfg = tiny_config(tmp_path)
    client = FakeMlflowClient()
    results = tmp_path / "out.jsonl"
    sink = build_sink(cfg, tiny=False, results_path=results, client=client)
    sink.log({"step": 1, "loss": 1.0})
    sink.finish()
    uploaded = [(path, dest) for (_run, path, dest) in client.artifacts_logged]
    assert (str(results), "results") in uploaded
