"""Metric sink port — where ``Trainer.log()``/``log_artifact()`` records go.

Two adapters behind :class:`MetricSink`:

* :class:`JsonlSink` — **AUTHORITATIVE**. Owns the exact byte-for-byte
  serialisation of ``results/*.jsonl`` that :mod:`entropic_substrate.telemetry`
  (the E3 execution gate) parses. This is a straight extraction of what
  ``Trainer.log()`` did inline before this module existed — same dict, same
  ``json.dumps(record, sort_keys=True) + "\\n"``, same append-only file — so a
  run through this sink is byte-identical to one that never went through a
  port. A write failure here (disk full, permissions) is a defect, not a
  degraded-mode event: it propagates, never gets softened.
* :class:`MlflowSink` — **BEST-EFFORT AFTER STARTUP**. Never authoritative,
  never gated on. Construction performs a STARTUP PROBE against the tracking
  server: an unreachable host or bad credentials raises
  :class:`MlflowStartupError` before any training work happens. Once
  construction succeeds, every later call is wrapped in a broad ``except`` —
  a mid-run blip is caught, counted, and reported once at ``finish()``; it
  never raises past this class and never slows or kills the run.

:class:`CompositeSink` fans one ``Trainer.log()``/``log_artifact()`` call out
to N sinks: the first is authoritative (its exceptions propagate), the rest
are already self-guarding per the port's contract above.

Derived kappa metrics (``kappa/spread_max``, per-layer spread, mean|kappa|)
are computed HERE, inside :class:`MlflowSink`, from the eval record's
``kappa`` field — never added to the record itself passed to ``JsonlSink``,
or the bytes the gate parses would change.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404 -- git_sha() only, fixed argv, see its own nosec
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from entropic_substrate.config import Config
from entropic_substrate.logging import logger

__all__ = [
    "MLFLOW_EXPERIMENT_ENV",
    "MLFLOW_TINY_ENV",
    "MLFLOW_TRACKING_ENV",
    "CompositeSink",
    "JsonlSink",
    "MetricSink",
    "MlflowSink",
    "MlflowStartupError",
    "build_sink",
    "config_hash",
    "git_sha",
    "mlflow_enabled",
]

# Enablement: MLflow is on iff MLFLOW_TRACKING_URI is set. --tiny additionally
# requires the explicit opt-in below — CPU smoke runs should not, by default,
# create tracking-server noise every time someone runs `just test-all`.
MLFLOW_TRACKING_ENV = "MLFLOW_TRACKING_URI"
MLFLOW_TINY_ENV = "ENTROPIC_MLFLOW_TINY"
MLFLOW_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
_DEFAULT_EXPERIMENT = "entropic-substrate-lm"

# Tags the aggregate module (entropic_substrate.mlflow_aggregate) reads back.
CONFIG_HASH_TAG = "config_hash"
EXP_TAG = "exp"

# mlflow's own defaults (7 retries with exponential backoff, 120s timeout) make
# an unreachable-host STARTUP ABORT take minutes, not milliseconds — GPU-hours
# reasoning depends on this being fast. setdefault() only: an operator who has
# already set these gets to keep their own values (e.g. a flaky WAN link that
# genuinely needs more retries).
#
# timeout=5/retries=0 (the original values here) gave a real unreachable host
# ~5s x N to abort, but ZERO margin on a healthy server that stalls once.
# MEASURED ROOT CAUSE (not "port-forward jitter" — that theory was tested and
# falsified below): the fresh `just container-test` mlflow container's own
# HEALTHCHECK hits a lightweight liveness-only `/health` endpoint that never
# touches Postgres, so "container healthy" fires before the DB-backed REST
# API (SQLAlchemy engine/connection-pool init, first-query-shape compile) has
# actually warmed up. Isolated curl probes through the SAME port forward,
# immediately after healthy, measured this directly: call 1 (a real
# get-by-name) 7.27s, call 2 1.22s, call 3 0.11s — a one-time cold-start
# tax, not sustained network jitter. The PRIMARY fix for that is a real
# warm-up probe against the actual REST API in
# `scripts/run_container_test.sh`, run once before any test (and therefore
# before this class's tight budget is ever exercised against a cold server).
# This module's timeout=10/retries=1 is deliberately secondary, residual
# hardening: `build_sink()`'s startup probe still makes ~7 sequential calls
# per `MlflowSink` construction, so even with the harness warmed, 0-margin
# on a single call would still turn any leftover stall into a hard red on a
# REQUIRED gate. Do not read this as "the timeout bump fixes flakiness" —
# empirically it does NOT, on its own, cover a genuine cold start: a 10s
# timeout with 1 retry still read-timed-out twice in a row (20s+) against an
# unwarmed server in one verbatim re-run. backoff_factor is pinned low (not
# left to mlflow's own default of 2, which compounds across retries) so a
# genuinely dead host still aborts in a bounded window, not minutes; the
# common misconfiguration modes (connection refused, DNS failure — see
# acceptance #5's `http://127.0.0.1:1` test) fail instantly regardless of
# this budget, since neither is a read-timeout in the first place.
_HTTP_TIMEOUT_ENV = "MLFLOW_HTTP_REQUEST_TIMEOUT"
_HTTP_RETRIES_ENV = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
_HTTP_BACKOFF_FACTOR_ENV = "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"


def _apply_fast_fail_http_defaults() -> None:
    os.environ.setdefault(_HTTP_TIMEOUT_ENV, "10")
    os.environ.setdefault(_HTTP_RETRIES_ENV, "1")
    os.environ.setdefault(_HTTP_BACKOFF_FACTOR_ENV, "1")


class MlflowStartupError(RuntimeError):
    """MLflow tracking is configured but unreachable/misconfigured at startup.

    Raised ONLY during :class:`MlflowSink` construction (the startup probe) —
    never once a run is underway. Callers (``entropic_substrate.cli``) must
    let this abort the process before any training work happens: catching it
    anywhere inside the training loop would violate STARTUP ABORT, RUNTIME
    CONTINUE (see this module's docstring).
    """


@runtime_checkable
class MetricSink(Protocol):
    """Port for one run's metric stream. See module docstring for the contract."""

    def log(self, record: Mapping[str, Any]) -> None:
        """Record one JSONL-shaped event (``run_start`` / ``train`` / ``eval`` / ``run_end``)."""
        ...

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        """Attach a file (a checkpoint) to this run."""
        ...

    def finish(self) -> None:
        """Close out the run. Idempotent — safe to call more than once."""
        ...


class JsonlSink:
    """AUTHORITATIVE. Byte-identical to ``Trainer.log()``'s pre-port serialisation."""

    __slots__ = ("path",)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        # DVC owns checkpoint artifacts (`dvc add`, downstream of training);
        # the JSONL sink carries metrics only.
        return None

    def finish(self) -> None:
        return None


def _kappa_metrics(kappa: Sequence[Sequence[float]]) -> dict[str, float]:
    """``kappa/L{i}/h{j}`` for every value, plus the derived spread/mean series.

    Shape-agnostic: the E3 per-head readout is ``n_layers`` rows of
    ``n_heads`` values at full scale (9x12 = 108), but this makes no
    assumption about either dimension — it reports whatever shape the
    eval record actually carries.
    """
    metrics: dict[str, float] = {}
    all_values: list[float] = []
    spreads: list[float] = []
    for i, row in enumerate(kappa):
        values = [float(v) for v in row]
        for j, value in enumerate(values):
            metrics[f"kappa/L{i}/h{j}"] = value
        if values:
            spread = max(values) - min(values)
            spreads.append(spread)
            metrics[f"kappa/L{i}/spread"] = spread
        all_values.extend(values)
    if spreads:
        metrics["kappa/spread_max"] = max(spreads)
    if all_values:
        metrics["kappa/mean_abs"] = sum(abs(v) for v in all_values) / len(all_values)
    return metrics


class MlflowSink:
    """BEST-EFFORT AFTER STARTUP. See module docstring for the full contract.

    ``client`` is injectable so callers (and tests) can pass a stub/fake that
    implements the small slice of ``mlflow.tracking.MlflowClient`` this class
    uses (``get_experiment_by_name``, ``create_experiment``, ``create_run``,
    ``set_tag``, ``log_param``, ``log_metric``, ``log_artifact``,
    ``set_terminated``) without a live server. When ``client`` is omitted,
    ``mlflow`` is imported lazily here — the only place in this module that
    touches the dependency — so untracked runs and unit tests that never
    build an ``MlflowSink`` pay no import cost.
    """

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self._failed_writes = 0
        self._finished = False

        if client is None:
            _apply_fast_fail_http_defaults()
            try:
                import mlflow.tracking
            except ImportError as exc:  # pragma: no cover — mlflow is a hard dep
                msg = f"mlflow is not importable: {exc}"
                raise MlflowStartupError(msg) from exc
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        self._client = client

        try:
            experiment = client.get_experiment_by_name(experiment_name)
            experiment_id = (
                experiment.experiment_id
                if experiment is not None
                else client.create_experiment(experiment_name)
            )
            run = client.create_run(
                experiment_id, tags={str(k): str(v) for k, v in (tags or {}).items()}
            )
            self._run_id = run.info.run_id
            client.set_tag(self._run_id, "mlflow.runName", run_name)
            for key, value in (params or {}).items():
                client.log_param(self._run_id, key, value)
        except MlflowStartupError:
            raise
        except Exception as exc:
            msg = f"MLflow tracking unreachable/misconfigured at {tracking_uri!r}: {exc}"
            raise MlflowStartupError(msg) from exc

        logger.info(f"mlflow: tracking run {run_name!r} at {tracking_uri} (run_id={self._run_id})")

    def log(self, record: Mapping[str, Any]) -> None:
        if record.get("event") != "eval":
            # Scope: metrics land per eval, not per train step — every-20-step
            # REST calls would be both noisy and unnecessary (JSONL already
            # carries the dense train-step trace, authoritatively).
            return
        try:
            step = int(record.get("step", 0))
            metrics: dict[str, float] = {}
            for key in ("loss", "nats_per_byte", "bits_per_byte", "perplexity"):
                if key in record:
                    metrics[key] = float(record[key])
            kappa = record.get("kappa")
            if kappa:
                metrics.update(_kappa_metrics(kappa))
            for key, value in metrics.items():
                self._client.log_metric(self._run_id, key, value, step=step)
        except Exception as exc:
            self._failed_writes += 1
            logger.warning(f"mlflow: metric write failed (#{self._failed_writes}): {exc}")

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        try:
            self._client.log_artifact(self._run_id, str(path), artifact_path=artifact_path)
            self._log_dvc_pointer_if_present(path)
        except Exception as exc:
            self._failed_writes += 1
            logger.warning(
                f"mlflow: artifact upload failed for {path} (#{self._failed_writes}): {exc}"
            )

    def _log_dvc_pointer_if_present(self, path: Path) -> None:
        """Best-effort: if ``<path>.dvc`` already exists, log its md5 + remote.

        ``dvc add`` for checkpoints runs downstream of training (see
        ``scripts/run_ladder.sh``, out of scope here), so this is normally a
        no-op during a live run — it only fires when re-logging an artifact
        for a checkpoint that has already been ``dvc add``-ed.
        """
        pointer = path.with_name(path.name + ".dvc")
        if not pointer.exists():
            return
        import yaml

        payload = yaml.safe_load(pointer.read_text(encoding="utf-8")) or {}
        outs = payload.get("outs") or []
        if not outs:
            return
        out = outs[0]
        md5 = out.get("md5")
        remote = out.get("remote")
        if md5:
            self._client.set_tag(self._run_id, f"dvc/{path.name}_md5", md5)
        if remote:
            self._client.set_tag(self._run_id, f"dvc/{path.name}_remote", remote)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self._failed_writes:
                self._client.set_tag(self._run_id, "mlflow_failed_writes", str(self._failed_writes))
                logger.warning(
                    f"mlflow: run {self._run_id} completed with {self._failed_writes} failed "
                    "tracking write(s) — results/*.jsonl is authoritative and complete "
                    "regardless; MLflow is the derived, best-effort view."
                )
            self._client.set_terminated(self._run_id, status="FINISHED")
        except Exception as exc:
            self._failed_writes += 1
            logger.warning(f"mlflow: finish() failed: {exc}")

    @property
    def failed_writes(self) -> int:
        """Count of tracking writes swallowed after startup — for tests/diagnostics."""
        return self._failed_writes

    @property
    def run_id(self) -> str:
        return self._run_id


class CompositeSink:
    """Fans one ``Trainer.log()``/``log_artifact()`` call out to N sinks.

    ``authoritative`` is never guarded here — a genuine defect in it must
    propagate. Every sink in ``best_effort`` is expected to already
    self-guard per :class:`MetricSink`'s contract (:class:`MlflowSink` does);
    this class does not add a second guard on top, so a sink that is NOT
    actually best-effort still surfaces its own failures honestly.
    """

    def __init__(self, authoritative: MetricSink, *best_effort: MetricSink) -> None:
        self._authoritative = authoritative
        self._best_effort = best_effort

    def log(self, record: Mapping[str, Any]) -> None:
        self._authoritative.log(record)
        for sink in self._best_effort:
            sink.log(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        self._authoritative.log_artifact(path, artifact_path=artifact_path)
        for sink in self._best_effort:
            sink.log_artifact(path, artifact_path=artifact_path)

    def finish(self) -> None:
        self._authoritative.finish()
        for sink in self._best_effort:
            sink.finish()

    @property
    def best_effort(self) -> tuple[MetricSink, ...]:
        """The non-authoritative sinks — for tests/diagnostics."""
        return self._best_effort


def git_sha() -> str:
    """The current commit SHA, or ``"unknown"`` outside a git checkout.

    ``GIT_*`` env vars are stripped before the subprocess call — mirroring
    ``scripts/run_exit_gate.py``'s ``_SAFE_ENV`` (the documented 2026-07-13
    ``core.worktree`` poisoning incident class). A pre-commit/pre-push hook
    context sets ``GIT_DIR``/``GIT_WORK_TREE`` in the parent process's
    environment; left in place, ``git rev-parse HEAD`` resolves against
    THAT repo regardless of this process's actual cwd, which is exactly how
    a directory outside any git checkout can still report a real SHA.
    Computed fresh per call (not cached at import time, unlike
    ``run_exit_gate.py``'s module-level constant) since this can run inside
    a long-lived process across a ``chdir``.
    """
    safe_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        # nosec B603 B607 -- "git" is deliberately resolved from PATH; args are
        # a fixed argv list (never shell-interpolated), matching the
        # scripts/run_exit_gate.py convention for the same "git" invocation.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            env=safe_env,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def config_hash(cfg: Config) -> str:
    """Short, stable hash of the resolved config — the seed-aggregation grouping key.

    ``train.seed`` is deliberately EXCLUDED: it is the one axis a seed sweep
    varies between otherwise-identical runs, and mlflow-aggregate groups runs
    BY this hash to find seeds of the same experiment. Hashing the seed in
    would give every seed run of the same config a distinct hash — a group
    of 1 that ``mlflow-aggregate --n-seeds 3`` could never complete.
    """
    payload = cfg.to_dict()
    payload["train"] = {k: v for k, v in payload["train"].items() if k != "seed"}
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def mlflow_enabled(*, tiny: bool) -> bool:
    """Whether a training run should build an :class:`MlflowSink`.

    ``MLFLOW_TRACKING_URI`` unset -> never. ``--tiny`` additionally requires
    the explicit opt-in env var (``ENTROPIC_MLFLOW_TINY=1``) — CPU smoke runs
    stay untracked unless someone deliberately asks otherwise.
    """
    uri = os.environ.get(MLFLOW_TRACKING_ENV, "").strip()
    if not uri:
        return False
    if tiny and os.environ.get(MLFLOW_TINY_ENV) != "1":
        return False
    return True


def build_sink(
    cfg: Config,
    *,
    tiny: bool,
    results_path: str | Path,
    client: Any | None = None,
) -> MetricSink:
    """Build the sink ``Trainer`` writes through.

    Raises :class:`MlflowStartupError` when MLflow is enabled but the
    tracking server is unreachable/misconfigured — callers must let this
    abort BEFORE constructing a ``Trainer`` (see the CLI's ``cmd_train``),
    so a bad ``MLFLOW_TRACKING_URI`` never burns GPU hours.
    """
    jsonl = JsonlSink(results_path)
    if not mlflow_enabled(tiny=tiny) and client is None:
        return jsonl

    tracking_uri = os.environ.get(MLFLOW_TRACKING_ENV, "")
    experiment_name = os.environ.get(MLFLOW_EXPERIMENT_ENV, _DEFAULT_EXPERIMENT)
    chash = config_hash(cfg)
    mlflow_sink = MlflowSink(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=cfg.name,
        params={
            EXP_TAG: cfg.name,
            "seed": cfg.train.seed,
            CONFIG_HASH_TAG: chash,
            "git_sha": git_sha(),
        },
        tags={EXP_TAG: cfg.name, CONFIG_HASH_TAG: chash},
        client=client,
    )
    return CompositeSink(jsonl, mlflow_sink)
