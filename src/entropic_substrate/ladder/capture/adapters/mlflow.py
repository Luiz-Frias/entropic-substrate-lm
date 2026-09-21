"""``MlflowSink`` -- the BEST-EFFORT AFTER STARTUP :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink` adapter.

Landed at p3b from ``entropic_substrate.sinks:341`` (quarantined
``entropic_substrate.__legacy.sinks``). **The only module in the tree
allowed to import ``mlflow``** (``.importlinter``'s forbidden contract,
p3b door, ``frozen_after: p13``). Never authoritative, never gated on.

Construction performs a STARTUP PROBE against the tracking server: an
unreachable host or bad credentials raises :class:`MlflowStartupError`
(kept in ``entropic_substrate.__legacy.sinks`` -- residue, p8) before any
training work happens. Once construction succeeds, every later call is
wrapped in a broad ``except`` -- a mid-run blip is caught, counted, and
reported once at :meth:`finish`; it never raises past this class and never
slows or kills the run.

Derived kappa metrics (``kappa/spread_max``, per-layer spread, mean|kappa|)
are computed HERE from the eval record's ``kappa`` field -- never added to
the record itself passed to
:class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`, or the
bytes the gate parses would change.

**Cycle note (architect constraint, design gate 2026-09-21):** the D1
ruling keeps ``MlflowStartupError`` and the ``MLFLOW_*_ENV`` constants in
``entropic_substrate.__legacy.sinks`` while this class lands here, and the
facade (``__legacy/sinks.py``) imports THIS module at module level to
re-export :class:`MlflowSink` under the old prefix. A module-level import
in both directions would be a genuine Python import cycle, so this module
imports ``entropic_substrate.__legacy.sinks`` ONLY inside function bodies
(never at module scope), after the facade has finished initialising.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    LegacyMetricExtractor,
    MetricExtractor,
)

__all__ = ["MlflowSink"]

# How long a PRESENT-but-cold tracking server may take to answer its first write
# before we give up. Deliberately generous: it is only reachable when something
# already accepted the connection, so it trades startup latency on a cold server
# for never failing a REQUIRED live gate on a warm-up race. An absent server does
# not reach this budget at all — see the ABSENT vs COLD note at the call site.
_STARTUP_COLD_BUDGET_S = 90.0

# Transport-level read-timeout types, matched by NAME rather than imported.
# mlflow re-wraps these in MlflowException, and importing urllib3/requests here
# to isinstance-check them would couple this module to mlflow's own transitive
# dependencies for no gain.
_COLD_START_EXC_NAMES = frozenset({"ReadTimeout", "ReadTimeoutError", "Timeout"})


def _is_cold_start(exc: BaseException) -> bool:
    """Is ``exc`` a server-is-slow failure rather than a server-is-absent one?

    Walks the ``__cause__``/``__context__`` chain, because the transport error
    that actually distinguishes the two is wrapped by the time it surfaces. The
    message-text fallback exists only for the case where the chain was broken by
    a re-raise that dropped the cause.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _COLD_START_EXC_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return "read timed out" in str(exc).lower()


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


@runtime_checkable
class _SystemMetricsMonitorLike(Protocol):
    """Structural shape this module needs from mlflow's own
    ``SystemMetricsMonitor`` — just ``start()``/``finish()``, so tests can
    inject a fake without ever importing the real (thread-spawning) class.
    """

    def start(self) -> None: ...
    def finish(self) -> None: ...


def _system_metrics_logging_enabled() -> bool:
    """Env-gated pass-through — same env var mlflow itself reads, parsed
    best-effort (garbage -> disabled) rather than mlflow's own raise-on-garbage
    ``.get()``; see the ``MLFLOW_SYSTEM_METRICS_ENV`` constant's comment.

    ``MLFLOW_SYSTEM_METRICS_ENV`` stays in ``entropic_substrate.__legacy.sinks``
    (residue, p8) -- imported here at function scope, never at module scope,
    per this module's cycle note.
    """
    from entropic_substrate.__legacy.sinks import MLFLOW_SYSTEM_METRICS_ENV

    return os.environ.get(MLFLOW_SYSTEM_METRICS_ENV, "").strip().lower() in {"true", "1"}


def _default_system_metrics_monitor_factory(
    run_id: str, tracking_uri: str
) -> _SystemMetricsMonitorLike:
    """Lazy import: the ONLY place psutil/pynvml reach this module, and only
    when system-metrics logging is actually enabled — see the module-level
    "no unconditional import" invariant pinned by
    ``tests/test_sinks.py::test_sinks_module_has_no_unconditional_psutil_or_pynvml_import``.
    """
    from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

    return SystemMetricsMonitor(run_id, tracking_uri=tracking_uri)


def _read_dvc_pointer_digest(pointer_path: Path) -> str | None:
    """Validated read of a DVC pointer's ``outs[0].md5`` digest.

    ``None`` when ``pointer_path`` does not exist — the normal, expected case
    (most configs are not enwik8, and even an enwik8 config may run before
    ``dvc pull``). Raises ``ValueError`` when the pointer file EXISTS but is
    structurally invalid (unparsable YAML, no ``outs``, empty ``outs``, or a
    missing/blank ``md5``) — a corrupt pointer is a defect to surface loudly
    from this function, never to silently treat as absent. The call site
    (``MlflowSink.__init__``) is what decides best-effort: it catches this
    and counts it as a swallowed tracking-write failure, exactly like every
    other post-startup MLflow operation in this class.
    """
    if not pointer_path.exists():
        return None
    import yaml

    try:
        payload = yaml.safe_load(pointer_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"{pointer_path}: unparsable YAML: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{pointer_path}: empty or non-mapping DVC pointer"
        raise ValueError(msg)
    outs = payload.get("outs")
    if not isinstance(outs, list) or not outs:
        msg = f"{pointer_path}: no 'outs' entries"
        raise ValueError(msg)
    first = outs[0]
    if not isinstance(first, dict):
        msg = f"{pointer_path}: outs[0] is not a mapping"
        raise ValueError(msg)
    digest = first.get("md5")
    if not digest:
        msg = f"{pointer_path}: outs[0] missing a non-blank 'md5'"
        raise ValueError(msg)
    return str(digest)


def _is_already_exists(exc: BaseException) -> bool:
    """Idempotent registry creation: a registered model that already exists is
    SUCCESS, not failure — every seed run of one experiment shares the same
    registered-model name (``run_name``) and will all attempt to (re-)create
    it. Matched by ``error_code`` (the string mlflow's own ``MlflowException``
    exposes, e.g. ``"RESOURCE_ALREADY_EXISTS"``) rather than an isinstance
    check against ``mlflow.exceptions.MlflowException`` — this keeps the
    check working against any injected fake client's own exception type,
    the same "match by shape, not by mlflow's own class" idiom
    ``_is_cold_start`` already uses for transport errors above.
    """
    return getattr(exc, "error_code", None) == "RESOURCE_ALREADY_EXISTS"


class MlflowSink:
    """BEST-EFFORT AFTER STARTUP. See module docstring for the full contract.

    ``client`` is injectable so callers (and tests) can pass a stub/fake that
    implements the small slice of ``mlflow.tracking.MlflowClient`` this class
    uses (``get_experiment_by_name``, ``create_experiment``, ``create_run``,
    ``set_tag``, ``log_param``, ``log_metric``, ``log_artifact``,
    ``set_terminated``, plus the registry trio ``create_registered_model`` /
    ``create_model_version`` / ``set_registered_model_alias`` for
    :meth:`register_citation`) without a live server. When ``client`` is
    omitted, ``mlflow`` is imported lazily here — the only place in this
    module that touches the dependency — so untracked runs and unit tests
    that never build an ``MlflowSink`` pay no import cost.

    ``system_metrics_monitor_factory`` is the same injectable-seam idiom for
    mlflow's OWN ``SystemMetricsMonitor``: production leaves it at the
    default (a lazy import, gated on ``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING``),
    tests inject a fake so the unit suite never spawns the real class's
    background thread against a real ``MlflowClient``.

    This class stays deliberately Config-free: dataset lineage
    (``dataset_pointer_path`` / ``dataset_split_spec``) and the citation
    registry key (``config_hash``) are plain values ``build_sink`` — which
    DOES have the ``Config`` — extracts and passes through.
    """

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        params: Mapping[str, Any] | None = None,
        tags: Mapping[str, Any] | None = None,
        results_path: str | Path | None = None,
        config_hash: str | None = None,
        dataset_pointer_path: str | Path | None = None,
        dataset_split_spec: Mapping[str, Any] | None = None,
        client: Any | None = None,
        metric_extractor: MetricExtractor | None = None,
        system_metrics_monitor_factory: Callable[[str, str], _SystemMetricsMonitorLike]
        | None = None,
    ) -> None:
        # MlflowStartupError stays in entropic_substrate.__legacy.sinks
        # (residue, p8) -- imported here at function scope, never at module
        # scope, per this module's cycle note.
        from entropic_substrate.__legacy.sinks import MlflowStartupError

        self._failed_writes = 0
        self._finished = False
        self._metric_extractor = metric_extractor or LegacyMetricExtractor()
        # The authoritative results JSONL this run writes locally; finish()
        # uploads it as a derived, best-effort run artifact (operator ruling
        # 2026-08-17 — previously pod-side glue any launch script could
        # forget). None => nothing to upload, e.g. tests that only probe
        # metric fan-out.
        self._results_path = Path(results_path) if results_path is not None else None
        # Citation-registry key (config_hash()'s output) — None => the caller
        # never gave one, and register_citation() becomes a no-op rather than
        # guessing a value that would silently collide across configs.
        self._config_hash = config_hash
        # "Registered model per experiment" == this run's run_name, the ONLY
        # value build_sink ever passes here (== cfg.name at that call site).
        # Named for what it IS USED AS (the registry's registered-model name),
        # not for the constructor arg it happens to equal — CodeRabbit review
        # flagged the prior name (`_exp_name`) as confusable with the separate
        # `experiment_name` constructor argument, which is a different value
        # (the MLflow tracking experiment) entirely.
        self._registered_model_name = run_name

        if client is None:
            _apply_fast_fail_http_defaults()
            try:
                import mlflow.tracking
            except ImportError as exc:  # pragma: no cover — mlflow is a hard dep
                msg = f"mlflow is not importable: {exc}"
                raise MlflowStartupError(msg) from exc
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        self._client = client

        # ABSENT vs COLD are different failures and must not share a policy.
        # "Connection refused" / DNS failure means nothing is listening: that is
        # misconfiguration, and the startup-abort invariant requires failing
        # immediately, before GPU hours are spent. A READ timeout means something
        # accepted the connection and then took too long — a cold gunicorn worker
        # or a first-request DB migration — which is a slow server, not a wrong
        # URI, and retrying is correct. Treating them alike is what made the live
        # gate flaky: the budget that fails fast on a bad URI is far too short for
        # a cold server on a small VM.
        deadline = time.monotonic() + _STARTUP_COLD_BUDGET_S
        attempt = 0
        while True:
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
                break
            except MlflowStartupError:
                raise
            except Exception as exc:
                if not _is_cold_start(exc) or time.monotonic() >= deadline:
                    msg = f"MLflow tracking unreachable/misconfigured at {tracking_uri!r}: {exc}"
                    raise MlflowStartupError(msg) from exc
                attempt += 1
                backoff = min(2.0**attempt, 8.0)
                logger.warning(
                    f"mlflow: {tracking_uri} answered but read-timed-out "
                    f"(attempt {attempt}); cold server, retrying in {backoff:.0f}s"
                )
                time.sleep(backoff)

        # DATASET LINEAGE (best-effort, own guard) — deliberately OUTSIDE the
        # retry loop above: a corrupt/malformed pointer is neither a cold
        # start (must not be retried against the STARTUP_COLD_BUDGET) nor a
        # reason to raise MlflowStartupError (only network unreachability
        # gets that authority). dataset_pointer_path is None whenever
        # build_sink's config isn't sourced from enwik8; the reader itself
        # returns None (no-op) when the pointer file doesn't exist.
        if dataset_pointer_path is not None:
            try:
                digest = _read_dvc_pointer_digest(Path(dataset_pointer_path))
                if digest is not None:
                    client.set_tag(self._run_id, "dvc/enwik8_md5", digest)
                    for key, value in (dataset_split_spec or {}).items():
                        client.log_param(self._run_id, f"data/{key}", value)
            except Exception as exc:
                self._failed_writes += 1
                logger.warning(f"mlflow: dataset lineage logging failed: {exc}")

        # SYSTEM METRICS (best-effort, own guard) — env-gated pass-through to
        # mlflow's OWN SystemMetricsMonitor; see MLFLOW_SYSTEM_METRICS_ENV's
        # comment for why this class re-parses the env var itself rather than
        # calling mlflow's accessor.
        self._system_metrics_monitor: _SystemMetricsMonitorLike | None = None
        if _system_metrics_logging_enabled():
            factory = system_metrics_monitor_factory or _default_system_metrics_monitor_factory
            try:
                monitor = factory(self._run_id, tracking_uri)
                monitor.start()
                self._system_metrics_monitor = monitor
            except Exception as exc:
                self._failed_writes += 1
                logger.warning(f"mlflow: system metrics monitor failed to start: {exc}")

        logger.info(f"mlflow: tracking run {run_name!r} at {tracking_uri} (run_id={self._run_id})")

    def log(self, record: object) -> None:
        try:
            extracted = self._metric_extractor.extract(record)
            if extracted is None:
                return
            for key, value in extracted.metrics.items():
                self._client.log_metric(self._run_id, key, value, step=extracted.step)
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
        # OWN GUARD: a monitor that fails to stop must not block run
        # termination below. Nested inside the `_finished` early-return
        # above, so a second finish() call never double-stops it.
        if self._system_metrics_monitor is not None:
            try:
                self._system_metrics_monitor.finish()
            except Exception as exc:
                self._failed_writes += 1
                logger.warning(f"mlflow: system metrics monitor finish() failed: {exc}")
        # OWN GUARD, deliberately separate from the termination block below: a
        # dead artifact store must not leave the run stuck in RUNNING —
        # set_terminated still fires and the failure is counted (pinned by
        # tests/test_sinks.py). CompositeSink.finish() runs the authoritative
        # JsonlSink first, so the file is complete before this upload reads it.
        if self._results_path is not None:
            try:
                if self._results_path.exists():
                    self._client.log_artifact(
                        self._run_id, str(self._results_path), artifact_path="results"
                    )
                else:
                    logger.warning(
                        f"mlflow: results file {self._results_path} missing at finish() — "
                        "skipping the derived upload; the local JSONL is authoritative"
                    )
            except Exception as exc:
                self._failed_writes += 1
                logger.warning(f"mlflow: results JSONL upload failed: {exc}")
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

    def register_citation(self) -> None:
        """Best-effort, EXPLICIT (never auto-called from :meth:`finish`):
        catalog this run in the MLflow model registry as a CITATION POINTER
        ONLY — the DVC-tracked ``.pt`` checkpoint remains the sole load
        authority, and this method never uploads or references model
        weights.

        One registered model per experiment (``self._registered_model_name`` ==
        ``run_name`` == ``cfg.name`` at the only call site, ``build_sink``),
        aliased ``champion-<config_hash>`` so re-running the same config
        repoints the alias to this run's citation rather than accumulating
        duplicates (``config_hash`` is deliberately seed-invariant — see
        :func:`entropic_substrate.__legacy.sinks.config_hash`). A no-op, not
        an error, when this sink was built without a ``config_hash`` (e.g. a
        bare :class:`MlflowSink` in a test that never threads one through).

        The registered model version's ``source`` points at this run's
        ``results`` artifact (telemetry JSONL, uploaded by :meth:`finish`) —
        verified 2026-08-17 against a real, local sqlite-backed
        ``mlflow.tracking.MlflowClient`` that ``create_model_version`` both
        accepts an arbitrary ``runs:/<id>/<artifact>`` source at creation time
        (no MLmodel flavour required to REGISTER, only to LOAD, which this
        method never does) and that the alias resolves back to it afterward.
        """
        if self._config_hash is None:
            logger.warning("mlflow: register_citation() called without a config_hash — skipping")
            return
        try:
            try:
                self._client.create_registered_model(self._registered_model_name)
            except Exception as exc:
                if not _is_already_exists(exc):
                    raise
            source = f"runs:/{self._run_id}/results"
            version = self._client.create_model_version(
                name=self._registered_model_name,
                source=source,
                run_id=self._run_id,
                tags={"citation_only": "true"},
            )
            self._client.set_registered_model_alias(
                self._registered_model_name, f"champion-{self._config_hash}", version.version
            )
        except Exception as exc:
            self._failed_writes += 1
            logger.warning(f"mlflow: registry citation failed: {exc}")
