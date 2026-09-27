"""Metric sink port — where ``Trainer.log()``/``log_artifact()`` records go.

QUARANTINE FACADE (p3b, docs/plans/src-hexagonal-refactor.plan.md #p3b,
DESIGN-GATE RULING D1, conductor 2026-09-21, wf_bc9ada06-18e): this module
STAYS quarantined (``landed: false`` — a shim has ONE target while this
module's names are consumed from the old prefix by out-of-envelope
importers rewrite-on-arrival forbids this phase from touching, e.g.
``tests/test_mlflow_live.py:28`` imports ``build_sink`` and
``MLFLOW_TRACKING_ENV`` from ``entropic_substrate.sinks``). p3b EXTRACTED
the port, the three E-series adapters and the run-descriptor door into
their landed homes and rewrote this module's body into a thin re-export
facade over those landed addresses:

* :class:`~entropic_substrate.ladder.capture.ports.metric_sink.MetricSink`
  (was ``sinks.py:217``)
* :class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`
  (was ``sinks.py:233``)
* :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`
  (was ``sinks.py:341``) — the psutil/pynvml-avoidance guard
  (``tests/test_sinks.py``) now reads that module's own ``__file__``, not
  this one's.
* :class:`~entropic_substrate.ladder.capture.adapters.composite.CompositeSink`
  and :func:`~entropic_substrate.ladder.capture.adapters.composite.build_sink_for_run`
  (were ``sinks.py:640`` / ``sinks.py:737``)
* :class:`~entropic_substrate.ladder.capture.ports.run_descriptor.TrackingRunDescriptor`
  (was ``entropic_substrate.tracking_contracts.TrackingRunDescriptor``)

Every name the old prefix exported keeps working through this facade,
either as one of the re-export imports above or, for the RESIDUE below (no
landed home yet — p8 owns it), because it is still physically defined here:
:func:`build_sink`, :func:`config_hash`, :func:`git_sha`, ``CONFIG_HASH_TAG``,
``EXP_TAG``, the ``MLFLOW_*_ENV`` constants, :class:`MlflowStartupError`, and
the import-time ``_ENWIK8_DVC_POINTER`` side effect.

RETRACTION (house rule 4, architect constraint, design gate 2026-09-21): the
brief's Allowed-files entry for this module states "MEASURED for sinks: 15
names". That figure is WRONG. MEASURED directly against this module's own
``__all__`` at the p3-quarantine tip (53e9dc2): **18 names**, not 15 —
``MLFLOW_EXPERIMENT_ENV``, ``MLFLOW_SYSTEM_METRICS_ENV``, ``MLFLOW_TINY_ENV``,
``MLFLOW_TRACKING_ENV``, ``CompositeSink``, ``ExtractedMetrics``,
``JsonlSink``, ``LegacyMetricExtractor``, ``MetricExtractor``, ``MetricSink``,
``MlflowSink``, ``MlflowStartupError``, ``TrackingRunDescriptor``,
``build_sink``, ``build_sink_for_run``, ``config_hash``, ``git_sha``,
``mlflow_enabled``. This facade's own ``__all__`` below re-states the same 18
names, byte-for-byte, so the compat surface this fix proves is unchanged.

Derived kappa metrics (``kappa/spread_max``, per-layer spread, mean|kappa|)
are computed inside
:class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`, from
the eval record's ``kappa`` field — never added to the record itself passed
to :class:`~entropic_substrate.ladder.capture.adapters.jsonl.JsonlSink`, or
the bytes the gate parses would change.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404 -- git_sha() only, fixed argv, see its own nosec
from pathlib import Path
from typing import Any

from entropic_substrate.kernel.runtime.config import Config
from entropic_substrate.kernel.runtime.paths import RepoLayoutError, repo_root
from entropic_substrate.ladder.capture.adapters.composite import CompositeSink, build_sink_for_run
from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.adapters.mlflow import MlflowSink
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.capture.ports.run_descriptor import (
    ExtractedMetrics,
    LegacyMetricExtractor,
    MetricExtractor,
    TrackingRunDescriptor,
)

__all__ = [
    "MLFLOW_EXPERIMENT_ENV",
    "MLFLOW_SYSTEM_METRICS_ENV",
    "MLFLOW_TINY_ENV",
    "MLFLOW_TRACKING_ENV",
    "CompositeSink",
    "ExtractedMetrics",
    "JsonlSink",
    "LegacyMetricExtractor",
    "MetricExtractor",
    "MetricSink",
    "MlflowSink",
    "MlflowStartupError",
    "TrackingRunDescriptor",
    "build_sink",
    "build_sink_for_run",
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

# mlflow-NATIVE env var (mlflow.environment_variables.
# MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING) — passed through to mlflow's OWN
# SystemMetricsMonitor by entropic_substrate.ladder.capture.adapters.mlflow
# (never re-implements psutil/pynvml collection); the constant itself stays
# here as residue (p8) and is imported by that module at function scope.
# Deliberately re-parsed rather than read via mlflow's own accessor: mlflow's
# `.get()` raises ValueError on a garbage value; the best-effort spirit of
# the whole MlflowSink class says garbage -> disabled, not abort.
MLFLOW_SYSTEM_METRICS_ENV = "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"

# Fixed repo location of the enwik8 DVC pointer (docs/architecture.md — "enwik8
# is DVC-tracked in data/"). Not derived from cfg.data.path (that field names
# the extracted training file, e.g. "data/enwik8/enwik8"; the pointer that
# carries the digest lives one level up). A module constant rather than a cfg
# field: dataset lineage is scoped to exactly the one source this repo tracks
# in DVC today (see build_sink's source == "enwik8" gate) — FineWeb-Edu is
# streamed and carries no local pointer to read.


def _default_enwik8_dvc_pointer() -> Path:
    """Resolve the enwik8 DVC pointer's default location.

    Anchored the same way as ``entropic_substrate.config.configs_dir()``
    (layout-invariant repo-root resolution via
    ``entropic_substrate.paths.repo_root()`` -- E-3/DN-8, replacing the
    prior ``Path(__file__).resolve().parents[2]`` depth anchor) so the
    pointer resolves regardless of the caller's CWD, falling back to the
    plain CWD-relative candidate this constant carried before if the
    anchored one is not a file. Stays best-effort by construction either
    way: build_sink's later ``.exists()`` check treats a miss on both
    candidates as an absent pointer, not a startup failure. The resolver
    call is guarded so that a ``RepoLayoutError`` (never expected in a
    normal installed checkout) cannot turn this MODULE-IMPORT-TIME call
    (``_ENWIK8_DVC_POINTER`` below) into a raising path.
    """
    try:
        anchored = repo_root() / "data" / "enwik8.dvc"
    except RepoLayoutError:
        anchored = None
    if anchored is not None and anchored.is_file():
        return anchored
    return Path("data/enwik8.dvc")


_ENWIK8_DVC_POINTER = _default_enwik8_dvc_pointer()

# Tags the aggregate module (entropic_substrate.mlflow_aggregate) reads back.
CONFIG_HASH_TAG = "config_hash"
EXP_TAG = "exp"


class MlflowStartupError(RuntimeError):
    """MLflow tracking is configured but unreachable/misconfigured at startup.

    Raised ONLY during
    :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`
    construction (the startup probe) — never once a run is underway. Callers
    (``entropic_substrate.cli``) must let this abort the process before any
    training work happens: catching it anywhere inside the training loop
    would violate STARTUP ABORT, RUNTIME CONTINUE (see that adapter's module
    docstring).

    Stays here as residue (p8): the adapter imports this class at function
    scope (never module scope) to avoid a module-level import cycle with
    this facade, which imports the adapter to re-export ``MlflowSink``.
    """


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
    """Whether a training run should build an
    :class:`~entropic_substrate.ladder.capture.adapters.mlflow.MlflowSink`.

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
    """Legacy ``Config`` adapter over
    :func:`~entropic_substrate.ladder.capture.adapters.composite.build_sink_for_run`.
    """
    chash = config_hash(cfg)
    is_enwik8 = cfg.data.source == "enwik8"
    descriptor = TrackingRunDescriptor(
        run_name=cfg.name,
        tiny=tiny,
        params={
            EXP_TAG: cfg.name,
            "seed": cfg.train.seed,
            CONFIG_HASH_TAG: chash,
            "git_sha": git_sha(),
        },
        tags={EXP_TAG: cfg.name, CONFIG_HASH_TAG: chash},
        config_hash=chash,
        dataset_pointer_path=_ENWIK8_DVC_POINTER if is_enwik8 else None,
        dataset_split_spec=(
            {"source": cfg.data.source, "val_fraction": cfg.data.val_fraction}
            if is_enwik8
            else None
        ),
    )
    return build_sink_for_run(descriptor, results_path=results_path, client=client)
