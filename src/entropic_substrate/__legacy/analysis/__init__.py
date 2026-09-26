"""Interpretation charts over ``results/*.jsonl`` telemetry.

Two lanes share this package's ``series``/``palette``/``charts`` modules:

* **Lane A** (``entropic_substrate.cli`` ``charts`` subcommand, lazy-imported
  like ``cmd_capture_results``) renders matplotlib figures and attaches them
  retroactively to MLflow runs via :meth:`mlflow.tracking.MlflowClient.
  log_figure` (``mlflow_attach``).
* **Lane B** (``notebooks/*.py``, marimo) renders the same chart set
  interactively in plotly, pulled from DVC-captured JSONLs.

Every JSONL read in this package goes through :mod:`entropic_substrate.
telemetry` — never ad-hoc ``json.loads`` on a results file — so an
inadmissible run fails loud (``TelemetryGap``) in both lanes identically.
Charts RENDER predicates (thresholds, CIs) visually; they never decide — the
gates this repo runs on stay sanity+selection, judgment human.
"""

from __future__ import annotations

__all__: list[str] = []
