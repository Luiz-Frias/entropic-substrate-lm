"""Attach the H chart set to the MLflow runs that produced the data -- and fail LOUD.

The E-lane attach (:mod:`entropic_substrate.ladder.analyze.mlflow_attach`) is
deliberately fail-soft. This one is not: every (run, figure) pair is attempted, every
failure is collected, and one :class:`HAttachError` naming every failed run id is
raised afterwards, so a half-attached wave can never read as success. The client is
injected (built by the CLI through the adapter's ``build_mlflow_client``); this
module imports no MLflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["HAttachError", "attach_h_figures"]


class HAttachError(Exception):
    """At least one run could not be attached; ``failures`` maps run (or arm) to reason."""

    def __init__(self, failures: Mapping[str, str], attached: Mapping[str, list[str]]) -> None:
        self.failures = dict(failures)
        self.attached = {k: list(v) for k, v in attached.items()}
        super().__init__(
            "could not attach charts to: "
            + "; ".join(f"{run}: {reason}" for run, reason in sorted(self.failures.items()))
        )


def attach_h_figures(
    client: Any,
    targets: Mapping[str, str | None],
    figures: Mapping[str, Figure],
) -> dict[str, list[str]]:
    """``client.log_figure`` every figure to every target run under ``charts/``.

    ``targets`` maps an arm label to its MLflow run id (``None`` when the captured
    worker log named none, which is a failure of that arm, never a skip). Returns
    ``{run_id: [artifact paths]}``; raises :class:`HAttachError` after attempting all.
    """
    attached: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    for label, run_id in sorted(targets.items()):
        if run_id is None:
            failures[label] = "the captured worker log names no mlflow.run run_id"
            continue
        done: list[str] = []
        for name, figure in figures.items():
            artifact_file = f"charts/{name}.png"
            try:
                client.log_figure(run_id, figure, artifact_file)
            except Exception as exc:  # collected and re-raised below, never swallowed
                failures[run_id] = f"{artifact_file}: {exc!r}"
                break
            done.append(artifact_file)
        attached[run_id] = done
    if failures:
        raise HAttachError(failures, attached)
    return attached
