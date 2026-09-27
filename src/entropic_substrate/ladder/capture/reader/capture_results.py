"""Retrieval-side capture of run results JSONLs into DVC.

Pods run credential-free (operator ruling 2026-08-17): a training pod holds
no R2/DVC identity and uploads its authoritative ``results/*.jsonl`` to the
tracking server as a run artifact (``MlflowSink.finish()``). This module is
the other half of that contract — it runs on an operator machine that DOES
hold R2 credentials, pulls those artifacts down, validates each one through
:func:`entropic_substrate.telemetry.final_eval_record` (the same admissibility
predicate every other JSONL consumer uses), and hands the capture directory to
DVC.

Measurement path: it fails LOUD. An unknown experiment raises, an
inadmissible file raises :class:`~entropic_substrate.telemetry.TelemetryGap`
(the offending file is kept on disk as evidence, named in the exception), and
nothing is ever captured silently. ``-tiny`` smoke runs are plumbing checks,
not science — excluded unless explicitly asked for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404 -- verify_cloud_sync runs `dvc status` with a fixed argv
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from entropic_substrate.kernel.runtime.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.reader.telemetry import final_eval_record

__all__ = ["capture_checkpoints", "capture_finished_runs", "dvc_commands", "verify_cloud_sync"]

_RESULTS_PREFIX = "results"
_CHECKPOINT_PREFIX = "checkpoints"
_FINAL_SUFFIX = "-final.pt"
_TINY_SUFFIX = "-tiny"


def _resolve_experiment_ids(client: Any, experiment_names: Sequence[str]) -> list[str]:
    experiment_ids: list[str] = []
    for name in experiment_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            msg = f"experiment {name!r} not found on the tracking server"
            raise LookupError(msg)
        experiment_ids.append(experiment.experiment_id)
    return experiment_ids


def _download_artifact(
    client: Any, run_id: str, artifact_path: str, dest: Path, target: Path
) -> None:
    """Download one MLflow artifact straight to ``target`` inside ``dest``.

    ``download_artifacts`` recreates the artifact tree under whatever ``dst``
    it is given; staging in a scratch dir first (then moving the one file we
    want) keeps ``dest`` holding only the flat capture set, shared by both
    the results-JSONL and the checkpoint capture lanes below.
    """
    with tempfile.TemporaryDirectory(dir=dest) as scratch:
        local = Path(client.download_artifacts(run_id, artifact_path, scratch))
        shutil.move(str(local), target)


def _validated_run_name(name: str) -> str:
    """Refuse an MLflow run name that is not a single path component.

    ``run_name`` is REMOTE data interpolated into a capture target path.
    ``dest / name`` silently REPLACES dest when ``name`` is absolute
    (pathlib semantics) and ``..`` segments traverse out of it — so a run
    rename on the server could steer where capture writes. Fail loud, never
    skip silently (a skipped run would read as "captured everything").
    """
    if name != Path(name).name or name in ("", ".", ".."):
        msg = (
            f"MLflow run name {name!r} is not a safe filename component — "
            "refusing to build a capture path from it"
        )
        raise ValueError(msg)
    return name


def capture_finished_runs(
    client: Any,
    experiment_names: Sequence[str],
    dest: str | Path,
    *,
    include_tiny: bool = False,
) -> list[Path]:
    """Download every FINISHED run's results JSONLs into ``dest``.

    Returns the newly captured paths; files already present in ``dest`` are
    skipped (idempotent — re-running after new runs finish captures only the
    delta, and ``dvc add`` on the directory picks up exactly that delta).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    experiment_ids = _resolve_experiment_ids(client, experiment_names)

    captured: list[Path] = []
    for run in client.search_runs(experiment_ids, max_results=500):
        info = run.info
        if info.status != "FINISHED":
            continue
        if not include_tiny and info.run_name.endswith(_TINY_SUFFIX):
            continue
        run_name = _validated_run_name(info.run_name)
        for artifact in client.list_artifacts(info.run_id, _RESULTS_PREFIX):
            if not artifact.path.endswith(".jsonl"):
                continue
            target = dest / f"{run_name}-{info.run_id[:8]}-{Path(artifact.path).name}"
            if target.exists():
                continue
            _download_artifact(client, info.run_id, artifact.path, dest, target)
            # Admissibility gate — raises TelemetryGap on a file the readers
            # would refuse; target stays on disk as evidence, named in the
            # exception message.
            final_eval_record(target)
            captured.append(target)
            logger.info(f"captured {target}")
    return captured


def capture_checkpoints(
    client: Any,
    experiment_names: Sequence[str],
    dest: str | Path,
    *,
    include_tiny: bool = False,
    strip_optimizer: bool = True,
) -> list[Path]:
    """Download every FINISHED run's FINAL checkpoint into ``dest``.

    Mirrors :func:`capture_finished_runs`'s FINISHED / ``-tiny`` / idempotent
    contract, over ``checkpoints/*-final.pt`` artifacts instead of
    ``results/*.jsonl``. Only the ``-final.pt`` twin is captured — the
    ``-best.pt`` checkpoint lives under the same ``checkpoints/`` prefix, but
    E1.5-D's diagnostics compare a run's FINAL trained state against its
    reconstructed init table, and the plan's own size budget (~340MB/ckpt x
    18 runs ≈ 6GB) is sized for exactly one checkpoint per run.

    Validation is fail-loud through
    :func:`entropic_substrate.checkpoint.normalize_checkpoint` — the same
    admissibility gate every other checkpoint reader in this repo uses —
    rather than telemetry's JSONL predicate. A malformed/truncated ``.pt``
    raises :class:`~entropic_substrate.checkpoint.CheckpointFormatError`
    naming the path; the file is kept on disk as evidence, same idiom as
    :func:`capture_finished_runs`'s ``TelemetryGap``.

    ``strip_optimizer`` (default ``True``): re-saves the downloaded
    checkpoint as ``{format_version, step, model, config}`` only. The
    optimiser and sampler state are training-resume machinery E1.5-D's
    diagnostics never touch, and dropping them is most of the ~1GB ->
    ~340MB/ckpt size reduction the plan's own budget depends on. Tensors are
    moved to CPU before the re-save (``map_location="cpu"`` on load, plus an
    explicit ``.cpu()`` per tensor below) so a captured checkpoint stays
    loadable on an analysis machine with no GPU, regardless of which device
    trained it.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    experiment_ids = _resolve_experiment_ids(client, experiment_names)

    captured: list[Path] = []
    for run in client.search_runs(experiment_ids, max_results=500):
        info = run.info
        if info.status != "FINISHED":
            continue
        if not include_tiny and info.run_name.endswith(_TINY_SUFFIX):
            continue
        run_name = _validated_run_name(info.run_name)
        for artifact in client.list_artifacts(info.run_id, _CHECKPOINT_PREFIX):
            if not artifact.path.endswith(_FINAL_SUFFIX):
                continue
            target = dest / f"{run_name}-{info.run_id[:8]}-{Path(artifact.path).name}"
            if target.exists():
                continue
            _download_artifact(client, info.run_id, artifact.path, dest, target)
            try:
                raw = torch.load(target, map_location="cpu", weights_only=True)
            except Exception as exc:  # re-raised as a named, typed refusal below
                msg = (
                    f"{target}: could not be loaded as a torch checkpoint ({exc}) — "
                    "kept on disk as evidence"
                )
                raise CheckpointFormatError(msg) from exc
            state = normalize_checkpoint(raw, source=str(target))
            # Validated unconditionally, not only under strip_optimizer: a
            # checkpoint missing these keys is not one Trainer.state_dict()
            # ever wrote regardless of whether THIS call strips it — leaving
            # it uncaught when strip_optimizer=False would let a malformed
            # capture sit on disk unflagged until some later, harder-to-
            # trace consumer (e.g. load_checkpoint_model's own state.get(
            # "step", 0) silently defaults instead of refusing) trips on it.
            missing = [key for key in ("step", "model", "config") if key not in state]
            if missing:
                msg = (
                    f"{target}: normalized checkpoint is missing required key(s) "
                    f"{missing} — not a checkpoint Trainer.state_dict() ever wrote, "
                    "kept on disk as evidence"
                )
                raise CheckpointFormatError(msg)
            if strip_optimizer:
                model_state = {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in state["model"].items()
                }
                stripped = {
                    "format_version": state["format_version"],
                    "step": state["step"],
                    "model": model_state,
                    "config": state["config"],
                }
                # Atomic swap, not in-place: `target.exists()` above is the
                # idempotency key this loop re-checks on every future
                # invocation, so a strip write interrupted partway through
                # would otherwise leave a truncated file sitting AT that
                # exact path — corrupt, but still "captured" forever as far
                # as this function is concerned. Writing the stripped
                # payload to a sibling path first and swapping it in with
                # `os.replace` (same filesystem, so it is atomic) means a
                # failed strip leaves the ORIGINAL, still-valid download in
                # place instead.
                tmp_target = target.with_suffix(target.suffix + ".partial")
                try:
                    torch.save(stripped, tmp_target)
                    os.replace(tmp_target, target)
                finally:
                    # An interrupted strip leaves the .partial sibling
                    # behind; `dvc add <dest>` (cmd_capture_results, after
                    # this function returns) would otherwise track that
                    # orphan alongside the real capture set. A no-op once
                    # os.replace has already consumed it.
                    tmp_target.unlink(missing_ok=True)
            captured.append(target)
            logger.info(f"captured {target}")
    return captured


def dvc_commands(dest: str | Path) -> list[list[str]]:
    """The exact DVC invocation pair the CLI runs after a capture."""
    return [["dvc", "add", str(Path(dest))], ["dvc", "push"]]


def verify_cloud_sync(dest: str | Path, *, runner: Callable[..., Any] = subprocess.run) -> None:
    """Affirmatively verify the remote holds ``dest``'s current content.

    Measured 2026-08-22: a checkpoint-capture ``dvc push`` exited 1 (seven
    ``IncompleteBody`` UploadPart errors against R2) AFTER ``dvc add`` had
    already rewritten the pointer file — the pointer on disk referenced a
    blob the remote did not hold, and committing it shipped a capture nobody
    could pull. ``dvc status --cloud`` exits 0 whether or not the remote is
    in sync (measured same day), so this parses the ``--json`` payload
    instead of trusting the return code: only an empty object means the
    remote verifiably has everything the pointer names.
    """
    pointer = Path(f"{dest}.dvc")
    if not pointer.exists():
        msg = f"{pointer} does not exist — nothing was dvc-added for {dest}"
        raise RuntimeError(msg)
    proc = runner(  # nosec B603 B607 -- fixed argv; "dvc" from PATH, same idiom as dvc_commands
        ["dvc", "status", "--cloud", "--json", str(pointer)],
        capture_output=True,
        text=True,
        check=True,
    )
    try:
        status = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        msg = (
            "dvc status --cloud produced unparseable output — remote sync is "
            f"UNVERIFIED, treat the capture as not pushed. stdout={proc.stdout!r} "
            f"stderr={proc.stderr!r}"
        )
        raise RuntimeError(msg) from exc
    if not isinstance(status, dict) or status:
        msg = (
            f"remote is NOT in sync for {pointer}: {status!r} — the pointer is "
            "not safe to commit; re-run dvc push and re-verify"
        )
        raise RuntimeError(msg)
