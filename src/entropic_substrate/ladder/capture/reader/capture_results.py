"""Retrieval-side capture of run results JSONLs into DVC.

Pods run credential-free (operator ruling 2026-08-17): a training pod holds
no R2/DVC identity and uploads its authoritative ``results/*.jsonl`` to the
tracking server as a run artifact (``MlflowSink.finish()``). This module is
the other half of that contract — it runs on an operator machine that DOES
hold R2 credentials, pulls those artifacts down, validates each one through
:func:`entropic_substrate.telemetry.final_eval_record` (the same admissibility
predicate every other JSONL consumer uses), and hands the capture directory to
DVC.

Two lanes, partitioned by the run's ``schema`` tag. A run with no ``schema``
tag, or a tag that does not start with ``h-``, is the E lane: every
``results/*.jsonl`` goes through ``final_eval_record``. A run whose tag starts
with ``h-`` (the H worker's ``h-probe-worker-result-v1``) is the H lane: its
``results/metrics.jsonl`` must be admissible through
:func:`~entropic_substrate.ladder.capture.reader.h_telemetry.final_h_run_end`
(exactly one ``run_start``, one ``run_end`` and a committed step row) and any
other ``results/*.jsonl`` (``worker-log.jsonl``) must be well-formed JSON
lines. The lane is chosen per run, never by a flag that could skip runs.

Measurement path: it fails LOUD. An unknown experiment raises, an
inadmissible file raises :class:`~entropic_substrate.telemetry.TelemetryGap`
(the offending file is kept on disk as evidence, named in the exception), and
nothing is ever captured silently. ``-tiny`` smoke runs are plumbing checks,
not science — excluded unless explicitly asked for.

A capture pass is atomic with respect to ``dest`` (FOR-1650), in both lanes.
Every artifact of the pass is downloaded and validated in a per-pass staging
directory ``<dest>.pending/<pass-id>/``, a SIBLING of ``dest`` so that
``dvc add <dest>`` can never track it. Only after the whole pass has validated
are its files moved into ``dest`` (``os.replace``) and returned. A refused or
failed pass moves nothing: ``dest`` is exactly as it was, and the staging
directory stays as evidence (the refused file is named in the exception). So "a
file in ``dest``" means "captured by a fully admissible pass", and the
``target.exists()`` idempotency key keeps that meaning. Before this, a refusal
midway left the admissible files already moved into ``dest`` untracked, the next
pass skipped them, and the CLI read that as "nothing new to capture".

Status gate (DN-1642-FAILED-CAPTURE, conductor 2026-10-15, option A). The gate is an
allowlist, lane-aware: the E lane admits ``FINISHED`` only; the H lane admits ``FINISHED``
and ``FAILED``, because a failed H run that committed a step is evidence (the worker now
terminates its MLflow run ``FAILED`` on its exception path and its ``run_end`` says
``failed``). Every other status (``RUNNING``, ``SCHEDULED``, ``KILLED``) is skipped on both
lanes before any tag is read or any run name validated. A ``FAILED`` run that is not
``h-`` tagged is skipped without further checks, so the E lane observably keeps its
FINISHED-only gate. The same admissibility predicate applies, so a ``FAILED`` H run that
died before its first committed step, before ``run_start``, or without ``metrics.jsonl``
is refused and fails the whole pass with exit 2, exactly as the same run would under
``FINISHED``.

AMENDMENT (house rule 4, FOR-1693, DH-46, 2026-11-01): the sentence above stands for every
run EXCEPT one whose ``run_end`` records a scientific stop (``status == failed`` and
``result_status == scientific_gap``): a codec warm-up that diverged before its first
committed step is a recorded scientific outcome, and ``final_h_run_end`` now admits that
stream with zero committed rows, so the capture pass no longer exits 2 on it. The exception is
keyed on the recorded stop, never on the row count; a ``FAILED`` run that died before its
first committed step WITHOUT that record is refused exactly as before.

``capture_checkpoints`` keeps its FINISHED-only gate (H runs upload no checkpoints).

An H-tagged run with no ``h-run-lifecycle-v1`` records is refused whatever produced it: a
worker from before FOR-1639 is. The local tracer (schema ``h-tiny-trace-v1``) and the local
``h-train-probe`` (schema ``h-train-probe-v1``) bypass ``object_worker`` and write none, so
since FOR-1642 (TRACER-PARTITION, conductor 2026-10-14) they name their runs with the
``-tiny`` suffix (``h-tiny-trace-<identity>-tiny``, ``h-train-probe-<identity>-tiny``): the
``include_tiny`` gate, which runs before the lane partition, excludes them by default and,
when opted in, they are refused here as lifecycle-less. This module carries no legacy arm
and no skip for these runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess  # nosec B404 -- verify_cloud_sync runs `dvc status` with a fixed argv
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from entropic_substrate.kernel.runtime.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.reader.h_telemetry import (
    final_h_run_end,
    require_json_object_lines,
)
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap, final_eval_record

__all__ = ["capture_checkpoints", "capture_finished_runs", "dvc_commands", "verify_cloud_sync"]

_RESULTS_PREFIX = "results"
_CHECKPOINT_PREFIX = "checkpoints"
_FINAL_SUFFIX = "-final.pt"
_TINY_SUFFIX = "-tiny"
_FINISHED_STATUS = "FINISHED"
_FAILED_STATUS = "FAILED"
_CAPTURE_STATUSES = (_FINISHED_STATUS, _FAILED_STATUS)
_SCHEMA_TAG = "schema"
_H_SCHEMA_PREFIX = "h-"
_H_METRICS_ARTIFACT = f"{_RESULTS_PREFIX}/metrics.jsonl"


def _resolve_experiment_ids(client: Any, experiment_names: Sequence[str]) -> list[str]:
    experiment_ids: list[str] = []
    for name in experiment_names:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            msg = f"experiment {name!r} not found on the tracking server"
            raise LookupError(msg)
        experiment_ids.append(experiment.experiment_id)
    return experiment_ids


class _CapturePass:
    """One capture pass: stage and validate everything, then commit it to ``dest``.

    Files are staged flat under ``<dest>.pending/<pass-id>/`` (a sibling of
    ``dest``, built with the same ``f"{dest}..."`` idiom as
    :func:`verify_cloud_sync`'s pointer path), named exactly as they will be in
    ``dest``. The staging directory is created at the first download, so a pass
    that fails before downloading anything leaves no trace.

    :meth:`commit` is the pass's only effect on ``dest`` and its last fallible
    step. Callers run it after their whole loop, never from a handler, so ANY
    exception before it (a refusal, a bad run name, an MLflow or OS error, an
    interrupt) leaves ``dest`` unchanged and the staging directory in place.
    """

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self._root = Path(f"{dest}.pending")
        self._dir: Path | None = None
        self._staged: list[Path] = []

    def claims(self, name: str) -> bool:
        """True when ``name`` is already staged in this pass (first wins)."""
        return any(path.name == name for path in self._staged)

    def stage(self, client: Any, run_id: str, artifact_path: str, name: str) -> Path:
        """Download one MLflow artifact into staging as ``name`` and return its path.

        ``download_artifacts`` recreates the artifact tree under whatever ``dst``
        it is given, and every H run carries the same ``results/metrics.jsonl``.
        So each download lands in a scratch directory of its own, is renamed to
        its flat, run-scoped ``name`` using the path ``download_artifacts``
        RETURNS, and the scratch directory is removed before the next download:
        no two artifacts ever share a relative path in staging, and a staged file
        is never overwritten.
        """
        if self._dir is None:
            self._dir = self._root / uuid.uuid4().hex
            self._dir.mkdir(parents=True)
        staged = self._dir / name
        scratch = self._dir / ".download"
        local = Path(client.download_artifacts(run_id, artifact_path, str(scratch)))
        os.replace(local, staged)
        shutil.rmtree(scratch, ignore_errors=True)
        self._staged.append(staged)
        return staged

    def commit(self) -> list[Path]:
        """Move every staged file into ``dest`` and return the ``dest`` paths.

        ``os.replace`` only, never ``shutil.move``: across filesystems the latter
        silently falls back to copy-then-delete, which is not atomic, and a
        cross-filesystem staging setup must fail loud at the first rename with
        nothing moved. Nothing that can fail for a non-rename reason runs between
        the first and the last rename; the staging directory is removed only
        after the last one and its removal can never raise.

        A rename that fails after earlier ones landed (``EACCES``, ``ENOSPC``)
        rolls those back into staging before the error propagates, so ``dest``
        is as it was and the whole pass is retried next time. The rollback is
        best effort: if a rollback rename itself fails, that file stays in
        ``dest`` and the original error still propagates.
        """
        moved: list[tuple[Path, Path]] = []
        try:
            for staged in self._staged:
                target = self.dest / staged.name
                os.replace(staged, target)
                moved.append((staged, target))
        except BaseException:
            for staged, target in reversed(moved):
                with contextlib.suppress(OSError):
                    os.replace(target, staged)
            raise
        self._cleanup()
        for _, target in moved:
            logger.info(f"captured {target}")
        return [target for _, target in moved]

    def _cleanup(self) -> None:
        """Remove a fully committed pass's directory; never raises, never deletes content.

        ``rmdir`` only, not ``rmtree``: after the last rename the pass directory
        is empty, so removal succeeds. Anything still inside is not a committed
        file, so it is left in place and logged with its path rather than
        deleted.
        """
        if self._dir is None:
            return
        with contextlib.suppress(OSError):
            self._dir.rmdir()
        if self._dir.exists():
            leftover = sorted(path.name for path in self._dir.iterdir())
            logger.warning(
                f"capture staging left behind after a clean pass: {self._dir} {leftover}"
            )
            return
        with contextlib.suppress(OSError):
            self._root.rmdir()  # only succeeds when no refused pass's evidence remains


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


def _run_schema_tag(run: Any, run_id: str) -> str | None:
    """The run's ``schema`` tag, or None when the run carries none.

    A run object with no ``data`` or no ``tags`` (an older client, a fake) is
    treated as untagged, i.e. the E lane. A tag that is present but not a
    string is a defect in the tracking data and raises.
    """
    try:
        tags = run.data.tags
    except AttributeError:
        return None
    tag = tags.get(_SCHEMA_TAG)
    if tag is None:
        return None
    if not isinstance(tag, str):
        msg = f"run {run_id}: tag {_SCHEMA_TAG!r} is {type(tag).__name__}, not a string"
        raise TypeError(msg)
    return tag


def _capture_h_run(
    client: Any, run_id: str, run_name: str, dest: Path, staging: _CapturePass
) -> None:
    """Stage and validate one H-lane run's ``results/*.jsonl`` through the H predicates.

    The run must carry exactly one ``results/metrics.jsonl``: a worker that
    died before its first ``sink.log`` still finishes its MLflow run, and
    capturing its ``worker-log.jsonl`` alone would read as success. Downloads
    go to the pass's staging directory and are committed to ``dest`` by the
    caller only when the WHOLE pass is admissible (FOR-1650), so a refusal here
    moves nothing and the refused file stays in staging, named in the exception.
    A file already in ``dest`` is re-validated rather than skipped, so one that
    fails validation (for example stranded there before FOR-1650) is refused
    again on the next invocation instead of reading as "nothing new to
    capture"; it is not returned, being already captured. A target name already
    staged earlier in this pass is skipped, first wins.
    """
    paths = [
        artifact.path
        for artifact in client.list_artifacts(run_id, _RESULTS_PREFIX)
        if artifact.path.endswith(".jsonl")
    ]
    if paths.count(_H_METRICS_ARTIFACT) != 1:
        msg = (
            f"run {run_id} ({run_name}): an H run must carry exactly one "
            f"{_H_METRICS_ARTIFACT}, found {paths.count(_H_METRICS_ARTIFACT)} "
            f"among {sorted(paths)} — refusing to capture its other files alone"
        )
        raise TelemetryGap(msg)
    for artifact_path in paths:
        name = f"{run_name}-{run_id[:8]}-{Path(artifact_path).name}"
        if staging.claims(name):
            continue
        local = dest / name
        if not local.exists():
            local = staging.stage(client, run_id, artifact_path, name)
        # Admissibility gate — raises TelemetryGap; the file stays on disk as
        # evidence, named in the exception message.
        if artifact_path == _H_METRICS_ARTIFACT:
            final_h_run_end(local)
        else:
            require_json_object_lines(local)


def capture_finished_runs(
    client: Any,
    experiment_names: Sequence[str],
    dest: str | Path,
    *,
    include_tiny: bool = False,
) -> list[Path]:
    """Download every admitted run's results JSONLs into ``dest``, atomically.

    Admitted means ``FINISHED`` (both lanes) or ``FAILED`` on the H lane only; see the
    module docstring's status gate (DN-1642-FAILED-CAPTURE).

    The pass is atomic with respect to ``dest`` (FOR-1650): every file is
    downloaded and validated in the pass's staging directory
    ``<dest>.pending/<pass-id>/`` and moved into ``dest`` only once the whole
    pass is admissible. Any exception (a refusal, an MLflow or OS error)
    moves nothing and leaves the staging directory as evidence.

    Returns the newly captured paths, in ``dest``; files already present in
    ``dest`` are skipped (idempotent — re-running after new runs finish
    captures only the delta, and ``dvc add`` on the directory picks up
    exactly that delta).

    Runs are partitioned by their ``schema`` tag (see the module docstring):
    the E lane below is unchanged; an ``h-`` tagged run takes the H lane
    through :func:`_capture_h_run`. The status, ``-tiny`` and run-name gates run
    first (a ``FAILED`` run is admitted only if it is ``h-`` tagged).
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    experiment_ids = _resolve_experiment_ids(client, experiment_names)

    staging = _CapturePass(dest)
    for run in client.search_runs(experiment_ids, max_results=500):
        info = run.info
        # Allowlist, never a RUNNING denylist: SCHEDULED and KILLED are skipped on both
        # lanes before any tag read (DN-1642-FAILED-CAPTURE).
        if info.status not in _CAPTURE_STATUSES:
            continue
        if info.status == _FAILED_STATUS:
            # FAILED is admitted on the H lane only. The lane is decided first, with the
            # unchanged tag reader, so a FAILED E run is skipped with no further check.
            failed_tag = _run_schema_tag(run, info.run_id)
            if failed_tag is None or not failed_tag.startswith(_H_SCHEMA_PREFIX):
                continue
        if not include_tiny and info.run_name.endswith(_TINY_SUFFIX):
            continue
        run_name = _validated_run_name(info.run_name)
        schema_tag = _run_schema_tag(run, info.run_id)
        if schema_tag is not None and schema_tag.startswith(_H_SCHEMA_PREFIX):
            _capture_h_run(client, info.run_id, run_name, dest, staging)
            continue
        for artifact in client.list_artifacts(info.run_id, _RESULTS_PREFIX):
            if not artifact.path.endswith(".jsonl"):
                continue
            name = f"{run_name}-{info.run_id[:8]}-{Path(artifact.path).name}"
            if (dest / name).exists() or staging.claims(name):
                continue
            staged = staging.stage(client, info.run_id, artifact.path, name)
            # Admissibility gate — raises TelemetryGap on a file the readers
            # would refuse; the staged file stays on disk as evidence, named in
            # the exception message, and nothing of this pass reaches dest.
            final_eval_record(staged)
    return staging.commit()


def capture_checkpoints(
    client: Any,
    experiment_names: Sequence[str],
    dest: str | Path,
    *,
    include_tiny: bool = False,
    strip_optimizer: bool = True,
) -> list[Path]:
    """Download every FINISHED run's FINAL checkpoint into ``dest``, atomically.

    Mirrors :func:`capture_finished_runs`'s FINISHED / ``-tiny`` / idempotent
    / atomic-pass contract (FOR-1650: each checkpoint is downloaded, validated
    and stripped in the pass's staging directory ``<dest>.pending/<pass-id>/``,
    one at a time to keep the disk budget below, and moved into ``dest`` only
    once EVERY checkpoint of the pass is admissible), over ``checkpoints/*-final.pt`` artifacts instead of
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
    naming the staged path; the file is kept in staging as evidence, same
    idiom as :func:`capture_finished_runs`'s ``TelemetryGap``.

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

    staging = _CapturePass(dest)
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
            name = f"{run_name}-{info.run_id[:8]}-{Path(artifact.path).name}"
            if (dest / name).exists() or staging.claims(name):
                continue
            staged = staging.stage(client, info.run_id, artifact.path, name)
            try:
                raw = torch.load(staged, map_location="cpu", weights_only=True)
            except Exception as exc:  # re-raised as a named, typed refusal below
                msg = (
                    f"{staged}: could not be loaded as a torch checkpoint ({exc}) — "
                    "kept on disk as evidence"
                )
                raise CheckpointFormatError(msg) from exc
            state = normalize_checkpoint(raw, source=str(staged))
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
                    f"{staged}: normalized checkpoint is missing required key(s) "
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
                # Atomic swap, not in-place, and inside staging: a strip write
                # interrupted partway through would otherwise leave a truncated
                # file at the staged path that the rest of this pass (and the
                # evidence a refused pass keeps) would take for the real
                # checkpoint. Writing the stripped payload to a sibling path
                # first and swapping it in with `os.replace` (same directory,
                # so it is atomic) means a failed strip leaves the ORIGINAL,
                # still-valid download in place instead. The swap happens per
                # file, before the next download, so staging never holds more
                # than one unstripped checkpoint.
                tmp_target = staged.with_suffix(staged.suffix + ".partial")
                try:
                    torch.save(stripped, tmp_target)
                    os.replace(tmp_target, staged)
                finally:
                    # An interrupted strip leaves the .partial sibling behind
                    # in staging; remove it so the evidence a failed pass keeps
                    # is the original download alone. A no-op once os.replace
                    # has already consumed it. (It could never reach `dest`:
                    # only the pass's staged names are committed, and a
                    # committed pass has no failed strip.)
                    tmp_target.unlink(missing_ok=True)
    return staging.commit()


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
