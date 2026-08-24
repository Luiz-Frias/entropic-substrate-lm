"""Mutation guards for typed D1 checkpoint admission and continuation authority."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from d1_checkpoint_fixtures import CheckpointCase, checkpoint_state, set_path
from entropic_substrate.data.batching import SequenceSampler, TokenSplit
from entropic_substrate.predictive_state.checkpoint_envelope import (
    D1CheckpointError,
    validate_d1_checkpoint,
)
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

pytestmark = pytest.mark.unit


def _validate(state: object, case: CheckpointCase, *, resume: bool = False) -> object:
    return validate_d1_checkpoint(
        state,
        system=case.system,
        optimizer=case.optimizer,
        sampler=case.sampler,
        run=case.run,
        protocol=case.protocol,
        for_resume=resume,
    )


def test_valid_typed_checkpoint_is_admitted_for_resume() -> None:
    state, case = checkpoint_state()
    assert _validate(state, case, resume=True) is state


def test_cold_unfitted_runtime_admits_serialized_baseline_before_state_load() -> None:
    state, case = checkpoint_state()
    family = case.run.family
    assert family is not None
    system = D1PredictiveSystem.from_family(
        family,
        case.run.draft.arm,
        initializer=SeededD1BranchInitializer(seed=case.run.execution.seed),
    )
    optimizer = torch.optim.AdamW(system.optimizer_parameters(), lr=6e-4)
    sampler = SequenceSampler(
        TokenSplit(np.arange(4096, dtype=np.int64) % 11),
        case.protocol.batching.batch_size,
        case.protocol.batching.sequence_length,
        seed=case.run.execution.seed,
    )

    assert system.mean_displacement_baseline.is_fitted is False
    assert (
        validate_d1_checkpoint(
            state,
            system=system,
            optimizer=optimizer,
            sampler=sampler,
            run=case.run,
            protocol=case.protocol,
            for_resume=True,
        )
        is state
    )


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("kind",), "legacy", "kind"),
        (("schema_version",), 0, "schema"),
        (("rung",), "d2", "rung"),
        (("successful_step",), 1, "evaluation boundary"),
        (("identity", "construction_digest"), "4" * 64, "construction"),
        (("identity", "family_digest"), "4" * 64, "family"),
        (("identity", "aggregation_digest"), "4" * 64, "aggregation"),
        (("identity", "execution_digest"), "4" * 64, "execution"),
        (("identity", "run_manifest"), {}, "run_manifest"),
        (("baseline_calibration", "protocol_digest"), "4" * 64, "protocol"),
        (("baseline_calibration", "spec_digest"), "4" * 64, "spec"),
        (("baseline_calibration", "split_digest"), "4" * 64, "split_digest"),
        (("baseline_calibration", "mean_parameter_digest"), "4" * 64, "mean_parameter"),
        (("baseline_calibration", "num_batches"), 31, "num_batches"),
        (("baseline_calibration", "online_positions"), 1, "online_positions"),
        (("baseline_calibration", "source_span_positions"), 1, "source_span_positions"),
        (("baseline_calibration", "codec_digest"), "4" * 64, "codec"),
        (("baseline_calibration", "initialization_seed"), 8, "initialization_seed"),
        (("baseline_calibration", "successful_step_at_fit"), 1, "step zero"),
        (("resource_ledger", "last_training_step"), 1, "last_training_step"),
        (("resource_ledger", "logical_run_id"), "other", "segment"),
        (("resource_ledger", "execution_segment_id"), "wrong", "segment"),
        (("journal", "logical_run_id"), "other", "logical_run_id"),
        (("journal", "execution_digest"), "4" * 64, "execution_digest"),
        (("journal", "successful_step"), 1, "successful_step"),
        (("journal", "prefix_sha256"), "bad", "prefix_sha256"),
        (("sampler", "generator"), None, "sampler.generator"),
        (("rng", "torch_cpu"), None, "rng.torch_cpu"),
        (("best_nats_per_byte",), float("nan"), "best_nats_per_byte"),
    ],
)
def test_identity_and_typed_state_mutations_fail_loud(
    path: tuple[str, ...], value: object, match: str
) -> None:
    state, case = checkpoint_state()
    set_path(state, path, value)
    with pytest.raises(D1CheckpointError, match=match):
        _validate(state, case)


def test_obsolete_resource_keys_and_ema_cursor_drift_are_refused() -> None:
    state, case = checkpoint_state()
    set_path(state, ("resource_ledger", "source_bytes"), 0)
    with pytest.raises(D1CheckpointError, match="resource ledger"):
        _validate(state, case)

    state, case = checkpoint_state()
    model = cast(dict[str, object], state["model"])
    extra = cast(dict[str, object], model["_extra_state"])
    cast(dict[str, object], extra["target_policy"])["update_cursor"] = 1
    with pytest.raises(D1CheckpointError, match="EMA update_cursor"):
        _validate(state, case)


def test_resource_and_journal_logical_run_continuity_is_required() -> None:
    state, case = checkpoint_state()
    set_path(state, ("resource_ledger", "logical_run_id"), "other")
    set_path(state, ("resource_ledger", "execution_segment_id"), "other:segment:0")

    with pytest.raises(D1CheckpointError, match="logical_run_id"):
        _validate(state, case)


@pytest.mark.parametrize(("status", "reason"), [("completed", "filed"), ("aborted", "oom")])
def test_terminal_snapshots_are_archival_but_not_resumable(status: str, reason: str) -> None:
    state, case = checkpoint_state()
    set_path(state, ("lifecycle", "state"), status)
    set_path(
        state,
        ("lifecycle", "completion_reason" if status == "completed" else "abort_reason"),
        reason,
    )

    assert _validate(state, case) is state
    with pytest.raises(D1CheckpointError, match="not resumable"):
        _validate(state, case, resume=True)
