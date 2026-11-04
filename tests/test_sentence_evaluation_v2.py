"""FOR-1704 (L4d): the ``h-predictive-evaluation-v2`` record, its reader and its telemetry.

These drive the real engine, so they need the reader members' bytes and ride the DG-A patch.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_h_worker_log_lines import bind_logs, read_log
from tests.test_sentence_execution import engine

from entropic_substrate.core.sentence import scoring
from entropic_substrate.core.sentence.law import DirectionalMagnitudeLaw, LawSpec
from entropic_substrate.core.sentence.scoring import (
    LatentUsageOperand,
    derived_stream_keys,
    importance_evaluate,
)
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run import evaluation_records, trace_capture
from entropic_substrate.ladder.run.evaluation_records import read_evaluation
from entropic_substrate.ladder.run.execution import SentenceExecution, _log_rows
from entropic_substrate.ladder.run.h_metric_extractor import (
    EVALUATION_SCHEMAS,
    project_evaluation,
)
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.runner import run_probe_training

pytestmark = pytest.mark.unit

LIMIT = 10_000_000
V1_EXAMPLE = {
    "source_batch",
    "source_examples",
    "source_bytes",
    "boundary_symbols",
    "by_sample_count",
    "log_weights",
    "decoder_log_prob",
    "prior_log_prob",
    "proposal_log_prob",
}


def _run(root: Path, *, mode: str = "magnitude", batches: int = 1) -> tuple[Any, Path, Any]:
    """One finished run; ``(engine template, output root, first evaluation artifact)``."""
    run = engine(root)
    base = run.profile
    changed = replace(
        base,
        law=LawSpec(8, mode),  # type: ignore[arg-type]
        schedule=replace(base.schedule, evaluation_batches=batches),
    )
    template = SentenceExecution(changed, run.source_root, max_source_bytes=1_000_000)
    output = root / "run"
    output.mkdir()
    result = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        latent_usage=POSTERIOR_MEAN,
    )
    progress = read_json(output, result.receipt, max_bytes=LIMIT)
    return template, output, artifact_from_mapping(progress["evaluations"][0]["artifact"])


def _read(template: Any, output: Path, artifact: Any) -> dict[str, Any]:
    return read_evaluation(
        output,
        artifact,
        template.profile,
        template.source_root,
        max_bytes=LIMIT,
        max_source_bytes=1_000_000,
    )


def _tampered(
    template: Any, output: Path, artifact: Any, mutate: Callable[[dict[str, Any]], None]
) -> Any:
    payload = copy.deepcopy(read_json(output, artifact, max_bytes=LIMIT))
    mutate(payload)
    # one fresh name per call: an artifact path is write-once
    name = f"tampered-{len(list(output.glob('tampered-*.json')))}-evaluation.json"
    return write_json(output, name, payload, max_bytes=LIMIT)


# ------------------------------------------------------------------------------ record shape
@pytest.mark.parametrize("mode", ["magnitude", "sphere"])
def test_the_v2_record_has_exact_key_sets_with_q_at_its_v1_keys(tmp_path: Path, mode: str) -> None:
    template, output, artifact = _run(tmp_path, mode=mode)
    record = read_json(output, artifact, max_bytes=LIMIT)
    assert record["schema"] == "h-predictive-evaluation-v2"
    assert record["latent_operand"] == "posterior_mean"
    assert set(record) == {
        "schema",
        "meaning",
        "run_identity",
        "step",
        "examples",
        "wall_seconds",
        "cuda_seconds",
        "checkpoint_seal",
        "latent_operand",
    }
    example = record["examples"][0]
    assert set(example) == V1_EXAMPLE | {"proposals", "diagnostics", "latent_usage"}
    assert set(example["proposals"]) == {"prior", "mixture"}
    assert set(example["proposals"]["prior"]) == {
        "proposal",
        "by_sample_count",
        "log_weights",
        "decoder_log_prob",
        "prior_log_prob",
    }
    assert set(example["proposals"]["mixture"]) == {
        "proposal",
        "allocation",
        "component",
        "by_sample_count",
        "log_weights",
        "decoder_log_prob",
        "q_log_prob",
        "prior_log_prob",
    }
    assert example["proposals"]["prior"]["proposal"] == "prior"
    assert example["proposals"]["mixture"]["proposal"] == "mixture"
    diagnostics = example["diagnostics"]
    assert set(diagnostics) == {
        "prior_log_radius_mean",
        "q_log_radius_mean",
        "prior_q_cosine",
        "log_weight_sd",
    }
    assert set(diagnostics["log_weight_sd"]) == {"q", "prior", "mixture"}
    radius_present = mode == "magnitude"
    assert (diagnostics["prior_log_radius_mean"] is not None) is radius_present
    assert (diagnostics["q_log_radius_mean"] is not None) is radius_present
    assert set(example["latent_usage"]) == {"q_decoder_log_prob", "null_decoder_log_prob"}
    assert _read(template, output, artifact)["schema"] == "h-predictive-evaluation-v2"


def test_the_extractor_reads_the_real_record_the_same_in_memory_and_after_the_json_round_trip(
    tmp_path: Path,
) -> None:
    template, output, artifact = _run(tmp_path)
    record = _read(template, output, artifact)
    metrics = project_evaluation(record)
    assert metrics is not None
    for key in (
        "eval/ess_k2",
        "eval/ess_prior_k4",
        "eval/ess_mixture_k8",
        "eval/log_marginal_per_byte_ratio_of_sums_prior_k2",
        "eval/log_weight_sd_mixture_k4",
        "eval/prior_log_radius_mean",
        "eval/q_log_radius_mean",
        "eval/prior_q_cosine_mean",
        "eval/latent_usage_posterior_mean_nats_per_byte",
    ):
        assert key in metrics, key
    assert project_evaluation(json.loads(json.dumps(record))) == metrics


def test_a_sphere_cell_emits_no_log_radius_keys_end_to_end(tmp_path: Path) -> None:
    template, output, artifact = _run(tmp_path, mode="sphere")
    metrics = project_evaluation(_read(template, output, artifact))
    assert metrics is not None
    assert "eval/prior_log_radius_mean" not in metrics and "eval/q_log_radius_mean" not in metrics
    assert "eval/prior_q_cosine_mean" in metrics


def test_the_attempt_record_keeps_the_exact_v1_example_key_set(tmp_path: Path) -> None:
    """The attempt's ``examples`` was the SAME list as the returned record's; a v2-shaped example
    inside the unchanged-name v1 forensic record would be an unruled widening."""
    run = _prediction_run(tmp_path, 1)
    record = run.evaluate(POSTERIOR_MEAN)
    attempt = run.last_evaluation
    assert attempt is not None and attempt["schema"] == "h-evaluation-attempt-v1"
    assert attempt["examples"] is not record["examples"]
    for example in attempt["examples"]:  # type: ignore[attr-defined]
        assert set(example) == V1_EXAMPLE
    for example in record["examples"]:  # type: ignore[attr-defined]
        assert set(example) == V1_EXAMPLE | {"proposals", "diagnostics", "latent_usage"}


def test_evaluate_has_no_default_estimator() -> None:
    run_signature = SentenceExecution.evaluate.__code__
    assert run_signature.co_varnames[: run_signature.co_argcount] == ("self", "latent_usage")
    assert SentenceExecution.evaluate.__defaults__ is None


def test_the_record_names_the_operand_the_injected_implementation_declares(
    tmp_path: Path,
) -> None:
    class Declared:
        operand = LatentUsageOperand.draw_average

        def q_decoder_log_prob(self, score, posterior):  # type: ignore[no-untyped-def]
            return score(scoring.posterior_mean_vector(posterior))

    run = _prediction_run(tmp_path, 1)
    assert run.evaluate(Declared())["latent_operand"] == "draw_average"  # type: ignore[arg-type]
    assert run.evaluate(POSTERIOR_MEAN)["latent_operand"] == "posterior_mean"


# ------------------------------------------------------------------------------ RNG isolation
def _q_oracle(run: SentenceExecution) -> list[dict[str, Any]]:
    """The v1 q-only evaluation, replayed in this process under the same step-keyed seed."""
    run.validation_sampler.restore(run.validation_sampler.snapshot() | {"cursor": 0})
    rows: list[dict[str, Any]] = []
    run.model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed((run.profile.seed * 0x9E3779B1 + run.step) % 2**32)
        for _ in range(run.profile.schedule.evaluation_batches):
            batch = run.validation_sampler.next_batch()
            with torch.autocast(
                run.device.type, dtype=torch.bfloat16, enabled=run.profile.precision == "bf16"
            ):
                q = importance_evaluate(
                    run.model,
                    batch.history,
                    batch.target,
                    sample_counts=run.profile.schedule.evaluated_samples,
                    sample_chunk_size=run.profile.schedule.importance_chunk,
                )
            rows.append(
                {
                    "log_weights": _log_rows(q.log_weights),
                    "decoder_log_prob": _log_rows(q.decoder_log_prob),
                    "prior_log_prob": _log_rows(q.prior_log_prob),
                    "proposal_log_prob": _log_rows(q.proposal_log_prob),
                    "by_sample_count": {
                        str(k): [r.__dict__ for r in v] for k, v in q.by_sample_count.items()
                    },
                }
            )
    return rows


def _prediction_run(root: Path, batches: int) -> SentenceExecution:
    run = engine(root)
    changed = replace(
        run.profile, schedule=replace(run.profile.schedule, evaluation_batches=batches)
    )
    template = SentenceExecution(changed, run.source_root, max_source_bytes=1_000_000)
    template.advance()
    template.advance()
    template.begin_prediction()
    template.advance()
    return template


def test_the_q_rows_equal_the_v1_algorithm_over_several_batches_in_process(
    tmp_path: Path,
) -> None:
    """C3/C4: with 2+ batches an extra draw on the shared stream would shift batch 2's q draws,
    which a single-batch test cannot see. The oracle is the v1 q-only loop, same process."""
    run = _prediction_run(tmp_path, 3)
    record = run.evaluate(POSTERIOR_MEAN)
    oracle = _q_oracle(run)
    assert len(record["examples"]) == 3  # type: ignore[arg-type]
    for example, expected in zip(record["examples"], oracle, strict=True):  # type: ignore[arg-type]
        for name, value in expected.items():
            assert example[name] == value, name  # exact equality, not a tolerance


def test_the_oracle_goes_red_when_the_extra_draws_share_the_q_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard teeth for the oracle above: no isolation (a no-op stream) moves batch 2 onward."""
    run = _prediction_run(tmp_path, 3)

    @contextlib.contextmanager
    def shared(key: int, like: torch.Tensor):  # type: ignore[no-untyped-def]
        torch.manual_seed(key)  # still seeds, but never restores: the q stream is consumed
        yield

    monkeypatch.setattr(scoring, "_stream", shared)
    record = run.evaluate(POSTERIOR_MEAN)
    monkeypatch.undo()
    oracle = _q_oracle(run)
    equal = [
        example["log_weights"] == expected["log_weights"]  # type: ignore[index]
        for example, expected in zip(record["examples"], oracle, strict=True)  # type: ignore[arg-type]
    ]
    assert not all(equal)


def test_the_extra_streams_depend_only_on_seed_step_batch_and_proposal(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = _prediction_run(tmp_path / "a", 2)
    second = _prediction_run(tmp_path / "b", 2)
    torch.manual_seed(1)
    a = first.evaluate(POSTERIOR_MEAN)
    torch.manual_seed(2)  # the CALLER's stream state is irrelevant: the evaluation re-seeds
    b = second.evaluate(POSTERIOR_MEAN)
    for x, y in zip(a["examples"], b["examples"], strict=True):  # type: ignore[arg-type]
        assert x["proposals"] == y["proposals"]
        assert x["log_weights"] == y["log_weights"]
    keys = {derived_stream_keys(seed=42, step=3, batch=i, q_key=7) for i in range(2)}
    assert len(keys) == 2


# ------------------------------------------------------------------------------ exact -inf
def test_an_antipodal_mixture_draw_survives_the_artifact_write_and_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A7: an exact antipode is a genuine zero density (-inf). It must reach the record as the
    string "-inf" in the columns that admit it and be re-derived by the reader, not kill the
    run at the write (journal.write_json uses allow_nan=False)."""
    original = SentenceExecution.evaluate

    def antipodal(self: SentenceExecution, latent_usage: Any) -> dict[str, object]:
        original_rsample = DirectionalMagnitudeLaw.rsample
        state = {"calls": 0, "q": None}

        def rsample(law: DirectionalMagnitudeLaw, shape: tuple[int, ...] = ()) -> torch.Tensor:
            n = state["calls"]
            state["calls"] += 1
            value = original_rsample(law, shape)
            if n % 4 == 2:  # the mixture's q draws: remember the posterior mean direction
                state["q"] = law.direction
            if n % 4 == 3:  # the mixture's prior draws: the exact antipode of q's mean direction
                return (-state["q"]).expand_as(value).contiguous()  # type: ignore[operator]
            return value

        with monkeypatch.context() as patch:
            patch.setattr(DirectionalMagnitudeLaw, "rsample", rsample)
            return original(self, latent_usage)

    monkeypatch.setattr(SentenceExecution, "evaluate", antipodal)
    template, output, artifact = _run(tmp_path, mode="sphere")
    record = read_json(output, artifact, max_bytes=LIMIT)
    mixture = record["examples"][0]["proposals"]["mixture"]
    flat = [v for row in mixture["q_log_prob"] for v in row]
    assert "-inf" in flat and any(v != "-inf" for v in flat)
    assert "-inf" not in [v for row in mixture["decoder_log_prob"] for v in row]
    assert _read(template, output, artifact)["schema"] == "h-predictive-evaluation-v2"
    assert project_evaluation(_read(template, output, artifact)) is not None


# ------------------------------------------------------------------------------ the reader
def test_the_reader_refuses_a_producing_component_with_a_zero_density(tmp_path: Path) -> None:
    template, output, artifact = _run(tmp_path, mode="sphere")

    def zero_producer(payload: dict[str, Any]) -> None:
        mixture = payload["examples"][0]["proposals"]["mixture"]
        mixture["q_log_prob"][0][0] = "-inf"
        mixture["component"][0][0] = 0  # drawn from q, so its own density cannot be zero

    with pytest.raises(ValueError, match="mixture importance terms"):
        _read(template, output, _tampered(template, output, artifact, zero_producer))


def test_a_v1_named_record_with_v2_keys_and_a_v2_record_missing_one_are_refused(
    tmp_path: Path,
) -> None:
    template, output, artifact = _run(tmp_path)

    def v1_with_v2(payload: dict[str, Any]) -> None:
        payload["schema"] = "h-predictive-evaluation-v1"

    def v2_missing(payload: dict[str, Any]) -> None:
        del payload["latent_operand"]

    for mutate in (v1_with_v2, v2_missing):
        with pytest.raises(ValueError, match="recovery payload fields"):
            _read(template, output, _tampered(template, output, artifact, mutate))


def test_a_v1_record_is_read_old_when_its_keys_are_exactly_the_v1_set(tmp_path: Path) -> None:
    """The H0 evaluation records are v1; the reader keeps admitting exactly that shape."""
    template, output, artifact = _run(tmp_path)

    def to_v1(payload: dict[str, Any]) -> None:
        payload["schema"] = "h-predictive-evaluation-v1"
        del payload["latent_operand"]
        for example in payload["examples"]:
            for key in ("proposals", "diagnostics", "latent_usage"):
                del example[key]

    observed = _read(template, output, _tampered(template, output, artifact, to_v1))
    assert observed["schema"] == "h-predictive-evaluation-v1"
    assert "latent_operand" not in observed


@pytest.mark.parametrize("operand", ["draw_average", "q_draw", 3])
def test_only_the_posterior_mean_operand_is_readable(tmp_path: Path, operand: object) -> None:
    template, output, artifact = _run(tmp_path)

    def swap(payload: dict[str, Any]) -> None:
        payload["latent_operand"] = operand

    with pytest.raises(ValueError, match="latent operand"):
        _read(template, output, _tampered(template, output, artifact, swap))


def _mut(path: tuple[Any, ...], value: object) -> Callable[[dict[str, Any]], None]:
    def apply(payload: dict[str, Any]) -> None:
        node: Any = payload["examples"][0]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


def _bump(path: tuple[Any, ...], delta: float) -> Callable[[dict[str, Any]], None]:
    def apply(payload: dict[str, Any]) -> None:
        node: Any = payload["examples"][0]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] += delta

    return apply


@pytest.mark.parametrize(
    ("label", "mutation", "expect"),
    [
        (
            "stored mixture component",
            # a +1 on the SMALLER component moves a logsumexp by less than float noise, so the
            # tamper sets a value that dominates the draw's density
            _mut(("proposals", "mixture", "q_log_prob", 0, 0), 1000.0),
            "mixture",
        ),
        ("mixture weights", _bump(("proposals", "mixture", "log_weights", 0, 0), 1.0), "mixture"),
        (
            "mixture summary",
            _bump(("proposals", "mixture", "by_sample_count", "2", 0, "log_estimate"), 1.0),
            "summary",
        ),
        ("prior weights", _bump(("proposals", "prior", "log_weights", 0, 0), 1.0), "prior"),
        (
            "prior summary",
            _bump(("proposals", "prior", "by_sample_count", "2", 0, "log_estimate"), 1.0),
            "summary",
        ),
        ("allocation", _mut(("proposals", "mixture", "allocation"), "alternating"), "half"),
        ("component", _mut(("proposals", "mixture", "component", 0, 0), 2), "component"),
        ("proposal name", _mut(("proposals", "prior", "proposal"), "mixture"), "prior block"),
        ("cosine range", _mut(("diagnostics", "prior_q_cosine", 0), 1.5), "cosine"),
        ("cosine type", _mut(("diagnostics", "prior_q_cosine", 0), True), "number"),
        (
            "radius mode",
            _mut(("diagnostics", "prior_log_radius_mean"), None),
            "prior_log_radius_mean",
        ),
        ("sd value", _bump(("diagnostics", "log_weight_sd", "q", "2", 0), 1.0), "log-weight sd"),
        (
            "sd missing",
            _mut(("diagnostics", "log_weight_sd", "prior", "2", 0), None),
            "log-weight sd",
        ),
        ("usage sign", _mut(("latent_usage", "q_decoder_log_prob", 0), 0.5), "not be positive"),
        (
            "usage length",
            _mut(("latent_usage", "null_decoder_log_prob"), [-1.0]),
            "one number per row",
        ),
        ("extra block key", _mut(("proposals", "prior", "extra"), 1), "recovery payload"),
    ],
)
def test_every_v2_addition_is_rederived_and_a_tampered_one_is_refused(
    tmp_path: Path, label: str, mutation: Callable[[dict[str, Any]], None], expect: str
) -> None:
    template, output, artifact = _run(tmp_path)
    assert _read(template, output, artifact)  # the untouched record is admitted
    with pytest.raises(ValueError, match=expect):
        _read(template, output, _tampered(template, output, artifact, mutation))


def test_a_sphere_cell_refuses_a_recorded_log_radius(tmp_path: Path) -> None:
    template, output, artifact = _run(tmp_path, mode="sphere")
    with pytest.raises(ValueError, match="sphere-law cell"):
        _read(
            template,
            output,
            _tampered(
                template, output, artifact, _mut(("diagnostics", "q_log_radius_mean"), [0.0, 0.0])
            ),
        )


def test_the_schema_literals_are_one_closed_set_across_writer_reader_and_capture() -> None:
    assert EVALUATION_SCHEMAS == ("h-predictive-evaluation-v1", "h-predictive-evaluation-v2")
    schemas = trace_capture._SCHEMAS  # pyright: ignore[reportPrivateUsage]
    assert set(EVALUATION_SCHEMAS) <= schemas
    assert len(schemas) == 18
    # the reader's admitted pair and the mirror's literal tuple are one closed set
    assert (evaluation_records._V1, evaluation_records._V2) == EVALUATION_SCHEMAS  # pyright: ignore[reportPrivateUsage]


# ------------------------------------------------------------------------------ telemetry
def test_the_instruments_ride_the_one_evaluation_end_event(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    bind_logs(log)
    (tmp_path / "w").mkdir()
    template, output, artifact = _run(tmp_path / "w")
    rows = read_log(log)
    ends = [r for r in rows if r.get("event") == "evaluation.end"]
    starts = [r for r in rows if r.get("event") == "evaluation.start"]
    assert len(ends) == len(starts) >= 1  # exactly one per evaluation, never per example or k
    record = _read(template, output, artifact)
    metrics = project_evaluation(record)
    assert metrics is not None
    first = ends[0]
    assert first["message"].startswith("evaluation.end global_step=")
    assert "eval/ess_prior_k4" not in first["message"]  # the sentence is unchanged
    assert {"global_step", "seconds"} <= set(first)
    for key in metrics:
        if key in ("eval/wall_seconds", "eval/cuda_seconds"):
            continue
        assert key in first, key
    assert first["eval/latent_usage_posterior_mean_nats_per_byte"] == pytest.approx(
        metrics["eval/latent_usage_posterior_mean_nats_per_byte"]
    )
    assert math.isfinite(first["eval/ess_prior_k4"])
