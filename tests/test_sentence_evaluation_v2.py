"""FOR-1704 (L4d): the ``h-predictive-evaluation-v2`` record, its reader and its telemetry.

These drive the real engine, so they need the reader members' bytes and ride the DG-A patch.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_h_worker_log_lines import bind_logs, read_log
from tests.test_sentence_execution import engine

from entropic_substrate.core.sentence import scoring
from entropic_substrate.core.sentence.codec import teacher_forcing_prefix
from entropic_substrate.core.sentence.law import DirectionalMagnitudeLaw, LawSpec
from entropic_substrate.core.sentence.scoring import (
    LatentUsageOperand,
    derived_stream_keys,
    importance_evaluate,
    importance_statistics,
)
from entropic_substrate.ladder.gate.ladder_campaign import artifact_from_mapping
from entropic_substrate.ladder.run import (
    evaluation_records,
    execution,
    object_worker,
    runner,
    trace_capture,
    worker_execution,
)
from entropic_substrate.ladder.run.evaluation_records import read_evaluation
from entropic_substrate.ladder.run.execution import SentenceExecution, _log_rows
from entropic_substrate.ladder.run.h_metric_extractor import (
    EVALUATION_SCHEMAS,
    project_evaluation,
)
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training

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
    template, output, receipt = _run_with_receipt(root, mode=mode, batches=batches)
    progress = read_json(output, receipt, max_bytes=LIMIT)
    return template, output, artifact_from_mapping(progress["evaluations"][0]["artifact"])


def _run_with_receipt(
    root: Path, *, mode: str = "magnitude", batches: int = 1
) -> tuple[Any, Path, Any]:
    """One finished run; ``(engine template, output root, training receipt)``."""
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
    return template, output, result.receipt


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


class _SpyingPosteriorMean:
    """The registered operand, plus what scoring handed back for each batch (the binding oracle)."""

    operand = POSTERIOR_MEAN.operand

    def __init__(self) -> None:
        self.q: list[list[float]] = []
        self.null: list[list[float]] = []
        self.q_log_radius: list[list[float] | None] = []

    def q_decoder_log_prob(self, score, posterior):  # type: ignore[no-untyped-def]
        q = score(scoring.posterior_mean_vector(posterior))
        self.q.append(q.tolist())
        self.null.append(score(posterior.direction * scoring.NULL_LATENT_RADIUS).tolist())
        radius = posterior.log_radius_mean
        self.q_log_radius.append(None if radius is None else radius.double().tolist())
        return q


def _train_decoder_on_the_first_batch(run: SentenceExecution) -> Any:
    """Teach the decoder to read the q posterior mean of the evaluation's first batch."""
    sampler = run.validation_sampler
    sampler.restore(sampler.snapshot() | {"cursor": 0})
    batch = sampler.next_batch()
    codec = run.model.codec
    prefix = teacher_forcing_prefix(batch.target, codec.spec.max_sentence_bytes)
    optimizer = torch.optim.AdamW(codec.decoder.parameters(), lr=1e-2)
    run.model.eval()
    with torch.enable_grad():
        for _ in range(150):
            latent = scoring.posterior_mean_vector(codec.posterior(batch.target)).detach()
            logits = codec.decoder(latent, prefix)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch.target.reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return batch


def test_the_record_binds_latent_usage_and_the_log_radii_to_their_writers_end_to_end(
    tmp_path: Path,
) -> None:
    """E07/E08: the scoring-level and extractor-fixture tests cannot see the engine copy the two
    latent-usage arrays (or the two log-radius arrays) into the WRONG record fields, because both
    sides of each pair are only ever compared with themselves. Here a trained decoder makes q beat
    the null, scoring's own return values are captured by a spy, and the persisted record must
    carry exactly those values under exactly those names."""
    run = _prediction_run(tmp_path, 1)
    batch = _train_decoder_on_the_first_batch(run)
    spy = _SpyingPosteriorMean()
    record = run.evaluate(spy)  # type: ignore[arg-type]
    example = record["examples"][0]  # type: ignore[index]
    recorded_q = example["latent_usage"]["q_decoder_log_prob"]
    recorded_null = example["latent_usage"]["null_decoder_log_prob"]
    assert recorded_q == spy.q[0] and recorded_null == spy.null[0]
    assert all(q > n for q, n in zip(recorded_q, recorded_null, strict=True)), (
        recorded_q,
        recorded_null,
    )

    metrics = project_evaluation(record)  # type: ignore[arg-type]
    assert metrics is not None
    usage = metrics["eval/latent_usage_posterior_mean_nats_per_byte"]
    expected = sum(q - n for q, n in zip(spy.q[0], spy.null[0], strict=True)) / sum(
        example["source_bytes"]
    )
    assert usage > 0 and usage == pytest.approx(expected, rel=1e-12)

    diagnostics = example["diagnostics"]
    assert spy.q_log_radius[0] is not None  # the default cell is magnitude-law
    assert diagnostics["q_log_radius_mean"] == spy.q_log_radius[0]
    with torch.no_grad():
        prior_radius = run.model.predict(batch.history).law.log_radius_mean
    assert prior_radius is not None
    assert diagnostics["prior_log_radius_mean"] == pytest.approx(
        prior_radius.double().tolist(), abs=1e-9
    )
    assert diagnostics["prior_log_radius_mean"] != diagnostics["q_log_radius_mean"]


def test_evaluate_has_no_default_estimator() -> None:
    run_signature = SentenceExecution.evaluate.__code__
    assert run_signature.co_varnames[: run_signature.co_argcount] == ("self", "latent_usage")
    assert SentenceExecution.evaluate.__defaults__ is None


@pytest.mark.parametrize(
    "seam",
    [
        runner.run_probe_training,
        object_worker.run_object_worker,
        worker_execution.execute_probe_request,
    ],
    ids=lambda seam: seam.__name__,
)
def test_every_seam_takes_the_estimator_as_a_required_keyword(seam: Callable[..., Any]) -> None:
    """A6, DH-52: the estimator is injected at the composition root, so no seam on the way to
    ``SentenceExecution.evaluate`` may hold a default, not even ``None``. MEASURED 2026-11-05 at
    0c1d3a35: a ``= None`` default on any one of the three left 153 tests GREEN (L4d review,
    round 5); with this pin each such mutant turns exactly its own case RED."""
    parameter = inspect.signature(seam).parameters["latent_usage"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


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


def test_evaluate_hands_each_batch_its_own_index_to_the_derived_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S47: the pure key function is tested above, but nothing pinned that ``evaluate`` calls it
    with the batch INDEX. A constant there makes every batch draw its prior and mixture noise on
    the same keys, and every other test still passes."""
    run = _prediction_run(tmp_path, 3)
    seen: list[dict[str, int]] = []

    def spy(**kwargs: int) -> tuple[int, int]:
        seen.append(dict(kwargs))
        return derived_stream_keys(**kwargs)

    monkeypatch.setattr(execution, "derived_stream_keys", spy)
    run.evaluate(POSTERIOR_MEAN)
    q_key = (run.profile.seed * 0x9E3779B1 + run.step) % 2**32
    assert [call["batch"] for call in seen] == [0, 1, 2]
    assert {(call["seed"], call["step"], call["q_key"]) for call in seen} == {
        (run.profile.seed, run.step, q_key)
    }
    assert len({derived_stream_keys(**call) for call in seen}) == 3


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


def _rederived_mixture(profile: Any, *, zero_producer: bool) -> Callable[[dict[str, Any]], None]:
    """A tamper that keeps the mixture block SELF-CONSISTENT: weights, every k-prefix summary and
    every log-weight sd are recomputed from the (possibly forged) recorded terms. With
    ``zero_producer`` the first draw is forged as produced by q while q's density there is zero."""

    def apply(payload: dict[str, Any]) -> None:
        example = payload["examples"][0]
        mixture = example["proposals"]["mixture"]
        if zero_producer:
            mixture["q_log_prob"][0][0] = "-inf"
            mixture["component"][0][0] = 0  # drawn from q, so its own density cannot be zero
            # the density is now prior - log 2, so the weight is decoder + log 2
            mixture["log_weights"][0][0] = mixture["decoder_log_prob"][0][0] + math.log(2)
        weights = torch.tensor(mixture["log_weights"], dtype=torch.float64)
        for k in profile.schedule.evaluated_samples:
            mixture["by_sample_count"][str(k)] = [
                asdict(row) for row in importance_statistics(weights[:k])
            ]
            example["diagnostics"]["log_weight_sd"]["mixture"][str(k)] = [
                float(column.std(unbiased=True)) for column in weights[:k].unbind(dim=1)
            ]

    return apply


def test_the_reader_refuses_a_consistently_forged_zero_density_producer(tmp_path: Path) -> None:
    """S42h: the older test above also trips the weight reconciliation, because its tamper does
    not recompute the weights, so it cannot tell whether the producing-component check works.
    Here every derived field is recomputed, so ONLY the producing check can refuse the record.
    The control re-derives without forging and must be admitted: the helper is faithful."""
    template, output, artifact = _run(tmp_path, mode="sphere")
    control = _rederived_mixture(template.profile, zero_producer=False)
    assert _read(template, output, _tampered(template, output, artifact, control))
    forged = _rederived_mixture(template.profile, zero_producer=True)
    with pytest.raises(ValueError, match="mixture importance terms"):
        _read(template, output, _tampered(template, output, artifact, forged))


def _rederived_positive_decoder(
    profile: Any, block: str, *, forge: bool
) -> Callable[[dict[str, Any]], None]:
    """A tamper that forges ONE positive decoder log-probability (impossible for a probability)
    in the ``prior`` or ``mixture`` block and keeps the block SELF-CONSISTENT: the weight, every
    k-prefix summary and every log-weight sd are recomputed from the forged term, so the weight
    reconciliation passes and ONLY the ``decoder > 0`` check can refuse the record."""

    def apply(payload: dict[str, Any]) -> None:
        example = payload["examples"][0]
        proposal = example["proposals"][block]
        if forge:
            proposal["decoder_log_prob"][0][0] = 0.5
            if block == "prior":
                proposal["log_weights"][0][0] = 0.5  # decoder + prior - prior
            else:
                both = torch.tensor(
                    [proposal["q_log_prob"][0][0], proposal["prior_log_prob"][0][0]],
                    dtype=torch.float64,
                )
                density = float(torch.logsumexp(both, 0)) - math.log(2)
                proposal["log_weights"][0][0] = 0.5 + proposal["prior_log_prob"][0][0] - density
        weights = torch.tensor(proposal["log_weights"], dtype=torch.float64)
        for k in profile.schedule.evaluated_samples:
            proposal["by_sample_count"][str(k)] = [
                asdict(row) for row in importance_statistics(weights[:k])
            ]
            example["diagnostics"]["log_weight_sd"][block][str(k)] = [
                float(column.std(unbiased=True)) for column in weights[:k].unbind(dim=1)
            ]

    return apply


@pytest.mark.parametrize("block", ["prior", "mixture"])
def test_the_reader_refuses_a_consistently_forged_positive_decoder_log_prob(
    tmp_path: Path, block: str
) -> None:
    """The control re-derives without forging and must be admitted: the helper is faithful."""
    template, output, artifact = _run(tmp_path, mode="sphere")
    control = _rederived_positive_decoder(template.profile, block, forge=False)
    assert _read(template, output, _tampered(template, output, artifact, control))
    forged = _rederived_positive_decoder(template.profile, block, forge=True)
    with pytest.raises(ValueError, match=f"{block} importance terms"):
        _read(template, output, _tampered(template, output, artifact, forged))


def _weights(rows: list[list[Any]]) -> torch.Tensor:
    """A recorded weight matrix as the reader parses it: the string ``"-inf"`` is a zero density."""
    return torch.tensor(
        [[-math.inf if v == "-inf" else v for v in row] for row in rows], dtype=torch.float64
    )


def _resummarize(example: dict[str, Any], terms: dict[str, Any], name: str, profile: Any) -> None:
    """Recompute one population's k-prefix summaries and log-weight sds from its recorded weights;
    a column holding a zero-density weight gets the honest ``None`` sd."""
    weights = _weights(terms["log_weights"])
    for k in profile.schedule.evaluated_samples:
        terms["by_sample_count"][str(k)] = [
            asdict(row) for row in importance_statistics(weights[:k])
        ]
        example["diagnostics"]["log_weight_sd"][name][str(k)] = [
            float(column.std(unbiased=True)) if bool(torch.isfinite(column).all()) else None
            for column in weights[:k].unbind(dim=1)
        ]


def _rederived_extreme_mixture(profile: Any, *, forge: bool) -> Callable[[dict[str, Any]], None]:
    """A SELF-CONSISTENT mixture block whose first draw has both component densities far below
    float64's exp floor (about -745): ``exp`` of either is exactly 0, so only a logsumexp keeps
    the mixture density finite. The weight, every k-prefix summary and every sd are recomputed."""

    def apply(payload: dict[str, Any]) -> None:
        example = payload["examples"][0]
        mixture = example["proposals"]["mixture"]
        if forge:
            mixture["q_log_prob"][0][0] = -1000.0
            mixture["prior_log_prob"][0][0] = -2500.0
            both = torch.tensor([-1000.0, -2500.0], dtype=torch.float64)
            density = float(torch.logsumexp(both, 0)) - math.log(2)
            mixture["log_weights"][0][0] = mixture["decoder_log_prob"][0][0] - 2500.0 - density
        _resummarize(example, mixture, "mixture", profile)

    return apply


def test_the_reader_admits_a_mixture_draw_with_both_components_below_the_exp_floor(
    tmp_path: Path,
) -> None:
    """The reader re-derives the mixture density itself; it must be a logsumexp there too, not
    only in the writer. The control re-derives without forging: the helper is faithful.
    MEASURED 2026-11-05 at 0c1d3a35: replacing the reader's logsumexp with a log of a sum left
    351 tests GREEN (L4d review, round 5); with this pin it is RED, ``evaluation mixture
    importance terms do not reconcile``."""
    assert float(torch.tensor([-1000.0, -2500.0], dtype=torch.float64).exp().sum()) == 0.0
    template, output, artifact = _run(tmp_path, mode="sphere")
    control = _rederived_extreme_mixture(template.profile, forge=False)
    assert _read(template, output, _tampered(template, output, artifact, control))
    extreme = _rederived_extreme_mixture(template.profile, forge=True)
    tampered = _tampered(template, output, artifact, extreme)
    mixture = read_json(output, tampered, max_bytes=LIMIT)["examples"][0]["proposals"]["mixture"]
    assert (mixture["q_log_prob"][0][0], mixture["prior_log_prob"][0][0]) == (-1000.0, -2500.0)
    assert _read(template, output, tampered)["schema"] == "h-predictive-evaluation-v2"


def _rederived_zero_density_column(
    profile: Any, block: str, *, sd: float | None
) -> Callable[[dict[str, Any]], None]:
    """A SELF-CONSISTENT record whose first ``block`` weight is a genuine zero density (-inf):
    the prior density there is zero while the producing density is finite, so the weight is
    ``"-inf"`` and its column has no sample sd. Every k-prefix summary and sd is recomputed, and
    the first column's sd is then set to ``sd`` (``None`` is the honest value)."""

    def apply(payload: dict[str, Any]) -> None:
        example = payload["examples"][0]
        terms = example if block == "q" else example["proposals"][block]
        terms["prior_log_prob"][0][0] = "-inf"
        if block == "mixture":
            terms["component"][0][0] = 0  # drawn from q, whose density there is finite
        terms["log_weights"][0][0] = "-inf"
        _resummarize(example, terms, block, profile)
        for k in profile.schedule.evaluated_samples:
            example["diagnostics"]["log_weight_sd"][block][str(k)][0] = sd

    return apply


@pytest.mark.parametrize("block", ["q", "mixture"])
def test_a_log_weight_sd_on_a_zero_density_row_is_refused(tmp_path: Path, block: str) -> None:
    """A numeric sd on a column whose weights include -inf is a fabricated diagnostic, and the
    projection would average it into ``eval/log_weight_sd_*``. The honest ``None`` is admitted.
    MEASURED 2026-11-05 at 0c1d3a35: with the zero-density refusal replaced by ``pass`` the
    forged sd was admitted (DID NOT RAISE) for both q and the mixture."""
    template, output, artifact = _run(tmp_path, mode="sphere")
    honest = _rederived_zero_density_column(template.profile, block, sd=None)
    assert _read(template, output, _tampered(template, output, artifact, honest))
    forged = _rederived_zero_density_column(template.profile, block, sd=0.5)
    with pytest.raises(ValueError, match="zero-density row"):
        _read(template, output, _tampered(template, output, artifact, forged))


def test_a_gap_label_on_a_finite_prior_row_is_refused(tmp_path: Path) -> None:
    """Relabelling a finite row as ``zero_importance_weight_mass`` would drop it out of the
    projection's means and ratios. Only the gap field changes, so only the gap check can fire.
    MEASURED 2026-11-05 at 0c1d3a35: the guard was GREEN on 445 tests when removed (L4d review,
    round 5); with the refusal replaced by ``pass`` this relabel is admitted (DID NOT RAISE)."""
    template, output, artifact = _run(tmp_path)
    assert _read(template, output, artifact)  # the untouched record is admitted
    relabel = _mut(
        ("proposals", "prior", "by_sample_count", "2", 0, "gap"), "zero_importance_weight_mass"
    )
    with pytest.raises(ValueError, match="gap differs from measured weight mass"):
        _read(template, output, _tampered(template, output, artifact, relabel))


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "h-predictive-evaluation-v0"),
        ("meaning", "not_the_importance_weighted_log_marginal_estimate"),
        ("run_identity", "0" * 64),
    ],
)
def test_a_record_with_a_foreign_schema_meaning_or_run_identity_is_refused(
    tmp_path: Path, field: str, value: str
) -> None:
    """The admission chain ``evaluation construction or scoring meaning mismatch`` had no test
    (MEASURED 2026-11-05: ``grep -rn "meaning mismatch" tests`` was empty at the tip and at
    main; replacing the schema term with ``False`` left 43 tests GREEN). The record is the exact
    v1 shape so ``recovery_record`` admits it and only the admission chain can refuse it."""
    template, output, artifact = _run(tmp_path)

    def foreign(payload: dict[str, Any]) -> None:
        payload["schema"] = "h-predictive-evaluation-v1"
        del payload["latent_operand"]
        for example in payload["examples"]:
            for key in ("proposals", "diagnostics", "latent_usage"):
                del example[key]
        payload[field] = value

    with pytest.raises(ValueError, match="meaning mismatch"):
        _read(template, output, _tampered(template, output, artifact, foreign))


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


def test_an_admitted_report_that_does_not_project_fails_loud_not_with_empty_instruments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2 (L3 review): the runner's fail-LOUD guard is the only thing between a defect and a quiet
    omission. Without the raise, ``instruments = {}`` leaves 51 tests GREEN (a fail-open
    measurement path), so the guard is pinned by forcing the projection to refuse."""
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    monkeypatch.setattr(runner, "project_evaluation", lambda report: None)
    with pytest.raises(ValueError, match="did not project"):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
            latent_usage=POSTERIOR_MEAN,
        )


def test_a_resume_admits_a_receipt_whose_evaluation_row_was_written_as_v1(tmp_path: Path) -> None:
    """R3b (L3 review): a run paused by pre-L4d code carries v1 evaluation rows, and the resume
    gate must keep admitting them, as ``read_evaluation`` does by design. Dropping the v1 arm of
    the gate's exact-string check left 258 tests GREEN; the other three exact-string gates and
    ``trace_capture._SCHEMAS`` all go RED on such a drop."""
    template, output, receipt = _run_with_receipt(tmp_path)
    progress = copy.deepcopy(read_json(output, receipt, max_bytes=LIMIT))

    def to_v1(payload: dict[str, Any]) -> None:
        payload["schema"] = "h-predictive-evaluation-v1"
        del payload["latent_operand"]
        for example in payload["examples"]:
            for key in ("proposals", "diagnostics", "latent_usage"):
                del example[key]

    # every roster entry becomes a v1-written row, so the whole roster is read-old
    progress["evaluations"] = [
        {
            "step": entry["step"],
            "artifact": artifact_reference(
                output,
                _tampered(template, output, artifact_from_mapping(entry["artifact"]), to_v1),
            ),
        }
        for entry in progress["evaluations"]
    ]
    written = write_json(output, "v1-roster-progress.json", progress, max_bytes=LIMIT)
    admitted = read_training_progress(
        output,
        written,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    assert admitted["evaluations"] == progress["evaluations"]
