"""Numerical sentence-run engine; authorization and artifact filing live outside it."""

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.corpus import CorpusSplit, PreparedCorpus
from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.objective import (
    SentenceLoss,
    codec_warmup_loss,
    prediction_loss,
)
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.rollout import SentenceRollout
from entropic_substrate.core.sentence.runtime_precision import require_execution_precision
from entropic_substrate.core.sentence.sampling import (
    SamplingSpec,
    SentenceBatch,
    SentenceSampler,
    VerifiedCorpus,
)
from entropic_substrate.core.sentence.scoring import (
    MIXTURE_ALLOCATION,
    LatentUsageEstimator,
    ProposalPopulation,
    derived_stream_keys,
    evaluate_proposals,
)
from entropic_substrate.core.sentence.spine import CarrierBlock
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.capture.resource.resource_meter_models import ResourcePhase
from entropic_substrate.ladder.capture.resource.resource_meter_runtime import (
    TorchCudaPeakMemory,
    TorchCudaPhaseTimer,
    build_torch_cuda_adapters,
)
from entropic_substrate.ladder.run.checkpoint import load_checkpoint
from entropic_substrate.ladder.run.mechanism_evidence import collect_mechanism
from entropic_substrate.ladder.run.training import (
    SentencePhase,
    apply_sentence_step,
    build_sentence_optimizer,
)

if TYPE_CHECKING:  # seals imports this module, so the memo type is named for the checker only
    from entropic_substrate.ladder.run.seals import CorpusMemo


def _log_rows(values: torch.Tensor) -> list[list[float | str]]:
    """Finite floats, and the string ``"-inf"`` only where a zero density is admissible.

    A NaN or ``+inf`` is not representable (the journal writes with ``allow_nan=False``) and
    never reaches the record; the reader decides per column whether ``"-inf"`` is admitted.
    """
    return [["-inf" if v == -math.inf else float(v) for v in row] for row in values.tolist()]


def _population_record(population: ProposalPopulation) -> dict[str, object]:
    """One non-q proposal's v2 block: its draws' terms and importance rows, named."""
    record: dict[str, object] = {"proposal": population.proposal}
    if population.component is not None:
        record["allocation"] = MIXTURE_ALLOCATION
        record["component"] = [[int(v) for v in row] for row in population.component.tolist()]
    record.update(
        by_sample_count={
            str(k): [asdict(r) for r in rows] for k, rows in population.by_sample_count.items()
        },
        log_weights=_log_rows(population.log_weights),
        decoder_log_prob=_log_rows(population.decoder_log_prob),
    )
    if population.q_log_prob is not None:
        record["q_log_prob"] = _log_rows(population.q_log_prob)
    record["prior_log_prob"] = _log_rows(population.prior_log_prob)
    return record


def batch_coordinates(batch: SentenceBatch) -> list[dict[str, object]]:
    return [
        {
            "document": e.target.document.identity,
            "target": [e.target.start, e.target.end, e.target.index],
            "history": [[s.start, s.end, s.index] for s in e.history],
            "terminal": e.terminal,
        }
        for e in batch.examples
    ]


class SentenceExecution:
    """Same model and update path on CPU and CUDA, with measured resource boundaries.

    Construction checks source before initializing model RNG. This numerical
    engine does not allocate a pod or mint scientific/spending authorization.
    """

    def __init__(
        self,
        profile: HProfile,
        source_root: Path,
        *,
        max_source_bytes: int,
        corpora: "CorpusMemo | None" = None,
    ) -> None:
        require_execution_precision()
        self.profile = profile
        self.source_root, self.max_source_bytes = source_root, max_source_bytes
        # FOR-1693 (DH-41): the process's memo of this source root, when the caller owns one.
        # With it the corpus is replayed once per process and root, and every sampler this
        # engine builds receives the proof instead of replaying again.
        self.corpora = corpora
        expected = ContentHash("sha256", profile.source_digest)
        self._verified: VerifiedCorpus | None = (
            None
            if corpora is None
            else corpora.read(source_root, expected, max_bytes=max_source_bytes)
        )
        self.corpus: PreparedCorpus = (
            read_prepared_corpus(source_root, expected, max_bytes=max_source_bytes)
            if self._verified is None
            else self._verified.corpus
        )
        if any(
            len(s.source) > profile.codec.max_sentence_bytes
            for r in self.corpus.records
            for s in r.segmentation.spans
        ):
            raise ValueError("source sentence exceeds the model limit")
        self.codec = SentenceByteCodec(profile.permutation)
        self.device = torch.device("cuda:0" if profile.device == "cuda" else "cpu")
        if profile.device == "cuda" and (
            not torch.cuda.is_available() or torch.cuda.device_count() != 1
        ):
            raise ValueError("sentence CUDA execution requires one visible accelerator")
        self.validation_sampler = self._sampler("validation", profile.spine.context_sentences)
        self.sampler = self._sampler("train", 0)
        torch.manual_seed(profile.seed)
        self.model = SentencePredictor(
            profile.codec,
            profile.spine,
            profile.law,
            consumer=profile.consumer,
            solver=profile.solver,
        ).to(self.device)
        self.model.spine.normalize_weights()
        self.phase: SentencePhase = "codec_warmup"
        self.step = 0
        self.optimizer: torch.optim.Optimizer = self._optimizer()
        self.timer: TorchCudaPhaseTimer | None = None
        self.memory: TorchCudaPeakMemory | None = None
        if profile.device == "cuda":
            self.timer, self.memory = build_torch_cuda_adapters(device=0)
            if self.timer is None or self.memory is None:
                raise ValueError("CUDA execution lacks measured resource adapters")
        self.last_attempt: dict[str, object] | None = None
        self.last_evaluation: dict[str, object] | None = None
        self.failed = False
        self.rollout: SentenceRollout | None = None

    def _sampler(self, split: CorpusSplit, history: int) -> SentenceSampler:
        return SentenceSampler(
            self.corpus if self._verified is None else self._verified,
            SamplingSpec(
                split, self.profile.schedule.batch_size, history, self.profile.source_seed
            ),
            self.codec,
            self.device,
        )

    def _optimizer(self) -> torch.optim.AdamW:
        policy = self.profile.optimization
        return build_sentence_optimizer(
            self.model,
            self.phase,
            learning_rate=policy.learning_rate,
            weight_decay=policy.weight_decay,
            betas=policy.betas,
            epsilon=policy.epsilon,
        )

    def begin_prediction(self) -> None:
        if self.failed:
            raise ValueError("failed execution must recover before transition")
        if self.phase != "codec_warmup" or self.step != self.profile.schedule.warmup_steps:
            raise ValueError("prediction requires completed codec warm-up")
        self._prediction_components()

    def _prediction_components(self) -> None:
        self.model.codec.freeze_encoder()
        self.model.zero_grad(set_to_none=True)
        self.phase, self.step = "prediction", 0
        self.optimizer = self._optimizer()
        self.sampler = self._sampler("train", self.profile.spine.context_sentences)

    def resume(
        self,
        root: Path,
        name: str,
        expected: ContentHash,
        *,
        phase: SentencePhase,
        max_bytes: int,
        measurement: ContentHash | None = None,
    ) -> None:
        """Restore only into a fresh engine; rejected attempts cannot continue training."""
        if (
            self.failed
            or self.phase != "codec_warmup"
            or self.step != 0
            or self.sampler.cursor != 0
        ):
            raise ValueError("resume requires a fresh execution engine")
        self.failed = True
        if phase == "prediction":
            self._prediction_components()
        restored = load_checkpoint(
            root,
            name,
            expected,
            self.model,
            self.optimizer,
            self.sampler,
            phase=phase,
            run_identity=self.profile.identity,
            max_bytes=max_bytes,
            expected_measurement=measurement,
        )
        self.model, self.optimizer, self.sampler, self.step = (
            restored.model,
            restored.optimizer,
            restored.sampler,
            restored.step,
        )
        self.rollout = restored.rollout
        self.failed = False

    def advance(self) -> dict[str, object]:
        if self.failed:
            raise ValueError("failed execution must recover before continuing")
        budget = (
            self.profile.schedule.warmup_steps
            if self.phase == "codec_warmup"
            else self.profile.schedule.steps
        )
        if self.step >= budget:
            raise ValueError("execution phase is complete")
        if self.sampler.cursor != self.step:
            raise ValueError("failed attempt must recover before continuing")
        batch = self.sampler.next_batch()
        row: dict[str, object] = {
            "schema": "h-training-step-v2",
            "run_identity": self.profile.identity,
            "phase": self.phase,
            "step": self.step + 1,
            "source_batch": batch.identity,
            "source_examples": batch_coordinates(batch),
            "status": "started",
        }
        self.last_attempt = row
        measured: SentenceLoss | None = None

        def objective() -> torch.Tensor:
            nonlocal measured
            require_execution_precision()
            if (
                self.model.consumer_mode != self.profile.consumer
                or self.model.solver != self.profile.solver
                or self.model.spine.spec != self.profile.spine
                or self.model.codec.spec != self.profile.codec
                or self.model.codec.law_spec != self.profile.law
                or self.model.head.spec != self.profile.law
            ):
                raise ValueError("live construction differs from the declared profile")
            with torch.autocast(
                self.device.type, dtype=torch.bfloat16, enabled=self.profile.precision == "bf16"
            ):
                measured = (
                    codec_warmup_loss(self.model, batch.target)
                    if self.phase == "codec_warmup"
                    else prediction_loss(
                        self.model,
                        batch.history,
                        batch.target,
                        future_weight=self.profile.optimization.future_weight,
                    )
                )
            row["mechanism"] = collect_mechanism(
                measured.readout,
                self.profile,
                phase=self.phase,
                zero_views=[
                    cast(CarrierBlock, block).zero_views for block in self.model.spine.blocks
                ],
            )
            return measured.total

        started = time.perf_counter_ns()
        ticket = None
        try:
            ticket = None if self.timer is None else self.timer.start(ResourcePhase.TRAIN)
            result = apply_sentence_step(
                self.model,
                self.optimizer,
                self.phase,
                objective,
                clip_norm=self.profile.optimization.clip_norm,
            )
            if measured is None:
                raise RuntimeError("training transaction omitted its objective evidence")
            row.update(asdict(result))
            row.update(
                meaning=measured.meaning,
                decoder_ce=float(measured.decoder_ce.detach()),
                future_nll_per_dimension=None
                if measured.future_nll_per_dimension is None
                else float(measured.future_nll_per_dimension.detach()),
                source_bytes=measured.source_bytes,
                boundary_symbols=measured.boundary_symbols,
                history_bytes=measured.history_bytes,
                encoder_bytes=measured.encoder_bytes,
                status="committed",
            )
            self.step += 1
        except Exception as error:
            self.failed = True
            row.update(
                status="scientific_gap"
                if isinstance(error, UndefinedDirection)
                else "execution_defect",
                error_code=getattr(error, "code", type(error).__name__),
                detail=str(error),
            )
            if isinstance(error, UndefinedDirection):
                row["affected_positions"] = error.positions
            raise
        finally:
            try:
                row["cuda_seconds"] = (
                    None
                    if self.timer is None or ticket is None
                    else self.timer.stop(ticket, ResourcePhase.TRAIN)
                )
                row["cuda_peak_bytes"] = None if self.memory is None else self.memory.peak_bytes()
                row["peak_scope"] = "process_allocator_high_water"
            except Exception as error:
                self.failed = True
                row.update(
                    prior_status=row["status"], status="execution_defect", resource_error=str(error)
                )
                raise
            finally:
                row["wall_seconds"] = (time.perf_counter_ns() - started) / 1e9
        return row

    def evaluate(self, latent_usage: LatentUsageEstimator) -> dict[str, object]:
        """Run the frozen predictive evaluation and return the ``h-predictive-evaluation-v2`` body.

        ``latent_usage`` is the injected q-side operand of the latent-usage instrument (DH-52);
        there is deliberately no default, and the record names the operand the implementation
        itself declares. The q rows keep their v1 keys; the prior and mixture proposals, the
        diagnostics and the latent-usage rows are additive keys of the v2 example. The attempt
        record keeps the EXACT v1 example key set: it is a distinct forensic record
        (``h-evaluation-attempt-v1``) and a v2-shaped example inside it would widen it unruled.
        """
        if self.failed:
            raise ValueError("failed execution must recover before evaluation")
        if self.phase != "prediction":
            raise ValueError("predictive evaluation requires the frozen prediction phase")
        training = self.model.training
        devices = [0] if self.device.type == "cuda" else []
        operand = latent_usage.operand
        examples: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        started = time.perf_counter_ns()
        ticket = None
        attempt: dict[str, object] = {
            "schema": "h-evaluation-attempt-v1",
            "status": "started",
            "run_identity": self.profile.identity,
            "step": self.step,
            "examples": examples,
            "current_source_batch": None,
            "current_source_examples": None,
        }
        self.last_evaluation = attempt
        try:
            require_execution_precision()
            self.validation_sampler.restore(self.validation_sampler.snapshot() | {"cursor": 0})
            ticket = None if self.timer is None else self.timer.start(ResourcePhase.EVAL)
            self.model.eval()
            with torch.random.fork_rng(devices=devices):
                # FOR-1664 door B: one noise draw repeated per checkpoint made every
                # checkpoint's k-curve share its sampling error. Key the seed by step:
                # (seed * 0x9E3779B1 + step) % 2**32. The CPU Mersenne Twister consumes
                # only the low 32 bits of torch.manual_seed (measured: seeds 42 and 43
                # under (seed << 32) | step drew identically), so the key is a 32-bit
                # value. The multiplier is odd: at a fixed step this is a bijection on
                # seed in [0, 2**32) (HProfile refuses larger), so paired seeds never
                # share noise; at a fixed seed it is a bijection on step. seed + step is
                # not injective (42 at step 4 == 44 at step 2). Pure in (seed, step).
                q_key = (self.profile.seed * 0x9E3779B1 + self.step) % 2**32
                torch.manual_seed(q_key)
                for index in range(self.profile.schedule.evaluation_batches):
                    batch = self.validation_sampler.next_batch()
                    attempt["current_source_batch"] = batch.identity
                    attempt["current_source_examples"] = batch_coordinates(batch)
                    # FOR-1704: the prior and mixture draws run on streams derived from
                    # (seed, step, batch index, proposal) inside nested fork_rng blocks, so q's
                    # stream above advances across batches exactly as it did before them.
                    prior_stream, mixture_stream = derived_stream_keys(
                        seed=self.profile.seed, step=self.step, batch=index, q_key=q_key
                    )
                    with torch.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.profile.precision == "bf16",
                    ):
                        scores = evaluate_proposals(
                            self.model,
                            batch.history,
                            batch.target,
                            sample_counts=self.profile.schedule.evaluated_samples,
                            sample_chunk_size=self.profile.schedule.importance_chunk,
                            latent_usage=latent_usage,
                            prior_stream=prior_stream,
                            mixture_stream=mixture_stream,
                        )
                    q = scores.q
                    example: dict[str, object] = {
                        "source_batch": batch.identity,
                        "source_examples": batch_coordinates(batch),
                        "source_bytes": q.source_bytes,
                        "boundary_symbols": q.boundary_symbols,
                        "by_sample_count": {
                            str(k): [asdict(r) for r in rows]
                            for k, rows in q.by_sample_count.items()
                        },
                        "log_weights": _log_rows(q.log_weights),
                        "decoder_log_prob": _log_rows(q.decoder_log_prob),
                        "prior_log_prob": _log_rows(q.prior_log_prob),
                        "proposal_log_prob": _log_rows(q.proposal_log_prob),
                    }
                    examples.append(example)
                    records.append(
                        {
                            **example,
                            "proposals": {
                                "prior": _population_record(scores.prior),
                                "mixture": _population_record(scores.mixture),
                            },
                            "diagnostics": {
                                "prior_log_radius_mean": None
                                if scores.prior_log_radius_mean is None
                                else list(scores.prior_log_radius_mean),
                                "q_log_radius_mean": None
                                if scores.q_log_radius_mean is None
                                else list(scores.q_log_radius_mean),
                                "prior_q_cosine": list(scores.prior_q_cosine),
                                "log_weight_sd": {
                                    name: {str(k): list(sd) for k, sd in by_k.items()}
                                    for name, by_k in scores.log_weight_sd.items()
                                },
                            },
                            "latent_usage": {
                                "q_decoder_log_prob": list(scores.q_latent_log_prob),
                                "null_decoder_log_prob": list(scores.null_latent_log_prob),
                            },
                        }
                    )
            attempt["status"] = "complete"
        except Exception as error:
            self.failed = True
            attempt.update(
                status="scientific_gap"
                if isinstance(error, UndefinedDirection)
                else "execution_defect",
                error_type=type(error).__name__,
                error_code=getattr(error, "code", type(error).__name__),
                detail=str(error),
            )
            if isinstance(error, UndefinedDirection):
                attempt["affected_positions"] = error.positions
            raise
        finally:
            self.model.train(training)
            try:
                cuda_seconds = (
                    None
                    if self.timer is None or ticket is None
                    else self.timer.stop(ticket, ResourcePhase.EVAL)
                )
                attempt["cuda_seconds"] = cuda_seconds
            except Exception as error:
                self.failed = True
                attempt.update(
                    prior_status=attempt["status"],
                    status="execution_defect",
                    resource_error=str(error),
                    resource_error_type=type(error).__name__,
                )
                raise
            finally:
                attempt["wall_seconds"] = (time.perf_counter_ns() - started) / 1e9
        return {
            "schema": "h-predictive-evaluation-v2",
            "meaning": "importance_weighted_log_marginal_estimate",
            "run_identity": self.profile.identity,
            "step": self.step,
            "latent_operand": str(operand),
            "examples": records,
            "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
            "cuda_seconds": cuda_seconds,
        }

    def inventory(self) -> dict[str, object]:
        components = {
            "encoder": self.model.codec.encoder,
            "decoder": self.model.codec.decoder,
            "spine": self.model.spine,
            "future_law": self.model.head,
        }
        return {
            "total_deployed": sum(p.numel() for p in self.model.parameters()),
            "optimizer_active": sum(
                p.numel() for g in self.optimizer.param_groups for p in g["params"]
            ),
            "components": {
                name: sum(p.numel() for p in component.parameters())
                for name, component in components.items()
            },
        }
