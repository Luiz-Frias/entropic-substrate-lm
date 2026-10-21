"""Numerical sentence-run engine; authorization and artifact filing live outside it."""

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.corpus import CorpusSplit
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
)
from entropic_substrate.core.sentence.scoring import importance_evaluate
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

    def __init__(self, profile: HProfile, source_root: Path, *, max_source_bytes: int) -> None:
        require_execution_precision()
        self.profile = profile
        self.source_root, self.max_source_bytes = source_root, max_source_bytes
        self.corpus = read_prepared_corpus(
            source_root, ContentHash("sha256", profile.source_digest), max_bytes=max_source_bytes
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
            self.corpus,
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

    def evaluate(self) -> dict[str, object]:
        if self.failed:
            raise ValueError("failed execution must recover before evaluation")
        if self.phase != "prediction":
            raise ValueError("predictive evaluation requires the frozen prediction phase")
        training = self.model.training
        devices = [0] if self.device.type == "cuda" else []
        examples: list[dict[str, object]] = []
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
                # (seed << 32) | step is injective over (seed, step) in [0, 2**32)^2 and
                # lies in [0, 2**64); seed + step is not (42 at step 4 == 44 at step 2).
                # Pure in (seed, step), so the same step reproduces and paired arms share.
                torch.manual_seed((self.profile.seed << 32) | self.step)
                for _ in range(self.profile.schedule.evaluation_batches):
                    batch = self.validation_sampler.next_batch()
                    attempt["current_source_batch"] = batch.identity
                    attempt["current_source_examples"] = batch_coordinates(batch)
                    with torch.autocast(
                        self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.profile.precision == "bf16",
                    ):
                        scores = importance_evaluate(
                            self.model,
                            batch.history,
                            batch.target,
                            sample_counts=self.profile.schedule.evaluated_samples,
                            sample_chunk_size=self.profile.schedule.importance_chunk,
                        )

                    def log_rows(values: torch.Tensor) -> list[list[float | str]]:
                        return [
                            ["-inf" if v == -math.inf else float(v) for v in row]
                            for row in values.tolist()
                        ]

                    examples.append(
                        {
                            "source_batch": batch.identity,
                            "source_examples": batch_coordinates(batch),
                            "source_bytes": scores.source_bytes,
                            "boundary_symbols": scores.boundary_symbols,
                            "by_sample_count": {
                                str(k): [asdict(r) for r in rows]
                                for k, rows in scores.by_sample_count.items()
                            },
                            "log_weights": log_rows(scores.log_weights),
                            "decoder_log_prob": log_rows(scores.decoder_log_prob),
                            "prior_log_prob": log_rows(scores.prior_log_prob),
                            "proposal_log_prob": log_rows(scores.proposal_log_prob),
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
            "schema": "h-predictive-evaluation-v1",
            "meaning": "importance_weighted_log_marginal_estimate",
            "run_identity": self.profile.identity,
            "step": self.step,
            "examples": examples,
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
