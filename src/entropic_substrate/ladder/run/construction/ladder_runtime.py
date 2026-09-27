"""Scientific D2-D5 composition over D1 domains; no launch authority is inferred."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from entropic_substrate.core.corpus.batching import PredictiveBatch
from entropic_substrate.core.latent.coarse_graining import CausalCoarseGrainer
from entropic_substrate.core.latent.dynamics import inverse_fisher_residual
from entropic_substrate.core.latent.future_law import (
    DiagonalGaussianFutureLaw,
    FutureLawLoss,
    MeanDisplacementBaseline,
)
from entropic_substrate.core.latent.horizons import (
    AlignedFutureTarget,
    HorizonFutureLaw,
    horizon_displacement,
)
from entropic_substrate.core.latent.initialization import SeededD1BranchInitializer
from entropic_substrate.core.latent.objective import PredictiveObjectiveTerms
from entropic_substrate.core.latent.stabilization import CollapseResult
from entropic_substrate.core.latent.state import LatentTrajectory
from entropic_substrate.core.latent.target_updates import OptimizationStepCommitted
from entropic_substrate.core.latent.target_values import FutureTarget, StabilizedTargets
from entropic_substrate.core.observation.contracts import CodecIdentity, ObservationAlignment
from entropic_substrate.core.observation.permuted_byte_codec import PermutedByteObservationCodec
from entropic_substrate.ladder.register.d1_family import D1DraftSpec
from entropic_substrate.ladder.register.ladder import RungSpec, Treatment
from entropic_substrate.ladder.run.construction.runtime import D1PredictiveSystem


@dataclass(frozen=True, slots=True)
class LadderPredictiveResult:
    """Present objective terms and named absences remain separate evidence."""

    language_logits: torch.Tensor
    online_states: LatentTrajectory
    target: StabilizedTargets | None
    future_law_loss: FutureLawLoss | None
    collapse: CollapseResult
    objective: PredictiveObjectiveTerms
    targets_by_horizon: Mapping[int, StabilizedTargets]
    losses_by_horizon: Mapping[int, FutureLawLoss]
    metrics: Mapping[str, torch.Tensor]
    absence_reasons: Mapping[str, str]
    router_penalty: torch.Tensor
    alignments: tuple[ObservationAlignment, ...]

    @property
    def total_loss(self) -> torch.Tensor:
        return self.objective.total + self.router_penalty


class LadderPredictiveSystem(nn.Module):
    """Fresh paired initialization with explicit rung-specific mechanisms.

    The D1 core supplies matched online/target encoders, stabilization policies,
    untied readout, and collapse regularizer. Additional scientific operators
    compose around those owners without pretending the result is a D1 arm.
    Inputs and returned logits use canonical source-byte labels in D5.
    """

    def __init__(self, spec: RungSpec) -> None:
        super().__init__()
        if type(spec) is not RungSpec:
            raise TypeError("ladder runtime requires a RungSpec")
        self.spec = spec
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(spec.seed)
            self.core = D1PredictiveSystem(
                spec.base, initializer=SeededD1BranchInitializer(seed=spec.seed)
            )
            width = spec.base.encoder_pair.latent_width
            self.laws = HorizonFutureLaw(
                latent_width=width,
                hidden_width=spec.base.future_law.predictor_hidden_width,
                horizons=spec.horizons,
            )
            # The first head is the paired D1 head, not an unused duplicate.
            self.laws.heads[str(spec.horizons[0])] = self.core.future_law
            self.baselines = nn.ModuleDict(
                {
                    str(h): self.core.mean_displacement_baseline
                    if h == spec.horizons[0]
                    else MeanDisplacementBaseline(latent_width=width)
                    for h in spec.horizons
                }
            )
            self.coarse = (
                None
                if spec.coarse is None
                else CausalCoarseGrainer(
                    width=width,
                    mode=spec.coarse.mode,
                    window=spec.coarse.window,
                    smoothing=spec.coarse.smoothing,
                    random_seed=spec.coarse.random_seed,
                )
            )
        self.codec = (
            None if spec.permutation is None else PermutedByteObservationCodec(spec.permutation)
        )
        if spec.treatment is Treatment.L:
            self.core.target_branch.requires_grad_(False)
            self.laws.requires_grad_(False)
        if self.coarse is not None and spec.coarse is not None and spec.coarse.mode != "learned":
            self.coarse.router.requires_grad_(False)

    @property
    def draft(self) -> D1DraftSpec:
        return self.core.draft

    @property
    def construction_identity(self) -> str:
        return self.spec.digest

    @property
    def codec_identity(self) -> CodecIdentity:
        return self.draft.codec if self.codec is None else self.codec.identity

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def optimizer_roles(self) -> Mapping[str, tuple[nn.Parameter, ...]]:
        owners: dict[str, nn.Module] = {
            "online_branch": self.core.online_branch,
            "language_readout": self.core.language_readout,
            "future_law": self.laws,
            "target_branch": self.core.target_branch,
        }
        if self.coarse is not None:
            owners["macro_state"] = self.coarse.macro
            owners["macro_readout"] = self.coarse.readout
            owners["boundary_router"] = self.coarse.router
        return {
            name: tuple(p for p in owner.parameters() if p.requires_grad)
            for name, owner in owners.items()
            if any(p.requires_grad for p in owner.parameters())
        }

    def gradient_absence_reasons(self) -> Mapping[str, str]:
        absent: dict[str, str] = {}
        if self.spec.treatment is Treatment.L:
            absent.update(target_branch="objective_disabled", future_law="objective_disabled")
        elif not any(p.requires_grad for p in self.core.target_branch.parameters()):
            absent["target_branch"] = "stop_gradient_target"
        if self.coarse is not None and self.spec.coarse is not None:
            if self.spec.coarse.mode != "learned":
                absent["boundary_router"] = "fixed_boundary_control"
        return absent

    def validate_optimizer_ownership(self, optimizer: object) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer ownership requires a torch optimizer")
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        expected = {id(p) for p in self.optimizer_parameters()}
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("optimizer must own every and only trainable ladder parameters")

    def target_policy_state(self) -> Mapping[str, object]:
        if self.spec.treatment is Treatment.L:
            return {"kind": "objective_disabled", "version": 1}
        return self.core.target_policy_state()

    def after_successful_optimizer_step(self, event: OptimizationStepCommitted) -> None:
        if type(event) is not OptimizationStepCommitted:
            raise TypeError("post-step requires OptimizationStepCommitted")
        if self.spec.treatment is not Treatment.L:
            self.core.after_successful_optimizer_step(event)

    def parameter_attestation(self) -> dict[str, int | str]:
        inventory = self.core.parameter_attestation()
        inventory.update(
            trainable_param_count=sum(p.numel() for p in self.optimizer_parameters()),
            frozen_param_count=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            frozen_mask_digest=hashlib.sha256(
                "\n".join(
                    f"{name}:{tuple(p.shape)}"
                    for name, p in self.named_parameters()
                    if not p.requires_grad
                ).encode()
            ).hexdigest(),
        )
        return inventory

    def _ids(self, canonical: torch.Tensor) -> torch.Tensor:
        if self.codec is None:
            return canonical
        return canonical.new_tensor(self.codec.mapping)[canonical]

    def _validate_batch(self, batch: PredictiveBatch) -> None:
        if type(batch) is not PredictiveBatch or batch.horizon != max(self.spec.horizons):
            raise ValueError("batch must use the construction's maximum future horizon")
        if batch.online_observations.device != next(self.parameters()).device:
            raise ValueError("batch and model must share device")
        time = batch.online_observations.shape[1]
        if time > self.draft.encoder_pair.online.block_size:
            raise ValueError("online sequence exceeds registered encoder block size")
        for ids in (
            batch.online_observations,
            batch.target_encoder_context,
            batch.language_targets,
        ):
            if bool(((ids < 0) | (ids >= self.draft.codec.cardinality)).any()):
                raise ValueError("batch observation IDs exceed the declared codec")
        for start, actual in zip(batch.source_starts.tolist(), batch.alignments, strict=True):
            if actual != ObservationAlignment.predictive(
                start=start, sequence_length=time, horizon=batch.horizon
            ):
                raise ValueError("batch source alignment does not match construction")

    def _targets(self, batch: PredictiveBatch) -> dict[int, AlignedFutureTarget]:
        time = batch.online_observations.shape[1]
        result = {}
        for horizon in self.spec.horizons:
            alignment = ObservationAlignment.predictive(
                start=int(batch.source_starts[0]), sequence_length=time, horizon=horizon
            )

            # Policy owns the whole encoder execution (no-grad for EMA targets).
            def target_forward(
                h: int = horizon, a: ObservationAlignment = alignment
            ) -> FutureTarget:
                trajectory = self.core.target_branch(
                    self._ids(batch.target_encoder_context[:, : time + h]), self.codec_identity
                )
                return cast(FutureTarget, horizon_displacement(trajectory, a).target)

            target = self.core.target_policy.run_target_branch(target_forward)
            result[horizon] = AlignedFutureTarget(target, alignment)
        return result

    def fit_mean_displacement_baseline(self, batches: Iterable[PredictiveBatch]) -> int:
        """Fit comparators and return the consumed target-input position count."""
        if self.spec.treatment is Treatment.L:
            raise ValueError("language-only treatment has no future baseline to fit")
        sums: dict[int, torch.Tensor] = {}
        count = 0
        target_positions = 0
        with torch.no_grad():
            for batch in batches:
                self._validate_batch(batch)
                for horizon, aligned in self._targets(batch).items():
                    target_positions += batch.target_encoder_context[
                        :, : batch.online_observations.shape[1] + horizon
                    ].numel()
                    total = aligned.target.values.float().sum(dim=(0, 1))
                    sums[horizon] = total if horizon not in sums else sums[horizon] + total
                count += batch.online_observations.numel()
        if count == 0:
            raise ValueError("baseline fitting requires nonempty training batches")
        for horizon, total in sums.items():
            cast(MeanDisplacementBaseline, self.baselines[str(horizon)]).fit(
                FutureTarget((total / count).view(1, 1, -1), self.codec_identity)
            )
        return target_positions

    def encode_observations(self, canonical_ids: torch.Tensor) -> LatentTrajectory:
        """Encode canonical observations via the actual production mechanism.

        Probe labels and future context never enter this path. Operator graphs
        remain local to the call, including during frozen-probe evaluation.
        """
        return self._encode_observations(canonical_ids)[0]

    def _encode_observations(
        self, canonical_ids: torch.Tensor
    ) -> tuple[LatentTrajectory, dict[str, torch.Tensor], torch.Tensor]:
        if canonical_ids.dtype != torch.int64 or canonical_ids.ndim != 2:
            raise ValueError("online observations must be rank-two int64")
        if any(size < 1 for size in canonical_ids.shape):
            raise ValueError("online observations must be nonempty")
        if canonical_ids.device != next(self.parameters()).device:
            raise ValueError("online observations and runtime must share device")
        if canonical_ids.shape[1] > self.draft.encoder_pair.online.block_size:
            raise ValueError("online observations exceed registered block size")
        if bool(((canonical_ids < 0) | (canonical_ids >= self.draft.codec.cardinality)).any()):
            raise ValueError("online observations exceed declared codec")
        metrics: dict[str, torch.Tensor] = {}

        def consume(states: torch.Tensor, proposal: torch.Tensor) -> torch.Tensor:
            predictors = [
                cast(DiagonalGaussianFutureLaw, head).predictor for head in self.laws.heads.values()
            ]
            parameter_names = [
                tuple(name for name, _ in head.named_parameters()) for head in predictors
            ]
            parameters = tuple(p for head in predictors for p in head.parameters())

            def parameterized_mean_map(
                point: torch.Tensor, weights: tuple[torch.Tensor, ...]
            ) -> torch.Tensor:
                normalized = self.core.online_branch.encoder.final_norm(point[None, None])
                # Equal-weight product-law metric agrees with composite NLL scaling.
                predictions = []
                offset = 0
                for predictor, names in zip(predictors, parameter_names, strict=True):
                    mapping = dict(zip(names, weights[offset : offset + len(names)], strict=True))
                    predictions.append(
                        torch.func.functional_call(
                            predictor, mapping, (normalized.float(),)
                        ).reshape(-1)
                    )
                    offset += len(names)
                return torch.cat(predictions) / len(predictions) ** 0.5

            solution = inverse_fisher_residual(
                lambda point: parameterized_mean_map(point, parameters),
                states,
                proposal,
                self.spec.solver,
                parameterized_mean_map=parameterized_mean_map,
                mean_parameters=parameters,
            )
            for name in (
                "iterations",
                "residual_norm",
                "tolerance",
                "metric_action_norm",
                "damping_action_norm",
                "update_norm",
            ):
                metrics[f"fisher_{name}"] = getattr(solution.diagnostics, name)
            return solution.update

        encoder = self.core.online_branch.encoder
        observations = self.core.online_branch.adapter(
            self._ids(canonical_ids), self.codec_identity
        )
        states = encoder(
            observations, final_residual=consume if self.spec.treatment is Treatment.G else None
        )
        router_penalty = states.values.new_zeros(())
        if self.coarse is not None and self.spec.coarse is not None:
            heuristic_ids = (
                canonical_ids if self.spec.coarse.mode in {"whitespace", "punctuation"} else None
            )
            coarse = self.coarse(states.values, byte_ids=heuristic_ids)
            states = LatentTrajectory(coarse.states, states.codec_identity)
            router_penalty = coarse.rate_penalty * self.spec.coarse.rate_weight
            for name in (
                "boundaries",
                "durations",
                "segment_counts",
                "pending_lengths",
                "boundary_rate",
                "boundary_entropy",
            ):
                metrics[f"coarse_{name}"] = getattr(coarse.diagnostics, name)
            metrics["coarse_macro_updates"] = states.values.new_tensor(
                coarse.diagnostics.macro_updates
            )
            metrics["coarse_macro_candidate_evaluations"] = states.values.new_tensor(
                coarse.diagnostics.macro_candidate_evaluations
            )
        return states, metrics, router_penalty

    def forward(self, batch: PredictiveBatch) -> LadderPredictiveResult:
        """Score canonical next bytes and aligned future targets separately."""
        self._validate_batch(batch)
        states, metrics, router_penalty = self._encode_observations(batch.online_observations)
        metrics["online_source_positions"] = states.values.new_tensor(
            batch.online_observations.numel()
        )
        metrics["target_encoder_positions"] = states.values.new_tensor(
            0
            if self.spec.treatment is Treatment.L
            else sum(
                batch.online_observations.shape[0] * (batch.online_observations.shape[1] + h)
                for h in self.spec.horizons
            )
        )
        logits = self.core.language_readout(states.values)
        if self.codec is not None:
            logits = self.codec.canonical_logits(logits, codec_identity=self.codec_identity)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            language_loss = F.cross_entropy(
                logits.float().flatten(0, 1), batch.language_targets.flatten()
            )
        aligned = {} if self.spec.treatment is Treatment.L else self._targets(batch)
        losses: dict[int, FutureLawLoss] = {}
        future = language_loss.new_zeros(())
        if aligned:
            result = self.laws.loss(
                states,
                aligned,
                mean_baselines={
                    h: cast(MeanDisplacementBaseline, self.baselines[str(h)])
                    for h in self.spec.horizons
                },
            )
            losses, future = dict(result.per_horizon), result.nll
        collapse = self.core.collapse_regularizer(states)
        objective = PredictiveObjectiveTerms(
            language_loss,
            future,
            collapse.weighted_variance_penalty,
            collapse.weighted_covariance_penalty,
        )
        targets = {h: cast(StabilizedTargets, value.target) for h, value in aligned.items()}
        first = self.spec.horizons[0]
        return LadderPredictiveResult(
            logits,
            states,
            targets.get(first),
            losses.get(first),
            collapse,
            objective,
            targets,
            losses,
            metrics,
            self.gradient_absence_reasons(),
            router_penalty,
            batch.alignments,
        )

    def get_extra_state(self) -> dict[str, object]:
        return {"version": 1, "construction_identity": self.construction_identity}

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise ValueError("ladder checkpoint construction identity is incompatible")
