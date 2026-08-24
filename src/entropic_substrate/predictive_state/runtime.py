"""Thin D1 runtime composition over the registered predictive-state domains."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as functional
from torch import nn

from entropic_substrate.data.batching import PredictiveBatch
from entropic_substrate.experiments.d1_family import D1DraftSpec, D1ExperimentFamily
from entropic_substrate.experiments.d1_primitives import (
    D1Arm,
)
from entropic_substrate.observation import CodecIdentity, LanguageReadout, LearnedObservationAdapter
from entropic_substrate.predictive_state.future_law import (
    DiagonalGaussianFutureLaw,
    MeanDisplacementBaseline,
)
from entropic_substrate.predictive_state.initialization import (
    D1InitializationAttestation,
    SeededD1BranchInitializer,
)
from entropic_substrate.predictive_state.objective import PredictiveObjectiveTerms
from entropic_substrate.predictive_state.runtime_components import (
    D1PredictiveResult,
    D1StateBranch,
)
from entropic_substrate.predictive_state.runtime_construction import (
    collapse_regularizer_from_draft,
    encoder_config_from_draft,
    target_policy_from_draft,
)
from entropic_substrate.predictive_state.state import build_encoder_pair
from entropic_substrate.predictive_state.target_updates import (
    OnlineTargetUpdate,
    OptimizationStepCommitted,
    StopGradientEmaUpdate,
)
from entropic_substrate.predictive_state.target_values import (
    FutureTarget,
    StabilizedTargets,
    one_step_displacement,
)

__all__ = ["D1PredictiveResult", "D1PredictiveSystem"]


class D1PredictiveSystem(nn.Module):
    """Complete D1 tracer with strategy-owned stabilization behavior.

    The online branch produces the sole causal state consumed by language and
    future-law losses.  The target branch owns a separate adapter and encoder;
    the injected target policy alone decides whether it has gradients or EMA
    synchronization.  The hot forward path therefore has no arm conditionals.
    """

    _STATE_VERSION = 2

    def __init__(
        self,
        draft: D1DraftSpec,
        *,
        initializer: SeededD1BranchInitializer,
        family: D1ExperimentFamily | None = None,
    ) -> None:
        super().__init__()
        if type(draft) is not D1DraftSpec:
            msg = "D1 predictive system requires a validated D1DraftSpec"
            raise TypeError(msg)
        if family is not None:
            if type(family) is not D1ExperimentFamily:
                msg = "D1 predictive system family must be a D1ExperimentFamily"
                raise TypeError(msg)
            if family.draft_for(draft.arm) != draft:
                msg = "D1 runtime draft is not the selected registered family cell"
                raise ValueError(msg)
        self.draft = draft
        self._family_identity = None if family is None else family.family_digest
        config = encoder_config_from_draft(draft)
        online_adapter = LearnedObservationAdapter(
            vocab_size=draft.codec.cardinality,
            latent_width=draft.encoder_pair.latent_width,
        )
        target_adapter = LearnedObservationAdapter(
            vocab_size=draft.codec.cardinality,
            latent_width=draft.encoder_pair.latent_width,
        )
        target_adapter.load_state_dict(online_adapter.state_dict(), strict=True)
        encoders = build_encoder_pair(config)
        encoders.online.final_norm.eps = draft.encoder_pair.online.rms_epsilon
        encoders.target.final_norm.eps = draft.encoder_pair.target.rms_epsilon
        self.online_branch = D1StateBranch(adapter=online_adapter, encoder=encoders.online)
        self.target_branch = D1StateBranch(adapter=target_adapter, encoder=encoders.target)
        initializer.require_identity(draft.encoder_pair.online.initialization)
        self.initialization_attestation: D1InitializationAttestation = initializer.initialize_pair(
            self.online_branch, self.target_branch
        )
        self._attest_branch_pair()
        self.language_readout = LanguageReadout(
            latent_width=draft.encoder_pair.latent_width,
            vocab_size=draft.codec.cardinality,
        )
        self.future_law = DiagonalGaussianFutureLaw(
            latent_width=draft.encoder_pair.latent_width,
            hidden_width=draft.future_law.predictor_hidden_width,
        )
        self.mean_displacement_baseline = MeanDisplacementBaseline(
            latent_width=draft.encoder_pair.latent_width
        )
        self.target_policy: OnlineTargetUpdate | StopGradientEmaUpdate = target_policy_from_draft(
            draft
        )
        self.target_policy.configure_target_encoder(self.target_branch)
        self.collapse_regularizer = collapse_regularizer_from_draft(draft)
        self._attest_readout_is_untied()

    @classmethod
    def from_family(
        cls,
        family: D1ExperimentFamily,
        arm: D1Arm,
        *,
        initializer: SeededD1BranchInitializer,
    ) -> D1PredictiveSystem:
        """Build one complete runtime from the selected registered family cell."""
        if type(family) is not D1ExperimentFamily:
            msg = "D1 predictive system requires a D1ExperimentFamily"
            raise TypeError(msg)
        if type(arm) is not D1Arm:
            msg = "D1 predictive system arm must be a D1Arm"
            raise TypeError(msg)
        return cls(family.draft_for(arm), family=family, initializer=initializer)

    @property
    def construction_identity(self) -> str:
        """Digest binding this instantiated runtime to its D1 cell construction."""
        return self.draft.construction_digest

    @property
    def family_identity(self) -> str | None:
        """Registered family digest when constructed from a family cell."""
        return self._family_identity

    def optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return exactly the parameters owned by the selected optimizer treatment."""
        return tuple(parameter for parameter in self.parameters() if parameter.requires_grad)

    def validate_optimizer_ownership(self, optimizer: object) -> None:
        """Require the optimizer to own every and only trainable D1 parameter."""
        if not isinstance(optimizer, torch.optim.Optimizer):
            msg = "D1 optimizer ownership validation requires a torch optimizer"
            raise TypeError(msg)
        if isinstance(self.target_policy, StopGradientEmaUpdate):
            self.target_policy.validate_optimizer_excludes_target(optimizer, self.target_branch)
        actual = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if isinstance(parameter, nn.Parameter)
        ]
        actual_ids = {id(parameter) for parameter in actual}
        expected_ids = {id(parameter) for parameter in self.optimizer_parameters()}
        if len(actual) != len(actual_ids) or actual_ids != expected_ids:
            msg = "D1 optimizer must own every and only the runtime's trainable parameters"
            raise ValueError(msg)

    def target_policy_state(self) -> Mapping[str, object]:
        """Return the target-update state that must travel with checkpoints."""
        return self.target_policy.state_dict()

    def after_successful_optimizer_step(self, event: OptimizationStepCommitted) -> None:
        """Advance EMA only after the caller commits a successful optimizer step."""
        if type(event) is not OptimizationStepCommitted:
            msg = "D1 post-step hook requires an OptimizationStepCommitted event"
            raise TypeError(msg)
        self.target_policy.after_optimization_commit(self.online_branch, self.target_branch, event)

    def parameter_attestation(self) -> dict[str, int | str]:
        """Report exact optimizer ownership without omitting control-arm evidence."""
        frozen_names = sorted(
            f"{name}:{tuple(parameter.shape)}"
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if not parameter.requires_grad
        )
        return {
            "trainable_param_count": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "frozen_param_count": sum(
                parameter.numel() for parameter in self.parameters() if not parameter.requires_grad
            ),
            "frozen_mask_digest": hashlib.sha256("\n".join(frozen_names).encode()).hexdigest(),
            "initialization_digest": self.initialization_attestation.identity.construction_digest,
            "initialization_name": self.initialization_attestation.identity.name,
            "initialization_parameter_digest": (
                self.initialization_attestation.online_parameter_digest
            ),
            "initialization_seed": self.initialization_attestation.seed,
        }

    def fit_mean_displacement_baseline(self, batches: Iterable[PredictiveBatch]) -> None:
        """Fit the frozen comparator once from declared pre-optimization batches."""
        target_sum: torch.Tensor | None = None
        target_count = 0
        codec: CodecIdentity | None = None
        with torch.no_grad():
            for batch in batches:
                self._validate_batch(batch)
                target = self._target_for(batch)
                values = target.values.float()
                target_sum = (
                    values.sum(dim=(0, 1))
                    if target_sum is None
                    else target_sum + values.sum(dim=(0, 1))
                )
                target_count += values.shape[0] * values.shape[1]
                codec = target.codec_identity
        if target_sum is None or codec is None or target_count == 0:
            msg = "D1 mean-displacement baseline requires at least one predictive batch"
            raise ValueError(msg)
        self.mean_displacement_baseline.fit(
            FutureTarget(values=(target_sum / target_count).view(1, 1, -1), codec_identity=codec)
        )

    def forward(self, batch: PredictiveBatch) -> D1PredictiveResult:
        """Score one aligned horizon-one batch through the selected D1 cell."""
        self._validate_batch(batch)
        online_states = self.online_branch(batch.online_observations, self.draft.codec)
        language_logits = self.language_readout(online_states.values)
        language_loss = _language_loss(language_logits, batch.language_targets, self.draft.codec)
        target = self._target_for(batch)
        future_law_loss = self.future_law.loss(
            online_states,
            target,
            mean_baseline=self.mean_displacement_baseline,
        )
        collapse = self.collapse_regularizer(online_states)
        objective = PredictiveObjectiveTerms(
            language=language_loss,
            future=future_law_loss.nll,
            variance=collapse.weighted_variance_penalty,
            covariance=collapse.weighted_covariance_penalty,
        )
        return D1PredictiveResult(
            language_logits=language_logits,
            online_states=online_states,
            target=target,
            future_law_loss=future_law_loss,
            collapse=collapse,
            objective=objective,
        )

    def get_extra_state(self) -> dict[str, object]:
        """Persist policy state and construction identities in ``state_dict``."""
        return {
            "version": self._STATE_VERSION,
            "construction_identity": self.construction_identity,
            "family_identity": self.family_identity,
            "initialization": {
                "identity": self.initialization_attestation.identity.manifest,
                "online_parameter_digest": (
                    self.initialization_attestation.online_parameter_digest
                ),
                "seed": self.initialization_attestation.seed,
                "target_parameter_digest": (
                    self.initialization_attestation.target_parameter_digest
                ),
            },
            "target_policy": self.target_policy_state(),
        }

    def set_extra_state(self, state: object) -> None:
        """Restore only the policy state belonging to this exact construction."""
        if not isinstance(state, dict) or state.get("version") != self._STATE_VERSION:
            msg = "D1 runtime extra state is malformed or incompatible"
            raise ValueError(msg)
        if state.get("construction_identity") != self.construction_identity:
            msg = "D1 runtime checkpoint construction identity does not match this system"
            raise ValueError(msg)
        if state.get("family_identity") != self.family_identity:
            msg = "D1 runtime checkpoint family identity does not match this system"
            raise ValueError(msg)
        if state.get("initialization") != self.get_extra_state()["initialization"]:
            msg = "D1 runtime checkpoint initialization identity does not match this system"
            raise ValueError(msg)
        policy_state = state.get("target_policy")
        if isinstance(self.target_policy, StopGradientEmaUpdate):
            self.target_policy.load_state_dict(policy_state)
        elif policy_state != self.target_policy.state_dict():
            msg = "D1 runtime checkpoint target policy does not match this system"
            raise ValueError(msg)

    def _target_for(self, batch: PredictiveBatch) -> StabilizedTargets:
        return self.target_policy.run_target_branch(
            lambda: one_step_displacement(
                self.target_branch(batch.target_encoder_context, self.draft.codec)
            )
        )

    def _validate_batch(self, batch: PredictiveBatch) -> None:
        if type(batch) is not PredictiveBatch:
            msg = "D1 predictive system requires a PredictiveBatch"
            raise TypeError(msg)
        if batch.horizon != 1:
            msg = f"D1 predictive system requires horizon=1, got {batch.horizon}"
            raise ValueError(msg)
        if batch.online_observations.device != next(self.parameters()).device:
            msg = "D1 predictive batch and runtime parameters must share one device"
            raise ValueError(msg)
        if batch.online_observations.shape[1] > self.draft.encoder_pair.online.block_size:
            msg = "D1 predictive batch sequence length exceeds the registered encoder block size"
            raise ValueError(msg)
        for start, alignment in zip(batch.source_starts.tolist(), batch.alignments, strict=True):
            expected = self._expected_alignment(start, batch.online_observations.shape[1])
            if alignment != expected:
                msg = "D1 predictive batch alignment is not the registered horizon-one layout"
                raise ValueError(msg)

    @staticmethod
    def _expected_alignment(start: int, sequence_length: int):
        from entropic_substrate.observation import ObservationAlignment

        return ObservationAlignment.predictive(
            start=start, sequence_length=sequence_length, horizon=1
        )

    def _attest_branch_pair(self) -> None:
        online = dict(self.online_branch.named_parameters())
        target = dict(self.target_branch.named_parameters())
        if online.keys() != target.keys():
            msg = "D1 online and target branches must have matching parameter names"
            raise ValueError(msg)
        for name, online_parameter in online.items():
            target_parameter = target[name]
            if (
                online_parameter.shape != target_parameter.shape
                or online_parameter.dtype != target_parameter.dtype
            ):
                msg = f"D1 online and target branch parameter mismatch for {name!r}"
                raise ValueError(msg)
            if online_parameter.data_ptr() == target_parameter.data_ptr():
                msg = f"D1 online and target branch storage is shared for {name!r}"
                raise ValueError(msg)
            if not torch.equal(online_parameter, target_parameter):
                msg = f"D1 online and target branch initialization differs for {name!r}"
                raise ValueError(msg)

    def _attest_readout_is_untied(self) -> None:
        if (
            self.language_readout.projection.weight.data_ptr()
            == self.online_branch.adapter.embedding.weight.data_ptr()
        ):
            msg = "D1 language readout must remain untied from the online observation adapter"
            raise ValueError(msg)


def _language_loss(
    logits: torch.Tensor, targets: torch.Tensor, codec: CodecIdentity
) -> torch.Tensor:
    if targets.shape != logits.shape[:2]:
        msg = "D1 language targets must align with language logits"
        raise ValueError(msg)
    if int(targets.min()) < 0 or int(targets.max()) >= codec.cardinality:
        msg = f"D1 language targets must be in [0, {codec.cardinality - 1}]"
        raise ValueError(msg)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return functional.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
