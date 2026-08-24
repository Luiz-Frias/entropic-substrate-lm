"""Executable initialization identity for the D1 causal-state branches."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from entropic_substrate.experiments.d1_primitives import (
    D1_ENCODER_INITIALIZATION_DIGEST,
    D1_ENCODER_INITIALIZATION_MANIFEST,
    D1_ENCODER_INITIALIZATION_NAME,
    EncoderInitializationIdentity,
    canonical_digest,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.observation import LearnedObservationAdapter
from entropic_substrate.predictive_state import CausalStateEncoder
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime_components import D1StateBranch
from predictive_state_helpers import model_config

pytestmark = pytest.mark.unit


def _branch() -> D1StateBranch:
    return D1StateBranch(
        adapter=LearnedObservationAdapter(vocab_size=17, latent_width=8),
        encoder=CausalStateEncoder(model_config()),
    )


def _parameter_map(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def test_registered_initialization_identity_is_the_digest_of_the_executable_manifest() -> None:
    identity = default_d1_encoder_initialization_identity()

    assert identity.name == D1_ENCODER_INITIALIZATION_NAME
    assert identity.construction_digest == D1_ENCODER_INITIALIZATION_DIGEST
    assert identity.construction_digest == canonical_digest(D1_ENCODER_INITIALIZATION_MANIFEST)


def test_seeded_initializer_materializes_repeatable_unaliased_branch_pairs() -> None:
    online_a, target_a = _branch(), _branch()
    online_b, target_b = _branch(), _branch()

    attestation_a = SeededD1BranchInitializer(seed=29).initialize_pair(online_a, target_a)
    attestation_b = SeededD1BranchInitializer(seed=29).initialize_pair(online_b, target_b)

    assert attestation_a == attestation_b
    assert attestation_a.seed == 29
    assert attestation_a.identity == default_d1_encoder_initialization_identity()
    assert attestation_a.online_parameter_digest == attestation_a.target_parameter_digest
    for name, online in online_a.named_parameters():
        target = dict(target_a.named_parameters())[name]
        torch.testing.assert_close(online, target, atol=0.0, rtol=0.0)
        assert online.data_ptr() != target.data_ptr()
        torch.testing.assert_close(online, dict(online_b.named_parameters())[name])


def test_run_seed_changes_materialized_parameters_without_changing_algorithm_identity() -> None:
    first_online, first_target = _branch(), _branch()
    second_online, second_target = _branch(), _branch()

    first = SeededD1BranchInitializer(seed=1).initialize_pair(first_online, first_target)
    second = SeededD1BranchInitializer(seed=2).initialize_pair(second_online, second_target)

    assert first.identity == second.identity
    assert first.online_parameter_digest != second.online_parameter_digest
    assert any(
        not torch.equal(value, _parameter_map(second_online)[name])
        for name, value in _parameter_map(first_online).items()
    )


def test_initializer_uses_registered_parameter_rules_not_module_constructor_state() -> None:
    online, target = _branch(), _branch()
    with torch.no_grad():
        for parameter in (*online.parameters(), *target.parameters()):
            parameter.fill_(123.0)

    SeededD1BranchInitializer(seed=3).initialize_pair(online, target)

    layer_norms = [module for module in online.modules() if isinstance(module, nn.LayerNorm)]
    assert layer_norms
    for layer_norm in layer_norms:
        torch.testing.assert_close(layer_norm.weight, torch.ones_like(layer_norm.weight))
        if layer_norm.bias is not None:
            torch.testing.assert_close(layer_norm.bias, torch.zeros_like(layer_norm.bias))
    assert all(not torch.all(parameter == 123.0) for parameter in online.parameters())


def test_runtime_can_reject_a_declared_identity_without_its_executable_initializer() -> None:
    registered = default_d1_encoder_initialization_identity()
    unimplemented = replace(registered, construction_digest="f" * 64)
    initializer = SeededD1BranchInitializer(seed=5)

    with pytest.raises(ValueError, match="does not materialize"):
        initializer.require_identity(unimplemented)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_initializer_refuses_invalid_run_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        SeededD1BranchInitializer(seed=seed)  # type: ignore[arg-type]


def test_initializer_refuses_shared_or_structurally_different_branch_pairs() -> None:
    branch = _branch()
    initializer = SeededD1BranchInitializer(seed=7)
    with pytest.raises(ValueError, match="shared"):
        initializer.initialize_pair(branch, branch)

    wider = D1StateBranch(
        adapter=LearnedObservationAdapter(vocab_size=17, latent_width=16),
        encoder=CausalStateEncoder(model_config(d_model=16, n_heads=2, n_layers=2, block_size=16)),
    )
    with pytest.raises(ValueError, match="parameter"):
        initializer.initialize_pair(_branch(), wider)


def test_identity_value_still_rejects_malformed_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        EncoderInitializationIdentity(name=D1_ENCODER_INITIALIZATION_NAME, construction_digest="x")
