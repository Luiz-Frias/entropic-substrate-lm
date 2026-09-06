"""Small local probe composition; scientific runs retain separate admission."""

from __future__ import annotations

from dataclasses import replace

from entropic_substrate.experiments.d1_execution import D1AmpPolicy, default_d1_execution_protocol
from entropic_substrate.experiments.d1_family import build_d1_experiment_family
from entropic_substrate.experiments.d1_primitives import (
    CausalEncoderSpec,
    D1Arm,
    MatchedEncoderPairSpec,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.experiments.ladder import CoarseSpec, Rung, RungSpec, Treatment
from entropic_substrate.observation import ByteObservationCodec


def tiny_rung_spec(experiment: str, *, seed: int) -> RungSpec:
    """Select explicit synthetic smoke mechanisms, with no predecessor authority.

    Later profiles carry S / horizon-1 / fixed-window constructions solely for
    exercising code. A successful probe cannot select a scientific winner.
    """
    encoder = CausalEncoderSpec(
        latent_width=8,
        n_layers=1,
        n_heads=2,
        mlp_ratio=2.0,
        block_size=128,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=default_d1_encoder_initialization_identity(),
    )
    protocol = default_d1_execution_protocol(sequence_length=8)
    protocol = replace(
        protocol,
        batching=replace(protocol.batching, batch_size=2),
        optimization=replace(protocol.optimization, amp=D1AmpPolicy.OFF),
    )
    base = build_d1_experiment_family(
        codec=ByteObservationCodec().codec_identity,
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
        execution_protocol=protocol,
    ).draft_for(D1Arm.BOTH)
    treatments = {"d2-l": Treatment.L, "d2-s": Treatment.S, "d2-g": Treatment.G}
    d2 = RungSpec(base, Rung.D2, treatments.get(experiment, Treatment.S), seed)
    if experiment in treatments:
        return d2
    horizons = {
        "d3-h1": (1,),
        "d3-h4": (4,),
        "d3-h16": (16,),
        "d3-h64": (64,),
        "d3-composite": (1, 4, 16, 64),
    }
    d3 = RungSpec(
        base, Rung.D3, Treatment.S, seed, horizons=horizons.get(experiment, (1,)), predecessor=d2
    )
    if experiment in horizons:
        return d3
    modes = {
        "identity",
        "fixed",
        "learned",
        "always",
        "never",
        "random",
        "fixed-duration",
        "whitespace",
        "punctuation",
    }
    mode = experiment.removeprefix("d4-")
    d4 = RungSpec(
        base,
        Rung.D4,
        Treatment.S,
        seed,
        predecessor=d3,
        coarse=CoarseSpec(mode.replace("-", "_") if mode in modes else "fixed"),
    )
    if experiment.startswith("d4-") and mode in modes:
        return d4
    if experiment in {"d5-identity", "d5-permuted"}:
        mapping = tuple((value * 73 + 19) % 256 for value in range(256))
        return RungSpec(
            base,
            Rung.D5,
            Treatment.S,
            seed,
            predecessor=d4,
            coarse=d4.coarse,
            permutation=mapping if experiment == "d5-permuted" else None,
        )
    raise ValueError(f"unknown tiny D-ladder experiment: {experiment}")
