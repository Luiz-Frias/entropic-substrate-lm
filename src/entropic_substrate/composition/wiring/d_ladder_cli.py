"""Small local probe composition; scientific runs retain separate admission."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from entropic_substrate.core.observation.byte_codec import ByteObservationCodec
from entropic_substrate.kernel.identity.d1_primitives import (
    CausalEncoderSpec,
    D1Arm,
    MatchedEncoderPairSpec,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.ladder.register.d1_execution import (
    D1AmpPolicy,
    default_d1_execution_protocol,
)
from entropic_substrate.ladder.register.d1_family import build_d1_experiment_family
from entropic_substrate.ladder.register.ladder import CoarseSpec, Rung, RungSpec, Treatment


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


def add_d_ladder_parser(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> None:
    """Attach the bounded local tracer command to the repository CLI."""
    parser = sub.add_parser("d-probe", help="execute a local synthetic D-ladder tiny probe")
    parser.add_argument("settings", nargs="*", help="optional explicit key=value tiny settings")
    parser.add_argument("--exp")
    parser.add_argument("--tiny", choices=("true", "false"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--manifest", type=Path, help="D1's existing validated smoke manifest")


def _resolve_probe_options(args: argparse.Namespace) -> None:
    fields = {
        "exp": "exp",
        "tiny": "tiny",
        "seed": "seed",
        "steps": "steps",
        "out": "out_dir",
        "manifest": "manifest",
        "resume": "resume",
    }
    for item in args.settings:
        key, separator, value = item.partition("=")
        if separator != "=" or key not in fields or not value:
            raise ValueError("probe settings require a known nonempty key=value")
        field = fields[key]
        if getattr(args, field) is not None:
            raise ValueError(f"duplicate or conflicting probe setting: {key}")
        parsed: str | int | Path = value
        if key in {"seed", "steps"}:
            parsed = int(value)
        elif key in {"out", "manifest", "resume"}:
            parsed = Path(value)
        setattr(args, field, parsed)
    # These are excluded-smoke conveniences, never scientific registration defaults.
    args.exp = args.exp if args.exp is not None else "d2-g"
    args.tiny = args.tiny if args.tiny is not None else "true"
    args.out_dir = args.out_dir if args.out_dir is not None else Path("results/d-probe")


def cmd_d_probe(args: argparse.Namespace) -> int:
    """Run a real tiny model and emit only a smoke receipt."""
    from entropic_substrate.historical.d_ladder.d_ladder_execution import (
        SmokeRequest,
        run_local_smoke,
    )

    _resolve_probe_options(args)
    if args.tiny != "true":
        raise ValueError(
            "d-probe admits local tiny runs only; scientific runs require registration"
        )
    if args.exp.startswith("d1-"):
        from entropic_substrate.composition.wiring.d1_probe_cli import run_d1_probe

        return run_d1_probe(args)
    if args.manifest is not None:
        raise ValueError("--manifest on d-probe is reserved for the existing D1 trainer")
    request = SmokeRequest(
        tiny_rung_spec(args.exp, seed=42 if args.seed is None else args.seed),
        max_steps=2 if args.steps is None else args.steps,
    )
    result = run_local_smoke(request, args.out_dir, resume=args.resume)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0
