"""FOR-1694 (host CUDA pair-verifier note): the host verifier equals ``admit_codec_pair``.

``admit_codec_pair`` restores both checkpoints on ``profile.device`` and is pod-only for a cuda
pair. ``admit_codec_pair_on_host`` admits the same pair from collected bytes with CPU-mapped
loads. Equality is proven on BOTH arms over L4a's own fixtures: the good arm yields an identical
``VerifiedHandoff``, and every row of ``ADMIT_HAZARDS`` (plus the own-claim forgery and the
independent warm-up) raises the same ``CodecHandoffDefect`` code from both readers. A verifier
that skipped a digest would stay GREEN on the good arm only; the hazard rows are its teeth.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from tests.test_h_codec_handoff import (
    ADMIT_HAZARDS,
    LIMIT,
    SOURCE_LIMIT,
    World,
    admit,
    finish,
    forged_root,
    listing,
    make_pair,
    materialize,
    nudge_codec,
    train,
    warmup_final,
)

from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run import host_pair_verifier
from entropic_substrate.ladder.run.codec_handoff import CodecHandoffDefect, VerifiedHandoff
from entropic_substrate.ladder.run.host_pair_verifier import admit_codec_pair_on_host

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    """L4a's shared donor runs, rebuilt here (its fixture lives in another module's namespace)."""
    pair = make_pair(tmp_path_factory.mktemp("pair"))
    donor, other, independent = pair.root("donor"), pair.root("other"), pair.root("independent")
    train(pair, pair.donor, donor)
    train(pair, pair.donor, other)
    control = train(pair, pair.recipient, independent)
    return World(pair, donor, other, independent, control.last_seal, listing(donor))


def on_host(
    world: World, donor_root: Path, recipient_root: Path, recipient_seal: ArtifactIdentity
) -> VerifiedHandoff:
    pair = world.pair
    return admit_codec_pair_on_host(
        donor_root,
        warmup_final(donor_root),
        pair.donor,
        recipient_root,
        recipient_seal,
        pair.recipient,
        pair.source,
        max_source_bytes=SOURCE_LIMIT,
        max_artifact_bytes=LIMIT,
    )


def test_the_host_verifier_yields_the_same_verified_handoff_on_identical_checkpoint_bytes(
    world: World,
) -> None:
    pair = world.pair
    root = pair.root("host-good")
    handoff = materialize(pair, world.donor, root)
    done = finish(pair, root, handoff)
    for seal in (handoff.seal, done.last_seal):  # the hand-off root and a later prediction seal
        pod_side = admit(pair, world.donor, root, seal)
        host_side = on_host(world, world.donor, root, seal)
        assert host_side == pod_side == handoff.verified
        assert host_side.recipient_identity == pair.recipient.identity
        assert host_side.donor_identity == pair.donor.identity and host_side.warmup_steps == 2


def refusal(call: Any) -> CodecHandoffDefect:
    with pytest.raises(CodecHandoffDefect) as raised:
        call()
    return raised.value


@pytest.mark.parametrize("kind", sorted(ADMIT_HAZARDS))
def test_every_hazard_row_is_refused_by_the_host_verifier_with_the_pod_verifiers_code(
    world: World, kind: str
) -> None:
    pair = world.pair
    overrides, nudge, code = ADMIT_HAZARDS[kind]
    if overrides == "manifest":
        overrides = {"donor_manifest": {**pair.donor.manifest, "name": "someone-else"}}
    assert overrides is None or isinstance(overrides, dict)
    root, seal = forged_root(world, f"host-{kind}", record=overrides, nudge=nudge)
    pod_side = refusal(lambda: admit(pair, world.donor, root, seal))
    host_side = refusal(lambda: on_host(world, world.donor, root, seal))
    assert host_side.code == pod_side.code == code, kind
    assert host_side.detail == pod_side.detail, kind


def test_the_central_forgery_is_named_by_the_donor_side_on_the_host_too(world: World) -> None:
    """A recipient that warmed differently and signs its OWN digests equals its claim; only the
    donor's payload can say no, and the host verifier reads the donor's stored payload."""
    root, seal = forged_root(world, "host-own-claim", nudge=nudge_codec, claim_own=True)
    pod_side = refusal(lambda: admit(world.pair, world.donor, root, seal))
    host_side = refusal(lambda: on_host(world, world.donor, root, seal))
    assert host_side.code == pod_side.code == "payload_mismatch"
    assert "donor" in host_side.detail and host_side.detail == pod_side.detail


def test_an_independent_recipient_and_a_foreign_donor_are_named_the_same_way(
    world: World,
) -> None:
    pair = world.pair
    independent = refusal(
        lambda: on_host(world, world.donor, world.independent, world.independent_seal)
    )
    assert independent.code == "independent_warmup"
    root = pair.root("host-foreign")
    handoff = materialize(pair, world.donor, root)
    foreign = refusal(lambda: on_host(world, world.other_donor, root, handoff.seal))
    pod_side = refusal(lambda: admit(pair, world.other_donor, root, handoff.seal))
    assert foreign.code == pod_side.code == "foreign_donor"


def test_the_host_verifier_never_builds_an_engine_and_never_touches_a_cuda_device(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It must run on a host with no CUDA: no ``SentenceExecution`` is constructed, and the
    checkpoints are loaded CPU-mapped."""
    from entropic_substrate.ladder.run import execution

    pair = world.pair
    root = pair.root("host-noengine")
    handoff = materialize(pair, world.donor, root)

    def forbidden(*_: Any, **__: Any) -> None:
        raise AssertionError("the host verifier must not construct an execution engine")

    monkeypatch.setattr(execution.SentenceExecution, "__init__", forbidden)
    loads: list[object] = []
    real_load = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        loads.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(host_pair_verifier.torch, "load", spy)
    assert on_host(world, world.donor, root, handoff.seal) == handoff.verified
    # every load (the seal walks' and the verifier's own two payload reads) is CPU-mapped
    assert loads and set(loads) == {"cpu"}


def test_a_pods_cuda_rng_states_are_part_of_the_digest_the_verifier_compares(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CPU fixtures carry an empty CUDA tuple, so a verifier that digested only the CPU state
    would pass them. A synthetic non-empty tuple in the stored payload changes the digest, and a
    donor whose stored RNG differs from the record's is refused, never ignored."""
    pair = world.pair
    root = pair.root("host-cuda-rng")
    handoff = materialize(pair, world.donor, root)
    real_load = torch.load

    def with_cuda_state(*args: Any, **kwargs: Any) -> Any:
        saved = real_load(*args, **kwargs)
        saved["rng"] = {**saved["rng"], "cuda": (torch.arange(16, dtype=torch.uint8),)}
        return saved

    monkeypatch.setattr(host_pair_verifier.torch, "load", with_cuda_state)
    failed = refusal(lambda: on_host(world, world.donor, root, handoff.seal))
    assert failed.code == "payload_mismatch" and "donor" in failed.detail
