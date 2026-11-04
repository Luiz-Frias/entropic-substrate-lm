"""FOR-1694 (host CUDA pair-verifier note): the host verifier equals ``admit_codec_pair``.

``admit_codec_pair`` restores both checkpoints on ``profile.device`` and is pod-only for a cuda
pair. ``admit_codec_pair_on_host`` admits the same pair from collected bytes with CPU-mapped
loads. Equality is proven on BOTH arms over L4a's own fixtures: the good arm yields an identical
``VerifiedHandoff``, and every row of ``ADMIT_HAZARDS`` (plus the own-claim forgery and the
independent warm-up) raises the same ``CodecHandoffDefect`` code from both readers. A verifier
that skipped a digest would stay GREEN on the good arm only; the hazard rows are its teeth.
"""

from dataclasses import replace
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

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run import host_pair_verifier
from entropic_substrate.ladder.run.codec_handoff import CodecHandoffDefect, VerifiedHandoff
from entropic_substrate.ladder.run.host_pair_verifier import (
    _guard_construction,  # pyright: ignore[reportPrivateUsage]
    _Stored,  # pyright: ignore[reportPrivateUsage]
    admit_codec_pair_on_host,
)

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


def test_a_pair_of_one_carrier_is_refused_by_the_pair_predicate_before_any_file_is_read(
    world: World, tmp_path: Path
) -> None:
    """The host verifier's first act is ``require_pair``, as ``admit_codec_pair``'s is: a donor
    and a recipient of the same spine carrier name ``pair_predicate`` with the pod verifier's own
    detail. The roots do not exist, so a verifier that skipped the predicate would fail with an
    artifact read error instead of the pair refusal."""
    from entropic_substrate.ladder.run.codec_handoff import admit_codec_pair

    pair = world.pair
    missing = tmp_path / "no-such-root"
    seal = ArtifactIdentity("no-seal.json", ContentHash("sha256", "0" * 64))
    arguments: dict[str, Any] = {
        "max_source_bytes": SOURCE_LIMIT,
        "max_artifact_bytes": LIMIT,
    }
    same = (missing, seal, pair.donor, missing, seal, pair.donor, pair.source)
    host_side = refusal(lambda: admit_codec_pair_on_host(*same, **arguments))
    pod_side = refusal(lambda: admit_codec_pair(*same, **arguments))
    assert host_side.code == pod_side.code == "pair_predicate"
    assert host_side.detail == pod_side.detail == "a pair needs two different spine carriers"


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
    safety: list[object] = []
    real_load = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        loads.append(kwargs.get("map_location"))
        safety.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(host_pair_verifier.torch, "load", spy)
    assert on_host(world, world.donor, root, handoff.seal) == handoff.verified
    # every load (the seal walks' and the verifier's own two payload reads) is CPU-mapped
    assert loads and set(loads) == {"cpu"}
    # ... and every load refuses arbitrary pickled globals: the bytes are pod-written, so a
    # full-pickle load here would be host-side code execution (the verifier's own two reads are
    # in this set; flipping either to ``weights_only=False`` makes it ``{True, False}``)
    assert set(safety) == {True}


def test_the_verifier_refuses_a_checkpoint_whose_bytes_changed_between_the_seal_walk_and_its_read(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``read_seal`` admits each checkpoint's hash through its own read; the verifier then reads
    the same bytes a second time for the three payload keys. A second read that returns other
    bytes (with their own honest hash) is refused against the SEALED hash, never loaded."""
    import hashlib

    pair = world.pair
    root = pair.root("host-swap")
    handoff = materialize(pair, world.donor, root)
    assert on_host(world, world.donor, root, handoff.seal) == handoff.verified  # control
    real_read = host_pair_verifier.read_d1_artifact_bytes

    def swapped(*args: Any, **kwargs: Any) -> Any:
        head, payload, _ = real_read(*args, **kwargs)
        other = payload + b"\0"
        return head, other, hashlib.sha256(other).hexdigest()

    monkeypatch.setattr(host_pair_verifier, "read_d1_artifact_bytes", swapped)
    with pytest.raises(ValueError, match="checkpoint content hash mismatch"):
        on_host(world, world.donor, root, handoff.seal)


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


def test_a_donor_chain_rooted_at_a_hand_off_is_chained_on_the_host(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A donor whose own codec arrived by hand-off is ``chained_handoff``: a donor warms up for
    itself. Reaching that root through the pair's real profiles needs a donor-profile hand-off,
    which no fixture builds (the pod-side chain is pinned by the materialize test that refuses a
    donor which was itself handed off), so the donor walk's root kind is the one thing overridden."""
    pair = world.pair
    root = pair.root("host-chained")
    handoff = materialize(pair, world.donor, root)
    real = host_pair_verifier.read_seal

    def handed_off(
        donor_root: Path, seal: ArtifactIdentity, profile: Any, *a: Any, **k: Any
    ) -> Any:
        sealed = real(donor_root, seal, profile, *a, **k)
        return replace(sealed, root_kind="handoff") if donor_root == world.donor else sealed

    monkeypatch.setattr(host_pair_verifier, "read_seal", handed_off)
    assert (
        refusal(lambda: on_host(world, world.donor, root, handoff.seal)).code == "chained_handoff"
    )


def test_a_hand_off_rooted_donor_with_no_warm_up_final_is_chained_before_it_is_foreign(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod verifier refuses a hand-off-rooted donor chain on the CITED seal, before it asks for
    a warm-up-final, so a chain that is both says ``chained_handoff``. The host mirrors that order
    (its second hand-off check, after the warm-up-final is resolved, would say ``foreign_donor``)."""
    pair = world.pair
    root = pair.root("host-chained-first")
    handoff = materialize(pair, world.donor, root)
    real = host_pair_verifier.read_seal

    def handed_off_without_final(
        donor_root: Path, seal: ArtifactIdentity, profile: Any, *a: Any, **k: Any
    ) -> Any:
        sealed = real(donor_root, seal, profile, *a, **k)
        if donor_root != world.donor:
            return sealed
        return replace(sealed, root_kind="handoff", warmup_final=None)

    monkeypatch.setattr(host_pair_verifier, "read_seal", handed_off_without_final)
    assert (
        refusal(lambda: on_host(world, world.donor, root, handoff.seal)).code == "chained_handoff"
    )


def test_a_donor_chain_without_a_warm_up_final_is_a_foreign_donor_on_the_host(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host reads the donor's chain through ``read_seal``; a chain that holds no warm-up-final
    seal is ``foreign_donor``. Control: the unmodified walk admits the same pair."""
    pair = world.pair
    root = pair.root("host-no-final")
    handoff = materialize(pair, world.donor, root)
    assert on_host(world, world.donor, root, handoff.seal) == handoff.verified  # control
    real = host_pair_verifier.read_seal

    def without_final(
        donor_root: Path, seal: ArtifactIdentity, profile: Any, *a: Any, **k: Any
    ) -> Any:
        sealed = real(donor_root, seal, profile, *a, **k)
        return replace(sealed, warmup_final=None) if donor_root == world.donor else sealed

    monkeypatch.setattr(host_pair_verifier, "read_seal", without_final)
    assert refusal(lambda: on_host(world, world.donor, root, handoff.seal)).code == "foreign_donor"


def stored(
    *, binding: dict[str, Any] | None = None, model: dict[str, object] | None = None
) -> _Stored:
    base: dict[str, Any] = {
        "run_identity": "donor-cell",
        "spine": {"carrier": "fp32", "width": 4},
        "groups": [["w"]],
    }
    return _Stored(
        binding or base,
        model if model is not None else {"w": torch.zeros(2), "codec.s": torch.ones(2)},
        {},
        None,
    )


def test_the_construction_guard_refuses_a_binding_inventory_or_dtype_mismatch() -> None:
    donor = stored()
    recipient = stored(
        binding={
            "run_identity": "recipient-cell",
            "spine": {"carrier": "bf16", "width": 4},
            "groups": [["w"]],
        }
    )
    _guard_construction(donor, recipient)  # control: only the run identity and carrier differ
    width = stored(
        binding={"run_identity": "r", "spine": {"carrier": "fp32", "width": 8}, "groups": [["w"]]}
    )
    with pytest.raises(CodecHandoffDefect) as binding:
        _guard_construction(donor, width)
    assert binding.value.code == "construction_divergence"
    extra = stored(model={"w": torch.zeros(2), "codec.s": torch.ones(2), "extra": torch.zeros(1)})
    with pytest.raises(CodecHandoffDefect, match="inventories differ"):
        _guard_construction(donor, extra)
    dtype = stored(model={"w": torch.zeros(2, dtype=torch.float64), "codec.s": torch.ones(2)})
    with pytest.raises(CodecHandoffDefect, match="non-codec state w"):
        _guard_construction(donor, dtype)
    codec_only = stored(model={"w": torch.zeros(2), "codec.s": torch.zeros(2)})
    _guard_construction(donor, codec_only)  # codec state is copied, never guarded
