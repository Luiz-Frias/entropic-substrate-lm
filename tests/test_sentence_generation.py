"""Generation continues the trained lineage and preserves pending byte sampling."""

import copy
import json
from collections.abc import Iterator
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_sentence_execution import engine
from tests.test_sentence_rollout import scripted_tokens

import entropic_substrate.ladder.run.generation as generation_module
from entropic_substrate.core.sentence.recovery_state import capture_rng, rollout_from_payload
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.generation import read_generation, run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference, read_json
from entropic_substrate.ladder.run.runner import run_probe_training

pytestmark = pytest.mark.unit
LIMIT = 10_000_000


def trained(root: Path) -> tuple[SentenceExecution, Path, ArtifactIdentity]:
    template = engine(root)
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


def test_generation_is_bound_to_training_and_retains_shadow_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 46, 256, 66, 46, 257])
    result = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    assert result.status == "generation_complete"
    report = read_generation(
        output,
        result.receipt,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    # FOR-1674: the profile fixture draws N=2 continuations and the script is cyclic, so
    # each predicate is re-addressed to continuations[0] (the original single rollout),
    # and the documented N=2 aggregate is asserted beside it.
    first = report["continuations"][0]
    assert first["outcome"] == "document_end"
    assert first["sentences"] == ["A.", "B."]
    assert report["summary"]["outcome"] == "document_end"  # the last continuation's outcome
    assert report["summary"]["observations"] == [65, 46, 256, 66, 46, 257] * 2
    assert report["summary"]["sentences"] == ["A.", "B."] * 2
    assert report["scientific_authority"] == "none"


def test_pending_generation_resumes_without_sampling_a_new_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 46, 256, 66, 46, 257])
    paused = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        pause_after=2,
    )
    assert paused.status == "paused"
    resumed = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        resume=paused.receipt,
    )
    report = read_generation(
        output,
        resumed.receipt,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    # FOR-1674: re-addressed to continuations[0]; the aggregate is both continuations.
    assert report["continuations"][0]["sentences"] == ["A.", "B."]
    assert report["summary"]["sentences"] == ["A.", "B."] * 2


def test_invalid_generated_utf8_remains_a_recorded_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [255, 257])
    result = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    report = read_generation(
        output,
        result.receipt,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
    )
    # FOR-1674: each continuation ends invalid_utf8 on its own pending [255]; the
    # per-continuation predicates are asserted beside the (unchanged) aggregate.
    assert [c["outcome"] for c in report["continuations"]] == ["invalid_utf8"] * 2
    assert all(c["sentences"] == [] and c["boundary"] == 257 for c in report["continuations"])
    assert report["summary"]["outcome"] == "invalid_utf8"
    assert report["summary"]["pending"] == [255]
    assert report["summary"]["sentences"] == []


def test_generation_rejects_a_partial_training_receipt(tmp_path: Path) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    partial = run_probe_training(
        template.profile,
        template.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        stop_after_seal=lambda phase, step: phase == "codec_warmup",
        latent_usage=POSTERIOR_MEAN,
    )
    with pytest.raises(ValueError, match="complete training"):
        run_probe_generation(
            output,
            partial.receipt,
            template.profile,
            template.source_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )


@pytest.mark.parametrize("mutation", ["prompt", "rng", "summary"])
def test_rehashed_generation_forgery_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65])
    result = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        pause_after=1,
    )
    report = read_json(output, result.receipt, max_bytes=LIMIT)
    if mutation == "summary":
        report["summary"]["observations"] = []
    else:
        state = torch.load(output / report["snapshot"]["path"], weights_only=True)
        if mutation == "prompt":
            state["rollout"]["history"] = (b"Different prompt.",)
        else:
            state["rng"]["cpu"] = torch.zeros(2, dtype=torch.uint8)
        target = output / "altered.pt"
        torch.save(state, target)
        report["snapshot"] = {
            "path": target.name,
            "sha256": sha256(target.read_bytes()).hexdigest(),
        }
        # FOR-1674: the paused continuation's own entry carries the same reference (the
        # reader refuses a top-level reference that differs from it), so the forgery
        # rewrites both and is rejected for the altered CONTENT, not for the mismatch.
        report["continuations"][-1]["snapshot"] = report["snapshot"]
    raw = json.dumps(report).encode()
    target = output / "altered.json"
    target.write_bytes(raw)
    artifact = ArtifactIdentity(target.name, ContentHash("sha256", sha256(raw).hexdigest()))
    rng = torch.get_rng_state()
    with pytest.raises(ValueError):
        read_generation(
            output,
            artifact,
            training,
            template.profile,
            template.source_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_unscripted_generation_matches_after_mid_sentence_resume(tmp_path: Path) -> None:
    template, output, training = trained(tmp_path)
    arguments = {"max_source_bytes": 1_000_000, "max_artifact_bytes": LIMIT}
    whole = run_probe_generation(
        output, training, template.profile, template.source_root, **arguments
    )
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=1, **arguments
    )
    assert paused.status == "paused"
    resumed = run_probe_generation(
        output, training, template.profile, template.source_root, resume=paused.receipt, **arguments
    )
    first = read_generation(
        output, whole.receipt, training, template.profile, template.source_root, **arguments
    )
    second = read_generation(
        output, resumed.receipt, training, template.profile, template.source_root, **arguments
    )
    assert first["summary"] == second["summary"]


def test_generation_defect_preserves_forensic_state_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entropic_substrate.ladder.run.generation as generation

    template, output, training = trained(tmp_path)

    def broken(*args: object) -> None:
        raise RuntimeError("injected generation defect")

    monkeypatch.setattr(generation, "advance_rollout", broken)
    with pytest.raises(RuntimeError, match="injected"):
        run_probe_generation(
            output,
            training,
            template.profile,
            template.source_root,
            max_source_bytes=1_000_000,
            max_artifact_bytes=LIMIT,
        )
    assert len(list(output.glob("*-generation-failure.json"))) == 1
    assert len(list(output.glob("*-generation-failure.pt"))) == 1
    assert not list(output.glob("*-generation.json"))


# ------------------------------------------------------------------------------------------
# FOR-1674: N continuations of one prompt, the sampling law recorded (receipt v3).
#
# Design gate G2: each continuation draws only inside its own torch.random.fork_rng island,
# seeded once with torch.manual_seed(seed_i). G1 (an explicit torch.Generator threaded
# through advance_rollout) is the recorded alternative: Beta.rsample takes no generator, so
# it would need a change to core/sentence/law.py outside this lane.
#
# Seed expression: seed_i = (profile.seed * 0x9E3779B1 + ((S << 8) | i)) % 2**32 with
# S = schedule.steps. The typed integers below are computed by hand, never by the function
# under test.

ARGS = {"max_source_bytes": 1_000_000, "max_artifact_bytes": LIMIT}
SAMPLER = "multinomial_temperature_1"


def _framed_cpu_digest(seed: int) -> str:
    """Independent recomputation of rng_digest on this CPU host (CPU-measured).

    sha256 over: 8-byte big-endian stream count, then for each stream an 8-byte big-endian
    length and the raw uint8 bytes. A CPU-only host has exactly one stream.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        raw = bytes(torch.get_rng_state().numpy().tobytes())
    digest = sha256()
    digest.update((1).to_bytes(8, "big"))
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def test_continuation_seed_known_answers_are_typed_literals() -> None:
    seed = generation_module.continuation_seed
    assert seed(42, 4, 0) == 4112120586
    assert seed(42, 4, 1) == 4112120587
    assert seed(0, 1, 0) == 256
    assert seed(2**32 - 1, 4, 3) == (4294967295 * 0x9E3779B1 + 1027) % 2**32


def test_continuation_seed_is_injective_in_the_index_and_below_2_32() -> None:
    seed = generation_module.continuation_seed
    for steps in (1, 4, 1000, 2**24 - 1):
        values = [seed(42, steps, i) for i in range(256)]
        assert len(set(values)) == 256
        assert all(0 <= v < 2**32 for v in values)


def test_continuation_seed_is_a_bijection_in_the_profile_seed() -> None:
    seed = generation_module.continuation_seed
    grid = (0, 1, 2, 3, 42, 1000, 2**31, 2**32 - 2, 2**32 - 1)
    for index in (0, 1, 255):
        assert len({seed(s, 4, index) for s in grid}) == len(grid)


def test_continuation_seeds_never_alias_an_evaluation_step_seed() -> None:
    seed = generation_module.continuation_seed
    for steps in (1, 4, 1000):
        evaluation = {
            (s * 0x9E3779B1 + t) % 2**32 for s in (0, 1, 42, 2**32 - 1) for t in range(steps + 1)
        }
        generated = {seed(s, steps, i) for s in (0, 1, 42, 2**32 - 1) for i in range(256)}
        assert not evaluation & generated


def test_continuation_index_256_aliases_the_next_step_so_the_count_is_capped() -> None:
    seed = generation_module.continuation_seed
    assert (42 * 0x9E3779B1 + ((4 << 8) | 256)) % 2**32 == (
        42 * 0x9E3779B1 + ((5 << 8) | 0)
    ) % 2**32
    with pytest.raises(ValueError, match="continuation"):
        seed(42, 4, 256)


@pytest.mark.parametrize(
    ("seed", "steps", "index"),
    [
        (42, 0, 0),
        (42, 2**24, 0),
        (42, 4, -1),
        (42, 4, 256),
        (2**32, 4, 0),
        (True, 4, 0),
        (42, 4, True),
    ],
)
def test_continuation_seed_refuses_values_outside_its_documented_domain(
    seed: object, steps: object, index: object
) -> None:
    with pytest.raises(ValueError, match="continuation"):
        generation_module.continuation_seed(seed, steps, index)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def generated(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity]]:
    """One real (unscripted) trained lineage and its whole N=2 generation run."""
    template, output, training = trained(tmp_path_factory.mktemp("v3"))
    torch.manual_seed(2026)
    whole = run_probe_generation(output, training, template.profile, template.source_root, **ARGS)
    yield template, output, training, whole.receipt


def _read(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
    receipt: ArtifactIdentity,
) -> dict[str, object]:
    template, output, training, _ = generated
    return read_generation(
        output, receipt, training, template.profile, template.source_root, **ARGS
    )


def _law_only(report: dict[str, object]) -> list[dict[str, object]]:
    """Everything a continuation records except its (run-specific) snapshot reference."""
    return [
        {k: v for k, v in entry.items() if k != "snapshot"}
        for entry in report["continuations"]  # type: ignore[attr-defined]
    ]


def test_receipt_v3_records_every_continuation_with_its_sampling_law(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    template, output, _, receipt = generated
    profile = template.profile
    count = profile.schedule.generation_continuations
    raw = read_json(output, receipt, max_bytes=LIMIT)
    assert raw["schema"] == "h-generation-receipt-v3"  # M4: a v2 writer is RED here
    assert len(raw["continuations"]) == count == 2  # M3: one continuation regardless is RED
    for index, entry in enumerate(raw["continuations"]):
        seed = generation_module.continuation_seed(profile.seed, profile.schedule.steps, index)
        assert set(entry) == {
            "index",
            "seed",
            "sampling_law",
            "sentences",
            "log_scores",
            "outcome",
            "boundary",
            "snapshot",
        }
        assert entry["index"] == index and entry["seed"] == seed
        # M2: dropping the sampling-law block, or its fields, is RED here
        assert entry["sampling_law"] == {
            "sampler": SAMPLER,
            "latent": "rsample",
            "seed": seed,
            "rng_digest": _framed_cpu_digest(seed),
        }
        assert entry["outcome"] != "paused"
        assert entry["log_scores"] and entry["boundary"] in (256, 257, None)
    assert raw["continuations"][0]["seed"] != raw["continuations"][1]["seed"]
    assert raw["sampling"] == {
        "sampler": SAMPLER,
        "temperature": 1.0,
        "continuations": count,
        "seed": profile.seed,
        "rng_digest": generation_module.sampling_digest(
            [e["sampling_law"]["rng_digest"] for e in raw["continuations"]]
        ),
    }
    assert raw["snapshot"] == raw["continuations"][-1]["snapshot"]
    summary = raw["summary"]
    assert set(summary) == {
        "outcome",
        "sentences",
        "observations",
        "pending",
        "log_scores",
        "prompt",
    }
    assert summary["sentences"] == [s for e in raw["continuations"] for s in e["sentences"]]
    assert summary["log_scores"] == [x for e in raw["continuations"] for x in e["log_scores"]]
    assert len(summary["observations"]) == len(summary["log_scores"])
    assert summary["outcome"] == raw["continuations"][-1]["outcome"]
    assert _read(generated, receipt) == raw  # the reader re-derives every field


def test_sampling_digest_is_independent_of_the_seed_integer() -> None:
    """Teeth for M1: the digest reads the RNG state, so an unseeded island cannot match."""
    assert _framed_cpu_digest(5) != _framed_cpu_digest(6)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        assert generation_module.rng_state_digest(capture_rng()) == _framed_cpu_digest(5)
        assert generation_module.rng_state_digest(capture_rng()) != sha256(b"5").hexdigest()


def test_different_continuations_draw_different_latents_and_laws(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    _, output, _, receipt = generated
    raw = read_json(output, receipt, max_bytes=LIMIT)
    digests = {e["sampling_law"]["rng_digest"] for e in raw["continuations"]}
    assert len(digests) == len(raw["continuations"])
    latents = []
    for entry in raw["continuations"]:
        rollout = torch.load(output / entry["snapshot"]["path"], weights_only=True)["rollout"]
        first = (
            rollout["sample"]
            if rollout["sample"] is not None
            else rollout["emissions"][0]["shadow"]
        )
        latents.append(first)
    # the first sampled latent is continuous, so distinct seeds differ almost surely;
    # byte observations alone could repeat in a tiny decoder and would prove nothing
    assert not torch.equal(latents[0], latents[1])


def test_same_seed_reproduces_byte_identically_whatever_the_ambient_rng(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    template, output, training, receipt = generated
    before = _read(generated, receipt)
    torch.manual_seed(31337)  # an ambient state unlike the fixture's: M1 (ignore the seed) is RED
    again = run_probe_generation(output, training, template.profile, template.source_root, **ARGS)
    after = _read(generated, again.receipt)
    assert _law_only(after) == _law_only(before)
    assert after["summary"] == before["summary"] and after["sampling"] == before["sampling"]
    first = torch.load(
        output / read_json(output, receipt, max_bytes=LIMIT)["snapshot"]["path"], weights_only=True
    )
    second = torch.load(
        output / read_json(output, again.receipt, max_bytes=LIMIT)["snapshot"]["path"],
        weights_only=True,
    )
    assert torch.equal(first["rng"]["cpu"], second["rng"]["cpu"])


def test_generation_leaves_the_callers_rng_untouched(tmp_path: Path) -> None:
    template, output, training = trained(tmp_path)
    torch.manual_seed(404)
    before = torch.get_rng_state()
    result = run_probe_generation(output, training, template.profile, template.source_root, **ARGS)
    assert torch.equal(torch.get_rng_state(), before)
    read_generation(
        output, result.receipt, training, template.profile, template.source_root, **ARGS
    )
    assert torch.equal(torch.get_rng_state(), before)


def test_the_continuation_count_comes_from_the_profile(tmp_path: Path) -> None:
    template = engine(tmp_path)
    profile = replace(
        template.profile, schedule=replace(template.profile.schedule, generation_continuations=3)
    )
    three = SentenceExecution(profile, template.source_root, max_source_bytes=1_000_000)
    output = tmp_path / "run"
    output.mkdir()
    training = run_probe_training(
        three.profile,
        three.source_root,
        output,
        max_source_bytes=1_000_000,
        max_artifact_bytes=LIMIT,
        latent_usage=POSTERIOR_MEAN,
    ).receipt
    result = run_probe_generation(output, training, profile, three.source_root, **ARGS)
    raw = read_json(output, result.receipt, max_bytes=LIMIT)
    assert len(raw["continuations"]) == 3 and raw["sampling"]["continuations"] == 3
    assert len({e["seed"] for e in raw["continuations"]}) == 3


def _reissue(output: Path, report: dict[str, object], name: str) -> ArtifactIdentity:
    raw = json.dumps(report).encode()
    target = output / name
    target.write_bytes(raw)
    return ArtifactIdentity(target.name, ContentHash("sha256", sha256(raw).hexdigest()))


def _refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
    report: dict[str, object],
    name: str,
    match: str | None = None,
) -> None:
    template, output, training, _ = generated
    artifact = _reissue(output, report, name)
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match=match):
        read_generation(output, artifact, training, template.profile, template.source_root, **ARGS)
    assert torch.equal(torch.get_rng_state(), before)


def _tamper_dropping_the_sampling_law(report: dict[str, object]) -> None:
    for entry in report["continuations"]:  # type: ignore[attr-defined]
        del entry["sampling_law"]


def _tamper_one_seed(report: dict[str, object]) -> None:
    report["continuations"][0]["seed"] += 1  # type: ignore[index]


def _tamper_one_digest(report: dict[str, object]) -> None:
    report["continuations"][1]["sampling_law"]["rng_digest"] = "0" * 64  # type: ignore[index]


def _tamper_the_sampling_digest(report: dict[str, object]) -> None:
    report["sampling"]["rng_digest"] = "1" * 64  # type: ignore[index]


def _tamper_the_family_root(report: dict[str, object]) -> None:
    report["sampling"]["seed"] += 1  # type: ignore[index]


def _tamper_a_dropped_continuation(report: dict[str, object]) -> None:
    del report["continuations"][-1]  # type: ignore[attr-defined]


def _tamper_the_sampler_label(report: dict[str, object]) -> None:
    report["continuations"][0]["sampling_law"]["sampler"] = "multinomial_temperature_2"  # type: ignore[index]


def _tamper_the_boundary(report: dict[str, object]) -> None:
    entry = report["continuations"][0]  # type: ignore[index]
    entry["boundary"] = None if entry["boundary"] is not None else 256


def _tamper_the_index(report: dict[str, object]) -> None:
    report["continuations"][1]["index"] = 5  # type: ignore[index]


def _tamper_the_top_level_snapshot(report: dict[str, object]) -> None:
    report["snapshot"] = report["continuations"][0]["snapshot"]  # type: ignore[index]


@pytest.mark.parametrize(
    "tamper",
    [
        _tamper_dropping_the_sampling_law,
        _tamper_one_seed,
        _tamper_one_digest,
        _tamper_the_sampling_digest,
        _tamper_the_family_root,
        _tamper_a_dropped_continuation,
        _tamper_the_sampler_label,
        _tamper_the_boundary,
        _tamper_the_index,
        _tamper_the_top_level_snapshot,
    ],
)
def test_a_rehashed_v3_receipt_that_misstates_its_law_is_refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
    tamper: object,
) -> None:
    report = read_json(generated[1], generated[3], max_bytes=LIMIT)
    tamper(report)  # type: ignore[operator]
    _refused(generated, report, f"{tamper.__name__}.json")  # type: ignore[attr-defined]


@pytest.mark.parametrize("old", ["h-generation-receipt-v1", "h-generation-receipt-v2"])
def test_old_receipts_require_their_original_reader(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity], old: str
) -> None:
    """Ruling D3: no implicit migration. v2 replays only against its own epoch's reader."""
    report = read_json(generated[1], generated[3], max_bytes=LIMIT)
    report["schema"] = old
    del report["continuations"], report["sampling"]
    _refused(generated, report, f"{old}.json", match="requires its original reader")


def test_a_v2_parent_receipt_requires_its_original_reader(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    _, output, _, receipt = generated
    old = read_json(output, receipt, max_bytes=LIMIT)
    old["schema"] = "h-generation-receipt-v2"
    old["status"] = "paused"
    parent = _reissue(output, old, "parent-v2.json")
    child = read_json(output, receipt, max_bytes=LIMIT)
    child["parent_receipt"] = {"path": parent.path, "sha256": parent.content_hash.digest}
    _refused(generated, child, "child-of-v2.json", match="requires its original reader")


def test_a_trace_without_the_sealed_final_step_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    original = generation_module.resume_seal

    def off_by_one(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)  # type: ignore[arg-type]
        args[2].step -= 1  # type: ignore[attr-defined]
        return result

    monkeypatch.setattr(generation_module, "resume_seal", off_by_one)
    with pytest.raises(ValueError, match="sealed final step"):
        run_probe_generation(output, training, template.profile, template.source_root, **ARGS)


def _continuation_lengths(report: dict[str, object]) -> list[int]:
    return [len(e["log_scores"]) for e in report["continuations"]]  # type: ignore[attr-defined]


def test_pause_inside_the_second_continuation_resumes_to_the_whole_run(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    template, output, training, receipt = generated
    whole = _read(generated, receipt)
    first, second = _continuation_lengths(whole)
    assert second >= 2, "the pinned seeds must leave continuation 1 room for an interior pause"
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=first + 1, **ARGS
    )
    assert paused.status == "paused"
    mid = _read(generated, paused.receipt)
    assert len(mid["continuations"]) == 2 and mid["continuations"][1]["outcome"] == "paused"
    assert mid["summary"]["outcome"] == "paused"
    # the finished continuation is carried forward by content address, never rewritten
    assert mid["continuations"][0]["snapshot"] != mid["continuations"][1]["snapshot"]
    resumed = run_probe_generation(
        output, training, template.profile, template.source_root, resume=paused.receipt, **ARGS
    )
    # a double seed or a skipped seed on resume would change continuation 1's draws
    done = _read(generated, resumed.receipt)
    assert done["status"] == "generation_complete"
    assert _law_only(done) == _law_only(whole) and done["summary"] == whole["summary"]
    assert done["continuations"][0]["snapshot"] == mid["continuations"][0]["snapshot"]


def test_a_paused_v3_parent_resumes_again_and_still_matches_the_whole_run(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    # PR #381 review (CodeRabbit 4370049138): resume a receipt whose own parent is a
    # paused v3 receipt. Two resumes in a row must reproduce the uninterrupted run.
    template, output, training, receipt = generated
    whole = _read(generated, receipt)
    total = sum(_continuation_lengths(whole))
    assert total >= 3, "the pinned seeds must leave room for two interior pauses"
    first = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=1, **ARGS
    )
    assert first.status == "paused"
    second = run_probe_generation(
        output,
        training,
        template.profile,
        template.source_root,
        resume=first.receipt,
        pause_after=total - 1,
        **ARGS,
    )
    assert second.status == "paused"
    mid = read_json(output, second.receipt, max_bytes=LIMIT)
    parent = read_json(output, artifact_from_mapping(mid["parent_receipt"]), max_bytes=LIMIT)
    assert parent["schema"] == "h-generation-receipt-v3" and parent["status"] == "paused"
    _read(generated, second.receipt)  # a v3 receipt with a v3 parent is admitted
    final = run_probe_generation(
        output, training, template.profile, template.source_root, resume=second.receipt, **ARGS
    )
    done = _read(generated, final.receipt)
    assert done["status"] == "generation_complete"
    assert _law_only(done) == _law_only(whole) and done["summary"] == whole["summary"]


def test_pause_at_the_last_observation_of_the_first_continuation_is_still_paused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    template, output, training, receipt = generated
    whole = _read(generated, receipt)
    first = _continuation_lengths(whole)[0]
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=first, **ARGS
    )
    mid = _read(generated, paused.receipt)
    assert paused.status == "paused" and mid["status"] == "paused"
    assert len(mid["continuations"]) == 1 and mid["continuations"][0]["outcome"] != "paused"
    assert mid["summary"]["outcome"] == "paused"
    with pytest.raises(ValueError, match="cursor"):
        run_probe_generation(
            output,
            training,
            template.profile,
            template.source_root,
            resume=paused.receipt,
            pause_after=first,
            **ARGS,
        )
    resumed = run_probe_generation(
        output, training, template.profile, template.source_root, resume=paused.receipt, **ARGS
    )
    done = _read(generated, resumed.receipt)
    assert _law_only(done) == _law_only(whole) and done["summary"] == whole["summary"]


def test_scripted_pause_at_the_continuation_boundary_resumes_into_a_fresh_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 46, 256, 66, 46, 257])
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=6, **ARGS
    )
    mid = read_json(output, paused.receipt, max_bytes=LIMIT)
    assert paused.status == "paused" and [e["outcome"] for e in mid["continuations"]] == [
        "document_end"
    ]
    assert mid["summary"]["pending"] == []
    resumed = run_probe_generation(
        output, training, template.profile, template.source_root, resume=paused.receipt, **ARGS
    )
    report = read_generation(
        output, resumed.receipt, training, template.profile, template.source_root, **ARGS
    )
    assert [e["outcome"] for e in report["continuations"]] == ["document_end"] * 2
    assert [e["index"] for e in report["continuations"]] == [0, 1]
    assert report["summary"]["observations"] == [65, 46, 256, 66, 46, 257] * 2


def test_a_defect_in_the_second_continuation_keeps_its_island_rng_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template, output, training = trained(tmp_path)
    scripted_tokens(monkeypatch, [65, 46, 256, 66, 46, 257])
    real = generation_module.advance_rollout
    seen: list[dict[str, object]] = []

    def flaky(*args: object) -> object:
        seen.append(capture_rng())  # inside the island, before the draw
        if len(seen) == 8:  # continuation 0 took 6 advances; this is continuation 1's second
            raise RuntimeError("injected generation defect")
        return real(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(generation_module, "advance_rollout", flaky)
    torch.manual_seed(77)
    ambient = torch.get_rng_state()
    with pytest.raises(RuntimeError, match="injected"):
        run_probe_generation(output, training, template.profile, template.source_root, **ARGS)
    assert len(list(output.glob("*-generation-failure.json"))) == 1
    assert len(list(output.glob("*-generation-failure.pt"))) == 1
    assert not list(output.glob("*-generation.json"))
    assert torch.equal(torch.get_rng_state(), ambient)  # the island did not leak
    failure = torch.load(next(output.glob("*-generation-failure.pt")), weights_only=True)
    # the snapshot holds the failing continuation's island state, not the restored ambient
    assert torch.equal(failure["rng"]["cpu"], seen[-1]["cpu"])
    assert not torch.equal(failure["rng"]["cpu"], ambient)
    assert failure["rollout"]["observations"] == (65,)  # continuation 1 had advanced once


def _extra_continuation_forgery(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> dict[str, object]:
    template, output, _, receipt = generated
    report = read_json(output, receipt, max_bytes=LIMIT)
    extra = copy.deepcopy(report["continuations"][1])  # type: ignore[index]
    extra["index"] = 2
    law = generation_module._derived_law(template, 2)
    extra["seed"] = law["seed"]
    extra["sampling_law"] = law
    first_len = len(report["continuations"][0]["log_scores"])  # type: ignore[index]
    obs = report["summary"]["observations"]  # type: ignore[index]
    report["continuations"].append(extra)  # type: ignore[attr-defined]
    report["snapshot"] = extra["snapshot"]  # type: ignore[index]
    report["summary"]["sentences"] = report["summary"]["sentences"] + extra["sentences"]  # type: ignore[index]
    report["summary"]["log_scores"] = report["summary"]["log_scores"] + extra["log_scores"]  # type: ignore[index]
    report["summary"]["observations"] = obs + obs[first_len:]  # type: ignore[index]
    return report


def test_forged_extra_continuation_beyond_the_profile_count_is_refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    """Teeth for the upper bound of the continuation count: a third entry is refused."""
    _refused(
        generated,
        _extra_continuation_forgery(generated),
        "extra-continuation.json",
        match="differ from the profile count",
    )


def _started(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
    snapshot: object,
    index: int,
) -> object:
    template, output, _, _ = generated
    identity = artifact_from_mapping(snapshot)
    saved = generation_module._snapshot(output, identity, LIMIT)
    state = rollout_from_payload(saved["rollout"], template.model, template.codec)
    return generation_module._Started(
        index, state, saved["rng"], artifact_reference(output, identity)
    )


def _paused_first_forgery(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> dict[str, object]:
    """A complete receipt whose continuation 0 is an unfinished (paused) snapshot."""
    template, output, training, receipt = generated
    whole = read_json(output, receipt, max_bytes=LIMIT)
    first = len(whole["continuations"][0]["log_scores"])  # type: ignore[index]
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=first + 1, **ARGS
    )
    mid = read_json(output, paused.receipt, max_bytes=LIMIT)
    assert mid["continuations"][1]["outcome"] == "paused"  # type: ignore[index]
    s0 = _started(generated, mid["continuations"][1]["snapshot"], 0)  # type: ignore[index]
    s1 = _started(generated, whole["continuations"][1]["snapshot"], 1)  # type: ignore[index]
    expected = [
        generation_module._entry(template, s, generation_module._derived_law(template, s.index))  # type: ignore[attr-defined]
        for s in (s0, s1)
    ]
    forged = dict(whole)
    forged["continuations"] = expected
    forged["snapshot"] = expected[-1]["snapshot"]
    forged["summary"] = generation_module._summary(template, [s0, s1], expected)  # type: ignore[arg-type, list-item]
    forged["status"] = generation_module._status(expected, template)
    assert expected[0]["outcome"] == "paused" and forged["status"] == "generation_complete"
    return forged


def test_forged_unfinished_earlier_continuation_is_refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    """Teeth for the 'only the last started continuation can be unfinished' guard."""
    _refused(
        generated,
        _paused_first_forgery(generated),
        "paused-first.json",
        match="only the last started continuation",
    )


def test_forged_status_on_a_complete_receipt_is_refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    report = read_json(generated[1], generated[3], max_bytes=LIMIT)
    assert report["status"] == "generation_complete"
    report["status"] = "paused"
    _refused(generated, report, "status-paused.json", match="status differs")


def test_forged_complete_status_on_a_paused_receipt_is_refused(
    generated: tuple[SentenceExecution, Path, ArtifactIdentity, ArtifactIdentity],
) -> None:
    template, output, training, receipt = generated
    first = len(read_json(output, receipt, max_bytes=LIMIT)["continuations"][0]["log_scores"])
    paused = run_probe_generation(
        output, training, template.profile, template.source_root, pause_after=first + 1, **ARGS
    )
    report = read_json(output, paused.receipt, max_bytes=LIMIT)
    assert report["status"] == "paused"
    report["status"] = "generation_complete"
    _refused(generated, report, "status-complete.json", match="status differs")


def test_sampling_digest_known_answer_pins_the_length_framing() -> None:
    """Teeth for the framing: hand-computed from the documented framing, not the function.

    Dropping the per-digest 8-byte length prefix yields 9fe9ad8e... instead.
    """
    assert (
        generation_module.sampling_digest(["a" * 64, "b" * 64])
        == (
            "6eb863582500f8e20167d5e376032278097047c31dd07ecb6966337864c3f732"  # pragma: allowlist secret
        )
    )
