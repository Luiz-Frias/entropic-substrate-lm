"""Generation continues the trained lineage and preserves pending byte sampling."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
import torch
from tests.test_sentence_execution import engine
from tests.test_sentence_rollout import scripted_tokens

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.generation import read_generation, run_probe_generation
from entropic_substrate.ladder.run.journal import read_json
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
    assert report["summary"]["outcome"] == "document_end"
    assert report["summary"]["sentences"] == ["A.", "B."]
    assert report["summary"]["observations"] == [65, 46, 256, 66, 46, 257]
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
    assert report["summary"]["sentences"] == ["A.", "B."]


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
