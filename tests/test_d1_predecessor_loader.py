"""Portable predecessor evidence uses real Git objects and existing filing readers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.gate.ladder_campaign import artifact_mapping
from entropic_substrate.ladder.register.d1_config import load_d1_profile_catalog
from entropic_substrate.ladder.register.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.ladder.register.d1_registration_codec import (
    serialize_decision_contract,
    serialize_registered_design,
)
from entropic_substrate.ladder.run.provider.d1_command_runner import D1BoundedProcessResult
from entropic_substrate.ladder.run.worker.d1_predecessor_loader import (
    load_d1_predecessor,
    verified_git_bundle,
)

from . import ladder_campaign_fixtures

pytestmark = pytest.mark.unit


def _artifact(path: Path) -> ArtifactIdentity:
    return ArtifactIdentity(
        str(path), ContentHash("sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


@pytest.mark.parametrize("poison", [False, True])
def test_bundle_proves_actual_blob_and_ancestry_without_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: bool
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-b", "proof")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.test")
    (repo / "proof.txt").write_text("first\n")
    _git(repo, "add", "proof.txt")
    _git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "proof.txt").write_text("second\n")
    _git(repo, "add", "proof.txt")
    _git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-m", "second")
    second = _git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "proof.bundle"
    _git(repo, "bundle", "create", str(bundle), "proof")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    _git(decoy, "init", "-b", "decoy")
    foreign_index = decoy / ".git/index"
    foreign_index.write_bytes(b"untouched foreign index")
    if poison:
        monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(decoy))
        monkeypatch.setenv("GIT_INDEX_FILE", str(foreign_index))
    with verified_git_bundle(tmp_path, _artifact(bundle)) as git:
        assert git.read_file(first, "proof.txt") == b"first\n"
        assert git.read_file(second, "proof.txt") == b"second\n"
        assert git.is_ancestor(first, second)
        assert not git.is_ancestor(second, first)
    assert not (tmp_path / ".git").exists()
    assert foreign_index.read_bytes() == b"untouched foreign index"


def test_bad_bundle_hash_is_rejected_before_git(tmp_path: Path) -> None:
    bundle = tmp_path / "proof.bundle"
    bundle.write_bytes(b"not git")
    identity = ArtifactIdentity(str(bundle), ContentHash("sha256", "0" * 64))
    with pytest.raises(ValueError, match="digest"):
        with verified_git_bundle(tmp_path, identity):
            pytest.fail("bad bundle admitted")


def test_bundle_clone_cannot_inherit_repository_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.d1_predecessor_loader.os",
        SimpleNamespace(environ={"GIT_DIR": "foreign", "PATH": "/fixture/bin"}),
    )

    class InspectClone:
        def __init__(self, *, timeout_seconds: float, max_output_bytes: int) -> None:
            assert timeout_seconds > 0 and max_output_bytes > 0

        def run(
            self, argv: tuple[str, ...], *, cwd: Path, input: bytes, env: Mapping[str, str]
        ) -> D1BoundedProcessResult:
            assert "clone" in argv and cwd.is_dir() and input == b""
            assert dict(env) == {"PATH": "/fixture/bin"}
            return D1BoundedProcessResult(1, b"", b"fixture rejection")

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.d1_predecessor_loader.D1BoundedProcessRunner",
        InspectClone,
    )
    bundle = tmp_path / "proof.bundle"
    bundle.write_bytes(b"fake bundle for environment inspection")
    with pytest.raises(ValueError, match="complete bare proof"):
        with verified_git_bundle(tmp_path, _artifact(bundle)):
            pytest.fail("rejected clone admitted")


def test_predecessor_never_accepts_summary_authorization(tmp_path: Path) -> None:
    declaration = tmp_path / "predecessor.json"
    declaration.write_bytes(b'{"authorized":true}')
    with pytest.raises(ValueError, match="fields"):
        load_d1_predecessor(tmp_path, _artifact(declaration))


def _predecessor_case(root: Path, monkeypatch: pytest.MonkeyPatch):
    observed = []
    original = ladder_campaign_fixtures.verify_d1_filing

    def observe(*args):
        observed.append(args)
        return original(*args)

    monkeypatch.setattr(ladder_campaign_fixtures, "verify_d1_filing", observe)
    gate = ladder_campaign_fixtures.d1_gate(root, monkeypatch)
    detached, _capture, revisions, _store, git = observed[0]
    contract_path = root / "decision.json"
    contract_path.write_text(serialize_decision_contract(detached.decision_contract))
    preregistration = detached.manifest.artifacts["preregistration"]
    design = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=load_d1_profile_catalog()[0].family,
            decision_contract=detached.decision_contract,
            registered_seeds=tuple(detached.decision_contract.registered_constants["seeds"]),
            preregistration=D1PreregistrationArtifact(
                path=detached.verdict.preregistration.path,
                blob_sha256=preregistration.sha256.digest,
                commit=detached.verdict.preregistration.commit,
            ),
        )
    )
    design_path = root / "registered-design.json"
    design_path.write_text(serialize_registered_design(design))
    lifecycle = {
        "schema": "d1-predecessor-lifecycle-v1",
        "events": [
            {
                "event_id": event.event_id,
                "kind": event.kind.value,
                "actor": {
                    "actor_id": event.actor.actor_id,
                    "session_id": event.actor.session_id,
                    "role": event.actor.role.value,
                },
                "artifacts": [artifact_mapping(item) for item in event.artifacts],
                "sealed_packet": None
                if event.sealed_packet is None
                else artifact_mapping(event.sealed_packet),
            }
            for event in gate.record.events
        ],
    }
    lifecycle_path = root / "lifecycle.json"
    lifecycle_path.write_text(json.dumps(lifecycle))
    lifecycle_revision = "3" * 40
    git.files[(design.preregistration.commit, "decision.json")] = contract_path.read_bytes()
    git.files[(revisions.capture, "registered-design.json")] = design_path.read_bytes()
    git.files[(lifecycle_revision, "lifecycle.json")] = lifecycle_path.read_bytes()
    git.ancestors.add((revisions.filing, lifecycle_revision))

    def relative(path):
        return {"path": str(path.relative_to(root)), "sha256": _artifact(path).content_hash.digest}

    declaration = {
        "schema": "d1-predecessor-bundle-v1",
        "filing_manifest": relative(root / "filing/manifest.json"),
        "decision_contract": relative(contract_path),
        "registered_design": relative(design_path),
        "capture_root": "finite-lineage/case/capture",
        "lifecycle": relative(lifecycle_path),
        "lifecycle_revision": lifecycle_revision,
        "revisions": dict(
            zip(
                ("capture", "analysis", "alias_commitment", "review", "alias_reveal", "filing"),
                revisions.ordered,
                strict=True,
            )
        ),
        "git_bundle": None,
    }
    source = root / "predecessor.json"
    source.write_text(json.dumps(declaration))
    return source, git, gate


def test_loader_rebuilds_capture_registration_filing_and_persisted_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, git, expected = _predecessor_case(tmp_path, monkeypatch)
    result = load_d1_predecessor(tmp_path, _artifact(source), git=git)
    assert result.evidence == expected.evidence
    assert result.record.events == expected.record.events
    assert result.successor_rung == "d2"


def test_loader_refuses_uncommitted_decision_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, git, _ = _predecessor_case(tmp_path, monkeypatch)
    for key in tuple(git.files):
        if key[1] == "decision.json":
            del git.files[key]
    with pytest.raises(ValueError, match="committed bytes"):
        load_d1_predecessor(tmp_path, _artifact(source), git=git)


def test_committed_lifecycle_cannot_skip_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, git, _ = _predecessor_case(tmp_path, monkeypatch)
    declaration = json.loads(source.read_bytes())
    lifecycle = tmp_path / declaration["lifecycle"]["path"]
    events = json.loads(lifecycle.read_bytes())
    events["events"] = [event for event in events["events"] if event["kind"] != "Analyze"]
    lifecycle.write_text(json.dumps(events))
    declaration["lifecycle"]["sha256"] = _artifact(lifecycle).content_hash.digest
    source.write_text(json.dumps(declaration))
    git.files[(declaration["lifecycle_revision"], "lifecycle.json")] = lifecycle.read_bytes()
    with pytest.raises(RuntimeError, match="illegal lifecycle transition"):
        load_d1_predecessor(tmp_path, _artifact(source), git=git)
