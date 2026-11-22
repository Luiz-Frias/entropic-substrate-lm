"""Teeth for ``scripts/logger_smoke.py``: the series registry and its probe.

The probe asks one question per registered emission path: did the tape record
what the code emitted, and did the human stderr line (the one a pod's system
log shows) say the same thing. These tests pin the registry's coverage and
show the probe going RED when an emitter drops or garbles the event.
"""

from __future__ import annotations

from pathlib import Path

import logger_smoke as smoke
import pytest

import repo_paths
from entropic_substrate.kernel.runtime.logging import logger

pytestmark = pytest.mark.unit

ACTIVE = smoke.SERIES_PATHS[0]


def test_every_log_vocabulary_module_has_a_registry_entry() -> None:
    package = repo_paths.REPO_ROOT / "src"
    found = {
        ".".join(path.relative_to(package).with_suffix("").parts)
        for path in package.rglob("*_log_events.py")
    }
    assert found, "the glob matched no vocabulary module, so this test examined nothing"
    registered = {entry.vocabulary for entry in smoke.SERIES_PATHS}
    assert found <= registered, f"unregistered log vocabularies: {sorted(found - registered)}"


def test_registry_names_are_unique_and_at_least_one_path_is_active() -> None:
    names = [entry.name for entry in smoke.SERIES_PATHS]
    assert len(names) == len(set(names))
    assert smoke.select("active")


def test_registry_entries_resolve_to_real_events_and_emitters() -> None:
    for entry in smoke.SERIES_PATHS:
        assert isinstance(smoke.event_of(entry), str)
        assert callable(smoke.emitter_of(entry))


def test_probe_fails_when_the_emitter_drops_the_event(tmp_path: Path) -> None:
    def silent(_level: str, _event: str, **_fields: object) -> None:
        return None

    result = smoke.probe(ACTIVE, tmp_path / "tape.jsonl", emit=silent)

    assert not result.ok
    assert "tape" in result.detail


def test_probe_fails_when_the_human_line_is_not_the_rendered_sentence(tmp_path: Path) -> None:
    def garbled(level: str, event: str, **fields: object) -> None:
        getattr(logger.bind(event=event, **fields), level)("something else entirely")

    result = smoke.probe(ACTIVE, tmp_path / "tape.jsonl", emit=garbled)

    assert not result.ok
    assert "message" in result.detail or "stderr" in result.detail


def test_probe_fails_when_a_bound_field_never_reaches_the_tape(tmp_path: Path) -> None:
    def lossy(level: str, event: str, **fields: object) -> None:
        sentence = smoke.render_of(ACTIVE)(event, **fields)
        getattr(logger.bind(event=event), level)(sentence)

    result = smoke.probe(ACTIVE, tmp_path / "tape.jsonl", emit=lossy)

    assert not result.ok
    assert "field" in result.detail


def test_run_with_nothing_active_is_refused_not_green(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke.run("active", paths=()) == smoke.EXIT_NOTHING_EXAMINED
    assert "nothing" in capsys.readouterr().err


def test_run_with_no_retired_paths_passes_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert smoke.run("retired", paths=()) == 0
    assert "no retired" in capsys.readouterr().out


def test_image_ref_reads_the_digest_pin_from_the_deploy_doc(tmp_path: Path) -> None:
    digest = "sha256:" + "a" * 64
    doc = tmp_path / "deploy.md"
    doc.write_text(f"prose\n\n```text\nghcr.io/owner/name@{digest}\n```\n")

    assert smoke.image_ref(doc) == f"ghcr.io/owner/name@{digest}"


def test_image_ref_refuses_a_doc_without_a_digest_pin(tmp_path: Path) -> None:
    doc = tmp_path / "deploy.md"
    doc.write_text("ghcr.io/owner/name:latest\n")

    with pytest.raises(ValueError, match="digest"):
        smoke.image_ref(doc)


@pytest.mark.slow
def test_retired_paths_still_record_what_they_emit(tmp_path: Path) -> None:
    for entry in smoke.select("retired"):
        result = smoke.probe(entry, tmp_path / f"{entry.name}.jsonl")
        assert result.ok, f"{entry.name}: {result.detail}"
