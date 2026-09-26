"""Trainer exports become visible only through a marker committed last."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import entropic_substrate.ladder.run.transport.d1_trainer_output_publication as publication_module
from d1_billing_finalizer_fixtures import completed_case
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import D1ExportedTrainerHandoff
from entropic_substrate.ladder.run.transport.d1_trainer_output_publication import (
    load_d1_published_trainer_output,
    publish_d1_trainer_output,
)

pytestmark = pytest.mark.unit

_PREFIX = "published/segment-0"


def test_handoff_is_published_before_marker_and_round_trips_real_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completed_case(tmp_path)
    (tmp_path / "published").mkdir()
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    calls: list[str] = []
    real_publish = publication_module.publish_d1_atomic_bytes

    def record_publication(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(str(kwargs["label"]))
        return real_publish(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(publication_module, "publish_d1_atomic_bytes", record_publication)

    marker = publish_d1_trainer_output(
        candidate,
        artifact_root=tmp_path,
        publication_prefix=_PREFIX,
    )
    observed = load_d1_published_trainer_output(
        artifact_root=tmp_path,
        publication_prefix=_PREFIX,
    )

    assert calls == ["trainer handoff", "trainer completion marker"]
    assert marker.path == str(tmp_path / f"{_PREFIX}.complete.json")
    assert observed == candidate
    manifest = json.loads(Path(marker.path).read_bytes())
    assert manifest["handoff_kind"] == "fresh_completion"
    assert manifest["artifacts"]["handoff"]["path"] == f"{_PREFIX}.handoff.json"
    assert manifest["artifacts"]["journal"]["path"] == "none.jsonl"
    assert manifest["artifacts"]["checkpoint"]["path"] == "none.pt"
    assert manifest["artifacts"]["quarantine"] == {"kind": "none"}


def test_occupied_handoff_or_missing_marker_never_looks_complete(tmp_path: Path) -> None:
    case = completed_case(tmp_path)
    published = tmp_path / "published"
    published.mkdir()
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    handoff_path = tmp_path / f"{_PREFIX}.handoff.json"
    marker_path = tmp_path / f"{_PREFIX}.complete.json"
    handoff_path.write_bytes(b"occupied")

    with pytest.raises(ValueError, match="occupied"):
        publish_d1_trainer_output(
            candidate,
            artifact_root=tmp_path,
            publication_prefix=_PREFIX,
        )
    assert not marker_path.exists()
    with pytest.raises(ValueError, match="marker"):
        load_d1_published_trainer_output(
            artifact_root=tmp_path,
            publication_prefix="published/missing",
        )


def test_post_marker_handoff_or_journal_drift_is_refused(tmp_path: Path) -> None:
    case = completed_case(tmp_path)
    (tmp_path / "published").mkdir()
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    publish_d1_trainer_output(
        candidate,
        artifact_root=tmp_path,
        publication_prefix=_PREFIX,
    )
    handoff_path = tmp_path / f"{_PREFIX}.handoff.json"
    original_handoff = handoff_path.read_bytes()
    handoff_path.write_bytes(b"x" * len(original_handoff))

    with pytest.raises(ValueError, match="handoff"):
        load_d1_published_trainer_output(
            artifact_root=tmp_path,
            publication_prefix=_PREFIX,
        )

    handoff_path.write_bytes(original_handoff)
    with case.journal_path.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="journal"):
        load_d1_published_trainer_output(
            artifact_root=tmp_path,
            publication_prefix=_PREFIX,
        )


def test_journal_drift_between_handoff_and_marker_aborts_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completed_case(tmp_path)
    (tmp_path / "published").mkdir()
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    real_publish = publication_module.publish_d1_atomic_bytes

    def drift_after_handoff(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        artifact = real_publish(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs["label"] == "trainer handoff":
            with case.journal_path.open("ab") as handle:
                handle.write(b"{}\n")
        return artifact

    monkeypatch.setattr(publication_module, "publish_d1_atomic_bytes", drift_after_handoff)

    with pytest.raises(ValueError, match="journal"):
        publish_d1_trainer_output(
            candidate,
            artifact_root=tmp_path,
            publication_prefix=_PREFIX,
        )

    assert not (tmp_path / f"{_PREFIX}.complete.json").exists()


def test_handoff_swap_after_marker_admission_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completed_case(tmp_path)
    (tmp_path / "published").mkdir()
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    publish_d1_trainer_output(
        candidate,
        artifact_root=tmp_path,
        publication_prefix=_PREFIX,
    )
    handoff_path = tmp_path / f"{_PREFIX}.handoff.json"
    value = json.loads(handoff_path.read_bytes())
    value["final_evaluation"]["future_nll"] += 0.125
    replacement = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    real_admit = publication_module._admit_reference  # pyright: ignore[reportPrivateUsage]

    def swap_after_admission(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        path = real_admit(*args, **kwargs)  # type: ignore[arg-type]
        if kwargs["label"] == "trainer handoff":
            handoff_path.write_bytes(replacement)
        return path

    monkeypatch.setattr(publication_module, "_admit_reference", swap_after_admission)

    with pytest.raises(ValueError, match="handoff"):
        load_d1_published_trainer_output(
            artifact_root=tmp_path,
            publication_prefix=_PREFIX,
        )
