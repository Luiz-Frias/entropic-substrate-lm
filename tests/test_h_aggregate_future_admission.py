"""DH-48 under DH-51 (FOR-1716): the loader's admission of a null future term.

An H1 profile requires ``optimization.future_weight == 0``, so the objective writes a null
``future_nll_per_dimension`` on every prediction row. ``load_h_run`` admits that null only when
``run_start.profile`` is present, authentic (rebuilds into an ``HProfile`` whose identity is
``run_start.profile_identity``) and records a future weight of exactly zero. These are the
synthetic-stream cases; the real pair-tracer cell is pinned in tests/test_h_pair_request.py
(it needs the worker bytes that write the manifest).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_aggregate import REQUEST_A, build_h_run_rows, write_h_run
from tests.test_sentence_profile import profile

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.analyze.h_aggregate import load_h_run
from entropic_substrate.ladder.capture.reader.telemetry import TelemetryGap

pytestmark = pytest.mark.unit

# the refusal text as it stood before L4d, pinned VERBATIM (S24): a ``match=`` regex searches a
# prefix, so a suffix appended to the message (for example " (no manifest)") would pass it
NULL_TAIL = "step 1 (prediction) has a null future_nll_per_dimension"


def assert_refused_verbatim(files: tuple[Path, Path], **kwargs: Any) -> None:
    """``load_h_run`` refuses with exactly ``<metrics path>: <NULL_TAIL>``, nothing before or after."""
    with pytest.raises(TelemetryGap) as caught:
        load_h_run(*files, **kwargs)
    assert str(caught.value) == f"{files[0]}: {NULL_TAIL}"


def _h1(weight: float = 0.0) -> HProfile:
    base = profile()
    return replace(
        base,
        name="h1-isotropic-tiny",
        stage="H1",
        consumer="isotropic",
        optimization=replace(base.optimization, future_weight=weight),
    )


def _write(
    tmp_path: Path,
    prof: HProfile,
    *,
    manifest: object = "from-profile",
    identity: str | None = None,
    null_future: bool = True,
    drop_profile: bool = False,
) -> tuple[Path, Path]:
    def mutate(rows: list[dict[str, Any]]) -> None:
        start = rows[0]
        start["profile_identity"] = prof.identity if identity is None else identity
        if not drop_profile:
            start["profile"] = prof.manifest if manifest == "from-profile" else manifest
        for row in rows:
            if "run_identity" in row:
                row["run_identity"] = start["profile_identity"]
            if null_future and row.get("phase") == "prediction":
                row["future_nll_per_dimension"] = None

    return write_h_run(tmp_path, prof.identity, REQUEST_A, mutate_metrics=mutate)


def test_a_zero_recorded_weight_admits_the_null_and_summarizes_it_as_absent(
    tmp_path: Path,
) -> None:
    run = load_h_run(*_write(tmp_path, _h1()))
    prediction = [scalars for _, phase, scalars in run.training if phase == "prediction"]
    assert prediction and all("future_nll_per_dimension" not in s for s in prediction)
    assert run.run_start["profile"] == _h1().manifest  # the manifest rides HRun.run_start


def test_the_same_files_with_the_manifest_stripped_keep_todays_refusal(tmp_path: Path) -> None:
    assert_refused_verbatim(_write(tmp_path, _h1(), drop_profile=True))


def test_a_null_under_a_recorded_nonzero_weight_keeps_the_existing_message(
    tmp_path: Path,
) -> None:
    """The fixture must use a stage that admits a weight (H0): an H1 profile refuses it."""
    with pytest.raises(ValueError, match="H1 preservation cannot bundle"):
        _h1(0.3)
    h0 = replace(profile(), optimization=replace(profile().optimization, future_weight=0.3))
    assert h0.stage == "H0"
    assert_refused_verbatim(_write(tmp_path, h0))


def test_an_authentic_nonzero_weight_with_finite_values_loads_unchanged(tmp_path: Path) -> None:
    h0 = profile()
    run = load_h_run(*_write(tmp_path, h0, null_future=False))
    assert all(
        "future_nll_per_dimension" in scalars
        for _, phase, scalars in run.training
        if phase == "prediction"
    )


@pytest.mark.parametrize("weight", [0.0, 0])
def test_both_numeric_zero_spellings_are_admitted(tmp_path: Path, weight: float) -> None:
    prof = _h1(weight)
    assert load_h_run(*_write(tmp_path, prof)).profile_identity == prof.identity


def test_a_manifest_with_another_identity_is_refused_by_name_even_without_a_null(
    tmp_path: Path,
) -> None:
    other = replace(_h1(), name="h1-other-tiny")
    files = _write(tmp_path, _h1(), manifest=other.manifest, null_future=False)
    with pytest.raises(TelemetryGap) as raised:
        load_h_run(*files)
    message = str(raised.value)
    assert "run_start.profile" in message
    assert other.identity in message and _h1().identity in message


def test_a_forged_recorded_weight_breaks_the_identity_and_is_refused(tmp_path: Path) -> None:
    """Forging weight 0 into a non-zero profile's manifest changes the hash: refused by name."""
    h0 = profile()
    forged = copy.deepcopy(h0.manifest)
    forged["optimization"]["future_weight"] = 0.0  # type: ignore[index]
    files = _write(tmp_path, h0, manifest=forged)
    with pytest.raises(TelemetryGap, match=r"run_start\.profile has content identity"):
        load_h_run(*files)


@pytest.mark.parametrize(
    "broken",
    [
        {},
        "not-a-manifest",
        None,
        lambda m: {**m, "schema": "h-run-profile-v2"},
        lambda m: {k: v for k, v in m.items() if k != "optimization"},
    ],
)
def test_a_present_but_unrebuildable_manifest_is_refused_never_treated_as_absent(
    tmp_path: Path, broken: object
) -> None:
    prof = _h1()
    manifest = broken(prof.manifest) if isinstance(broken, Callable) else broken
    files = _write(tmp_path, prof, manifest=manifest, null_future=False)
    with pytest.raises(TelemetryGap, match=r"run_start\.profile is not an authentic HProfile"):
        load_h_run(*files)


def test_a_warmup_null_stays_admitted_without_any_manifest(tmp_path: Path) -> None:
    run = load_h_run(*write_h_run(tmp_path))
    assert run.training[0][1] == "codec_warmup"


def test_the_default_builder_run_has_no_profile_key(tmp_path: Path) -> None:
    rows, _ = build_h_run_rows("0" * 63 + "1", REQUEST_A)
    assert "profile" not in rows[0]


def test_the_admission_is_the_same_on_the_warmup_steps_override(tmp_path: Path) -> None:
    files = _write(tmp_path, _h1())
    run = load_h_run(*files, warmup_steps=2)
    assert any(phase == "prediction" for _, phase, _ in run.training)
    stripped = _write(tmp_path / "s", _h1(), drop_profile=True)
    assert_refused_verbatim(stripped, warmup_steps=2)
