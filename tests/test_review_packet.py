"""Blinding contract for the review-packet assembler.

The packet's blinding is only as good as the scrub: ``run_start`` carries the
arm axis and EVERY record carries ``run``, so filename relabeling alone is
theater (measured on the 2026-08-18 captures — see
scripts/assemble_review_packet.py). These tests pin the transform per rung and
prove each fail-loud gate actually fires on a planted defect.

The gates are what make this generalizable across rungs at all: E1's two arms
were parameter-identical, so its packet never had to scrub ``params``. E1.5a's
untied variant carries 85.33M against the tied 85.13M, and that single integer
unblinds the packet on its own. An allow-list of "fields we remembered to
scrub" would not have caught it; comparing WHOLE post-scrub configs does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "assemble_review_packet",
    Path(__file__).resolve().parents[1] / "scripts" / "assemble_review_packet.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["assemble_review_packet"] = _MOD
_SPEC.loader.exec_module(_MOD)

_E1 = _MOD.RUNGS["e1"]
_E15A = _MOD.RUNGS["e1-5a"]


def _run_start(**model: Any) -> dict[str, Any]:
    base = {"embedding_init": "fisher_sqrt", "fisher_init_path": "data/f.pt"}
    base.update(model)
    return {
        "event": "run_start",
        "run": "e1-fisher",
        "params": 85_130_000,
        "config": {"name": "e1-fisher", "model": base},
    }


# --------------------------------------------------------------------------
# E1 — the filed rung. These pin the behaviour the committed packets were
# built with, so the registry refactor cannot silently move a filed artifact.
# --------------------------------------------------------------------------


def test_e1_scrub_maps_every_identifying_field() -> None:
    record = _run_start()
    out = _MOD.scrub_record(record, "A", _E1)
    assert out["run"] == "arm-A"
    assert out["config"]["name"] == "arm-A"
    assert out["config"]["model"]["embedding_init"] == "init-A"
    assert "fisher_init_path" not in out["config"]["model"]
    # E1's arms were parameter-identical, so params was never scrubbed there
    assert out["params"] == 85_130_000
    # the original is not mutated — capture files stay canonical
    assert record["config"]["model"]["embedding_init"] == "fisher_sqrt"


def test_e1_scrub_touches_only_arm_fields() -> None:
    record = {"event": "train", "run": "e1-random", "step": 40, "data_order_digest": "abc"}
    out = _MOD.scrub_record(record, "B", _E1)
    assert out == {"event": "train", "run": "arm-B", "step": 40, "data_order_digest": "abc"}


# --------------------------------------------------------------------------
# E1.5a — three arms, and a wider scrub than E1 needed.
# --------------------------------------------------------------------------


def test_e15a_scrub_drops_the_freeze_axis_and_the_param_count() -> None:
    record = _run_start(embedding_freeze="input", tie_embeddings=False)
    record["params"] = 85_330_000  # untied: the integer that would unblind
    out = _MOD.scrub_record(record, "A", _E15A)
    model = out["config"]["model"]
    assert "embedding_freeze" not in model
    assert "tie_embeddings" not in model
    assert "fisher_init_path" not in model
    assert "params" not in out


def test_e15a_init_token_is_shared_not_per_arm() -> None:
    """All three E1.5a arms share one initialization.

    A per-arm token here would MANUFACTURE a difference between A and B that
    the parity gate would then correctly report as a leak — the scrub would be
    creating the very signal it exists to remove.
    """
    a = _MOD.scrub_record(_run_start(), "A", _E15A)
    b = _MOD.scrub_record(_run_start(), "B", _E15A)
    assert a["config"]["model"]["embedding_init"] == "init-shared"
    assert b["config"]["model"]["embedding_init"] == "init-shared"


def test_e15a_blinded_arms_are_identical_after_scrub() -> None:
    """The two variants differ ONLY at the axis, so scrubbing must equalize them."""
    a = _MOD.scrub_record(_run_start(embedding_freeze="input", tie_embeddings=False), "A", _E15A)
    b = _MOD.scrub_record(_run_start(embedding_freeze="both", tie_embeddings=True), "B", _E15A)
    _MOD.assert_blinded_parity({"A": a["config"], "B": b["config"]}, _E15A)


# --------------------------------------------------------------------------
# Guard teeth — each gate must FAIL on a planted defect, not merely pass clean.
# --------------------------------------------------------------------------


def test_parity_gate_fires_on_an_unscrubbed_arm_field() -> None:
    """Plant a field the scrub does not know about; the gate must catch it.

    This is the load-bearing proof: a future rung adding an arm-identifying
    config key gets a build failure rather than a silently-unblinded packet.
    """
    a = _MOD.scrub_record(_run_start(), "A", _E15A)
    b = _MOD.scrub_record(_run_start(), "B", _E15A)
    a["config"]["model"]["some_new_axis"] = "variant-one"
    b["config"]["model"]["some_new_axis"] = "variant-two"
    with pytest.raises(AssertionError, match="differ post-scrub"):
        _MOD.assert_blinded_parity({"A": a["config"], "B": b["config"]}, _E15A)


def test_parity_gate_ignores_the_disclosed_control() -> None:
    """Arm C's role is disclosed, so its config legitimately differs."""
    a = _MOD.scrub_record(_run_start(), "A", _E15A)
    b = _MOD.scrub_record(_run_start(), "B", _E15A)
    c = _MOD.scrub_record(_run_start(), "C", _E15A)
    c["config"]["model"]["extra_control_only_key"] = 1
    _MOD.assert_blinded_parity({"A": a["config"], "B": b["config"], "C": c["config"]}, _E15A)


def test_banned_scan_fires_on_a_planted_leak(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"run": "arm-A"}\n', encoding="utf-8")
    assert _MOD.banned_scan(clean, _E1.banned) == []
    leaky = tmp_path / "leaky.jsonl"
    leaky.write_text('{"run": "arm-A", "note": "Fisher init"}\n', encoding="utf-8")
    assert "fisher" in _MOD.banned_scan(leaky, _E1.banned)


def test_e15a_bans_its_own_axis_vocabulary(tmp_path: Path) -> None:
    leaky = tmp_path / "leaky.jsonl"
    leaky.write_text('{"config": {"model": {"embedding_freeze": "both"}}}\n', encoding="utf-8")
    assert "embedding_freeze" in _MOD.banned_scan(leaky, _E15A.banned)
    # ...and E1's banned list would NOT have caught it
    assert "embedding_freeze" not in _MOD.banned_scan(leaky, _E1.banned)


def test_select_captures_fails_loud_on_a_missing_seed(tmp_path: Path) -> None:
    """An incomplete packet still assembles and still looks complete.

    Silently changing n is the worst failure this script can have, so the
    absence of a seed is an error, never an omission.
    """
    for arm, seeds in (("e1-5a-input", "12"), ("e1-5a-both", "1234"), ("e1-fisher", "1234")):
        for seed in seeds:
            (tmp_path / f"{arm}-abcd1234-{arm}-s{seed}.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="no capture for"):
        _MOD.select_captures(tmp_path, _E15A)


def test_select_captures_ignores_other_rungs(tmp_path: Path) -> None:
    """One capture dir holds every rung; selection must take only this one's."""
    for arm in ("e1-5a-input", "e1-5a-both", "e1-fisher"):
        for seed in "1234":
            (tmp_path / f"{arm}-abcd1234-{arm}-s{seed}.jsonl").write_text("{}\n", encoding="utf-8")
    for noise in ("e1-random-aaaa1111-e1-random-s1", "e1-5c-prob-bbbb2222-e1-5c-prob-s1"):
        (tmp_path / f"{noise}.jsonl").write_text("{}\n", encoding="utf-8")
    selected = _MOD.select_captures(tmp_path, _E15A)
    assert len(selected) == 12
    assert {arm.label for _, arm, _ in selected} == {"A", "B", "C"}


def test_arm_resolution_prefers_the_longest_prefix() -> None:
    """``e1-5c-field`` must not shadow ``e1-5c-field-matched-total``.

    Both live in one capture directory, and the shorter name is a prefix of
    the longer — first-match-wins would assign every matched-total run to the
    plain field arm.
    """
    spec = _MOD.RungSpec(
        name="probe",
        arms=(
            _MOD.ArmSpec("A", "e1-5c-field"),
            _MOD.ArmSpec("B", "e1-5c-field-matched-total"),
        ),
        seeds=("1",),
        abstract="",
        preregistration="",
        sealed_slug="probe",
        sealed_mapping_note="",
        manifest_title="probe",
    )
    resolved = spec.resolve_arm("e1-5c-field-matched-total-1234abcd-x-s1.jsonl")
    assert resolved is not None
    assert resolved.label == "B"


def test_parity_exempts_keys_declared_per_arm_but_not_shared_ones() -> None:
    """E1's arms MUST differ at ``embedding_init`` — that token is the blind label.

    Caught by regenerating the filed E1 packet through the new engine
    (2026-08-22): the first parity rule flagged E1's own declared difference as
    a leak. A per-arm token is a declaration; anything else that differs is not.
    """
    a = _MOD.scrub_record(_run_start(), "A", _E1)
    b = _MOD.scrub_record(_run_start(), "B", _E1)
    assert a["config"]["model"]["embedding_init"] != b["config"]["model"]["embedding_init"]
    _MOD.assert_blinded_parity({"A": a["config"], "B": b["config"]}, _E1)

    # ...but E1.5a maps the SAME key to a shared token, so there it must match.
    a15 = _MOD.scrub_record(_run_start(), "A", _E15A)
    b15 = _MOD.scrub_record(_run_start(), "B", _E15A)
    b15["config"]["model"]["embedding_init"] = "init-tampered"
    with pytest.raises(AssertionError, match="differ post-scrub"):
        _MOD.assert_blinded_parity({"A": a15["config"], "B": b15["config"]}, _E15A)


def test_build_reader_strips_prose_and_shims_logging() -> None:
    source = (
        '"""Module docstring naming the e3 ladder."""\n'
        "from entropic_substrate.logging import configure_logging, logger\n\n\n"
        "def f() -> int:\n"
        '    """Docstring naming fisher."""\n'
        "    return 1\n"
    )
    reader = _MOD.build_reader(source)
    lowered = reader.lower()
    assert "e3" not in lowered.replace("prose", "")  # header prose is neutral
    assert "fisher" not in lowered
    assert "entropic_substrate" not in reader
    assert "def configure_logging" in reader
    assert "return 1" in reader
