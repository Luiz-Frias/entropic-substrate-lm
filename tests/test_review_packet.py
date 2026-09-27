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

import hashlib
import hmac
import importlib.util
import json
import re
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
    _MOD.assert_blinded_parity({"A": a, "B": b}, _E15A)


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
        _MOD.assert_blinded_parity({"A": a, "B": b}, _E15A)


def test_parity_gate_ignores_the_disclosed_control() -> None:
    """Arm C's role is disclosed, so its config legitimately differs."""
    a = _MOD.scrub_record(_run_start(), "A", _E15A)
    b = _MOD.scrub_record(_run_start(), "B", _E15A)
    c = _MOD.scrub_record(_run_start(), "C", _E15A)
    c["config"]["model"]["extra_control_only_key"] = 1
    _MOD.assert_blinded_parity({"A": a, "B": b, "C": c}, _E15A)


def test_banned_scan_fires_on_a_planted_leak(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"run": "arm-A"}\n', encoding="utf-8")
    assert _MOD.banned_scan(clean, _E1.banned) == []
    leaky = tmp_path / "leaky.jsonl"
    leaky.write_text('{"run": "arm-A", "note": "Fisher init"}\n', encoding="utf-8")
    assert "fisher" in _MOD.banned_scan(leaky, _E1.banned)


def test_banned_scan_ignores_hex_digests_but_not_prose(tmp_path: Path) -> None:
    """A 64-char hex value contains "e3" with p ≈ 22% (63 positions × 1/256),
    so digest-bearing packets (data_order_digest, rekeyed frozen_mask_digest)
    would spuriously abort assembly more often than not. Every banned token
    except "e3" carries a non-hex character and CANNOT occur inside a pure
    hex run — masking long hex runs loses no lexical detection."""
    hex_with_e3 = "ab" * 15 + "e3" + "cd" * 16  # 64 hex chars containing "e3"
    assert "e3" in hex_with_e3
    hexy = tmp_path / "hexy.jsonl"
    hexy.write_text(json.dumps({"data_order_digest": hex_with_e3}) + "\n", encoding="utf-8")
    assert _MOD.banned_scan(hexy, _E1.banned) == []
    # the same token OUTSIDE a hex run still trips
    prose = tmp_path / "prose.jsonl"
    prose.write_text('{"note": "compared against e3 anchors"}\n', encoding="utf-8")
    assert "e3" in _MOD.banned_scan(prose, _E1.banned)
    # a short hex-looking fragment is NOT masked (only runs >= 32 chars)
    short = tmp_path / "short.jsonl"
    short.write_text('{"note": "deadbeefe3"}\n', encoding="utf-8")
    assert "e3" in _MOD.banned_scan(short, _E1.banned)


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
    _MOD.assert_blinded_parity({"A": a, "B": b}, _E1)

    # ...but E1.5a maps the SAME key to a shared token, so there it must match.
    a15 = _MOD.scrub_record(_run_start(), "A", _E15A)
    b15 = _MOD.scrub_record(_run_start(), "B", _E15A)
    b15["config"]["model"]["embedding_init"] = "init-tampered"
    with pytest.raises(AssertionError, match="differ post-scrub"):
        _MOD.assert_blinded_parity({"A": a15, "B": b15}, _E15A)


def test_build_reader_strips_prose_and_shims_logging() -> None:
    source = (
        '"""Module docstring naming the e3 ladder."""\n'
        "from entropic_substrate.kernel.runtime.logging import configure_logging, logger\n\n\n"
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


# --------------------------------------------------------------------------
# Unblinded freeze attestation (review finding MAJOR-1). The packet scrubs
# every freeze-axis config field, so nothing in it lets a reviewer verify the
# arms froze DIFFERENT subsets. The attestation is the channel that does —
# and a RAW sha256 over parameter names is dictionary-attackable (guess the
# architecture, enumerate candidate masks, hash, compare), so it is rekeyed
# under a per-assembly HMAC salt that lives only in the sealed mapping.
# --------------------------------------------------------------------------


def _attested(digest: str, trainable: int = 85_130_000, frozen: int = 0) -> dict[str, Any]:
    record = _run_start()
    record["trainable_param_count"] = trainable
    record["frozen_param_count"] = frozen
    record["frozen_mask_digest"] = digest
    return record


def test_rekey_replaces_the_digest_and_preserves_equality_structure() -> None:
    salt = "0" * 64
    same_a = _MOD.rekey_attestation(_attested("aaaa"), salt)
    same_b = _MOD.rekey_attestation(_attested("aaaa"), salt)
    other = _MOD.rekey_attestation(_attested("bbbb"), salt)

    # the preimage never reaches the packet...
    assert same_a["frozen_mask_digest"] != "aaaa"
    # ...but it IS the HMAC of it, so the operator can verify from the sealed salt
    assert (
        same_a["frozen_mask_digest"] == hmac.new(salt.encode(), b"aaaa", hashlib.sha256).hexdigest()
    )
    # equality structure survives: same mask -> same value, different -> different
    assert same_a["frozen_mask_digest"] == same_b["frozen_mask_digest"]
    assert same_a["frozen_mask_digest"] != other["frozen_mask_digest"]


def test_rekey_uses_a_different_salt_per_assembly() -> None:
    """Two packets of the same runs must not share rekeyed values — otherwise
    one disclosed sealed mapping de-blinds every other packet."""
    a = _MOD.rekey_attestation(_attested("aaaa"), "0" * 64)
    b = _MOD.rekey_attestation(_attested("aaaa"), "1" * 64)
    assert a["frozen_mask_digest"] != b["frozen_mask_digest"]


def test_rekey_passes_the_counts_through_unmodified() -> None:
    """The reviewer asked for counts; coarse integers identify no project."""
    out = _MOD.rekey_attestation(_attested("aaaa", trainable=85_121_808, frozen=8_192), "0" * 64)
    assert out["trainable_param_count"] == 85_121_808
    assert out["frozen_param_count"] == 8_192


def test_rekey_is_a_no_op_on_a_pre_attestation_record() -> None:
    """E1 and E1.5a were captured before the attestation existed."""
    record = _run_start()
    assert _MOD.rekey_attestation(record, "0" * 64) == record


def test_rekey_fails_loud_on_a_non_string_digest() -> None:
    """The one branch that could pass a raw digest through unrekeyed.

    A present-but-malformed value is counted as attested by
    ``assert_uniform_attestation`` and exempted by the parity gate, so
    nothing downstream would catch it — and this is a blinding path, where
    silence is the expensive failure.
    """
    record = _attested("aaaa")
    record["frozen_mask_digest"] = 12345
    with pytest.raises(AssertionError, match="non-string frozen_mask_digest"):
        _MOD.rekey_attestation(record, "0" * 64)


def test_parity_gate_ignores_the_attestation_keys() -> None:
    """The attestation keys are EXPECTED to differ between arms — that is the
    whole point of the channel. Every other record-level difference is a leak.
    """
    a = _MOD.scrub_record(_attested("aaaa", frozen=8_192), "A", _E15A)
    b = _MOD.scrub_record(_attested("bbbb", frozen=16_384), "B", _E15A)
    _MOD.assert_blinded_parity({"A": a, "B": b}, _E15A)


def test_parity_gate_fires_on_an_unscrubbed_record_level_field() -> None:
    """Record-level parity is not attestation-shaped: a NEW top-level field
    that differs is still a leak, exempted keys aside."""
    a = _MOD.scrub_record(_attested("aaaa"), "A", _E15A)
    b = _MOD.scrub_record(_attested("bbbb"), "B", _E15A)
    a["some_new_record_field"] = "variant-one"
    b["some_new_record_field"] = "variant-two"
    with pytest.raises(AssertionError, match="differ post-scrub"):
        _MOD.assert_blinded_parity({"A": a, "B": b}, _E15A)


def _probe_spec(**overrides: Any) -> Any:
    fields = {
        "name": "probe",
        "arms": (_MOD.ArmSpec("A", "probe-one"), _MOD.ArmSpec("B", "probe-two")),
        "seeds": ("1",),
        "abstract": "",
        "preregistration": "",
        "sealed_slug": "probe",
        "sealed_mapping_note": "",
        "manifest_title": "probe",
    }
    fields.update(overrides)
    return _MOD.RungSpec(**fields)


def test_required_attestation_fails_loud_on_a_missing_key() -> None:
    """A rung whose runs were ALL captured post-attestation says so, and a
    run missing the channel is then an assembly error naming that run."""
    records = {"A-seed1.jsonl": _attested("aaaa"), "B-seed1.jsonl": _run_start()}
    with pytest.raises(AssertionError, match=re.escape("B-seed1.jsonl")):
        _MOD.assert_attestation_present(records, _probe_spec(attestation_required=True))


def test_required_attestation_rejects_a_partially_attested_record() -> None:
    """All three keys or none of the run: counts without a digest cannot
    distinguish the two variants (their frozen counts are equal)."""
    partial = _attested("aaaa")
    del partial["frozen_mask_digest"]
    with pytest.raises(AssertionError, match="frozen_mask_digest"):
        _MOD.assert_attestation_present(
            {"A-seed1.jsonl": partial}, _probe_spec(attestation_required=True)
        )


def test_mixed_attestation_passes_when_not_required() -> None:
    """The DEFAULT, and it is load-bearing: the E1.5a A/C extension mixes
    pre-attestation captures with post-attestation reruns inside ONE arm, so
    a uniformity rule would refuse a legitimate packet."""
    records = {"A-seed1.jsonl": _attested("aaaa"), "A-seed2.jsonl": _run_start()}
    _MOD.assert_attestation_present(records, _probe_spec())
    for rung in (_E1, _E15A):
        assert rung.attestation_required is False


def test_assembly_writes_the_rekey_salt_only_to_the_sealed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: the salt ``main`` rekeys under is the salt it seals.

    The unit tests above verify the HMAC against a salt CONSTANT; only a real
    assembly proves the two halves are the same value — a mismatch would seal
    a key that verifies nothing, and no other gate would notice.
    """
    monkeypatch.setitem(_MOD.RUNGS, "probe", _probe_spec(attestation_required=True))
    # Salt PINNED, not random: the banned scan is a plain substring test and
    # "e3" is two characters, so a random 64-char hex digest trips it about
    # 22% of the time (63 positions x 1/256). That hazard predates this change
    # — every packet already carries sha256 data_order_digest values — but a
    # random salt would make this test itself a coin flip. Measured: under
    # this salt neither rekeyed value contains a banned substring.
    salt = "0" * 64
    monkeypatch.setattr(_MOD.secrets, "token_hex", lambda _n: salt)

    captured = tmp_path / "captured"
    captured.mkdir()
    raw_digests = {"probe-one": "a" * 64, "probe-two": "b" * 64}
    for run_name, raw in raw_digests.items():
        record = {
            "event": "run_start",
            "run": run_name,
            "config": {"name": run_name, "model": {"d_model": 32}},
            "trainable_param_count": 100,
            "frozen_param_count": 8,
            "frozen_mask_digest": raw,
        }
        path = captured / f"{run_name}-abcd1234-{run_name}-s1.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    times = tmp_path / "times.json"
    run_id_full = "abcd1234deadbeef"  # fixture run id, not a credential  # pragma: allowlist secret
    times.write_text(
        json.dumps({run_id_full: {"start": 1_700_000_000_000, "end": 1_700_000_060_000}}),
        encoding="utf-8",
    )
    packet = tmp_path / "packet"
    sealed = tmp_path / "sealed.md"
    # No --telemetry override: this exercises the CLI's DEFAULT telemetry
    # source, same as test_default_telemetry_source_resolves_through_the_module
    # below. Pinning an explicit override here (the old form pointed at the
    # literal src path, which is now a generated shim) would decouple this
    # test from the default-resolution fix -- FOR-1524's guard tooth
    # ("revert the default -> both tests RED") depends on this test using
    # the default too.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_review_packet.py",
            "--rung=probe",
            f"--captured={captured}",
            f"--packet-out={packet}",
            f"--sealed-out={sealed}",
            f"--run-times={times}",
            "--prereg-commit=abc123 2026-01-01T00:00:00+00:00",
        ],
    )
    assert _MOD.main() == 0

    # the sealed file is where the key is disclosed — and it must be the SAME
    # key the assembly rekeyed under, or it verifies nothing
    assert salt in sealed.read_text(encoding="utf-8")
    emitted = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((packet / "data").glob("*.jsonl"))
    }
    assert (
        emitted["A-seed1.jsonl"]["frozen_mask_digest"]
        == hmac.new(salt.encode(), raw_digests["probe-one"].encode(), hashlib.sha256).hexdigest()
    )
    assert (
        emitted["B-seed1.jsonl"]["frozen_mask_digest"]
        == hmac.new(salt.encode(), raw_digests["probe-two"].encode(), hashlib.sha256).hexdigest()
    )
    # counts survive the assembly untouched
    assert emitted["A-seed1.jsonl"]["frozen_param_count"] == 8

    # ...and neither the preimage nor the key is anywhere in the packet
    blob = "".join(p.read_text(encoding="utf-8") for p in packet.rglob("*") if p.is_file())
    assert salt not in blob
    for raw in raw_digests.values():
        assert raw not in blob


def test_default_telemetry_source_resolves_through_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FOR-1524: the CLI's default ``--telemetry`` must resolve to the real
    module, never the literal ``src/entropic_substrate/telemetry.py`` path.

    p3-quarantine's move made that literal path a 4-statement generated
    shim (``import sys`` / ``import ...__legacy.telemetry as _target`` /
    ``sys.modules[__name__] = _target``). ``build_reader`` only strips
    known prose and D1-only ``ImportFrom`` nodes; a bare ``ast.Import`` of
    ``entropic_substrate.__legacy.telemetry`` survives that filter and
    lands in ``reader.py`` verbatim, and ``entropic`` is itself a banned
    identity token -- so the packet fails closed on the leak (an
    AssertionError from ``banned_scan``, not a silent no-op reader). This
    pins that the DEFAULT is fixed at the source: it must resolve through
    ``entropic_substrate.telemetry`` (whose sys.modules rebind makes
    ``__file__`` the real ``__legacy/telemetry.py``), so the produced
    reader carries real telemetry code and no repository-identifying
    dotted name.
    """
    monkeypatch.setitem(_MOD.RUNGS, "probe", _probe_spec(attestation_required=False))
    captured = tmp_path / "captured"
    captured.mkdir()
    record = {
        "event": "run_start",
        "run": "probe-one",
        "config": {"name": "probe-one", "model": {"d_model": 32}},
        "trainable_param_count": 100,
        "frozen_param_count": 8,
    }
    (captured / "probe-one-abcd1234-probe-one-s1.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    record2 = dict(record, run="probe-two")
    (captured / "probe-two-abcd1234-probe-two-s1.jsonl").write_text(
        json.dumps(record2) + "\n", encoding="utf-8"
    )

    times = tmp_path / "times.json"
    run_id_full = "abcd1234deadbeef"  # fixture run id, not a credential  # pragma: allowlist secret
    times.write_text(
        json.dumps({run_id_full: {"start": 1_700_000_000_000, "end": 1_700_000_060_000}}),
        encoding="utf-8",
    )
    packet = tmp_path / "packet"
    sealed = tmp_path / "sealed.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assemble_review_packet.py",
            "--rung=probe",
            f"--captured={captured}",
            f"--packet-out={packet}",
            f"--sealed-out={sealed}",
            f"--run-times={times}",
            "--prereg-commit=abc123 2026-01-01T00:00:00+00:00",
        ],
    )
    assert _MOD.main() == 0

    reader = (packet / "reader.py").read_text(encoding="utf-8")
    assert "sys.modules[__name__]" not in reader
    assert "__legacy" not in reader
    assert "entropic_substrate" not in reader
    assert "TelemetryGap" in reader  # a real telemetry symbol, not a shim


def test_build_reader_pre_ruff_output_carries_no_leaked_identity() -> None:
    """FOR-1556 remedy (P13-FOR-1556): re-point the identity-leak guard to
    ``build_reader``'s PRE-ruff output, over the REAL telemetry module,
    resolved the same way ``main()`` resolves it.

    The assertion above (``:556``, over ``packet / "reader.py"`` AFTER
    ``main()`` runs ``ruff format`` + ``ruff check --fix-only``) is real
    coverage of the shipped artifact and stays exactly as it is. But it is
    NOT a guard against every leak class: an ``entropic_substrate.*`` import
    ``build_reader`` fails to strip (e.g. a ``_PACKET_READER_EXCLUDED_IMPORT_PREFIXES``
    entry that no longer matches a real import) can still leave the packet
    clean, because the excluded functions (``main``/``_build_parser``/
    ``read_d1_terminal_defect``) are the only consumers of those D1 imports:
    once they are stripped, the leaked import is textbook ruff F401
    "unused import" and ``--fix-only`` silently deletes it before line 556
    ever reads the file. MEASURED in-branch (mutation reverted, never
    committed), both instruments, all three FOR-1556 classes -- dropping
    either ``_PACKET_READER_EXCLUDED_IMPORT_PREFIXES`` entry, or neutralizing
    the ``_LOGGING_SHIM`` substitution, leaves the OLD post-ruff instrument
    green (blind) while THIS pre-ruff instrument goes RED on every one:
    no class is left green under the new instrument (P13-TEETH-HONESTY has
    nothing to retract). The baseline (no mutation) is green under both.

    This is additive: it does not replace or weaken the :556 assertion, and
    it changes no production behavior -- ``_PACKET_READER_EXCLUDED_IMPORT_PREFIXES``
    and ``_is_shim_source`` are read here, never redefined.
    """
    import entropic_substrate.ladder.capture.reader.telemetry as _telemetry

    source = Path(_telemetry.__file__).read_text(encoding="utf-8")
    reader = _MOD.build_reader(source, source_path=Path(_telemetry.__file__))
    assert "entropic_substrate" not in reader
    assert "__legacy" not in reader
    assert "TelemetryGap" in reader  # a real telemetry symbol, not a shim


def test_build_reader_refuses_a_literal_shim_source(tmp_path: Path) -> None:
    """The refusal guard: a shim-form source can never become a packet
    reader again, even via an explicit ``--telemetry`` override."""
    shim_path = tmp_path / "telemetry.py"
    # P13-EXCEPTIONS (addendum 3): the exact 3-statement rebind shape
    # `_is_shim_source` detects is kept; only the dotted target changes
    # (the generator and its `__legacy` shim form retire in B2, but the
    # refusal guard must still recognize the shape on sight).
    shim_path.write_text(
        "import sys\n"
        "import entropic_substrate.ladder.capture.reader.telemetry as _target\n"
        "sys.modules[__name__] = _target\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape(str(shim_path))):
        _MOD.build_reader(shim_path.read_text(encoding="utf-8"), source_path=shim_path)
