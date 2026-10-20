"""Assemble a double-blind review packet for a rung (docs/peer-review-protocol.md).

Operator-side tool. It relabels the arms, SCRUBS arm-identifying fields from
every record of the captured JSONLs (filename relabeling alone is not
blinding: ``run_start`` carries the arm axis and every record carries
``run``), emits a self-contained reader with all prose stripped
(docstrings/comments carry project vocabulary), and writes ``manifest.txt``.
The arm mapping is written OUTSIDE the packet (``--sealed-out``) and
disclosed only after the review is filed.

Per-rung differences live in a ``RungSpec`` (the ``RUNGS`` registry below) —
arms, which config keys carry the arm axis, which records fields must be
dropped, the banned vocabulary, and the packet prose. The engine is shared;
selecting a rung is ``--rung``.

Fail-loud gates, all of which abort assembly:

1. **Banned-substring scan** over every emitted file. A packet that leaks arm
   identity is worse than no packet — the review it produces looks blind and
   is not.
2. **Completeness** — the captures must yield exactly one file per (arm,
   seed) named by the spec. A missing or duplicated pair is never silently
   dropped.
3. **Parity** — post-scrub, the *blinded* arms' ``run_start`` records must be
   identical outside the declared exemptions. This is the gate that catches a
   NEW arm-identifying field appearing in the schema: E1's arms were
   parameter-identical, but E1.5a's differ in ``params`` (85.33M untied vs
   85.13M tied), which alone would unblind the packet.
4. **Attestation presence** — for a rung declaring ``attestation_required``,
   every ``run_start`` must carry all three freeze-attestation keys. NOT a
   uniformity rule: the E1.5a A/C extension mixes pre- and post-attestation
   captures inside one arm, which is legitimate.

The freeze attestation (review finding MAJOR-1) is the one channel that is
DELIBERATELY not equalized: the scrub removes every freeze-axis config field,
so without it a reviewer cannot verify that the arms froze different parameter
subsets at all. ``trainable_param_count``/``frozen_param_count`` pass through
raw; ``frozen_mask_digest`` is rekeyed under a per-assembly HMAC salt written
only to ``--sealed-out``, because a raw sha256 over parameter names is
dictionary-attackable.

Layout (FOR-1609): the specs, the preregistration prose and ``RUNGS`` live in
the sibling ``review_packet_rungs.py``; this file keeps the engine and
``main()`` and re-exports the sibling's names, so ``_MOD.RUNGS`` and friends
resolve exactly as before. ``scripts/`` is not a package (no ``__init__.py``)
and ``tests/test_review_packet.py`` loads this file with
``spec_from_file_location``, which does NOT put ``scripts/`` on ``sys.path``;
a bare ``import review_packet_rungs`` would therefore work under
``python scripts/assemble_review_packet.py`` (the interpreter prepends the
script's directory) and fail under the loader. The mechanism is an explicit
``sys.path`` entry for this file's own directory, inserted before the sibling
import and only when absent. Measured 2026-10-20: both entry paths import the
sibling (``python scripts/assemble_review_packet.py --help`` from the repo
root, exit 0, and the file-path loader in the test module), and the line
counts are recorded in the FOR-1609 commit message.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import re
import secrets
import shutil
import subprocess  # nosec B404 -- ruff-normalize the packet only, fixed argv below
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from review_packet_rungs import (  # noqa: E402 -- needs the sys.path entry above
    BANNED_BASE,
    RUNGS,
    ArmSpec,
    RungSpec,
)

# Re-exported for the file-path loader (tests/test_review_packet.py reads these
# off the loaded module); the engine below uses them directly as well.
__all__ = ["BANNED_BASE", "RUNGS", "ArmSpec", "RungSpec", "main"]

_READER_HEADER = '"""Validated reader for the packet\'s JSONL run series.\n\nProse (docstrings/comments) was stripped from the original for blinding;\nlogic is unmodified. Parse the data ONLY through these functions.\n"""\n'
_LOGGING_SHIM = (
    "import logging\n"
    'logger = logging.getLogger("reader")\n'
    "def configure_logging() -> None:\n"
    "    logging.basicConfig(level=logging.INFO)\n"
)


def md5(path: Path) -> str:
    # integrity manifest, not crypto — declared via usedforsecurity
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def scrub_record(record: dict[str, Any], arm: str, spec: RungSpec) -> dict[str, Any]:
    """Map arm-identifying values to blind labels; drop arm-axis keys."""
    out = dict(record)
    if "run" in out:
        out["run"] = f"arm-{arm}"
    for key in spec.record_drop:
        out.pop(key, None)
    config = out.get("config")
    if isinstance(config, dict):
        config = json.loads(json.dumps(config))  # deep copy
        if "name" in config:
            config["name"] = f"arm-{arm}"
        model = config.get("model", {})
        for key, token in spec.value_scrub.items():
            if key in model:
                model[key] = token.format(arm=arm)
        for key in spec.model_drop:
            model.pop(key, None)
        out["config"] = config
    return out


def rekey_attestation(record: Mapping[str, Any], salt: str) -> dict[str, Any]:
    """Replace the raw frozen-parameter digest with an HMAC under ``salt``.

    The train-time digest is a sha256 over frozen parameter names and shapes
    (backbone.py ``parameter_attestation``). Passed through raw it is
    dictionary-attackable: a reviewer who guesses the architecture enumerates
    candidate frozen masks, hashes them, and reads the arm off the packet.
    HMAC under a per-assembly salt kills preimage enumeration while keeping
    the only structure the reviewer needs — equality: same arm and axis maps
    to the same value, different masks to different values.

    The salt lives ONLY in the sealed mapping file, outside the packet, so it
    is disclosed on the same schedule as the arm mapping itself.

    The counts pass through unmodified: the reviewer asked for counts, and
    coarse parameter integers identify no project. Absence of the digest is a
    no-op — pre-attestation captures (E1, E1.5a) carry none, and a rung whose
    arms MUST carry one says so via ``attestation_required``
    (:func:`assert_attestation_present`).
    """
    out = dict(record)
    if "frozen_mask_digest" not in out:
        return out
    raw = out["frozen_mask_digest"]
    if not isinstance(raw, str):
        # Present-but-malformed is the one shape that would reach the packet
        # unrekeyed: assert_uniform_attestation counts it as attested and the
        # parity gate exempts it, so nothing downstream would catch it.
        msg = f"non-string frozen_mask_digest {raw!r} — corrupt attestation, refusing to assemble"
        raise AssertionError(msg)
    out["frozen_mask_digest"] = hmac.new(
        salt.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return out


def assert_attestation_present(records: Mapping[str, Mapping[str, Any]], spec: RungSpec) -> None:
    """Under ``attestation_required``, every ``run_start`` must carry all three keys.

    Deliberately NOT a whole-packet uniformity rule. The E1.5a A/C extension
    mixes pre-attestation captures (seeds 1-4) with post-attestation reruns
    INSIDE one arm, so neither whole-packet nor per-arm uniformity is
    satisfiable there — requiring it would refuse a legitimate packet.

    A rung whose runs were ALL captured after the attestation landed declares
    ``attestation_required=True``, and a missing key is then an assembly
    error naming the offending run, never a silent omission.
    """
    if not spec.attestation_required:
        return
    incomplete = {
        name: sorted(set(RungSpec.ATTESTATION_KEYS) - set(record))
        for name, record in records.items()
        if not set(RungSpec.ATTESTATION_KEYS) <= set(record)
    }
    if incomplete:
        msg = (
            f"{spec.name}: attestation_required, but these runs are missing "
            f"attestation keys: {dict(sorted(incomplete.items()))}"
        )
        raise AssertionError(msg)


_StripNode = TypeVar("_StripNode", ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node: _StripNode) -> _StripNode:
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:] or [ast.Pass()]
        node.body = body
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


_PACKET_READER_EXCLUDED_FUNCTIONS = {
    "_build_parser",
    "main",
    "read_d1_terminal_defect",
}
_PACKET_READER_EXCLUDED_IMPORT_PREFIXES = (
    # p13-sweep, B2 row 7 (P12-D5 cut, E-4 door): the two entries this tuple
    # carried at the old prefix are rewritten to their landed addresses.
    # They are pinned verbatim by tests/test_cli_surface.py's
    # _FROZEN_LITERAL_SITES_PATH.
    "entropic_substrate.ladder.capture.reader.d1_",
    "entropic_substrate.ladder.capture.records.d1_",
)


def _is_shim_source(source: str) -> bool:
    """True if ``source`` is (structurally) a generated alias shim: the
    ``import <target> as _target`` / ``sys.modules[__name__] = _target``
    rebind the pre-p13 shim generator emitted (the generator and every
    emitted shim retired in p13's breaking commit; this guard still
    recognizes the shape on sight, in case a future move reintroduces it),
    with or without its leading entrypoint ``if __name__ == "__main__":``
    guard.

    A reader built from a source this shallow degrades to a no-op that
    still "succeeds" (FOR-1524, MEASURED 2026-09-20: p3-quarantine's move
    made ``src/entropic_substrate/telemetry.py`` exactly this shape, and
    the bare ``import entropic_substrate.__legacy.telemetry as _target``
    line survives ``build_reader``'s ImportFrom-only filter, leaking the
    banned ``entropic`` token into the packet instead of failing loud).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    body = tree.body
    if body and isinstance(body[0], ast.If):
        body = body[1:]
    if len(body) != 3:
        return False
    s0, s1, s2 = body
    imports_sys = (
        isinstance(s0, ast.Import)
        and len(s0.names) == 1
        and s0.names[0].name == "sys"
        and s0.names[0].asname is None
    )
    imports_target = (
        isinstance(s1, ast.Import) and len(s1.names) == 1 and s1.names[0].asname == "_target"
    )
    rebinds = (
        isinstance(s2, ast.Assign)
        and len(s2.targets) == 1
        and isinstance(s2.targets[0], ast.Subscript)
        and isinstance(s2.targets[0].value, ast.Attribute)
        and s2.targets[0].value.attr == "modules"
        and isinstance(s2.targets[0].value.value, ast.Name)
        and s2.targets[0].value.value.id == "sys"
        and isinstance(s2.value, ast.Name)
        and s2.value.id == "_target"
    )
    return imports_sys and imports_target and rebinds


def build_reader(telemetry_source: str, *, source_path: Path | None = None) -> str:
    """Strip prose and D1-only imports from the legacy packet reader.

    Also drops the operator CLI (``main``/``_build_parser``/``__main__``
    block) — its argparse prog/help strings carry repository names, and the
    reviewer parses through the functions, never argv. D1 evidence readers
    live in separate modules and are not dependencies of the legacy E-series
    packet reader; copying their package imports would both break standalone
    execution and leak repository identity into a blinded packet.

    Refuses a shim-form ``telemetry_source`` outright (FOR-1524): a literal
    alias shim can never become a packet reader, whether reached through the
    default resolution or an explicit ``--telemetry`` override.
    """
    if _is_shim_source(telemetry_source):
        where = f" ({source_path})" if source_path is not None else ""
        msg = (
            f"refusing to build a review-packet reader from a generated alias "
            f"shim{where} -- it has no real telemetry code to read; point "
            "--telemetry at the real module (FOR-1524)"
        )
        raise ValueError(msg)
    tree = _DocstringStripper().visit(ast.parse(telemetry_source))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.FunctionDef) and node.name in _PACKET_READER_EXCLUDED_FUNCTIONS
        )
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith(_PACKET_READER_EXCLUDED_IMPORT_PREFIXES)
        )
        and not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        # __all__ would dangle names of the dropped CLI (ruff F822)
        and not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        )
    ]
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    # p13-sweep, B2 row 18: the old-spelling strip
    # ("from entropic_substrate.logging import configure_logging, logger")
    # retired as dead code -- no source under the retired namespace exists
    # to produce that string any more. The landed spelling below is the
    # only one any reader source can carry now.
    code = code.replace(
        "from entropic_substrate.kernel.runtime.logging import configure_logging, logger",
        _LOGGING_SHIM,
    )
    return _READER_HEADER + code + "\n"


# Long pure-hex runs are masked before the substring scan. Every token in
# the banned vocabulary except "e3" carries a non-hex character and cannot
# occur inside one; "e3" inside a digest is two coincidental nibbles, not a
# lexical reference — and at 64 hex chars it appears with p ≈ 22% (63
# positions × 1/256), so digest-bearing packets (data_order_digest, the
# rekeyed frozen_mask_digest) would spuriously abort more often than not.
_HEX_RUN = re.compile(r"[0-9a-f]{32,}")


def banned_scan(path: Path, banned: Sequence[str]) -> list[str]:
    # The "#" placeholder keeps masked runs from splicing their neighbours
    # into a banned word ("e" + masked-run + "3" stays "e#3", never "e3").
    text = _HEX_RUN.sub("#", path.read_text(encoding="utf-8").lower())
    return [word for word in banned if word in text]


def select_captures(captured: Path, spec: RungSpec) -> list[tuple[Path, ArmSpec, str]]:
    """Resolve capture files to (path, arm, seed), fail-loud on incompleteness.

    A capture directory accumulates every rung's runs, so files that match no
    arm of THIS rung are skipped — but the selection is then required to be
    exactly one file per (arm, seed). Silent under-selection is the failure
    mode this gate exists to prevent: a packet missing a seed still assembles,
    still looks complete, and quietly changes n.
    """
    chosen: dict[tuple[str, str], tuple[Path, ArmSpec, str]] = {}
    for src in sorted(captured.glob("*.jsonl")):
        arm = spec.resolve_arm(src.name)
        if arm is None:
            continue
        seed = src.stem.rsplit("-s", 1)[-1]
        if seed not in spec.seeds:
            continue
        key = (arm.label, seed)
        if key in chosen:
            msg = f"duplicate capture for arm {arm.label} seed {seed}: {chosen[key][0].name}, {src.name}"
            raise AssertionError(msg)
        chosen[key] = (src, arm, seed)

    missing = [
        f"{arm.label}-seed{seed}"
        for arm in spec.arms
        for seed in spec.seeds
        if (arm.label, seed) not in chosen
    ]
    if missing:
        msg = f"{spec.name}: no capture for {missing}"
        raise AssertionError(msg)
    return [chosen[(arm.label, seed)] for arm in spec.arms for seed in spec.seeds]


def assert_blinded_parity(records: Mapping[str, dict[str, Any]], spec: RungSpec) -> None:
    """Post-scrub, blinded arms must carry identical run_start RECORDS.

    This is the gate that catches a NEW arm-identifying field entering the
    schema between rungs. Comparing whole records (minus declared exemptions)
    rather than an allow-list means a field nobody thought about is a FAILURE,
    not an omission. The comparison covers the RECORD, not only its ``config``
    — the freeze attestation (MAJOR-1) enters at record top level, and so did
    ``params``, the one field E1's packet never had to scrub.

    A key whose ``value_scrub`` token interpolates ``{arm}`` is a DECLARED
    difference — that token IS the blind label (E1's ``embedding_init`` ->
    ``init-A``/``init-B``) — so it is exempt. A key mapped to a shared token,
    or not mapped at all, must match. That asymmetry is the whole rule: the
    per-arm tokens are the only differences anyone declared, and every other
    difference is a leak by definition.

    Two record-level exemptions are structural, applying to every rung:
    ``run`` (scrubbed to ``arm-<label>``, the record-level analogue of
    ``config.name``) and :attr:`RungSpec.ATTESTATION_KEYS`, whose whole
    purpose is to differ between arms.
    """
    blinded = [arm.label for arm in spec.arms if arm.blinded]
    if len(blinded) < 2:
        return

    per_arm = {key for key, token in spec.value_scrub.items() if "{arm}" in token}
    record_exempt = ("run", *RungSpec.ATTESTATION_KEYS)

    def strip(record: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in record.items() if k not in record_exempt}
        cfg = out.get("config")
        if isinstance(cfg, dict):
            cfg = {k: v for k, v in cfg.items() if k not in spec.parity_exempt}
            model = cfg.get("model")
            if isinstance(model, dict) and per_arm:
                cfg["model"] = {k: v for k, v in model.items() if k not in per_arm}
            out["config"] = cfg
        return out

    reference_label = blinded[0]
    reference = strip(records[reference_label])
    for label in blinded[1:]:
        other = strip(records[label])
        if other != reference:
            diff = sorted(
                k
                for k in set(reference) | set(other)
                if reference.get(k, "<absent>") != other.get(k, "<absent>")
            )
            detail = {k: (reference.get(k, "<absent>"), other.get(k, "<absent>")) for k in diff}
            msg = (
                f"blinded arms {reference_label} and {label} differ post-scrub "
                f"at {diff} -> {detail}; these fields identify the arm"
            )
            raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=sorted(RUNGS), default="e1")
    parser.add_argument("--captured", type=Path, default=Path("results/captured"))
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="defaults to the real module behind "
        "entropic_substrate.ladder.capture.reader.telemetry, "
        "resolved THROUGH the module (not a literal repo path) so it follows "
        "the sys.modules rebind to wherever the module actually lives "
        "(FOR-1524)",
    )
    parser.add_argument("--packet-out", type=Path, required=True)
    parser.add_argument("--sealed-out", type=Path, required=True)
    parser.add_argument("--run-times", type=Path, required=True, help="run_id -> times JSON")
    parser.add_argument("--prereg-commit", required=True, help="'<sha> <iso-utc>'")
    parser.add_argument("--amendment-commit", default=None, help="'<sha> <iso-utc>'")
    parser.add_argument(
        "--ext-prereg-commit",
        default=None,
        help="'<sha> <iso-utc>' — include the extension registration (n=9 packet)",
    )
    parser.add_argument(
        "--precedence-commit",
        default=None,
        help="'<sha> <iso-utc>' of the clause-precedence declaration",
    )
    parser.add_argument(
        "--abandoned-note",
        action="append",
        default=[],
        help="operator disclosure line(s) for abandoned/incomplete runs "
        "(the complete-run-ledger requirement)",
    )
    args = parser.parse_args()
    spec = RUNGS[args.rung]

    telemetry_path = args.telemetry
    if telemetry_path is None:
        # Resolve THROUGH the module, never a literal repo path: this
        # stays correct across a later structural move on its own, and
        # historically (pre-p13) also followed the shim generator's
        # sys.modules rebind to wherever the real target module lived
        # (FOR-1524). The shim generator retired in p13; resolving through
        # the module is kept as the general-case default regardless.
        import entropic_substrate.ladder.capture.reader.telemetry as _telemetry

        telemetry_path = Path(_telemetry.__file__)

    packet = args.packet_out
    if packet.exists():
        shutil.rmtree(packet)
    (packet / "data").mkdir(parents=True)

    run_times = json.loads(args.run_times.read_text(encoding="utf-8"))

    (packet / "abstract.md").write_text(spec.abstract, encoding="utf-8")
    prereg_text = spec.preregistration
    if args.ext_prereg_commit and spec.extension_registration:
        prereg_text += spec.extension_registration
    (packet / "preregistration.md").write_text(prereg_text, encoding="utf-8")
    (packet / "reader.py").write_text(
        build_reader(telemetry_path.read_text(encoding="utf-8"), source_path=telemetry_path),
        encoding="utf-8",
    )

    # One fresh salt per assembly: two packets over the same runs must not
    # share rekeyed digests, or disclosing one sealed mapping de-blinds the
    # other. Written only to --sealed-out, never into the packet.
    packet_salt = secrets.token_hex(32)

    sealed_rows: list[str] = []
    time_rows: list[str] = []
    start_records: dict[str, dict[str, Any]] = {}
    attested_starts: dict[str, dict[str, Any]] = {}
    for src, arm, seed in select_captures(args.captured, spec):
        dst = packet / "data" / f"{arm.label}-seed{seed}.jsonl"
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                scrubbed = scrub_record(record, arm.label, spec)
                if record.get("event") == "run_start":
                    scrubbed = rekey_attestation(scrubbed, packet_salt)
                    attested_starts[f"{arm.label}-seed{seed}"] = scrubbed
                    start_records.setdefault(arm.label, scrubbed)
                fout.write(json.dumps(scrubbed) + "\n")
        run_id8 = src.name[len(arm.run_name) + 1 :][:8]
        full = next((k for k in run_times if k.startswith(run_id8)), None)
        if full is None:
            msg = f"no tracking-server times for {src.name} (id prefix {run_id8})"
            raise LookupError(msg)
        for stamp_key in ("start", "end"):
            stamp = run_times[full][stamp_key]
            iso = datetime.fromtimestamp(stamp / 1000, tz=UTC).isoformat()
            time_rows.append(f"# {dst.name}  {stamp_key}={iso}")
        sealed_rows.append(f"| {src.name} | {md5(src)} | {dst.name} | {md5(dst)} | run_id {full} |")

    assert_attestation_present(attested_starts, spec)
    assert_blinded_parity(start_records, spec)

    # Hook-stability: the repo's lint hooks rewrite any committed .py — if
    # that happened AFTER hashing, the manifest would invalidate the packet
    # (measured on the first commit attempt, 2026-08-18: ruff applied 9
    # autofixes to reader.py post-manifest). Normalize BEFORE hashing.
    # nosec B603 B607 -- fixed argv, packet path from our own CLI arg
    subprocess.run(["ruff", "format", "--quiet", str(packet)], check=True)  # nosec B603 B607
    subprocess.run(  # nosec B603 B607
        ["ruff", "check", "--fix-only", "--quiet", str(packet)], check=True
    )

    files = sorted(p for p in packet.rglob("*") if p.is_file())
    leaks = {str(p): banned_scan(p, spec.banned) for p in files if banned_scan(p, spec.banned)}
    if leaks:
        msg = f"identity leak(s) in packet files: {leaks}"
        raise AssertionError(msg)

    manifest_lines = [f"# {spec.manifest_title}", "#", "# md5  file"]
    manifest_lines += [f"{md5(p)}  {p.relative_to(packet)}" for p in files]
    manifest_lines += [
        "#",
        "# Registration commits (UTC):",
        f"# preregistration committed: {args.prereg_commit}",
        *(
            [f"# amendment 1 (epsilon) committed: {args.amendment_commit}"]
            if args.amendment_commit
            else []
        ),
        *(
            [f"# extension registration committed: {args.ext_prereg_commit}"]
            if args.ext_prereg_commit
            else []
        ),
        *(
            [f"# clause precedence committed: {args.precedence_commit}"]
            if args.precedence_commit
            else []
        ),
        "#",
        "# Run times, operator-supplied from the tracking server (UTC).",
        "# These are asserted metadata, not raw record content:",
        *time_rows,
        *(
            ["#", "# Complete run ledger — abandoned/incomplete runs (operator-declared):"]
            + [f"# {note}" for note in args.abandoned_note]
            if args.abandoned_note
            else []
        ),
    ]
    (packet / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    args.sealed_out.parent.mkdir(parents=True, exist_ok=True)
    args.sealed_out.write_text(
        "---\n"
        f"name: {spec.sealed_slug}\n"
        f'description: "SEALED arm mapping for the {spec.name} review packet — '
        'disclose only after the review is filed (docs/peer-review-protocol.md)"\n'
        "metadata:\n  type: project\n---\n\n"
        f"# SEALED — {spec.name} review packet arm mapping\n\n"
        f"{spec.sealed_mapping_note}\n"
        "## Freeze-attestation salt\n\n"
        "HMAC-SHA256 key the packet's `frozen_mask_digest` values were rekeyed\n"
        "under; recomputing a digest from a candidate frozen-parameter mask "
        "needs it.\nFresh per assembly, so it de-blinds THIS packet and no "
        "other.\n\n"
        f"    {packet_salt}\n\n"
        "| capture file | capture md5 | packet file | packet md5 | run |\n"
        "|---|---|---|---|---|\n" + "\n".join(sealed_rows) + "\n",
        encoding="utf-8",
    )
    print(f"packet: {packet} ({len(files)} files) | sealed: {args.sealed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
