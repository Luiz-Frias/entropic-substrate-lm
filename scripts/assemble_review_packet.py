"""Assemble the double-blind E1 review packet (docs/peer-review-protocol.md).

Operator-side tool. It relabels the arms A/B (A = the registered-favored
arm), SCRUBS arm-identifying fields from every record of the captured JSONLs
(filename relabeling alone is not blinding: ``run_start`` carries
``embedding_init`` and every record carries ``run``), emits a self-contained
reader with all prose stripped (docstrings/comments carry project vocabulary),
and writes ``manifest.txt``. The A/B mapping is written OUTSIDE the packet
(``--sealed-out``) and disclosed only after the review is filed.

Fail-loud gate: every emitted packet file is scanned for BANNED substrings;
any hit aborts assembly. A packet that leaks arm identity is worse than no
packet — the review it produces looks blind and is not.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess  # nosec B404 -- ruff-normalize the packet only, fixed argv below
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

# Arm mapping: A is the registered-favored arm by protocol construction —
# the packet restates the direction as "A better than B"; what stays sealed
# is WHICH initialization A is.
ARM_MAP = {"e1-fisher": "A", "e1-random": "B"}
INIT_MAP = {"fisher_sqrt": "init-A", "random_sphere": "init-B"}

# Substrings that must not survive into any packet file (case-insensitive).
BANNED = ("entropic", "fisher", "random_sphere", "sqrt", "e1-", "e3", "ladder", "anchor")

ABSTRACT = """\
# Abstract

Paired comparison of two embedding initializations (A, B) for a byte-level
language model trained on enwik8 (~85M parameters, 20,000 steps). Five seed
pairs (seeds 1-5); within a pair both arms share the seed, data order, and
all non-embedding initialization. Primary metric: final nats/byte on a held
out split, lower is better. The registered hypothesis is that arm A attains
lower nats/byte than arm B. This packet contains the pre-registration, the
captured per-run JSONL series, a self-contained validated reader, and a
manifest of file hashes and registration timestamps.
"""

PREREGISTRATION = """\
# Pre-registration — paired two-arm comparison (arms A, B)

Committed before any confirmatory run; binding. Amendment 1 (epsilon) was
committed after the runs completed but before any analysis or arm-level
readout (timestamps in manifest.txt).

## Hypothesis

H1: arm A's embedding initialization improves final nats/byte over arm B's
(lower is better). With d_s = y_B,s - y_A,s, H1 predicts mean(d) > 0.
Architecture, data, and per-seed conditions are identical between arms; only
the embedding initialization differs.

## Design

- Pairing: same seed s implies identical data order and all non-embedding
  init; evidenced per run by the data_order_digest field (Appendix A). A
  digest mismatch within a pair INVALIDATES the pair (reported, excluded).
- Stage 1 seeds: {1, 2, 3, 4, 5} (df = 4), analyzed once.
- Stage 2 seeds: {6, 7, 8, 9} (cumulative n = 9, df = 8) — run ONLY if the
  stage-1 CI straddles 0.
- Primary metric: final nats/byte, read exclusively via reader.py's
  final_eval_metric(path, "nats_per_byte").
- Estimator: CI = mean(d) +/- t_{0.9875, n-1} * sd(d)/√n — the total 5%
  alpha is split across the two looks (Bonferroni, per-look 2.5%):
  t_{0.9875,4} ~= 3.50 (stage 1), t_{0.9875,8} ~= 2.75 (stage 2, cumulative).
- Verdict rule: CI entirely above 0 -> H1 supported. CI entirely below
  +epsilon -> no practically relevant improvement. Otherwise -> inconclusive
  (stage 2 if not yet run; else reported as inconclusive).
- Exclusion rule (only this one): a run is excluded solely for a non-finite
  final loss; an exclusion always drops the whole PAIR and is reported.
- Pilot declaration: one declared pilot run of arm A (2026-08-17) established
  cost/duration/plumbing. Pilots are never part of the confirmatory set.

## Amendment 1 — epsilon fixed (2026-08-18, post-run / pre-analysis)

epsilon = 0.01 nats/byte. Basis, chosen with NO access to any confirmatory
outcome of this comparison (the amender had seen baseline runs outside this
comparison and the declared pilot): one enwik8 reporting step (0.01 bits/byte
~= 0.0069 nats/byte) rounded up to ~=1.3% of a byte-level baseline outside
this comparison (0.7738 nats/byte). Every other registration term unchanged.

## Appendix A — data_order_digest

SHA-256 over the source offsets of the first 32 training batches, emitted
ONCE as the additive fields data_order_digest + data_order_batches on the
first logged train record at/after the window — or on the final record of a
run shorter than 32 steps, which then attests its actual batch count. Fresh
runs only: a resumed run never attests (absence = not attested, never "same
order").
"""

EXTENSION_REGISTRATION = """\

## Extension registration — seeds 6-9 (outcome-aware; committed before any extension run)

The stage-1 confirmatory phase closed with its filed verdict. This extension
was registered AFTER that outcome was known and BEFORE any extension run
launched: seeds {6, 7, 8, 9}, same paired design, same container image
digest, same GPU class, same corpus and configs.

- Primary: cumulative paired analysis over seeds {1..9}. STRONGLY_SUPPORTED
  iff mean(d) / (sd(d)/√9) > t_{0.99865, 8} = 4.277 (the one-sided 99.865%
  lower confidence bound of mean(d) exceeds 0). Otherwise
  NOT_STRONGLY_SUPPORTED. This label is reported ALONGSIDE — never
  replacing — the stage-1 verdict.
- Practical relevance is read against the ratified epsilon = 0.01 nats/byte,
  unchanged.
- Clause precedence (declared after extension runs launched, before any
  completed or was analyzed): the direction label is primary; the
  practical-relevance rider is mandatory on every quotation of the verdict.
- Exclusion rule unchanged (non-finite final loss drops the whole pair);
  pairing rule unchanged (a within-pair data_order_digest mismatch
  invalidates the pair).
"""

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


def scrub_record(record: dict[str, Any], arm: str) -> dict[str, Any]:
    """Map arm-identifying values to the blind label; drop init-path keys."""
    out = dict(record)
    if "run" in out:
        out["run"] = f"arm-{arm}"
    config = out.get("config")
    if isinstance(config, dict):
        config = json.loads(json.dumps(config))  # deep copy
        if "name" in config:
            config["name"] = f"arm-{arm}"
        model = config.get("model", {})
        if "embedding_init" in model:
            model["embedding_init"] = INIT_MAP[model["embedding_init"]]
        model.pop("fisher_init_path", None)
        # symmetric across arms but project vocabulary — a staged-variant key
        model.pop("anchor_mode", None)
        out["config"] = config
    return out


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


def build_reader(telemetry_source: str) -> str:
    """Strip all prose from the telemetry module; shim the package import.

    Also drops the operator CLI (``main``/``_build_parser``/``__main__``
    block) — its argparse prog/help strings carry repository names, and the
    reviewer parses through the functions, never argv.
    """
    tree = _DocstringStripper().visit(ast.parse(telemetry_source))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in ("main", "_build_parser"))
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
    code = code.replace(
        "from entropic_substrate.logging import configure_logging, logger",
        _LOGGING_SHIM,
    )
    return _READER_HEADER + code + "\n"


def banned_scan(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").lower()
    return [word for word in BANNED if word in text]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, default=Path("results/captured"))
    parser.add_argument(
        "--telemetry", type=Path, default=Path("src/entropic_substrate/telemetry.py")
    )
    parser.add_argument("--packet-out", type=Path, required=True)
    parser.add_argument("--sealed-out", type=Path, required=True)
    parser.add_argument("--run-times", type=Path, required=True, help="run_id -> times JSON")
    parser.add_argument("--prereg-commit", required=True, help="'<sha> <iso-utc>'")
    parser.add_argument("--amendment-commit", required=True, help="'<sha> <iso-utc>'")
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

    packet = args.packet_out
    if packet.exists():
        shutil.rmtree(packet)
    (packet / "data").mkdir(parents=True)

    run_times = json.loads(args.run_times.read_text(encoding="utf-8"))

    (packet / "abstract.md").write_text(ABSTRACT, encoding="utf-8")
    prereg_text = PREREGISTRATION
    if args.ext_prereg_commit:
        prereg_text += EXTENSION_REGISTRATION
    (packet / "preregistration.md").write_text(prereg_text, encoding="utf-8")
    (packet / "reader.py").write_text(
        build_reader(args.telemetry.read_text(encoding="utf-8")), encoding="utf-8"
    )

    sealed_rows: list[str] = []
    time_rows: list[str] = []
    for src in sorted(args.captured.glob("e1-*.jsonl")):
        # capture names look like: e1-fisher-<runid8>-e1-fisher-s<N>.jsonl
        arm_name = "e1-fisher" if "fisher" in src.name else "e1-random"
        seed = src.stem.rsplit("-s", 1)[-1]
        arm = ARM_MAP[arm_name]
        dst = packet / "data" / f"{arm}-seed{seed}.jsonl"
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(json.dumps(scrub_record(json.loads(line), arm)) + "\n")
        run_id8 = src.name.split("-")[2]
        full = next((k for k in run_times if k.startswith(run_id8)), None)
        if full is None:
            msg = f"no tracking-server times for {src.name} (id prefix {run_id8})"
            raise LookupError(msg)
        for stamp_key in ("start", "end"):
            stamp = run_times[full][stamp_key]
            iso = datetime.fromtimestamp(stamp / 1000, tz=UTC).isoformat()
            time_rows.append(f"# {dst.name}  {stamp_key}={iso}")
        sealed_rows.append(f"| {src.name} | {md5(src)} | {dst.name} | {md5(dst)} | run_id {full} |")

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
    leaks = {str(p): banned_scan(p) for p in files if banned_scan(p)}
    if leaks:
        msg = f"identity leak(s) in packet files: {leaks}"
        raise AssertionError(msg)

    manifest_lines = ["# E1 review packet manifest", "#", "# md5  file"]
    manifest_lines += [f"{md5(p)}  {p.relative_to(packet)}" for p in files]
    manifest_lines += [
        "#",
        "# Registration commits (UTC):",
        f"# preregistration committed: {args.prereg_commit}",
        f"# amendment 1 (epsilon) committed: {args.amendment_commit}",
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
        "name: e1-review-sealed-mapping\n"
        'description: "SEALED A/B mapping for the E1 review packet — disclose only '
        'after the review is filed (docs/peer-review-protocol.md)"\n'
        "metadata:\n  type: project\n---\n\n"
        "# SEALED — E1 review packet arm mapping\n\n"
        "Mapping: **A = e1-fisher, B = e1-random** (A = registered-favored arm by\n"
        "protocol construction; the blind hides which initialization A is).\n\n"
        "Scrub transform applied to every record: run -> arm-<X>; config.name ->\n"
        "arm-<X>; config.model.embedding_init -> init-<X>; config.model.\n"
        "fisher_init_path and config.model.anchor_mode dropped. reader.py =\n"
        "telemetry.py with all docstrings/comments stripped (ast.unparse), the\n"
        "package logging import shimmed, and the operator CLI removed.\n\n"
        "| capture file | capture md5 | packet file | packet md5 | run |\n"
        "|---|---|---|---|---|\n" + "\n".join(sealed_rows) + "\n",
        encoding="utf-8",
    )
    print(f"packet: {packet} ({len(files)} files) | sealed: {args.sealed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
