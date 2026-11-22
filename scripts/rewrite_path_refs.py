#!/usr/bin/env python3
"""Rewrite stale path references in bulk after files move.

Point it at a source of old-to-new paths and at the trees to sweep:

    # preview what a move between two revisions left behind under docs/
    uv run python scripts/rewrite_path_refs.py --git-renames main HEAD docs

    # apply, leaving filed verdicts stale on purpose
    uv run python scripts/rewrite_path_refs.py --git-renames main HEAD --apply \\
        --exclude 'docs/verdicts/*' --exclude 'docs/h0/verdict-*' docs

    # gate form: exit 1 while any stale reference remains
    uv run python scripts/rewrite_path_refs.py --git-renames main HEAD --check docs

Three parts, each replaceable: a MAPPING SOURCE (git renames between two
revisions, or a JSON object of old to new), a RULE SET compiled from the
mapping (path form always, dotted-module form on ``--dotted``), and a SWEEP
over target trees with exclude globs. Dry-run is the default; nothing is
written without ``--apply``. An empty mapping is refused: a rewriter that
examined nothing must not report clean.

A reference is rewritten only as a whole token. ``tests/test_a.py`` inside
``src/tests/test_a.py`` or ``tests/test_a.pyc`` is a different path and is
left alone (pinned by tests/repo/test_rewrite_path_refs.py).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess  # nosec B404 -- fixed git argv, validated revisions, no shell
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

RENAME_FIELDS = 3


class EmptyMappingError(ValueError):
    """The mapping source produced no old-to-new pairs."""


@dataclass(frozen=True)
class Rule:
    """One compiled reference form: a pattern and what each match becomes."""

    pattern: re.Pattern[str]
    replacements: Mapping[str, str]


@dataclass
class Report:
    """What a sweep found, split by what was done about it."""

    rewritten: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.rewritten.values()) + sum(self.excluded.values())


def mapping_from_git(repo: Path, base: str, head: str) -> dict[str, str]:
    """Old-to-new paths for every rename git detects between two revisions."""
    for revision in (base, head):
        if not revision or revision.startswith("-"):
            message = f"not a revision: {revision!r}"
            raise ValueError(message)
    done = subprocess.run(  # nosec B603 B607 -- git from PATH by design; argv is fixed plus two validated revisions
        ["git", "diff", "--name-status", "-M", "--diff-filter=R", "-z", base, head],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    fields = [part for part in done.stdout.split("\0") if part]
    if len(fields) % RENAME_FIELDS:
        message = f"unexpected git rename output: {len(fields)} fields"
        raise ValueError(message)
    triples = zip(
        fields[0::RENAME_FIELDS], fields[1::RENAME_FIELDS], fields[2::RENAME_FIELDS], strict=True
    )
    return {old: new for _status, old, new in triples}


def mapping_from_json(path: Path) -> dict[str, str]:
    """Old-to-new paths from a JSON object of strings."""
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in loaded.items()
    ):
        message = f"{path} must hold one JSON object mapping old path to new path"
        raise ValueError(message)
    return dict(loaded)


def _alternation(keys: Sequence[str]) -> str:
    return "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True))


def _dotted(path: str) -> str | None:
    return path[: -len(".py")].replace("/", ".") if path.endswith(".py") else None


def compile_rules(mapping: Mapping[str, str], *, dotted: bool = False) -> tuple[Rule, ...]:
    """Compile a mapping into whole-token rewrite rules."""
    pairs = {old: new for old, new in mapping.items() if old != new}
    if not pairs:
        message = "no old-to-new pairs: nothing would be examined"
        raise EmptyMappingError(message)
    rules = [Rule(re.compile(rf"(?<![\w./-])(?:{_alternation(list(pairs))})(?![\w-]|\.\w)"), pairs)]
    if dotted:
        modules = {
            old_module: new_module
            for old, new in pairs.items()
            if (old_module := _dotted(old)) and (new_module := _dotted(new))
        }
        if modules:
            rules.append(
                Rule(re.compile(rf"(?<![\w.])(?:{_alternation(list(modules))})(?![\w])"), modules)
            )
    return tuple(rules)


def rewrite_text(text: str, rules: Sequence[Rule]) -> tuple[str, int]:
    """Apply every rule; return the new text and how many references changed."""
    total = 0
    for rule in rules:
        text, count = rule.pattern.subn(
            lambda match, rule=rule: rule.replacements[match.group(0)], text
        )
        total += count
    return text, total


def _files(root: Path, targets: Sequence[Path]) -> list[Path]:
    found: list[Path] = []
    for target in targets:
        path = root / target
        if path.is_file():
            found.append(path)
        else:
            found.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
            )
    return found


def run(
    root: Path,
    targets: Sequence[Path],
    rules: Sequence[Rule],
    *,
    excludes: Sequence[str],
    apply: bool,
) -> Report:
    """Sweep the targets under ``root``; write only when ``apply`` is set."""
    report = Report()
    for path in _files(root, targets):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text, count = rewrite_text(text, rules)
        if not count:
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, glob) for glob in excludes):
            report.excluded[relative] = count
            continue
        report.rewritten[relative] = count
        if apply:
            path.write_text(new_text, encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--git-renames",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="map every rename between two revisions",
    )
    source.add_argument("--mapping-json", type=Path, help="JSON object of old path to new path")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true", help="write the rewrites (default is a dry run)"
    )
    mode.add_argument(
        "--check", action="store_true", help="exit 1 if any non-excluded stale reference remains"
    )
    parser.add_argument(
        "--dotted", action="store_true", help="also rewrite dotted module references"
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB", help="leave matching files stale"
    )
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root (default: cwd)"
    )
    parser.add_argument(
        "targets", nargs="+", type=Path, help="files or directories to sweep, relative to the root"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    mapping = (
        mapping_from_git(root, *args.git_renames)
        if args.git_renames
        else mapping_from_json(args.mapping_json)
    )
    try:
        rules = compile_rules(mapping, dotted=args.dotted)
    except EmptyMappingError as error:
        print(f"rewrite_path_refs: {error}", file=sys.stderr)
        return 2
    report = run(root, args.targets, rules, excludes=args.exclude, apply=args.apply)
    verb = "rewrote" if args.apply else "would rewrite"
    for name, count in sorted(report.rewritten.items()):
        print(f"{verb} {count:4d}  {name}")
    for name, count in sorted(report.excluded.items()):
        print(f"left stale {count:4d}  {name}  (excluded)")
    print(
        f"{len(mapping)} mapped paths; {sum(report.rewritten.values())} references in {len(report.rewritten)} files "
        f"{'rewritten' if args.apply else 'pending'}; {sum(report.excluded.values())} left stale by exclusion"
    )
    return 1 if args.check and report.rewritten else 0


if __name__ == "__main__":
    raise SystemExit(main())
