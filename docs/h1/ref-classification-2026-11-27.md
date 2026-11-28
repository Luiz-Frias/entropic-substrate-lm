# Ref classification by patch equivalence, 2026-11-27

Horizon M0 of the platform refactor classifies every ref that is not an ancestor of `main`
before any branch or worktree is cleaned up (plan row M0, source review risk 11). This file is a
record. It deletes nothing and authorizes no delete.

## Why ancestry is the wrong test here

This repository squash-merges most pull requests. A squash writes one new commit on `main`, so
the branch's own commits never become ancestors of `main`, and `git branch --no-merged` reports
landed work as unmerged. `git cherry` compares patch identities instead, which survives a rebase
but not a squash of several commits into one. So three checks are used, in order:

1. ancestry: is the tip an ancestor of `origin/main`;
2. patch equivalence: `git cherry origin/main <ref>`, where `-` marks a commit whose patch is
   already on `main` and `+` one whose patch is not;
3. for every `+` that a merged pull request claims, blob identity of the changed files against
   the merge commit.

## Method

Measured on 2026-11-28 UTC in the M0 worktree, after `git fetch origin`. Every value below is
`[measured]` by the command named in its column.

| reference point | commit |
| --- | --- |
| `origin/main` | `5f6b6dd92b9da007624ce255492f55b779fd4bfd` |
| `origin/refactor/platform-topology`, the refactor trunk | `9d7917c81af71c9faf23591469db374f1457c9e0` |
| tag `epoch-pre-platform-refactor` | tag object `0edde818`, peels to `5f6b6dd9` |

Namespaces scanned: `refs/heads`, `refs/remotes/origin`, `refs/gt-fetch-head`, `refs/codex` and
`refs/tags`. A ref whose tip is an ancestor of `origin/main` is merged by ancestry and is not
listed. All 24 tags are ancestors of `origin/main`. The stash is empty.

## Classification

Columns: `ahead` is `git rev-list --count origin/main..<ref>`; `+` and `-` are the two counts
from `git cherry origin/main <ref>`; `held on origin by` lists the `origin` branches that contain
the tip.

### Landed on main, every patch equivalent

Ten refs under `refs/gt-fetch-head`, Graphite's local fetch cache. Each has zero `+` commits.

| ref | tip | ahead | + | - | pull request |
| --- | --- | --- | --- | --- | --- |
| `claude/relaxed-thompson-k1huxv` | `9c0a5a10` | 4 | 0 | 4 | 412, merged |
| `docs/series-plugin-topology-plan` | `181fb4de` | 5 | 0 | 5 | 418, merged |
| `docs/series-plugin-topology-plan-corpus-review` | `ecc10e08` | 1 | 0 | 1 | 426, merged |
| `docs/series-plugin-topology-plan-d11` | `fa1da8d4` | 2 | 0 | 2 | 427, merged |
| `docs/series-plugin-topology-plan-engines` | `51e25052` | 6 | 0 | 6 | 419, merged |
| `docs/series-plugin-topology-plan-member-rules` | `07129a58` | 8 | 0 | 8 | 421, merged |
| `docs/series-plugin-topology-plan-observability` | `a681231c` | 7 | 0 | 7 | 420, merged |
| `feat/for-1689-enwik8-documents` | `24ac0ce1` | 27 | 0 | 27 | 417, merged |
| `fix/host-suite-lock` | `ff516ab5` | 9 | 0 | 9 | 422, merged |
| `fix/ship-it-holds-suite-lock` | `e3e266ba` | 10 | 0 | 10 | 423, merged |

None of the ten is held by an `origin` branch. Nothing is lost if they go: every patch is on
`main`.

### Landed on main by squash, patch identity differs

| ref | tip | ahead | + | - | pull request | held on origin by |
| --- | --- | --- | --- | --- | --- | --- |
| `refs/gt-fetch-head/pilot-run-authority` | `638b911f` | 2 | 2 | 0 | 406, merged as `467b5d7a` | `refs/pull/406/head` only |

This is the case the method exists for. Both commits read as unmerged under ancestry and under
`git cherry`, because the pull request squashed two commits into one. Check 3 settles it: the
ref changes two files, and both blobs are identical in the squash commit.

| file | blob at the ref | blob at `467b5d7a` |
| --- | --- | --- |
| `src/entropic_substrate/ladder/run/run_authority.py` | `af235966f3b5` | `af235966f3b5` |
| `tests/test_run_authority.py` | `ba5671cf39c3` | `ba5671cf39c3` |

The source file is still that blob on `main`. The test has since moved to
`tests/unit/ladder/run/test_run_authority.py` with one line changed.

### On the refactor trunk, not yet on main

| ref | tip | ahead | + | held on origin by |
| --- | --- | --- | --- | --- |
| `refs/heads/refactor/platform-topology` and its `origin` twin | `9d7917c8` | 3 | 2 | itself |
| `refs/gt-fetch-head/docs/plan/platform-refactor-rung-seven` | `8754e614` | 2 | 2 | `origin/refactor/platform-topology` |

The two `+` commits are the prelaunch commits `a50ea2a8` and `8754e614`. The third commit is the
merge commit of pull request 445. They reach `main` when the trunk merges, which is the
operator's action at a horizon exit.

The M0 stack's own branches under `refs/heads/refactor/charter/` sit on top of the trunk and are
in flight while this file is written. They are not classified here: their tips move with each
commit of this horizon, and they are published as pull requests into the trunk. The worktree's
starting branch `m0-flush-freeze-charter-15fc9b` held no commit of its own and was removed once
the stack's first branch replaced it.

### Shelved by ruling, not landed

| ref | tip | ahead | + | own commits | held on origin by | ruling |
| --- | --- | --- | --- | --- | --- | --- |
| `shelf/h1/l4d-evaluation-instruments`, local and `origin`; also `refs/gt-fetch-head/h1-l4d-evaluation-instruments` | `35b0b687` | 23 | 20 | 20, plus 3 merge commits; 65 files, +5219 and -605 against the merge base | `origin/shelf/h1/l4d-evaluation-instruments` | D12: kept shelved, read as design input at M6 and M7 |
| `h1-l4e-segmenter-alignment`, local and `origin` | `73dc7968` | 13 | 13 | 2: `69cf0fd1`, `73dc7968`; tests only | `origin/h1-l4e-segmenter-alignment` | D13: ported into M1 as a rail |
| `feat/for-1713-codec-handoff-residual-pins`, local and `origin` | `9c3729e1` | 12 | 12 | 1: `9c3729e1`; tests only | `origin/feat/for-1713-codec-handoff-residual-pins` | D13: ported into M1 as a rail |

Own commits for the second and third rows are `git cherry` against the L4d shelf, not against
`main`. Both refs branch from `391693b8`, eleven commits into the L4d line, so a diff against
`main` shows those eleven as well. The closed pull request 435 pointed at the L4d head; `origin`
now holds that head under the shelf branch too, so it no longer depends on `refs/pull/435/head`.

What the three refs contain, and what resuming each one takes, is in
`docs/h1/shelved-state-2026-11-27.md`.

### Open pull requests against main

| ref | tip | + | pull request | touches |
| --- | --- | --- | --- | --- |
| `origin/fix/FOR-1724` | `787e862e` | 1 | 444, open draft | `.claude/hooks/worktree-post-create.sh`, `.claude/workflows/mass-patch-fanout.js` |
| `origin/dependabot/github_actions/actions/checkout-7` | `7cd1a639` | 1 | 19, open | `.github/workflows/ci.yaml` |
| `origin/dependabot/github_actions/astral-sh/setup-uv-7` | `30ba6fe9` | 1 | 20, open | `.github/workflows/ci.yaml` |
| `origin/dependabot/uv/dvc-s3--gte-3.67.1` | `95c5cc33` | 1 | 22, open | `pyproject.toml`, `uv.lock` |
| `origin/dependabot/uv/mlflow-skinny-3.15.1` | `d2e13dd1` | 1 | 23, open | `pyproject.toml`, `uv.lock` |
| `origin/dependabot/uv/numpy-2.5.2` | `6af04035` | 1 | 21, open | `pyproject.toml`, `uv.lock` |
| `origin/dependabot/uv/ruff-0.16.2` | `3c18f6de` | 1 | 25, open | `pyproject.toml`, `uv.lock` |
| `origin/dependabot/uv/setuptools-84.0.0` | `ebf8cf0f` | 1 | 24, open | `pyproject.toml` |
| `origin/dependabot/uv/uv-18c674f953` | `1794527d` | 1 | 11, open | `uv.lock` |

Six of the eight dependency branches touch `pyproject.toml` or `uv.lock`, the files horizons M3,
M10, M12 and MF own. The two GitHub Actions branches touch neither.

### Not commits

| ref | object |
| --- | --- |
| `refs/codex/turn-diffs/captures/1793040477804/cf18f840-d7b1-4778-978e-90603233a88f/base` | tree `c70a13d6` |
| `refs/codex/turn-diffs/checkpoints/b5d97b24758d618cac04afe5f6566c5b/dbd65325a96bd747f7989efad155ee4d/1793043129241/b76876bf-e19f-48ce-a01e-3a6e617480b6` | tree `81a31206` |

These are tree snapshots written by another tool. Patch equivalence does not apply to a tree.

## Unreachable objects

`git fsck --unreachable --no-reflogs` reports 1489 commits, 2175 trees and 526 blobs that no ref
reaches. The read is non-destructive. The source review relays that dropped stash entries dated
2026-11-22 are among them. No `git gc`, `git prune` or `git stash clear` has been run in M0, and
none should be run until someone has read those entries: a prune deletes them for good.

## What this table supports

| question | answer from the table |
| --- | --- |
| is any in-flight work held only by a local ref | no. Every ref with a `+` commit that is not on `main` is held by an `origin` branch, except `pilot-run-authority`, whose content is on `main` |
| which refs carry no information `main` lacks | the ten patch-equivalent refs and `pilot-run-authority` |
| which refs must outlive the refactor | the three shelved H1 refs |
| what may be deleted | nothing by this file. Each delete is the operator's call, taken with this table in hand |
