"""p0 CLI-surface door (frozen_after: p13): the subcommand set for
``entropic_substrate.cli``, its ``--help`` rendering width, and the frozen
dotted-name literal set used by the worker-launch modules and the operator
scripts. Also PINS (measured, does not touch) the two machine-consumed
stdout sites this phase's Observability rewrite-on-arrival scope must never
route through ``logger``: ``telemetry final-eval`` and ``cli eval --json``.

Rendering width is set explicitly (``COLUMNS=80`` in the subprocess env)
rather than inherited from the ambient terminal, per the architect ruling:
argparse's help formatter reads ``shutil.get_terminal_size()``, which checks
``COLUMNS`` before falling back to fd 1's tty size, so an unpinned width
makes this test's own green regime-dependent. Interpreters are always
``sys.executable`` -- the ambient ``python3`` on this machine is CPython
3.14.7, a different regime than the project's pinned 3.12.13 venv (measured
in tests/test_identity_epoch.py's docstring), so a bare ``python3`` here
would both fail (no torch) and prove nothing about this project's CLI.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROG = "entropic-substrate"

_FROZEN_SUBCOMMANDS = (
    "capture-results",
    "charts",
    "d-probe",
    "d-score-gain",
    "d1-campaign-run",
    "d1-capture-finalize",
    "d1-capture-prepare",
    "d1-inspect",
    "d1-run-fresh",
    "d1-run-resume",
    "embedding-geometry",
    "eval",
    "field-pretrain",
    "h-capture-model",
    "h-diagnose-spans",
    "h-generate-probe",
    "h-inspect-custody",
    "h-inspect-generation",
    "h-inspect-model",
    "h-inspect-source",
    "h-inspect-spans",
    "h-inspect-training",
    "h-prepare-source",
    "h-train-probe",
    "init-fisher",
    "mlflow-aggregate",
    "train",
)

# The frozen dotted-name literal set (architect ruling: pinned by exact
# STRING match, never by line number -- an edit above a site shifts its line
# for a reason unrelated to a contract change).
_FROZEN_DOTTED_LITERALS = (
    "entropic_substrate.composition.entrypoints.cli",
    "entropic_substrate.ladder.capture.reader.telemetry",
    "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command",
    "entropic_substrate.historical.d_ladder.d_ladder_worker_command",
    "entropic_substrate.ladder.run.worker_command",
    "entropic_substrate.ladder.review.d1_external_worker",
    "entropic_substrate.ladder.capture.reader.d1_",
    "entropic_substrate.ladder.capture.records.d1_",
)

# {relative file path: literal substrings that file must contain verbatim}.
# Split into module-importable sites (src/) keyed by dotted name and path-only sites.
_FROZEN_LITERAL_SITES_MODULE: dict[str, tuple[str, ...]] = {
    "ladder.run.provider.provider_launch": (
        "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command",
        "entropic_substrate.historical.d_ladder.d_ladder_worker_command",
    ),
    "ladder.run.provider.d1_runpod_adapter": (
        "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command",
    ),
    "ladder.run.h_probe_launch": ("entropic_substrate.ladder.run.worker_command",),
    "ladder.review.d1_external_analysis_adapter": (
        "entropic_substrate.ladder.review.d1_external_worker",
    ),
}

_FROZEN_LITERAL_SITES_PATH: dict[str, tuple[str, ...]] = {
    "scripts/assemble_review_packet.py": (
        "entropic_substrate.ladder.capture.reader.d1_",
        "entropic_substrate.ladder.capture.records.d1_",
    ),
    "scripts/entrypoints.env": (
        "entropic_substrate.composition.entrypoints.cli",
        "entropic_substrate.ladder.capture.reader.telemetry",
        "entropic_substrate.ladder.run.worker_command",
        "entropic_substrate.composition.entrypoints.d1_trainer_mailbox_command",
        "entropic_substrate.historical.d_ladder.d_ladder_worker_command",
        "entropic_substrate.ladder.review.d1_external_worker",
    ),
}

# Union for the frozen literal set check: all literals across both dicts
_FROZEN_LITERAL_SITES: dict[str, tuple[str, ...]] = {
    **{
        f"src/entropic_substrate/{k.replace('.', '/')}.py": v
        for k, v in _FROZEN_LITERAL_SITES_MODULE.items()
    },
    **_FROZEN_LITERAL_SITES_PATH,
}


def _run_help(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["COLUMNS"] = "80"
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    return subprocess.run(
        [sys.executable, "-m", "entropic_substrate.composition.entrypoints.cli", *args, "--help"],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


class TestSubcommandSetIsImportDerivedAndFrozen:
    def test_build_parser_subcommand_set_matches_frozen_tuple(self) -> None:
        from entropic_substrate.composition.entrypoints.cli import build_parser

        parser = build_parser()
        subparsers_action = next(
            action
            for action in parser._subparsers._group_actions  # type: ignore[union-attr]
            if hasattr(action, "choices")
        )
        discovered = tuple(sorted(subparsers_action.choices.keys()))
        assert discovered == _FROZEN_SUBCOMMANDS

    def test_prog_name_is_frozen(self) -> None:
        from entropic_substrate.composition.entrypoints.cli import build_parser

        assert build_parser().prog == _PROG


class TestHelpSnapshotIsWidthPinned:
    def test_top_level_help_exits_zero_and_names_prog(self) -> None:
        result = _run_help()
        assert result.returncode == 0
        assert _PROG in result.stdout

    def test_top_level_help_is_stable_across_two_runs_at_the_pinned_width(self) -> None:
        first = _run_help()
        second = _run_help()
        assert first.stdout == second.stdout

    def test_eval_subcommand_help_exposes_the_json_flag(self) -> None:
        result = _run_help("eval")
        assert result.returncode == 0
        assert "--json" in result.stdout


class TestFrozenDottedNameLiteralSet:
    @pytest.mark.parametrize(
        ("dotted_name", "literals"),
        list(_FROZEN_LITERAL_SITES_MODULE.items()),
    )
    def test_module_site_contains_its_frozen_literals_verbatim(
        self, dotted_name: str, literals: tuple[str, ...]
    ) -> None:
        # Resolve by importlib to survive p3's shim rebind
        module_name = f"entropic_substrate.{dotted_name}"
        module = importlib.import_module(module_name)
        assert module.__file__ is not None, f"{module_name} has no __file__"
        text = Path(module.__file__).read_text(encoding="utf-8")
        for literal in literals:
            assert literal in text, f"{module_name!s} lost frozen literal {literal!r}"

    @pytest.mark.parametrize(
        ("relative_path", "literals"),
        list(_FROZEN_LITERAL_SITES_PATH.items()),
    )
    def test_path_site_contains_its_frozen_literals_verbatim(
        self, relative_path: str, literals: tuple[str, ...]
    ) -> None:
        # Path-based sites (scripts/) cannot be imported as modules
        text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for literal in literals:
            assert literal in text, f"{relative_path!s} lost frozen literal {literal!r}"

    def test_frozen_literal_set_is_exactly_the_pinned_tuple(self) -> None:
        all_literals: set[str] = set()
        for literals in _FROZEN_LITERAL_SITES.values():
            all_literals.update(literals)
        assert all_literals == set(_FROZEN_DOTTED_LITERALS)


class TestMachineConsumedStdoutSitesArePinnedNotRouted:
    """These two sites are the ONLY permitted raw ``print()`` calls under the
    Observability rewrite-on-arrival rule (P-1). p0 pins their existence and
    shape; it does not touch them -- routing them through logger would break
    the machine-readable contract a later phase's scripts parse."""

    def test_telemetry_final_eval_subcommand_exists_and_is_raw_print(self) -> None:
        # Resolve by importlib to survive p8's shim rebind
        module = importlib.import_module("entropic_substrate.ladder.capture.reader.telemetry")
        assert module.__file__ is not None, (
            "entropic_substrate.ladder.capture.reader.telemetry has no __file__"
        )
        text = Path(module.__file__).read_text(encoding="utf-8")
        assert '"final-eval"' in text
        assert "print(final_eval_metric(args.path, args.metric))" in text

    def test_cli_eval_json_flag_exists_and_is_raw_print(self) -> None:
        # Resolve by importlib to survive p8's shim rebind
        module = importlib.import_module("entropic_substrate.composition.entrypoints.cli")
        assert module.__file__ is not None, (
            "entropic_substrate.composition.entrypoints.cli has no __file__"
        )
        text = Path(module.__file__).read_text(encoding="utf-8")
        assert 'evaluate.add_argument("--json"' in text
        # Scoped to cmd_eval itself (AST, not a whole-file substring): cli.py
        # has a second, unrelated --json report site with the identical
        # print(json.dumps(...)) idiom, so a file-wide substring check cannot
        # tell "cmd_eval still raw-prints" from "some other command still does".
        tree = ast.parse(text)
        cmd_eval = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "cmd_eval"
        )
        body = ast.get_source_segment(text, cmd_eval)
        assert body is not None, "cmd_eval has no source segment"
        assert "print(json.dumps(payload, sort_keys=True))" in body


_D1_COMMANDS = frozenset(
    {
        "d1-campaign-run",
        "d1-capture-finalize",
        "d1-capture-prepare",
        "d1-inspect",
        "d1-run-fresh",
        "d1-run-resume",
    }
)

# command -> (module holding the handler, handler name). Resolved lazily at
# call time, so patching the module attribute reaches the dispatch.
_LAZY_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("d-probe", "composition.wiring.d_ladder_cli", "cmd_d_probe"),
    ("d-score-gain", "composition.wiring.d_ladder_probe_cli", "cmd_score_gain"),
    ("h-train-probe", "composition.cli", "cmd_sentence"),
    ("h-inspect-training", "composition.cli", "cmd_sentence"),
    ("h-generate-probe", "composition.cli", "cmd_sentence"),
    ("h-inspect-generation", "composition.cli", "cmd_sentence"),
    ("h-capture-model", "composition.capture_cli", "cmd_capture"),
    ("h-inspect-model", "composition.capture_cli", "cmd_capture"),
    ("h-inspect-custody", "composition.capture_cli", "cmd_capture"),
    ("h-diagnose-spans", "composition.span_cli", "cmd_span"),
    ("h-inspect-spans", "composition.span_cli", "cmd_span"),
    ("h-prepare-source", "composition.source_cli", "cmd_source"),
    ("h-inspect-source", "composition.source_cli", "cmd_source"),
)


def _main_function() -> tuple[ast.FunctionDef, str]:
    module = importlib.import_module("entropic_substrate.composition.entrypoints.cli")
    assert module.__file__ is not None
    text = Path(module.__file__).read_text(encoding="utf-8")
    main = next(
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    return main, text


def _reads_args_command(node: ast.AST) -> bool:
    return any(
        isinstance(inner, ast.Attribute)
        and inner.attr == "command"
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "args"
        for inner in ast.walk(node)
    )


class TestMainDispatchesThroughOneTable:
    def test_only_the_d1_early_return_branches_on_args_command(self) -> None:
        main, _ = _main_function()
        branches = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.If) and _reads_args_command(node.test)
        ]
        # The D1 early return is the single sanctioned `if args.command` in main().
        assert len(branches) == 1, [ast.unparse(node.test) for node in branches]
        assert "D1CliCommand" in ast.unparse(branches[0].test)

    def test_every_non_d1_subcommand_resolves_through_the_handlers_table(self) -> None:
        main, _ = _main_function()
        table = next(
            node.value
            for node in ast.walk(main)
            if isinstance(node, ast.Assign | ast.AnnAssign)
            and any(
                isinstance(t, ast.Name) and t.id == "handlers"
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            and isinstance(node.value, ast.Dict)
        )
        keys = {key.value for key in table.keys if isinstance(key, ast.Constant)}
        assert len(keys) == len(table.keys), "handlers keys must be unique string literals"
        assert keys == set(_FROZEN_SUBCOMMANDS) - _D1_COMMANDS

    @pytest.mark.parametrize(("command", "module_name", "handler"), _LAZY_ROUTES)
    def test_lazy_handler_routes_return_value_and_receives_parsed_args(
        self,
        monkeypatch: pytest.MonkeyPatch,
        command: str,
        module_name: str,
        handler: str,
    ) -> None:
        import argparse

        cli = importlib.import_module("entropic_substrate.composition.entrypoints.cli")
        seen: list[argparse.Namespace] = []

        def fake(args: argparse.Namespace) -> int:
            seen.append(args)
            return 37

        namespace = argparse.Namespace(command=command)

        class _Parser:
            def parse_args(self, argv: object) -> argparse.Namespace:
                del argv
                return namespace

        monkeypatch.setattr(cli, "build_parser", lambda: _Parser())
        monkeypatch.setattr(
            importlib.import_module(f"entropic_substrate.{module_name}"), handler, fake
        )
        assert cli.main([command]) == 37
        assert seen == [namespace]
