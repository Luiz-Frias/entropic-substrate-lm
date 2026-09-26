"""RED-first relocated-guard tests for the two fail-closed pod source guards
(``d_ladder_worker_runtime.verify_worker_source`` and
``sentence_prediction.worker_execution._runtime``) plus the resolver pins for
``entropic_substrate.paths`` that both guards now share (E-3/DN-8).

Both guards were rewritten this phase onto ``entropic_substrate.paths`` so
their "am I executing the bound tree" check is layout-invariant: the actual
side of the comparison comes from the installed ``entropic_substrate``
package's own resolved location, never from the calling module's own
``__file__`` depth. Loading a COPY of each guard module from an unrelated
directory (``importlib.util.spec_from_file_location``) proves this directly:
today, before this phase, ``Path(__file__).resolve().parent`` inside a
relocated copy points at the copy's tmp location, so the guard would
wrongly refuse even a legitimately bound tree — measured: grepping either
raise message across ``tests/`` and ``src/`` before this phase returns only
the two ``raise`` sites, zero test assertions. These tests are the first
assertions on either message.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from entropic_substrate import paths

pytestmark = pytest.mark.unit

# Repo-root convention used by every other test file in this suite (e.g.
# tests/test_identity_epoch.py's ``_REPO_ROOT``) -- deliberately NOT
# ``paths.repo_root()``: this file's fixtures build the paths to the SOURCE
# files being relocated, so they must stay independent of the resolver under
# test. Using the resolver here would make the "point the resolver at a
# wrong root" mutation corrupt these fixtures instead of exercising the
# guards, producing a vacuous (ERROR, not RED-on-the-guard) arm.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_D_LADDER_RUNTIME_SOURCE = _REPO_ROOT / "src" / "entropic_substrate" / "d_ladder_worker_runtime.py"
_WORKER_EXECUTION_SOURCE = (
    _REPO_ROOT / "src" / "entropic_substrate" / "sentence_prediction" / "worker_execution.py"
)

_GUARD_TEXT_D_LADDER = "worker is not executing the source tree selected for verification"
_GUARD_TEXT_WORKER_EXECUTION = "remote worker must execute its bound source tree"


def _load_relocated_copy(source: Path, relocated_dir: Path, module_name: str) -> ModuleType:
    """Copy ``source`` into ``relocated_dir`` and import it under a distinct
    module name, so its own ``__file__`` differs from the installed
    package's copy while its absolute ``from entropic_substrate...`` imports
    still resolve to the real installed package (unaffected by the copy)."""
    relocated_dir.mkdir(parents=True, exist_ok=True)
    target = relocated_dir / source.name
    shutil.copy2(source, target)
    spec = importlib.util.spec_from_file_location(module_name, target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def relocated_d_ladder_runtime(tmp_path: Path) -> Iterator[ModuleType]:
    module_name = "test_relocated_d_ladder_worker_runtime"
    try:
        yield _load_relocated_copy(_D_LADDER_RUNTIME_SOURCE, tmp_path / "elsewhere", module_name)
    finally:
        sys.modules.pop(module_name, None)


@pytest.fixture
def relocated_worker_execution(tmp_path: Path) -> Iterator[ModuleType]:
    module_name = "test_relocated_worker_execution"
    try:
        yield _load_relocated_copy(_WORKER_EXECUTION_SOURCE, tmp_path / "elsewhere", module_name)
    finally:
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# entropic_substrate.paths — resolver pins (no tests/test_paths.py in the
# allowed set for this phase; the resolver is pinned here alongside its two
# consumer guards).
# ---------------------------------------------------------------------------


class TestPathsResolvers:
    def test_source_root_contains_the_real_package_init(self) -> None:
        assert (paths.source_root() / "entropic_substrate" / "__init__.py").is_file()

    def test_repo_root_contains_pyproject_toml(self) -> None:
        assert (paths.repo_root() / "pyproject.toml").is_file()

    def test_repo_root_is_one_level_above_source_root(self) -> None:
        assert paths.repo_root() == paths.source_root().parent

    def test_repo_root_cannot_serve_as_the_closure_resolvers_source_root(self) -> None:
        """The two anchors are NOT interchangeable (architect constraint):
        d1_reader_source_closure needs source_root, not repo_root.

        A bare inequality (source_root() != repo_root()) is nearly vacuous
        given test_repo_root_is_one_level_above_source_root already pins
        repo_root() == source_root().parent -- a path is never equal to its
        own parent. This test instead pins the actual contract
        d1_reader_source_closure._source_root() enforces: the candidate root
        must directly contain a real ``entropic_substrate`` package
        directory. repo_root() fails that contract (the package sits one
        level deeper, under ``src/``), which is the concrete reason the two
        anchors are not interchangeable."""
        assert not (paths.repo_root() / "entropic_substrate").is_dir()

    def test_repo_root_raises_repo_layout_error_when_package_file_is_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import entropic_substrate

        monkeypatch.setattr(entropic_substrate, "__file__", None, raising=False)
        with pytest.raises(paths.RepoLayoutError):
            paths.repo_root()

    def test_source_root_raises_repo_layout_error_when_package_dir_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import entropic_substrate

        missing = tmp_path / "does-not-exist" / "entropic_substrate" / "__init__.py"
        monkeypatch.setattr(entropic_substrate, "__file__", str(missing), raising=False)
        with pytest.raises(paths.RepoLayoutError):
            paths.source_root()


# ---------------------------------------------------------------------------
# verify_worker_source (d_ladder_worker_runtime.py) — relocated copy
# ---------------------------------------------------------------------------


class _FakeRequestNoBundle:
    registration_git_bundle = None


class TestVerifyWorkerSourceRelocated:
    def test_relocated_copy_accepts_the_bound_tree(
        self, relocated_d_ladder_runtime: ModuleType
    ) -> None:
        """Must stay GREEN: the guard accepts the bound tree from any
        location. Proceeding past the guard reaches the NEXT check (a
        missing Git bundle), a distinct message from the guard's own."""
        with pytest.raises(ValueError, match="pod worker requires a bound Git bundle"):
            relocated_d_ladder_runtime.verify_worker_source(
                objects=None,  # type: ignore[arg-type]
                request=_FakeRequestNoBundle(),
                repo_root=_REPO_ROOT,
            )

    def test_relocated_copy_refuses_an_unbound_tree(
        self, relocated_d_ladder_runtime: ModuleType, tmp_path: Path
    ) -> None:
        """Must go RED: fail-closed semantics survive the rewrite."""
        unbound_root = tmp_path / "unbound-repo"
        (unbound_root / "src" / "entropic_substrate").mkdir(parents=True)
        with pytest.raises(ValueError, match=_GUARD_TEXT_D_LADDER):
            relocated_d_ladder_runtime.verify_worker_source(
                objects=None,  # type: ignore[arg-type]
                request=_FakeRequestNoBundle(),
                repo_root=unbound_root,
            )


# ---------------------------------------------------------------------------
# _runtime (sentence_prediction/worker_execution.py) — relocated copy
# ---------------------------------------------------------------------------


class _FakeProfile:
    run_class = "remote_tiny"


class _FakeProbeRequest:
    profile = _FakeProfile()
    code_bundle = object()
    code_commit = "deadbeef" * 5  # 40 hex chars, shape-only stand-in


class _SentinelPastGuard(Exception):
    """Raised by a monkeypatched verified_git_bundle to prove the guard
    already accepted the tree before this point was reached."""


class TestRuntimeGuardRelocated:
    def test_relocated_copy_accepts_the_bound_tree(
        self,
        relocated_worker_execution: ModuleType,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Must stay GREEN: the guard accepts the bound tree from any
        location. A monkeypatched next-step raises a distinct sentinel,
        proving the guard's own message was never hit."""

        def _raise_sentinel(*_args: object, **_kwargs: object) -> None:
            raise _SentinelPastGuard

        monkeypatch.setattr(relocated_worker_execution, "verified_git_bundle", _raise_sentinel)
        with pytest.raises(_SentinelPastGuard):
            relocated_worker_execution._runtime(
                request=_FakeProbeRequest(),  # type: ignore[arg-type]
                root=tmp_path,
                repo_root=_REPO_ROOT,
            )

    def test_relocated_copy_refuses_an_unbound_tree(
        self, relocated_worker_execution: ModuleType, tmp_path: Path
    ) -> None:
        """Must go RED: fail-closed semantics survive the rewrite."""
        unbound_root = tmp_path / "unbound-repo"
        (unbound_root / "src" / "entropic_substrate" / "sentence_prediction").mkdir(parents=True)
        with pytest.raises(ValueError, match=_GUARD_TEXT_WORKER_EXECUTION):
            relocated_worker_execution._runtime(
                request=_FakeProbeRequest(),  # type: ignore[arg-type]
                root=tmp_path,
                repo_root=unbound_root,
            )

    def test_local_tiny_run_class_never_reaches_the_guard(
        self, relocated_worker_execution: ModuleType, tmp_path: Path
    ) -> None:
        """Zero/blank case: a local_tiny request short-circuits before the
        repo_root/anchor check entirely, even with repo_root=None."""

        class _LocalProfile:
            run_class = "local_tiny"

        class _LocalRequest:
            profile = _LocalProfile()
            code_bundle = None
            code_commit = None

        result = relocated_worker_execution._runtime(
            request=_LocalRequest(),  # type: ignore[arg-type]
            root=tmp_path,
            repo_root=None,
        )
        assert result["kind"] == "local_tiny"


# ---------------------------------------------------------------------------
# Guard-teeth mutation, automated: "point the resolver at a wrong root" must
# turn BOTH guards' acceptance of a legitimately bound tree RED. Mutating
# entropic_substrate.paths.source_root() on disk and reloading would ALSO
# break analysis.d1_external_analysis_control's own unrelated module-level
# call to installed_d1_reader_source_files() (measured separately: it
# cascades into an ERROR before either guard's own comparison ever
# executes -- not evidence the GUARD depends on the resolver, only that the
# wider import graph does). Patching the bound name each guard module
# imported (``from entropic_substrate.paths import source_root``) isolates
# exactly the guard's own comparison line, on the REAL (non-relocated)
# modules -- the same single resolver reference both guards hold.
# ---------------------------------------------------------------------------


def test_mutating_the_shared_resolver_turns_both_guards_red(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import entropic_substrate.d_ladder_worker_runtime as dlwr
    import entropic_substrate.sentence_prediction.worker_execution as we

    # BEFORE: both guards accept the bound tree and proceed past their own
    # check into a later, distinct failure.
    with pytest.raises(ValueError, match="pod worker requires a bound Git bundle"):
        dlwr.verify_worker_source(
            objects=None,  # type: ignore[arg-type]
            request=_FakeRequestNoBundle(),
            repo_root=_REPO_ROOT,
        )

    def _raise_sentinel(*_args: object, **_kwargs: object) -> None:
        raise _SentinelPastGuard

    monkeypatch.setattr(we, "verified_git_bundle", _raise_sentinel)
    with pytest.raises(_SentinelPastGuard):
        we._runtime(request=_FakeProbeRequest(), root=Path("/tmp"), repo_root=_REPO_ROOT)  # type: ignore[arg-type]

    # MUTATION: point the resolver each guard consumes at a wrong root.
    wrong_root = lambda: Path("/tmp/MUTATED_WRONG_ROOT")  # noqa: E731
    monkeypatch.setattr(dlwr, "source_root", wrong_root)
    monkeypatch.setattr(we, "source_root", wrong_root)

    with pytest.raises(ValueError, match=_GUARD_TEXT_D_LADDER):
        dlwr.verify_worker_source(
            objects=None,  # type: ignore[arg-type]
            request=_FakeRequestNoBundle(),
            repo_root=_REPO_ROOT,
        )
    with pytest.raises(ValueError, match=_GUARD_TEXT_WORKER_EXECUTION):
        we._runtime(request=_FakeProbeRequest(), root=Path("/tmp"), repo_root=_REPO_ROOT)  # type: ignore[arg-type]

    # monkeypatch reverts both attributes automatically at teardown.
