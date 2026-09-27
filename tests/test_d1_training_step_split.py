"""RED-first split-ownership test for d1_training_step / d1_calibration
(ruling R-parallel; contract C17-p5).

Both modules split, still under __legacy/, into a record half (dataclass/
StrEnum evidence types, math-only, no torch/numpy) and a torch half (the
forward/backward/clip transaction, kept at the module's existing dotted
name). p8 later moves only the record halves, p9 only the torch halves, so
the two future wave lanes must share no file -- which requires the record
symbols to be OWNED by the record half from the moment the split lands,
never merely present there.

Ownership, not presence, is what this file pins (architect constraint,
design gate 2026-09-22): the torch half RE-EXPORTS every record name for
its 11 out-of-envelope importers (7 __legacy modules, 4 test files) that
import them by the old dotted name -- a plain `from records import X`
import makes ``X.__module__`` say the RECORD half and the torch-half
attribute ``is`` (identity, not equality) the SAME object the record half
defines. A regression that instead DEFINES a same-named class directly in
the torch half would still pass every other test in this phase (the class
would work identically), but ``__module__`` would point at the torch half
and the two attributes would no longer be identical objects -- this is the
guard-teeth mutation for house rule 5's "define a record symbol in the
torch half" row, performed and reverted in the PR body.

This test module fails to even import until both record modules exist,
which is what makes it RED-first: authored before the split commit, it
ModuleNotFoundErrors at collection; the split commit is what turns it
green.
"""

from __future__ import annotations

import pytest

from entropic_substrate.ladder.capture.records import d1_training_step_records
from entropic_substrate.ladder.capture.records.d1_calibration_records import (
    D1BaselineCalibrationAttestation,
)
from entropic_substrate.ladder.capture.records.d1_training_step_records import (
    D1ComponentGradientEvidence,
    D1GradientEvidenceKind,
    D1GradientReachability,
    D1TrainStepResult,
)
from entropic_substrate.ladder.run.trainer import d1_calibration, d1_training_step

pytestmark = pytest.mark.unit

_TRAINING_STEP_RECORD_MODULE = "entropic_substrate.ladder.capture.records.d1_training_step_records"
_CALIBRATION_RECORD_MODULE = "entropic_substrate.ladder.capture.records.d1_calibration_records"

_TRAINING_STEP_RECORD_NAMES = (
    "D1ComponentGradientEvidence",
    "D1GradientEvidenceKind",
    "D1GradientReachability",
    "D1TrainStepResult",
)


@pytest.mark.parametrize("name", _TRAINING_STEP_RECORD_NAMES)
def test_d1_training_step_record_symbol_is_owned_by_the_record_half(name: str) -> None:
    """Ownership, not presence: the class/enum's own __module__ must say the
    record half, never the torch half -- this is what a re-DEFINITION in the
    torch half (the guard-teeth mutation) would break."""
    obj = getattr(d1_training_step, name)
    assert obj.__module__ == _TRAINING_STEP_RECORD_MODULE


@pytest.mark.parametrize("name", _TRAINING_STEP_RECORD_NAMES)
def test_d1_training_step_torch_half_reexports_the_record_object_by_identity(
    name: str,
) -> None:
    """The torch half's attribute must be the SAME object the record half
    defines (``is``, not merely an equal one) -- a re-export, never a
    parallel re-definition."""
    torch_half_attr = getattr(d1_training_step, name)
    record_half_attr = getattr(d1_training_step_records, name)
    assert torch_half_attr is record_half_attr


def test_d1_calibration_record_symbol_is_owned_by_the_record_half() -> None:
    assert D1BaselineCalibrationAttestation.__module__ == _CALIBRATION_RECORD_MODULE


def test_d1_calibration_torch_half_reexports_the_record_object_by_identity() -> None:
    assert d1_calibration.D1BaselineCalibrationAttestation is D1BaselineCalibrationAttestation


def test_both_record_halves_are_importable_from_their_own_module_before_any_split_move() -> None:
    """Both halves stay under __legacy/ this phase (non-goal: neither half
    moves out yet) -- this is the "each half importable from its own module"
    green arm the guard-teeth table names, run against the real modules
    rather than a synthetic fixture."""
    assert D1TrainStepResult.__module__ == _TRAINING_STEP_RECORD_MODULE
    assert D1ComponentGradientEvidence.__module__ == _TRAINING_STEP_RECORD_MODULE
    assert D1GradientEvidenceKind.__module__ == _TRAINING_STEP_RECORD_MODULE
    assert D1GradientReachability.__module__ == _TRAINING_STEP_RECORD_MODULE
    assert D1BaselineCalibrationAttestation.__module__ == _CALIBRATION_RECORD_MODULE


def test_d1_training_step_torch_half_declares_every_record_name_in_its_own_all() -> None:
    """C17-p5: the torch half keeps every record name in its own literal
    __all__ list (11 out-of-envelope importers, gen_shims' .pyi stub
    derivation, and ruff F401 on the re-export lines all depend on this)."""
    for name in _TRAINING_STEP_RECORD_NAMES:
        assert name in d1_training_step.__all__


def test_d1_calibration_torch_half_declares_the_record_name_in_its_own_all() -> None:
    assert "D1BaselineCalibrationAttestation" in d1_calibration.__all__


def test_record_halves_declare_no_torch_or_numpy_import() -> None:
    """Neither record half may acquire a runtime torch or numpy import
    (architect constraint): both stay math/dataclass/StrEnum-only so a
    later reader (evidence.d1_telemetry_*) can import the record symbols
    without pulling in torch. Checked at the AST level, not by import
    absence alone (a lazy/function-local import would not show up as a
    ModuleNotFoundError here but is exactly what this constraint forbids)."""
    import ast
    from pathlib import Path

    for relpath in (
        "src/entropic_substrate/ladder/capture/records/d1_training_step_records.py",
        "src/entropic_substrate/ladder/capture/records/d1_calibration_records.py",
    ):
        source = (Path(__file__).resolve().parents[1] / relpath).read_text()
        tree = ast.parse(source)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        assert "torch" not in imported_roots, f"{relpath} must not import torch"
        assert "numpy" not in imported_roots, f"{relpath} must not import numpy"


def test_record_halves_cold_import_without_torch_or_numpy() -> None:
    """Strengthens the AST check above: that check only inspects the two
    record files' own source, so a project-module dependency THEY import
    (e.g. the calibration half's `d1_primitives`) could transitively pull
    in torch/numpy without tripping it. This runs the record-half imports
    in an isolated subprocess with a meta-path finder that raises on any
    attempted `torch`/`numpy` import, direct or transitive, and keeps the
    AST check above for the direct-import case (subprocess overhead).

    Guard-teeth, both arms (house rule 5): the negative control at the top
    of the subprocess script proves the finder itself actually intercepts
    `torch`/`numpy` (asserting AssertionError if it does not) before the
    green arm below it proves the record halves import clean under that
    same finder -- a finder that silently stopped intercepting would still
    let the green arm pass, which is exactly what the negative control
    catches."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = """
import importlib.abc
import sys


class RefuseTorchOrNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("record modules must not import torch")
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise RuntimeError("record modules must not import numpy")
        return None


sys.meta_path.insert(0, RefuseTorchOrNumpy())
for forbidden_module in ("torch", "numpy"):
    try:
        __import__(forbidden_module)
    except RuntimeError:
        pass
    else:
        raise AssertionError(f"guard accepted {forbidden_module}")

for module_name in sys.argv[1:]:
    __import__(module_name)
"""
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "src"), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            _TRAINING_STEP_RECORD_MODULE,
            _CALIBRATION_RECORD_MODULE,
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
