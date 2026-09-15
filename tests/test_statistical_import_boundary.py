"""A training worker can validate statistical contracts without installing SciPy."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("module", ["seed_sizing", "ladder_registration"])
def test_contract_import_does_not_require_the_analysis_numerical_stack(module: str) -> None:
    script = """
import importlib.abc
import sys
class RefuseScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'scipy' or fullname.startswith('scipy.'):
            raise RuntimeError('training worker must not import scipy')
sys.meta_path.insert(0, RefuseScipy())
__import__('entropic_substrate.analysis.' + sys.argv[1])
"""
    result = subprocess.run([sys.executable, "-c", script, module], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
