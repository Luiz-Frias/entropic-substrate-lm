"""A training worker can validate statistical contracts without installing SciPy.

P13-Q2-BOUNDARY-SKIP (2026-09-26): the argv-built module prefix was
'entropic_substrate.analysis.' -- a bare package name no ruled map source
resolves (the rewrite tool never sees it: it is built at runtime by string
concatenation, not a static AST import target or a whole quoted chain). The
two parametrized leaves, 'seed_sizing' and 'ladder_registration', are both
measured to shim at 'entropic_substrate.ladder.analyze.<leaf>', so B2
hand-edits the prefix to that landed package directly.
"""

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
__import__('entropic_substrate.ladder.analyze.' + sys.argv[1])
"""
    result = subprocess.run([sys.executable, "-c", script, module], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
