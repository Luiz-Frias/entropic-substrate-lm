"""RED-first guard-teeth for the RunClass.SMOKE refusal (p7-register-gate
brief ACCEPTANCE: "experiments/d1_launch_attempts.py's non-PILOT/non-
CONFIRMATORY refusal is TEST-PINNED"). MEASURED 2026-09-23: no existing test
reaches ``D1LaunchAttemptIdentity.__post_init__``'s ``:71`` raise --
``tests/test_d1_launch_attempt_ledger.py``'s ``registered_design_digest``
and ``"pilot"``-matching cases both exercise the DIFFERENT branch at ``:80``/
``:86``; no test anywhere constructs a ``D1LaunchAttemptIdentity`` with
``run_class=RunClass.SMOKE``, the only third member of ``RunClass``.

Architect constraint (design gate 2026-09-23): every OTHER field is
OTHERWISE-VALID -- a real sha256 ``family_digest``, a real ``D1Arm``, a
nonnegative int ``seed``, a nonempty ``logical_run_id``,
``registered_design_digest=None`` -- so ``RunClass.SMOKE`` is the ONLY
reason construction can raise. If any other field were invalid, the
unmutated code would raise at ``:71`` first for the wrong reason (five
further raise sites follow it: ``require_sha256`` on ``family_digest``, the
``D1Arm`` ``TypeError``, the seed check, ``require_nonempty_string`` on
``logical_run_id``, and the ``:80``/``:86`` digest arms), and a later
mutation that drops ``RunClass.SMOKE`` from the rejected set would raise a
DIFFERENT ``ValueError`` whose message still fails the ``match=`` -- RED for
a reason that has nothing to do with the refusal this test pins.
"""

from __future__ import annotations

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.register.d1_launch_attempts import D1LaunchAttemptIdentity

pytestmark = pytest.mark.unit

_SMOKE_FAMILY_DIGEST = "c" * 64  # pragma: allowlist secret


def test_smoke_run_class_is_refused_by_the_attempt_ledger_identity() -> None:
    """Every field but ``run_class`` is otherwise-valid; the raise at
    ``d1_launch_attempts.py:71`` must be the one that fires."""
    with pytest.raises(ValueError, match="records only pilot or confirmatory"):
        D1LaunchAttemptIdentity(
            run_class=RunClass.SMOKE,
            family_digest=_SMOKE_FAMILY_DIGEST,
            arm=D1Arm.NONE,
            seed=0,
            logical_run_id="d1-smoke-refusal-probe",
            registered_design_digest=None,
        )
