"""Evidence-domain contracts for the D1 experimental flywheel.

The package deliberately owns only immutable records, validation, and pure
decisions.  Capture, analysis, review-packet assembly, CLI wiring, and DVC
transport remain adapters around this domain core.
"""

from __future__ import annotations
