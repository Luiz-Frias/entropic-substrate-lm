"""Validated, rung-specific experiment construction contracts.

These contracts are intentionally separate from the runtime factory.  They
make the scientific identity of a rung explicit before configuration is
translated into trainable modules.
"""

from __future__ import annotations
