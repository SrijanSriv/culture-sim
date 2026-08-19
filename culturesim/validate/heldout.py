"""Held-out statistics validation (SPEC §9.1).

Task 7.

Fit using only the rate and burst groups of the fingerprint, then check whether the
avalanche exponents and the crackling-noise scaling relation come out right *without
having been fitted*. If they do, the model captured mechanism rather than curve-fitting
-- and that distinction is the difference between a test bench and a lookup table.

Report the result either way (SPEC §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..stats.fingerprint import Fingerprint, FingerprintSpec

__all__ = ["HeldoutResult", "FITTED_GROUPS", "HELDOUT_GROUPS", "run_heldout"]

FITTED_GROUPS = ("rates", "bursts")
HELDOUT_GROUPS = ("avalanches", "branching", "connectivity")


@dataclass(frozen=True)
class HeldoutResult:
    fitted_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]
    predicted: Fingerprint
    observed: Fingerprint
    # Per-statistic z-scores on the held-out groups, in across-culture units.
    heldout_z_scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    notes: str = ""


def run_heldout(
    observed: Fingerprint,
    spec: FingerprintSpec,
    **kwargs: Any,
) -> HeldoutResult:
    raise NotImplementedError("Task 7 (SPEC §9.1)")
