"""Fingerprint distance (SPEC §8.1).

Task 5.

Each statistic is z-scored by its **across-culture** variability in the real dataset,
not its within-culture variability. This is the load-bearing choice: within-culture
scatter for a stable statistic can be tiny, which would make the distance explode over
differences far smaller than the biological spread between two healthy cultures, and
that single statistic would then dominate the fit. Across-culture scale makes the
tolerance mean "as close as two real cultures are to each other".

Distributional components (the log-spaced histogram bins) are compared by Wasserstein
distance rather than bin-by-bin, so that a distribution shifted by one bin scores as
nearly-right instead of as wrong in two bins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..stats.fingerprint import Fingerprint, FingerprintSpec

__all__ = ["ScaleReference", "distance", "component_distances"]


@dataclass(frozen=True)
class ScaleReference:
    """Per-statistic across-culture scale, estimated from several real cultures.

    Built from a set of real fingerprints, one per culture. Fitting to a single
    culture leaves no way to estimate this, which is the first of several reasons
    SPEC §14 warns against it.
    """

    names: tuple[str, ...]
    scale: np.ndarray  # across-culture std per statistic
    center: np.ndarray  # across-culture median per statistic
    n_cultures: int

    @classmethod
    def from_fingerprints(
        cls,
        fingerprints: Sequence[Fingerprint],
        *,
        min_scale: float = 1e-9,
    ) -> ScaleReference:
        """Estimate the scale from >= 2 real cultures.

        ``min_scale`` floors degenerate statistics so a zero-variance entry cannot
        divide the distance by zero; such statistics are reported, not hidden.
        """
        raise NotImplementedError("Task 5 (SPEC §8.1)")


def distance(
    fp_sim: Fingerprint,
    fp_real: Fingerprint,
    weights: Mapping[str, float] | np.ndarray | None = None,
    *,
    scale: ScaleReference | None = None,
    spec: FingerprintSpec | None = None,
) -> float:
    """Weighted distance between two fingerprints.

    Default weights are uniform *after* z-scoring (SPEC §8.1). NaN entries -- the
    documented sentinel for an undefined statistic -- are excluded from the sum and
    counted, so a simulation that produces no bursts at all cannot score well by
    having nothing to compare.
    """
    raise NotImplementedError("Task 5 (SPEC §8.1)")


def component_distances(
    fp_sim: Fingerprint,
    fp_real: Fingerprint,
    *,
    scale: ScaleReference | None = None,
    spec: FingerprintSpec | None = None,
) -> dict[str, float]:
    """Per-group contributions to the total distance, for diagnosis and the report."""
    raise NotImplementedError("Task 5 (SPEC §8.1)")
