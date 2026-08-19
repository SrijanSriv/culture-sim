"""Cross-culture validation (SPEC §9.2).

Task 7.

Fit culture A, independently fit culture B, and check whether the posteriors are
neighbours in parameter space rather than scattered. Quantified as posterior overlap,
so the claim is a number rather than an impression from two plots.

If the posteriors do not overlap, the honest reading is that the model has no single
parameter regime describing dissociated cultures in general, and the fitted values are
culture-specific. Fitting one culture and claiming generality is the failure mode this
test exists to catch (SPEC §14).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CrossCultureResult", "posterior_overlap", "run_cross_culture"]


@dataclass(frozen=True)
class CrossCultureResult:
    culture_keys: tuple[str, ...]
    # Pairwise overlap per parameter and overall, in [0, 1].
    per_parameter_overlap: dict[str, float]
    joint_overlap: float
    posterior_means: np.ndarray  # (n_cultures, n_params)
    posterior_stds: np.ndarray
    passed: bool = False
    notes: str = ""


def posterior_overlap(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Overlap coefficient between two posterior sample sets.

    Estimated on the marginals and the joint; the joint estimate is the honest one but
    degrades with dimension, so both are reported.
    """
    raise NotImplementedError("Task 7 (SPEC §9.2)")


def run_cross_culture(posteriors: Sequence[Any], **kwargs: Any) -> CrossCultureResult:
    raise NotImplementedError("Task 7 (SPEC §9.2)")
