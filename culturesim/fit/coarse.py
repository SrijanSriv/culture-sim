"""Coarse fitting: grid search then local optimisation (SPEC §8.2).

Task 5.

The grid stage over ``w_e``, ``g`` and ``tau_rec`` is not optional. It answers a
question no optimiser can: whether the model can produce the target behaviour *at
all*. If nothing in the grid bursts, that is a structural bug in the network -- and an
optimiser handed a structural bug returns a converged-looking point estimate at the
edge of the parameter box, which reads exactly like a hard fitting problem. The grid
also produces the distance-landscape figure required for Task 5 acceptance.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model.params import FreeParams, ModelParams
from ..stats.fingerprint import Fingerprint

__all__ = ["GridResult", "CoarseFitResult", "grid_search", "local_optimize", "coarse_fit"]

# SPEC §8.2: the three parameters that dominate whether bursting happens at all.
GRID_PARAMETERS = ("w_e", "g", "tau_rec")


@dataclass(frozen=True)
class GridResult:
    """Distances over the parameter grid, and the landscape figure's data."""

    parameter_names: tuple[str, ...]
    grid_values: tuple[np.ndarray, ...]
    distances: np.ndarray  # shape = tuple(len(v) for v in grid_values)
    fingerprints: dict[tuple[int, ...], Fingerprint] = field(default_factory=dict)
    n_failed: int = 0

    @property
    def best_index(self) -> tuple[int, ...]:
        flat_best = int(np.nanargmin(self.distances))
        return tuple(int(i) for i in np.unravel_index(flat_best, self.distances.shape))

    @property
    def any_bursting(self) -> bool:
        """False means the model cannot burst anywhere in the grid -- a structural bug."""
        return bool(np.any(np.isfinite(self.distances)))


@dataclass(frozen=True)
class CoarseFitResult:
    best: FreeParams
    best_distance: float
    baseline_distance: float  # hand-tuned Task 1 parameters, for the >=50% acceptance test
    grid: GridResult | None
    optimizer: str
    n_evaluations: int
    history: list[tuple[FreeParams, float]] = field(default_factory=list)

    @property
    def improvement_fraction(self) -> float:
        if not np.isfinite(self.baseline_distance) or self.baseline_distance == 0:
            return float("nan")
        return 1.0 - self.best_distance / self.baseline_distance


def grid_search(
    objective: Callable[[FreeParams], float],
    base: ModelParams,
    grid: Mapping[str, Sequence[float]],
) -> GridResult:
    """Exhaustive search over 2-3 parameters (SPEC §8.2)."""
    raise NotImplementedError("Task 5 (SPEC §8.2)")


def local_optimize(
    objective: Callable[[FreeParams], float],
    start: FreeParams,
    base: ModelParams,
    *,
    method: str = "nelder-mead",
    max_evaluations: int = 400,
) -> CoarseFitResult:
    """Nelder-Mead or CMA-ES over all 8 free parameters for a point estimate.

    A point estimate is a starting point for SBI, not a result. SPEC §8.3: the
    posterior is the deliverable.
    """
    raise NotImplementedError("Task 5 (SPEC §8.2)")


def coarse_fit(
    target: Fingerprint,
    base: ModelParams,
    **kwargs: Any,
) -> CoarseFitResult:
    """Grid stage then local stage, in that order."""
    raise NotImplementedError("Task 5 (SPEC §8.2)")
