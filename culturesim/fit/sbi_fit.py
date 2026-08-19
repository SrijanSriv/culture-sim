"""Simulation-based inference with SNPE-C (SPEC §8.3).

Task 6.

The posterior is the deliverable, not a point estimate. Read the marginals honestly:

* A tight marginal means the data genuinely identifies that parameter.
* A flat marginal means the fingerprint cannot see that parameter. That is a
  **finding** about what MEA statistics constrain, not a failure of the fit, and it
  must be reported as such -- collapsing it to a MAP value would assert precision the
  data does not support.
* Pairwise posterior correlations reveal degeneracies: two parameters that trade off
  along a ridge are jointly constrained but individually not, which is a different
  statement again.

Posterior predictive checks close the loop: sample parameters from the posterior,
simulate, and confirm the resulting fingerprints bracket the real one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..model.params import FreeParams, ModelParams, PriorBox
from ..stats.fingerprint import Fingerprint

__all__ = [
    "SBIResult",
    "PosteriorSummary",
    "simulate_training_set",
    "train_posterior",
    "posterior_predictive_check",
]


@dataclass(frozen=True)
class PosteriorSummary:
    """Per-parameter marginal summary, plus the identifiability verdict."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    quantiles: Mapping[float, np.ndarray]
    prior_std: np.ndarray
    correlations: np.ndarray  # (n_params, n_params) pairwise posterior correlation
    identified: Mapping[str, bool]
    std_ratio_threshold: float

    def identified_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if self.identified[n])

    def unidentified_names(self) -> tuple[str, ...]:
        """Parameters the fingerprint cannot constrain. Goes in the README verbatim."""
        return tuple(n for n in self.names if not self.identified[n])


@dataclass(frozen=True)
class SBIResult:
    posterior: Any  # sbi DirectPosterior; kept untyped so importing this is cheap
    summary: PosteriorSummary
    observed_fingerprint: Fingerprint
    n_simulations: int
    n_excluded: int  # draws that crashed or returned non-finite fingerprints
    theta: np.ndarray
    fingerprints: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        raise NotImplementedError("Task 6 (SPEC §8.3)")

    @classmethod
    def load(cls, path: str | Path) -> SBIResult:
        raise NotImplementedError("Task 6 (SPEC §8.3)")


def simulate_training_set(
    base: ModelParams,
    prior: PriorBox,
    n_simulations: int,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw from the prior, simulate, and return ``(theta, fingerprints, n_excluded)``.

    Draws whose simulation crashes or yields a non-finite fingerprint are excluded and
    counted, never silently replaced: the exclusion rate is itself informative about
    which regions of the prior box the model cannot run in.
    """
    raise NotImplementedError("Task 6 (SPEC §8.3)")


def train_posterior(
    theta: np.ndarray,
    fingerprints: np.ndarray,
    observed: Fingerprint,
    config: Mapping[str, Any],
) -> SBIResult:
    """Train SNPE-C with an embedding net over the fingerprint and condition on ``observed``."""
    raise NotImplementedError("Task 6 (SPEC §8.3)")


def posterior_predictive_check(
    result: SBIResult,
    base: ModelParams,
    n_draws: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """Simulate from posterior samples and test whether the real fingerprint is bracketed.

    Returns per-statistic coverage. A statistic the posterior predictive misses is
    reported as missed -- that is the check doing its job.
    """
    raise NotImplementedError("Task 6 (SPEC §8.3)")


def posterior_to_free_params(samples: np.ndarray) -> list[FreeParams]:
    """Posterior sample matrix -> parameter objects, in ``FREE_PARAM_NAMES`` order."""
    return [FreeParams.from_vector(row) for row in np.atleast_2d(samples)]
