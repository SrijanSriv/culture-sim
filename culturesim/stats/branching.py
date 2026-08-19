"""Branching ratio estimation (SPEC §6.4).

Task 3.

The naive descendants/ancestors estimator is severely biased under subsampling,
and the bias is a function of how many electrodes you happen to record from. That
means identical biology measured on a 60-electrode array and a 1024-electrode
array yields different "criticality", and the bias runs toward reporting the
network as more subcritical than it is.

The fingerprint therefore uses the multistep-regression (MR) estimator of Wilting
& Priesemann (2018, Nat Commun 9:2325): compute the autocorrelation ``r_k`` of the
binned population activity for lags ``k = 1..k_max``, fit ``r_k = C * m**k``, and
recover ``m`` from the fitted decay. This is consistent under subsampling.

:func:`naive_branching_ratio` is provided, clearly labelled, for one purpose only:
demonstrating that bias in a figure and in ``tests/test_branching.py``. It must not
enter the fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spiketrains import SpikeRecording

__all__ = [
    "BranchingStats",
    "mr_branching_ratio",
    "naive_branching_ratio",
    "population_activity",
]

DEFAULT_BIN_WIDTH_S = 0.004  # 4 ms, ~the propagation time of one generation
DEFAULT_K_MAX = 150  # lags; must span several autocorrelation times


@dataclass(frozen=True)
class BranchingStats:
    branching_ratio_mr: float
    branching_mr_fit_r2: float
    autocorrelation_time_s: float
    # Labelled, excluded from the fingerprint, kept only to document the bias.
    branching_ratio_naive: float
    lags: np.ndarray
    autocorrelation: np.ndarray


def population_activity(
    recording: SpikeRecording,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
) -> np.ndarray:
    """Array-wide spike count per bin."""
    raise NotImplementedError("Task 3 (SPEC §6.4)")


def mr_branching_ratio(
    recording: SpikeRecording,
    *,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
    k_max: int = DEFAULT_K_MAX,
) -> BranchingStats:
    """MR estimator (Wilting & Priesemann 2018), consistent under subsampling.

    Sentinel: NaN for ``m`` when the activity is too sparse for the regression.
    A Poisson process gives ``m ~ 0`` (SPEC §12).
    """
    raise NotImplementedError("Task 3 (SPEC §6.4)")


def naive_branching_ratio(
    recording: SpikeRecording,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
) -> float:
    """Naive descendants/ancestors estimator. BIASED UNDER SUBSAMPLING.

    Do not use this to characterise a network. It exists so that
    ``tests/test_branching.py`` can demonstrate, on a branching process with a
    known ``m``, that this estimator's answer depends on the observed fraction
    while the MR estimator's does not.
    """
    raise NotImplementedError("Task 3 (SPEC §6.4)")
