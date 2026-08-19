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
    if bin_width_s <= 0:
        raise ValueError(f"bin_width_s must be positive, got {bin_width_s}")
    n_bins = max(1, int(np.ceil(recording.duration / bin_width_s)))
    edges = np.linspace(0.0, n_bins * bin_width_s, n_bins + 1)
    counts, _ = np.histogram(recording.times, bins=edges)
    return counts.astype(np.float64)


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
    activity = population_activity(recording, bin_width_s)
    lags = np.arange(1, min(k_max, activity.size - 1) + 1, dtype=np.int64)
    if lags.size == 0 or np.var(activity) <= 0:
        empty = np.array([], dtype=np.float64)
        return BranchingStats(
            float("nan"),
            float("nan"),
            float("nan"),
            naive_branching_ratio(recording),
            lags,
            empty,
        )

    demeaned = activity - np.mean(activity)
    denom = float(np.dot(demeaned, demeaned))
    autocorr = np.asarray(
        [float(np.dot(demeaned[:-lag], demeaned[lag:]) / denom) for lag in lags],
        dtype=np.float64,
    )
    if autocorr.size == 0 or not np.isfinite(autocorr[0]) or autocorr[0] < 0.05:
        return BranchingStats(
            branching_ratio_mr=0.0,
            branching_mr_fit_r2=0.0,
            autocorrelation_time_s=0.0,
            branching_ratio_naive=naive_branching_ratio(recording, bin_width_s),
            lags=lags,
            autocorrelation=autocorr,
        )

    positive = np.isfinite(autocorr) & (autocorr > 0.0)
    if np.count_nonzero(positive) < 2:
        m = float(np.clip(autocorr[0], 0.0, 1.0))
        return BranchingStats(
            branching_ratio_mr=m,
            branching_mr_fit_r2=float("nan"),
            autocorrelation_time_s=_autocorr_time(m, bin_width_s),
            branching_ratio_naive=naive_branching_ratio(recording, bin_width_s),
            lags=lags,
            autocorrelation=autocorr,
        )

    fit_lags = lags[positive].astype(np.float64)
    fit_y = np.log(autocorr[positive])
    slope, intercept = np.polyfit(fit_lags, fit_y, deg=1)
    predicted = slope * fit_lags + intercept
    ss_res = float(np.sum((fit_y - predicted) ** 2))
    ss_tot = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    m = float(np.clip(np.exp(slope), 0.0, 1.0))
    return BranchingStats(
        branching_ratio_mr=m,
        branching_mr_fit_r2=float(r2),
        autocorrelation_time_s=_autocorr_time(m, bin_width_s),
        branching_ratio_naive=naive_branching_ratio(recording, bin_width_s),
        lags=lags,
        autocorrelation=autocorr,
    )


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
    activity = population_activity(recording, bin_width_s)
    if activity.size < 2:
        return float("nan")
    ancestors = activity[:-1]
    descendants = activity[1:]
    mask = ancestors > 0
    if not np.any(mask):
        return float("nan")
    active_fraction = float(np.mean(recording.spike_counts() > 0))
    return float(np.sum(descendants[mask]) / np.sum(ancestors[mask]) * active_fraction)


def _autocorr_time(m: float, bin_width_s: float) -> float:
    if not np.isfinite(m) or m <= 0.0:
        return float("nan")
    if m >= 1.0:
        return float("inf")
    return float(-bin_width_s / np.log(m))
