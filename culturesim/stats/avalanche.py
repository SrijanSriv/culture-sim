"""Neuronal avalanches and the crackling-noise scaling relation (SPEC §6.3).

Task 3.

Three rules here are not negotiable, because breaking any of them manufactures a
power law out of almost any data (SPEC §14):

1. The bin width is the mean inter-spike interval across the whole array, computed
   from the recording. Hard-coding a bin width can create or destroy a power law.
2. ``xmin`` is estimated by the Clauset MLE procedure (the ``powerlaw`` package),
   never assumed to be 1.
3. Every power-law fit is compared against a lognormal alternative and the
   loglikelihood ratio is reported. A power law is never claimed without it.

The scaling relation is the discriminating statistic: matching alpha alone is easy,
matching ``gamma = (beta - 1)/(alpha - 1)`` is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spiketrains import SpikeRecording

__all__ = [
    "Avalanches",
    "AvalancheStats",
    "avalanche_bin_width",
    "detect_avalanches",
    "fit_power_law",
    "PowerLawFit",
    "scaling_relation",
    "avalanche_stats",
]


@dataclass(frozen=True)
class Avalanches:
    """Avalanche sizes (total spikes) and durations (number of bins)."""

    sizes: np.ndarray
    durations: np.ndarray
    bin_width_s: float

    @property
    def n_avalanches(self) -> int:
        return int(self.sizes.size)


@dataclass(frozen=True)
class PowerLawFit:
    """Clauset MLE fit with its lognormal comparison."""

    exponent: float  # the positive exponent, p(x) ~ x**-exponent
    xmin: float
    n_tail: int  # samples at or above xmin, i.e. what the fit actually used
    loglik_ratio_lognormal: float  # >0 favours the power law
    p_value_lognormal: float


@dataclass(frozen=True)
class AvalancheStats:
    avalanche_alpha: float
    avalanche_beta: float
    avalanche_gamma_fit: float
    avalanche_gamma_predicted: float  # (beta - 1) / (alpha - 1)
    avalanche_scaling_discrepancy: float
    avalanche_size_xmin: float
    avalanche_duration_xmin: float
    avalanche_size_loglik_ratio_lognormal: float
    avalanche_duration_loglik_ratio_lognormal: float
    avalanches: Avalanches


def avalanche_bin_width(recording: SpikeRecording) -> float:
    """Mean inter-spike interval across the whole array, in seconds (SPEC §6.3).

    This is the standard convention and the only bin width this module accepts as
    a default. Sentinel: NaN for fewer than two array-wide spikes.
    """
    raise NotImplementedError("Task 3 (SPEC §6.3)")


def detect_avalanches(
    recording: SpikeRecording,
    bin_width_s: float | None = None,
) -> Avalanches:
    """Runs of consecutive non-empty bins bracketed by empty bins.

    ``bin_width_s=None`` uses :func:`avalanche_bin_width`, which is what the
    fingerprint always does.
    """
    raise NotImplementedError("Task 3 (SPEC §6.3)")


def fit_power_law(samples: np.ndarray, *, discrete: bool = True) -> PowerLawFit:
    """Clauset MLE fit via the ``powerlaw`` package, with xmin estimation.

    Sizes and durations are counts, so ``discrete=True`` is correct for both;
    fitting them as continuous biases the exponent.
    """
    raise NotImplementedError("Task 3 (SPEC §6.3)")


def scaling_relation(avalanches: Avalanches) -> tuple[float, float]:
    """Fit ``<S>(D) ~ D**gamma``; returns ``(gamma, r_squared)``.

    Fitted on log-binned average size given duration so that the many short
    avalanches do not dominate the regression.
    """
    raise NotImplementedError("Task 3 (SPEC §6.3)")


def avalanche_stats(recording: SpikeRecording) -> AvalancheStats:
    raise NotImplementedError("Task 3 (SPEC §6.3)")
