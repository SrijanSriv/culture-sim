"""Delegated avalanche distributions plus local power-law fits (SPEC §6.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interop import cl_analysis
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
    """The CL criticality bin-width convention used by the wrapper."""
    del recording
    return cl_analysis.DEFAULT_CRITICALITY_BIN_SIZE_S


def detect_avalanches(
    recording: SpikeRecording,
    bin_width_s: float | None = None,
) -> Avalanches:
    """Avalanche sizes and durations from CL criticality analysis."""
    if recording.n_spikes == 0:
        return Avalanches(np.array([], dtype=np.int64), np.array([], dtype=np.int64), float("nan"))
    bin_width_s = avalanche_bin_width(recording) if bin_width_s is None else bin_width_s
    try:
        result = cl_analysis.analyse_criticality(recording, bin_size_sec=bin_width_s)
    except ValueError:
        return Avalanches(np.array([], dtype=np.int64), np.array([], dtype=np.int64), bin_width_s)
    dump = result.model_dump()
    return Avalanches(
        sizes=np.asarray(dump["avalanche_spike_counts"], dtype=np.int64),
        durations=np.asarray(dump["avalanche_durations"], dtype=np.int64),
        bin_width_s=float(dump["bin_size_sec"]),
    )


def fit_power_law(samples: np.ndarray, *, discrete: bool = True) -> PowerLawFit:
    """Clauset MLE fit via the ``powerlaw`` package, with xmin estimation.

    Sizes and durations are counts, so ``discrete=True`` is correct for both;
    fitting them as continuous biases the exponent.
    """
    samples = np.asarray(samples)
    samples = samples[np.isfinite(samples)]
    samples = samples[samples > 0]
    if samples.size < 2 or np.unique(samples).size < 2:
        return _empty_power_law_fit(samples)
    try:
        import powerlaw

        fit = powerlaw.Fit(samples, discrete=discrete, verbose=False)
        ratio, p_value = fit.distribution_compare("power_law", "lognormal")
        xmin = float(fit.power_law.xmin)
        n_tail = int(np.count_nonzero(samples >= xmin))
        return PowerLawFit(
            exponent=float(fit.power_law.alpha),
            xmin=xmin,
            n_tail=n_tail,
            loglik_ratio_lognormal=float(ratio),
            p_value_lognormal=float(p_value),
        )
    except Exception:
        return _empty_power_law_fit(samples)


def _empty_power_law_fit(samples: np.ndarray) -> PowerLawFit:
    return PowerLawFit(
        exponent=float("nan"),
        xmin=float("nan"),
        n_tail=int(samples.size),
        loglik_ratio_lognormal=float("nan"),
        p_value_lognormal=float("nan"),
    )


def scaling_relation(avalanches: Avalanches) -> tuple[float, float]:
    """Fit ``<S>(D) ~ D**gamma``; returns ``(gamma, r_squared)``.

    Fitted on log-binned average size given duration so that the many short
    avalanches do not dominate the regression.
    """
    if avalanches.sizes.size < 2:
        return float("nan"), float("nan")
    durations = np.asarray(avalanches.durations, dtype=np.float64)
    sizes = np.asarray(avalanches.sizes, dtype=np.float64)
    valid = (durations > 0) & (sizes > 0)
    durations, sizes = durations[valid], sizes[valid]
    unique = np.unique(durations)
    if unique.size < 2:
        return float("nan"), float("nan")
    mean_sizes = np.asarray([np.mean(sizes[durations == d]) for d in unique], dtype=np.float64)
    x = np.log(unique)
    y = np.log(mean_sizes)
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def avalanche_stats(recording: SpikeRecording) -> AvalancheStats:
    avalanches = detect_avalanches(recording)
    size_fit = fit_power_law(avalanches.sizes)
    duration_fit = fit_power_law(avalanches.durations)
    gamma_fit, _ = scaling_relation(avalanches)
    predicted = (
        (duration_fit.exponent - 1.0) / (size_fit.exponent - 1.0)
        if np.isfinite(size_fit.exponent)
        and np.isfinite(duration_fit.exponent)
        and size_fit.exponent != 1.0
        else float("nan")
    )
    discrepancy = (
        abs(gamma_fit - predicted)
        if np.isfinite(gamma_fit) and np.isfinite(predicted)
        else float("nan")
    )
    return AvalancheStats(
        avalanche_alpha=size_fit.exponent,
        avalanche_beta=duration_fit.exponent,
        avalanche_gamma_fit=gamma_fit,
        avalanche_gamma_predicted=predicted,
        avalanche_scaling_discrepancy=discrepancy,
        avalanche_size_xmin=size_fit.xmin,
        avalanche_duration_xmin=duration_fit.xmin,
        avalanche_size_loglik_ratio_lognormal=size_fit.loglik_ratio_lognormal,
        avalanche_duration_loglik_ratio_lognormal=duration_fit.loglik_ratio_lognormal,
        avalanches=avalanches,
    )
