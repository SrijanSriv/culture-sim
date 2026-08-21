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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..model.params import FREE_PARAM_NAMES, FreeParams, PriorBox

__all__ = [
    "CrossCultureResult",
    "posterior_overlap",
    "marginal_overlaps",
    "run_cross_culture",
]

# Overlap coefficient above this (joint, mean of marginals) counts as "neighbours".
PASS_JOINT_OVERLAP = 0.1
PASS_MEAN_MARGINAL = 0.25
# When only one full posterior exists, parameter distance in prior-span units.
PASS_PRIOR_UNIT_DISTANCE = 0.5


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
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "culture_keys": list(self.culture_keys),
            "per_parameter_overlap": self.per_parameter_overlap,
            "joint_overlap": self.joint_overlap,
            "posterior_means": self.posterior_means.tolist(),
            "posterior_stds": self.posterior_stds.tolist(),
            "passed": self.passed,
            "notes": self.notes,
            "diagnostics": self.diagnostics,
        }


def posterior_overlap(samples_a: np.ndarray, samples_b: np.ndarray) -> float:
    """Overlap coefficient between two posterior sample sets.

    Estimated on the joint via a product of 1-D histogram overlaps (independence
    assumption). That underestimates true joint overlap when parameters correlate,
    but stays defined in 8-D where a joint histogram cannot. Report
    :func:`marginal_overlaps` alongside this number.
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("samples must be (n_draws, n_params)")
    if a.shape[1] != b.shape[1]:
        raise ValueError("sample sets must share parameter dimension")
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    overlaps = []
    for dim in range(a.shape[1]):
        overlaps.append(_histogram_overlap(a[:, dim], b[:, dim]))
    finite = [o for o in overlaps if np.isfinite(o)]
    if not finite:
        return float("nan")
    # Geometric mean of marginal overlaps ≈ joint under independence.
    return float(np.exp(np.mean(np.log(np.maximum(finite, 1e-12)))))


def marginal_overlaps(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    names: Sequence[str] = FREE_PARAM_NAMES,
) -> dict[str, float]:
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    if a.shape[1] != len(names):
        raise ValueError("names length must match parameter dimension")
    return {str(names[i]): float(_histogram_overlap(a[:, i], b[:, i])) for i in range(a.shape[1])}


def run_cross_culture(
    posteriors: Sequence[Any] | None = None,
    *,
    samples_by_culture: Mapping[str, np.ndarray] | None = None,
    point_estimates: Mapping[str, FreeParams] | None = None,
    prior: PriorBox | None = None,
    **kwargs: Any,
) -> CrossCultureResult:
    """Compare two (or more) culture fits.

    Preferred input: ``samples_by_culture`` with ≥2 keys, each an ``(n, 8)`` sample
    matrix from an independent SBI posterior.

    Fallback when only culture A's posterior exists: pass A's samples plus a
    ``point_estimates`` entry for culture B (e.g. coarse fit). Overlap is then
    undefined; we report prior-normalised distance between A's mean and B's point
    and refuse to claim a passing posterior overlap.
    """
    del posteriors, kwargs
    if samples_by_culture is None or len(samples_by_culture) < 1:
        raise ValueError("run_cross_culture needs samples_by_culture with ≥1 culture")

    keys = tuple(sorted(samples_by_culture))
    stacked = {k: np.asarray(samples_by_culture[k], dtype=np.float64) for k in keys}
    for key, arr in stacked.items():
        if arr.ndim != 2 or arr.shape[1] != len(FREE_PARAM_NAMES):
            raise ValueError(f"{key}: expected (n, {len(FREE_PARAM_NAMES)}) samples")

    means = np.vstack([arr.mean(axis=0) for arr in stacked.values()])
    stds = np.vstack([arr.std(axis=0, ddof=1) for arr in stacked.values()])

    if len(keys) >= 2:
        per = marginal_overlaps(stacked[keys[0]], stacked[keys[1]])
        joint = posterior_overlap(stacked[keys[0]], stacked[keys[1]])
        mean_marginal = float(np.nanmean(list(per.values())))
        passed = bool(
            np.isfinite(joint)
            and joint >= PASS_JOINT_OVERLAP
            and mean_marginal >= PASS_MEAN_MARGINAL
        )
        notes = (
            f"Pairwise overlap {keys[0]} vs {keys[1]}: joint={joint:.3g}, "
            f"mean marginal={mean_marginal:.3g}."
        )
        return CrossCultureResult(
            culture_keys=keys,
            per_parameter_overlap=per,
            joint_overlap=joint,
            posterior_means=means,
            posterior_stds=stds,
            passed=passed,
            notes=notes,
        )

    # Single posterior + optional point estimate for culture B.
    key_a = keys[0]
    samples_a = stacked[key_a]
    mean_a = samples_a.mean(axis=0)
    diagnostics: dict[str, Any] = {"mode": "single_posterior_plus_point"}

    if point_estimates is None or len(point_estimates) < 1:
        return CrossCultureResult(
            culture_keys=keys,
            per_parameter_overlap={n: float("nan") for n in FREE_PARAM_NAMES},
            joint_overlap=float("nan"),
            posterior_means=means,
            posterior_stds=stds,
            passed=False,
            notes=(
                "Only one culture posterior was available. Cross-culture overlap "
                "needs an independent fit of culture B (SPEC §9.2). Deferred to a "
                "second SBI run (see docs/SBI_REFIT.md)."
            ),
            diagnostics=diagnostics,
        )

    # Pick a B that is not A if possible.
    key_b = next((k for k in point_estimates if k != key_a), next(iter(point_estimates)))
    vec_b = point_estimates[key_b].to_vector()
    if prior is None:
        span = np.ones(len(FREE_PARAM_NAMES), dtype=np.float64)
    else:
        span = np.maximum(prior.high - prior.low, 1e-12)
    dist = float(np.sqrt(np.mean(((mean_a - vec_b) / span) ** 2)))
    diagnostics.update(
        {
            "culture_b": key_b,
            "prior_unit_rms_distance": dist,
            "mean_a": mean_a.tolist(),
            "point_b": vec_b.tolist(),
        }
    )
    # Not a true overlap pass; distance below threshold is only a weak neighbour signal.
    neighbour = dist <= PASS_PRIOR_UNIT_DISTANCE
    notes = (
        f"Culture A posterior mean vs culture B point estimate ({key_b}): "
        f"RMS distance {dist:.3g} in prior-span units "
        f"(threshold {PASS_PRIOR_UNIT_DISTANCE:g}). "
        "This is not posterior overlap — run SBI on culture B for SPEC §9.2."
    )
    return CrossCultureResult(
        culture_keys=(key_a, key_b),
        per_parameter_overlap={n: float("nan") for n in FREE_PARAM_NAMES},
        joint_overlap=float("nan"),
        posterior_means=np.vstack([mean_a, vec_b]),
        posterior_stds=np.vstack(
            [samples_a.std(axis=0, ddof=1), np.full(len(FREE_PARAM_NAMES), np.nan)]
        ),
        passed=False,  # SPEC asks for posterior overlap; refuse a false pass.
        notes=notes + (f" Neighbour signal: {neighbour}." if neighbour else ""),
        diagnostics=diagnostics,
    )


def _histogram_overlap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_bins: int = 40,
) -> float:
    """1-D overlap coefficient ∫ min(p,q) from histograms on a shared grid."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, n_bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(ha, hb) * widths))
