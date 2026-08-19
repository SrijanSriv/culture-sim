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
from scipy.stats import wasserstein_distance

from ..stats.fingerprint import Fingerprint, FingerprintSpec, HistogramSpec

__all__ = ["ScaleReference", "UNDEFINED_Z", "distance", "component_distances"]

# A statistic the target has and the simulation does not is scored as this many
# across-culture sigmas, not dropped. Dropping it would let a silent network win
# by having nothing to compare (SPEC §8.1).
UNDEFINED_Z = 5.0


@dataclass(frozen=True)
class ScaleReference:
    """Per-statistic across-culture scale, estimated from several real cultures.

    Built from a set of real fingerprints, one per culture. Fitting to a single
    culture leaves no way to estimate this, which is the first of several reasons
    SPEC §14 warns against it.
    """

    names: tuple[str, ...]
    scale: np.ndarray  # across-culture std per statistic (hist bins unused)
    center: np.ndarray  # across-culture median per statistic
    n_cultures: int
    histogram_scale: dict[str, float]
    histogram_bin_names: frozenset[str]

    def __post_init__(self) -> None:
        scale = np.ascontiguousarray(np.asarray(self.scale, dtype=np.float64)).copy()
        center = np.ascontiguousarray(np.asarray(self.center, dtype=np.float64)).copy()
        scale.setflags(write=False)
        center.setflags(write=False)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        object.__setattr__(self, "histogram_scale", dict(self.histogram_scale))
        object.__setattr__(self, "histogram_bin_names", frozenset(self.histogram_bin_names))
        if scale.size != len(self.names) or center.size != len(self.names):
            raise ValueError("scale/center length must match names")

    def require_names(self, names: tuple[str, ...]) -> None:
        if names != self.names:
            raise ValueError("ScaleReference names do not match the fingerprint order")

    @classmethod
    def from_fingerprints(
        cls,
        fingerprints: Sequence[Fingerprint],
        *,
        spec: FingerprintSpec | None = None,
        min_scale: float = 1e-9,
    ) -> ScaleReference:
        """Estimate the scale from >= 2 real cultures.

        ``min_scale`` floors degenerate statistics so a zero-variance entry cannot
        divide the distance by zero; such statistics are reported, not hidden.
        """
        if len(fingerprints) < 2:
            raise ValueError(
                "across-culture scale needs fingerprints from at least two cultures "
                "(SPEC §8.1, §14). Fitting a single recording cannot estimate it."
            )
        names = fingerprints[0].names
        for fingerprint in fingerprints[1:]:
            if fingerprint.names != names:
                raise ValueError("fingerprints used for scale must share the same names")
        stack = np.vstack([fp.values for fp in fingerprints])
        with np.errstate(all="ignore"):
            center = np.nanmedian(stack, axis=0)
            scale = np.nanstd(stack, axis=0, ddof=1)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)
        scale = np.where(np.isfinite(scale), np.maximum(scale, min_scale), min_scale)

        spec = spec if spec is not None else FingerprintSpec.load()
        bin_names = _histogram_bin_names(spec)
        histogram_scale: dict[str, float] = {}
        for histogram in spec.histograms:
            pairwise = _pairwise_wasserstein(fingerprints, histogram)
            if pairwise.size == 0 or not np.any(np.isfinite(pairwise)):
                histogram_scale[histogram.stat] = min_scale
            else:
                histogram_scale[histogram.stat] = float(max(min_scale, np.nanmedian(pairwise)))
        return cls(
            names=names,
            scale=scale,
            center=center,
            n_cultures=len(fingerprints),
            histogram_scale=histogram_scale,
            histogram_bin_names=bin_names,
        )


def distance(
    fp_sim: Fingerprint,
    fp_real: Fingerprint,
    weights: Mapping[str, float] | np.ndarray | None = None,
    *,
    scale: ScaleReference | None = None,
    spec: FingerprintSpec | None = None,
) -> float:
    """Weighted RMS of z-scored components (SPEC §8.1).

    Default weights are uniform *after* z-scoring. NaN entries -- the documented
    sentinel for an undefined statistic -- are excluded when both sides lack them,
    and penalised by :data:`UNDEFINED_Z` when the target has the statistic and the
    simulation does not, so a silent network cannot score well by having nothing
    to compare.
    """
    terms = _z_terms(fp_sim, fp_real, weights=weights, scale=scale, spec=spec)
    if not terms:
        return float("nan")
    z = np.array([item[0] for item in terms], dtype=np.float64)
    w = np.array([item[1] for item in terms], dtype=np.float64)
    if not np.any(w > 0):
        return float("nan")
    return float(np.sqrt(np.average(np.square(z), weights=w)))


def component_distances(
    fp_sim: Fingerprint,
    fp_real: Fingerprint,
    *,
    scale: ScaleReference | None = None,
    spec: FingerprintSpec | None = None,
) -> dict[str, float]:
    """Per-group RMS of z-scored components, for diagnosis and the report."""
    spec = spec if spec is not None else FingerprintSpec.load()
    grouped: dict[str, list[tuple[float, float]]] = {group: [] for group in spec.groups}
    for z, weight, group in _z_terms(fp_sim, fp_real, scale=scale, spec=spec):
        grouped.setdefault(group, []).append((z, weight))
    out: dict[str, float] = {}
    for group, terms in grouped.items():
        if not terms:
            out[group] = float("nan")
            continue
        z = np.array([item[0] for item in terms], dtype=np.float64)
        w = np.array([item[1] for item in terms], dtype=np.float64)
        out[group] = (
            float(np.sqrt(np.average(np.square(z), weights=w))) if np.any(w > 0) else float("nan")
        )
    return out


def _z_terms(
    fp_sim: Fingerprint,
    fp_real: Fingerprint,
    weights: Mapping[str, float] | np.ndarray | None = None,
    *,
    scale: ScaleReference | None,
    spec: FingerprintSpec | None,
) -> list[tuple[float, float, str]]:
    if scale is None:
        raise ValueError(
            "distance requires a ScaleReference built from >= 2 real cultures "
            "(SPEC §8.1). Fitting a single recording cannot z-score the vector."
        )
    spec = spec if spec is not None else FingerprintSpec.load()
    fp_sim.require_match(spec)
    fp_real.require_match(spec)
    scale.require_names(fp_sim.names)

    weight_vec = _weight_vector(fp_sim.names, weights)
    group_of = spec.group_of
    terms: list[tuple[float, float, str]] = []
    bin_names = scale.histogram_bin_names

    for i, name in enumerate(fp_sim.names):
        if name in bin_names:
            continue
        sim = float(fp_sim.values[i])
        real = float(fp_real.values[i])
        group = group_of[name]
        weight = float(weight_vec[i])
        z = _scalar_z(sim, real, float(scale.scale[i]))
        if z is None:
            continue
        terms.append((z, weight, group))

    for histogram in spec.histograms:
        sim_hist = _histogram_mass(fp_sim, histogram)
        real_hist = _histogram_mass(fp_real, histogram)
        group = spec.group_of[histogram.names[0]]
        weight = float(weight_vec[spec.index_of(histogram.names[0])])
        hist_scale = float(scale.histogram_scale.get(histogram.stat, 1e-9))
        z = _histogram_z(sim_hist, real_hist, histogram, hist_scale)
        if z is None:
            continue
        terms.append((z, weight, group))
    return terms


def _scalar_z(sim: float, real: float, scale: float) -> float | None:
    sim_ok, real_ok = np.isfinite(sim), np.isfinite(real)
    if not real_ok:
        return None
    if not sim_ok:
        return UNDEFINED_Z
    denom = scale if scale > 0 and np.isfinite(scale) else 1e-9
    return abs(sim - real) / denom


def _histogram_z(
    sim: np.ndarray | None,
    real: np.ndarray | None,
    histogram: HistogramSpec,
    hist_scale: float,
) -> float | None:
    if real is None:
        return None
    if sim is None:
        return UNDEFINED_Z
    locations = 0.5 * (histogram.edges[:-1] + histogram.edges[1:])
    value = wasserstein_distance(locations, locations, u_weights=sim, v_weights=real)
    denom = hist_scale if hist_scale > 0 and np.isfinite(hist_scale) else 1e-9
    return float(value / denom)


def _histogram_mass(fingerprint: Fingerprint, histogram: HistogramSpec) -> np.ndarray | None:
    values = np.array([fingerprint[name] for name in histogram.names], dtype=np.float64)
    if not np.any(np.isfinite(values)):
        return None
    values = np.where(np.isfinite(values), np.clip(values, 0.0, np.inf), 0.0)
    widths = np.diff(histogram.edges)
    mass = values * widths if histogram.normalize == "density" else values
    if float(mass.sum()) <= 0:
        return None
    return mass


def _histogram_bin_names(spec: FingerprintSpec) -> frozenset[str]:
    return frozenset(name for histogram in spec.histograms for name in histogram.names)


def _pairwise_wasserstein(
    fingerprints: Sequence[Fingerprint],
    histogram: HistogramSpec,
) -> np.ndarray:
    masses = [_histogram_mass(fp, histogram) for fp in fingerprints]
    locations = 0.5 * (histogram.edges[:-1] + histogram.edges[1:])
    values: list[float] = []
    for i in range(len(masses)):
        if masses[i] is None:
            continue
        for j in range(i + 1, len(masses)):
            if masses[j] is None:
                continue
            values.append(
                float(
                    wasserstein_distance(
                        locations, locations, u_weights=masses[i], v_weights=masses[j]
                    )
                )
            )
    return np.asarray(values, dtype=np.float64)


def _weight_vector(
    names: tuple[str, ...],
    weights: Mapping[str, float] | np.ndarray | None,
) -> np.ndarray:
    if weights is None:
        return np.ones(len(names), dtype=np.float64)
    if isinstance(weights, Mapping):
        return np.array([float(weights.get(name, 1.0)) for name in names], dtype=np.float64)
    vector = np.asarray(weights, dtype=np.float64).ravel()
    if vector.size != len(names):
        raise ValueError(f"weights have length {vector.size}, fingerprint has {len(names)}")
    return vector
