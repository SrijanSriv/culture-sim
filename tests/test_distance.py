"""Fingerprint distance (SPEC §8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.fit.distance import UNDEFINED_Z, ScaleReference, component_distances, distance
from culturesim.stats.fingerprint import Fingerprint, FingerprintSpec


def _fingerprint(spec: FingerprintSpec, **overrides: float) -> Fingerprint:
    values = np.ones(len(spec), dtype=np.float64)
    for histogram in spec.histograms:
        for i, name in enumerate(histogram.names):
            values[spec.index_of(name)] = 1.0 if i == 0 else 0.0
    for name, value in overrides.items():
        values[spec.index_of(name)] = value
    return Fingerprint(values=values, names=spec.names, version=spec.version)


def _scale(spec: FingerprintSpec) -> ScaleReference:
    a = _fingerprint(spec, rate_mean=1.0, burst_rate_per_min=2.0)
    b = _fingerprint(spec, rate_mean=2.0, burst_rate_per_min=4.0)
    return ScaleReference.from_fingerprints([a, b], spec=spec)


def test_identical_fingerprints_have_distance_zero() -> None:
    spec = FingerprintSpec.load()
    scale = _scale(spec)
    target = _fingerprint(spec, rate_mean=1.5)
    assert distance(target, target, scale=scale, spec=spec) == pytest.approx(0.0)


def test_scale_requires_two_cultures() -> None:
    spec = FingerprintSpec.load()
    with pytest.raises(ValueError, match="at least two cultures"):
        ScaleReference.from_fingerprints([_fingerprint(spec)], spec=spec)


def test_distance_requires_a_scale() -> None:
    spec = FingerprintSpec.load()
    fp = _fingerprint(spec)
    with pytest.raises(ValueError, match="ScaleReference"):
        distance(fp, fp, spec=spec)


def test_a_one_sigma_rate_offset_is_order_one() -> None:
    spec = FingerprintSpec.load()
    scale = _scale(spec)
    real = _fingerprint(spec, rate_mean=1.0)
    sim = _fingerprint(spec, rate_mean=1.0 + float(scale.scale[spec.index_of("rate_mean")]))
    # One sigma on a single scalar, zeros elsewhere: RMS is 1/sqrt(n_terms).
    value = distance(sim, real, scale=scale, spec=spec)
    assert 0.0 < value < 1.0


def test_missing_bursts_do_not_score_well() -> None:
    spec = FingerprintSpec.load()
    scale = _scale(spec)
    real = _fingerprint(spec, burst_rate_per_min=3.0)
    silent = _fingerprint(spec, burst_rate_per_min=float("nan"))
    matched = _fingerprint(spec, burst_rate_per_min=3.0)
    assert distance(silent, real, scale=scale, spec=spec) > distance(
        matched, real, scale=scale, spec=spec
    )
    # The penalty for a missing scalar the target has is UNDEFINED_Z, not a drop.
    components = component_distances(silent, real, scale=scale, spec=spec)
    assert components["bursts"] >= UNDEFINED_Z / 10.0


def test_shifted_histogram_is_closer_than_a_disjoint_one() -> None:
    spec = FingerprintSpec.load()
    hist = spec.histogram_for("ibi_seconds")

    def peaked(bin_index: int) -> Fingerprint:
        overrides = {name: 0.0 for name in hist.names}
        overrides[hist.names[bin_index]] = 1.0
        return _fingerprint(spec, **overrides)

    near = peaked(1)
    far = peaked(len(hist.names) - 1)
    target = peaked(0)
    other = peaked(2)
    scale = ScaleReference.from_fingerprints([target, other], spec=spec)
    assert distance(near, target, scale=scale, spec=spec) < distance(
        far, target, scale=scale, spec=spec
    )
