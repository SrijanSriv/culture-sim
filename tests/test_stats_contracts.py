"""The SPEC §12 statistics tests, as explicit obligations.

Every test here is the *real* test SPEC §12 requires -- against an input whose answer
is known analytically -- with the assertion written out and marked ``xfail(strict=True,
raises=NotImplementedError)`` while the statistic is unimplemented.

``strict=True`` is the point: the moment Task 3 implements a function, its test stops
raising and pytest reports XPASS as a failure. That forces the marker to be removed
deliberately rather than leaving a permanently-green placeholder, which is how a test
suite quietly stops testing anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.stats.avalanche import avalanche_bin_width, avalanche_stats, fit_power_law
from culturesim.stats.branching import mr_branching_ratio, naive_branching_ratio
from culturesim.stats.bursts import burst_stats
from culturesim.stats.connectivity import connectivity_stats
from culturesim.stats.fingerprint import FingerprintSpec, compute_fingerprint
from culturesim.stats.rates import isi_cv, rate_stats
from culturesim.stats.spiketrains import SpikeRecording

todo = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="Task 3 (SPEC §6)")


# -- SPEC §12: analytically known answers ---------------------------------


@todo
def test_poisson_isi_cv_is_one(poisson_recording: SpikeRecording) -> None:
    """A homogeneous Poisson process has exponential ISIs, so CV = 1."""
    assert rate_stats(poisson_recording).isi_cv_pooled == pytest.approx(1.0, abs=0.1)


@todo
def test_regular_train_isi_cv_is_zero(regular_recording: SpikeRecording) -> None:
    assert isi_cv(regular_recording.times_of(0)) == pytest.approx(0.0, abs=1e-9)


@todo
def test_poisson_branching_ratio_is_zero(poisson_recording: SpikeRecording) -> None:
    """Poisson activity has no propagation, so the MR estimator must return ~0."""
    assert mr_branching_ratio(poisson_recording).branching_ratio_mr == pytest.approx(0.0, abs=0.05)


@todo
def test_mr_estimator_recovers_a_known_branching_ratio() -> None:
    """SPEC §12: within 5% on a directly simulated critical branching process."""
    recording = _branching_process(m=0.95, n_channels=60, duration_s=300.0, seed=1)
    assert mr_branching_ratio(recording).branching_ratio_mr == pytest.approx(0.95, rel=0.05)


@todo
def test_naive_estimator_is_biased_under_subsampling() -> None:
    """SPEC §6.4/§12: this test exists to *document* the naive estimator's bias.

    The same branching process observed through fewer electrodes must move the naive
    estimate (toward subcritical) while leaving the MR estimate intact. If this ever
    stops holding, the demonstration figure is wrong.
    """
    m_true = 0.95
    full = _branching_process(m=m_true, n_channels=1024, duration_s=300.0, seed=1)
    subsampled = full.drop_channels(np.arange(60, 1024))

    naive_full = naive_branching_ratio(full)
    naive_sub = naive_branching_ratio(subsampled)
    mr_full = mr_branching_ratio(full).branching_ratio_mr
    mr_sub = mr_branching_ratio(subsampled).branching_ratio_mr

    assert naive_sub < naive_full, "the naive estimator must be pulled down by subsampling"
    assert naive_sub < m_true * 0.9, "and it must be badly wrong, not marginally so"
    assert mr_sub == pytest.approx(mr_full, rel=0.1), "the MR estimator must be robust"


@todo
def test_power_law_fit_recovers_a_known_exponent() -> None:
    """SPEC §12: synthetic power-law sample with a known exponent."""
    rng = np.random.default_rng(0)
    alpha = 2.5
    samples = np.floor((1.0 - rng.random(200_000)) ** (-1.0 / (alpha - 1.0))).astype(np.int64)
    fit = fit_power_law(samples[samples >= 1])
    assert fit.exponent == pytest.approx(alpha, rel=0.05)


@todo
def test_avalanche_bin_width_is_the_mean_isi(poisson_recording: SpikeRecording) -> None:
    """SPEC §6.3: a hard-coded bin width manufactures or destroys power laws."""
    expected = 1.0 / poisson_recording.mean_rate
    assert avalanche_bin_width(poisson_recording) == pytest.approx(expected, rel=0.05)


@todo
def test_avalanche_fit_reports_the_lognormal_comparison(poisson_recording: SpikeRecording) -> None:
    """SPEC §6.3: never claim a power law without the lognormal loglikelihood ratio."""
    stats = avalanche_stats(poisson_recording)
    assert np.isfinite(stats.avalanche_size_loglik_ratio_lognormal)


@todo
def test_scaling_relation_discrepancy_is_reported(poisson_recording: SpikeRecording) -> None:
    """SPEC §6.3: |gamma - (beta - 1)/(alpha - 1)| is the discriminating statistic."""
    stats = avalanche_stats(poisson_recording)
    predicted = (stats.avalanche_beta - 1.0) / (stats.avalanche_alpha - 1.0)
    assert stats.avalanche_gamma_predicted == pytest.approx(predicted)
    assert stats.avalanche_scaling_discrepancy == pytest.approx(
        abs(stats.avalanche_gamma_fit - predicted)
    )


# -- SPEC §12: degenerate inputs return sentinels, never crash -------------


@pytest.mark.parametrize("case", ["empty", "single_spike", "single_channel"])
@todo
def test_rate_stats_survive_degenerate_input(case: str, edge_case_recordings) -> None:
    stats = rate_stats(edge_case_recordings[case])
    assert np.isfinite(stats.rate_mean), "a rate is always defined; it may be 0"
    assert stats.per_electrode_rates.size == edge_case_recordings[case].n_channels


@pytest.mark.parametrize("case", ["empty", "single_spike", "single_channel"])
@todo
def test_burst_stats_survive_degenerate_input(case: str, edge_case_recordings, rng) -> None:
    stats = burst_stats(edge_case_recordings[case], rng)
    assert stats.burst_rate_per_min == pytest.approx(0.0)
    assert stats.ibi_seconds.size == 0


@pytest.mark.parametrize("case", ["empty", "single_spike", "single_channel"])
@todo
def test_connectivity_stats_survive_degenerate_input(case: str, edge_case_recordings, rng) -> None:
    stats = connectivity_stats(edge_case_recordings[case], rng)
    recording = edge_case_recordings[case]
    assert stats.adjacency.shape == (recording.n_channels, recording.n_channels)
    assert not stats.adjacency.diagonal().any(), "no self-loops"


@pytest.mark.parametrize("case", ["empty", "single_spike", "single_channel"])
@todo
def test_fingerprint_survives_degenerate_input(case: str, edge_case_recordings) -> None:
    """Undefined statistics become NaN sentinels; the vector keeps its full length."""
    spec = FingerprintSpec.load("fingerprint.yaml")
    fingerprint = compute_fingerprint(edge_case_recordings[case], spec)
    assert len(fingerprint) == len(spec)
    assert fingerprint.names == spec.names


@todo
def test_fingerprint_computes_within_the_time_budget(poisson_recording: SpikeRecording) -> None:
    """Task 3 acceptance: end-to-end in under 10 s on simulated data."""
    import time

    spec = FingerprintSpec.load("fingerprint.yaml")
    started = time.perf_counter()
    compute_fingerprint(poisson_recording, spec)
    assert time.perf_counter() - started < 10.0


def _branching_process(
    m: float, n_channels: int, duration_s: float, seed: int, bin_width_s: float = 0.004
) -> SpikeRecording:
    """A directly simulated critical branching process with a known ``m``.

    Each spike independently produces ``Poisson(m)`` descendants in the next bin, so
    the true branching ratio is ``m`` by construction rather than by fitting. Neurons
    are assigned to channels uniformly, which is what makes dropping channels a clean
    subsampling experiment.
    """
    rng = np.random.default_rng(seed)
    n_bins = int(duration_s / bin_width_s)
    drive = 5.0  # spontaneous spikes per bin, keeps a subcritical process alive
    activity = np.zeros(n_bins, dtype=np.int64)
    ancestors = 0
    for index in range(n_bins):
        descendants = rng.poisson(m * ancestors) if ancestors else 0
        ancestors = int(descendants + rng.poisson(drive))
        activity[index] = ancestors

    times = np.concatenate(
        [
            rng.uniform(index * bin_width_s, (index + 1) * bin_width_s, size=count)
            for index, count in enumerate(activity)
            if count
        ]
    )
    order = np.argsort(times, kind="stable")
    return SpikeRecording(
        times=times[order],
        channels=rng.integers(0, n_channels, size=times.size).astype(np.int32)[order],
        n_channels=n_channels,
        duration=duration_s,
        source="synthetic-branching",
        metadata={"m_true": m, "bin_width_s": bin_width_s},
    )
