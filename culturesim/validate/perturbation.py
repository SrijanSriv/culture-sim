"""Perturbation-response validation (SPEC §9.3).

Task 7. **This is the test that actually matters.**

Fit on spontaneous activity only, then ask whether the model predicts the *evoked*
response to electrical stimulation: PSTH shape, response probability as a function of
stimulus amplitude, and post-stimulus network burst probability. None of these are
fitted (SPEC §0 lists evoked response as validation only).

Every downstream project stimulates the culture -- closed-loop timing, decoder drift,
criticality control. A model that only matches resting behaviour is not fit for any of
them. If this test fails, SPEC §9.3 requires saying so plainly in the README rather
than quietly dropping it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..model.params import FreeParams, ModelParams
from ..observation.virtual_mea import ElectrodeLayout
from ..stats.bursts import detect_bursts
from ..stats.spiketrains import SpikeRecording

__all__ = [
    "StimulusProtocol",
    "EvokedResponse",
    "PerturbationResult",
    "evoked_response",
    "protocol_to_stimulus",
    "run_perturbation",
]

# Amplitude in mV injected at the electrode for protocol.normalised == 1.0.
DEFAULT_AMPLITUDE_SCALE_MV = 40.0
RESPONSE_WINDOW_S = (0.002, 0.080)
PSTH_BIN_S = 0.005
BURST_WINDOW_S = 0.500
PASS_PSTH_CORR = 0.3
PASS_AMP_CURVE_ERROR = 0.35
PASS_BURST_ERROR = 0.35


@dataclass(frozen=True)
class StimulusProtocol:
    """Biphasic current stimulation through one electrode.

    Defaults follow common MEA practice: biphasic pulses at 0.2-1.0 V equivalent
    amplitude, inter-stimulus intervals long enough for the network to recover.
    """

    electrode: int
    amplitudes: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # normalised drive
    n_pulses_per_amplitude: int = 50
    inter_stimulus_interval_s: float = 5.0
    pulse_width_ms: float = 0.4
    # Stimulus artefact blanking; real recordings cannot see spikes in this window,
    # so the model must not be credited with spikes there either.
    blanking_ms: float = 2.0

    @property
    def total_duration_s(self) -> float:
        n = len(self.amplitudes) * self.n_pulses_per_amplitude
        return n * self.inter_stimulus_interval_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "electrode": self.electrode,
            "amplitudes": list(self.amplitudes),
            "n_pulses_per_amplitude": self.n_pulses_per_amplitude,
            "inter_stimulus_interval_s": self.inter_stimulus_interval_s,
            "pulse_width_ms": self.pulse_width_ms,
            "blanking_ms": self.blanking_ms,
            "total_duration_s": self.total_duration_s,
        }


@dataclass(frozen=True)
class EvokedResponse:
    amplitude: float
    psth_bin_edges_s: np.ndarray
    psth_hz: np.ndarray
    response_probability: float  # P(>=1 spike in the response window | stimulus)
    mean_latency_s: float
    post_stimulus_burst_probability: float
    n_trials: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "amplitude": self.amplitude,
            "psth_bin_edges_s": self.psth_bin_edges_s.tolist(),
            "psth_hz": self.psth_hz.tolist(),
            "response_probability": self.response_probability,
            "mean_latency_s": self.mean_latency_s,
            "post_stimulus_burst_probability": self.post_stimulus_burst_probability,
            "n_trials": self.n_trials,
        }


@dataclass(frozen=True)
class PerturbationResult:
    protocol: StimulusProtocol
    simulated: tuple[EvokedResponse, ...]
    observed: tuple[EvokedResponse, ...]
    psth_correlation: float
    amplitude_curve_error: float
    burst_probability_error: float
    passed: bool = False
    notes: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "simulated": [r.to_dict() for r in self.simulated],
            "observed": [r.to_dict() for r in self.observed],
            "psth_correlation": self.psth_correlation,
            "amplitude_curve_error": self.amplitude_curve_error,
            "burst_probability_error": self.burst_probability_error,
            "passed": self.passed,
            "notes": self.notes,
            "diagnostics": self.diagnostics,
        }


def protocol_to_stimulus(
    protocol: StimulusProtocol,
    layout: ElectrodeLayout,
    *,
    amplitude_scale_mv: float = DEFAULT_AMPLITUDE_SCALE_MV,
    radius_um: float = 100.0,
    start_s: float = 0.0,
) -> dict[str, Any]:
    """Brian2-ready stimulus dict for :func:`culturesim.model.network.build_network`."""
    if protocol.electrode < 0 or protocol.electrode >= layout.n_electrodes:
        raise ValueError(
            f"electrode {protocol.electrode} out of range for "
            f"{layout.n_electrodes}-electrode layout"
        )
    times: list[float] = []
    amplitudes: list[float] = []
    t = float(start_s)
    for amp in protocol.amplitudes:
        for _ in range(protocol.n_pulses_per_amplitude):
            times.append(t)
            amplitudes.append(float(amp) * float(amplitude_scale_mv))
            t += float(protocol.inter_stimulus_interval_s)
    return {
        "times_s": np.asarray(times, dtype=np.float64),
        "amplitudes_mv": np.asarray(amplitudes, dtype=np.float64),
        "electrode_x_um": float(layout.x_um[protocol.electrode]),
        "electrode_y_um": float(layout.y_um[protocol.electrode]),
        "radius_um": float(radius_um),
        "blanking_ms": float(protocol.blanking_ms),
        "pulse_width_ms": float(protocol.pulse_width_ms),
    }


def evoked_response(
    recording: SpikeRecording,
    stimulus_times_s: Sequence[float],
    *,
    amplitude: float,
    blanking_ms: float = 2.0,
    response_window_s: tuple[float, float] = RESPONSE_WINDOW_S,
    psth_bin_s: float = PSTH_BIN_S,
    burst_window_s: float = BURST_WINDOW_S,
    rng: np.random.Generator | None = None,
) -> EvokedResponse:
    """PSTH, response probability, latency and post-stimulus burst probability."""
    times = np.asarray(stimulus_times_s, dtype=np.float64)
    if times.size == 0:
        edges = np.arange(0.0, response_window_s[1] + psth_bin_s, psth_bin_s)
        return EvokedResponse(
            amplitude=float(amplitude),
            psth_bin_edges_s=edges,
            psth_hz=np.zeros(max(edges.size - 1, 0), dtype=np.float64),
            response_probability=0.0,
            mean_latency_s=float("nan"),
            post_stimulus_burst_probability=0.0,
            n_trials=0,
        )

    blank = float(blanking_ms) * 1e-3
    win_lo, win_hi = float(response_window_s[0]), float(response_window_s[1])
    # Spikes relative to each stimulus, after blanking.
    rel_spikes: list[np.ndarray] = []
    responded = 0
    latencies: list[float] = []
    spike_t = np.asarray(recording.times, dtype=np.float64)

    for stim in times:
        lo = stim + max(blank, win_lo)
        hi = stim + win_hi
        mask = (spike_t >= lo) & (spike_t < hi)
        rel = spike_t[mask] - stim
        rel_spikes.append(rel)
        if rel.size:
            responded += 1
            latencies.append(float(rel.min()))

    all_rel = np.concatenate(rel_spikes) if rel_spikes else np.array([], dtype=np.float64)
    edges = np.arange(0.0, win_hi + psth_bin_s, psth_bin_s)
    if all_rel.size:
        counts, _ = np.histogram(all_rel, bins=edges)
        psth_hz = counts.astype(np.float64) / (times.size * psth_bin_s)
    else:
        psth_hz = np.zeros(edges.size - 1, dtype=np.float64)

    # Burst probability: network burst onset inside (stim, stim+burst_window).
    rng = np.random.default_rng(0) if rng is None else rng
    bursts = detect_bursts(recording, rng)
    n_burst_hits = 0
    if bursts.available and bursts.n_bursts:
        for stim in times:
            lo = stim + blank
            hi = stim + float(burst_window_s)
            if np.any((bursts.starts >= lo) & (bursts.starts < hi)):
                n_burst_hits += 1

    return EvokedResponse(
        amplitude=float(amplitude),
        psth_bin_edges_s=edges,
        psth_hz=psth_hz,
        response_probability=float(responded / times.size),
        mean_latency_s=float(np.mean(latencies)) if latencies else float("nan"),
        post_stimulus_burst_probability=float(n_burst_hits / times.size),
        n_trials=int(times.size),
    )


def run_perturbation(
    *,
    base: ModelParams,
    free: FreeParams,
    protocol: StimulusProtocol | None = None,
    observation_config: Mapping[str, Any] | None = None,
    observed_recording: SpikeRecording | None = None,
    observed_stimulus_times_s: Sequence[float] | None = None,
    amplitude_scale_mv: float = DEFAULT_AMPLITUDE_SCALE_MV,
    run_index: int = 30_000,
    **kwargs: Any,
) -> PerturbationResult:
    """Simulate the evoked protocol under the spontaneous-fit parameters.

    ``observed_recording`` is optional. Without a real stim session to compare
    against, the test records the simulated curve and **fails** with an explicit
    note — SPEC §9.3 forbids quietly dropping the comparison.
    """
    del kwargs
    from ..config import load_config
    from ..model.runner import RunRequest, SimulationError, run_one
    from ..observation.virtual_mea import ObservationConfig

    observation = (
        dict(observation_config)
        if observation_config is not None
        else dict(load_config("observation.yaml"))
    )
    obs_cfg = ObservationConfig.from_config(observation)
    protocol = protocol or StimulusProtocol(electrode=obs_cfg.layout.n_electrodes // 2)

    # Short default for laptop runs; callers can pass a fuller protocol.
    stim = protocol_to_stimulus(
        protocol,
        obs_cfg.layout,
        amplitude_scale_mv=amplitude_scale_mv,
        start_s=float(base.simulation.transient_s),
    )
    duration_s = float(protocol.total_duration_s + 1.0)
    sim_base = replace(
        base,
        simulation=replace(base.simulation, duration_s=duration_s),
    )

    notes_parts: list[str] = []
    simulated: list[EvokedResponse] = []
    try:
        result = run_one(
            RunRequest(
                params=sim_base.with_free(free),
                run_index=run_index,
                observation_config=observation,
                stimulus=stim,
            )
        )
        recording = result.recording
        notes_parts.append(
            f"Simulated {protocol.total_duration_s:.0f} s stim protocol "
            f"({recording.n_spikes} electrode spikes)."
        )
        times_s = np.asarray(stim["times_s"], dtype=np.float64)
        amps = np.asarray(stim["amplitudes_mv"], dtype=np.float64) / float(amplitude_scale_mv)
        # Align times to recording clock (transient stripped in runner).
        times_rec = times_s - float(base.simulation.transient_s)
        for amp in protocol.amplitudes:
            stim_times = times_rec[np.isclose(amps, amp)]
            simulated.append(
                evoked_response(
                    recording,
                    stim_times,
                    amplitude=float(amp),
                    blanking_ms=protocol.blanking_ms,
                )
            )
    except SimulationError as exc:
        notes_parts.append(f"Stimulation sim failed: {exc}.")
        simulated = []

    observed: list[EvokedResponse] = []
    if observed_recording is not None and observed_stimulus_times_s is not None:
        # Single amplitude unknown for many Wagenaar stim files — score as one curve.
        observed.append(
            evoked_response(
                observed_recording,
                observed_stimulus_times_s,
                amplitude=float("nan"),
                blanking_ms=protocol.blanking_ms,
            )
        )
        notes_parts.append(f"Observed stim recording: {len(observed_stimulus_times_s)} pulses.")
    else:
        notes_parts.append(
            "No matched evoked recording supplied; cannot score PSTH/amplitude/"
            "burst curves against data (SPEC §9.3)."
        )

    psth_corr, amp_err, burst_err = _compare_curves(simulated, observed)
    passed = bool(
        observed
        and simulated
        and np.isfinite(psth_corr)
        and psth_corr >= PASS_PSTH_CORR
        and amp_err <= PASS_AMP_CURVE_ERROR
        and burst_err <= PASS_BURST_ERROR
    )
    if not observed:
        passed = False
    notes_parts.append(
        f"psth_corr={psth_corr:.3g}, amp_curve_err={amp_err:.3g}, "
        f"burst_err={burst_err:.3g}; passed={passed}."
    )
    return PerturbationResult(
        protocol=protocol,
        simulated=tuple(simulated),
        observed=tuple(observed),
        psth_correlation=psth_corr,
        amplitude_curve_error=amp_err,
        burst_probability_error=burst_err,
        passed=passed,
        notes=" ".join(notes_parts),
        diagnostics={
            "amplitude_scale_mv": amplitude_scale_mv,
            "n_simulated_amplitudes": len(simulated),
            "n_observed_curves": len(observed),
        },
    )


def _compare_curves(
    simulated: Sequence[EvokedResponse],
    observed: Sequence[EvokedResponse],
) -> tuple[float, float, float]:
    if not simulated or not observed:
        return float("nan"), float("nan"), float("nan")
    # Collapse multi-amplitude sim to mean PSTH; compare to the (often single) observed.
    sim_psth = np.mean(np.vstack([r.psth_hz for r in simulated]), axis=0)
    obs = observed[0]
    n = min(sim_psth.size, obs.psth_hz.size)
    if n < 2:
        psth_corr = float("nan")
    else:
        a, b = sim_psth[:n], obs.psth_hz[:n]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            psth_corr = 0.0
        else:
            psth_corr = float(np.corrcoef(a, b)[0, 1])
    sim_resp = float(np.mean([r.response_probability for r in simulated]))
    obs_resp = float(np.mean([r.response_probability for r in observed]))
    sim_burst = float(np.mean([r.post_stimulus_burst_probability for r in simulated]))
    obs_burst = float(np.mean([r.post_stimulus_burst_probability for r in observed]))
    amp_err = float(abs(sim_resp - obs_resp))
    burst_err = float(abs(sim_burst - obs_burst))
    return psth_corr, amp_err, burst_err
