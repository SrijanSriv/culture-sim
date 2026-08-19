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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..stats.spiketrains import SpikeRecording

__all__ = ["StimulusProtocol", "EvokedResponse", "PerturbationResult", "run_perturbation"]


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


@dataclass(frozen=True)
class EvokedResponse:
    amplitude: float
    psth_bin_edges_s: np.ndarray
    psth_hz: np.ndarray
    response_probability: float  # P(>=1 spike in the response window | stimulus)
    mean_latency_s: float
    post_stimulus_burst_probability: float
    n_trials: int


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


def evoked_response(
    recording: SpikeRecording,
    stimulus_times_s: Sequence[float],
    **kwargs: Any,
) -> EvokedResponse:
    """PSTH, response probability, latency and post-stimulus burst probability."""
    raise NotImplementedError("Task 7 (SPEC §9.3)")


def run_perturbation(**kwargs: Any) -> PerturbationResult:
    raise NotImplementedError("Task 7 (SPEC §9.3)")
