"""Statistics computed on electrode-level recordings (SPEC §6).

Every function here takes a :class:`~culturesim.stats.spiketrains.SpikeRecording`
and works identically on simulated and real data. Nothing here may special-case
the source of the recording.
"""

from __future__ import annotations

from .fingerprint import Fingerprint, FingerprintSpec, compute_fingerprint
from .spiketrains import SpikeRecording, load_recording, save_recording

__all__ = [
    "Fingerprint",
    "FingerprintSpec",
    "SpikeRecording",
    "compute_fingerprint",
    "load_recording",
    "save_recording",
]
