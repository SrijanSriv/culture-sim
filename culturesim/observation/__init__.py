"""Observation model: the electrode-level bottleneck (SPEC §5)."""

from __future__ import annotations

from .virtual_mea import ElectrodeLayout, ObservationConfig, observe

__all__ = ["ElectrodeLayout", "ObservationConfig", "observe"]
