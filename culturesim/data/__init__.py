"""Real data loading and the simulation cache (SPEC §7, §11)."""

from __future__ import annotations

from .cache import SimulationCache
from .loaders import DATASETS, DatasetInfo, available_datasets

__all__ = ["DATASETS", "DatasetInfo", "SimulationCache", "available_datasets"]
