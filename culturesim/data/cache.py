"""Content-addressed cache for simulation output (SPEC §11).

Keyed by the hash of the config that produced the recording, so re-running an
identical simulation is free. Each entry stores the recording next to the manifest
of the run that made it -- a cached result with no provenance is not reusable
evidence.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..config import config_hash
from ..manifest import Manifest
from ..stats.spiketrains import SpikeRecording

__all__ = ["SimulationCache", "DEFAULT_CACHE_DIR"]

DEFAULT_CACHE_DIR = Path(".cache/simulations")


class SimulationCache:
    """Store and retrieve :class:`SpikeRecording` objects by config hash."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)

    def key(self, *configs: Any) -> str:
        """Cache key for the configs that fully determine a simulation."""
        return config_hash(*configs)

    def _entry_dir(self, key: str) -> Path:
        # Two-character prefix keeps directory listings usable after a few
        # thousand SBI draws.
        return self.root / key[:2] / key

    def path_for(self, key: str) -> Path:
        return self._entry_dir(key) / "recording.h5"

    def contains(self, key: str) -> bool:
        return self.enabled and self.path_for(key).exists()

    def get(self, key: str) -> SpikeRecording | None:
        if not self.contains(key):
            return None
        try:
            return SpikeRecording.from_hdf5(self.path_for(key))
        except (OSError, ValueError, KeyError):
            # A truncated entry from an interrupted write is worse than a miss.
            self.evict(key)
            return None

    def put(
        self,
        key: str,
        recording: SpikeRecording,
        manifest: Manifest | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None
        entry = self._entry_dir(key)
        entry.mkdir(parents=True, exist_ok=True)
        # Write to a sibling then rename, so a crash mid-write cannot leave a
        # half-written file that later looks like a valid hit.
        target = self.path_for(key)
        staging = target.with_suffix(".h5.partial")
        recording.to_hdf5(staging)
        staging.replace(target)
        if manifest is not None:
            manifest.write(entry / "manifest.json")
        return target

    def evict(self, key: str) -> None:
        shutil.rmtree(self._entry_dir(key), ignore_errors=True)

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*/*/recording.h5")) if self.root.exists() else 0

    def __repr__(self) -> str:
        return f"SimulationCache(root={str(self.root)!r}, enabled={self.enabled}, n={len(self)})"
