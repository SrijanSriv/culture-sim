"""Subprocess-isolated simulation execution (SPEC §4.5).

Task 1.

Brian2's ``cpp_standalone`` device cannot be reinitialised cleanly many times in one
process -- looping ``device.reinit()`` thousands of times leaks or fails outright,
and the SBI stage needs thousands of runs. So each simulation gets its own
subprocess with its own build directory under a temp root, returns a
:class:`~culturesim.stats.spiketrains.SpikeRecording`, and cleans up after itself.

Each subprocess derives its numpy and Brian2 seeds deterministically from the master
seed plus the run index, so results do not depend on the order the pool happens to
finish in.

Performance budget: 300 s of biological time in well under 60 s wall-clock. If a run
misses that, profile before continuing rather than starting the fit -- 5000 draws at
60 s each is a different project than 5000 at 10 s each.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..stats.spiketrains import SpikeRecording
from .params import FreeParams, ModelParams

__all__ = ["RunRequest", "RunResult", "run_one", "run_many", "SimulationError"]

# SPEC §4.5 performance target, asserted by tests/test_runner.py.
WALL_CLOCK_BUDGET_S = 60.0
BUDGET_BIOLOGICAL_S = 300.0


class SimulationError(RuntimeError):
    """A simulation subprocess failed.

    Raised rather than returning a sentinel so that a crashed draw cannot be
    mistaken for a draw that produced a silent network. SBI collects these and
    reports how many were excluded (SPEC §8.3).
    """


@dataclass(frozen=True)
class RunRequest:
    """One simulation to execute."""

    params: ModelParams
    run_index: int = 0
    observation_config: dict[str, Any] | None = None
    stimulus: dict[str, Any] | None = None  # Task 7; None for spontaneous activity

    @property
    def seed(self) -> int:
        """Deterministic per-run seed derived from the master seed and run index."""
        from ..rng import derive_seed

        return derive_seed(self.params.seed, "simulation", self.run_index)


@dataclass(frozen=True)
class RunResult:
    """A completed simulation.

    ``recording`` is electrode-level: the virtual MEA has already been applied
    inside the subprocess, so the neuron-level spikes never cross the process
    boundary. That is deliberate -- it keeps the pickled payload small and makes it
    impossible for downstream code to accidentally use neuron-level data (SPEC §5).
    """

    recording: SpikeRecording
    wall_clock_s: float
    seed: int
    run_index: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_one(request: RunRequest, *, build_root: Path | None = None) -> RunResult:
    """Execute one simulation in a fresh subprocess."""
    raise NotImplementedError("Task 1 (SPEC §4.5)")


def run_many(
    requests: Sequence[RunRequest] | Iterable[RunRequest],
    *,
    n_workers: int | None = None,
    build_root: Path | None = None,
    timeout_s: float | None = None,
) -> list[RunResult]:
    """Run simulations in parallel over ``multiprocessing.Pool``.

    ``n_workers=None`` uses ``cpu_count() - 1``, leaving one core so the machine
    stays usable during an overnight SBI sweep.
    """
    raise NotImplementedError("Task 1 (SPEC §4.5)")


def run_free_params(
    draws: Sequence[FreeParams],
    base: ModelParams,
    **kwargs: Any,
) -> list[RunResult]:
    """Convenience wrapper for SBI: run a batch of parameter draws off one base config."""
    raise NotImplementedError("Task 1 (SPEC §4.5)")
