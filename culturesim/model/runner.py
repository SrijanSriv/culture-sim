"""Subprocess-isolated simulation execution (SPEC §4.5).

Brian2's ``cpp_standalone`` device cannot be reinitialised cleanly many times in one
process -- looping ``device.reinit()`` thousands of times leaks or fails outright, and
the SBI stage needs thousands of runs. So each simulation gets its own process with its
own build directory under a temp root, returns a
:class:`~culturesim.stats.spiketrains.SpikeRecording`, and cleans up after itself.

Parallelism uses ``multiprocessing.Pool(maxtasksperchild=1)``. The ``maxtasksperchild``
is not a tuning knob: without it a pooled worker would be reused for a second
simulation and hit exactly the standalone-reinitialisation problem this module exists to
avoid. One task per process, then the process is replaced.

Each subprocess derives its numpy and Brian2 seeds deterministically from the master
seed plus the run index, so results do not depend on the order the pool finishes in.

Performance budget: 300 s of biological time in well under 60 s wall-clock. Compilation
is a fixed per-run cost under standalone, so ``RunResult.diagnostics`` separates build
time from run time -- if the budget is missed it matters a great deal which one is to
blame.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..rng import derive_seed
from ..stats.spiketrains import SpikeRecording
from .params import FreeParams, ModelParams

__all__ = [
    "RunRequest",
    "RunResult",
    "SimulationError",
    "run_one",
    "run_many",
    "run_free_params",
    "WALL_CLOCK_BUDGET_S",
    "BUDGET_BIOLOGICAL_S",
]

# SPEC §4.5 performance target, asserted by tests/test_runner.py.
WALL_CLOCK_BUDGET_S = 60.0
BUDGET_BIOLOGICAL_S = 300.0


class SimulationError(RuntimeError):
    """A simulation subprocess failed.

    Raised rather than returning a sentinel so that a crashed draw cannot be mistaken
    for a draw that produced a silent network. SBI collects these and reports how many
    were excluded (SPEC §8.3).
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
        return derive_seed(self.params.seed, "simulation", self.run_index)


@dataclass(frozen=True)
class RunResult:
    """A completed simulation.

    ``recording`` is electrode-level whenever an observation config was supplied: the
    virtual MEA runs inside the subprocess, so neuron-level spikes never cross the
    process boundary. That keeps the pickled payload small and makes it hard for
    downstream code to use neuron-level data by accident (SPEC §5, §14).

    Without an observation config the recording is neuron-level, tagged
    ``metadata["observation"] == "none"``. That mode exists for the Task 1 raster and
    ablation figures only; ``compute_fingerprint`` refuses such recordings.
    """

    recording: SpikeRecording
    wall_clock_s: float
    seed: int
    run_index: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        """Whether this run met the SPEC §4.5 budget, scaled to its own duration."""
        biological_s = self.recording.metadata.get("duration_s", self.recording.duration)
        allowed = WALL_CLOCK_BUDGET_S * float(biological_s) / BUDGET_BIOLOGICAL_S
        return self.wall_clock_s <= allowed


def _simulate(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one simulation. Executes in a fresh subprocess; never call directly.

    Returns a plain dict rather than a ``RunResult`` so that a failure can be reported
    across the process boundary without depending on the exception type pickling
    cleanly, which Brian2's exceptions do not reliably do.
    """
    started = time.perf_counter()
    build_dir = Path(payload["build_dir"])
    try:
        import brian2 as b2

        from ..observation.virtual_mea import ObservationConfig, observe
        from .network import build_network, place_neurons

        b2.prefs.codegen.target = "cython"  # only used before set_device takes effect
        b2.set_device("cpp_standalone", directory=str(build_dir), build_on_run=False)
        b2.defaultclock.dt = payload["dt_ms"] * b2.ms
        # Brian2 prints a progress banner and codegen chatter per run; with thousands of
        # runs that is noise, and it slows the pool down measurably.
        b2.prefs.logging.console_log_level = "ERROR"

        params: ModelParams = payload["params"]
        seed = int(payload["seed"])
        rng = np.random.default_rng(seed)

        positions = place_neurons(params, rng)
        network, components = build_network(params, positions, seed, stimulus=payload["stimulus"])

        total_s = params.simulation.total_duration_s
        built_at = time.perf_counter()
        network.run(total_s * b2.second)
        b2.device.build(directory=str(build_dir), compile=True, run=True, clean=False)
        finished_at = time.perf_counter()

        monitor = components["spike_monitor"]
        spike_times_s = np.asarray(monitor.t / b2.second, dtype=np.float64)
        spike_neurons = np.asarray(monitor.i, dtype=np.int64)

        # Discard the warm-up transient and re-zero (SPEC §4.5 / model_default.yaml).
        transient_s = params.simulation.transient_s
        keep = spike_times_s >= transient_s
        spike_times_s = spike_times_s[keep] - transient_s
        spike_neurons = spike_neurons[keep]
        duration_s = params.simulation.duration_s

        diagnostics = {
            "n_synapses_exc": components["n_synapses_exc"],
            "n_synapses_inh": components["n_synapses_inh"],
            "stp_enabled": components["stp_enabled"],
            "n_neuron_spikes": int(spike_times_s.size),
            "neuron_mean_rate_hz": float(
                spike_times_s.size / (duration_s * params.network.n_neurons)
            ),
            "build_and_run_s": round(finished_at - built_at, 3),
            "setup_s": round(built_at - started, 3),
        }

        if payload["observation_config"] is None:
            recording_metadata = {
                # Flags this as pre-observation data. Statistics computed on it are
                # not comparable to anything measured on an electrode array.
                "observation": "none",
                "duration_s": duration_s,
                "seed": seed,
                **diagnostics,
            }
            if params.simulation.record_neuron_positions:
                recording_metadata["neuron_x_um"] = positions.x_um.tolist()
                recording_metadata["neuron_y_um"] = positions.y_um.tolist()
            recording = SpikeRecording(
                times=spike_times_s,
                channels=spike_neurons.astype(np.int32),
                n_channels=params.network.n_neurons,
                duration=duration_s,
                source="simulation-neuron-level",
                metadata=recording_metadata,
            )
        else:
            observation = ObservationConfig.from_config(payload["observation_config"])
            recording = observe(
                spike_times_s,
                spike_neurons,
                positions.x_um,
                positions.y_um,
                duration_s,
                observation,
                np.random.default_rng(derive_seed(seed, "observation")),
                source="simulation",
                metadata={"duration_s": duration_s, "seed": seed, **diagnostics},
            )

        return {
            "ok": True,
            "recording": recording,
            "wall_clock_s": round(time.perf_counter() - started, 3),
            "seed": seed,
            "run_index": int(payload["run_index"]),
            "diagnostics": diagnostics,
        }
    except Exception as exc:  # noqa: BLE001 - reported across the process boundary
        return {
            "ok": False,
            "run_index": int(payload["run_index"]),
            "seed": int(payload["seed"]),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        if not payload["keep_build_dir"]:
            shutil.rmtree(build_dir, ignore_errors=True)


def _payload(
    request: RunRequest,
    build_root: Path | None,
    keep_build_dir: bool,
) -> dict[str, Any]:
    root = Path(build_root) if build_root is not None else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    build_dir = Path(
        tempfile.mkdtemp(prefix=f"culturesim_run{request.run_index:06d}_", dir=str(root))
    )
    return {
        "params": request.params,
        "run_index": request.run_index,
        "seed": request.seed,
        "observation_config": request.observation_config,
        "stimulus": request.stimulus,
        "dt_ms": request.params.simulation.dt_ms,
        "build_dir": str(build_dir),
        "keep_build_dir": keep_build_dir,
    }


def _to_result(outcome: dict[str, Any]) -> RunResult:
    if not outcome["ok"]:
        raise SimulationError(
            f"run {outcome['run_index']} (seed {outcome['seed']}) failed: "
            f"{outcome['error']}\n{outcome['traceback']}"
        )
    return RunResult(
        recording=outcome["recording"],
        wall_clock_s=outcome["wall_clock_s"],
        seed=outcome["seed"],
        run_index=outcome["run_index"],
        diagnostics=outcome["diagnostics"],
    )


def default_workers() -> int:
    """``cpu_count() - 1``, leaving a core free so the machine stays usable."""
    return max(1, (os.cpu_count() or 2) - 1)


def run_one(
    request: RunRequest,
    *,
    build_root: Path | None = None,
    keep_build_dir: bool = False,
    timeout_s: float | None = None,
) -> RunResult:
    """Execute one simulation in a fresh subprocess.

    Uses a one-worker pool rather than calling ``_simulate`` inline, so that a single
    run and a batch go through exactly the same isolation path -- a runner that behaves
    differently when run alone is a runner whose tests prove nothing.
    """
    return run_many(
        [request],
        n_workers=1,
        build_root=build_root,
        keep_build_dir=keep_build_dir,
        timeout_s=timeout_s,
    )[0]


def run_many(
    requests: Sequence[RunRequest] | Iterable[RunRequest],
    *,
    n_workers: int | None = None,
    build_root: Path | None = None,
    keep_build_dir: bool = False,
    timeout_s: float | None = None,
    on_error: str = "raise",
) -> list[RunResult]:
    """Run simulations in parallel, one subprocess per simulation.

    ``on_error='raise'`` propagates the first failure as a :class:`SimulationError`.
    ``on_error='skip'`` drops failed runs and is what SBI uses, since some corners of
    the prior box legitimately cannot be simulated (SPEC §8.3).
    """
    requests = list(requests)
    if not requests:
        return []
    if on_error not in {"raise", "skip"}:
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    workers = min(default_workers() if n_workers is None else int(n_workers), len(requests))
    payloads = [_payload(r, build_root, keep_build_dir) for r in requests]

    # 'spawn' rather than 'fork': Brian2 keeps module-level device state, and a forked
    # child would inherit a partially-configured device from the parent.
    context = mp.get_context("spawn")
    results: list[RunResult] = []
    with context.Pool(processes=workers, maxtasksperchild=1) as pool:
        async_result = pool.map_async(_simulate, payloads)
        outcomes = async_result.get(timeout=timeout_s)

    for outcome in outcomes:
        if outcome["ok"]:
            results.append(_to_result(outcome))
        elif on_error == "raise":
            _to_result(outcome)
    return results


def run_free_params(
    draws: Sequence[FreeParams],
    base: ModelParams,
    *,
    observation_config: dict[str, Any] | None = None,
    start_index: int = 0,
    **kwargs: Any,
) -> list[RunResult]:
    """Run a batch of parameter draws off one base config. Used by the fit stages."""
    requests = [
        RunRequest(
            params=replace(base, free=draw),
            run_index=start_index + index,
            observation_config=observation_config,
        )
        for index, draw in enumerate(draws)
    ]
    return run_many(requests, **kwargs)
