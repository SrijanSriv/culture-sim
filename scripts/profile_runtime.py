#!/usr/bin/env python3
"""Profile the simulation against the SPEC §4.5 budget.

SPEC §4.5 requires 300 s of biological time in well under 60 s wall clock, and requires
profiling rather than guessing if the budget is missed. This runs the same network with
individual components disabled, one run at a time so no measurement competes for cores.

    .venv/bin/python scripts/profile_runtime.py --duration 60
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import RunRequest, run_one  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--neurons", type=int, default=1000)
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"].update(duration_s=args.duration, transient_s=0.0)
    config["network"]["n_neurons"] = args.neurons
    base = ModelParams.from_config(config)

    variants: list[tuple[str, ModelParams]] = [
        ("full model", base),
        (
            "1 background afferent",
            replace(base, fixed=replace(base.fixed, n_background_synapses=1)),
        ),
        ("no recurrent synapses", replace(base, free=base.free.replace(p_conn=0.0))),
        (
            "no background, no recurrence",
            replace(
                base,
                fixed=replace(base.fixed, n_background_synapses=1),
                free=base.free.replace(p_conn=0.0),
            ),
        ),
        ("dt = 0.2 ms", replace(base, simulation=replace(base.simulation, dt_ms=0.2))),
    ]

    print(f"{args.duration}s biological, {args.neurons} neurons, one run at a time")
    print(f"{'variant':32s}{'wall':>8}{'setup':>8}{'sim':>8}{'spikes':>10}{'Hz/neuron':>11}")
    print("-" * 77)
    for label, params in variants:
        result = run_one(RunRequest(params=params, run_index=0))
        diagnostics = result.diagnostics
        print(
            f"{label:32s}{result.wall_clock_s:8.1f}{diagnostics['setup_s']:8.1f}"
            f"{diagnostics['build_and_run_s']:8.1f}{diagnostics['n_neuron_spikes']:10d}"
            f"{diagnostics['neuron_mean_rate_hz']:11.3f}"
        )
    print("\n'setup' is topology construction in Python; 'sim' is C++ compile plus the run.")

    # Compilation is a fixed cost per run under cpp_standalone, so it does not shrink
    # with a shorter simulation. Two durations separate the intercept from the slope,
    # which decides whether the budget problem is the compiler or the integration loop.
    print("\nseparating fixed compile cost from per-second integration cost:")
    print(f"{'biological s':>14}{'wall':>8}{'sim':>8}")
    print("-" * 30)
    measurements = []
    for duration in (10.0, args.duration):
        params = replace(base, simulation=replace(base.simulation, duration_s=duration))
        result = run_one(RunRequest(params=params, run_index=0))
        measurements.append((duration, result.diagnostics["build_and_run_s"]))
        print(f"{duration:14.0f}{result.wall_clock_s:8.1f}{measurements[-1][1]:8.1f}")

    (short_s, short_cost), (long_s, long_cost) = measurements
    per_second = (long_cost - short_cost) / (long_s - short_s)
    fixed_cost = short_cost - per_second * short_s
    print(f"\nfixed cost (compile etc): {fixed_cost:.1f}s")
    print(f"per biological second    : {per_second:.3f}s")
    print(f"projected for 300s       : {fixed_cost + 300 * per_second:.0f}s vs 60s budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
