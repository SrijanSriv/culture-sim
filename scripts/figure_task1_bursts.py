#!/usr/bin/env python3
"""Task 1 acceptance figure and measurements (SPEC §13).

Runs the hand-tuned parameter set and the static-synapse ablation on identical topology
and seed, and reports:

* whether network bursts appear, separated by quiet periods, with inter-burst intervals
  in the 1-60 s range;
* that the ablation demonstrably fails to produce them;
* the wall-clock time for 300 s of biological time against the 60 s budget (SPEC §4.5).

    .venv/bin/python scripts/figure_task1_bursts.py --duration 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.figures import (  # noqa: E402
    apply_style,
    population_rate_panel,
    raster_panel,
    save_figure,
)
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import (  # noqa: E402
    BUDGET_BIOLOGICAL_S,
    WALL_CLOCK_BUDGET_S,
    RunRequest,
    run_many,
)
from culturesim.stats.spiketrains import SpikeRecording  # noqa: E402

BIN_S = 0.025


def burst_intervals(recording: SpikeRecording, threshold_factor: float = 8.0) -> dict[str, float]:
    """Crude burst timing, for the acceptance check only.

    Deliberately independent of ``stats/bursts.py``: this figure has to be able to say
    "the network bursts" before the fitted burst detector exists, and reusing the
    detector would make the claim circular.
    """
    n_bins = max(1, int(np.ceil(recording.duration / BIN_S)))
    index = np.minimum((recording.times / BIN_S).astype(np.int64), n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    if counts.sum() == 0:
        return {"n_bursts": 0, "median_ibi_s": float("nan"), "peak_over_mean": float("nan")}

    # Absolute floor of 10 spikes/bin so a silent network cannot produce "bursts" out of
    # a handful of spikes by having a tiny mean.
    threshold = max(10.0, threshold_factor * counts.mean())
    active = counts > threshold
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    stops = np.flatnonzero(active & ~np.r_[active[1:], False])
    ibis = np.diff(starts) * BIN_S
    return {
        "n_bursts": int(starts.size),
        "burst_rate_per_min": float(starts.size / (recording.duration / 60.0)),
        "median_ibi_s": float(np.median(ibis)) if ibis.size else float("nan"),
        "min_ibi_s": float(ibis.min()) if ibis.size else float("nan"),
        "max_ibi_s": float(ibis.max()) if ibis.size else float("nan"),
        # IBI spread is the discriminator that matters. A culture's inter-burst intervals
        # are broadly distributed; a periodic oscillator's are not, and a periodic
        # oscillator is what a network without synaptic depression degenerates into.
        "ibi_cv": float(ibis.std() / ibis.mean()) if ibis.size > 1 else float("nan"),
        "mean_duration_s": float(np.mean((stops - starts + 1) * BIN_S)) if starts.size else 0.0,
        "peak_over_mean": float(counts.max() / counts.mean()),
        "quiet_fraction": float(np.mean(counts == 0)),
        "mean_rate_hz_per_neuron": float(
            recording.n_spikes / (recording.duration * recording.n_channels)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=BUDGET_BIOLOGICAL_S)
    parser.add_argument("--transient", type=float, default=10.0)
    parser.add_argument("--window", type=float, default=60.0, help="seconds shown in the raster")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"].update(duration_s=args.duration, transient_s=args.transient)
    base = ModelParams.from_config(config)

    # Same seed for both, so the ablation differs only in short-term plasticity.
    requests = [
        RunRequest(params=base, run_index=0),
        RunRequest(
            params=replace(base, simulation=replace(base.simulation, static_synapses=True)),
            run_index=0,
        ),
    ]
    started = time.perf_counter()
    results = run_many(requests, n_workers=2)
    print(f"both runs completed in {time.perf_counter() - started:.1f}s wall clock\n")

    stp, static = results[0], results[1]
    summaries = {}
    for label, result in (("stp", stp), ("static", static)):
        summary = burst_intervals(result.recording)
        summary["wall_clock_s"] = result.wall_clock_s
        summary["within_budget"] = bool(result.within_budget)
        summaries[label] = summary
        print(f"[{label}] wall clock {result.wall_clock_s:.1f}s for {args.duration}s biological")
        for key, value in summary.items():
            print(f"    {key:24s}: {value}")
        print()

    budget = WALL_CLOCK_BUDGET_S * args.duration / BUDGET_BIOLOGICAL_S
    print(f"SPEC §4.5 budget for {args.duration}s biological: {budget:.0f}s wall clock")
    print(f"  STP run: {stp.wall_clock_s:.1f}s -> {'PASS' if stp.within_budget else 'FAIL'}")

    ibi = summaries["stp"]["median_ibi_s"]
    in_range = summaries["stp"]["n_bursts"] >= 2 and 1.0 <= ibi <= 60.0
    print(f"  bursts with 1-60s IBI: {'PASS' if in_range else 'FAIL'} (median {ibi:.2f}s)")

    # The ablation does not go silent -- it degenerates into a fast periodic oscillation.
    # So the criterion is not "fewer bursts" but "not culture-like bursting": a
    # near-constant sub-second period at a far higher firing rate.
    static_summary, stp_summary = summaries["static"], summaries["stp"]
    periodic = static_summary["ibi_cv"] < 0.3 or static_summary["median_ibi_s"] < 1.0
    hyperactive = (
        static_summary["mean_rate_hz_per_neuron"] > 5 * stp_summary["mean_rate_hz_per_neuron"]
    )
    ablation_fails = periodic or hyperactive
    print(
        f"  static ablation fails to burst like a culture: {'PASS' if ablation_fails else 'FAIL'}"
    )
    print(
        f"    STP:    IBI {stp_summary['median_ibi_s']:.1f}s (CV {stp_summary['ibi_cv']:.2f}), "
        f"{stp_summary['mean_rate_hz_per_neuron']:.2f} Hz/neuron"
    )
    print(
        f"    static: IBI {static_summary['median_ibi_s']:.2f}s "
        f"(CV {static_summary['ibi_cv']:.2f}), "
        f"{static_summary['mean_rate_hz_per_neuron']:.2f} Hz/neuron"
    )

    # -- figure -----------------------------------------------------------
    apply_style()
    import matplotlib.pyplot as plt

    window = min(args.window, args.duration)
    figure, axes = plt.subplots(
        4, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 3, 1]}
    )
    for offset, (label, result) in enumerate(
        (("Tsodyks-Markram STP", stp), ("static synapses", static))
    ):
        sliced = result.recording.time_slice(0.0, window)
        raster_panel(axes[2 * offset], sliced, title=f"{label} ({sliced.n_spikes} spikes shown)")
        population_rate_panel(axes[2 * offset + 1], sliced)
    axes[-1].set_xlabel("time (s)")
    figure.suptitle(
        "Task 1: short-term plasticity is what makes network bursts terminate",
        x=0.02,
        ha="left",
    )
    path = save_figure(figure, "task1_bursts_vs_static", directory=args.out)
    print(f"\nwrote {path}")

    summary_path = path.with_suffix(".json")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0 if (in_range and ablation_fails and stp.within_budget) else 1


if __name__ == "__main__":
    raise SystemExit(main())
