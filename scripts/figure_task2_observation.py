#!/usr/bin/env python3
"""Task 2 acceptance figure: 60- vs 1024-electrode observation (SPEC §13).

Takes one neuron-level simulation and pushes it through both shipped layouts, so the
biology is identical and only the observational bottleneck changes.

    .venv/bin/python scripts/figure_task2_observation.py --duration 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.figures import apply_style, save_figure  # noqa: E402
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import RunRequest, run_one  # noqa: E402
from culturesim.observation.virtual_mea import ObservationConfig, observe  # noqa: E402
from culturesim.stats.branching import mr_branching_ratio, naive_branching_ratio  # noqa: E402
from culturesim.stats.fingerprint import FingerprintSpec, compute_fingerprint  # noqa: E402
from culturesim.stats.rates import rate_stats  # noqa: E402

LAYOUTS = ("mcs_60", "hd_mea_1024")


def _observe_both(neuron_recording, observation_seed: int) -> dict:
    x_um = np.asarray(neuron_recording.metadata["neuron_x_um"], dtype=np.float64)
    y_um = np.asarray(neuron_recording.metadata["neuron_y_um"], dtype=np.float64)
    observed = {}
    for name in LAYOUTS:
        config = ObservationConfig.load("observation.yaml", layout_name=name)
        observed[name] = observe(
            neuron_recording.times,
            neuron_recording.channels,
            x_um,
            y_um,
            neuron_recording.duration,
            config,
            # Same seed so per-neuron amplitude scatter is identical; dead-electrode
            # draws then differ only because the layouts have different counts.
            np.random.default_rng(observation_seed),
            source="simulation",
            metadata={"layout": name, "seed": observation_seed},
        )
    return observed


def _summary(recording, fingerprint) -> dict[str, float]:
    rates = rate_stats(recording)
    branching = mr_branching_ratio(recording)
    return {
        "n_channels": float(recording.n_channels),
        "n_spikes": float(recording.n_spikes),
        "n_dead": float(len(recording.metadata.get("dead_electrodes", []))),
        "rate_mean": rates.rate_mean,
        "rate_std": rates.rate_std,
        "rate_p10": rates.rate_p10,
        "rate_p90": rates.rate_p90,
        "active_electrode_fraction": rates.active_electrode_fraction,
        "isi_cv_pooled": rates.isi_cv_pooled,
        "branching_ratio_mr": branching.branching_ratio_mr,
        "branching_ratio_naive": naive_branching_ratio(recording),
        "burst_rate_per_min": fingerprint["burst_rate_per_min"],
        "n_finite_fingerprint": float(np.count_nonzero(np.isfinite(fingerprint.values))),
        "n_fingerprint": float(len(fingerprint)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--transient", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"].update(duration_s=args.duration, transient_s=args.transient)
    params = ModelParams.from_config(config)
    result = run_one(RunRequest(params=params, run_index=0))
    neuron = result.recording
    print(
        f"neuron-level: {neuron.n_spikes} spikes in {neuron.duration:.0f}s "
        f"({result.wall_clock_s:.1f}s wall)"
    )

    observed = _observe_both(neuron, observation_seed=result.seed)
    spec = FingerprintSpec.load("fingerprint.yaml")
    fingerprints = {name: compute_fingerprint(rec, spec) for name, rec in observed.items()}
    summaries = {name: _summary(observed[name], fingerprints[name]) for name in LAYOUTS}

    for name, summary in summaries.items():
        print(f"\n[{name}]")
        for key, value in summary.items():
            print(f"    {key:28s}: {value}")

    n_spikes_differ = summaries["mcs_60"]["n_spikes"] != summaries["hd_mea_1024"]["n_spikes"]
    active_differ = (
        summaries["mcs_60"]["active_electrode_fraction"]
        != summaries["hd_mea_1024"]["active_electrode_fraction"]
    )
    naive_differ = (
        summaries["mcs_60"]["branching_ratio_naive"]
        != summaries["hd_mea_1024"]["branching_ratio_naive"]
    )
    hd_bursts_undefined = not np.isfinite(summaries["hd_mea_1024"]["burst_rate_per_min"])
    print("\nacceptance:")
    print(f"  spike counts differ:         {'PASS' if n_spikes_differ else 'FAIL'}")
    print(f"  active fractions differ:     {'PASS' if active_differ else 'FAIL'}")
    print(f"  naive branching differs:     {'PASS' if naive_differ else 'FAIL'}")
    print(f"  HD-MEA CL bursts undefined:  {'PASS' if hd_bursts_undefined else 'FAIL'}")
    print(
        f"  60-electrode CL bursts finite: "
        f"{'PASS' if np.isfinite(summaries['mcs_60']['burst_rate_per_min']) else 'FAIL'}"
    )

    apply_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for name, color in (("mcs_60", "0.15"), ("hd_mea_1024", "0.55")):
        rates = rate_stats(observed[name]).per_electrode_rates
        axes[0].hist(
            rates,
            bins=24,
            range=(0.0, max(float(rates.max()), 0.05)),
            density=True,
            histtype="step",
            color=color,
            lw=1.4,
            label=f"{name} (n={observed[name].n_channels})",
        )
    axes[0].set_xlabel("electrode firing rate (Hz)")
    axes[0].set_ylabel("density")
    axes[0].set_title("rate distribution", loc="left")
    axes[0].legend()

    labels = ["naive m", "MR m"]
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(
        x - width / 2,
        [
            summaries["mcs_60"]["branching_ratio_naive"],
            summaries["mcs_60"]["branching_ratio_mr"],
        ],
        width,
        color="0.15",
        label="60 elec.",
    )
    axes[1].bar(
        x + width / 2,
        [
            summaries["hd_mea_1024"]["branching_ratio_naive"],
            summaries["hd_mea_1024"]["branching_ratio_mr"],
        ],
        width,
        color="0.55",
        label="1024 elec.",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("branching ratio")
    axes[1].set_title("subsampling bias", loc="left")
    axes[1].legend()

    comparable = [
        "rate_mean",
        "rate_std",
        "active_electrode_fraction",
        "isi_cv_pooled",
        "branching_ratio_mr",
    ]
    y = np.arange(len(comparable))
    left = [summaries["mcs_60"][k] for k in comparable]
    right = [summaries["hd_mea_1024"][k] for k in comparable]
    axes[2].barh(y + 0.18, left, 0.35, color="0.15", label="60 elec.")
    axes[2].barh(y - 0.18, right, 0.35, color="0.55", label="1024 elec.")
    axes[2].set_yticks(y, comparable)
    axes[2].set_xlabel("value")
    axes[2].set_title("fingerprint scalars that both arrays can compute", loc="left")
    axes[2].legend()

    hd_finite = int(summaries["hd_mea_1024"]["n_finite_fingerprint"])
    n_stats = int(summaries["hd_mea_1024"]["n_fingerprint"])
    figure.suptitle(
        "Task 2: the same 1000 neurons, two arrays. "
        f"HD-MEA fills {hd_finite}/{n_stats} fingerprint entries "
        f"(`cl-sdk==1.0.0` cannot store channel ids > 255).",
        x=0.02,
        ha="left",
        fontsize=9,
    )
    figure.tight_layout()
    path = save_figure(figure, "task2_60_vs_1024", directory=args.out)
    print(f"\nwrote {path}")
    summary_path = path.with_suffix(".json")
    payload = {
        "summaries": summaries,
        "n_neuron_spikes": neuron.n_spikes,
        "wall_clock_s": result.wall_clock_s,
        "dead_electrodes": {
            name: observed[name].metadata.get("dead_electrodes", []) for name in LAYOUTS
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"wrote {summary_path}")

    passed = n_spikes_differ and active_differ and naive_differ and hd_bursts_undefined
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
