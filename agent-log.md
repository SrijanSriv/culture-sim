# Agent log

Working brief for agents. Read this before starting; append to it before finishing. See
`AGENTS.md` §0 for the protocol. Older or superseded entries live in
`agent-log-archive.md`.

Newest entries at the top.

---

## 2026-08-19 — Task 2 60-vs-1024 figure and geometry fix

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 2.

Acceptance figure: `scripts/figure_task2_observation.py --duration 60`. Identical
neuron-level spikes (42,405 in 60 s) observed on both shipped layouts.

| | mcs_60 | hd_mea_1024 |
|---|---|---|
| electrode spikes | 3,791 | 63,661 |
| rate mean (Hz) | 1.05 | 1.04 |
| ISI CV pooled | 7.27 | 23.25 |
| naive branching | 0.911 | 0.923 |
| MR branching | 0.470 | 0.456 |
| burst_rate_per_min | 109 | NaN (CL uint8) |
| finite fingerprint entries | 62/66 | 17/66 |

The rate *means* barely move; the observational bottleneck shows up as spike count,
pooled ISI CV (1024 electrodes see overlapping copies of the same somata), and the
CL-delegated half of the vector going undefined. MR branching is the more stable of
the two estimators, which is the point of SPEC §6.4.

Also fixed a silent geometry bug: `place_neurons` put somata in `[0, W]×[0, H]` while
electrode layouts are centred on the origin, so the virtual MEA only saw one corner
of the culture. Neurons are now in `[-W/2, W/2]`. Dead-electrode simulation has a
direct test (soma on a marked-dead site records nothing).

`detect_bursts` returns NaN sentinels rather than crashing when CL cannot ingest the
recording. pytest: **162 passed**.

### Still open

1. Task 3: freeze `fingerprint.yaml`.
2. Task 4: write `load_wagenaar`.

---

## 2026-08-19 — Task 1 runtime budget closed

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 1.

Shipped defaults are now `background_mode: poisson` and `dt_ms: 0.2`. Diffusion stays
as a testable approximation only; do not turn it on to chase wall-clock.

Measured on this machine, poisson, 1000 neurons, 10 s vs 40 s to separate compile from
integration:

| dt | fixed | per biological s | projected 300 s |
|---|---|---|---|
| 0.1 ms | 5.4 s | 0.177 s | **58.5 s** (inside 60 s, not "well under") |
| 0.2 ms | 4.8 s | 0.124 s | **42.1 s** |

Apple clang here has no libomp, so OpenMP is not available; even if it were, threading
inside each SBI worker would oversubscribe the pool. `dt = 0.2 ms` is the budget lever
that helps every draw independently. Burst/avalanche analysis bins are 50 ms and
`tau_m = 20 ms` is still 100 steps, so this does not coarsen the fingerprint.

Full 300 s acceptance run (`scripts/figure_task1_bursts.py --duration 300`):

- STP: **53.7 s** wall-clock (PASS), 34 bursts, median IBI **8.55 s** (range 3.9–16.9 s),
  0.69 Hz/neuron, quiet fraction 0.87. Raster shows isolated network bursts.
- Static ablation: 54.8 s, 10.88 Hz/neuron (~16x), median IBI 3.45 s. Still fails to
  look like a culture via the hyperactive criterion (not via IBI-CV-as-metronome; at
  these defaults the ablation is dense bursting, not the CV 0.11 oscillator seen earlier).

`compute_fingerprint` now refuses `metadata["observation"] == "none"`. pytest: **159
passed**.

---

## Standing notes (do not rediscover)

**Dataset access is verified public** (details in `culturesim/data/loaders.py`). Wagenaar
is threshold crossings at `neurodatasharing.bme.gatech.edu`; channel 60 is a stimulus
marker. Braingeneers DANDI 001603 is a draft version only — pin checksums. The paper
said Wagenaar was gated; the archive is open. Schema is in the `load_wagenaar` docstring.
Do not silently substitute CRCNS hc-8 / DANDI 001611 / G-Node Kapucu.

**Three model corrections** (full writeup in `agent-log-archive.md`, comments on the
fields in `params.py`): `n_background_synapses=1000`, `w_e` is an EPSP amplitude converted
via `epsp_peak_ratio()`, `w_background` decoupled from `w_e`. Without all three the
network is silent across the prior box.

**Sharp knobs:** `rate_bg` 4.41 Hz silent / 4.98 bursting / 5.35 tonic. `g` must stay
low (g=4 is asynchronous, no bursts). `tau_rec` sets IBI. Static-synapse ablation is
hyperactive, not silent — do not test it by burst count.

**Do not retry the diffusion background.** It was 0.91x slower and shifted rates/bursts.
`scripts/check_background_modes.py` is the comparison. Poisson is the scientific default.
