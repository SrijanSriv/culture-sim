# Agent log

Working brief for agents. Read this before starting; append to it before finishing. See
`AGENTS.md` §0 for the protocol. Older or superseded entries live in
`agent-log-archive.md`.

Newest entries at the top.

---

## 2026-08-21 — Task 6 MCMC sampling (rejection hung)

Agent: Cursor Grok. Branch `main`.

Sims finished 3000/3000 (kept 2372). SNPE training converged (101 epochs).
Default rejection/direct sampling sat at 0/10000 for >8 min. Stopped the
worker; switched `configs/fit_sbi.yaml` to `sample_with: mcmc` (2000 draws)
and re-launch from the existing checkpoint — no Brian2 re-sim.

---

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 6 (machinery; overnight sims).

SBI is designed so an agent (or human) starts it and walks away:

- `culture-sim fit sbi ... --detach` returns immediately after spawning a worker
- `output/task6_status.json` updates every batch (`running` → `done`/`failed`)
- README Task 6 row is rewritten from that status file after every batch and on
  finish (including identified / unidentified parameter lists)
- Check later: `.venv/bin/python scripts/check_task6.py` (exit 0=done, 2=running)
- Checkpoints at `output/posterior.checkpoint.npz`; resume is the default

Config: 3000 sims, 60 s biological per draw (same as Task 5; edit
`configs/fit_sbi.yaml` for 300 s). PPC draws 50 posterior samples at the end.

Do **not** wait on the agent for this. Launch with `--detach`, leave the machine.

### Still open

1. Wait for the detached campaign to finish, then confirm README shows **Done**
   and commit the status/README result if desired.
2. Tasks 7–8.

---

## 2026-08-20 — Task 5 coarse fit

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 5.

Acceptance: baseline 4.50 → best 1.29 (**71%**). Landscape figure written.
Half the grid cells were non-finite; finite half still found a bursting
neighbourhood. Duration per draw 60 s.

---

## 2026-08-19 — Task 4 Wagenaar loader

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 4.

`load_wagenaar` reads `.spk.txt.bz2`, splits channel 60 as stim, recovers
plating/culture/DIV from the filename. Default `1-1-14`: 224,547 spikes,
duration **2716 s**, complete 66-stat fingerprint. CL median IBI 0.30 s vs
Wagenaar published 1–300 s is a detector mismatch (documented).

---

## 2026-08-19 — Task 3 fingerprint freeze

Agent: Cursor Grok. Branch `main`. SPEC §13 Task 3.

Froze `configs/fingerprint.yaml` at version **1.0.0**, 66 statistics, SHA-256
`603d25df56e75289848d69e764a4d5ef74eede671b2c5a97821af7afc171df0e`. CI already runs
`scripts/freeze_fingerprint.py --check`. Adding, removing, or reordering an entry after
this requires bumping the version and re-running every fit.

§12 tests already passed (naive-estimator subsampling bias included). `cl.analysis`
imports are confined to `interop/` by `tests/test_reproducibility.py`. Fingerprint
end-to-end is under 10 s (`test_fingerprint_computes_within_the_time_budget`).

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
