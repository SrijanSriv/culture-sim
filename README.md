# culture-sim

A parameter-calibrated in-silico model of a dissociated neuronal culture on a
multi-electrode array, implemented in Brian2 and fitted by simulation-based inference.

The scope statement below is reproduced verbatim from `SPEC.md` §0, as that section
requires. **Read it before using this model for anything.**

---

## 0. Purpose and Scope

### What this is

A parameter-calibrated in-silico model of a dissociated neuronal culture on a multi-electrode array (MEA), implemented in Brian2. The model is fitted to reproduce a defined list of statistical properties of real MEA recordings, and ships with quantified uncertainty over its fitted parameters.

### What it is for

This model exists to serve as a validated test bench for downstream experiments on:

1. Closed-loop feedback timing (how loop latency and jitter affect STDP-based learning)
2. Decoder drift and stabilization across days
3. Online control of network criticality

### What it must reproduce (in scope)

- Per-electrode firing rate distribution
- Inter-spike-interval statistics
- Network burst rate, duration, size, and participation
- Inter-burst-interval distribution
- Neuronal avalanche size and duration distributions and their scaling relation
- Subsampled branching ratio
- Functional connectivity degree distribution
- Evoked response to electrical stimulation (validation only, not fitted)

### What it does NOT model (explicit non-goals)

- Dendritic computation, morphology, or compartmental detail
- Glia, astrocytes, or metabolic state
- Neurotransmitter diffusion or volume transmission
- Developmental changes in cell count or physical network growth
- Spatial wave propagation across the culture
- Ion channel dynamics below the LIF abstraction
- 3-D organoid geometry (this is a 2-D dissociated culture model)

**This scope statement must be reproduced verbatim in the repo README. It is the honest-limitations section of any resulting paper, written before there is any incentive to fudge it.**

### Relationship to existing tools

The Cortical Labs CL SDK Simulator is **not** prior art for this project and does not overlap with it. Its default data source generates spikes from a Poisson distribution or replays a fixed recording; it contains no neuron model, no synapses, no plasticity, and produces no response to stimulation. It is an API mock for developing CL1 application code without hardware, which is a different purpose.

The two are complementary, and this is the intended long-term integration path: expose `culture-sim` as a custom CL data source so that application code written against the CL API can be driven by a calibrated network model instead of Poisson noise. **Do not build this integration during Tasks 0–8.** It is noted here only so that the `SpikeRecording` interface and the observation model are designed to make it straightforward later.

---

## Current status

**Tasks 0–3 are complete. Nothing has been fitted, and no claim in `SPEC.md`
§15 is yet supported by evidence.** The table below is the honest state of the build;
`SPEC.md` §13 has the acceptance criterion for each task.

| Task | What it covers | State |
|---|---|---|
| 0 | Scaffold, configs, canonical data structures, HDF5 round-trip, CI | **Done** |
| 0.5 | CL SDK API probe, license/install check, H5 schema, analysis signatures | **Done** (`culturesim/interop/CL_API_PROBE.md`) |
| 1 | Brian2 network with Tsodyks–Markram synapses, subprocess runner | **Done** (300 s biological in 53.7 s wall-clock; poisson drive; `dt=0.2` ms) |
| 2 | Virtual MEA observation model + CL adapter | **Done** (60-vs-1024 figure; dead electrodes; HD-MEA CL stats blocked by `cl-sdk==1.0.0` uint8 channel ids) |
| 3 | Statistics and the frozen fingerprint | **Done** (66 statistics, version 1.0.0, hash `603d25df56e7`) |
| 4 | Real data loaders | Access verified (see below); loaders not written |
| 5 | Coarse fit | Not started |
| 6 | SBI posterior | Not started |
| 7 | Validation suite | Not started |
| 8 | HTML report | Not started |

### Results not yet available

These sections exist so that they are filled in rather than quietly omitted:

- **Which parameters the data identifies** (SPEC §8.3) — requires Task 6. A flat
  posterior marginal is a finding about what MEA statistics can constrain, and will be
  reported as such rather than collapsed to a point estimate.
- **Held-out statistics** (SPEC §9.1) — requires Task 7.
- **Cross-culture posterior overlap** (SPEC §9.2) — requires Task 7.
- **Whether the model predicts evoked responses** (SPEC §9.3) — requires Task 7. This
  is the test that matters, since every downstream project stimulates the culture. If
  it fails, that failure gets stated here plainly.

### Dataset access is verified

`SPEC.md` §7 requires confirming that a dataset actually resolves before writing a
loader, and forbids silently substituting a different one. Both targets were checked on
2026-08-19 by downloading real data, and both are public:

| Target | State | Evidence |
|---|---|---|
| `wagenaar2006` | **public**, no registration | downloaded a 874 KB spike file (224,547 events) from `neurodatasharing.bme.gatech.edu` |
| `braingeneers_organoid` | **public** (DANDI 001603) | anonymous byte-range fetch returned valid HDF5 |

Worth recording because it inverts the usual warning: the Wagenaar paper tells readers to
email the author for access and the original Potter lab host is dead, so the *publication*
reads as gated while the archive is in fact open. Verification beats the paper in both
directions.

Two caveats are carried in `culturesim/data/loaders.py`. Wagenaar provides threshold
crossings rather than sorted units — which is what §7 prefers, since the virtual MEA
models threshold detection. DANDI 001603 exists only as a draft version, so asset
checksums must be pinned for a fit against it to stay reproducible.

The loaders themselves are still Task 4, so the fit commands exit with code 3:

```
$ culture-sim fit coarse --data wagenaar2006 --out coarse.json
not implemented: Task 4 (SPEC §7) -- loader not yet written
```

---

## Install

The repo expects the virtual environment at `.venv`:

```bash
python3.12 -m venv .venv              # or any Python >= 3.12
.venv/bin/python -m pip install -e ".[dev]"
```

Dependency versions are pinned exactly in `pyproject.toml`. That is deliberate: a
fitted posterior is only reproducible against the library versions that produced it,
and every run records those versions in its manifest. `cl-sdk==1.0.0` is pinned for
the same reason: delegated statistic definitions are part of the scientific method.
In some environments `pip install` needs `--break-system-packages`; prefer the venv.

`cl-sdk==1.0.0` reports license `CC BY-NC 4.0`. That is compatible with
non-commercial research use with attribution, but not with unrestricted commercial use
or a pure-MIT redistribution story. Treat this as a release constraint.

Verify:

```bash
.venv/bin/python scripts/check_environment.py   # imports + Brian2 standalone compiles
.venv/bin/python -m pytest -q
.venv/bin/culture-sim --help
```

`check_environment.py` builds a throwaway `cpp_standalone` simulation, because that
device needs a working C++ compiler and the SPEC §4.5 runtime budget is unreachable
without it. Verified working on Python 3.13.3 with Brian2 2.10.1.

## Usage

```
culture-sim simulate    --config configs/model_default.yaml --out run.h5
culture-sim fingerprint --input run.h5 --out fp.json
culture-sim fit coarse   --data <dataset> --out coarse.json
culture-sim fit sbi      --data <dataset> --n-sims 5000 --out posterior.pkl
culture-sim validate     --posterior posterior.pkl --test all
culture-sim report       --out report.html
```

Commands whose task has not landed exit with code 3 and an explanatory message. That
is distinct from failure (code 1) on purpose: a half-built pipeline should not look
broken, and a broken one should not look unbuilt.

## How it is put together

```
configs/          model, observation, fingerprint and SBI configs
culturesim/
  model/          Brian2 network, parameter split, subprocess runner
  observation/    virtual MEA: neuron spikes -> electrode spikes
  interop/        CL H5 adapter and delegated CL analysis wrappers
  stats/          statistics and the fingerprint vector
  data/           dataset loaders and the simulation cache
  fit/            distance, coarse search, SBI
  validate/       held-out, cross-culture, perturbation tests
scripts/          check_environment.py, freeze_fingerprint.py, figure generation
tests/
```

`tests/test_stats_contracts.py` holds the SPEC §12 statistical tests -- each written
against an input with an analytically known answer or a documented CL delegation
sentinel. These now run as real tests rather than Task 3 placeholders.

Three design rules run through all of it, each of which exists because breaking it
invalidates results rather than merely making them worse:

**Everything becomes a `SpikeRecording` first.** Simulated and real data are converted
to one representation (`culturesim/stats/spiketrains.py`) before any statistic is
computed. New recordings write CL recording H5 as the native on-disk format, with a
small culture-sim sidecar for exact round-trip identity.

**Statistics are only ever computed on electrode-level data.** The virtual MEA
(`culturesim/observation/virtual_mea.py`) is the observational bottleneck: neurons are
observed through a finite number of electrodes with a distance-dependent amplitude, a
detection threshold, dead time, and some dead channels. Comparing statistics from 1000
simulated neurons against statistics from 60 electrodes is invalid, and it is the most
common error in this literature.

**The fingerprint order freezes at Task 3.** Adding a statistic later invalidates every
fit made against the old vector. `configs/fingerprint.yaml` records a hash of the
expanded statistic names, CI checks it, and the loader refuses to run a frozen spec
whose order has changed. Adding statistics at Task 7 means the project has drifted from
building a tool into collecting a hobby.

**CL statistics stay behind `interop`.** `cl.analysis` is never imported directly from
`stats/` or `fit/`; `culturesim/interop/cl_analysis.py` is the only boundary. This keeps
upstream convention changes visible and makes a fallback possible if the SDK changes.

## Reproducibility

- Every run writes a manifest beside its output: git commit, full config, master seed,
  package versions, wall-clock time.
- All randomness flows from one master seed through explicit `np.random.Generator`
  instances. There is no global `np.random.seed` and no bare `random`, and
  `tests/test_reproducibility.py` parses the package to enforce that.
- Simulation outputs are cached by config hash, so re-running is cheap.

## License

MIT.
