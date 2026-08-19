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

---

## Current status

**Task 0 (scaffold) is complete. Nothing has been fitted, and no claim in `SPEC.md`
§15 is yet supported by evidence.** The table below is the honest state of the build;
`SPEC.md` §13 has the acceptance criterion for each task.

| Task | What it covers | State |
|---|---|---|
| 0 | Scaffold, configs, canonical data structures, HDF5 round-trip, CI | **Done** |
| 1 | Brian2 network with Tsodyks–Markram synapses, subprocess runner | Not started |
| 2 | Virtual MEA observation model | Layouts and config done; detection not started |
| 3 | Statistics and the frozen fingerprint | Contracts and frozen order defined; not implemented |
| 4 | Real data loaders | **Blocked: dataset access not verified** (see below) |
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

### Dataset access is unverified

`SPEC.md` §7 requires confirming that a dataset actually resolves before writing a
loader, and forbids silently substituting a different one. Neither target has been
checked yet, so both are marked `unverified` in `culturesim/data/loaders.py` and the
fit commands refuse to run against them:

```
$ culture-sim fit coarse --data wagenaar2006 --out coarse.json
not implemented: dataset 'wagenaar2006' has access state 'unverified'. SPEC §7
requires verifying that ... resolves and reading its data-availability statement
before a loader is written. Do not substitute another dataset.
```

---

## Install

The repo expects the virtual environment at `.venv`:

```bash
python3 -m venv .venv                 # if it does not exist yet
.venv/bin/python -m pip install -e ".[dev]"
```

Dependency versions are pinned exactly in `pyproject.toml`. That is deliberate: a
fitted posterior is only reproducible against the library versions that produced it,
and every run records those versions in its manifest. In some environments
`pip install` needs `--break-system-packages`; prefer the venv.

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
  stats/          statistics and the fingerprint vector
  data/           dataset loaders and the simulation cache
  fit/            distance, coarse search, SBI
  validate/       held-out, cross-culture, perturbation tests
scripts/          check_environment.py, freeze_fingerprint.py, figure generation
tests/
```

`tests/test_stats_contracts.py` holds the SPEC §12 statistical tests -- each written
against an input with an analytically known answer, and marked
`xfail(strict=True, raises=NotImplementedError)` until Task 3 implements the function.
`strict=True` is deliberate: once a statistic is implemented its test stops raising and
pytest reports the unexpected pass as a failure, forcing the marker to be removed rather
than leaving a permanently-green placeholder.

Three design rules run through all of it, each of which exists because breaking it
invalidates results rather than merely making them worse:

**Everything becomes a `SpikeRecording` first.** Simulated and real data are converted
to one representation (`culturesim/stats/spiketrains.py`) before any statistic is
computed. That is what makes the comparison valid.

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

## Reproducibility

- Every run writes a manifest beside its output: git commit, full config, master seed,
  package versions, wall-clock time.
- All randomness flows from one master seed through explicit `np.random.Generator`
  instances. There is no global `np.random.seed` and no bare `random`, and
  `tests/test_reproducibility.py` parses the package to enforce that.
- Simulation outputs are cached by config hash, so re-running is cheap.

## License

MIT.
