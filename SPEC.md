# `culture-sim` — Implementation Spec

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

## 1. Repository Layout

```
culture-sim/
├── README.md                  # includes §0 verbatim
├── pyproject.toml
├── configs/
│   ├── model_default.yaml     # fixed + free parameter defaults
│   ├── observation.yaml       # virtual MEA config
│   ├── fingerprint.yaml       # which statistics, which weights
│   └── fit_sbi.yaml
├── culturesim/
│   ├── __init__.py
│   ├── model/
│   │   ├── network.py         # Brian2 network builder
│   │   ├── params.py          # dataclasses, free/fixed split, bounds
│   │   └── runner.py          # subprocess-isolated simulation execution
│   ├── observation/
│   │   └── virtual_mea.py     # neuron -> electrode observation model
│   ├── stats/
│   │   ├── spiketrains.py     # canonical spike data structures
│   │   ├── rates.py
│   │   ├── bursts.py
│   │   ├── avalanche.py
│   │   ├── branching.py
│   │   ├── connectivity.py
│   │   └── fingerprint.py     # assembles the full vector
│   ├── data/
│   │   ├── loaders.py         # real dataset -> canonical format
│   │   └── cache.py
│   ├── fit/
│   │   ├── distance.py
│   │   ├── coarse.py          # grid / Nelder-Mead / CMA-ES
│   │   └── sbi_fit.py         # SNPE posterior
│   ├── validate/
│   │   ├── heldout.py
│   │   ├── cross_culture.py
│   │   └── perturbation.py
│   └── cli.py
├── tests/
└── scripts/
```

---

## 2. Environment

- Python 3.11+
- `brian2`, `numpy`, `scipy`, `pandas`, `matplotlib`
- `sbi` (simulation-based inference), `torch`
- `powerlaw` (Clauset MLE fitting)
- `pyyaml`, `h5py`, `pynwb` (for NWB-format real data)
- `pytest`, `hypothesis`

Pin versions in `pyproject.toml`. Note that `pip install` in some environments requires `--break-system-packages`.

---

## 3. Canonical Data Structures

Everything — simulated and real — is converted to one representation before any statistic is computed. This is non-negotiable; it is what makes the comparison valid.

```python
@dataclass(frozen=True)
class SpikeRecording:
    """Canonical spike data. Units: seconds."""
    times: np.ndarray          # float64, shape (n_spikes,), sorted ascending
    channels: np.ndarray       # int32, shape (n_spikes,), electrode index
    n_channels: int
    duration: float            # recording length in seconds
    source: str                # 'simulation' | dataset identifier
    metadata: dict             # DIV, culture id, sampling rate, etc.
```

```python
@dataclass(frozen=True)
class Fingerprint:
    """Fixed-order summary statistic vector."""
    values: np.ndarray         # float64, shape (n_stats,)
    names: tuple[str, ...]     # frozen order, defined in fingerprint.yaml
    def __post_init__(self):   # assert len(values) == len(names)
```

**Rule: the fingerprint's `names` tuple is frozen once Task 3 is complete. Adding statistics later invalidates all prior fits. If a statistic must be added, bump a version field and re-run everything.**

---

## 4. Model Specification

### 4.1 Neuron model

Current-based leaky integrate-and-fire with spike-frequency adaptation.

```
dv/dt   = (E_L - v + I_e + I_i + I_ext - a) / tau_m   : volt (unless refractory)
dI_e/dt = -I_e / tau_e                                 : volt
dI_i/dt = -I_i / tau_i                                 : volt
da/dt   = -a / tau_a                                   : volt
```

- Threshold: `v > v_th` → emit spike
- Reset: `v = v_reset`; `a += b`
- Refractory: `t_ref`

### 4.2 Synapses — Tsodyks–Markram short-term plasticity

**This is the structural requirement. A network with static synapses cannot terminate a network burst and will never reproduce realistic inter-burst intervals no matter how it is tuned. Implement this before attempting any fitting.**

Per-synapse state, integrated between presynaptic spikes:

```
du/dt = -u / tau_f          : 1 (event-driven)
dx/dt = (1 - x) / tau_rec   : 1 (event-driven)
```

On presynaptic spike, in this exact order:

```
u += U * (1 - u)
I_post += w * u * x
x -= u * x
```

Use `(event-driven)` in the Brian2 equations so the integration is lazy — this matters a great deal for runtime.

### 4.3 Network topology

- `N = 1000` neurons (raise from 500; burst statistics are unstable below ~800)
- 80% excitatory, 20% inhibitory
- Neurons placed on a 2-D sheet matching the physical MEA sensing area
- Distance-dependent connection probability: `p(d) = p_conn * exp(-d / lambda_conn)`
- Background drive: independent Poisson input to each neuron at `rate_bg`

### 4.4 Parameter split

**Free (8) — these are what you fit:**

| name | symbol | prior range | units |
|---|---|---|---|
| connection probability | `p_conn` | 0.01 – 0.30 | 1 |
| excitatory weight | `w_e` | 0.05 – 3.0 | mV |
| inhibition/excitation ratio | `g` | 1.0 – 12.0 | 1 |
| background rate | `rate_bg` | 0.1 – 20.0 | Hz |
| membrane time constant | `tau_m` | 10 – 40 | ms |
| STP utilization | `U` | 0.05 – 0.8 | 1 |
| STP recovery | `tau_rec` | 100 – 5000 | ms |
| adaptation increment | `b` | 0.0 – 5.0 | mV |

**Fixed — set from literature and documented with a citation comment in `params.py`:**
`E_L = -70 mV`, `v_th = -50 mV`, `v_reset = -60 mV`, `t_ref = 2 ms`, `tau_e = 5 ms`, `tau_i = 10 ms`, `tau_a = 200 ms`, `tau_f = 100 ms`, `lambda_conn` = one third of array width.

Every fixed parameter needs a one-line comment naming its source. Reviewers ask.

### 4.5 Execution — Brian2 standalone gotcha

Brian2's `cpp_standalone` device does not cleanly support many sequential runs in one process. **Do not** try to loop `device.reinit()` thousands of times.

Required approach: `runner.py` executes each simulation in a **separate subprocess** with its own build directory under a temp root, returns a `SpikeRecording`, and cleans up. Parallelize with `multiprocessing.Pool` sized to `cpu_count() - 1`. Each subprocess sets its own `numpy` and Brian2 seed derived deterministically from a master seed plus run index.

Target: a 300-second biological-time simulation should complete in well under 60 seconds wall-clock. If it does not, profile before proceeding — the SBI stage needs thousands of these.

---

## 5. Observation Model (`virtual_mea.py`)

**This module is the scientific crux of the project. Statistics must be computed on simulated data that has passed through the same observational bottleneck as the real data. Computing statistics on all 1000 simulated neurons and comparing them to statistics from 60 electrodes is invalid, and it is the single most common error in this literature.**

```python
def observe(spikes_neuron_level, config) -> SpikeRecording
```

Implementation:

1. Place `n_electrodes` on a regular grid over the sheet, matching the geometry of the target real dataset (electrode count, pitch, array extent). Configurable — MCS 60-electrode and MaxWell HD-MEA layouts both needed.
2. Each neuron's spike produces an extracellular amplitude at each electrode decaying with distance: `A = A_0 / (1 + (d / d_0)**2)`.
3. Add Gaussian recording noise with the real system's RMS (2–5 µV typical).
4. A spike is detected on an electrode if `A > k * noise_rms`, with `k` configurable (default 5, the common threshold convention).
5. Apply a per-electrode dead time to mimic detector refractoriness.
6. Optionally mark a configurable fraction of electrodes as dead/broken — real arrays always have some.

Output is a `SpikeRecording` with `n_channels = n_electrodes`. Everything downstream sees only this.

---

## 6. Statistics (`stats/`)

Each function takes a `SpikeRecording` and returns floats or arrays. All must work identically on simulated and real data.

### 6.1 Rates (`rates.py`)
- Mean firing rate per electrode; return mean, std, and the 10th/50th/90th percentiles of the across-electrode distribution. **The heterogeneity is a target, not noise** — real arrays have a few hot electrodes and many quiet ones. A uniform simulation is a failed simulation.
- ISI coefficient of variation, pooled and per-electrode.
- Fraction of active electrodes (rate > 0.01 Hz).

### 6.2 Bursts (`bursts.py`)
Network burst detection: bin all-array spike counts at 25 ms; a burst starts when the count exceeds a threshold derived from the surrogate distribution (shuffle ISIs to build a null), and ends when it drops below. Enforce a minimum duration and a minimum electrode participation.

Return: burst rate (per minute), mean and std duration, spikes per burst, fraction of electrodes participating, and the **full inter-burst-interval distribution** (report log-spaced histogram plus median and IQR). Real IBIs span roughly 1–300 seconds; that wide range is a strong discriminator and hard to fake.

### 6.3 Avalanches (`avalanche.py`)
- Bin width Δt = mean inter-spike interval across the whole array. This is the standard convention; hard-coding a fixed bin width is a known way to manufacture or destroy power laws.
- An avalanche is a run of consecutive non-empty bins bracketed by empty bins. Size = total spikes; duration = number of bins.
- Fit exponents with the `powerlaw` package (Clauset MLE with `xmin` estimation), for sizes (`alpha`) and durations (`beta`).
- **Compute the crackling-noise scaling relation**: fit `gamma` from the average-size-given-duration relation `<S>(D) ~ D**gamma`, then report the discrepancy `|gamma - (beta - 1)/(alpha - 1)|`. Matching one exponent is easy; matching the relation between them is not, which is exactly why it belongs in the fingerprint.
- Always compare the power-law fit against a lognormal alternative and report the loglikelihood ratio. Never claim a power law without it.

### 6.4 Branching ratio (`branching.py`)
**Do not use the naive descendants/ancestors estimator.** It is severely biased under subsampling, and it biases toward reporting the network as more subcritical than it is — the bias depends on how many electrodes you have, so it produces different answers from identical biology.

Implement the multistep-regression (MR) estimator (Wilting & Priesemann): compute the autocorrelation `r_k` of the binned population activity for lags `k = 1..k_max`, fit `r_k = C * m**k`, and recover `m` from the fitted decay. Return `m` plus the fit quality.

Include the naive estimator too, clearly labelled, purely so the bias can be demonstrated in a figure.

### 6.5 Connectivity (`connectivity.py`)
Pairwise cross-correlation between electrode spike trains at a fixed bin width; threshold against a jitter-corrected surrogate null to build a binary functional connectivity graph. Return mean degree, degree-distribution skew, clustering coefficient.

### 6.6 Fingerprint assembly (`fingerprint.py`)
Assemble all of the above into a fixed-order vector. Scalars enter directly; distributions enter as a small fixed set of quantiles (10/25/50/75/90) plus a log-spaced histogram of fixed bin edges. Frozen order defined in `fingerprint.yaml`.

---

## 7. Real Data (`data/loaders.py`)

Write one loader per dataset, each returning `SpikeRecording`.

Primary target: the Wagenaar/Potter developmental dataset (dissociated rat cortical cultures, dense longitudinal recordings across the first weeks in vitro). Its longitudinal structure is what the downstream drift work needs.

Secondary: an organoid dataset from the Braingeneers DANDI archive, in NWB format via `pynwb`, for a 3-D comparison point.

**Before writing any loader, verify the accession/URL actually resolves and check the data-availability statement.** Do not assume a dataset is public because a paper says so. If a target dataset turns out to be gated, report this and stop rather than substituting silently.

Handle: spike-sorted vs threshold-crossing data (prefer threshold crossings for comparability with the observation model), differing sampling rates, and electrode geometry metadata.

---

## 8. Fitting (`fit/`)

### 8.1 Distance (`distance.py`)
```python
def distance(fp_sim: Fingerprint, fp_real: Fingerprint, weights) -> float
```
Z-score each statistic by its **across-culture** variability in the real dataset (not its within-culture variability) so that no single statistic dominates and so the tolerance reflects genuine biological spread. Distributional components compared by Wasserstein distance on the log-spaced histograms. Weights configurable; default uniform after z-scoring.

### 8.2 Coarse fit (`coarse.py`)
Grid search over 2–3 parameters (`w_e`, `g`, `tau_rec`) to establish that bursting occurs at all and to locate the rough neighbourhood. Then Nelder-Mead or CMA-ES over all 8 for a point estimate.

**Do not skip the grid stage.** It builds the intuition for which knob does what, and it catches the case where the model cannot produce the target behaviour at all — which is a structural bug, not an optimization failure, and no optimizer will tell you the difference.

### 8.3 SBI (`sbi_fit.py`)
Use `sbi` with SNPE-C:

1. Uniform prior over the 8-parameter box in §4.4
2. Simulate 3000–5000 parameter draws (overnight on a laptop with the subprocess pool)
3. Embedding network over the fingerprint vector
4. Train the density estimator, condition on the real fingerprint, sample the posterior

**The posterior is the deliverable, not a point estimate.** A tight marginal means that parameter is genuinely identified by the data. A flat marginal means the data cannot see it — which is a finding, not a failure, and must be reported as such. Also report pairwise posterior correlations, which reveal parameter degeneracies.

Run posterior predictive checks: sample parameters from the posterior, simulate, and confirm the resulting fingerprints bracket the real one.

---

## 9. Validation (`validate/`)

Three tests, increasing in strength. Report all three regardless of outcome.

1. **Held-out statistics** (`heldout.py`) — fit using only rate and burst statistics; then check whether avalanche exponents and the crackling relation come out right *without having been fitted*. If they do, the model captured mechanism rather than curve-fitting.

2. **Cross-culture** (`cross_culture.py`) — fit culture A, then independently fit culture B, and check the posteriors are neighbours in parameter space rather than scattered. Quantify with posterior overlap.

3. **Perturbation response** (`perturbation.py`) — **this is the test that actually matters.** Fit on spontaneous activity only, then check whether the model predicts the *evoked* response to stimulation: PSTH shape, response probability vs. stimulus amplitude, and post-stimulus network burst probability. Every downstream project stimulates the culture, so a model that only matches resting behaviour is not fit for purpose. If this fails, say so plainly in the README rather than quietly dropping the test.

---

## 10. CLI (`cli.py`)

```
culture-sim simulate   --config configs/model_default.yaml --out run.h5
culture-sim fingerprint --input run.h5 --out fp.json
culture-sim fit coarse  --data <dataset> --out coarse.json
culture-sim fit sbi     --data <dataset> --n-sims 5000 --out posterior.pkl
culture-sim validate    --posterior posterior.pkl --test all
culture-sim report      --out report.html
```

---

## 11. Reproducibility Requirements

- Every run writes a manifest: git commit hash, full config, master seed, package versions, wall-clock time.
- All randomness flows from a single master seed via explicit `np.random.Generator` instances. No global `np.random.seed`, no bare `random`.
- Simulation outputs cached by config hash so re-running is cheap.
- Every figure regenerable from a single command.

---

## 12. Testing (`tests/`)

Statistics code must be tested against inputs with analytically known answers. Minimum set:

- Homogeneous Poisson process → ISI CV ≈ 1.0 (within tolerance); MR branching ratio ≈ 0
- Known critical branching process simulated directly → MR estimator recovers `m` within 5%, and the naive estimator demonstrably does not under subsampling (this test *documents the bias*)
- Synthetic power-law sample with known exponent → `powerlaw` fit recovers it
- Regular spike train → ISI CV ≈ 0
- Empty recording, single-spike recording, single-active-electrode recording → all statistics return without crashing, with documented sentinel values
- Round-trip: `SpikeRecording` → HDF5 → `SpikeRecording` is identity

Property-based tests via `hypothesis` for the spike-train data structures (sortedness, channel bounds, duration consistency).

---

## 13. Task Order and Acceptance Criteria

**Task 0 — Scaffold.** Repo layout, `pyproject.toml`, config schemas, `SpikeRecording` and `Fingerprint` dataclasses with HDF5 round-trip, CI running pytest.
*Accept:* `pytest` passes on the round-trip and property tests. `culture-sim --help` works.

**Task 1 — Network with STP.** Implement §4 in full including Tsodyks–Markram synapses and the subprocess runner.
*Accept:* A raster plot from a hand-tuned parameter set shows clear network bursts separated by quiet periods, with inter-burst intervals in the 1–60 s range. A static-synapse ablation is included and demonstrably fails to produce them. Simulation of 300 s biological time completes in under 60 s wall-clock.

**Task 2 — Virtual MEA.** Implement §5 with configurable geometry for both a 60-electrode and an HD-MEA layout.
*Accept:* Given identical neuron-level spikes, a 60-electrode and a 1024-electrode observation yield measurably different fingerprints, and the difference is documented in a figure. Dead-electrode simulation works.

**Task 3 — Statistics.** Implement §6 in full. Freeze `fingerprint.yaml`.
*Accept:* All tests in §12 pass, including the demonstration of naive-estimator subsampling bias. Fingerprint computes end-to-end on simulated data in under 10 s.

**Task 4 — Real data.** Implement §7. Verify dataset access first.
*Accept:* At least one real recording loads into `SpikeRecording` and produces a complete fingerprint. Values are printed alongside published values for that dataset where available, with any discrepancy investigated and explained.

**Task 5 — Coarse fit.** Implement §8.1–8.2.
*Accept:* Distance to the real fingerprint is reduced by at least 50% relative to the hand-tuned Task 1 parameters. A distance-landscape figure over the 2-D grid exists.

**Task 6 — SBI.** Implement §8.3.
*Accept:* Posterior over 8 parameters obtained from ≥3000 simulations. Marginal and pairwise plots produced. Posterior predictive check shows simulated fingerprints bracketing the real one. Identified vs. unidentified parameters explicitly listed in the README.

**Task 7 — Validation.** Implement §9.
*Accept:* All three tests run and their results reported honestly in the README, including failures.

**Task 8 — Report.** `culture-sim report` produces a single HTML document with the scope statement, fitted posteriors, all validation results, and stated limitations.
*Accept:* Report regenerates from scratch with one command on a clean checkout.

---

## 14. Known Pitfalls

- **Scope creep in calibration.** There is always one more statistic to match. The fingerprint is frozen at Task 3. If you are still adding statistics at Task 7, the project has drifted from building a tool to collecting a hobby. Stop.
- **Brian2 standalone in a loop** will fail or leak. Subprocess isolation, always.
- **Comparing full-network statistics to electrode-level statistics** invalidates everything downstream. The observation model is not optional.
- **Manufactured power laws.** Fixed bin widths, no `xmin` estimation, and no lognormal comparison will produce "power laws" from almost anything. Follow §6.3 exactly.
- **Fitting to a single culture** and claiming generality. Cross-culture validation exists for a reason.
- **Silent dataset substitution.** If the intended dataset is inaccessible, stop and report rather than quietly switching to something else with different geometry.

---

## 15. Definition of Done

The repo produces, from a clean checkout and a single command, a report stating: this model reproduces statistics X, Y, Z within quantified tolerance of dataset D; parameters P1, P2 are identified by the data and P3, P4 are not; the model does/does not predict evoked responses; and it is not valid for the purposes listed in §0.

That last clause is the whole point. A tool whose limits are precisely stated is usable by other people. One that claims to be realistic is not.