# Agent log

Working brief for agents. Read this before starting; append to it before finishing. See
`AGENTS.md` §0 for the protocol. Older or superseded entries live in
`agent-log-archive.md`.

Newest entries at the top.

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

### Still open

1. Task 2 remainder: 60-vs-1024 fingerprint comparison figure; HD-MEA CL channel-id
   blocker in `cl-sdk==1.0.0`.
2. Task 3: freeze `fingerprint.yaml`.
3. Task 4: write `load_wagenaar`.

---

## 2026-08-19 — commit the parallel Task 0.5–4 work

Agent: Cursor Grok. Branch `main`.

Split the previously uncommitted tree into seven focused commits (see `git log`).
Verified before committing: `ruff check` + `ruff format --check` clean; `pytest -q`
→ **156 passed**. `figures/` stays gitignored.

Commits (newest last among this batch): SPEC/`cl-sdk` pin → CL interop + H5 → Task 1
network → virtual MEA → Task 3 stats → dataset access verification → README/log.

### Still open (unchanged by the commit pass)

Superseded by the Task 1 runtime-budget entry at the top. Remaining: Task 2 60-vs-1024
figure, fingerprint freeze, `load_wagenaar`.

---

## 2026-08-19 — Task 1 (network + runner), and a blocking spec change

Agent: Claude Opus 5. Branch `main`, originated at `7eec5cb`; technical findings below.
Procedural "uncommitted / parallel edits" framing was resolved by the commit pass above.

### Spec change that landed mid-Task-1 (archived detail)

Task 0.5 / §6.0 CL delegation / CL-native H5 / Python 3.12+ were added to `SPEC.md`
while the network work was in flight. Narrative moved to `agent-log-archive.md`; the
live contract is `SPEC.md`.

### `cl-sdk` — installed and probed (see `interop/CL_API_PROBE.md`)

Pin is `cl-sdk==1.0.0`; license recorded as CC BY-NC 4.0. Task 0.5 acceptance is met
by the probe document plus the adapter smoke tests.

### Dataset access: VERIFIED PUBLIC — Task 4 is unblocked

Both `SPEC.md` §7 targets are open, verified on 2026-08-19 by downloading real bytes
rather than by loading a landing page. States and evidence are recorded in
`culturesim/data/loaders.py`; `available_datasets()` now returns both, so the CLI's
access gate no longer refuses them.

- **`wagenaar2006` — PUBLIC, no registration.** Live at
  `neurodatasharing.bme.gatech.edu/development-data/`. Confirmed by downloading
  `simple-text/daily/spont/dense/1-1-14.spk.txt.bz2` (874,682 bytes, 224,547 events).
  Directory indexes are enabled. Citation requirement; bundled code GPL v2.
- **`braingeneers_organoid` — PUBLIC**, DANDI **001603**, OPEN, 111 assets / 322 GB,
  CC-BY-4.0, anonymous byte-range download confirmed. **Draft version only, no immutable
  published version**, so pin asset checksums in the manifest or a refit is not
  reproducible.

**The interesting failure mode here ran opposite to the one `SPEC.md` §7 warns about.**
The spec warns against assuming a dataset is public because a paper says so. In this case
the paper says the opposite of the truth: the BMC Neuroscience text says to email Steve
Potter for access, and the old `potterlab.gatech.edu` host is dead, so the publication
reads as gated while the archive is wide open. Verifying beats trusting the paper in
*either* direction.

Wagenaar gives **threshold crossings, not sorted units**, which is what `SPEC.md` §7
prefers — the virtual MEA models threshold detection, so sorted units would be compared
against something the observation model does not reproduce.

The full file schema needed to write the loader (path scheme, column format, 25 kHz,
0.3335 uV LSB, 8x8-minus-corners geometry, and the fact that **channel 60 is a stimulus
marker rather than an electrode**) is documented in the `load_wagenaar` docstring.

Also verified and recorded in `VERIFIED_ALTERNATIVES`, none of them a substitute without
saying so: **CRCNS hc-8** is GATED (free but manually-reviewed account request; otherwise
the closest match to Wagenaar and sorted); **DANDI 001611** is public, dissociated, 2-D
and the only candidate with an immutable published version, but its protocol is repeated
stimulation rather than spontaneous activity; **G-Node Kapucu** is public but 2.3 TiB and
on Axion plates rather than MCS. `EMPTY_DANDISETS` names four zero-asset dandisets that
rank well in search and waste time.

Remaining Task 4 work: write the loader, then satisfy the §13 acceptance criterion —
print fingerprint values alongside published values for the dataset and explain any
discrepancy.

### Task 1: the network now bursts, and three model corrections were needed

The network was silent at every point in the prior box as originally written. Three
separate causes, all worth understanding because each is invisible in the equations:

**1. `rate_bg` needs a background afferent count.** With one afferent per neuron,
crossing the 20 mV rest-to-threshold gap needs ~4.7 kHz of drive — three orders of
magnitude outside the 0.1–20 Hz prior. Added `n_background_synapses = 1000` to
`FixedParams` (Brunel 2000 uses `C_ext = C_E = 1000`).

**2. `w_e` is an EPSP amplitude, not a current jump.** This is the important one. Fed
directly into `I_e`, a `w_e` of 0.85 mV produces a membrane deflection of only ~0.03 mV,
because the current is attenuated by the membrane filter and then scaled by `U`.
Recurrent excitation therefore cannot recruit anything, at any parameter setting. The
peak deflection per unit current jump is
`(tau_syn/(tau_m - tau_syn)) * (exp(-t*/tau_m) - exp(-t*/tau_syn))` — **0.157** at the
defaults, so a synapse must carry ~6.4x its target EPSP amplitude. See
`epsp_peak_ratio()` and `synaptic_current_jumps()` in `model/network.py`. Weights are
also divided by `U` so `w_e` means the amplitude a *rested* synapse delivers and `U`
controls only dynamics.

**3. The background amplitude must be decoupled from `w_e`.** While the background was
driven at `w_e`, baseline membrane noise scaled with recurrent gain: raising `w_e` to get
bursts to ignite simultaneously raised the spontaneous rate into tonic firing, so no
sparse-baseline bursting regime existed anywhere in the prior. Added
`w_background = 0.1 mV` to `FixedParams`. Now `rate_bg` sets how far below threshold the
network idles and `w_e` sets how explosively it recruits.

Both new constants are absent from `SPEC.md` §4.4. Both are required for the model to
function; the reasoning is documented at each field in `model/params.py`.

### Hand-tuned Task 1 parameters (in `configs/model_default.yaml`)

`p_conn 0.20, w_e 1.5, g 2.0, rate_bg 4.98, tau_m 20, U 0.25, tau_rec 2500, b 2.5`

Result at 300 s: 27 bursts (5.4/min), median IBI 10.0 s, baseline **0.007 Hz/neuron**
between bursts. Three of these are sharp — do not nudge them casually:

- `rate_bg` puts the mean background depolarisation at ~15.8 mV against a 20 mV gap. The
  window is narrow: 4.41 Hz is silent, 5.35 Hz is continuously active with no quiet
  periods.
- `g` must stay low. By `g = 4` excitation and inhibition balance, which suppresses
  synchrony into an asynchronous irregular state with **no bursts at all**.
- `tau_rec` sets the IBI, since depleted synaptic resources are what end a burst.

### Static-synapse ablation passes, but not the way you would expect

It does not go silent — it degenerates into a **metronome**: median IBI 1.03 s with
CV 0.11, at 17.7 Hz/neuron versus 0.91 for the STP network. So the acceptance check is
"not culture-like bursting" (degenerate IBI spread or a far higher rate), not "fewer
bursts". A count-based check passes the ablation and is wrong.

### Runtime: 66 s projection — superseded

The 66 s / 0.1 ms projection and the "OpenMP not tried" note are closed by the
runtime-budget entry at the top of this file. Component costs from that profile are
in `agent-log-archive.md`.

### Dead end: do not retry the diffusion-approximation background

I replaced `PoissonInput` with an equivalent Ornstein-Uhlenbeck current in `I_e` (moments
matched exactly: mean `J*R*tau_e`, variance `J^2*R*tau_e/2`) expecting a large speedup,
since the background was 6.3 s of a 16.9 s run.

**It was 0.91x — slower, not faster.** Brian2's `xi` noise costs one `randn` per neuron
per timestep, which is no cheaper than the binomial sampler. It also shifted statistics:
rate 0.433 vs 0.565 Hz/neuron, 8.7 vs 11.3 bursts, because the network sits close enough
to threshold to be sensitive to fluctuation shape.

The code is kept as `background_mode = "poisson" | "diffusion"` with `poisson` as the
default, so the approximation stays testable. `scripts/check_background_modes.py`
reproduces the comparison. Do not spend time on this again.

### Task 2 partially done

`observe()` in `observation/virtual_mea.py` is implemented: per-neuron amplitude scatter,
distance decay, detection, per-electrode dead time, dead electrodes. Detection uses
`p = Phi((A - k*rms)/rms)` per (neuron, electrode) pair — analytically identical to
drawing Gaussian noise per spike per electrode, but one evaluation per pair instead of
~90 million draws per 300 s run.

Remaining Task 2: the 60-vs-1024-electrode fingerprint comparison figure. CL H5
round-trip for 60 electrodes is done; HD-MEA is blocked by `cl-sdk==1.0.0` uint8
channel ids (see `CL_API_PROBE.md`).
