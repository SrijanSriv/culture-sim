# Re-fitting the SBI posterior (Task 6) on a bigger machine

The first overnight run **completed** but produced a weakly identified posterior
(only `rate_bg` clear; PPC covered ~50% of fingerprint stats). This note is for a
later re-fit on Colab, a workstation, or a cloud VM — not a requirement to pass
Tasks 7–8.

## What already exists (do not throw away)

| Path | Role |
|---|---|
| `output/posterior.checkpoint.npz` | 3000 prior draws → kept fingerprints (resume sims) |
| `output/posterior.pkl` | Current (weak) posterior |
| `output/posterior.summary.json` | Identified / unidentified + PPC summary |
| `output/task6_status.json` | Run status for README sync |
| `configs/fit_sbi.yaml` | Duration, n_sims, **MCMC** sampling |

## Why it was weak

1. **60 s sims vs ~2716 s real recording** — IBI / burst structure cannot match.
2. **Wide 8-D prior + only ~2.4k kept sims** — density estimate underfills the box.
3. **Default rejection sampling hung** at 0/10000 — fixed: use `sample_with: mcmc`.
4. **Near-constant fingerprint bins** across sims — bad features for the embedding.

GPU helps **torch training** a bit; it does **not** speed Brian2 `cpp_standalone`
much. Prefer a machine with **many CPU cores** and a working C++ compiler.

## Faster / better re-fit recipe

```bash
# On the big machine, after clone + .venv + pip install -e ".[dev]"
.venv/bin/python scripts/fetch_wagenaar.py --scale

# Edit configs/fit_sbi.yaml:
#   simulator.duration_s: 180   # or 300
#   inference.n_simulations: 5000
#   posterior.sample_with: mcmc
#   posterior.n_samples: 2000

# Optional: copy checkpoint from this laptop to skip re-simulating from scratch
# (only valid if duration_s / fingerprint version match).

.venv/bin/culture-sim fit sbi \
  --data wagenaar2006 \
  --n-sims 5000 \
  --out output/posterior.pkl \
  --detach

.venv/bin/python scripts/check_task6.py   # exit 0 when done
```

Tighter prior (often better than more sims alone): shrink `priors:` in
`configs/model_default.yaml` around the Task 5 coarse point
(`output/coarse.json` → `best`), then re-run SBI.

## Colab-specific notes

- Install a C++ toolchain before Brian2 standalone compiles.
- `n_workers` ≈ number of CPUs; avoid oversubscription.
- Persist `output/` to Drive so detach/resume survives runtime disconnects.
- Do not expect GPU to cut wall-clock by 10× for this workload.
