# reports/

Timestamped HTML snapshots from `culture-sim report`. Each run archives a new
file and refreshes `latest.html`.

## Layout

| Path | Role |
|---|---|
| `YYYY-MM-DDTHHMMSSZ_<label>.html` | Immutable snapshot for that run |
| `latest.html` | Copy of the most recent snapshot (always overwritten) |
| `history.jsonl` | One JSON object per archived report (path, commit, label, time) |
| `README.md` | This file (tracked in git) |

HTML snapshots are gitignored — they embed figures and can be large. The history
log is also gitignored. Keep this README so the folder exists in a clean
checkout.

## Commands

```bash
# Archive with auto timestamp + label "report"
.venv/bin/culture-sim report

# Named milestone
.venv/bin/culture-sim report --label first-fit-task8

# Explicit path (still works; skips history/latest unless under reports/)
.venv/bin/culture-sim report --out reports/special.html
```

Open the newest one with `open reports/latest.html`.
