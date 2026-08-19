"""Task-status artefacts for long overnight jobs (Task 6+).

Agents (and humans) should not babysit SBI. A run writes ``output/task6_status.json``
continuously; on finish it rewrites the Task 6 row in ``README.md``. Check progress
with ``.venv/bin/python scripts/check_task6.py``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from ..config import REPO_ROOT

__all__ = [
    "DEFAULT_STATUS_PATH",
    "Task6Status",
    "read_status",
    "write_status",
    "sync_readme_task6",
    "status_table_cell",
    "mark_running",
]

DEFAULT_STATUS_PATH = REPO_ROOT / "output" / "task6_status.json"
State = Literal["not_started", "running", "done", "failed"]


@dataclass
class Task6Status:
    """Serializable progress for the Task 6 SBI campaign."""

    state: State = "not_started"
    n_simulations_target: int = 0
    n_attempted: int = 0
    n_kept: int = 0
    n_excluded: int = 0
    batch_size: int = 0
    duration_s: float = 0.0
    started_at: str | None = None
    updated_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    out: str | None = None
    log: str | None = None
    checkpoint: str | None = None
    message: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Task6Status:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_status(path: str | Path = DEFAULT_STATUS_PATH) -> Task6Status:
    path = Path(path)
    if not path.exists():
        return Task6Status()
    return Task6Status.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_status(status: Task6Status, path: str | Path = DEFAULT_STATUS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    status.updated_at = _now()
    if status.state == "running" and status.started_at is None:
        status.started_at = status.updated_at
    if status.state in {"done", "failed"} and status.finished_at is None:
        status.finished_at = status.updated_at
    staging = path.with_suffix(path.suffix + ".partial")
    staging.write_text(json.dumps(status.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    staging.replace(path)
    return path


def status_table_cell(status: Task6Status) -> str:
    """One README table cell for the Task 6 State column."""
    if status.state == "not_started":
        return "Not started"
    if status.state == "running":
        return (
            f"**Running** ({status.n_attempted}/{status.n_simulations_target} sims; "
            f"kept {status.n_kept}, excluded {status.n_excluded}) — "
            f"see `output/task6_status.json`"
        )
    if status.state == "failed":
        detail = status.message or "see log"
        return f"**Failed** ({detail})"
    identified = status.summary.get("identified") or []
    unidentified = status.summary.get("unidentified") or []
    n = status.summary.get("n_simulations", status.n_kept)
    return (
        f"**Done** ({n} sims; identified: {', '.join(identified) or 'none'}; "
        f"unidentified: {', '.join(unidentified) or 'none'})"
    )


_TASK6_ROW = re.compile(
    r"^\| 6 \| SBI posterior \| .* \|$",
    re.MULTILINE,
)


def sync_readme_task6(
    status: Task6Status | None = None,
    *,
    readme: str | Path | None = None,
    status_path: str | Path = DEFAULT_STATUS_PATH,
) -> str:
    """Rewrite the Task 6 State cell from the status artefact. Returns the new cell."""
    status = status if status is not None else read_status(status_path)
    cell = status_table_cell(status)
    path = Path(readme) if readme is not None else REPO_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    replacement = f"| 6 | SBI posterior | {cell} |"
    if not _TASK6_ROW.search(text):
        raise ValueError(f"could not find Task 6 row in {path}")
    path.write_text(_TASK6_ROW.sub(replacement, text, count=1), encoding="utf-8")
    return cell


def mark_running(
    *,
    n_simulations: int,
    duration_s: float,
    batch_size: int,
    out: str | Path,
    log: str | Path | None = None,
    checkpoint: str | Path | None = None,
    message: str = "",
    path: str | Path = DEFAULT_STATUS_PATH,
) -> Task6Status:
    previous = read_status(path)
    status = Task6Status(
        state="running",
        n_simulations_target=int(n_simulations),
        n_attempted=previous.n_attempted if previous.state == "running" else 0,
        n_kept=previous.n_kept if previous.state == "running" else 0,
        n_excluded=previous.n_excluded if previous.state == "running" else 0,
        batch_size=int(batch_size),
        duration_s=float(duration_s),
        started_at=previous.started_at if previous.state == "running" else _now(),
        pid=os.getpid(),
        out=str(out),
        log=None if log is None else str(log),
        checkpoint=None if checkpoint is None else str(checkpoint),
        message=message,
        summary=previous.summary if previous.state == "running" else {},
    )
    write_status(status, path)
    sync_readme_task6(status, status_path=path)
    return status
