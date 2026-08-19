"""Task 6 status file and README sync."""

from __future__ import annotations

from pathlib import Path

from culturesim.fit.task_status import (
    Task6Status,
    status_table_cell,
    sync_readme_task6,
    write_status,
)


def test_readme_sync_rewrites_the_task6_cell(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "| Task | What it covers | State |\n|---|---|---|\n| 6 | SBI posterior | Not started |\n",
        encoding="utf-8",
    )
    status = Task6Status(
        state="running",
        n_simulations_target=3000,
        n_attempted=120,
        n_kept=100,
        n_excluded=20,
    )
    write_status(status, tmp_path / "status.json")
    cell = sync_readme_task6(status, readme=readme, status_path=tmp_path / "status.json")
    assert "Running" in cell
    assert "120/3000" in readme.read_text(encoding="utf-8")


def test_done_cell_lists_identifiability() -> None:
    status = Task6Status(
        state="done",
        n_kept=3000,
        summary={
            "n_simulations": 2900,
            "identified": ["w_e", "tau_rec"],
            "unidentified": ["b"],
        },
    )
    cell = status_table_cell(status)
    assert "Done" in cell
    assert "w_e" in cell
    assert "b" in cell
