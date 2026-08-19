"""Fitting: distance, coarse search, and SBI (SPEC §8)."""

from __future__ import annotations

from .coarse import CoarseFitResult, coarse_fit, grid_search
from .distance import ScaleReference, distance
from .sbi_fit import PosteriorSummary, SBIResult, run_sbi_fit, train_posterior
from .task_status import Task6Status, read_status, sync_readme_task6

__all__ = [
    "CoarseFitResult",
    "PosteriorSummary",
    "SBIResult",
    "ScaleReference",
    "Task6Status",
    "coarse_fit",
    "distance",
    "grid_search",
    "read_status",
    "run_sbi_fit",
    "sync_readme_task6",
    "train_posterior",
]
