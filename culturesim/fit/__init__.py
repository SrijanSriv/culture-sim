"""Fitting: distance, coarse search, and SBI (SPEC §8)."""

from __future__ import annotations

from .coarse import CoarseFitResult, coarse_fit, grid_search
from .distance import ScaleReference, distance
from .sbi_fit import PosteriorSummary, SBIResult, train_posterior

__all__ = [
    "CoarseFitResult",
    "PosteriorSummary",
    "SBIResult",
    "ScaleReference",
    "coarse_fit",
    "distance",
    "grid_search",
    "train_posterior",
]
