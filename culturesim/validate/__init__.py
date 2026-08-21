"""Validation suite (SPEC §9).

Three tests, increasing in strength. All three are reported regardless of outcome.
"""

from __future__ import annotations

from .cross_culture import CrossCultureResult, run_cross_culture
from .heldout import HeldoutResult, run_heldout
from .perturbation import PerturbationResult, run_perturbation
from .suite import ValidationReport, run_validation, update_readme_validation

__all__ = [
    "CrossCultureResult",
    "HeldoutResult",
    "PerturbationResult",
    "ValidationReport",
    "run_cross_culture",
    "run_heldout",
    "run_perturbation",
    "run_validation",
    "update_readme_validation",
]

# Names accepted by `culture-sim validate --test`.
TESTS = ("heldout", "cross_culture", "perturbation")
