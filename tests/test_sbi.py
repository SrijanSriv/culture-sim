"""SBI helpers that do not need Brian2 (SPEC §8.3)."""

from __future__ import annotations

import numpy as np

from culturesim.fit.sbi_fit import summarise_posterior
from culturesim.model.params import FREE_PARAM_NAMES, PriorBox


def test_summarise_posterior_marks_tight_marginals_identified() -> None:
    prior = PriorBox.default()
    rng = np.random.default_rng(0)
    # Samples tightly around the prior mid-point for w_e (index 1); flat elsewhere.
    samples = rng.uniform(prior.low, prior.high, size=(2000, len(FREE_PARAM_NAMES)))
    samples[:, 1] = 1.5 + 0.01 * rng.normal(size=2000)
    summary = summarise_posterior(samples, prior, std_ratio_threshold=0.5)
    assert summary.identified["w_e"] is True
    assert "w_e" in summary.identified_names()
