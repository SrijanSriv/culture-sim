"""SPEC §0 must appear verbatim in the README.

The spec is explicit that this is the honest-limitations section of any resulting
paper, "written before there is any incentive to fudge it". The incentive to soften it
arrives later, so the check is automated now rather than left to review.
"""

from __future__ import annotations

import pytest

from culturesim.config import REPO_ROOT

SPEC_PATH = REPO_ROOT / "SPEC.md"
README_PATH = REPO_ROOT / "README.md"
SCOPE_HEADING = "## 0. Purpose and Scope"


def _spec_scope_section() -> str:
    """SPEC §0, from its heading up to the horizontal rule that closes it."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    start = text.index(SCOPE_HEADING)
    end = text.index("\n---\n", start)
    return text[start:end].strip()


def test_readme_reproduces_the_scope_statement_verbatim() -> None:
    scope = _spec_scope_section()
    readme = README_PATH.read_text(encoding="utf-8")

    missing = [line for line in scope.splitlines() if line.strip() and line.strip() not in readme]
    assert not missing, (
        "SPEC §0 must be reproduced verbatim in README.md. Missing lines:\n" + "\n".join(missing)
    )
    assert scope in readme, "SPEC §0 must appear as one contiguous block, not scattered"


def test_readme_keeps_the_non_goals() -> None:
    """The list of things the model does not do is the part most likely to erode."""
    readme = README_PATH.read_text(encoding="utf-8")
    for non_goal in (
        "Dendritic computation",
        "Glia, astrocytes, or metabolic state",
        "Neurotransmitter diffusion",
        "Developmental changes in cell count",
        "Spatial wave propagation",
        "Ion channel dynamics below the LIF abstraction",
        "3-D organoid geometry",
    ):
        assert non_goal in readme, f"non-goal dropped from the README: {non_goal}"


def test_readme_states_evoked_response_is_not_fitted() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    assert "Evoked response to electrical stimulation (validation only, not fitted)" in readme


@pytest.mark.parametrize("task", ["Task 0", "Task 1", "Task 3", "Task 6", "Task 7"])
def test_readme_reports_build_status(task: str) -> None:
    """A reader must be able to tell what is actually done from the README alone."""
    readme = README_PATH.read_text(encoding="utf-8")
    number = task.split()[1]
    assert f"| {number} |" in readme or task in readme
