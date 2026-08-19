"""Real dataset loaders (SPEC §7).

Task 4. One loader per dataset, each returning a
:class:`~culturesim.stats.spiketrains.SpikeRecording`.

**Access must be verified before any loader is written.** SPEC §7 is explicit: confirm
the accession or URL actually resolves and read the data-availability statement. A
paper saying data is public is not evidence that it is. If a target dataset turns out
to be gated, report that and stop -- do not substitute a different dataset, because a
different array geometry silently invalidates every comparison the observation model
exists to make (SPEC §14).

``DATASETS`` below records the verification state of each target. It is filled in from
actual checks, not from what the papers claim.

Threshold crossings are preferred over spike-sorted output: the virtual MEA models
threshold detection, so sorted units would be compared against something the
observation model does not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..stats.spiketrains import SpikeRecording

__all__ = ["DatasetInfo", "DATASETS", "load_wagenaar", "load_dandi_nwb", "available_datasets"]

AccessState = Literal["unverified", "public", "gated", "unavailable"]


@dataclass(frozen=True)
class DatasetInfo:
    """A target dataset and the current state of our access to it."""

    key: str
    description: str
    url: str
    access: AccessState = "unverified"
    verified_on: str | None = None
    notes: str = ""
    spike_kind: Literal["threshold", "sorted", "unknown"] = "unknown"


DATASETS: dict[str, DatasetInfo] = {
    # Primary target (SPEC §7): dense longitudinal recordings of dissociated rat
    # cortical cultures across the first weeks in vitro. The longitudinal structure
    # is what the downstream decoder-drift work needs.
    "wagenaar2006": DatasetInfo(
        key="wagenaar2006",
        description=(
            "Wagenaar, Pine & Potter 2006, BMC Neurosci 7:11 -- developmental MEA "
            "recordings of dissociated rat cortical cultures, MCS 60-electrode arrays"
        ),
        url="https://potterlab.gatech.edu/potter-lab-data-code-and-designs/",
        access="unverified",
        notes="Access not yet checked. Do not write the loader before checking (SPEC §7).",
        spike_kind="threshold",
    ),
    # Secondary target: a 3-D comparison point, NWB format via pynwb.
    "braingeneers_organoid": DatasetInfo(
        key="braingeneers_organoid",
        description="Braingeneers organoid HD-MEA recordings, DANDI archive, NWB format",
        url="https://dandiarchive.org/",
        access="unverified",
        notes=(
            "Dandiset ID not yet pinned. This is a 3-D organoid dataset and the model "
            "is 2-D (SPEC §0) -- it is a comparison point, never a fitting target."
        ),
        spike_kind="unknown",
    ),
}


def available_datasets() -> dict[str, DatasetInfo]:
    """Datasets confirmed public. Empty until access has actually been verified."""
    return {k: v for k, v in DATASETS.items() if v.access == "public"}


def load_wagenaar(path: str | Path, **kwargs: Any) -> SpikeRecording:
    """Load one Wagenaar/Potter developmental recording (SPEC §7).

    Must handle: threshold-crossing vs spike-sorted data, differing sampling rates,
    and electrode geometry metadata -- and must record DIV and culture id in
    ``metadata``, since cross-culture validation (SPEC §9.2) keys off them.
    """
    raise NotImplementedError("Task 4 (SPEC §7) -- verify dataset access first")


def load_dandi_nwb(path: str | Path, **kwargs: Any) -> SpikeRecording:
    """Load an NWB-format recording from the DANDI archive via ``pynwb`` (SPEC §7)."""
    raise NotImplementedError("Task 4 (SPEC §7) -- verify dataset access first")
