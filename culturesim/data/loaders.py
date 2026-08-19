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
actual checks, not from what the papers claim. Both spec targets were verified public on
2026-08-19 by downloading real bytes, not by loading a landing page.

``VERIFIED_ALTERNATIVES`` holds datasets checked at the same time that are *not* spec
targets. They are recorded to preserve the evidence, not to be swapped in silently.

Threshold crossings are preferred over spike-sorted output: the virtual MEA models
threshold detection, so sorted units would be compared against something the
observation model does not reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..stats.spiketrains import SpikeRecording

__all__ = [
    "DatasetInfo",
    "DATASETS",
    "VERIFIED_ALTERNATIVES",
    "EMPTY_DANDISETS",
    "load_wagenaar",
    "load_dandi_nwb",
    "available_datasets",
]

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
        url="https://neurodatasharing.bme.gatech.edu/development-data/html/index.html",
        access="public",
        verified_on="2026-08-19",
        notes=(
            "Verified by downloading simple-text/daily/spont/dense/1-1-14.spk.txt.bz2 "
            "(874,682 bytes, 224,547 events) with no registration, cookie or referrer "
            "check. Directory indexes are enabled, so bulk retrieval works. Condition of "
            "use is a citation requirement; bundled code is GPL v2. "
            "Note the trap in the other direction: the paper body says to email Steve "
            "Potter for access and the old potterlab.gatech.edu host is dead, so the "
            "publication reads as gated while the archive is in fact open. "
            "Threshold crossings, which SPEC §7 prefers -- see load_wagenaar for the schema."
        ),
        spike_kind="threshold",
    ),
    # Secondary target: a 3-D comparison point, NWB format via pynwb.
    "braingeneers_organoid": DatasetInfo(
        key="braingeneers_organoid",
        description=(
            "van der Molen et al., 'Preconfigured neuronal firing sequences in human "
            "brain organoids', DANDI 001603 -- HD-MEA, NWB format"
        ),
        url="https://dandiarchive.org/dandiset/001603",
        access="public",
        verified_on="2026-08-19",
        notes=(
            "embargo_status OPEN, 111 assets / 322 GB, CC-BY-4.0. Anonymous byte-range "
            "download confirmed against an asset (presigned S3, valid HDF5 magic) with no "
            "API token. Carries both Units (sorted) and ElectricalSeries (raw) across 36 "
            "subjects. "
            "Two cautions: it exists as a DRAFT version only, with no immutable published "
            "version, so contents can change underneath a fitted result -- pin asset "
            "checksums in the manifest. And this is a 3-D organoid dataset while the model "
            "is 2-D (SPEC §0), so it is a comparison point, never a fitting target."
        ),
        spike_kind="sorted",
    ),
}

# Verified during the SPEC §7 access check but NOT spec targets. Recorded so the
# verification work is not lost and so a future gap is answered with evidence rather than
# another round of searching. SPEC §7 forbids silently substituting any of these for a
# spec target; using one is a documented decision, not a convenience.
VERIFIED_ALTERNATIVES: dict[str, DatasetInfo] = {
    "crcns_hc8": DatasetInfo(
        key="crcns_hc8",
        description=(
            "CRCNS hc-8 -- dissociated rat hippocampal cultures, DIV 6-35, MCS "
            "60-electrode 200 um lattice, 435 recordings of ~1 h, spike-sorted"
        ),
        url="https://download.crcns.org/hc-8/",
        access="gated",
        verified_on="2026-08-19",
        notes=(
            "URL returns 200 but the body is a username/password form. Registration is "
            "free yet is a manually-reviewed account request (real name, institution, "
            "position, statement of intent), so access is not instant. Scientifically the "
            "closest match to wagenaar2006 and it is sorted, which wagenaar2006 is not."
        ),
        spike_kind="sorted",
    ),
    "dandi_001611": DatasetInfo(
        key="dandi_001611",
        description=(
            "Mayama, Takahashi et al. -- dissociated rat cortical cultures on HD-MEAs "
            "under a repeated-stimulation protocol, 2700 NWB files / 39.9 GB"
        ),
        url="https://dandiarchive.org/dandiset/001611",
        access="public",
        verified_on="2026-08-19",
        notes=(
            "OPEN, CC-BY-4.0, and the only candidate found with an IMMUTABLE published "
            "version (0.260611.0634), which matters for reproducibility. Dissociated and "
            "2-D, so a closer structural match than the organoid data. The catch is in the "
            "title: repeated stimulation, not spontaneous activity, so the protocol is a "
            "confound for statistics that assume spontaneous dynamics."
        ),
        spike_kind="sorted",
    ),
    "gnode_kapucu": DatasetInfo(
        key="gnode_kapucu",
        description=(
            "Kapucu et al. (Tampere) comparative MEA dataset -- rat embryonic cortical "
            "and hPSC-derived networks on Axion multiwell plates, 12.5 kHz"
        ),
        url="https://gin.g-node.org/NeuroGroup_TUNI/Comparative_MEA_dataset",
        access="public",
        verified_on="2026-08-19",
        notes=(
            "Browses anonymously with no sign-in wall; CC-BY-4.0; raw HDF5 plus spike "
            "times plus analysis code. The full archive is 2.3 TiB, so fetch individual "
            "files rather than cloning. Axion plate geometry, not MCS."
        ),
        spike_kind="unknown",
    ),
}

# Registered on DANDI but empty (zero assets) as of 2026-08-19; two are owned by a contact
# named "Test, Manuel". They rank well in search and waste time, so they are named here.
EMPTY_DANDISETS: tuple[str, ...] = ("000893", "001263", "001355", "000626")


def available_datasets() -> dict[str, DatasetInfo]:
    """Datasets confirmed public. Empty until access has actually been verified."""
    return {k: v for k, v in DATASETS.items() if v.access == "public"}


def load_wagenaar(path: str | Path, **kwargs: Any) -> SpikeRecording:
    """Load one Wagenaar/Potter developmental recording (SPEC §7).

    Access is verified (see ``DATASETS['wagenaar2006']``), so this is unblocked.

    Schema, read off the archive's own ``help.html`` and a downloaded file:

    * Path scheme
      ``{full,simple-text,simple-matlab}/{daily,night}/{spont,stim}/{dense,small,sparse,smsp,ulsp}/{plating}-{culture}-{div}.ext``.
      ``simple-text`` is the right entry point: one recording is under 1 MB and the whole
      compact set is ~4 GB, against ~45 GB for full waveforms.
    * ``.spk.txt.bz2`` -- two whitespace-separated columns, ``time_seconds channel``,
      channels 0-59. Threshold crossings, multi-unit, **not** sorted.
    * ``.spike.gz`` -- binary: 8-byte time, then int16 channel, height, width,
      ``context[74]``, threshold. Context spans -1 ms to +2 ms around the peak.
    * 25 kHz sampling (40 us period); amplitude LSB 0.3335 uV.
    * MCS 8x8 grid minus the four corners = 60 electrodes. **Channel 60 is a stimulus
      marker, not an electrode**, and must be split out rather than counted as a channel.
      The electrode-number to row/column map is published in ``help.html``.
    * Recording durations are documented as ~30 min but were not confirmed per file.
      ``night.spont.dense.text.html`` returns 404, so overnight recordings may only be
      reachable by browsing the directory index.

    Must record DIV and culture id in ``metadata``: cross-culture validation
    (SPEC §9.2) keys off them, and they are recoverable only from the filename.
    """
    raise NotImplementedError("Task 4 (SPEC §7) -- loader not yet written")


def load_dandi_nwb(path: str | Path, **kwargs: Any) -> SpikeRecording:
    """Load an NWB-format recording from the DANDI archive via ``pynwb`` (SPEC §7)."""
    raise NotImplementedError("Task 4 (SPEC §7) -- verify dataset access first")
