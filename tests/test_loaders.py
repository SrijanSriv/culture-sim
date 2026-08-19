"""Tests for real-dataset loaders (SPEC §7, Task 4)."""

from __future__ import annotations

import bz2
from pathlib import Path

import numpy as np
import pytest

from culturesim.data.loaders import (
    WAGENAAR_N_ELECTRODES,
    load_wagenaar,
    parse_wagenaar_filename,
)


def _write_spk(path: Path, rows: str) -> Path:
    path.write_bytes(bz2.compress(rows.encode("ascii")))
    return path


def test_filename_recovers_plating_culture_and_div() -> None:
    ids = parse_wagenaar_filename("simple-text/daily/spont/dense/2-4-21.spk.txt.bz2")
    assert ids == {"plating": 2, "culture": 4, "div": 21}


def test_filename_rejects_an_unparseable_name() -> None:
    with pytest.raises(ValueError, match="cannot parse plating-culture-div"):
        parse_wagenaar_filename("spikes.txt")


def test_load_wagenaar_reads_text_bz2_and_splits_the_stim_channel(tmp_path: Path) -> None:
    path = _write_spk(
        tmp_path / "1-1-14.spk.txt.bz2",
        "0.10 0\n0.20 59\n1.50 60\n0.15 0\n",
    )
    recording = load_wagenaar(path, duration_s=2.0)
    assert recording.source == "wagenaar2006"
    assert recording.n_channels == WAGENAAR_N_ELECTRODES
    assert recording.duration == pytest.approx(2.0)
    assert recording.metadata["div"] == 14
    assert recording.metadata["plating"] == 1
    assert recording.metadata["culture"] == 1
    np.testing.assert_allclose(recording.times, [0.10, 0.15, 0.20])
    np.testing.assert_array_equal(recording.channels, [0, 0, 59])
    assert recording.metadata["n_stimulus_events"] == 1
    assert recording.metadata["stimulus_times_s"] == pytest.approx([1.50])
    assert 60 not in recording.channels.tolist()


def test_load_wagenaar_uses_the_nominal_half_hour_when_spikes_end_early(tmp_path: Path) -> None:
    path = _write_spk(tmp_path / "3-2-8.spk.txt.bz2", "10.0 1\n20.0 2\n")
    recording = load_wagenaar(path)
    assert recording.duration == pytest.approx(1800.0)
    assert recording.metadata["div"] == 8


def test_load_wagenaar_does_not_truncate_a_file_longer_than_30_min(tmp_path: Path) -> None:
    path = _write_spk(tmp_path / "1-1-14.spk.txt.bz2", "10.0 1\n2716.0 2\n")
    recording = load_wagenaar(path)
    assert recording.duration == pytest.approx(2716.0)


def test_load_wagenaar_without_a_cached_file_explains_how_to_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_wagenaar"):
        load_wagenaar()


@pytest.mark.requires_data
def test_default_wagenaar_recording_loads_from_the_verified_archive() -> None:
    """Needs data/raw/wagenaar2006/1-1-14.spk.txt.bz2 (gitignored)."""
    from culturesim.data.loaders import wagenaar_cache_path

    if not wagenaar_cache_path().exists():
        pytest.skip("Wagenaar cache not present")
    recording = load_wagenaar()
    assert recording.n_channels == 60
    assert recording.n_spikes > 1000
    assert recording.metadata["div"] == 14
    assert recording.duration >= recording.times[-1]
