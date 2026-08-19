"""Tests for the fingerprint container and its frozen order (SPEC §3, §6.6)."""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.config import load_config
from culturesim.stats.fingerprint import Fingerprint, FingerprintSpec


@pytest.fixture
def spec() -> FingerprintSpec:
    return FingerprintSpec.load("fingerprint.yaml")


# -- spec expansion -------------------------------------------------------


def test_shipped_config_expands(spec: FingerprintSpec) -> None:
    assert len(spec) == len(spec.names) == spec.weights.size
    assert len(set(spec.names)) == len(spec.names)
    assert spec.groups == ("rates", "bursts", "avalanches", "branching", "connectivity")


def test_expansion_order_is_scalars_then_quantiles_then_histograms(spec: FingerprintSpec) -> None:
    burst_names = [n for n in spec.names if spec.group_of[n] == "bursts"]
    assert burst_names[0] == "burst_rate_per_min"
    quantile_start = burst_names.index("ibi_seconds_p10")
    histogram_start = burst_names.index("ibi_seconds_hist_00")
    assert quantile_start < histogram_start
    assert burst_names[-1] == "ibi_seconds_hist_11"


def test_group_mask_selects_one_group(spec: FingerprintSpec) -> None:
    mask = spec.group_mask("avalanches")
    assert mask.sum() == sum(1 for n in spec.names if spec.group_of[n] == "avalanches")
    selected = [n for n, m in zip(spec.names, mask, strict=True) if m]
    assert all(spec.group_of[n] == "avalanches" for n in selected)


def test_unknown_group_raises(spec: FingerprintSpec) -> None:
    with pytest.raises(KeyError, match="unknown fingerprint group"):
        spec.group_mask("does_not_exist")


def test_naive_branching_ratio_is_not_in_the_vector(spec: FingerprintSpec) -> None:
    """SPEC §6.4: the naive estimator is subsampling-biased and must stay out."""
    assert "branching_ratio_naive" not in spec.names
    assert "branching_ratio_mr" in spec.names


def test_histogram_edges_are_log_spaced(spec: FingerprintSpec) -> None:
    histogram = spec.histogram_for("ibi_seconds")
    assert histogram.n_bins == 12
    assert histogram.edges[0] == pytest.approx(0.1)
    ratios = histogram.edges[1:] / histogram.edges[:-1]
    np.testing.assert_allclose(ratios, ratios[0])


def test_duplicate_statistic_is_rejected() -> None:
    config = {"groups": [{"name": "a", "scalars": ["x", "x"]}]}
    with pytest.raises(ValueError, match="duplicate fingerprint statistic"):
        FingerprintSpec.from_config(config)


def test_empty_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="declares no statistics"):
        FingerprintSpec.from_config({"groups": []})


# -- the freeze mechanism (SPEC §3) ---------------------------------------


def test_freeze_check_passes_when_hash_matches() -> None:
    config = {
        "version": "1.0.0",
        "frozen": True,
        "groups": [{"name": "g", "scalars": ["a", "b"]}],
    }
    spec = FingerprintSpec.from_config(config)
    config["names_sha256"] = spec.names_sha256
    FingerprintSpec.from_config(config).check_freeze()  # must not raise


def test_freeze_check_catches_a_reordered_frozen_spec() -> None:
    """Editing a frozen fingerprint silently invalidates every prior fit."""
    config = {
        "version": "1.0.0",
        "frozen": True,
        "groups": [{"name": "g", "scalars": ["a", "b"]}],
    }
    original_hash = FingerprintSpec.from_config(config).names_sha256
    config["groups"] = [{"name": "g", "scalars": ["b", "a"]}]
    config["names_sha256"] = original_hash
    with pytest.raises(ValueError, match="order has changed"):
        FingerprintSpec.from_config(config).check_freeze()


def test_frozen_spec_without_a_hash_is_rejected() -> None:
    config = {"version": "1.0.0", "frozen": True, "groups": [{"name": "g", "scalars": ["a"]}]}
    with pytest.raises(ValueError, match="declares no names_sha256"):
        FingerprintSpec.from_config(config).check_freeze()


def test_shipped_config_is_not_yet_frozen() -> None:
    """Freezing happens at the end of Task 3, not before (SPEC §13)."""
    assert load_config("fingerprint.yaml")["frozen"] is False


# -- the Fingerprint container --------------------------------------------


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="1 values but 2 names"):
        Fingerprint(values=np.array([1.0]), names=("a", "b"))


def test_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        Fingerprint(values=np.array([1.0, 2.0]), names=("a", "a"))


def test_values_are_immutable() -> None:
    fingerprint = Fingerprint(values=np.array([1.0]), names=("a",))
    with pytest.raises(ValueError):
        fingerprint.values[0] = 2.0


def test_lookup_by_name() -> None:
    fingerprint = Fingerprint(values=np.array([1.5, 2.5]), names=("a", "b"))
    assert fingerprint["b"] == pytest.approx(2.5)
    with pytest.raises(KeyError):
        fingerprint["c"]


def test_json_round_trip_preserves_non_finite_sentinels(tmp_path) -> None:
    """NaN is the documented sentinel, so it has to survive serialisation."""
    original = Fingerprint(
        values=np.array([1.0, np.nan, np.inf, -np.inf]),
        names=("a", "b", "c", "d"),
        version="0.1.0-draft",
        metadata={"source": "test"},
    )
    restored = Fingerprint.read_json(original.write_json(tmp_path / "fp.json"))

    assert restored.names == original.names
    assert restored.version == original.version
    assert restored.metadata == original.metadata
    np.testing.assert_array_equal(restored.values, original.values)
    assert restored.n_undefined == 3


def test_require_match_rejects_a_reordered_vector(spec: FingerprintSpec) -> None:
    """Comparing differently-ordered vectors compares different statistics."""
    values = np.zeros(len(spec))
    good = Fingerprint(values=values, names=spec.names, version=spec.version)
    good.require_match(spec)

    reordered = Fingerprint(values=values, names=tuple(reversed(spec.names)), version=spec.version)
    with pytest.raises(ValueError, match="order does not match"):
        reordered.require_match(spec)


def test_require_match_rejects_a_version_mismatch(spec: FingerprintSpec) -> None:
    stale = Fingerprint(values=np.zeros(len(spec)), names=spec.names, version="0.0.1-old")
    with pytest.raises(ValueError, match="version mismatch"):
        stale.require_match(spec)


def test_empty_values_matches_the_spec_length(spec: FingerprintSpec) -> None:
    values = spec.empty_values()
    assert values.size == len(spec)
    assert np.all(np.isnan(values))


def test_fingerprint_refuses_neuron_level_recordings(spec: FingerprintSpec) -> None:
    """SPEC §5 / §14: statistics on 1000 neurons are not comparable to 60 electrodes."""
    from culturesim.stats.fingerprint import compute_fingerprint
    from culturesim.stats.spiketrains import SpikeRecording

    recording = SpikeRecording(
        times=np.array([0.1, 0.2]),
        channels=np.array([0, 1]),
        n_channels=1000,
        duration=1.0,
        source="simulation-neuron-level",
        metadata={"observation": "none"},
    )
    with pytest.raises(ValueError, match="electrode-level"):
        compute_fingerprint(recording, spec)
