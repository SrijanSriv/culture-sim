"""SpikeRecording <-> Cortical Labs recording H5 adapter (SPEC §6.0.1)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..stats.spiketrains import SpikeRecording

CL_SDK_VERSION = "1.0.0"
DEFAULT_SAMPLING_RATE_HZ = 25_000
DEFAULT_UV_PER_SAMPLE_UNIT = 1.0
CL_WAVEFORM_SAMPLES = 75

CL_COMMON_CHANNEL_COUNT = 64
CL_COMMON_GROUND_CHANNELS = (0, 7, 56, 63)
CL_COMMON_REFERENCE_CHANNELS = (4,)

_CULTURE_SIM_ATTR_PREFIX = "culture_sim_"

__all__ = [
    "CL_COMMON_CHANNEL_COUNT",
    "CL_COMMON_GROUND_CHANNELS",
    "CL_COMMON_REFERENCE_CHANNELS",
    "DEFAULT_SAMPLING_RATE_HZ",
    "DEFAULT_UV_PER_SAMPLE_UNIT",
    "is_cl_recording",
    "to_cl_h5",
    "from_cl_h5",
    "cl_channel_mapping",
]


def cl_channel_mapping(n_channels: int) -> np.ndarray:
    """Map culture-sim channels to CL recording channels.

    CL criticality/connectivity in ``cl-sdk==1.0.0`` require the 64-channel common
    layout and reject spikes on ground channels. The 60-electrode layout therefore
    embeds into the non-ground channels of that layout.
    """
    if n_channels == 60:
        return np.asarray(
            [ch for ch in range(CL_COMMON_CHANNEL_COUNT) if ch not in CL_COMMON_GROUND_CHANNELS],
            dtype=np.int32,
        )
    if n_channels < 1:
        raise ValueError(f"n_channels must be positive, got {n_channels}")
    return np.arange(n_channels, dtype=np.int32)


def _cl_channel_count(mapping: np.ndarray) -> int:
    if mapping.size == 60 and set(mapping.tolist()).isdisjoint(CL_COMMON_GROUND_CHANNELS):
        return CL_COMMON_CHANNEL_COUNT
    return int(mapping.max()) + 1 if mapping.size else 0


def _sampling_rate(recording: SpikeRecording) -> int:
    raw = recording.metadata.get("sampling_rate_hz", DEFAULT_SAMPLING_RATE_HZ)
    rate = int(round(float(raw)))
    if rate <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {raw!r}")
    return rate


def _duration_frames(recording: SpikeRecording, sampling_rate_hz: int) -> int:
    return int(np.ceil(recording.duration * sampling_rate_hz))


def _json_attr(attrs: Mapping[str, Any], key: str, default: Any) -> Any:
    raw = attrs.get(key)
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _string_attr(attrs: Mapping[str, Any], key: str, default: str) -> str:
    raw = attrs.get(key, default)
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def is_cl_recording(path: str | Path) -> bool:
    """Return True when the path looks like a CL recording H5 file."""
    import h5py

    try:
        with h5py.File(Path(path), "r") as handle:
            return "spikes" in handle and "channel_count" in handle.attrs
    except OSError:
        return False


def to_cl_h5(
    recording: SpikeRecording,
    path: str | Path,
    *,
    uV_per_sample_unit: float | None = None,
) -> Path:
    """Write ``recording`` in the CL SDK recording H5 format."""
    import cl
    from cl._sim._recording_writer import RecordingWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sampling_rate_hz = _sampling_rate(recording)
    duration_frames = _duration_frames(recording, sampling_rate_hz)
    mapping = cl_channel_mapping(recording.n_channels)
    cl_n_channels = _cl_channel_count(mapping)
    if cl_n_channels > 255:
        raise ValueError(
            "cl-sdk==1.0.0 stores spike channels as uint8, so recordings with more "
            f"than 255 CL channels cannot be written losslessly (got {cl_n_channels})"
        )

    cl_channels = (
        mapping[recording.channels] if recording.n_spikes else np.array([], dtype=np.int32)
    )
    frame_indices = np.rint(recording.times * sampling_rate_hz).astype(np.int64)
    frame_indices = np.clip(frame_indices, 0, duration_frames)
    order = np.lexsort((cl_channels, frame_indices))
    cl_channels = cl_channels[order]
    frame_indices = frame_indices[order]

    uv_scale = float(
        DEFAULT_UV_PER_SAMPLE_UNIT if uV_per_sample_unit is None else uV_per_sample_unit
    )
    waveform = _default_waveform()
    attrs: dict[str, Any] = {
        "channel_count": cl_n_channels,
        "frames_per_second": sampling_rate_hz,
        "sampling_frequency": sampling_rate_hz,
        "duration_frames": duration_frames,
        "duration_seconds": float(recording.duration),
        "recording_start_timestamp": 0,
        "recording_stop_timestamp": duration_frames,
        "start_timestamp": 0,
        "uV_per_sample_unit": uv_scale,
        "app_info": {"name": "culture-sim", "cl_sdk_version": CL_SDK_VERSION},
        f"{_CULTURE_SIM_ATTR_PREFIX}format": "cl-recording-h5",
        f"{_CULTURE_SIM_ATTR_PREFIX}schema_version": 1,
        f"{_CULTURE_SIM_ATTR_PREFIX}source": recording.source,
        f"{_CULTURE_SIM_ATTR_PREFIX}original_n_channels": recording.n_channels,
        f"{_CULTURE_SIM_ATTR_PREFIX}metadata_json": json.dumps(recording.metadata, sort_keys=True),
        f"{_CULTURE_SIM_ATTR_PREFIX}channel_map_json": json.dumps(mapping.tolist()),
    }

    writer = RecordingWriter(
        path,
        channel_count=cl_n_channels,
        start_timestamp=0,
        include_spikes=True,
        include_stims=True,
        include_raw_samples=False,
        include_data_streams=False,
        initial_attributes=attrs,
    )
    writer.start()
    spikes = [
        cl.Spike(
            timestamp=int(timestamp),
            channel=int(channel),
            channel_mean_sample=float(waveform.mean()),
            samples=waveform,
        )
        for timestamp, channel in zip(frame_indices, cl_channels, strict=True)
    ]
    if spikes:
        writer.write_spikes(spikes)
    writer.update_attributes(attrs)
    writer.stop()
    _write_identity_sidecar(path, recording)
    return path


def _default_waveform() -> np.ndarray:
    return np.linspace(-10.0, -80.0, CL_WAVEFORM_SAMPLES, dtype=np.float32)


def _write_identity_sidecar(path: Path, recording: SpikeRecording) -> None:
    import h5py

    with h5py.File(path, "a") as handle:
        group = handle.create_group("culture_sim_spike_recording")
        group.attrs["schema_version"] = 1
        group.attrs["n_channels"] = recording.n_channels
        group.attrs["duration"] = recording.duration
        group.attrs["source"] = recording.source
        group.attrs["metadata_json"] = json.dumps(recording.metadata, sort_keys=True)
        group.create_dataset("times", data=recording.times, dtype="f8")
        group.create_dataset("channels", data=recording.channels, dtype="i4")


def from_cl_h5(path: str | Path) -> SpikeRecording:
    """Read a CL recording H5 file into ``SpikeRecording``."""
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        attrs = handle.attrs
        if "spikes" not in handle:
            raise ValueError(f"{path} is missing the CL /spikes table")
        if "culture_sim_spike_recording" in handle:
            group = handle["culture_sim_spike_recording"]
            source = group.attrs["source"]
            return SpikeRecording(
                times=group["times"][:],
                channels=group["channels"][:],
                n_channels=int(group.attrs["n_channels"]),
                duration=float(group.attrs["duration"]),
                source=source.decode() if isinstance(source, bytes) else str(source),
                metadata=json.loads(group.attrs["metadata_json"]),
            )
        sampling_rate_hz = float(attrs.get("sampling_frequency", attrs["frames_per_second"]))
        duration = float(
            attrs.get(
                "duration_seconds",
                float(attrs["duration_frames"]) / sampling_rate_hz,
            )
        )
        metadata = _json_attr(attrs, f"{_CULTURE_SIM_ATTR_PREFIX}metadata_json", {})
        source = _string_attr(attrs, f"{_CULTURE_SIM_ATTR_PREFIX}source", "cl-recording")
        mapping = np.asarray(
            _json_attr(
                attrs,
                f"{_CULTURE_SIM_ATTR_PREFIX}channel_map_json",
                list(range(int(attrs["channel_count"]))),
            ),
            dtype=np.int32,
        )
        n_channels = int(attrs.get(f"{_CULTURE_SIM_ATTR_PREFIX}original_n_channels", mapping.size))
        reverse = {int(cl_ch): int(local_ch) for local_ch, cl_ch in enumerate(mapping)}

        spikes = handle["spikes"][:]
        if spikes.size:
            local_channels: list[int] = []
            times: list[float] = []
            for row in spikes:
                cl_channel = int(row["channel"])
                local_channel = reverse.get(cl_channel)
                if local_channel is None:
                    continue
                local_channels.append(local_channel)
                times.append(float(row["timestamp"]) / sampling_rate_hz)
            channels_arr = np.asarray(local_channels, dtype=np.int32)
            times_arr = np.asarray(times, dtype=np.float64)
            order = np.argsort(times_arr, kind="stable")
            times_arr = times_arr[order]
            channels_arr = channels_arr[order]
        else:
            times_arr = np.array([], dtype=np.float64)
            channels_arr = np.array([], dtype=np.int32)

    return SpikeRecording(
        times=times_arr,
        channels=channels_arr,
        n_channels=n_channels,
        duration=duration,
        source=source,
        metadata=metadata,
    )
