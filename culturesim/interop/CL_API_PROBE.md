# CL API Probe

Task 0.5 probe run on Python 3.13.3 with `cl-sdk==1.0.0`.

## Install and License

`cl-sdk==1.0.0` installs without CL1 hardware in the project virtualenv.

Package metadata reports license `CC BY-NC 4.0`. This permits non-commercial use with attribution, but it is not an OSI open-source license and does not permit commercial use without separate permission from Cortical Labs. Treat that as a release/legal constraint before distributing any fitted artifact or package that depends on `cl-sdk`.

## Import Surface

```python
import cl
from cl.util.recording_view import RecordingView
```

Useful public methods are on `RecordingView`, not top-level functions in `cl.analysis`:

```python
RecordingView(path).analyse_network_bursts(
    bin_size_sec: float,
    onset_freq_hz: float,
    offset_freq_hz: float,
    min_active_channels: int | None = None,
)

RecordingView(path).analyse_criticality(
    bin_size_sec: float,
    percentile_threshold: float,
    max_lags_branching_ratio: int = 40,
    duration_thresholds: tuple[int, int] = (2, 5),
    min_spike_count_threshold: int = 10,
    n_bootstraps: int = 100,
    random_seed: int = 42,
)

RecordingView(path).analyse_functional_connectivity(
    bin_size_sec: float,
    correlation_threshold: float = 0.6,
)
```

`cl.analysis` exports result models such as `AnalysisResultNetworkBursts`, `AnalysisResultCriticality`, and `AnalysisResultsFunctionalConnectivity`.

## Recording H5 Schema

The SDK recording writer creates a PyTables H5 file with root tables:

- `/spikes`: table with columns `timestamp: int64`, `channel: uint8`, `samples: float32[75]`
- `/stims`: table with columns `timestamp: int64`, `channel: uint8`

Required root attributes for analysis:

- `channel_count`
- `frames_per_second`
- `sampling_frequency`
- `duration_frames`
- `duration_seconds`
- `uV_per_sample_unit`

The adapter writes additional `culture_sim_*` attributes for original channel count, channel mapping, source, and JSON metadata.

## Runnable Poisson Smoke

This is the minimal external-spike path that succeeded:

```python
import tempfile
from pathlib import Path

import cl
import numpy as np
from cl._sim._recording_writer import RecordingWriter
from cl.util.recording_view import RecordingView

path = Path(tempfile.mkdtemp()) / "probe64_valid.h5"
fps = 25_000
duration_frames = 250_000
ground = {0, 7, 56, 63}
attrs = {
    "channel_count": 64,
    "frames_per_second": fps,
    "sampling_frequency": fps,
    "duration_frames": duration_frames,
    "duration_seconds": duration_frames / fps,
    "recording_start_timestamp": 0,
    "recording_stop_timestamp": duration_frames,
    "uV_per_sample_unit": 1.0,
}

writer = RecordingWriter(
    path,
    channel_count=64,
    start_timestamp=0,
    include_spikes=True,
    include_stims=True,
    include_raw_samples=False,
    include_data_streams=False,
    initial_attributes=attrs,
)
writer.start()
rng = np.random.default_rng(0)
wave = np.linspace(-10, -80, 75, dtype=np.float32)
spikes = []
for channel in (ch for ch in range(64) if ch not in ground):
    for seconds in np.sort(rng.uniform(0.5, 9.5, size=10)):
        spikes.append(
            cl.Spike(
                timestamp=int(round(seconds * fps)),
                channel=channel,
                channel_mean_sample=float(wave.mean()),
                samples=wave,
            )
        )
spikes.sort(key=lambda spike: spike.timestamp)
writer.write_spikes(spikes)
writer.update_attributes(attrs)
writer.stop()

with RecordingView(str(path)) as recording:
    bursts = recording.analyse_network_bursts(
        bin_size_sec=0.05,
        onset_freq_hz=3.0,
        offset_freq_hz=1.0,
        min_active_channels=3,
    )
    criticality = recording.analyse_criticality(
        bin_size_sec=0.05,
        percentile_threshold=0.95,
        n_bootstraps=2,
        random_seed=1,
    )
    connectivity = recording.analyse_functional_connectivity(
        bin_size_sec=0.05,
        correlation_threshold=0.1,
    )
```

Observed result:

- `analyse_network_bursts` returned `AnalysisResultNetworkBursts` with burst boundaries, durations, spike counts, bin size, and spike frames by channel.
- `analyse_criticality` returned `AnalysisResultCriticality` with `avalanche_spike_counts`, `avalanche_durations`, shape/profile fields, upstream exponents, and branching fields.
- `analyse_functional_connectivity` returned `AnalysisResultsFunctionalConnectivity` with a weighted adjacency matrix, Louvain `graph_partition`, modularity, and clustering coefficient.

## Conventions That Matter

- Timestamps are frame indices at `frames_per_second`, usually 25 kHz.
- Burst thresholds are per-channel firing rates in Hz. The project wrappers default to `bin_size_sec=0.05`, `onset_freq_hz=3.0`, `offset_freq_hz=1.0`, and `min_active_channels=3` until calibration says otherwise.
- Criticality uses `percentile_threshold` in `[0, 1]`, not `[0, 100]`.
- Criticality and functional connectivity require the SDK's 64-channel common MEA layout. Ground channels `0, 7, 56, 63` must have no spikes or samples.
- The project maps its 60-electrode layout into the 60 non-ground channels of the SDK common layout for delegated analysis.

## Does Not Work

- `cl.analysis` does not expose public analysis functions directly; the callable public API is `RecordingView.analyse_*`.
- Criticality and connectivity reject non-common-layout recordings with `ValueError: Recording does not conform to common MEA layout.`
- `cl-sdk==1.0.0` writes spike channel ids as `uint8`, so native SDK-written spike tables cannot losslessly store channel ids above 255. A 1024-electrode HD-MEA recording therefore cannot be represented through the SDK writer without a separate compatibility decision.
- `cl.sim` is a simulator data-source API mock. It is not used by `culture-sim` for scientific simulation.
