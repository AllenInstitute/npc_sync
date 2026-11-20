"""Timebase of a sync file (npc_sync.timebase): dropped acquisition buffers located on camera
lines, displaced data before a replayed run, and patches built on the corrected clock.

Synthetic files: a truth event stream is cut into buffers and written in the order
`recorded buffer p <- true buffer source[p]`, the way the sync program writes events (a row
where the state differs from the previous written sample).
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

import npc_sync
from npc_sync import patch as P
from npc_sync import timebase as T

FS = 100_000
BUFFER = 50_000
LABELS = [""] * 32
LABELS[0] = "barcode_ephys"
LABELS[2] = "vsync_stim"
LABELS[4] = "stim_photodiode"
LABELS[5] = "stim_running"
LABELS[27] = "beh_cam_exposing"
LABELS[29] = "eye_cam_exposing"
LABELS[30] = "face_cam_exposing"
LABELS[31] = "lick_sensor"
CAM_HZ = {27: 60.0027, 29: 60.0012, 30: 60.0019}  # a buffer is 30 periods - 13..23 us
DURATION = 200.0
N_BUFFERS = int(DURATION * FS) // BUFFER


def _truth(seed: int = 0, stim_start: float = 5.0, n_frames: int = 10_800):
    rng = np.random.default_rng(seed)
    fi = np.full(n_frames - 1, 1 / 59.95) + rng.normal(0, 2e-5, n_frames - 1)
    for i in (700, 1500, 1520, 2400, 4000, 5200, 9000):  # dropped frames
        fi[i] = 3 / 59.95
    frames = stim_start + np.r_[0, np.cumsum(fi)]
    edges: dict[int, list[tuple[float, int]]] = {b: [] for b in range(32)}
    for t in frames:  # vsync: rising 2 ms before the falling edge
        edges[2] += [(t - 0.002, 1), (t, 0)]
    pol = 1
    for f in range(0, n_frames, 3):  # diode toggles every 3 frames
        edges[4].append((frames[f] + (0.020 if pol else 0.024), pol))
        pol ^= 1
    edges[5] += [(stim_start - 0.05, 1), (frames[-1] + 0.05, 0)]
    cams = {}
    for bit, hz in CAM_HZ.items():
        cam = np.arange(1.0 + bit * 1e-3, DURATION - 1, 1 / hz)
        cam = cam + rng.uniform(-2e-6, 2e-6, cam.size)  # a little jitter
        cams[bit] = cam
        for t in cam:
            edges[bit] += [(t, 1), (t + 0.003, 0)]
    licks = np.sort(rng.uniform(stim_start, frames[-1], 400))
    licks = licks[np.r_[True, np.diff(licks) > 0.1]]
    for t in licks:
        edges[31] += [(t, 1), (t + 0.03, 0)]
    barcodes = np.arange(3.0, DURATION - 3, 31.0)
    for t in barcodes:
        for k in range(8):
            edges[0] += [(t + 0.03 * k, 1), (t + 0.03 * k + 0.012, 0)]
    return dict(
        fi=fi, frames=frames, edges=edges, cams=cams, licks=licks, barcodes=barcodes
    )


def _events(edges: dict[int, list[tuple[float, int]]]) -> tuple[np.ndarray, np.ndarray]:
    items = sorted((int(round(t * FS)), b, v) for b, es in edges.items() for t, v in es)
    times, states = [0], [0]
    state = 0
    for s, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == s:
            states[-1] = state
        else:
            times.append(s)
            states.append(state)
    return np.asarray(times, np.int64), np.asarray(states, np.int64)


def _record(times, states, source: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Recorded event stream when recorded buffer p holds true buffer source[p]."""
    out_t: list[int] = []
    out_s: list[int] = []
    last = 0
    for p, c in enumerate(source):
        i, j = np.searchsorted(times, [c * BUFFER, (c + 1) * BUFFER])
        start_state = int(states[np.searchsorted(times, c * BUFFER, side="right") - 1])
        if (i == j or times[i] != c * BUFFER) and start_state != last:
            out_t.append(p * BUFFER)
            out_s.append(start_state)
        out_t += (times[i:j] - c * BUFFER + p * BUFFER).tolist()
        out_s += states[i:j].tolist()
        last = int(states[j - 1]) if j > i else start_state
    return np.asarray(out_t, np.int64), np.asarray(out_s, np.int64)


def _write(path: pathlib.Path, times, states, n_buffers: int) -> pathlib.Path:
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:03:20.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": int(n_buffers * BUFFER),
    }
    with h5py.File(path, "w") as f:
        f["data"] = np.c_[times.astype(np.uint32), states.astype(np.uint32)]
        f["meta"] = str(meta)
    return path


@pytest.fixture(scope="module")
def truth():
    tr = _truth()
    tr["times"], tr["states"] = _events(tr["edges"])
    return tr


def _file(tmp_path_factory, truth, name: str, source: list[int]) -> pathlib.Path:
    t, s = _record(truth["times"], truth["states"], source)
    return _write(tmp_path_factory.mktemp("sync") / f"{name}.h5", t, s, len(source))


@pytest.fixture(scope="module")
def dropped(tmp_path_factory, truth):
    """3 buffers dropped at 40 s (true 40.0-41.5 s missing), 1 at 90 s: no copies."""
    source = [c for c in range(N_BUFFERS) if c not in (80, 81, 82, 180)]
    return _file(tmp_path_factory, truth, "dropped", source)


@pytest.fixture(scope="module")
def displaced(tmp_path_factory, truth):
    """Buffers dropped at true 30.0 and 33.0 s, made up for by 2 copies recorded at
    36.0-37.0 s; and a run of 3 copies at 70.0-71.5 s with its 3 drops at the run start.
    """
    source = list(range(N_BUFFERS))
    # recorded 60..65 hold true 61..66; 66..71 hold true 68..73; 72, 73 are copies of 74
    source[60:66] = range(61, 67)
    source[66:72] = range(68, 74)
    source[72:74] = [74, 74]
    source[140:143] = [143, 143, 143]
    return _file(tmp_path_factory, truth, "displaced", source)


def _check_measured_lines(s: npc_sync.SyncDataset, truth, unknown) -> None:
    """Edges returned outside `unknown` are the true ones, on every kind of line."""
    for bit, name in ((31, "lick_sensor"), (0, "barcode_ephys")):
        true_r = np.array([t for t, v in truth["edges"][bit] if v])
        r = s.get_rising_edges(name, units="seconds")
        assert not P.in_intervals(r, unknown).any()
        expected = true_r[~P.in_intervals(true_r, unknown)]
        assert len(r) == len(expected) and np.max(np.abs(r - expected)) < 1.5 / FS


def test_drops_are_located(dropped, truth) -> None:
    raw = npc_sync.SyncDataset(dropped)
    assert raw.invalid_intervals == ()
    # each drop shortens one barcode onset interval
    assert [round(d, 3) for _, _, d in raw.timebase_discontinuities] == [-1.5, -0.5]
    tb = raw.find_timebase()
    assert [(j.sample, j.n_chunks, j.source) for j in tb.jumps] == [
        (80 * BUFFER, 3, "periodic_lines"),
        (177 * BUFFER, 1, "periodic_lines"),
    ]
    assert tb.dropped == ((80 * BUFFER, 83 * BUFFER), (180 * BUFFER, 181 * BUFFER))
    assert tb.total_samples_true == N_BUFFERS * BUFFER
    s = npc_sync.SyncDataset(dropped, timebase=tb, on_invalid="mask")
    assert s.invalid_intervals == ((40.0, 41.5), (90.0, 90.5))
    assert s.total_seconds == DURATION
    assert s.timebase_discontinuities == ()
    (block,) = s.vsync_times_in_blocks  # the 1.5 s gap doesn't split the block
    frames = truth["frames"]
    kept = frames[~P.in_intervals(frames, s._mask_intervals)]
    assert len(block) == len(kept) and np.max(np.abs(block - kept)) < 1.5 / FS
    _check_measured_lines(s, truth, s._mask_intervals)


def test_timebase_inherited_and_recorded_view(dropped) -> None:
    raw = npc_sync.SyncDataset(dropped)
    tb = raw.find_timebase()
    s = npc_sync.SyncDataset(dropped, timebase=tb)
    assert npc_sync.SyncDataset(s, on_invalid="ignore").timebase == tb
    recorded = npc_sync.SyncDataset(s, on_invalid="ignore", timebase=False)
    assert recorded.timebase is None and recorded.timebase_discontinuities
    assert recorded.find_timebase() == tb == s.find_timebase()
    n = raw.meta_data["total_samples"]
    assert len(recorded.get_all_events()) == len(raw.get_recorded_events())
    assert s.get_all_times("samples").max() < tb.total_samples_true
    assert recorded.get_all_times("samples").max() < n


def test_patch_reconstructs_dropped_data(dropped, truth) -> None:
    raw = npc_sync.SyncDataset(dropped, on_invalid="ignore")
    tb = raw.find_timebase()
    view = npc_sync.SyncDataset(raw, on_invalid="ignore", timebase=tb)
    assert len(view.vsync_times_in_blocks) == 1
    patch = P.build_sync_patch(
        raw, [P.StimBlock(0, truth["fi"], 3, "synthetic")], builder="test"
    )
    assert patch.timebase == tb
    assert [f.kind for f in patch.faults] == ["drop", "drop"]
    assert [(f.start, f.shift) for f in patch.faults] == [(40.0, -1.5), (88.5, -0.5)]
    patch = P.SyncPatch.from_json(patch.to_json())  # round trip, timebase included
    assert patch.timebase == tb
    s = npc_sync.SyncDataset(dropped, patch=patch)
    assert s.timebase == tb and s.invalid_intervals == ((40.0, 41.5), (90.0, 90.5))
    frames = truth["frames"]
    v = s.get_falling_edges("vsync_stim", units="seconds")
    assert len(v) == len(frames) and np.max(np.abs(v - frames)) < 2 / FS
    src = s.get_edge_sources("vsync_stim", "falling")
    assert (src == P.INTERPOLATED).sum() == P.in_intervals(
        frames, s.invalid_intervals
    ).sum()
    for bit, name in ((27, "beh_cam_exposing"), (29, "eye_cam_exposing")):
        cam = s.get_rising_edges(name, units="seconds")
        assert len(cam) == len(truth["cams"][bit])
        assert np.max(np.abs(cam - truth["cams"][bit])) < 5e-6 + 1 / FS
    d = np.sort(
        np.r_[
            s.get_rising_edges("stim_photodiode", units="seconds"),
            s.get_falling_edges("stim_photodiode", units="seconds"),
        ]
    )
    true_d = np.sort([t for t, _ in truth["edges"][4]])
    assert len(d) == len(true_d) and np.max(np.abs(d - true_d)) < 3 / FS
    unknown = s.get_unknown_intervals("lick_sensor")
    assert len(unknown) == 2
    _check_measured_lines(s, truth, unknown)


def test_displaced_data_before_replayed_run(displaced, truth) -> None:
    raw = npc_sync.SyncDataset(displaced, on_invalid="ignore")
    assert raw.invalid_intervals == ((36.0, 37.0), (70.0, 71.5))
    tb = raw.find_timebase()
    assert [
        (j.sample // BUFFER, j.n_chunks, j.source) for j in tb.jumps if j.n_chunks > 0
    ] == [
        (60, 1, "periodic_lines"),
        (66, 1, "periodic_lines"),
        (140, 3, "periodic_lines"),
    ]
    assert sum(j.n_chunks == -1 for j in tb.jumps) == 5
    assert tb.net_shift_samples == 0 and not tb.notes
    # true 30.0-30.5 and 33.5-34.0 s are missing; the copies at 36-37 s are not
    assert tb.dropped == (
        (60 * BUFFER, 61 * BUFFER),
        (67 * BUFFER, 68 * BUFFER),
        (140 * BUFFER, 143 * BUFFER),
    )
    patch = P.build_sync_patch(
        raw, [P.StimBlock(0, truth["fi"], 3, "synthetic")], builder="test"
    )
    s = npc_sync.SyncDataset(displaced, patch=patch)
    assert s.invalid_intervals == ((30.0, 30.5), (33.5, 34.0), (70.0, 71.5))
    frames = truth["frames"]
    v = s.get_falling_edges("vsync_stim", units="seconds")
    assert len(v) == len(frames) and np.max(np.abs(v - frames)) < 2 / FS
    assert s.timebase_discontinuities == ()
    unknown = s.get_unknown_intervals("lick_sensor")
    _check_measured_lines(s, truth, unknown)
    # without the timebase (the model before drops were located), licks recorded between a
    # drop and its run are 0.5-1 s early
    old = npc_sync.SyncDataset(displaced, on_invalid="mask", timebase=False)
    licks = old.get_rising_edges("lick_sensor", units="seconds")
    true_licks = truth["licks"]
    in_stretch = (licks >= 30.0) & (licks < 36.0)
    err = np.array([np.min(np.abs(true_licks - t)) for t in licks[in_stretch]])
    assert in_stretch.sum() > 5 and (err > 0.01).mean() > 0.5


def test_run_without_periodic_lines_is_assumed_at_its_start(displaced) -> None:
    raw = npc_sync.SyncDataset(displaced, on_invalid="ignore")
    tb = raw.find_timebase(periodic_lines=[])
    assert [
        (j.sample // BUFFER, j.n_chunks, j.source) for j in tb.jumps if j.n_chunks > 0
    ] == [
        (72, 2, "assumed_at_run"),
        (140, 3, "assumed_at_run"),
    ]
    assert tb.has_assumed_drops
    # same intervals as the replayed copies
    assert npc_sync.SyncDataset(
        raw, on_invalid="ignore", timebase=tb
    ).invalid_intervals == (
        (36.0, 37.0),
        (70.0, 71.5),
    )


def test_one_camera_is_not_enough(dropped) -> None:
    raw = npc_sync.SyncDataset(dropped)
    tb = raw.find_timebase(periodic_lines=["beh_cam_exposing"])
    assert tb.is_identity


def test_build_timebase_checks() -> None:
    with pytest.raises(ValueError, match="chunk grid"):
        T.build_timebase(1_000_000, replayed=[(600_001, 700_000)])
    with pytest.raises(ValueError, match="follows a replayed copy"):
        T.build_timebase(1_000_000, drops={13: 1}, replayed=[(600_000, 700_000)])
    tb = T.build_timebase(1_000_000)
    assert tb.is_identity and tb.dropped == () and tb.total_samples_true == 1_000_000
    assert T.Timebase.from_dict(tb.to_dict()) == tb
