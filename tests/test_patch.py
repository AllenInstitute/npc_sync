"""SyncPatch, strict invalid-interval policy and per-gap reconstruction, on synthetic sync files
with replayed buffers.

See npc_sessions outputs/sync-frame-timing-upgrade-plan.md (P1) and npc_sync.patch.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pytest

import npc_sync
from npc_sync import patch as P

FS = 100_000
BUFFER = 50_000
LABELS = [""] * 32
LABELS[2] = "vsync_stim"
LABELS[4] = "stim_photodiode"
LABELS[5] = "stim_running"
LABELS[27] = "beh_cam_exposing"
LABELS[31] = "lick_sensor"


def _truth(seed: int = 0, duration: float = 60.0, stim_start: float = 5.0, n_frames: int = 3000):
    rng = np.random.default_rng(seed)
    fi = np.full(n_frames - 1, 1 / 60) + rng.normal(0, 2e-5, n_frames - 1)
    for i in (700, 1500, 1520, 2400):  # dropped frames
        fi[i] = 3 / 60
    frames = stim_start + np.r_[0, np.cumsum(fi)]
    edges: dict[int, list[tuple[float, int]]] = {b: [] for b in range(32)}
    for t in frames:  # vsync: rising 2 ms before the falling edge
        edges[2] += [(t - 0.002, 1), (t, 0)]
    pol = 1
    for f in range(0, n_frames, 3):  # diode toggles every 3 frames, polarity-dependent latency
        edges[4].append((frames[f] + (0.020 if pol else 0.024), pol))
        pol ^= 1
    edges[5] += [(stim_start - 0.05, 1), (frames[-1] + 0.05, 0)]
    cam = np.arange(1.0, duration - 1, 1 / 60.02)
    for t in cam:
        edges[27] += [(t, 1), (t + 0.003, 0)]
    licks = np.sort(rng.uniform(stim_start, frames[-1], 150))
    licks = licks[np.r_[True, np.diff(licks) > 0.1]]
    for t in licks:
        edges[31] += [(t, 1), (t + 0.03, 0)]
    return fi, frames, edges, cam, licks


def _events(edges: dict[int, list[tuple[float, int]]]) -> tuple[np.ndarray, np.ndarray]:
    items = sorted((int(round(t * FS)), b, v) for b, es in edges.items() for t, v in es)
    times, states = [0], [0]  # initial state (the first event is never an edge)
    state = 0
    for s, b, v in items:
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times and times[-1] == s:
            states[-1] = state
        else:
            times.append(s)
            states.append(state)
    return np.asarray(times, np.int64), np.asarray(states, np.int64)


def _replay(times, states, genuine_buffer: int, n_copies: int):
    """Overwrite the `n_copies` buffers before `genuine_buffer` with copies of it."""
    g0 = genuine_buffer * BUFFER
    first = g0 - n_copies * BUFFER
    g = (times >= g0) & (times < g0 + BUFFER)
    rel, st = times[g] - g0, states[g]
    keep = (times < first) | (times >= g0)
    parts_t, parts_s = [times[keep & (times < first)]], [states[keep & (times < first)]]
    for c in range(n_copies):
        parts_t.append(first + c * BUFFER + rel)
        parts_s.append(st)
    parts_t.append(times[times >= g0])
    parts_s.append(states[times >= g0])
    return np.concatenate(parts_t), np.concatenate(parts_s), (first / FS, g0 / FS)


def _write(path: pathlib.Path, times, states, duration: float) -> pathlib.Path:
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:00.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": int(duration * FS),
    }
    with h5py.File(path, "w") as f:
        f["data"] = np.c_[times.astype(np.uint32), states.astype(np.uint32)]
        f["meta"] = str(meta)
    return path


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """One block with two replays: 2 s (buffers 36-39 copies of 40) and 0.5 s (60 of 61)."""
    fi, frames, edges, cam, licks = _truth()
    t, s = _events(edges)
    t, s, iv1 = _replay(t, s, genuine_buffer=40, n_copies=4)
    t, s, iv2 = _replay(t, s, genuine_buffer=61, n_copies=1)
    path = _write(tmp_path_factory.mktemp("sync") / "replayed.h5", t, s, 60.0)
    return dict(path=path, fi=fi, frames=frames, cam=cam, licks=licks, intervals=(iv1, iv2))


def test_detects_intervals(replayed) -> None:
    s = npc_sync.SyncDataset(replayed["path"])
    assert s.invalid_intervals == replayed["intervals"]


def test_strict_policy_raises(replayed) -> None:
    """Fails under <= 0.3.1, which returned the copies' edges."""
    s = npc_sync.SyncDataset(replayed["path"])
    for line in ("vsync_stim", "lick_sensor", "beh_cam_exposing", 3):
        with pytest.raises(npc_sync.SyncFaultError):
            s.get_rising_edges(line, units="seconds")
    with pytest.raises(npc_sync.SyncFaultError):
        s.vsync_times_in_blocks
    with pytest.raises(npc_sync.SyncFaultError):
        s.get_events_by_line("lick_sensor")


def test_ignore_policy_returns_copies(replayed) -> None:
    s = npc_sync.SyncDataset(replayed["path"], on_invalid="ignore")
    v = s.get_falling_edges("vsync_stim", units="seconds")
    assert len(v) != len(replayed["frames"])  # copies change the count


def test_mask_policy_bridges_blocks(replayed) -> None:
    s = npc_sync.SyncDataset(replayed["path"], on_invalid="mask")
    licks = s.get_rising_edges("lick_sensor", units="seconds")
    assert not P.in_intervals(licks, replayed["intervals"]).any()
    # the 2 s masked gap is longer than the 1 s block-splitting gap but must not split the block
    (block,) = s.vsync_times_in_blocks
    assert block.size < len(replayed["frames"])
    assert s.get_unknown_intervals("vsync_stim")


def test_per_gap_solve(replayed) -> None:
    raw = npc_sync.SyncDataset(replayed["path"], on_invalid="ignore")
    (block,) = raw.vsync_times_in_blocks
    rec = P.reconcile_vsyncs(block, replayed["fi"], raw.invalid_intervals)
    assert rec.balance == 0 and len(rec.gaps) == 2
    frames = replayed["frames"]
    for g in rec.gaps:
        truth = int(P.in_intervals(frames, [g.span]).sum())
        assert g.n_frames == truth
    assert np.nanmax(np.abs(rec.times - frames)) < 2 / FS


def test_gap_at_block_boundary_fails() -> None:
    fi = np.full(599, 1 / 60)
    tf = np.r_[0, np.cumsum(fi)] + 10.0
    rec = P.solve_vsync_gaps(tf, fi, [(9.9, 10.5)])
    assert rec.boundary_gap
    with pytest.raises(P.ReconciliationError):
        P.reconcile_vsyncs(tf, fi, [(9.9, 10.5)])


def test_unexplained_count_mismatch_fails() -> None:
    """Without an invalid interval there's nothing to reconcile: one extra vsync fails the
    balance (the stim matcher's old atol=1 accepted this)."""
    fi = np.full(599, 1 / 60)
    tf = np.r_[0, np.cumsum(fi)] + 10.0
    with pytest.raises(P.ReconciliationError, match="balance"):
        P.reconcile_vsyncs(np.sort(np.r_[tf, 12.0001]), fi, [])


def test_patch_applies(replayed) -> None:
    raw = npc_sync.SyncDataset(replayed["path"], on_invalid="ignore")
    patch = P.build_sync_patch(
        raw, [P.StimBlock(0, replayed["fi"], 3, "synthetic")], builder="test"
    )
    s = npc_sync.SyncDataset(replayed["path"], patch=patch)
    frames = replayed["frames"]
    v = s.get_falling_edges("vsync_stim", units="seconds")
    assert len(v) == len(frames) and np.max(np.abs(v - frames)) < 2 / FS
    src = s.get_edge_sources("vsync_stim", "falling")
    assert (src == P.INTERPOLATED).sum() == P.in_intervals(frames, s.invalid_intervals).sum()
    (block,) = s.vsync_times_in_blocks
    assert len(block) == len(frames)
    # vsync rising edges: reconstructed from the measured pulse width
    r = s.get_rising_edges("vsync_stim", units="seconds")
    assert len(r) == len(frames) and np.max(np.abs(r - (frames - 0.002))) < 3 / FS
    # camera: integer-period interpolation
    cam = s.get_rising_edges("beh_cam_exposing", units="seconds")
    assert len(cam) == len(replayed["cam"]) and np.max(np.abs(cam - replayed["cam"])) < 2 / FS
    # photodiode: predicted edges match the truth
    d = np.sort(
        np.r_[
            s.get_rising_edges("stim_photodiode", units="seconds"),
            s.get_falling_edges("stim_photodiode", units="seconds"),
        ]
    )
    pol = 1
    true_d = []
    for f in range(0, len(frames), 3):
        true_d.append(frames[f] + (0.020 if pol else 0.024))
        pol ^= 1
    assert len(d) == len(true_d) and np.max(np.abs(d - np.array(true_d))) < 3 / FS
    # licks: unknown inside the intervals, measured outside
    unknown = s.get_unknown_intervals("lick_sensor")
    assert all(any(a >= ua and z <= uz for ua, uz in unknown) for a, z in s.invalid_intervals)
    licks = s.get_rising_edges("lick_sensor", units="seconds")
    truth = replayed["licks"][~P.in_intervals(replayed["licks"], unknown)]
    assert np.allclose(licks, truth, atol=1 / FS)
    # unlabelled, unused lines have no edges anywhere: reconstructed as no edges, no raise
    assert s.get_rising_edges(3, units="seconds").size == 0
    assert s.get_unknown_intervals(3) == ()


def test_patch_round_trip_and_hash(replayed, tmp_path) -> None:
    raw = npc_sync.SyncDataset(replayed["path"], on_invalid="ignore")
    patch = P.build_sync_patch(raw, [P.StimBlock(0, replayed["fi"], 3)], builder="test")
    path = tmp_path / "patch.json"
    patch.save(path)
    loaded = P.SyncPatch.load(path)
    a = npc_sync.SyncDataset(replayed["path"], patch=patch)
    b = npc_sync.SyncDataset(replayed["path"], patch=loaded)
    for bit in range(32):
        assert np.array_equal(
            a.get_rising_edges(bit, units="seconds"), b.get_rising_edges(bit, units="seconds")
        )
    loaded.sync_content_hash = "0" * 64
    with pytest.raises(ValueError, match="different sync file"):
        npc_sync.SyncDataset(replayed["path"], patch=loaded)


def test_patch_without_stim_leaves_vsync_unknown(replayed) -> None:
    raw = npc_sync.SyncDataset(replayed["path"], on_invalid="ignore")
    patch = P.build_sync_patch(raw, [], builder="test")
    s = npc_sync.SyncDataset(replayed["path"], patch=patch)
    assert s.get_unknown_intervals("vsync_stim")
    assert not P.in_intervals(
        s.get_falling_edges("vsync_stim", units="seconds"), s.invalid_intervals
    ).any()
    (block,) = s.vsync_times_in_blocks  # bridged, not split
    assert block.size < len(replayed["frames"])


def test_clean_file_unchanged(tmp_path) -> None:
    fi, frames, edges, cam, licks = _truth()
    t, s = _events(edges)
    path = _write(tmp_path / "clean.h5", t, s, 60.0)
    d = npc_sync.SyncDataset(path)
    assert d.invalid_intervals == ()
    v = d.get_falling_edges("vsync_stim", units="seconds")
    assert np.max(np.abs(v - frames)) < 1 / FS
    assert (d.get_edge_sources("vsync_stim", "falling") == P.MEASURED).all()
    assert d.get_unknown_intervals("lick_sensor") == ()
