"""Photodiode phantom flicker and box-1 toggle alignment (`npc_sync.patch.align_diode_toggles`),
removal of measured edges by a `SyncPatch` (any file), and box-1 prediction inside replayed spans.

Evidence and design: npc_sessions experiments/phantom-flicker (README) and
outputs/phantom-flicker-report.md.
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
R = 1 / 60
LAT_R, LAT_F = 0.020, 0.024  # TTL latency after the vsync falling edge, by polarity (> 1 refresh)
LABELS = [""] * 32
LABELS[2], LABELS[4], LABELS[5] = "vsync_stim", "stim_photodiode", "stim_running"


def _box1(
    n_frames: int = 4000,
    holds: dict[int, int] | None = None,
    phantom_pairs: dict[int, list[int]] | None = None,
    glitches: tuple[int, ...] = (),
    late_returns: dict[int, float] | None = None,
    extra_edges: tuple[float, ...] = (),
    seed: int = 0,
    start: float = 5.0,
    default_holds: bool = True,
):
    """Synthetic box-1 block. holds: frame -> refreshes held; phantom_pairs: held frame -> missed
    refreshes (1..n-2) at which the display flickers to the other buffer for one refresh (TTL:
    the first edge 3 ms late, the restoring edge on time); glitches: frames whose toggle has
    chatter (F R F within 0.1 ms); late_returns: frame -> seconds its frame-signal falling edge
    (and the stim's frame interval) is late, the frame displayed on time; extra_edges: times
    of extra single edges (unexplained).

    Returns (frame_intervals, display times (refresh grid), vsync falling, vsync rising, edges).
    """
    defaults = {700: 3, 1500: 2, 1520: 4, 2400: 11, 3000: 3, 3300: 3, 3600: 2} if default_holds else {}
    holds = {**defaults, **(holds or {})}
    rng = np.random.default_rng(seed)
    n_ref = np.ones(n_frames - 1, int)
    for k, n in holds.items():
        n_ref[k] = n
    display = start + np.r_[0, np.cumsum(n_ref)] * R  # refresh-grid display times
    fall = display + 0.0002 + rng.normal(0, 2e-5, n_frames)  # flip() returns just after the vsync
    for k, d in (late_returns or {}).items():
        fall[k] += d
    rise = fall - 0.015
    for k, d in (late_returns or {}).items():
        rise[k] -= d  # the frame signal rose on time
    fi = np.diff(fall)
    edges: list[tuple[float, int, int]] = []
    for t_r, t_f in zip(rise, fall):
        edges += [(t_r, 2, 1), (t_f, 2, 0)]
    pol = 1
    for k in range(n_frames):
        t = display[k] + (LAT_R if pol else LAT_F)
        if k in glitches:
            edges += [(t, 4, pol), (t + 0.00005, 4, 1 - pol), (t + 0.0001, 4, pol)]
        else:
            edges.append((t, 4, pol))
        for r in (phantom_pairs or {}).get(k, []):
            # flicker to the other buffer at missed refresh r, back at r + 1
            edges.append((display[k] + r * R + (LAT_R if 1 - pol else LAT_F) + 0.003, 4, 1 - pol))
            edges.append((display[k] + (r + 1) * R + (LAT_R if pol else LAT_F), 4, pol))
        pol ^= 1
    for t in extra_edges:
        edges.append((t, 4, -1))  # polarity resolved below
    edges += [(start - 0.05, 5, 1), (fall[-1] + 0.05, 5, 0)]
    return fi, display, fall, rise, edges


def _events(edges):
    items = sorted((int(round(t * FS)), b, v) for t, b, v in edges)
    times, states, state = [0], [0], 0
    for s, b, v in items:
        if v == -1:  # toggle the bit
            v = 0 if (state >> b) & 1 else 1
        state = (state | (1 << b)) if v else (state & ~(1 << b))
        if times[-1] == s:
            states[-1] = state
        else:
            times.append(s)
            states.append(state)
    return np.asarray(times, np.int64), np.asarray(states, np.int64)


def _write(path: pathlib.Path, times, states, duration: float = 100.0) -> pathlib.Path:
    meta = {
        "ni_daq": {"sample_freq": float(FS), "counter_bits": 32, "event_bits": 32},
        "start_time": "2026-01-01 10:00:00.000000",
        "stop_time": "2026-01-01 10:01:40.000000",
        "line_labels": LABELS,
        "timeouts": [],
        "version": "synthetic",
        "total_samples": int(duration * FS),
    }
    with h5py.File(path, "w") as f:
        f["data"] = np.c_[times.astype(np.uint32), states.astype(np.uint32)]
        f["meta"] = str(meta)
    return path


def _diode(s: npc_sync.SyncDataset):
    r = s.get_rising_edges("stim_photodiode", units="seconds")
    f = s.get_falling_edges("stim_photodiode", units="seconds")
    e = np.r_[r, f]
    q = np.r_[np.ones(len(r), bool), np.zeros(len(f), bool)]
    o = np.argsort(e, kind="stable")
    return e[o], q[o]


def _align(fi, fall, rise, edges):
    t, s = _events(edges)
    bit = (s >> 4) & 1
    ch = np.flatnonzero(np.diff(bit) != 0) + 1
    e = t[ch] / FS
    q = bit[ch] == 1
    v = fall  # one vsync per frame
    return P.align_diode_toggles(v, fi, e, q, frame_signal_rising=rise), e, q


def test_phantom_pairs_and_glitches_are_classified() -> None:
    pairs = {700: [1], 2400: [1, 3, 5, 7, 9], 3000: [1]}
    fi, display, fall, rise, edges = _box1(phantom_pairs=pairs, glitches=(1000, 2001))
    al, e, q = _align(fi, fall, rise, edges)
    assert al.count(P.DIODE_PHANTOM_FLICKER) == 2 * sum(len(v) for v in pairs.values())
    assert al.count(P.DIODE_GLITCH) == 4
    assert al.count(P.DIODE_UNEXPLAINED) == 0 and len(al.missing_frames) == 0
    assert al.failures() == []
    held = set(al.edge_frame[al.edge_class == P.DIODE_PHANTOM_FLICKER].tolist())
    assert held == set(pairs)
    # the box-1 latency is longer than a refresh: no one-frame alias
    s = al.summary()
    assert s["latency_rising_ms"] == pytest.approx(LAT_R * 1e3, abs=0.5)
    assert s["latency_falling_ms"] == pytest.approx(LAT_F * 1e3, abs=0.5)
    toggles = e[al.frame_edge]
    assert np.max(np.abs(toggles - (display + np.where(al.frame_rising, LAT_R, LAT_F)))) < 0.2e-3


def test_late_flip_return_is_matched_to_the_displayed_frame() -> None:
    """Frame 1200's flip() returns 14 ms late: the stim's interval 1199 looks like a held frame
    (30.7 ms) and 1200 is short, but the frame was displayed on time; its toggle is 14 ms early
    relative to the late falling edge (outside the match tolerance) and is matched in the second
    pass, bounded by the frame-signal rising edge."""
    fi, display, fall, rise, edges = _box1(late_returns={1200: 0.014})
    al, e, q = _align(fi, fall, rise, edges)
    assert 1200 in set(al.second_pass.tolist())
    assert al.failures() == []
    assert len(al.long_interval_mismatch["late_return"]) >= 1
    # without the rising edges the toggle stays unmatched
    t, s = _events(edges)
    al2 = P.align_diode_toggles(fall, fi, e, q)
    assert 1200 in set(al2.missing_frames.tolist())


@pytest.mark.parametrize(
    "case",
    [
        "pair_in_regular_frame",  # 2 extra edges where no frame is held
        "odd_in_hold",  # 1 extra edge in a held frame: not state-restoring
        "pair_in_two_refresh_hold",  # 2 edges > n - 1 = 1 for a 2-refresh hold
    ],
)
def test_flicker_is_not_a_catch_all(case: str) -> None:
    fi, display, fall, rise, edges = _box1()
    if case == "pair_in_regular_frame":
        k = 900
        extra = (display[k] + 0.006 + LAT_F, display[k] + 0.010 + LAT_F)
    elif case == "odd_in_hold":
        k = 700  # 3-refresh hold
        extra = (display[k] + R + LAT_F + 0.003,)
    else:
        k = 1500  # 2-refresh hold
        extra = (display[k] + R + 0.002 + LAT_F, display[k] + R + 0.006 + LAT_F)
    _, _, _, _, edges = _box1(extra_edges=extra)
    al, e, q = _align(fi, fall, rise, edges)
    assert al.count(P.DIODE_PHANTOM_FLICKER) == 0
    assert al.failures(), case


def test_offset_undetermined_without_held_frames() -> None:
    """With no held frame the frame <-> edge offset can't be fixed (v[k] + L ~ v[k+1] + L - R)."""
    fi, display, fall, rise, edges = _box1(n_frames=2000, default_holds=False)
    with pytest.raises(P.DiodeOffsetUndetermined):
        _align(fi, fall, rise, edges)


@pytest.fixture()
def flicker_file(tmp_path):
    pairs = {700: [1], 2400: [1, 3, 5, 7], 3000: [1]}
    fi, display, fall, rise, edges = _box1(phantom_pairs=pairs, glitches=(1000,))
    t, s = _events(edges)
    path = _write(tmp_path / "flicker.h5", t, s)
    return dict(path=path, fi=fi, display=display, pairs=pairs)


def test_patch_removes_phantom_flicker_in_a_file_without_replays(flicker_file, tmp_path) -> None:
    raw = npc_sync.SyncDataset(flicker_file["path"])
    assert raw.invalid_intervals == ()
    n_pairs = sum(len(v) for v in flicker_file["pairs"].values())
    e_raw, _ = _diode(raw)
    assert len(e_raw) == len(flicker_file["display"]) + 2 * n_pairs + 2
    patch = P.build_sync_patch(raw, [P.StimBlock(0, flicker_file["fi"], 1, "synthetic")], builder="test")
    (rec,) = patch.diode_blocks
    assert rec.status == "cleaned" and rec.n_phantom_flicker == 2 * n_pairs and rec.n_glitch == 2
    assert patch.diode_summary()["phantom_flicker_removed"] == 2 * n_pairs
    s = npc_sync.SyncDataset(flicker_file["path"], patch=patch)
    e, q = _diode(s)
    assert len(e) == len(flicker_file["display"])  # one edge per frame
    assert (np.diff(q.astype(int)) != 0).all()  # alternating
    lat = np.where(q, LAT_R, LAT_F)
    assert np.max(np.abs(e - lat - flicker_file["display"])) < 0.2e-3
    # long diode intervals are exactly the stim's held frames
    d = np.diff(e - lat)
    assert set(np.flatnonzero(d > 1.5 * R).tolist()) == set(np.flatnonzero(flicker_file["fi"] > 1.5 * R).tolist())
    # every other line is untouched
    assert np.array_equal(s.get_falling_edges("vsync_stim"), raw.get_falling_edges("vsync_stim"))
    # JSON round trip
    path = tmp_path / "patch.json"
    patch.save(path)
    loaded = P.SyncPatch.load(path)
    assert loaded.diode_blocks[0].status == "cleaned"
    e2, _ = _diode(npc_sync.SyncDataset(flicker_file["path"], patch=loaded))
    assert np.array_equal(e, e2)


def test_version_1_patch_loads(flicker_file) -> None:
    raw = npc_sync.SyncDataset(flicker_file["path"])
    d = P.build_sync_patch(raw, [], builder="test").to_dict()
    d["format_version"] = 1
    d.pop("removed")
    d.pop("diode_blocks")
    p = P.SyncPatch.from_dict(d)
    assert p.removed == {} and p.diode_blocks == []


def test_unexplained_edges_reject_the_block_and_remove_nothing(tmp_path) -> None:
    k = 900
    _, display, fall, rise, edges = _box1(
        phantom_pairs={700: [1]},
        extra_edges=(5.0 + (k + 0.35) * R + LAT_F, 5.0 + (k + 0.6) * R + LAT_F),
    )
    t, s = _events(edges)
    path = _write(tmp_path / "unexplained.h5", t, s)
    raw = npc_sync.SyncDataset(path)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, np.diff(fall), 1)], builder="test")
    (rec,) = patch.diode_blocks
    assert rec.status == "rejected" and "neither a frame's toggle" in rec.reason
    assert patch.removed == {}
    e_raw, _ = _diode(raw)
    e, _ = _diode(npc_sync.SyncDataset(path, patch=patch))
    assert np.array_equal(e, e_raw)


def _replay(times, states, genuine_buffer: int, n_copies: int):
    g0 = genuine_buffer * BUFFER
    first = g0 - n_copies * BUFFER
    g = (times >= g0) & (times < g0 + BUFFER)
    rel, st = times[g] - g0, states[g]
    before = times < first
    parts_t = [times[before]] + [first + c * BUFFER + rel for c in range(n_copies)] + [times[times >= g0]]
    parts_s = [states[before]] + [st for _ in range(n_copies)] + [states[times >= g0]]
    return np.concatenate(parts_t), np.concatenate(parts_s), (first / FS, g0 / FS)


def test_box1_prediction_in_a_replayed_span(tmp_path) -> None:
    """A replayed span (buffers 28-29 copies of 30: 14-15 s) over a 3-refresh hold, with phantom
    flicker in a hold just after the span. The box-1 prediction is exact; the old pairing rule
    (`predict_diode_edges`, box interval 1) puts the held frame's toggle a refresh late."""
    start = 5.0
    k_hold = int((14.4 - start) / R)
    k_after = int((15.2 - start) / R)
    fi, display, fall, rise, edges = _box1(
        start=start, holds={k_hold: 3, k_after: 3}, phantom_pairs={k_after: [1]}
    )
    t, s = _events(edges)
    t, s, iv = _replay(t, s, genuine_buffer=30, n_copies=2)
    path = _write(tmp_path / "replay_box1.h5", t, s)
    raw = npc_sync.SyncDataset(path, on_invalid="ignore")
    assert raw.invalid_intervals == (iv,)
    patch = P.build_sync_patch(raw, [P.StimBlock(0, fi, 1, "synthetic")], builder="test")
    (rec,) = patch.diode_blocks
    assert rec.status == "cleaned" and rec.n_phantom_flicker == 2
    (span,) = patch.lines["stim_photodiode"]
    assert span.status == "reconstructed", span.reason
    pd = npc_sync.SyncDataset(path, patch=patch)
    e, q = _diode(pd)
    truth = display + np.where(np.arange(len(display)) % 2 == 0, LAT_R, LAT_F)
    assert len(e) == len(truth)
    assert np.max(np.abs(e - truth)) < 0.3e-3
    # the old rule, on the same frames: one toggle is a refresh off (the held frame's)
    rec_v = P.reconcile_vsyncs(raw.vsync_times_in_blocks[0], fi, raw.invalid_intervals)
    dr = raw.get_rising_edges("stim_photodiode", units="seconds")
    df = raw.get_falling_edges("stim_photodiode", units="seconds")
    sp = (iv[0] - 1 / FS, iv[1])
    rise_old, fall_old, _ = P.predict_diode_edges(rec_v.times, dr, df, sp, 1)
    old = np.sort(np.r_[rise_old, fall_old])
    inside = truth[P.in_intervals(truth, [sp])]
    err_old = np.array([np.min(np.abs(old - x)) if len(old) else np.inf for x in inside])
    assert err_old.max() > 0.5 * R


def test_predict_diode_edges_ignores_copies_of_other_spans() -> None:
    """Box interval 3, two replayed spans 1 s apart: the 3 s window after the first span used to
    include the second span's copies (here with the opposite toggle phase, as copies of a later
    buffer can have), which failed the polarity check (DRpilot_796012_20250625, 3310 s)."""
    fi = np.full(5999, R)
    ft = 10.0 + np.r_[0, np.cumsum(fi)]
    tog = ft[::3]
    pol = np.arange(len(tog)) % 2 == 0
    e = tog + np.where(pol, LAT_R, LAT_F)
    s1, s2 = (20.0, 20.5), (21.5, 22.0)
    copies = P.in_intervals(e, [s2])
    e_rec = np.r_[e[~copies], e[copies]]
    p_rec = np.r_[pol[~copies], ~pol[copies]]
    o = np.argsort(e_rec)
    e_rec, p_rec = e_rec[o], p_rec[o]
    r, f = e_rec[p_rec], e_rec[~p_rec]
    _, _, old = P.predict_diode_edges(ft, r, f, s1, 3)
    assert any("wrong polarity" in x for x in old.failures())
    _, _, new = P.predict_diode_edges(ft, r, f, s1, 3, exclude=[s2])
    assert new.failures() == []
