"""Timebase discontinuities (dropped/inserted sync data) detected from barcode onset intervals.

See outputs/sync-dropped-data.md.
"""

import pathlib

import numpy as np
import pytest

import npc_sync

PERIOD = 31.0
ONSETS = np.arange(12) * PERIOD + 10.0  # 10 ... 351
TOTAL = 400.0


def find(onsets, **kwargs):
    kwargs.setdefault("total_seconds", TOTAL)
    return npc_sync.find_barcode_discontinuities(onsets, **kwargs)


def test_regular_barcodes() -> None:
    assert find(ONSETS) == ()


def test_dropped_buffers() -> None:
    for drop in (0.5, 1.0, 2.5):
        late = np.r_[ONSETS[:6], ONSETS[6:] - drop]
        ((start, stop, shift),) = find(late)
        assert (start, stop) == (ONSETS[5], ONSETS[6] - drop)
        assert shift == pytest.approx(-drop)


def test_inserted_data() -> None:
    late = np.r_[ONSETS[:6], ONSETS[6:] + 0.5]
    assert find(late)[0][2] == pytest.approx(0.5)


def test_jitter_is_not_a_discontinuity() -> None:
    rng = np.random.default_rng(0)
    assert find(ONSETS + rng.normal(0, 0.0003, ONSETS.size)) == ()


def test_truncated_first_and_last_barcodes_are_ignored() -> None:
    # first starts 0.4 s after recording start, last 0.3 s before its end: both may be truncated
    onsets = np.r_[0.4, ONSETS[1:-1], TOTAL - 0.3]
    assert find(onsets) == ()


def test_missing_barcode_is_not_a_discontinuity() -> None:
    assert find(np.delete(ONSETS, 5)) == ()


def test_drop_across_missing_barcode() -> None:
    late = np.r_[ONSETS[:6], ONSETS[6:] - 1.0]
    ((_, _, shift),) = find(np.delete(late, 5))
    assert shift == pytest.approx(-1.0)


def test_barcodes_near_invalid_intervals_are_excluded() -> None:
    # barcode 6 is a partial copy (mid-barcode edge, +272 ms late) next to a replayed interval
    onsets = ONSETS.copy()
    onsets[6] += 0.272
    assert find(onsets) != ()
    assert find(onsets, invalid_intervals=((onsets[6] - 3.5, onsets[6] - 0.0),)) == ()


def test_too_few_or_no_barcodes() -> None:
    assert find([]) == ()
    assert find(ONSETS[:3] + np.array([0, 5, 0.0])) == ()  # 3 barcodes: ambiguous
    assert find(ONSETS[:4] + np.array([0, 0, 0, 5.0])) != ()
    assert find(ONSETS[:2]) == ()


def test_frame_display_time_blocks_fails_fast_on_discontinuity() -> None:
    sync = object.__new__(npc_sync.SyncDataset)
    sync.path = pathlib.Path("dropped.h5")
    sync.invalid_intervals = ()
    sync.timebase_discontinuities = ((2458.56, 2488.58, -1.0),)
    with pytest.raises(AssertionError, match="2458.560-2488.580 s"):
        _ = sync.frame_display_time_blocks


DATA = pathlib.Path(__file__).parents[1] / "experiments" / "vsync-frame-count" / "data"
KNOWN = {
    "DRpilot_836629_20260430": (),
    "DRpilot_839146_20260518": (),
    "DRpilot_847958_20260413": (),
    "DRpilot_856847_20260602": (),  # barcode at 805.04 s is partial: excluded, not flagged
    "DRpilot_856847_20260604": (),
    "DRpilot_862025_20260805": (),
    "DRpilot_840160_20260428": ((2458.56, 2488.58, -1.0),),
}


@pytest.mark.parametrize("session", sorted(KNOWN))
def test_known_sessions(session: str) -> None:
    paths = sorted((DATA / session).glob("*.h5"))
    if not paths:
        pytest.skip(f"local sync file not available for {session}")
    sync = npc_sync.SyncDataset(paths[0])
    assert sync.has_barcodes
    result = sync.timebase_discontinuities
    assert len(result) == len(KNOWN[session])
    for got, expected in zip(result, KNOWN[session]):
        assert got[:2] == pytest.approx(expected[:2], abs=0.01)
        assert got[2] == pytest.approx(expected[2], abs=0.005)
