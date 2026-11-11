"""Detection of replayed sync acquisition buffers.

See outputs/vsync-frame-count-discrepancies.md.
"""

import pathlib

import numpy as np
import pytest

import npc_sync


def _make_events(
    buffer_size: int,
    n_buffers: int,
    copies: dict[int, int],
    events_per_buffer: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random events in each buffer; `copies` maps copy buffer index -> source buffer index."""
    rng = np.random.default_rng(seed)
    rel = [
        np.sort(rng.choice(buffer_size, events_per_buffer, replace=False))
        for _ in range(n_buffers)
    ]
    state = [rng.integers(0, 2**32, events_per_buffer) for _ in range(n_buffers)]
    for copy, source in copies.items():
        rel[copy], state[copy] = rel[source].copy(), state[source].copy()
    times = np.concatenate([r + i * buffer_size for i, r in enumerate(rel)])
    return times, np.concatenate(state)


@pytest.mark.parametrize("buffer_size", [10_000, 50_000, 30_000])
def test_buffer_size_inferred(buffer_size: int) -> None:
    times, bits = _make_events(buffer_size, 40, {10: 13, 11: 13, 12: 13})
    intervals, size = npc_sync.find_replayed_buffers(times, bits)
    assert size == buffer_size
    assert intervals == ((10 * buffer_size, 13 * buffer_size),)


def test_multiple_runs() -> None:
    times, bits = _make_events(50_000, 40, {5: 6, 20: 22, 21: 22})
    intervals, size = npc_sync.find_replayed_buffers(times, bits)
    assert size == 50_000
    assert intervals == ((250_000, 300_000), (1_000_000, 1_100_000))


def test_no_replay() -> None:
    times, bits = _make_events(50_000, 40, {})
    assert npc_sync.find_replayed_buffers(times, bits) == ((), None)
    assert npc_sync.find_replayed_buffers([], []) == ((), None)


def test_trailing_event_difference() -> None:
    """The genuine buffer can have an extra trailing event that the copies lack
    (DRpilot_847958_20260413): its last slot doesn't match, but the copy buffers are
    still reported in full."""
    buffer_size = 50_000
    times, bits = _make_events(buffer_size, 40, {10: 12, 11: 12})
    genuine = (times >= 12 * buffer_size) & (times < 13 * buffer_size)
    extra_time = times[genuine].max() + 1
    i = np.searchsorted(times, extra_time)
    times = np.insert(times, i, extra_time)
    bits = np.insert(bits, i, 12345)
    intervals, size = npc_sync.find_replayed_buffers(times, bits)
    assert size == buffer_size
    assert intervals == ((10 * buffer_size, 12 * buffer_size),)


def test_frame_display_time_blocks_fails_fast_on_replay() -> None:
    sync = object.__new__(npc_sync.SyncDataset)
    sync.path = pathlib.Path("replayed.h5")
    sync.invalid_intervals = ((5247.5, 5257.0),)
    with pytest.raises(AssertionError, match="5247.500-5257.000 s"):
        _ = sync.frame_display_time_blocks


DATA = pathlib.Path(__file__).parents[1] / "experiments" / "vsync-frame-count" / "data"
KNOWN = {
    "DRpilot_836629_20260430": ((5247.5, 5257.0),),
    "DRpilot_839146_20260518": ((5622.0, 5629.0),),
    "DRpilot_856847_20260602": ((801.5, 805.0),),
    "DRpilot_856847_20260604": ((2743.0, 2744.5),),
    "DRpilot_862025_20260805": ((705.0, 706.5),),
    "DRpilot_847958_20260413": ((5706.0, 5707.5),),
    "DRpilot_840160_20260428": (),  # timebase drop, not a replay
}


@pytest.mark.parametrize("session", sorted(KNOWN))
def test_known_sessions(session: str) -> None:
    paths = sorted((DATA / session).glob("*.h5"))
    if not paths:
        pytest.skip(f"local sync file not available for {session}")
    assert npc_sync.SyncDataset(paths[0]).invalid_intervals == KNOWN[session]
