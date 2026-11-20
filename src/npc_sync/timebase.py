"""The timebase of a sync file: which stretch of true time each recorded acquisition buffer
("chunk") holds.

Background. The sync program writes fixed-size buffers (50,000 samples = 0.5 s at 100 kHz) and
numbers samples by their position in the file. Two faults break the assumption that sample
number = time:

- dropped chunks: k chunks are never written, with no gap in sample numbers, so everything
  recorded after the drop is early by k chunks
- replayed chunks: a run of n identical chunks, copies of the chunk that follows the run
  (`npc_sync.sync.find_replayed_buffers`). Each copy takes up one chunk of sample numbers
  without holding new data, so a run of n copies makes up for n dropped chunks

They are one mechanism: a run of copies follows single-chunk drops made up to several
seconds earlier, and the data between a drop and the run is displaced (recorded early). Drops
without copies shift the rest of the file. Model: recorded chunk p holds true chunk
`p + s(p)`, where the shift state s rises by k at a drop of k chunks and falls by 1 at each
copy.

Evidence (npc_sessions experiments/sync-timebase-drops): camera exposure lines are strictly
periodic and a chunk is not a whole number of camera periods (30 periods minus 12-26 us), so
the camera phase of a chunk is offset by s x that slip. Chunk-to-chunk phase differences give
an integer jump at every chunk boundary: -1.00 at every copy, +k at drops, 0 elsewhere. Barcode
onset intervals (`find_barcode_discontinuities`), which see a drop as a short interval, are
regular after the remap.

`find_chunk_jumps` measures the jumps, `build_timebase` turns jumps + replayed runs into a
`Timebase`, and `SyncDataset(path, timebase=...)` (or a `SyncPatch` carrying one) returns
events on the corrected ("true") sample clock: displaced data moved to where it belongs, copies
removed, and the intervals no recorded chunk covers (`Timebase.dropped`) reported as
`SyncDataset.invalid_intervals`, to be reconstructed like any other missing data.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SAMPLES = 50_000
"""Acquisition buffer size assumed when a file has no replayed runs to infer it from (every
drop located so far is on this grid: npc_sessions experiments/sync-timebase-drops)."""

REGULAR_INTERVAL_TOLERANCE = 0.25
"""An interval on a periodic line is regular if it is within this fraction of the median
interval (4 ms at 60 Hz): one period, whatever the phase step. Chunks with any irregular
interval (camera start/stop, startup PWM section, a skipped exposure) give no phase. It has to
allow the step of a large drop: 19 chunks is 0.5 ms on a 60 Hz camera (with 0.1 ms, drops of
7-19 chunks at 7 replayed runs were not seen: npc_sessions
experiments/sync-timebase-drops/survey_summary_v1.log)."""

MIN_CHUNK_EDGE_FRACTION = 0.6
"""A chunk gives a phase for a line only if it has at least this fraction of the edges a
full chunk would have."""

MIN_SLIP_SAMPLES = 0.5
"""A line is used only if a chunk differs from a whole number of its periods by at least this
many samples (5 us at 100 kHz; observed 12-26 us for 60 Hz cameras, with a noise of 1-2 us on
the difference of two chunk means)."""

DRIFT_WINDOW_CHUNKS = 201
"""Rolling-median window that removes the slow drift of a line's clock against the sync clock
from chunk-to-chunk phase differences."""

JUMP_MAX_FRACTION = 0.25
"""The combined jump of a boundary (median over lines, in chunks) must be within this of an
integer (observed <= 0.18 over ~60,000 boundaries in 5 files)."""

JUMP_LINE_TOLERANCE = 0.35
"""A line agrees with the combined jump if its own estimate is within this of it."""

JUMP_MIN_AGREEING_SOURCES = 2
"""A boundary is observed only if lines from at least this many sources (cameras) agree with
the combined jump."""

JUMP_MIN_AGREEING_FRACTION = 0.6
"""... and at least this fraction of the lines that have an estimate there."""


@dataclasses.dataclass(frozen=True)
class ChunkJumps:
    """Jump of the shift state at every chunk boundary, measured on periodic lines.

    Index b is the boundary between chunks b - 1 and b (index 0 is unused)."""

    chunk_samples: int
    value: npt.NDArray[np.float64]
    """Median over lines of the phase step in chunks (NaN: no line has an estimate)."""
    jump: npt.NDArray[np.int64]
    """`value` rounded (0 where NaN)."""
    observed: npt.NDArray[np.bool_]
    """The jump is trusted: near an integer and enough lines from enough sources agree."""
    n_lines: npt.NDArray[np.int64]
    lines: tuple[str, ...]
    """Lines used (label:edge kind)."""
    slip_samples: tuple[float, ...]
    """Per line: whole periods in a chunk x period - chunk, in samples."""

    @property
    def n_chunks(self) -> int:
        return len(self.value)


def chunk_phase(
    edge_samples: npt.ArrayLike,
    n_chunks: int,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Mean phase residual (samples) of a periodic line in every chunk, and what a shift of one
    chunk does to it.

    The residual of an edge is its time minus period x its index. Chunks with too few edges or
    an irregular interval are NaN, and the residual restarts after an irregular interval, so
    only differences between consecutive non-NaN chunks are meaningful.

    Returns (phase per chunk, period in samples, slip in samples: the phase step of data
    recorded one chunk early). (all-NaN, nan, nan) if the line can't be used.

    >>> t = np.round(np.arange(3000) * 1666.58).astype(np.int64)  # 60.0003 Hz
    >>> dropped = np.r_[t[t < 2_000_000], t[t >= 2_100_000] - 100_000]  # 2 chunks missing
    >>> phase, period, slip = chunk_phase(dropped, 90)
    >>> round(period, 2), abs(slip - (30 * 1666.58 - 50_000)) < 0.1
    (1666.58, True)
    >>> step = np.diff(phase)
    >>> int(np.nanargmax(np.abs(step))) + 1, round(float(np.nanmax(np.abs(step))) / abs(slip))
    (40, 2)
    """
    t = np.sort(np.asarray(edge_samples, dtype=np.int64))
    out = np.full(n_chunks, np.nan)
    if t.size < 100:
        return out, np.nan, np.nan
    dt = np.diff(t)
    p0 = float(np.median(dt))
    if p0 <= 0:
        return out, np.nan, np.nan
    regular = np.abs(dt - p0) < REGULAR_INTERVAL_TOLERANCE * p0
    if regular.sum() < 100:
        return out, np.nan, np.nan
    period = float(dt[regular].mean())
    n_per_chunk = round(chunk_samples / period)
    slip = n_per_chunk * period - chunk_samples
    if n_per_chunk < 5 or abs(slip) < MIN_SLIP_SAMPLES:
        return out, period, slip
    # residual: cumulative deviation of regular intervals from the period
    u = np.r_[0.0, np.cumsum(np.where(regular, dt - period, 0.0))]
    chunk = t // chunk_samples
    in_range = chunk < n_chunks
    edge_ok = np.r_[True, regular] & np.r_[regular, True]
    count = np.bincount(chunk[in_range], minlength=n_chunks)
    n_bad = np.bincount(chunk[in_range], weights=~edge_ok[in_range], minlength=n_chunks)
    total = np.bincount(chunk[in_range], weights=u[in_range], minlength=n_chunks)
    ok = (count >= MIN_CHUNK_EDGE_FRACTION * n_per_chunk) & (n_bad == 0)
    out[ok] = total[ok] / count[ok]
    return out, period, slip


def _rolling_nanmedian(
    x: npt.NDArray[np.float64], window: int, min_count: int
) -> npt.NDArray[np.float64]:
    """Centered rolling median ignoring NaN; NaN where fewer than `min_count` values.

    >>> _rolling_nanmedian(np.array([1.0, np.nan, 3.0, 10.0, 5.0]), 3, 2).tolist()
    [nan, 2.0, 6.5, 5.0, 7.5]
    """
    half = window // 2
    padded = np.r_[np.full(half, np.nan), x, np.full(half, np.nan)]
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    count = np.isfinite(view).sum(axis=1)
    out = np.full(len(x), np.nan)
    ok = count >= min_count
    if ok.any():
        out[ok] = np.nanmedian(view[ok], axis=1)
    return out


def find_chunk_jumps(
    edges: Mapping[str, npt.ArrayLike],
    total_samples: int,
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    sources: Mapping[str, str] | None = None,
) -> ChunkJumps:
    """Jump of the shift state (in chunks) at every chunk boundary, from periodic lines.

    Parameters
    ----------
    edges
        Line name -> edge sample indices as recorded (one edge kind per entry, e.g.
        'beh_cam_exposing:rising'). Lines that aren't periodic enough are skipped.
    total_samples
        Length of the recording.
    sources
        Line name -> independent source (e.g. the camera). Default: the name up to ':'.
        Rising and falling edges of one camera share a clock, so they count as one source.

    A drop of 2 chunks at 20 s seen by two cameras:

    >>> def cam(period):
    ...     t = np.round(np.arange(3000) * period).astype(np.int64)
    ...     return np.r_[t[t < 2_000_000], t[t >= 2_100_000] - 100_000]
    >>> j = find_chunk_jumps({'a:rising': cam(1666.58), 'b:rising': cam(1666.62)}, 4_500_000)
    >>> np.flatnonzero(j.jump).tolist(), j.jump[40].item(), bool(j.observed[40])
    ([40], 2, True)
    """
    n_chunks = int(np.ceil(total_samples / chunk_samples))
    names: list[str] = []
    slips: list[float] = []
    rows: list[npt.NDArray[np.float64]] = []
    for name, samples in edges.items():
        phase, _, slip = chunk_phase(samples, n_chunks, chunk_samples)
        if not np.isfinite(phase).any():
            continue
        step = np.full(n_chunks, np.nan)
        step[1:] = np.diff(phase)
        drift = _rolling_nanmedian(step, DRIFT_WINDOW_CHUNKS, min_count=20)
        rows.append((step - drift) / slip)
        names.append(name)
        slips.append(float(slip))
    if not rows:
        return ChunkJumps(
            chunk_samples,
            np.full(n_chunks, np.nan),
            np.zeros(n_chunks, dtype=np.int64),
            np.zeros(n_chunks, dtype=bool),
            np.zeros(n_chunks, dtype=np.int64),
            (),
            (),
        )
    d = np.array(rows)
    finite = np.isfinite(d)
    n_lines = finite.sum(axis=0)
    value = np.full(n_chunks, np.nan)
    has = n_lines > 0
    value[has] = np.nanmedian(d[:, has], axis=0)
    jump = np.where(has, np.round(np.nan_to_num(value)), 0).astype(np.int64)
    agree = finite & (np.abs(np.nan_to_num(d) - jump) <= JUMP_LINE_TOLERANCE)
    src = [(sources or {}).get(n, n.split(":")[0]) for n in names]
    n_sources = np.zeros(n_chunks, dtype=np.int64)
    for s in sorted(set(src)):
        n_sources += agree[[i for i, x in enumerate(src) if x == s]].any(axis=0)
    observed = (
        has
        & (np.abs(np.nan_to_num(value) - jump) <= JUMP_MAX_FRACTION)
        & (n_sources >= JUMP_MIN_AGREEING_SOURCES)
        & (agree.sum(axis=0) >= JUMP_MIN_AGREEING_FRACTION * n_lines)
    )
    return ChunkJumps(
        chunk_samples, value, jump, observed, n_lines, tuple(names), tuple(slips)
    )


JumpSource = Literal["periodic_lines", "assumed_at_run", "copy"]


@dataclasses.dataclass(frozen=True)
class TimebaseJump:
    """A change of the shift state at a chunk boundary."""

    sample: int
    """Recorded sample index of the boundary (first sample of the chunk after it)."""
    n_chunks: int
    """+k: k chunks dropped here. -1: the chunk before the boundary is a copy."""
    source: JumpSource
    """'periodic_lines': measured (`find_chunk_jumps`). 'assumed_at_run': the drops that a
    run of copies makes up for were not seen on periodic lines (cameras not running, or
    nothing found), so they are assumed to be at the start of the run, i.e. the copies
    overwrite the data they replace. 'copy': a replayed chunk."""
    value: float | None = None
    """Measured jump in chunks before rounding (None if not measured)."""


@dataclasses.dataclass(frozen=True)
class Timebase:
    """Map from recorded sample numbers to true sample numbers for one sync file.

    >>> tb = build_timebase(1_000_000, drops={10: 2})  # 2 chunks dropped at 5.0 s
    >>> tb.segments, tb.dropped, tb.total_samples_true
    (((0, 500000, 0), (500000, 1000000, 100000)), ((500000, 600000),), 1100000)
    >>> [x.tolist() for x in tb.to_true([499_999, 500_000])]
    [[499999, 600000], [True, True]]

    A run of 2 copies at 6.0-7.0 s, with the two drops they make up for at 4.0 and 5.0 s:
    data recorded at 4.0-5.0 s is 1 chunk early, 5.0-6.0 s is 2 chunks early, the copies are
    removed, and true 4.0-4.5 s and 5.5-6.0 s are missing.

    >>> tb = build_timebase(1_000_000, drops={8: 1, 10: 1}, replayed=[(600_000, 700_000)])
    >>> tb.segments
    ((0, 400000, 0), (400000, 500000, 50000), (500000, 600000, 100000), (700000, 1000000, 0))
    >>> tb.dropped, tb.total_samples_true
    (((400000, 450000), (550000, 600000)), 1000000)
    >>> true, keep = tb.to_true([450_000, 599_999, 600_000, 699_999, 700_000])
    >>> true[keep].tolist(), keep.tolist()
    ([500000, 699999, 700000], [True, True, False, False, True])
    """

    chunk_samples: int
    total_samples: int
    """Recorded samples."""
    segments: tuple[tuple[int, int, int], ...]
    """(start, stop, shift): recorded samples [start, stop) are at true sample + shift.
    Recorded samples in no segment are copies and are discarded."""
    dropped: tuple[tuple[int, int], ...]
    """True-sample intervals [start, stop) that no recorded chunk holds."""
    jumps: tuple[TimebaseJump, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def is_identity(self) -> bool:
        return (
            len(self.segments) == 1
            and self.segments[0][2] == 0
            and self.segments[0][:2] == (0, self.total_samples)
        )

    @property
    def total_samples_true(self) -> int:
        return self.total_samples + self.net_shift_samples

    @property
    def net_shift_samples(self) -> int:
        """Shift of the end of the recording: samples dropped and not made up for by copies."""
        return self.segments[-1][2] if self.segments else 0

    @property
    def drops(self) -> tuple[TimebaseJump, ...]:
        return tuple(j for j in self.jumps if j.n_chunks > 0)

    @property
    def has_assumed_drops(self) -> bool:
        return any(j.source == "assumed_at_run" for j in self.jumps)

    def to_true(
        self, samples: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Recorded sample numbers -> (true sample numbers, keep). `keep` is False for
        samples in discarded copies (their true value is not meaningful)."""
        s = np.asarray(samples).astype(np.int64)
        starts = np.array([a for a, _, _ in self.segments], dtype=np.int64)
        stops = np.array([z for _, z, _ in self.segments], dtype=np.int64)
        shifts = np.array([d for _, _, d in self.segments], dtype=np.int64)
        i = np.clip(np.searchsorted(starts, s, side="right") - 1, 0, len(starts) - 1)
        keep = (s >= starts[i]) & (s < stops[i])
        return s + shifts[i], keep

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_samples": int(self.chunk_samples),
            "total_samples": int(self.total_samples),
            "segments": [[int(x) for x in seg] for seg in self.segments],
            "dropped": [[int(x) for x in iv] for iv in self.dropped],
            "jumps": [
                {
                    "sample": int(j.sample),
                    "n_chunks": int(j.n_chunks),
                    "source": j.source,
                    "value": None if j.value is None else float(j.value),
                }
                for j in self.jumps
            ],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Timebase:
        return cls(
            chunk_samples=int(d["chunk_samples"]),
            total_samples=int(d["total_samples"]),
            segments=tuple((int(a), int(z), int(s)) for a, z, s in d["segments"]),
            dropped=tuple((int(a), int(z)) for a, z in d["dropped"]),
            jumps=tuple(TimebaseJump(**j) for j in d.get("jumps", [])),
            notes=tuple(d.get("notes", [])),
        )


def build_timebase(
    total_samples: int,
    drops: Mapping[int, int] | None = None,
    replayed: Sequence[tuple[int, int]] = (),
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    jumps: ChunkJumps | None = None,
) -> Timebase:
    """Timebase from drops and replayed runs.

    Parameters
    ----------
    total_samples
        Recorded length of the file.
    drops
        Chunk boundary index (boundary b is at sample b x chunk_samples) -> chunks dropped
        there. Taken from `jumps` (observed positive jumps outside copies) if not given.
    replayed
        Recorded sample intervals of replayed copies (`find_replayed_buffers`): whole chunks.
    jumps
        Measured jumps: the source of `drops` if that is None, and checked against the copies
        (each copy should measure -1).

    A run with more copies than drops seen before it (shift state would go negative) gets the
    missing drops at its own start ('assumed_at_run'): the copies then overwrite the data they
    replace, which is the model used before drops were located.

    >>> tb = build_timebase(1_000_000, replayed=[(600_000, 700_000)])
    >>> tb.segments, tb.dropped
    (((0, 600000, 0), (700000, 1000000, 0)), ((600000, 700000),))
    >>> [(j.sample, j.n_chunks, j.source) for j in tb.jumps]
    [(600000, 2, 'assumed_at_run'), (650000, -1, 'copy'), (700000, -1, 'copy')]
    """
    n_chunks = int(np.ceil(total_samples / chunk_samples))
    notes: list[str] = []
    is_copy = np.zeros(n_chunks, dtype=bool)
    for a, z in sorted(replayed):
        if a % chunk_samples or z % chunk_samples:
            raise ValueError(
                f"replayed interval ({a}, {z}) is not on the {chunk_samples}-sample chunk grid"
            )
        r0, r1 = a // chunk_samples, z // chunk_samples
        is_copy[r0:r1] = True
    # boundary b: between chunks b - 1 and b. Copy boundaries: b - 1 is a copy.
    copy_boundary = np.r_[False, is_copy[:-1]]
    up = np.zeros(n_chunks + 1, dtype=np.int64)
    measured: dict[int, float] = {}
    if drops is not None:
        for b, k in drops.items():
            if k <= 0:
                raise ValueError(f"drops must be positive: {b}: {k}")
            if b < len(copy_boundary) and copy_boundary[b]:
                raise ValueError(f"drop at boundary {b}, which follows a replayed copy")
            up[b] = k
    elif jumps is not None:
        if jumps.chunk_samples != chunk_samples:
            raise ValueError("jumps were measured with a different chunk size")
        nz = np.flatnonzero(jumps.observed & (jumps.jump != 0))
        for b in nz:
            if copy_boundary[b]:
                continue
            if jumps.jump[b] > 0:
                up[b] = jumps.jump[b]
                measured[int(b)] = float(jumps.value[b])
            else:
                notes.append(
                    f"jump of {jumps.jump[b]} chunks at sample {b * chunk_samples} is not "
                    "at a replayed copy: not applied"
                )
    if jumps is not None:
        cb = np.flatnonzero(copy_boundary)
        seen = cb[jumps.observed[cb]]
        wrong = seen[jumps.jump[seen] != -1]
        if wrong.size:
            notes.append(
                f"{wrong.size} of {seen.size} measured copy boundaries are not -1 chunk "
                f"(first at sample {int(wrong[0]) * chunk_samples})"
            )
    # shift state, adding the drops a run needs but that weren't seen, at the run's start
    starts = np.flatnonzero(is_copy & ~np.r_[False, is_copy[:-1]])
    stops = np.flatnonzero(is_copy & ~np.r_[is_copy[1:], False]) + 1
    run_length = dict(zip(starts.tolist(), (stops - starts).tolist(), strict=True))
    out_jumps: list[TimebaseJump] = []
    state = np.zeros(n_chunks, dtype=np.int64)
    s = 0
    for p in range(n_chunks):
        if copy_boundary[p]:
            out_jumps.append(TimebaseJump(p * chunk_samples, -1, "copy"))
            s -= 1
        if up[p]:
            out_jumps.append(
                TimebaseJump(
                    p * chunk_samples, int(up[p]), "periodic_lines", measured.get(p)
                )
            )
            s += int(up[p])
        if p in run_length and s < run_length[p]:
            out_jumps.append(
                TimebaseJump(p * chunk_samples, run_length[p] - s, "assumed_at_run")
            )
            s = run_length[p]
        state[p] = s
    if n_chunks and is_copy[-1]:
        notes.append("recording ends in a replayed copy")
    # segments of kept chunks with one shift
    segments: list[tuple[int, int, int]] = []
    kept = np.flatnonzero(~is_copy)
    if kept.size:
        brk = np.flatnonzero((np.diff(kept) != 1) | (np.diff(state[kept]) != 0))
        first = kept[np.r_[0, brk + 1]]
        last = kept[np.r_[brk, len(kept) - 1]]
        for a, z in zip(first, last, strict=True):
            segments.append(
                (
                    int(a) * chunk_samples,
                    min(int(z + 1) * chunk_samples, total_samples),
                    int(state[a]) * chunk_samples,
                )
            )
    # true intervals not covered
    dropped: list[tuple[int, int]] = []
    prev_stop = 0
    for a, z, d in segments:
        if a + d > prev_stop:
            dropped.append((prev_stop, a + d))
        elif a + d < prev_stop:
            raise ValueError(
                "timebase is not monotonic: a copy before the drop it makes up for"
            )
        prev_stop = z + d
    return Timebase(
        chunk_samples=chunk_samples,
        total_samples=int(total_samples),
        segments=tuple(segments),
        dropped=tuple(dropped),
        jumps=tuple(out_jumps),
        notes=tuple(notes),
    )
