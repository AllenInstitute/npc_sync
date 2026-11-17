"""Reconstruct sync line events inside invalid (replayed) intervals, and carry them in a
`SyncPatch` that `SyncDataset(path, patch=...)` applies.

Background: some sync files contain runs of acquisition buffers that are copies of a later
buffer (`SyncDataset.invalid_intervals`). The copies overwrite real data on every line, but the
timebase after the run is correct. Events inside the copies are not real, so each line needs one
of:

- a reconstruction from an independent source with a known error (vsync from the stim file's
  `frameIntervals`, camera exposures from the camera's fixed period, the digital photodiode
  from reconstructed vsyncs plus the measured local latency), or
- an explicit "unknown" span: edges inside are removed and consumers must treat the span as
  missing data (licks, barcodes, sound, opto, unlabelled lines).

Everything here works on arrays (seconds on the sync clock) so that the reconstructions can be
checked independently of `SyncDataset`. The builder that combines them is `build_sync_patch`.

Gates are the values in the npc_sessions plan (`outputs/sync-frame-timing-upgrade-plan.md`):
per-gap vsync residual <= 0.5 ms with the second-best frame count >= 10 ms worse and an exact
block balance; camera residual <= 0.1 ms with |gap / period - n| <= 0.01.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

PATCH_FORMAT_VERSION = 1

MEASURED: int = 0
"""Edge source: recorded on the sync line (outside any patched span)."""
INTERPOLATED: int = 1
"""Edge source: reconstructed inside a patched span (see the line's `method`)."""
MISSING: int = 2
"""Edge source: no time available (only used in per-frame arrays, never for edges)."""
SOURCE_NAMES = {MEASURED: "measured", INTERPOLATED: "interpolated", MISSING: "missing"}

VSYNC_MAX_GAP_RESIDUAL_SEC = 0.5e-3
"""Max |sync elapsed time - stim elapsed time| across one gap, for the best frame count."""
VSYNC_MIN_SECOND_BEST_MARGIN_SEC = 10e-3
"""The second-best frame count for a gap must be at least this much worse than the best."""
PERIODIC_MAX_RESIDUAL_SEC = 0.1e-3
"""Max residual of integer-period interpolation across a gap (camera lines)."""
PERIODIC_MAX_FRACTION = 0.01
"""Max |gap / period - round(gap / period)| for integer-period interpolation."""
PERIODIC_FIT_WINDOW_SEC = 10.0
"""Seconds of measured edges on each side used to fit the period."""
DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC = 0.5e-3
"""Predicted diode edges after a span must match the measured ones to within the pre-span
baseline error + this."""
DIODE_LATENCY_WINDOW_SEC = 3.0
"""Seconds of measured diode edges on each side of a span used to estimate the latency."""


class SyncFaultError(ValueError):
    """Sync data is invalid in a way that can't be used without a patch."""


class ReconciliationError(ValueError):
    """A reconstruction failed one of its gates."""


# ----------------------------------------------------------------------------- utilities


def in_intervals(
    times: npt.ArrayLike, intervals: Iterable[tuple[float, float]]
) -> npt.NDArray[np.bool_]:
    """Mask of `times` inside any half-open interval [start, stop).

    >>> in_intervals([0.5, 1.0, 1.5, 2.0], [(1.0, 2.0)]).tolist()
    [False, True, True, False]
    """
    t = np.asarray(times, dtype=float)
    mask = np.zeros(t.shape, dtype=bool)
    for a, z in intervals:
        mask |= (t >= a) & (t < z)
    return mask


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Sort and merge overlapping or touching intervals.

    >>> merge_intervals([(3, 4), (0, 1), (1, 2)])
    ((0.0, 2.0), (3.0, 4.0))
    """
    out: list[list[float]] = []
    for a, z in sorted((float(a), float(z)) for a, z in intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], z)
        else:
            out.append([a, z])
    return tuple((a, z) for a, z in out)


def splice_spans(
    edges: npt.ArrayLike,
    intervals: Iterable[tuple[float, float]],
    period: float,
) -> tuple[tuple[float, float], ...]:
    """Extend each invalid interval to cover splice artefacts on a periodic line.

    A buffer boundary can split a pulse, leaving a spurious edge just outside the interval
    that is closer than half a period to its neighbour on the far side (a 3.75 ms pulse in
    DRpilot_836629_20260430, a 3.8 ms interval in DRpilot_847958_20260413). Edges within half a
    period of the interval, whose interval to the next measured edge away from the interval is
    also < half a period, are included. Same rule as the prototype
    (npc_sync/experiments/vsync-frame-count/repair.py).

    Returns the (possibly extended) spans, half-open, merged.

    >>> e = np.r_[np.arange(0, 1, 0.1), 1.31, 1.33, np.arange(1.4, 2, 0.1)]
    >>> splice_spans(e, [(1.0, 1.3)], period=0.1)
    ((1.0, 1.3100000001),)
    """
    e = np.sort(np.asarray(edges, dtype=float))
    half = 0.5 * period
    spans = []
    for a, z in merge_intervals(intervals):
        kept = e[~in_intervals(e, [(a, z)])]
        new_a, new_z = a, z
        j = int(np.searchsorted(kept, z))
        while (
            j + 1 < len(kept) and kept[j] - z < half and kept[j + 1] - kept[j] < half
        ):
            new_z = kept[j] + 1e-10
            j += 1
        j = int(np.searchsorted(kept, a)) - 1
        while j >= 1 and a - kept[j] < half and kept[j] - kept[j - 1] < half:
            new_a = kept[j]
            j -= 1
        spans.append((new_a, new_z))
    return merge_intervals(spans)


# ----------------------------------------------------------------------------- vsyncs


@dataclasses.dataclass(frozen=True)
class GapSolution:
    """How the frames in one gap of a vsync block were solved."""

    span: tuple[float, float]
    """Removed span (invalid interval, extended over splice edges), seconds."""
    frame_before: int
    """Stim frame index of the last measured vsync before the span."""
    n_frames: int
    """Number of stim frames reconstructed inside the span."""
    n_removed: int
    """Number of recorded edges removed from the span (copies + splice edges)."""
    time_before: float
    time_after: float
    residual_sec: float
    """(sync elapsed time) - (stim elapsed time) across the gap, for `n_frames`."""
    second_best_sec: float
    """|residual| for the second-best frame count (NaN if there is none)."""


@dataclasses.dataclass
class VsyncReconciliation:
    """Per-frame vsync times for one stim block, with provenance."""

    times: npt.NDArray[np.float64]
    """One time per stim frame (NaN where missing)."""
    source: npt.NDArray[np.int8]
    """MEASURED / INTERPOLATED / MISSING per frame."""
    gaps: tuple[GapSolution, ...]
    balance: int
    """n_stim_frames - frames accounted for (measured + reconstructed). Must be 0."""
    n_stim_frames: int
    n_recorded: int
    """Number of vsyncs in the block as recorded (including copies)."""
    boundary_gap: bool = False
    """A span touches the first or last vsync of the block, so it can't be anchored on both
    sides."""

    def failures(
        self,
        max_residual_sec: float = VSYNC_MAX_GAP_RESIDUAL_SEC,
        min_margin_sec: float = VSYNC_MIN_SECOND_BEST_MARGIN_SEC,
    ) -> list[str]:
        out = []
        if self.boundary_gap:
            out.append("an invalid span touches the start or end of the block")
        if self.balance != 0:
            out.append(
                f"block balance {self.balance:+d} frames (stim frames - accounted)"
            )
        for g in self.gaps:
            if abs(g.residual_sec) > max_residual_sec:
                out.append(
                    f"gap {g.span[0]:.3f}-{g.span[1]:.3f} s: residual "
                    f"{g.residual_sec * 1e3:+.3f} ms > {max_residual_sec * 1e3:.3f} ms"
                )
            margin = g.second_best_sec - abs(g.residual_sec)
            if not (margin >= min_margin_sec):  # NaN fails
                out.append(
                    f"gap {g.span[0]:.3f}-{g.span[1]:.3f} s: second-best frame count only "
                    f"{margin * 1e3:.1f} ms worse (< {min_margin_sec * 1e3:.1f} ms)"
                )
        return out

    @property
    def max_abs_residual_sec(self) -> float:
        return max((abs(g.residual_sec) for g in self.gaps), default=0.0)


def solve_vsync_gaps(
    vsyncs: npt.ArrayLike,
    frame_intervals: npt.ArrayLike,
    invalid_intervals: Iterable[tuple[float, float]],
) -> VsyncReconciliation:
    """Match a block of recorded vsync times to its stim frames across invalid intervals,
    solving each gap's frame count separately. Does not apply gates: see
    `VsyncReconciliation.failures` and `reconcile_vsyncs`.

    Method (per gap, in time order): remove recorded edges in the span (copies, plus splice
    edges; see `splice_spans`), then choose the number of stim frames n inside the gap that
    minimises |dt_sync - dt_stim(n)|, where dt_sync is the time between the measured vsyncs on
    either side and dt_stim(n) = tf[k + 1 + n] - tf[k] (tf = [0, cumsum(frameIntervals)], k =
    stim index of the last measured vsync before the gap). Frames in the gap are placed by
    linear interpolation of the stim clock between the two anchors.

    The first recorded vsync of the block is assumed to be stim frame 0.

    Parameters
    ----------
    vsyncs
        Recorded vsync (falling-edge) times of one block, seconds, including any copies.
    frame_intervals
        The stim file's `frameIntervals` (n_stim_frames - 1 values), seconds.
    invalid_intervals
        Replayed intervals (`SyncDataset.invalid_intervals`). Only those that overlap the block
        are used.
    """
    v_raw = np.sort(np.asarray(vsyncs, dtype=float))
    fi = np.asarray(frame_intervals, dtype=float)
    tf = np.concatenate([[0.0], np.cumsum(fi)])
    n_stim = len(tf)
    period = float(np.median(np.diff(v_raw))) if len(v_raw) > 1 else np.nan
    overlapping = [
        (a, z) for a, z in merge_intervals(invalid_intervals) if a <= v_raw[-1] and z > v_raw[0]
    ]
    spans = splice_spans(v_raw, overlapping, period) if overlapping else ()
    removed = in_intervals(v_raw, spans)
    v = v_raw[~removed]
    boundary_gap = bool(removed[0] or removed[-1]) if len(v_raw) else False
    times = np.full(n_stim, np.nan)
    source = np.full(n_stim, MISSING, dtype=np.int8)
    gaps: list[GapSolution] = []
    if len(v) == 0:
        return VsyncReconciliation(
            times, source, (), n_stim, n_stim, len(v_raw), boundary_gap=True
        )
    inner = [(a, z) for a, z in spans if a > v[0] and z <= v[-1] + 1e-9]
    k = 0  # stim index of v[idx_v]
    idx_v = 0
    overflow = False
    for a, z in inner:
        j_before = int(np.searchsorted(v, a)) - 1
        j_after = j_before + 1
        k_before = k + (j_before - idx_v)
        # measured frames up to and including j_before
        stop = min(k_before + 1, n_stim)
        times[k:stop] = v[idx_v : idx_v + (stop - k)]
        source[k:stop] = MEASURED
        if k_before + 1 >= n_stim:
            overflow = True
            k, idx_v = k_before + 1, j_after
            break
        dt = v[j_after] - v[j_before]
        cand = np.arange(0, int(dt / (0.5 * period)) + 5)
        cand = cand[k_before + 1 + cand < n_stim]
        if cand.size == 0:
            overflow = True
            k, idx_v = k_before + 1, j_after
            break
        err = dt - (tf[k_before + 1 + cand] - tf[k_before])
        order = np.argsort(np.abs(err))
        n_g = int(cand[order[0]])
        resid = float(err[order[0]])
        second = float(np.abs(err[order[1]])) if len(order) > 1 else np.nan
        idx = np.arange(k_before + 1, k_before + 1 + n_g)
        stim_span = tf[k_before + 1 + n_g] - tf[k_before]
        scale = dt / stim_span if stim_span > 0 else 1.0
        times[idx] = v[j_before] + (tf[idx] - tf[k_before]) * scale
        source[idx] = INTERPOLATED
        gaps.append(
            GapSolution(
                span=(float(a), float(z)),
                frame_before=int(k_before),
                n_frames=n_g,
                n_removed=int(in_intervals(v_raw, [(a, z)]).sum()),
                time_before=float(v[j_before]),
                time_after=float(v[j_after]),
                residual_sec=resid,
                second_best_sec=second,
            )
        )
        k = k_before + 1 + n_g
        idx_v = j_after
    remaining = len(v) - idx_v
    if not overflow:
        stop = min(k + remaining, n_stim)
        times[k:stop] = v[idx_v : idx_v + (stop - k)]
        source[k:stop] = MEASURED
    balance = n_stim - (k + remaining)
    return VsyncReconciliation(
        times=times,
        source=source,
        gaps=tuple(gaps),
        balance=int(balance),
        n_stim_frames=n_stim,
        n_recorded=len(v_raw),
        boundary_gap=boundary_gap,
    )


def reconcile_vsyncs(
    vsyncs: npt.ArrayLike,
    frame_intervals: npt.ArrayLike,
    invalid_intervals: Iterable[tuple[float, float]],
    max_residual_sec: float = VSYNC_MAX_GAP_RESIDUAL_SEC,
    min_margin_sec: float = VSYNC_MIN_SECOND_BEST_MARGIN_SEC,
) -> VsyncReconciliation:
    """`solve_vsync_gaps`, raising `ReconciliationError` if any gate fails.

    Examples
    --------
    Frames at 60 Hz with two dropped frames; a replay copies 0.5 s of regular pulses over the
    real ones (which contained a 50 ms interval), so there are 2 extra recorded vsyncs:

    >>> fi = np.full(599, 1 / 60); fi[300] = 3 / 60
    >>> tf = np.r_[0, np.cumsum(fi)] + 10.0
    >>> copies = np.arange(15.0, 15.5, 1 / 60) + 0.001
    >>> recorded = np.sort(np.r_[tf[(tf < 15.0) | (tf >= 15.5)], copies])
    >>> len(recorded) - len(tf)
    2
    >>> r = reconcile_vsyncs(recorded, fi, [(15.0, 15.5)])
    >>> r.balance, r.gaps[0].n_frames, int((r.source == INTERPOLATED).sum())
    (0, 28, 28)
    >>> bool(np.allclose(r.times, tf))
    True
    """
    rec = solve_vsync_gaps(vsyncs, frame_intervals, invalid_intervals)
    failures = rec.failures(max_residual_sec, min_margin_sec)
    if failures:
        raise ReconciliationError("; ".join(failures))
    return rec


def locate_drop(
    vsyncs: npt.ArrayLike,
    frame_intervals: npt.ArrayLike,
    buffer_sec: float = 0.5,
    window: int = 600,
) -> tuple[int, int, float, float]:
    """For a block with fewer vsyncs than stim frames whose sync duration is short by k
    buffers (a timebase drop, see `SyncDataset.timebase_discontinuities`): find the vsync index
    where the missing stim frames are absent, scanning buffer boundaries.

    Ported from the prototype (npc_sync/experiments/vsync-frame-count/repair.py). Policy: drops
    are rejected, not patched (the timebase after the drop is shifted on every line); this is
    for diagnosis only.

    Returns (vsync index of first frame after the drop, n_missing, drop seconds, mean interval
    mismatch in ms).
    """
    v = np.asarray(vsyncs, dtype=float)
    fi = np.asarray(frame_intervals, dtype=float)
    n_stim = len(fi) + 1
    n_missing = n_stim - len(v)
    if n_missing <= 0:
        raise ReconciliationError(f"not a drop: {n_missing} missing frames")
    dur_diff = (v[-1] - v[0]) - fi.sum()
    k = round(-dur_diff / buffer_sec)
    drop_sec = k * buffer_sec
    if k < 1 or abs(-dur_diff - drop_sec) > 0.02:
        raise ReconciliationError(
            f"duration deficit {-dur_diff:.4f} s is not a multiple of {buffer_sec} s"
        )
    vi = np.diff(v)
    best_cost, best_i0 = np.inf, None
    bounds = np.arange(np.ceil(v[0] / buffer_sec), np.floor(v[-1] / buffer_sec) + 1) * buffer_sec
    for t in bounds:
        i0 = int(np.searchsorted(v, t))
        if i0 < window or i0 + window + n_missing + 1 >= len(v):
            continue
        before = np.abs(vi[i0 - window : i0 - 1] - fi[i0 - window : i0 - 1])
        after = np.abs(vi[i0 : i0 + window] - fi[i0 + n_missing : i0 + n_missing + window])
        cost = float(np.mean(before) + np.mean(after))
        if cost < best_cost:
            best_cost, best_i0 = cost, i0
    if best_i0 is None:
        raise ReconciliationError("no buffer boundary far enough from the block ends")
    return best_i0, n_missing, float(drop_sec), best_cost * 1e3


def reconcile_drop(
    vsyncs: npt.ArrayLike, frame_intervals: npt.ArrayLike, i0: int, n_missing: int
) -> VsyncReconciliation:
    """Per-frame times for a block with a located drop (see `locate_drop`): the `n_missing`
    frames from stim index `i0` are MISSING (NaN). Times after the drop are as recorded, i.e.
    shifted early by the dropped duration relative to true time."""
    v = np.asarray(vsyncs, dtype=float)
    n_stim = len(frame_intervals) + 1
    times = np.full(n_stim, np.nan)
    source = np.full(n_stim, MISSING, dtype=np.int8)
    times[:i0] = v[:i0]
    source[:i0] = MEASURED
    times[i0 + n_missing :] = v[i0:]
    source[i0 + n_missing :] = MEASURED
    return VsyncReconciliation(times, source, (), 0, n_stim, len(v))


# ----------------------------------------------------------------------------- periodic lines


@dataclasses.dataclass(frozen=True)
class PeriodicGapSolution:
    span: tuple[float, float]
    status: Literal["reconstructed", "inactive", "failed"]
    period_sec: float = np.nan
    n_inserted: int = 0
    n_removed: int = 0
    residual_sec: float = np.nan
    fraction: float = np.nan
    """|gap / period - n|"""
    fit_residual_max_sec: float = np.nan
    """Max residual of the linear fits on either side (clock regularity)."""
    reason: str = ""

    def failures(
        self,
        max_residual_sec: float = PERIODIC_MAX_RESIDUAL_SEC,
        max_fraction: float = PERIODIC_MAX_FRACTION,
    ) -> list[str]:
        if self.status == "failed":
            return [self.reason]
        if self.status == "inactive":
            return []
        out = []
        if not abs(self.residual_sec) <= max_residual_sec:
            out.append(
                f"residual {self.residual_sec * 1e3:+.4f} ms > {max_residual_sec * 1e3:.3f} ms"
            )
        if not self.fraction <= max_fraction:
            out.append(f"fraction {self.fraction:.4f} > {max_fraction}")
        return out


def interpolate_periodic_edges(
    edges: npt.ArrayLike,
    span: tuple[float, float],
    window_sec: float = PERIODIC_FIT_WINDOW_SEC,
    min_edges_per_side: int = 10,
) -> tuple[npt.NDArray[np.float64], PeriodicGapSolution]:
    """Fill a span on a line with a fixed hardware period (camera exposures) by integer-period
    interpolation between the measured edges on either side.

    The period is the mean of least-squares slopes (time vs index) over `window_sec` of
    measured edges on each side (the median of 10-us-quantised intervals is too coarse). The
    number of periods n across the gap is round(gap / period); the inserted edges are evenly
    spaced between the two anchors. Gates (see `PeriodicGapSolution.failures`) use the residual
    gap - n * period and the fraction |gap / period - n|.

    If the line has no edges within `window_sec` on either side it is taken to be inactive and
    nothing is inserted. If it is active on only one side (started or stopped near the span) the
    solution fails: the count inside is unknown.

    Returns the edges to insert (excluding the anchors) and the solution.

    >>> p = 1 / 60.123
    >>> e = np.arange(0, 2000) * p + 3.0
    >>> span = (15.0, 16.5)
    >>> recorded = e[~in_intervals(e, [span])]
    >>> new, sol = interpolate_periodic_edges(recorded, span)
    >>> sol.status, sol.n_inserted == int(in_intervals(e, [span]).sum())
    ('reconstructed', True)
    >>> bool(np.allclose(new, e[in_intervals(e, [span])]))
    True
    """
    a, z = span
    e = np.sort(np.asarray(edges, dtype=float))
    inside = in_intervals(e, [span])
    n_removed = int(inside.sum())
    kept = e[~inside]
    before = kept[(kept < a) & (kept >= a - window_sec)]
    after = kept[(kept >= z) & (kept < z + window_sec)]
    if before.size == 0 and after.size == 0:
        return np.array([]), PeriodicGapSolution(
            span, "inactive", n_removed=n_removed, reason="no edges within window"
        )
    if before.size < min_edges_per_side or after.size < min_edges_per_side:
        return np.array([]), PeriodicGapSolution(
            span,
            "failed",
            n_removed=n_removed,
            reason=f"line active on one side only ({before.size} edges before, {after.size} after)",
        )
    ib, ia = np.arange(before.size), np.arange(after.size)
    fb, fa = np.polyfit(ib, before, 1), np.polyfit(ia, after, 1)
    p = 0.5 * (fb[0] + fa[0])
    fit_resid = max(
        np.abs(before - np.polyval(fb, ib)).max(), np.abs(after - np.polyval(fa, ia)).max()
    )
    # the line must be running right up to the span on both sides
    if a - before[-1] > 2 * p or after[0] - z > 2 * p:
        return np.array([]), PeriodicGapSolution(
            span,
            "failed",
            period_sec=float(p),
            n_removed=n_removed,
            reason="line stops or starts near the span",
        )
    t0, t1 = before[-1], after[0]
    ratio = (t1 - t0) / p
    n = int(round(ratio))
    new = t0 + np.arange(1, n) * (t1 - t0) / n
    return new, PeriodicGapSolution(
        span,
        "reconstructed",
        period_sec=float(p),
        n_inserted=int(n - 1),
        n_removed=n_removed,
        residual_sec=float(t1 - (t0 + n * p)),
        fraction=float(abs(ratio - n)),
        fit_residual_max_sec=float(fit_resid),
    )


# ----------------------------------------------------------------------------- photodiode


@dataclasses.dataclass(frozen=True)
class DiodeSpanSolution:
    span: tuple[float, float]
    status: Literal["reconstructed", "failed"]
    latency_rising_sec: float = np.nan
    latency_falling_sec: float = np.nan
    """Median toggle-frame vsync -> diode edge latency, per polarity, before the span. The
    digital photodiode's latency depends on polarity (e.g. 23.5 vs 19.8 ms in
    DRpilot_862025_20260805), so one latency can't be used for both."""
    baseline_error_sec: float = np.nan
    """Max |measured - predicted| for measured edges in the window before the span."""
    after_error_sec: float = np.nan
    """Max |measured - predicted| for measured edges in the window after the span."""
    n_polarity_errors_after: int = 0
    """Measured edges after the span whose polarity differs from the predicted toggle's."""
    n_inserted: int = 0
    n_removed: int = 0
    reason: str = ""

    def failures(
        self, max_above_baseline_sec: float = DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC
    ) -> list[str]:
        if self.status == "failed":
            return [self.reason]
        out = []
        if self.n_polarity_errors_after:
            out.append(
                f"{self.n_polarity_errors_after} measured edges after the span have the wrong "
                "polarity for the predicted toggle count"
            )
        if not self.after_error_sec <= self.baseline_error_sec + max_above_baseline_sec:
            out.append(
                f"predicted edges after the span are off by {self.after_error_sec * 1e3:.2f} ms "
                f"(baseline {self.baseline_error_sec * 1e3:.2f} ms + "
                f"{max_above_baseline_sec * 1e3:.1f} ms allowed)"
            )
        return out


def _nearest_index(x: npt.NDArray, ref: npt.NDArray) -> npt.NDArray[np.intp]:
    """Index of the nearest value in sorted `ref` for each x."""
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1)
    return np.where(np.abs(x - ref[j - 1]) <= np.abs(ref[j] - x), j - 1, j)


def predict_diode_edges(
    frame_times: npt.ArrayLike,
    diode_rising: npt.ArrayLike,
    diode_falling: npt.ArrayLike,
    span: tuple[float, float],
    diode_box_frame_interval: int,
    window_sec: float = DIODE_LATENCY_WINDOW_SEC,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], DiodeSpanSolution]:
    """Predict digital photodiode edges inside a span from per-frame vsync times.

    The diode box toggles on frames f with f % `diode_box_frame_interval` == 0 (TaskControl
    `showFrame`), so consecutive toggle frames produce edges of alternating polarity.

    1. Each measured edge in `window_sec` before the span is paired with the latest toggle
       frame whose vsync is at most 2 ms after it. The last one fixes the polarity of every
       toggle (alternating from there).
    2. Latency per polarity = median (edge - toggle vsync) over the pre-span window.
    3. Predicted edge for every toggle = its vsync + the latency for its polarity. Those
       strictly between the last measured edge before the span and the first after it are
       inserted.
    4. Checks: measured edges in the window after the span are compared with the prediction
       (nearest predicted edge): error, and whether the polarity agrees (it doesn't if the
       toggle count inside the span is wrong).

    Parameters
    ----------
    frame_times
        Per-stim-frame vsync times for the block (e.g. `VsyncReconciliation.times`), seconds.
    diode_rising, diode_falling
        Recorded edges on the photodiode line, seconds (copies inside the span are ignored).
    span
        The removed span [start, stop).
    """
    a, z = span
    ft = np.asarray(frame_times, dtype=float)
    box = int(diode_box_frame_interval)
    tt = ft[(np.arange(len(ft)) % box == 0) & np.isfinite(ft)]
    r = np.asarray(diode_rising, dtype=float)
    f = np.asarray(diode_falling, dtype=float)
    all_e = np.concatenate([r, f])
    pol = np.concatenate([np.ones(r.size, bool), np.zeros(f.size, bool)])
    o = np.argsort(all_e)
    all_e, pol = all_e[o], pol[o]
    inside = in_intervals(all_e, [span])
    n_removed = int(inside.sum())
    e, pol = all_e[~inside], pol[~inside]
    pre = (e < a) & (e >= a - window_sec)
    post = (e >= z) & (e < z + window_sec)
    if pre.sum() < 6 or post.sum() < 6 or len(tt) < 2:
        return np.array([]), np.array([]), DiodeSpanSolution(
            span, "failed", n_removed=n_removed, reason="too few measured diode edges near span"
        )
    e_pre, pol_pre = e[pre], pol[pre]
    j_pre = np.searchsorted(tt, e_pre + 0.002, side="right") - 1
    if (j_pre < 0).any():
        return np.array([]), np.array([]), DiodeSpanSolution(
            span, "failed", n_removed=n_removed, reason="diode edges before the first toggle frame"
        )
    # polarity of every toggle from the last pre-span edge
    k0, p0 = int(j_pre[-1]), bool(pol_pre[-1])
    toggle_rising = ((np.arange(len(tt)) - k0) % 2 == 0) == p0
    # paired edges must agree with that polarity (else the pairing is ambiguous)
    n_bad_pre = int((toggle_rising[j_pre] != pol_pre).sum())
    if n_bad_pre > 1:  # allow one (e.g. a dropped frame at the window start)
        return np.array([]), np.array([]), DiodeSpanSolution(
            span,
            "failed",
            n_removed=n_removed,
            reason=f"{n_bad_pre} pre-span diode edges don't alternate with toggle frames",
        )
    lat = e_pre - tt[j_pre]
    lat_r = float(np.median(lat[pol_pre]))
    lat_f = float(np.median(lat[~pol_pre]))
    pred = tt + np.where(toggle_rising, lat_r, lat_f)
    order = np.argsort(pred)
    pred, pred_rising = pred[order], toggle_rising[order]
    pre_err = float(np.max(np.abs(e_pre - pred[_nearest_index(e_pre, pred)])))
    e_post, pol_post = e[post], pol[post]
    j_post = _nearest_index(e_post, pred)
    post_err = float(np.max(np.abs(e_post - pred[j_post])))
    n_bad_post = int((pred_rising[j_post] != pol_post).sum())
    last_before, first_after = e[e < a][-1], e[e >= z][0]
    margin = 0.25 * box / 60
    new = (pred > last_before + margin) & (pred < first_after - margin)
    return (
        pred[new & pred_rising],
        pred[new & ~pred_rising],
        DiodeSpanSolution(
            span,
            "reconstructed",
            latency_rising_sec=lat_r,
            latency_falling_sec=lat_f,
            baseline_error_sec=pre_err,
            after_error_sec=post_err,
            n_polarity_errors_after=n_bad_post,
            n_inserted=int(new.sum()),
            n_removed=n_removed,
        ),
    )


# ----------------------------------------------------------------------------- the patch


SpanStatus = Literal["reconstructed", "unknown"]


@dataclasses.dataclass
class SpanPatch:
    """Replacement edges for one line in one span."""

    span: tuple[float, float]
    status: SpanStatus
    """'reconstructed': `rising`/`falling` replace all recorded edges in the span.
    'unknown': recorded edges in the span are removed and nothing replaces them."""
    rising: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([])
    )
    falling: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([])
    )
    method: str = ""
    """e.g. 'frame_intervals', 'periodic', 'vsync+latency', 'no_edges', 'mask'"""
    error_bound_sec: float | None = None
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Anchors and residuals, JSON-serialisable."""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": list(self.span),
            "status": self.status,
            "rising": [float(x) for x in self.rising],
            "falling": [float(x) for x in self.falling],
            "method": self.method,
            "error_bound_sec": self.error_bound_sec,
            "diagnostics": _jsonable(self.diagnostics),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SpanPatch:
        return cls(
            span=(float(d["span"][0]), float(d["span"][1])),
            status=d["status"],
            rising=np.asarray(d["rising"], dtype=float),
            falling=np.asarray(d["falling"], dtype=float),
            method=d.get("method", ""),
            error_bound_sec=d.get("error_bound_sec"),
            diagnostics=dict(d.get("diagnostics", {})),
            reason=d.get("reason", ""),
        )


@dataclasses.dataclass(frozen=True)
class SyncFault:
    kind: Literal["replay", "timebase_discontinuity"]
    start: float
    stop: float
    shift: float = 0.0


@dataclasses.dataclass
class SyncPatch:
    """Per-line replacement edges inside invalid intervals of one sync file.

    Apply with `SyncDataset(path, patch=patch)`. A line with a `SpanPatch` for every invalid
    interval returns measured edges outside the spans plus the patch's edges inside; spans with
    status 'unknown' return nothing inside (see `SyncDataset.get_unknown_intervals`). A line
    without patches raises on edge queries (unless `on_invalid` is relaxed).

    The patch is keyed to the sync file's content hash (`SyncDataset.content_hash`) and records
    the builder that made it.
    """

    sync_content_hash: str
    builder: str
    invalid_intervals: tuple[tuple[float, float], ...]
    faults: tuple[SyncFault, ...] = ()
    lines: dict[str, list[SpanPatch]] = dataclasses.field(default_factory=dict)
    """line label -> span patches (sorted by start)."""
    created: str = dataclasses.field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    notes: list[str] = dataclasses.field(default_factory=list)

    # application ---------------------------------------------------------------------- #

    def covers(self, line: str) -> bool:
        """Whether every invalid interval is covered by a span of this line's patch."""
        spans = self.lines.get(line)
        if not spans:
            return not self.invalid_intervals
        merged = merge_intervals(s.span for s in spans)
        return all(
            any(a >= sa - 1e-9 and z <= sz + 1e-9 for sa, sz in merged)
            for a, z in self.invalid_intervals
        )

    def unknown_intervals(self, line: str) -> tuple[tuple[float, float], ...]:
        return tuple(s.span for s in self.lines.get(line, []) if s.status == "unknown")

    def apply(
        self,
        line: str,
        kind: Literal["rising", "falling"],
        recorded: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
        """Recorded edge times (seconds) -> patched times and per-edge source."""
        if not self.covers(line):
            raise SyncFaultError(f"patch does not cover every invalid interval for {line!r}")
        rec = np.asarray(recorded, dtype=float)
        spans = self.lines.get(line, [])
        keep = ~in_intervals(rec, [s.span for s in spans])
        parts = [rec[keep]]
        sources = [np.full(int(keep.sum()), MEASURED, dtype=np.int8)]
        for s in spans:
            new = s.rising if kind == "rising" else s.falling
            parts.append(np.asarray(new, dtype=float))
            sources.append(np.full(len(new), INTERPOLATED, dtype=np.int8))
        t = np.concatenate(parts)
        src = np.concatenate(sources)
        order = np.argsort(t, kind="stable")
        return t[order], src[order]

    # serialisation -------------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": PATCH_FORMAT_VERSION,
            "sync_content_hash": self.sync_content_hash,
            "builder": self.builder,
            "created": self.created,
            "invalid_intervals": [list(x) for x in self.invalid_intervals],
            "faults": [dataclasses.asdict(f) for f in self.faults],
            "lines": {k: [s.to_dict() for s in v] for k, v in self.lines.items()},
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SyncPatch:
        if d.get("format_version") != PATCH_FORMAT_VERSION:
            raise ValueError(f"unsupported patch format {d.get('format_version')!r}")
        return cls(
            sync_content_hash=d["sync_content_hash"],
            builder=d["builder"],
            created=d["created"],
            invalid_intervals=tuple((float(a), float(z)) for a, z in d["invalid_intervals"]),
            faults=tuple(SyncFault(**f) for f in d["faults"]),
            lines={k: [SpanPatch.from_dict(s) for s in v] for k, v in d["lines"].items()},
            notes=list(d.get("notes", [])),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_json(cls, s: str) -> SyncPatch:
        return cls.from_dict(json.loads(s))

    def save(self, path: Any) -> None:
        import npc_io

        npc_io.from_pathlike(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: Any) -> SyncPatch:
        import npc_io

        return cls.from_json(npc_io.from_pathlike(path).read_text())

    def summary(self) -> dict[str, dict[str, int]]:
        """line -> {'reconstructed': n spans, 'unknown': n spans}"""
        out: dict[str, dict[str, int]] = {}
        for line, spans in self.lines.items():
            c = out.setdefault(line, {"reconstructed": 0, "unknown": 0})
            for s in spans:
                c[s.status] += 1
        return out


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ----------------------------------------------------------------------------- builder


@dataclasses.dataclass(frozen=True)
class StimBlock:
    """What the builder needs from one stim file matched to a vsync block."""

    block_index: int
    frame_intervals: npt.NDArray[np.float64]
    diode_box_frame_interval: int | None = None
    """None: the stim file doesn't say, so the photodiode line can't be predicted."""
    name: str = ""


def line_key(line_labels: Sequence[str], bit: int) -> str:
    """Patch key for a bit: its label, or 'bit<n>' if unlabelled."""
    label = line_labels[bit] if bit < len(line_labels) else ""
    return label or f"bit{bit}"


CAMERA_LINE_SUFFIXES = ("_cam_exposing", "_cam_frame_readout")


def build_sync_patch(
    sync: Any,
    stim_blocks: Sequence[StimBlock] = (),
    builder: str = "",
    periodic_lines: Sequence[str] | None = None,
) -> SyncPatch:
    """Build a `SyncPatch` for a sync file with invalid intervals.

    - `vsync_stim`: for each span inside a block with a matched stim file, frames are solved
      per gap from `frameIntervals` (`reconcile_vsyncs`); rising edges are placed at the
      reconstructed falling edge minus the median measured pulse width in the block. Spans in
      blocks without a stim file, or that fail a gate, are 'unknown'. Spans outside every
      block with no recorded vsync edges are reconstructed as 'no_edges'.
    - `stim_photodiode`: predicted from reconstructed vsyncs (`predict_diode_edges`) where the
      stim file gives `diodeBoxFrameInterval`; else 'unknown'.
    - camera lines (`*_cam_exposing`, `*_cam_frame_readout`, or `periodic_lines`): integer-period
      interpolation (`interpolate_periodic_edges`), checked against its gates; 'unknown' if a
      gate fails.
    - `stim_running`: 'no_edges' if no edges were recorded in the span and the level is the
      same on both sides (assumes no stimulus started and stopped within the span); else
      'unknown'.
    - every other line: 'unknown' (licks, rewards, sound, opto, barcodes, unlabelled bits).

    Parameters
    ----------
    sync
        A `SyncDataset` (read with `on_invalid='ignore'` internally, so copies are visible).
    stim_blocks
        Stim data per vsync block index (`SyncDataset.vsync_times_in_blocks` order, unpatched).
    builder
        Identifier of the calling builder (recorded in the patch), e.g. 'npc_sessions 0.0.292'.
    """
    from npc_sync.sync import SyncDataset  # avoid import cycle

    raw = SyncDataset(sync, on_invalid="ignore")
    invalid = tuple(raw.invalid_intervals)
    import importlib.metadata

    patch = SyncPatch(
        sync_content_hash=raw.content_hash,
        builder=builder or f"npc_sync {importlib.metadata.version('npc_sync')}",
        invalid_intervals=invalid,
        faults=tuple(SyncFault("replay", a, z) for a, z in invalid)
        + tuple(SyncFault("timebase_discontinuity", a, z, d) for a, z, d in raw.timebase_discontinuities),
    )
    if raw.timebase_discontinuities:
        patch.notes.append(
            "file has timebase discontinuities: edges after them are shifted; not corrected by this patch"
        )
    if not invalid:
        return patch
    fs = raw.sample_freq
    # in 4 of 6 replays checked, the event at the sample before a copy already carries the
    # copy's state (experiments/sync-patch-manifest/acceptance_local7.log), so every span starts
    # one sample early
    spans_base = tuple((a - 1 / fs, z) for a, z in invalid)

    labels = [line_key(raw.line_labels, bit) for bit in range(len(raw.line_labels))]
    # --- vsync + photodiode ----------------------------------------------------------
    vsync_spans: list[SpanPatch] = []
    diode_spans: list[SpanPatch] = []
    blocks = raw.vsync_times_in_blocks
    vr_all = raw.get_rising_edges("vsync_stim", units="seconds")
    vf_all = raw.get_falling_edges("vsync_stim", units="seconds")
    has_diode = "stim_photodiode" in labels
    if has_diode:
        dr_all = raw.get_rising_edges("stim_photodiode", units="seconds")
        df_all = raw.get_falling_edges("stim_photodiode", units="seconds")
    by_block = {sb.block_index: sb for sb in stim_blocks}
    covered: list[tuple[float, float]] = []
    for bi, block in enumerate(blocks):
        overlapping = [(a, z) for a, z in spans_base if a <= block[-1] and z > block[0]]
        if not overlapping:
            continue
        sb = by_block.get(bi)
        if sb is None:
            for a, z in overlapping:
                vsync_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"block {bi}: no stim file"))
                diode_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"block {bi}: no stim file"))
                covered.append((a, z))
            continue
        rec = solve_vsync_gaps(block, sb.frame_intervals, overlapping)
        failures = rec.failures()
        if failures:
            for a, z in overlapping:
                reason = f"block {bi} ({sb.name}): " + "; ".join(failures)
                vsync_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=reason))
                diode_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=reason))
                covered.append((a, z))
            continue
        # pulse width from measured edges in the block
        widths = []
        in_block_f = vf_all[(vf_all >= block[0]) & (vf_all <= block[-1])]
        j = np.searchsorted(vr_all, in_block_f) - 1
        ok = j >= 0
        widths = in_block_f[ok] - vr_all[j[ok]]
        widths = widths[~in_intervals(in_block_f[ok], [g.span for g in rec.gaps])]
        width = float(np.median(widths))
        for g in rec.gaps:
            idx = np.arange(g.frame_before + 1, g.frame_before + 1 + g.n_frames)
            fall = rec.times[idx]
            vsync_spans.append(
                SpanPatch(
                    g.span,
                    "reconstructed",
                    rising=fall - width,
                    falling=fall,
                    method="frame_intervals",
                    error_bound_sec=VSYNC_MAX_GAP_RESIDUAL_SEC,
                    diagnostics={
                        "block_index": bi,
                        "stim": sb.name,
                        "frame_before": g.frame_before,
                        "n_frames": g.n_frames,
                        "n_removed": g.n_removed,
                        "anchor_residual_sec": g.residual_sec,
                        "second_best_sec": g.second_best_sec,
                        "pulse_width_sec": width,
                    },
                )
            )
            covered.append(g.span)
            if not has_diode:
                continue
            if sb.diode_box_frame_interval is None:
                diode_spans.append(SpanPatch(g.span, "unknown", method="mask", reason="diodeBoxFrameInterval unknown"))
                continue
            rise, fall_d, sol = predict_diode_edges(
                rec.times, dr_all, df_all, g.span, sb.diode_box_frame_interval
            )
            dfail = sol.failures()
            diag = _jsonable(dataclasses.asdict(sol))
            if dfail:
                diode_spans.append(SpanPatch(g.span, "unknown", method="mask", diagnostics=diag, reason="; ".join(dfail)))
            else:
                diode_spans.append(
                    SpanPatch(
                        g.span,
                        "reconstructed",
                        rising=rise,
                        falling=fall_d,
                        method="vsync+latency",
                        error_bound_sec=sol.baseline_error_sec + DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC,
                        diagnostics=diag,
                    )
                )
    # invalid intervals not overlapping any block: vsync + diode reconstructed as 'no edges'
    # if nothing was recorded there, else unknown
    for a, z in spans_base:
        if any(a >= ca - 1e-9 and z <= cz + 1e-9 for ca, cz in merge_intervals(covered)):
            continue
        for spans, line_edges in (
            (vsync_spans, (vr_all, vf_all)),
            (diode_spans, (dr_all, df_all) if has_diode else None),
        ):
            if line_edges is None:
                continue
            n_in = sum(int(in_intervals(x, [(a, z)]).sum()) for x in line_edges)
            if n_in == 0:
                spans.append(SpanPatch((a, z), "reconstructed", method="no_edges", reason="outside every vsync block"))
            else:
                spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"{n_in} edges outside every vsync block"))
    patch.lines["vsync_stim"] = sorted(vsync_spans, key=lambda s: s.span)
    if has_diode:
        patch.lines["stim_photodiode"] = sorted(diode_spans, key=lambda s: s.span)

    # --- periodic lines ---------------------------------------------------------------
    periodic = list(periodic_lines) if periodic_lines is not None else [
        lab for lab in labels if lab.endswith(CAMERA_LINE_SUFFIXES)
    ]
    for line in periodic:
        spans = []
        rising = raw.get_rising_edges(line, units="seconds")
        falling = raw.get_falling_edges(line, units="seconds")
        for a, z in spans_base:
            new_r, sol_r = interpolate_periodic_edges(rising, (a, z))
            new_f, sol_f = interpolate_periodic_edges(falling, (a, z))
            fails = sol_r.failures() + sol_f.failures()
            diag = {"rising": dataclasses.asdict(sol_r), "falling": dataclasses.asdict(sol_f)}
            if not fails and sol_r.status == "inactive" and sol_f.status == "inactive":
                spans.append(SpanPatch((a, z), "reconstructed", method="no_edges", diagnostics=diag, reason="line inactive"))
            elif fails or sol_r.status != sol_f.status:
                spans.append(SpanPatch((a, z), "unknown", method="mask", diagnostics=diag, reason="; ".join(fails) or "rising/falling disagree"))
            else:
                spans.append(
                    SpanPatch(
                        (a, z),
                        "reconstructed",
                        rising=new_r,
                        falling=new_f,
                        method="periodic",
                        error_bound_sec=PERIODIC_MAX_RESIDUAL_SEC,
                        diagnostics=diag,
                    )
                )
        patch.lines[line] = spans

    # --- stim_running -----------------------------------------------------------------
    if "stim_running" in labels:
        spans = []
        sr = raw.get_edges("all", "stim_running", units="seconds")
        bit = raw._line_to_bit("stim_running")
        all_t = raw.get_all_times("seconds")
        states = (raw.get_all_bits() >> bit) & 1
        for a, z in spans_base:
            n_in = int(in_intervals(sr, [(a, z)]).sum())
            i_before = int(np.searchsorted(all_t, a)) - 1
            i_after = int(np.searchsorted(all_t, z))
            same = (
                i_before >= 0
                and i_after < len(all_t)
                and states[i_before] == states[i_after]
            )
            if n_in == 0 and same:
                spans.append(SpanPatch((a, z), "reconstructed", method="no_edges", reason="no edges recorded and same level on both sides"))
            else:
                spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"{n_in} edges recorded in copies; level same on both sides: {same}"))
        patch.lines["stim_running"] = spans

    # --- everything else: unknown, unless the line has no edges anywhere in the file ------
    for line in labels:
        if line in patch.lines:
            continue
        if raw.get_edges("all", line, units="samples").size == 0:
            patch.lines[line] = [
                SpanPatch((a, z), "reconstructed", method="no_edges", reason="no edges anywhere in the file")
                for a, z in spans_base
            ]
            continue
        patch.lines[line] = [
            SpanPatch((a, z), "unknown", method="mask", reason="no reconstruction for this line")
            for a, z in spans_base
        ]
    _extend_unknown_spans_over_splice(raw, patch)
    return patch


def _extend_unknown_spans_over_splice(raw: Any, patch: SyncPatch) -> None:
    """The first event after a copy is compared with the copy's last state, so an edge there
    may be a splice artefact rather than a real transition. For 'unknown' spans (no
    reconstruction validates the edges after the span), include that edge in the span."""
    times = np.asarray(raw.get_all_times("samples"), dtype=np.int64)
    bits = np.asarray(raw.get_all_bits(), dtype=np.int64)
    fs = raw.sample_freq
    for line, spans in patch.lines.items():
        bit = raw._line_to_bit(line)
        for i, sp in enumerate(spans):
            if sp.status != "unknown":
                continue
            j = int(np.searchsorted(times, sp.span[1] * fs - 1e-6))
            if j == 0 or j >= len(times):
                continue
            if ((bits[j] >> bit) & 1) != ((bits[j - 1] >> bit) & 1):
                spans[i] = dataclasses.replace(
                    sp,
                    span=(sp.span[0], (times[j] + 1) / fs),
                    reason=(sp.reason + "; " if sp.reason else "")
                    + "extended over an ambiguous edge at the first event after the copy",
                )
