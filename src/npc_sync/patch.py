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

Gates (npc_sessions `outputs/sync-frame-timing-upgrade-plan.md`, re-derived from the P0
manifest): per-gap vsync residual <= 0.2 ms with the second-best frame count >= 10 ms worse and
an exact block balance; camera residual <= 0.1 ms with |gap / period - n| <= 0.01; predicted
photodiode edges in the 0.5 s after a span within the pre-span error + 0.5 ms.
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

PATCH_FORMAT_VERSION = 2
"""2: adds `SyncPatch.removed` (measured edges removed anywhere in the file) and
`SyncPatch.diode_blocks`. Version-1 patches still load."""

MEASURED: int = 0
"""Edge source: recorded on the sync line (outside any patched span)."""
INTERPOLATED: int = 1
"""Edge source: reconstructed inside a patched span (see the line's `method`)."""
MISSING: int = 2
"""Edge source: no time available (only used in per-frame arrays, never for edges)."""
SOURCE_NAMES = {MEASURED: "measured", INTERPOLATED: "interpolated", MISSING: "missing"}

VSYNC_MAX_GAP_RESIDUAL_SEC = 0.2e-3
"""Max |sync elapsed time - stim elapsed time| across one gap, for the best frame count.

Re-derived from the P0 manifest (npc_sessions experiments/sync-patch-manifest/summary.log,
2,971 sync files): in files with < 20 invalid intervals and no timebase discontinuity, 159 of
183 gaps are <= 0.1 ms (144 <= 0.05 ms), one is 0.13 ms, and the rest form a separate cluster
at 0.2-0.45 ms (16 gaps, 7 sessions) plus failures > 0.5 ms. That cluster matches the
unexplained ~0.4 ms-per-buffer residual of multi-run files, so the gate sits between the
modes (was 0.5 ms, which admitted it)."""
VSYNC_MIN_SECOND_BEST_MARGIN_SEC = 10e-3
"""The second-best frame count for a gap must be at least this much worse than the best."""
PERIODIC_MAX_RESIDUAL_SEC = 0.1e-3
"""Max residual of integer-period interpolation across a gap (camera lines)."""
PERIODIC_MAX_FRACTION = 0.01
"""Max |gap / period - round(gap / period)| for integer-period interpolation."""
PWM_WIDTH_TOLERANCE_SEC = 0.25e-3
"""Leading pulses whose width differs from the line's median by more than this are the camera's
startup PWM section (signature) and are excluded from period fits."""
PERIODIC_FIT_WINDOW_SEC = 10.0
"""Seconds of measured edges on each side used to fit the period."""
DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC = 0.5e-3
"""Predicted diode edges after a span must match the measured ones to within the pre-span
baseline error + this."""
DIODE_LATENCY_WINDOW_SEC = 3.0
"""Seconds of measured diode edges before a span used to estimate the latency and the baseline
error (and after it, for the polarity check)."""
DIODE_CHECK_WINDOW_SEC = 0.5
"""Seconds of measured diode edges right after a span compared with the prediction. Short, so
the check tests the continuation of the predicted toggle sequence rather than unrelated latency
outliers seconds later (with a 3 s window, 9/95 spans in the P0 manifest failed on edges
0.2-3 s after the span while the first edges matched to <= 0.1 ms:
npc_sessions experiments/sync-patch-manifest/diode_gate_check.log)."""


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


def startup_pwm_section(
    rising: npt.ArrayLike,
    falling: npt.ArrayLike,
    tolerance_sec: float = PWM_WIDTH_TOLERANCE_SEC,
) -> tuple[tuple[float, float], ...]:
    """Time range of the run of pulses at the start of a line whose width differs from the
    line's median width, or () if the first pulse has the median width.

    Cameras start with a short run of exposures of a distinctive width (MVR's startup
    signature, e.g. 10 x 2 ms then 16 ms on an eye camera in DRpilot_857681_20260720; npc_mvr
    `signature_exposure_duration`). Their period is regular but their falling edges are offset
    from the rest, so they must not be used to fit the falling-edge period. Only the leading run
    counts: readout-line widths vary throughout a recording.

    >>> r = np.arange(30) / 60
    >>> f = r + np.r_[np.full(10, 0.002), np.full(20, 0.016)]
    >>> [(round(a, 3), round(z, 3)) for a, z in startup_pwm_section(r, f)]
    [(0.0, 0.152)]
    >>> startup_pwm_section(r, r + 0.016)
    ()
    """
    r = np.sort(np.asarray(rising, dtype=float))
    f = np.sort(np.asarray(falling, dtype=float))
    if len(r) < 2 or len(f) < 2:
        return ()
    j = np.searchsorted(f, r)  # first falling edge after each rising edge
    next_r = np.r_[r[1:], np.inf]
    ok = j < len(f)
    ok[ok] &= f[j[ok]] < next_r[ok]  # a complete pulse (not paired across a gap)
    ok[ok] &= f[j[ok]] - r[ok] < np.median(np.diff(r))  # nor with a pulse whose rise is missing
    if not ok[0]:
        return ()
    r, fall = r[ok], f[j[ok]]
    odd = np.abs((fall - r) - np.median(fall - r)) > tolerance_sec
    if not odd[0]:
        return ()
    n = int(np.argmin(odd)) if not odd.all() else len(odd)
    return ((float(r[0]), float(fall[n - 1])),)


def interpolate_periodic_edges(
    edges: npt.ArrayLike,
    span: tuple[float, float],
    window_sec: float = PERIODIC_FIT_WINDOW_SEC,
    min_edges_per_side: int = 10,
    fit_exclude: Iterable[tuple[float, float]] = (),
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
    # edges excluded from the fit (a camera's startup PWM section; see
    # `startup_pwm_section`) stay in the data but aren't used as anchors or for the period.
    # If they're next to the span, the "stops or starts near the span" check fails.
    kept = kept[~in_intervals(kept, [(x - 1e-9, y + 1e-9) for x, y in fit_exclude])]
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
    """Max |measured - predicted| for measured edges in the `DIODE_CHECK_WINDOW_SEC` after the
    span."""
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
    exclude: Iterable[tuple[float, float]] = (),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], DiodeSpanSolution]:
    """Predict digital photodiode edges inside a span from per-frame vsync times, for
    `diode_box_frame_interval` > 1. For box interval 1 use `align_diode_toggles` and
    `predict_box1_diode_edges`: the pairing rule in step 1 assumes the vsync -> diode latency
    is shorter than one toggle period, and with box interval 1 it isn't (~20-25 ms vs 16.7 ms),
    so every edge is paired with the next frame and the pairing breaks at each dropped frame
    (that caused all 21 'wrong polarity after the span' failures of box-1 spans in the P0
    manifest: npc_sessions experiments/phantom-flicker/diag_polarity.log).

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
    exclude
        Other spans whose recorded edges are copies: ignored in the windows before and after
        `span` (they used to be compared with the prediction when two replayed spans were less
        than `window_sec` apart).
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
    copies = in_intervals(all_e, list(exclude)) & ~inside
    e, pol = all_e[~inside & ~copies], pol[~inside & ~copies]
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
    n_bad_post = int((pred_rising[j_post] != pol_post).sum())
    check = e_post < z + DIODE_CHECK_WINDOW_SEC
    if not check.any():
        return np.array([]), np.array([]), DiodeSpanSolution(
            span,
            "failed",
            n_removed=n_removed,
            reason=f"no measured diode edges within {DIODE_CHECK_WINDOW_SEC} s after the span",
        )
    post_err = float(np.max(np.abs(e_post[check] - pred[j_post[check]])))
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


# ----------------------------------------------------------------------------- photodiode, box 1
#
# diodeBoxFrameInterval = 1: every frame inverts the diode box (TaskControl `showFrame`), so frame
# k's toggle has polarity p_k = p_0 xor (k odd) and appears at v[k] + L(p_k) (v = vsync falling
# edge, L = per-polarity TTL latency, ~20 ms rising and ~25 ms falling: longer than a refresh).
# A frame held for n >= 2 refreshes (a dropped frame: stim `frameIntervals[k]`) must show no
# toggle until frame k+1's. Edges inside the hold are *phantom flicker*: the display re-presents
# the other buffer during the stall (old npc_sync sync.py: "phantom"/"spontaneous flicker").
# Evidence, 442 cached Task 2 pilot sessions (npc_sessions experiments/phantom-flicker):
# 3,952 phantom edges in 39 box-1 blocks, all in holds of >= 3 refreshes, always an even number
# (state-restoring pairs) and at most n - 1 per hold; none in 203,646 holds of 2 refreshes; all 39
# blocks were rejected by npc_stim's flip-count check. Their amplitude changes over a session:
# below the TTL threshold they don't register, but the next toggle can come up to ~9 ms early.

DIODE_TOGGLE = 0
DIODE_GLITCH = 1
DIODE_PHANTOM_FLICKER = 2
DIODE_UNEXPLAINED = 3
DIODE_OUTSIDE = 4
DIODE_EDGE_CLASSES = {
    DIODE_TOGGLE: "toggle",
    DIODE_GLITCH: "glitch",
    DIODE_PHANTOM_FLICKER: "phantom_flicker",
    DIODE_UNEXPLAINED: "unexplained",
    DIODE_OUTSIDE: "outside",
}
"""Classes of photodiode edges in a box-1 block (`align_diode_toggles`):

- toggle: frame k's toggle edge
- glitch: comparator chatter, consecutive edges < `DIODE_GLITCH_MAX_SEC` apart. Not a display
  change: the display changes at most once per refresh.
- phantom_flicker: state-restoring pairs strictly between the toggles of frame k and k+1, where
  frame k is held for n >= 3 refreshes, at most n - 1 edges (one per missed refresh)
- unexplained: any other edge between the first and last matched toggles
- outside: before the first or after the last matched toggle (e.g. the screen returning to grey)
"""

DIODE_GLITCH_MAX_SEC = 0.05 / 59.95
"""Same threshold as npc_stim's `DEFINITE_SHORT_INTERVAL_THRESHOLD` (0.83 ms)."""
HELD_FRAME_MIN_REFRESHES = 1.5
"""A stim frame interval at least this many refreshes long is a held (dropped) frame."""
DIODE_OFF_GRID_FRACTION = 0.1
"""A stim frame interval more than this fraction of a refresh from a whole number of refreshes is
off the refresh grid: a late `flip()` return at one of its ends (the frame-signal falling edge
and PsychoPy's frame interval are late; the frame was displayed on time)."""
DIODE_LANDMARK_LATENCY_RANGE_SEC = (0.005, 0.035)
"""Latency accepted for a landmark. Block medians: rising 19.5-20.4 ms, falling 23.4-25.9 ms
(survey_blocks.csv); one refresh more (>= 36 ms) is excluded: a late `flip()` return produces that
alias (stim interval k long, diode interval k+1 long)."""
DIODE_LANDMARK_LENGTH_TOL_REFRESHES = 0.5
DIODE_MIN_LANDMARKS = 3
DIODE_MIN_LANDMARK_AGREEMENT = 0.9
DIODE_LANDMARK_AGREEMENT_TOL_SEC = 0.008
"""A landmark agrees if its latency is within this of its polarity's mode. The latency drifts and
can switch between two states ~5.5 ms apart within a block (falling 23 vs 28.5 ms in
DRpilot_674562_20231004 block 6); an alias is a refresh (16.7 ms) away."""
DIODE_MAX_POLARITY_LATENCY_DIFF_SEC = 0.008
"""Rising and falling latencies differ by the TTL threshold asymmetry, never by a refresh."""
DIODE_TOGGLE_FIRST_PASS_TOL_SEC = 0.008
DIODE_TOGGLE_MATCH_TOL_SEC = 0.012
"""A toggle edge must be within this of its expected time: toggles after a hold can be ~9 ms
early (sub-threshold phantom flicker), and after-stall TTL edges late."""
DIODE_MIN_DISPLAY_AFTER_RISE_SEC = 0.015
"""A frame can't be displayed before its frame-signal rising edge + this (npc_stim
`MIN_VSYNC_DIODE_FLIP_SEPARATION_SEC`; >= 20.9 ms in 2.2 M analog-checked toggles). Lower bound of
the second matching pass, for frames whose falling edge is late."""
DIODE_AFTER_STALL_RESIDUAL_SEC = 0.25 / 59.95
"""Toggle residuals beyond this explain a disagreement of the long-interval check
(`DiodeToggleAlignment.long_interval_mismatch`) as a TTL timing deviation."""
DIODE_MIN_SEGMENT_TOGGLES = 3
DIODE_BLOCK_END_PADDING_SEC = 0.5
"""Diode edges up to this long after a block's last vsync are aligned with the block."""
DIODE_BOUNDARY_FRAMES = 1
"""A missing toggle in the first or last this-many frames of a block doesn't fail the check: the
first frame's toggle can be hidden by the transition from the initializing screen (npc_stim
corrects the whole-block offset), the last by the return to the grey screen."""


class DiodeOffsetUndetermined(ReconciliationError):
    """No (or too few, or inconsistent) long stim frame intervals show as long diode intervals:
    the frame <-> edge offset can't be fixed from the diode."""


@dataclasses.dataclass
class DiodeToggleAlignment:
    """Assignment of photodiode edges to the toggle frames of one box-1 block (see
    `align_diode_toggles`)."""

    edge_times: npt.NDArray[np.float64]
    edge_rising: npt.NDArray[np.bool_]
    edge_class: npt.NDArray[np.int8]
    """Per edge: `DIODE_EDGE_CLASSES` key."""
    edge_frame: npt.NDArray[np.intp]
    """Per edge: the frame whose toggle it is (toggle), the held frame whose hold it is in
    (phantom flicker), the frame before it (unexplained), else -1."""
    frame_edge: npt.NDArray[np.intp]
    """Per frame: index of its toggle edge, or -1."""
    frame_rising: npt.NDArray[np.bool_]
    """Per frame: polarity of its toggle."""
    expected: npt.NDArray[np.float64]
    """Per frame: expected toggle time (frame time + local per-polarity latency), seconds."""
    residual: npt.NDArray[np.float64]
    """Per frame: toggle edge - expected (NaN if unmatched)."""
    unobserved: npt.NDArray[np.bool_]
    """Per frame: unmatched, with the expected toggle in or next to an `unobserved` interval."""
    second_pass: npt.NDArray[np.intp]
    """Frames matched only in the second pass (late `flip()` return)."""
    refresh_sec: float
    n_refreshes: npt.NDArray[np.int_]
    """round(frameIntervals / refresh) per frame interval."""
    n_landmarks: int
    n_landmarks_disagree: int
    long_interval_mismatch: dict[str, npt.NDArray[np.intp]]
    """Long-interval check: frame intervals k where stim (>= 1.5 refreshes) and diode
    (latency-corrected toggle-to-toggle interval >= 1.5 refreshes) disagree, by cause:
    'late_return' (stim interval k or a neighbour off the refresh grid), 'after_stall' (toggle k
    or k+1 off its expected time by > `DIODE_AFTER_STALL_RESIDUAL_SEC`), 'other'."""

    def count(self, edge_class: int) -> int:
        return int((self.edge_class == edge_class).sum())

    @property
    def missing_frames(self) -> npt.NDArray[np.intp]:
        """Frames with no toggle edge, excluding unobserved ones."""
        return np.flatnonzero((self.frame_edge < 0) & ~self.unobserved)

    @property
    def interior_missing_frames(self) -> npt.NDArray[np.intp]:
        m = self.missing_frames
        n = len(self.frame_edge)
        return m[(m >= DIODE_BOUNDARY_FRAMES) & (m < n - DIODE_BOUNDARY_FRAMES)]

    @property
    def removable(self) -> npt.NDArray[np.intp]:
        """Indices of edges to remove: glitches and phantom flicker."""
        return np.flatnonzero(
            (self.edge_class == DIODE_GLITCH) | (self.edge_class == DIODE_PHANTOM_FLICKER)
        )

    def failures(self) -> list[str]:
        """Why the block's diode edges can't be explained (empty: they can)."""
        out = []
        if n := self.count(DIODE_UNEXPLAINED):
            out.append(
                f"{n} diode edge(s) are neither a frame's toggle, nor phantom flicker inside a "
                "held frame, nor a comparator glitch"
            )
        if n := len(self.interior_missing_frames):
            out.append(f"{n} toggle frame(s) have no diode edge")
        if n := len(self.long_interval_mismatch["other"]):
            out.append(
                f"{n} frame interval(s) where the stim file and the diode disagree on a held frame"
            )
        return out

    def summary(self) -> dict[str, Any]:
        return {
            **{DIODE_EDGE_CLASSES[c]: self.count(c) for c in DIODE_EDGE_CLASSES},
            "missing_frames": len(self.missing_frames),
            "missing_boundary_frames": len(self.missing_frames)
            - len(self.interior_missing_frames),
            "unobserved_frames": int(self.unobserved.sum()),
            "held_frames": int((self.n_refreshes >= 2).sum()),
            "holds_with_phantom_flicker": int(
                len(np.unique(self.edge_frame[self.edge_class == DIODE_PHANTOM_FLICKER]))
            ),
            "second_pass_toggles": len(self.second_pass),
            "landmarks": self.n_landmarks,
            "landmarks_disagree": self.n_landmarks_disagree,
            "first_frame_rising": bool(self.frame_rising[0]) if len(self.frame_rising) else None,
            "latency_rising_ms": _nanmedian_ms(self.expected - self._frame_times, self.frame_rising),
            "latency_falling_ms": _nanmedian_ms(self.expected - self._frame_times, ~self.frame_rising),
            "long_interval_mismatch": {k: len(v) for k, v in self.long_interval_mismatch.items()},
            "max_abs_toggle_residual_ms": float(np.nanmax(np.abs(self.residual)) * 1e3)
            if np.isfinite(self.residual).any()
            else None,
        }

    _frame_times: npt.NDArray[np.float64] = dataclasses.field(
        default_factory=lambda: np.array([]), repr=False
    )


def _nanmedian_ms(x: npt.NDArray, mask: npt.NDArray) -> float | None:
    v = x[mask & np.isfinite(x)]
    return float(np.median(v) * 1e3) if len(v) else None


def diode_glitch_mask(
    edge_times: npt.ArrayLike, max_sec: float = DIODE_GLITCH_MAX_SEC
) -> npt.NDArray[np.bool_]:
    """Edges to remove as comparator chatter: clusters of consecutive edges less than `max_sec`
    apart. An even cluster is removed whole (a pulse); an odd cluster keeps its last edge (a
    transition with chatter; npc_stim's anomalous-interval removal also kept the last).

    >>> diode_glitch_mask([0.0, 1.0, 1.0001, 1.0002, 2.0, 3.0, 3.0003]).tolist()
    [False, True, True, False, False, True, True]
    """
    e = np.asarray(edge_times, dtype=float)
    out = np.zeros(len(e), bool)
    if len(e) < 2:
        return out
    close = np.diff(e) < max_sec
    i = 0
    while i < len(close):
        if not close[i]:
            i += 1
            continue
        j = i
        while j < len(close) and close[j]:
            j += 1
        n = j - i + 1  # edges i..j
        out[i : j + 1 - (n % 2)] = True
        i = j + 1
    return out


def _diode_landmarks(
    v: npt.NDArray, fi: npt.NDArray, e: npt.NDArray, q: npt.NDArray, R: float,
    unobserved: Sequence[tuple[float, float]],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """(latency, polarity, frame) of each landmark: a diode interval >= 1.5 refreshes whose
    length matches (within half a refresh) a held frame k on the refresh grid, starting within
    `DIODE_LANDMARK_LATENCY_RANGE_SEC` after v[k]: its first edge is frame k's toggle."""
    frac = np.abs(fi / R - np.round(fi / R))
    on = frac <= DIODE_OFF_GRID_FRACTION
    on_nb = on & np.r_[True, on[:-1]] & np.r_[on[1:], True]
    long_k = np.flatnonzero((fi >= HELD_FRAME_MIN_REFRESHES * R) & on_nb & np.isfinite(v[:-1]))
    d = np.diff(e)
    long_j = np.flatnonzero(d >= HELD_FRAME_MIN_REFRESHES * R)
    if unobserved:  # a diode interval across a removed span is not a hold
        long_j = long_j[~np.array([any(a < e[j + 1] and z > e[j] for a, z in unobserved) for j in long_j], bool)]
    vs = v[long_k]
    lo_l, hi_l = DIODE_LANDMARK_LATENCY_RANGE_SEC
    offs, pols, ks = [], [], []
    for j in long_j:
        lo = np.searchsorted(vs, e[j] - hi_l)
        hi = np.searchsorted(vs, e[j] - lo_l, side="right")
        for i in range(lo, hi):
            k = long_k[i]
            if abs(d[j] - fi[k]) < DIODE_LANDMARK_LENGTH_TOL_REFRESHES * R:
                offs.append(e[j] - v[k])
                pols.append(q[j])
                ks.append(k)
    return np.array(offs, float), np.array(pols, bool), np.array(ks, int)


def _diode_landmark_offset(
    v: npt.NDArray, fi: npt.NDArray, e: npt.NDArray, q: npt.NDArray, R: float,
    unobserved: Sequence[tuple[float, float]],
) -> tuple[int, float, float, int, int]:
    """(parity of frames with rising toggles, L_rising, L_falling, n landmarks agreeing, n
    disagreeing). Raises `DiodeOffsetUndetermined`."""
    offs, pols, ks = _diode_landmarks(v, fi, e, q, R, unobserved)
    if len(offs) == 0:
        raise DiodeOffsetUndetermined("no long diode interval matches a held frame")
    lo_l, hi_l = DIODE_LANDMARK_LATENCY_RANGE_SEC
    bins = np.arange(lo_l, hi_l + 0.0005, 0.0005)

    def mode(x: npt.NDArray) -> float:
        hs = np.convolve(np.histogram(x, bins)[0], np.ones(5), mode="same")
        return float(bins[np.argmax(hs)] + 0.00025)

    modes = {pol: mode(offs[pols == pol]) for pol in (True, False) if (pols == pol).any()}
    if len(modes) == 2 and abs(modes[True] - modes[False]) > DIODE_MAX_POLARITY_LATENCY_DIFF_SEC:
        raise DiodeOffsetUndetermined(
            f"rising and falling landmark latencies differ by {abs(modes[True] - modes[False]) * 1e3:.1f} ms"
        )
    fallback = next(iter(modes.values()))
    ref = np.array([modes.get(p, fallback) for p in pols])
    agree = np.abs(offs - ref) <= DIODE_LANDMARK_AGREEMENT_TOL_SEC
    if agree.sum() < DIODE_MIN_LANDMARKS or agree.mean() < DIODE_MIN_LANDMARK_AGREEMENT:
        raise DiodeOffsetUndetermined(
            f"{int(agree.sum())} of {len(offs)} landmarks agree on the latency (need >= "
            f"{DIODE_MIN_LANDMARKS} and >= {DIODE_MIN_LANDMARK_AGREEMENT:.0%})"
        )
    Lr = float(np.median(offs[agree & pols])) if (agree & pols).any() else modes.get(True, fallback)
    Lf = float(np.median(offs[agree & ~pols])) if (agree & ~pols).any() else modes.get(False, fallback)
    # frames with rising toggles: rising landmarks are toggles of such frames
    par = np.where(pols[agree], ks[agree] % 2, 1 - ks[agree] % 2)
    if min(par.mean(), 1 - par.mean()) > 1 - DIODE_MIN_LANDMARK_AGREEMENT:
        raise DiodeOffsetUndetermined("landmarks disagree on which frames have rising toggles")
    return int(np.round(par.mean())), Lr, Lf, int(agree.sum()), int((~agree).sum())


def _match_toggles(
    t: npt.NDArray, p_rising: npt.NDArray, e: npt.NDArray, q: npt.NDArray, tol: float,
    early: npt.NDArray | None = None,
) -> npt.NDArray[np.intp]:
    """frame -> edge index (or -1). Each frame takes the edge of its polarity nearest its
    expected time t within [t - tol - early, t + tol]; an edge claimed by two frames goes to the
    nearer. Frames with non-finite t are not matched."""
    fe = np.full(len(t), -1, np.intp)
    early = np.zeros(len(t)) if early is None else early
    for pol in (True, False):
        ks = np.flatnonzero((p_rising == pol) & np.isfinite(t))
        js = np.flatnonzero(q == pol)
        if len(ks) == 0 or len(js) == 0:
            continue
        ee, tk = e[js], t[ks]
        lo = np.searchsorted(ee, tk - tol - early[ks])
        hi = np.searchsorted(ee, tk + tol, side="right")
        best = np.full(len(ks), -1, np.intp)
        dist = np.full(len(ks), np.inf)
        for off in range(int(np.max(hi - lo))):
            c = lo + off
            d = np.where(c < hi, np.abs(ee[np.clip(c, 0, len(ee) - 1)] - tk), np.inf)
            better = d < dist
            best[better], dist[better] = c[better], d[better]
        m = best >= 0
        if m.any():
            idx = np.flatnonzero(m)[np.lexsort((dist[m], best[m]))]
            dup = np.r_[False, best[idx][1:] == best[idx][:-1]]
            best[idx[dup]] = -1
        fe[ks[best >= 0]] = js[best[best >= 0]]
    return fe


def _segment_correction(
    res: npt.NDArray, p_rising: npt.NDArray, seg: npt.NDArray, n_seg: int
) -> npt.NDArray:
    """Per-frame latency correction: median residual per (segment, polarity), carried forward
    (then backward) into segments with too few matched toggles."""
    out = np.zeros(len(res))
    for pol in (True, False):
        sel = (p_rising == pol) & np.isfinite(res)
        med = np.full(n_seg, np.nan)
        if sel.any():
            s, r = seg[sel], res[sel]
            order = np.argsort(s, kind="stable")
            s, r = s[order], r[order]
            bounds = np.searchsorted(s, np.arange(n_seg + 1))
            for i in range(n_seg):
                a, z = bounds[i], bounds[i + 1]
                if z - a >= DIODE_MIN_SEGMENT_TOGGLES:
                    med[i] = np.median(r[a:z])
        ok = np.flatnonzero(np.isfinite(med))
        if len(ok):
            # nearest earlier segment with a value, else the nearest later one
            j = np.searchsorted(ok, np.arange(n_seg), side="right") - 1
            filled = np.where(j >= 0, med[ok[np.clip(j, 0, None)]], med[ok[0]])
        else:
            filled = np.zeros(n_seg)
        out[p_rising == pol] = filled[seg[p_rising == pol]]
    return out


def align_diode_toggles(
    frame_times: npt.ArrayLike,
    frame_intervals: npt.ArrayLike,
    edge_times: npt.ArrayLike,
    edge_rising: npt.ArrayLike,
    frame_signal_rising: npt.ArrayLike | None = None,
    unobserved: Iterable[tuple[float, float]] = (),
) -> DiodeToggleAlignment:
    """Align the photodiode edges of one box-1 block (diodeBoxFrameInterval = 1) with its
    frames, and classify every edge (`DIODE_EDGE_CLASSES`).

    1. Comparator glitches (consecutive edges < `DIODE_GLITCH_MAX_SEC` apart) are set aside.
    2. Frame <-> edge offset from landmarks (`_diode_landmarks`): with box interval 1 it can't be
       found from regular frames (v[k] + L ~ v[k+1] + L - R). Raises `DiodeOffsetUndetermined`
       if there are fewer than 3 consistent landmarks.
    3. Per-polarity latency, piecewise constant between held frames (it can switch at a stall).
    4. Each frame's toggle: the edge of its polarity nearest its expected time, within
       `DIODE_TOGGLE_MATCH_TOL_SEC`; frames left unmatched get a second pass down to their
       frame-signal rising edge + `DIODE_MIN_DISPLAY_AFTER_RISE_SEC` (late `flip()` return).
    5. Edges between the toggles of frames k and k+1 are phantom flicker if frame k is held for
       n >= 3 refreshes and they are an even number <= n - 1; otherwise unexplained.
    6. Long-interval check (`DiodeToggleAlignment.long_interval_mismatch`).

    Parameters
    ----------
    frame_times
        Per stim frame: vsync falling edge (seconds), measured or reconstructed
        (`VsyncReconciliation.times`). One per stim frame.
    frame_intervals
        The stim file's `frameIntervals`.
    edge_times, edge_rising
        Measured photodiode edges of the block (sorted) and their polarity. Edges inside
        `unobserved` are ignored.
    frame_signal_rising
        Per frame: the frame-signal (vsync line) rising edge before its falling edge, or NaN.
    unobserved
        Intervals without measured edges (replayed spans): frames whose toggle falls there are
        not counted as missing.
    """
    v = np.asarray(frame_times, dtype=float)
    fi = np.asarray(frame_intervals, dtype=float)
    N = len(v)
    if len(fi) != N - 1:
        raise ValueError(f"{N} frame times but {len(fi)} frame intervals")
    if not np.isfinite(v).all():
        raise ValueError("frame times must be finite (reconcile the vsyncs first)")
    spans = list(merge_intervals(unobserved))
    e_all = np.asarray(edge_times, dtype=float)
    q_all = np.asarray(edge_rising, dtype=bool)
    keep_in = ~in_intervals(e_all, spans)
    e_all, q_all = e_all[keep_in], q_all[keep_in]
    R = float(np.median(fi))
    nref = np.round(fi / R).astype(int)
    gl = diode_glitch_mask(e_all)
    keep = np.flatnonzero(~gl)
    e, q = e_all[keep], q_all[keep]
    c_r, Lr, Lf, n_agree, n_dis = _diode_landmark_offset(v, fi, e, q, R, spans)
    p_rising = (np.arange(N) % 2) == c_r
    # a new segment starts after every held frame
    seg = np.concatenate([[0], np.cumsum(fi >= HELD_FRAME_MIN_REFRESHES * R)])
    n_seg = int(seg[-1]) + 1
    base = np.where(p_rising, Lr, Lf)
    corr = np.zeros(N)
    for tol in (DIODE_TOGGLE_FIRST_PASS_TOL_SEC, DIODE_TOGGLE_MATCH_TOL_SEC):
        t = v + base + corr
        fe = _match_toggles(t, p_rising, e, q, tol)
        res = np.where(fe >= 0, e[np.clip(fe, 0, None)] - t, np.nan)
        corr = corr + _segment_correction(res, p_rising, seg, n_seg)
    t = v + base + corr
    fe = _match_toggles(t, p_rising, e, q, DIODE_TOGGLE_MATCH_TOL_SEC)
    second = np.array([], np.intp)
    if frame_signal_rising is not None:
        vr = np.asarray(frame_signal_rising, dtype=float)
        un = np.flatnonzero(fe < 0)
        used = np.zeros(len(e), bool)
        used[fe[fe >= 0]] = True
        early = t[un] - DIODE_TOGGLE_MATCH_TOL_SEC - (vr[un] + DIODE_MIN_DISPLAY_AFTER_RISE_SEC)
        early = np.where(np.isfinite(early), np.clip(early, 0, None), 0.0)
        free = np.flatnonzero(~used)
        fe2 = _match_toggles(t[un], p_rising[un], e[free], q[free], DIODE_TOGGLE_MATCH_TOL_SEC, early)
        take = fe2 >= 0
        fe[un[take]] = free[fe2[take]]
        second = un[take]
    # toggles must be in frame order: unmatch any out-of-order pair (rare)
    m = np.flatnonzero(fe >= 0)
    bad = np.flatnonzero(np.diff(fe[m]) <= 0)
    fe[m[bad]] = -1
    fe[m[bad + 1]] = -1
    m = np.flatnonzero(fe >= 0)
    residual = np.where(fe >= 0, e[np.clip(fe, 0, None)] - t, np.nan)
    near_span = in_intervals(t, [(a - DIODE_TOGGLE_MATCH_TOL_SEC, z + DIODE_TOGGLE_MATCH_TOL_SEC) for a, z in spans])
    unobserved_frames = (fe < 0) & near_span
    # classify
    M = len(e)
    cls = np.full(M, DIODE_UNEXPLAINED, np.int8)
    ef = np.full(M, -1, np.intp)
    cls[fe[m]] = DIODE_TOGGLE
    ef[fe[m]] = m
    if len(m):
        cls[: fe[m[0]]] = DIODE_OUTSIDE
        cls[fe[m[-1]] + 1 :] = DIODE_OUTSIDE
        for g in np.flatnonzero(np.diff(fe[m]) > 1):
            ka, kb = m[g], m[g + 1]
            between = np.arange(fe[ka] + 1, fe[kb])
            ef[between] = ka
            if (
                kb == ka + 1
                and nref[ka] >= 3
                and len(between) <= nref[ka] - 1
                and len(between) % 2 == 0
            ):
                cls[between] = DIODE_PHANTOM_FLICKER
    else:
        cls[:] = DIODE_OUTSIDE
    # long-interval check on latency-corrected toggle-to-toggle intervals
    both = np.flatnonzero((fe[:-1] >= 0) & (fe[1:] >= 0))
    ct = np.where(fe >= 0, e[np.clip(fe, 0, None)] - (t - v), np.nan)
    d_long = (ct[both + 1] - ct[both]) >= HELD_FRAME_MIN_REFRESHES * R
    s_long = fi[both] >= HELD_FRAME_MIN_REFRESHES * R
    mm = both[d_long != s_long]
    off_grid = np.r_[np.abs(fi / R - np.round(fi / R)) > DIODE_OFF_GRID_FRACTION, False]
    big = np.abs(np.nan_to_num(residual)) > DIODE_AFTER_STALL_RESIDUAL_SEC
    late = off_grid[mm] | off_grid[np.clip(mm - 1, 0, None)] | off_grid[mm + 1]
    after = ~late & (big[mm] | big[mm + 1])
    long_mismatch = {"late_return": mm[late], "after_stall": mm[after], "other": mm[~late & ~after]}
    # back to the caller's edge indices (glitches included)
    edge_class = np.full(len(e_all), DIODE_GLITCH, np.int8)
    edge_class[keep] = cls
    edge_frame = np.full(len(e_all), -1, np.intp)
    edge_frame[keep] = ef
    return DiodeToggleAlignment(
        edge_times=e_all,
        edge_rising=q_all,
        edge_class=edge_class,
        edge_frame=edge_frame,
        frame_edge=np.where(fe >= 0, keep[np.clip(fe, 0, None)], -1),
        frame_rising=p_rising,
        expected=t,
        residual=residual,
        unobserved=unobserved_frames,
        second_pass=second,
        refresh_sec=R,
        n_refreshes=nref,
        n_landmarks=n_agree,
        n_landmarks_disagree=n_dis,
        long_interval_mismatch=long_mismatch,
        _frame_times=v,
    )


def predict_box1_diode_edges(
    alignment: DiodeToggleAlignment,
    span: tuple[float, float],
    n_removed: int = 0,
    window_sec: float = DIODE_LATENCY_WINDOW_SEC,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], DiodeSpanSolution]:
    """Photodiode edges inside a replayed span of a box-1 block, from a block alignment made with
    the span `unobserved` (`align_diode_toggles`).

    Predicted edges: the expected toggle (frame time + local per-polarity latency) of every
    unobserved frame near the span; held frames get no toggle until the next frame's (phantom
    flicker can't be predicted and isn't a frame). Measured toggles are kept, and phantom flicker
    and glitches next to the span are removed with the block's cleaning (see `build_sync_patch`).

    Checks (`DiodeSpanSolution.failures`): the alignment holds across the span (frame count and
    polarity: measured edges in `window_sec` after it that aren't toggles, phantom flicker or
    glitches count as polarity errors, as do missing toggles), and the toggles in the
    `DIODE_CHECK_WINDOW_SEC` after the span are as close to their expected times as those in
    `window_sec` before it (+ `DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC`). The first toggle after a
    held frame and late `flip()` returns are left out of both, since their TTL timing deviates
    for reasons the prediction can't model.
    """
    al = alignment
    a, z = span
    ft = al._frame_times
    N = len(ft)
    t = al.expected
    near = in_intervals(t, [(a - DIODE_TOGGLE_MATCH_TOL_SEC, z + DIODE_TOGGLE_MATCH_TOL_SEC)])
    pred = np.flatnonzero(near & al.unobserved)
    after_hold = np.r_[False, al.n_refreshes >= 2]
    regular = (al.frame_edge >= 0) & ~after_hold
    regular[al.second_pass] = False
    e_t = np.where(al.frame_edge >= 0, al.edge_times[np.clip(al.frame_edge, 0, None)], np.nan)
    pre = regular & (e_t < a) & (e_t >= a - window_sec)
    post = regular & (e_t >= z) & (e_t < z + DIODE_CHECK_WINDOW_SEC)
    lat_r = _nanmedian_ms((t - ft)[near | pre], al.frame_rising[near | pre])
    lat_f = _nanmedian_ms((t - ft)[near | pre], ~al.frame_rising[near | pre])
    common = dict(
        latency_rising_sec=(lat_r or np.nan) / 1e3,
        latency_falling_sec=(lat_f or np.nan) / 1e3,
        n_removed=n_removed,
    )
    if pre.sum() < 6 or post.sum() < 1:
        return np.array([]), np.array([]), DiodeSpanSolution(
            span, "failed", reason="too few measured diode toggles near the span", **common
        )
    after_win = (al.edge_times >= z) & (al.edge_times < z + window_sec)
    n_bad = int((after_win & (al.edge_class == DIODE_UNEXPLAINED)).sum())
    frames_after = np.flatnonzero((t >= z) & (t < z + window_sec))
    n_missing = int(np.isin(frames_after, al.missing_frames).sum())
    rising = pred[al.frame_rising[pred]]
    falling = pred[~al.frame_rising[pred]]
    return (
        t[rising],
        t[falling],
        DiodeSpanSolution(
            span,
            "reconstructed",
            baseline_error_sec=float(np.max(np.abs(al.residual[pre]))),
            after_error_sec=float(np.max(np.abs(al.residual[post]))),
            n_polarity_errors_after=n_bad + n_missing,
            n_inserted=len(pred),
            **common,
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
class RemovedEdges:
    """Measured edges removed from one line, anywhere in the file (not tied to invalid
    intervals): e.g. phantom flicker and comparator glitches on the photodiode line."""

    reason: str
    """'phantom_flicker' or 'glitch' (see `DIODE_EDGE_CLASSES`)."""
    block_index: int
    """Vsync block the edges belong to (`SyncDataset.vsync_times_in_blocks`)."""
    rising_samples: npt.NDArray[np.int64] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    falling_samples: npt.NDArray[np.int64] = dataclasses.field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    """Sample indices of the removed edges, as recorded."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "block_index": int(self.block_index),
            "rising_samples": [int(x) for x in self.rising_samples],
            "falling_samples": [int(x) for x in self.falling_samples],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RemovedEdges:
        return cls(
            reason=d["reason"],
            block_index=int(d["block_index"]),
            rising_samples=np.asarray(d["rising_samples"], dtype=np.int64),
            falling_samples=np.asarray(d["falling_samples"], dtype=np.int64),
        )


DiodeBlockStatus = Literal["cleaned", "rejected", "not_checked"]


@dataclasses.dataclass
class DiodeBlockPatch:
    """Outcome of checking one vsync block's photodiode edges against its stim file.

    - 'cleaned': every edge is explained (`align_diode_toggles`). Phantom flicker and glitches
      (if any) are removed through `SyncPatch.removed`; consumers use the line as patched and
      must not remove further edges (npc_stim skips its own anomalous-flip removal).
    - 'rejected': edges that are none of a toggle, phantom flicker or a glitch, or toggle frames
      without an edge. Nothing is removed; consumers must not compute display times.
    - 'not_checked': not applicable or not possible (diodeBoxFrameInterval > 1, no held frames to
      fix the frame <-> edge offset, vsync count != stim frames, ...). Consumers keep their own
      handling.
    """

    block_index: int
    start: float
    """First vsync of the block (s)."""
    stop: float
    """Last vsync of the block (s)."""
    status: DiodeBlockStatus
    reason: str = ""
    stim: str = ""
    diode_box_frame_interval: int | None = None
    diagnostics: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Edge class counts, latencies, landmarks, long-interval check (JSON-serialisable)."""

    @property
    def n_phantom_flicker(self) -> int:
        return int(self.diagnostics.get("phantom_flicker", 0) or 0)

    @property
    def n_glitch(self) -> int:
        return int(self.diagnostics.get("glitch", 0) or 0)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DiodeBlockPatch:
        return cls(**{k: d[k] for k in d if k in {f.name for f in dataclasses.fields(cls)}})


@dataclasses.dataclass
class SyncPatch:
    """Corrections to the recorded edges of one sync file.

    - `lines`: per-line replacement edges inside invalid (replayed) intervals. A line with a
      `SpanPatch` for every invalid interval returns measured edges outside the spans plus the
      patch's edges inside; spans with status 'unknown' return nothing inside (see
      `SyncDataset.get_unknown_intervals`). A line without patches raises on edge queries
      (unless `on_invalid` is relaxed).
    - `removed`: measured edges removed anywhere in the file, including files with no invalid
      intervals (photodiode phantom flicker and glitches).
    - `diode_blocks`: per vsync block, whether its photodiode edges were checked and cleaned
      (`DiodeBlockPatch`).

    Apply with `SyncDataset(path, patch=patch)`. The patch is keyed to the sync file's content
    hash (`SyncDataset.content_hash`) and records the builder that made it.
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
    removed: dict[str, list[RemovedEdges]] = dataclasses.field(default_factory=dict)
    """line label -> measured edges removed."""
    diode_blocks: list[DiodeBlockPatch] = dataclasses.field(default_factory=list)

    # application ---------------------------------------------------------------------- #

    def removed_samples(self, line: str, kind: Literal["rising", "falling"]) -> npt.NDArray[np.int64]:
        """Sample indices of the measured `kind` edges removed from `line`."""
        parts = [
            r.rising_samples if kind == "rising" else r.falling_samples
            for r in self.removed.get(line, [])
        ]
        return np.unique(np.concatenate(parts)).astype(np.int64) if parts else np.array([], dtype=np.int64)

    def keep_recorded(
        self, line: str, kind: Literal["rising", "falling"], recorded_samples: npt.ArrayLike
    ) -> npt.NDArray[np.bool_]:
        """Mask of recorded edges (sample indices) that aren't removed by this patch."""
        rec = np.asarray(recorded_samples)
        removed = self.removed_samples(line, kind)
        if not removed.size:
            return np.ones(rec.shape, dtype=bool)
        return ~np.isin(np.round(rec).astype(np.int64), removed)

    def diode_block(self, start: float, stop: float) -> DiodeBlockPatch | None:
        """The diode record whose block overlaps [start, stop] (vsync times, seconds)."""
        for b in self.diode_blocks:
            if b.start <= stop and b.stop >= start:
                return b
        return None

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
            "removed": {k: [r.to_dict() for r in v] for k, v in self.removed.items()},
            "diode_blocks": [b.to_dict() for b in self.diode_blocks],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SyncPatch:
        if d.get("format_version") not in (1, PATCH_FORMAT_VERSION):
            raise ValueError(f"unsupported patch format {d.get('format_version')!r}")
        return cls(
            sync_content_hash=d["sync_content_hash"],
            builder=d["builder"],
            created=d["created"],
            invalid_intervals=tuple((float(a), float(z)) for a, z in d["invalid_intervals"]),
            faults=tuple(SyncFault(**f) for f in d["faults"]),
            lines={k: [SpanPatch.from_dict(s) for s in v] for k, v in d["lines"].items()},
            notes=list(d.get("notes", [])),
            removed={
                k: [RemovedEdges.from_dict(r) for r in v]
                for k, v in d.get("removed", {}).items()
            },
            diode_blocks=[DiodeBlockPatch.from_dict(b) for b in d.get("diode_blocks", [])],
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
        """line -> {'reconstructed': n spans, 'unknown': n spans, 'removed_<reason>': n edges}"""
        out: dict[str, dict[str, int]] = {}
        for line, spans in self.lines.items():
            c = out.setdefault(line, {"reconstructed": 0, "unknown": 0})
            for s in spans:
                c[s.status] += 1
        for line, removed in self.removed.items():
            c = out.setdefault(line, {"reconstructed": 0, "unknown": 0})
            for r in removed:
                key = f"removed_{r.reason}"
                c[key] = c.get(key, 0) + len(r.rising_samples) + len(r.falling_samples)
        return out

    def diode_summary(self) -> dict[str, int]:
        """Counts over `diode_blocks`: blocks by status, phantom flicker and glitch edges removed."""
        out = {"cleaned": 0, "rejected": 0, "not_checked": 0, "phantom_flicker_removed": 0, "glitch_removed": 0}
        for b in self.diode_blocks:
            out[b.status] += 1
        for r in self.removed.get("stim_photodiode", []):
            n = len(r.rising_samples) + len(r.falling_samples)
            out[f"{r.reason}_removed"] = out.get(f"{r.reason}_removed", 0) + n
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
    """Build a `SyncPatch` for a sync file: phantom-flicker removal on the photodiode line of
    every box-1 block with a stim file (any file), and reconstructions inside invalid (replayed)
    intervals.

    - `stim_photodiode`, every block with a matched stim file and diodeBoxFrameInterval 1: edges
      aligned with the block's frames (`align_diode_toggles`, with reconstructed vsyncs where the
      block has replayed spans). If every edge is explained, phantom flicker and glitches are
      removed (`SyncPatch.removed`) and the block is 'cleaned'; edges that can't be explained
      make it 'rejected' (nothing removed). `SyncPatch.diode_blocks` records the outcome for every
      block with a stim file.
    - `vsync_stim`: for each span inside a block with a matched stim file, frames are solved
      per gap from `frameIntervals` (`reconcile_vsyncs`); rising edges are placed at the
      reconstructed falling edge minus the median measured pulse width in the block. Spans in
      blocks without a stim file, or that fail a gate, are 'unknown'. Spans outside every
      block with no recorded vsync edges are reconstructed as 'no_edges'.
    - `stim_photodiode` inside spans: box interval 1 in a 'cleaned' block: expected toggles of
      the frames in the span (`predict_box1_diode_edges`); box interval > 1:
      `predict_diode_edges`; otherwise, or if a gate fails, 'unknown'.
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
        A `SyncDataset` (read with `on_invalid='ignore'` and no patch internally, so the edges
        are as recorded).
    stim_blocks
        Stim data per vsync block index (`SyncDataset.vsync_times_in_blocks` order, unpatched).
    builder
        Identifier of the calling builder (recorded in the patch), e.g. 'npc_sessions 0.0.292'.
    """
    from npc_sync.sync import SyncDataset  # avoid import cycle

    raw = SyncDataset(sync, on_invalid="ignore")
    raw.patch = None
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
    fs = raw.sample_freq
    # in 4 of 6 replays checked, the event at the sample before a copy already carries the
    # copy's state (experiments/sync-patch-manifest/acceptance_local7.log), so every span starts
    # one sample early
    spans_base = tuple((a - 1 / fs, z) for a, z in invalid)

    labels = [line_key(raw.line_labels, bit) for bit in range(len(raw.line_labels))]
    # --- vsync + photodiode ----------------------------------------------------------
    vsync_spans: list[SpanPatch] = []
    diode_spans: list[SpanPatch] = []
    has_vsync = "vsync_stim" in labels and raw.get_falling_edges("vsync_stim").size > 0
    blocks: Sequence[npt.NDArray] = ()
    if has_vsync:
        try:
            blocks = raw.vsync_times_in_blocks
        except (AssertionError, IndexError, ValueError) as exc:
            # e.g. unbalanced stim_running edges: no blocks, so nothing to check or reconstruct
            patch.notes.append(f"vsync blocks could not be computed: {exc!r}")
    vr_all = raw.get_rising_edges("vsync_stim", units="seconds") if has_vsync else np.array([])
    vf_all = raw.get_falling_edges("vsync_stim", units="seconds") if has_vsync else np.array([])
    has_diode = "stim_photodiode" in labels
    if has_diode:
        d_rs = np.asarray(raw.get_rising_edges("stim_photodiode", units="samples"), dtype=np.int64)
        d_fs = np.asarray(raw.get_falling_edges("stim_photodiode", units="samples"), dtype=np.int64)
        dr_all, df_all = d_rs / fs, d_fs / fs
        d_samples = np.concatenate([d_rs, d_fs])
        d_rising = np.concatenate([np.ones(len(d_rs), bool), np.zeros(len(d_fs), bool)])
        o = np.argsort(d_samples, kind="stable")
        d_samples, d_rising = d_samples[o], d_rising[o]
        d_times = d_samples / fs
        diode_bit = raw._line_to_bit("stim_photodiode")
    else:
        dr_all = df_all = np.array([])
    by_block = {sb.block_index: sb for sb in stim_blocks}
    covered: list[tuple[float, float]] = []

    def diode_record(
        bi: int,
        block: npt.NDArray,
        status: DiodeBlockStatus,
        reason: str = "",
        sb: StimBlock | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        patch.diode_blocks.append(
            DiodeBlockPatch(
                block_index=bi,
                start=float(block[0]),
                stop=float(block[-1]),
                status=status,
                reason=reason,
                stim=sb.name if sb else "",
                diode_box_frame_interval=sb.diode_box_frame_interval if sb else None,
                diagnostics=_jsonable(diagnostics or {}),
            )
        )

    for bi, block in enumerate(blocks):
        overlapping = [(a, z) for a, z in spans_base if a <= block[-1] and z > block[0]]
        sb = by_block.get(bi)
        if sb is None:
            for a, z in overlapping:
                vsync_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"block {bi}: no stim file"))
                diode_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=f"block {bi}: no stim file"))
                covered.append((a, z))
            continue
        rec: VsyncReconciliation | None = None
        if overlapping:
            rec = solve_vsync_gaps(block, sb.frame_intervals, overlapping)
            failures = rec.failures()
            if failures:
                reason = f"block {bi} ({sb.name}): " + "; ".join(failures)
                for a, z in overlapping:
                    vsync_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=reason))
                    diode_spans.append(SpanPatch((a, z), "unknown", method="mask", reason=reason))
                    covered.append((a, z))
                if has_diode:
                    diode_record(bi, block, "not_checked", "vsyncs not reconstructed: " + "; ".join(failures), sb)
                continue
            # pulse width from measured edges in the block
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
            frame_times = rec.times
            frame_measured = rec.source == MEASURED
        else:
            if len(block) != len(sb.frame_intervals) + 1:
                if has_diode:
                    diode_record(
                        bi, block, "not_checked",
                        f"{len(block)} vsyncs, {len(sb.frame_intervals) + 1} stim frames", sb,
                    )
                continue
            frame_times = np.asarray(block, dtype=float)
            frame_measured = np.ones(len(block), bool)
        if not has_diode:
            continue
        gap_spans = [g.span for g in rec.gaps] if rec is not None else []
        box = sb.diode_box_frame_interval
        if box is None:
            for sp in gap_spans:
                diode_spans.append(SpanPatch(sp, "unknown", method="mask", reason="diodeBoxFrameInterval unknown"))
            diode_record(bi, block, "not_checked", "diodeBoxFrameInterval unknown", sb)
            continue
        if box != 1:
            for sp in gap_spans:
                others = [x for x in spans_base if not (x[0] >= sp[0] - 1e-9 and x[1] <= sp[1] + 1e-9)]
                rise, fall_d, sol = predict_diode_edges(frame_times, dr_all, df_all, sp, box, exclude=others)
                _append_diode_span(diode_spans, sp, rise, fall_d, sol, method="vsync+latency")
            diode_record(
                bi, block, "not_checked",
                f"phantom flicker removal applies to diodeBoxFrameInterval 1 (this block: {box})", sb,
            )
            continue
        # box interval 1: align every measured edge with the block's frames
        d_spans = [_extend_span_over_splice(raw, diode_bit, sp) for sp in gap_spans]
        in_win = (d_times >= block[0]) & (d_times < block[-1] + DIODE_BLOCK_END_PADDING_SEC)
        sel = np.flatnonzero(in_win & ~in_intervals(d_times, d_spans))
        jr = np.searchsorted(vr_all, frame_times) - 1
        frame_rise = np.where(frame_measured & (jr >= 0), vr_all[np.clip(jr, 0, None)], np.nan)
        try:
            al = align_diode_toggles(
                frame_times,
                sb.frame_intervals,
                d_times[sel],
                d_rising[sel],
                frame_signal_rising=frame_rise,
                unobserved=d_spans,
            )
        except DiodeOffsetUndetermined as exc:
            for sp in d_spans:
                diode_spans.append(SpanPatch(sp, "unknown", method="mask", reason=f"diodeBoxFrameInterval 1: {exc}"))
            diode_record(bi, block, "not_checked", f"frame <-> diode edge offset undetermined: {exc}", sb)
            continue
        diag = al.summary()
        fails = al.failures()
        if fails:
            reason = "; ".join(fails)
            for sp in d_spans:
                diode_spans.append(
                    SpanPatch(sp, "unknown", method="mask", diagnostics=diag, reason=f"diode edges not explained: {reason}")
                )
            diode_record(bi, block, "rejected", reason, sb, diag)
            continue
        for cls_value, reason in ((DIODE_PHANTOM_FLICKER, "phantom_flicker"), (DIODE_GLITCH, "glitch")):
            idx = sel[np.flatnonzero(al.edge_class == cls_value)]
            if len(idx):
                patch.removed.setdefault("stim_photodiode", []).append(
                    RemovedEdges(
                        reason=reason,
                        block_index=bi,
                        rising_samples=d_samples[idx[d_rising[idx]]],
                        falling_samples=d_samples[idx[~d_rising[idx]]],
                    )
                )
        diode_record(bi, block, "cleaned", "", sb, diag)
        for sp in d_spans:
            n_in = int(in_intervals(d_times, [sp]).sum())
            rise, fall_d, sol = predict_box1_diode_edges(al, sp, n_removed=n_in)
            _append_diode_span(diode_spans, sp, rise, fall_d, sol, method="vsync+latency (box 1 alignment)")
    if not invalid:
        return patch
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
        clean_r = rising[~in_intervals(rising, spans_base)]
        clean_f = falling[~in_intervals(falling, spans_base)]
        pwm = startup_pwm_section(clean_r, clean_f)
        for a, z in spans_base:
            new_r, sol_r = interpolate_periodic_edges(rising, (a, z), fit_exclude=pwm)
            new_f, sol_f = interpolate_periodic_edges(falling, (a, z), fit_exclude=pwm)
            fails = sol_r.failures() + sol_f.failures()
            diag = {"rising": dataclasses.asdict(sol_r), "falling": dataclasses.asdict(sol_f),
                    "pwm_sections_excluded_from_fit": [list(x) for x in pwm]}
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


def _append_diode_span(
    spans: list[SpanPatch],
    span: tuple[float, float],
    rising: npt.NDArray,
    falling: npt.NDArray,
    sol: DiodeSpanSolution,
    method: str,
) -> None:
    fails = sol.failures()
    diag = _jsonable(dataclasses.asdict(sol))
    if fails:
        spans.append(SpanPatch(span, "unknown", method="mask", diagnostics=diag, reason="; ".join(fails)))
    else:
        spans.append(
            SpanPatch(
                span,
                "reconstructed",
                rising=rising,
                falling=falling,
                method=method,
                error_bound_sec=sol.baseline_error_sec + DIODE_MAX_RESIDUAL_ABOVE_BASELINE_SEC,
                diagnostics=diag,
            )
        )


def _extend_span_over_splice(raw: Any, bit: int, span: tuple[float, float]) -> tuple[float, float]:
    """The first event after a copy is compared with the copy's last state, so a change of `bit`
    there may be a splice artefact rather than a real edge: include that event in the span."""
    times = np.asarray(raw.get_all_times("samples"), dtype=np.int64)
    bits = np.asarray(raw.get_all_bits(), dtype=np.int64)
    fs = raw.sample_freq
    j = int(np.searchsorted(times, span[1] * fs - 1e-6))
    if 0 < j < len(times) and ((bits[j] >> bit) & 1) != ((bits[j - 1] >> bit) & 1):
        return (span[0], (times[j] + 1) / fs)
    return span


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
